#ifdef OVG_CUDA_ENABLED

#include "cuda_ops.h"
#include <cuda_runtime.h>

// ── Tiled GEMM ───────────────────────────────────────────────────────────────
//
// Tile size 16×16.  Each thread block computes one 16×16 tile of the output.
// Shared memory staging eliminates redundant global-memory reads.
// For the backward kernels we transpose one operand in the tile-load step
// and use atomicAdd so gradient accumulation is correct when a weight tensor
// appears as a parent of more than one matmul node in the graph.

#define TILE 16

// ── Forward: C = A @ B ───────────────────────────────────────────────────────

__global__ void matmul_fwd_kernel(const float *A, const float *B, float *C,
                                   int m, int K, int n)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // output row  (i)
    int col = blockIdx.x * TILE + threadIdx.x;   // output col  (j)

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;       // A column for this thread
        int b_row = t * TILE + threadIdx.y;       // B row for this thread

        sA[threadIdx.y][threadIdx.x] = (row < m && a_col < K) ? A[row * K + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < n) ? B[b_row * n + col]  : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = acc;
}

// ── Backward dA: dA += dC @ B^T ─────────────────────────────────────────────
//
// Output tile [m × K].  Thread (ty,tx) accumulates dA[row, col].
// Inner dimension j iterates over columns of dC / columns of B.
//   sD[ty][tx] = dC[row, t*TILE+tx]          (load row of dC)
//   sB[ty][tx] = B[col, t*TILE+ty]            (load row of B — equiv. col of B^T)
// Accumulate: acc += sum_k sD[ty][k] * sB[k][tx]

__global__ void matmul_bwd_dA_kernel(const float *dC, const float *B, float *dA,
                                      int m, int K, int n)
{
    __shared__ float sD[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // i  in [0, m)
    int col = blockIdx.x * TILE + threadIdx.x;   // ki in [0, K)

    float acc = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        int j_tx = t * TILE + threadIdx.x;        // j index for dC load
        int j_ty = t * TILE + threadIdx.y;        // j index for B load

        sD[threadIdx.y][threadIdx.x] = (row < m && j_tx < n) ? dC[row * n + j_tx] : 0.0f;
        // B^T[j, ki] = B[ki, j] = B[col * n + j_ty]
        sB[threadIdx.y][threadIdx.x] = (col < K && j_ty < n) ? B[col * n + j_ty]  : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            acc += sD[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < K)
        atomicAdd(&dA[row * K + col], acc);
}

// ── Backward dB: dB += A^T @ dC ─────────────────────────────────────────────
//
// Output tile [K × n].  Thread (ty,tx) accumulates dB[row, col].
// Inner dimension i iterates over rows of A / rows of dC.
//   sA[ty][tx] = A[t*TILE+ty, row]            (load col of A — equiv. row of A^T)
//   sD[ty][tx] = dC[t*TILE+ty, col]           (load row of dC)
// Accumulate: acc += sum_k sA[k][tx] * sD[k][tx]  — but note both share k index
//   => acc += sum_k sA[ty_k][tx] * sD[ty_k][col_tx]
//   Actually: acc += sA[k][threadIdx.x (row)] * sD[k][threadIdx.x (col)]
//   We must track both independently via separate indices.
//
// Standard pattern: sA[ty][tx] is A^T[row, i_local] transposed-loaded,
// sD[ty][tx] is dC[i_local, col] — then acc += sum_k sA[k][tx] * sD[k][tx]
// does NOT work because sA and sD share threadIdx.x in different roles.
//
// Correct layout:
//   sA[ty][tx]: thread (ty,tx) loads A^T[row=blockCol, i=t*TILE+ty]
//               i.e. A[t*TILE+ty, row] where row = blockIdx.x*TILE + tx
//   sD[ty][tx]: thread (ty,tx) loads dC[t*TILE+ty, col]
//               col = blockIdx.x*TILE + tx   (same tx, different semantics)
// Wait — row and col are BOTH indexed by threadIdx.x here.  We need two
// separate tile coordinates.  Use row = blockIdx.y*TILE+ty (ki) and
// col = blockIdx.x*TILE+tx (j) as the output coordinates, and i as the
// inner reduction dimension.
//
//   sA[ty][tx]: A^T tile row=ki, i_local=tx  => A[t*TILE+tx, ki] = A[(t*TILE+tx)*K + row]
//               But ty maps to ki (output row), tx maps to i_local; all threads
//               in same row load different i but same ki (row = blockIdx.y*TILE+ty).
//               So: sA[ty][tx] = A[(t*TILE+tx)*K + row]   where row = blockIdx.y*TILE+ty
//   sD[ty][tx]: dC[i_local, col] = dC[(t*TILE+ty)*n + col]
//               ty maps to i_local, tx maps to col.
// Then:  acc += sum_{k=0}^{TILE-1} sA[ty][k] * sD[k][tx]

__global__ void matmul_bwd_dB_kernel(const float *A, const float *dC, float *dB,
                                      int m, int K, int n)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sD[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // ki in [0, K)
    int col = blockIdx.x * TILE + threadIdx.x;   // j  in [0, n)

    float acc = 0.0f;

    for (int t = 0; t < (m + TILE - 1) / TILE; t++) {
        int i_tx = t * TILE + threadIdx.x;        // i index for A^T column load
        int i_ty = t * TILE + threadIdx.y;        // i index for dC row load

        // A^T[row=ki, i_local=tx] = A[i_local, ki] = A[i_tx * K + row]
        sA[threadIdx.y][threadIdx.x] = (i_tx < m && row < K) ? A[i_tx * K + row]  : 0.0f;
        // dC[i_local=ty, col=j]
        sD[threadIdx.y][threadIdx.x] = (i_ty < m && col < n) ? dC[i_ty * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            acc += sA[threadIdx.y][k] * sD[k][threadIdx.x];
        __syncthreads();
    }

    if (row < K && col < n)
        atomicAdd(&dB[row * n + col], acc);
}

// ── Host-side launchers — matmul ─────────────────────────────────────────────

void cuda_matmul_forward(const float *A, const float *B, float *C,
                         int m, int K, int n)
{
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    matmul_fwd_kernel<<<grid, block>>>(A, B, C, m, K, n);
}

void cuda_matmul_bwd_dA(const float *dC, const float *B, float *dA,
                        int m, int K, int n)
{
    dim3 block(TILE, TILE);
    dim3 grid((K + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    matmul_bwd_dA_kernel<<<grid, block>>>(dC, B, dA, m, K, n);
}

void cuda_matmul_bwd_dB(const float *A, const float *dC, float *dB,
                        int m, int K, int n)
{
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (K + TILE - 1) / TILE);
    matmul_bwd_dB_kernel<<<grid, block>>>(A, dC, dB, m, K, n);
}

// ── Element-wise kernels ──────────────────────────────────────────────────────

#define BLOCK 256

__global__ void add_fwd_k(const float *a, const float *b, float *o, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}
__global__ void add_bwd_k(const float *g, float *da, float *db, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) { atomicAdd(&da[i], g[i]); atomicAdd(&db[i], g[i]); }
}

__global__ void sub_fwd_k(const float *a, const float *b, float *o, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) o[i] = a[i] - b[i];
}
__global__ void sub_bwd_k(const float *g, float *da, float *db, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) { atomicAdd(&da[i], g[i]); atomicAdd(&db[i], -g[i]); }
}

__global__ void mul_fwd_k(const float *a, const float *b, float *o, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) o[i] = a[i] * b[i];
}
__global__ void mul_bwd_k(const float *a, const float *b, const float *g,
                           float *da, float *db, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) { atomicAdd(&da[i], b[i] * g[i]); atomicAdd(&db[i], a[i] * g[i]); }
}

__global__ void scale_fwd_k(const float *a, float s, float *o, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) o[i] = a[i] * s;
}
__global__ void scale_bwd_k(const float *g, float s, float *da, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) atomicAdd(&da[i], s * g[i]);
}

__global__ void pow_fwd_k(const float *a, float p, float *o, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) o[i] = powf(a[i], p);
}
__global__ void pow_bwd_k(const float *a, const float *g, float p, float *da, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) atomicAdd(&da[i], p * powf(a[i], p - 1.0f) * g[i]);
}

__global__ void relu_fwd_k(const float *a, float *o, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) o[i] = a[i] > 0.0f ? a[i] : 0.0f;
}
__global__ void relu_bwd_k(const float *a, const float *g, float *da, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) atomicAdd(&da[i], g[i] * (a[i] > 0.0f ? 1.0f : 0.0f));
}

__global__ void tanh_fwd_k(const float *a, float *o, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) o[i] = tanhf(a[i]);
}
__global__ void tanh_bwd_k(const float *out, const float *g, float *da, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) { float t = out[i]; atomicAdd(&da[i], (1.0f - t * t) * g[i]); }
}

__device__ float gelu_grad_d(float x) {
    const float k = 0.7978845608028654f;
    const float c = 0.044715f;
    float x2 = x * x;
    float u = k * (x + c * x * x2);
    float t = tanhf(u);
    float du = k * (1.0f + 3.0f * c * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
}

__global__ void gelu_fwd_k(const float *a, float *o, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    const float k = 0.7978845608028654f;
    const float c = 0.044715f;
    float x = a[i];
    o[i] = 0.5f * x * (1.0f + tanhf(k * (x + c * x * x * x)));
}
__global__ void gelu_bwd_k(const float *a, const float *g, float *da, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) atomicAdd(&da[i], gelu_grad_d(a[i]) * g[i]);
}

// ── Host launchers — element-wise ────────────────────────────────────────────

static int blocks(int n) { return (n + BLOCK - 1) / BLOCK; }

void cuda_add_fwd(const float *a, const float *b, float *o, int n)
    { add_fwd_k<<<blocks(n),BLOCK>>>(a,b,o,n); }
void cuda_add_bwd(const float *g, float *da, float *db, int n)
    { add_bwd_k<<<blocks(n),BLOCK>>>(g,da,db,n); }

void cuda_sub_fwd(const float *a, const float *b, float *o, int n)
    { sub_fwd_k<<<blocks(n),BLOCK>>>(a,b,o,n); }
void cuda_sub_bwd(const float *g, float *da, float *db, int n)
    { sub_bwd_k<<<blocks(n),BLOCK>>>(g,da,db,n); }

void cuda_mul_fwd(const float *a, const float *b, float *o, int n)
    { mul_fwd_k<<<blocks(n),BLOCK>>>(a,b,o,n); }
void cuda_mul_bwd(const float *a, const float *b, const float *g,
                  float *da, float *db, int n)
    { mul_bwd_k<<<blocks(n),BLOCK>>>(a,b,g,da,db,n); }

void cuda_scale_fwd(const float *a, float s, float *o, int n)
    { scale_fwd_k<<<blocks(n),BLOCK>>>(a,s,o,n); }
void cuda_scale_bwd(const float *g, float s, float *da, int n)
    { scale_bwd_k<<<blocks(n),BLOCK>>>(g,s,da,n); }

void cuda_pow_fwd(const float *a, float p, float *o, int n)
    { pow_fwd_k<<<blocks(n),BLOCK>>>(a,p,o,n); }
void cuda_pow_bwd(const float *a, const float *g, float p, float *da, int n)
    { pow_bwd_k<<<blocks(n),BLOCK>>>(a,g,p,da,n); }

void cuda_relu_fwd(const float *a, float *o, int n)
    { relu_fwd_k<<<blocks(n),BLOCK>>>(a,o,n); }
void cuda_relu_bwd(const float *a, const float *g, float *da, int n)
    { relu_bwd_k<<<blocks(n),BLOCK>>>(a,g,da,n); }

void cuda_tanh_fwd(const float *a, float *o, int n)
    { tanh_fwd_k<<<blocks(n),BLOCK>>>(a,o,n); }
void cuda_tanh_bwd(const float *out, const float *g, float *da, int n)
    { tanh_bwd_k<<<blocks(n),BLOCK>>>(out,g,da,n); }

void cuda_gelu_fwd(const float *a, float *out, int n)
    { gelu_fwd_k<<<blocks(n),BLOCK>>>(a,out,n); }
void cuda_gelu_bwd(const float *a, const float *g, float *da, int n)
    { gelu_bwd_k<<<blocks(n),BLOCK>>>(a,g,da,n); }

// ── Transpose ────────────────────────────────────────────────────────────────

__global__ void transpose_k(const float *a, float *o, int rows, int cols) {
    // Use blockDim (not BLOCK) — this kernel launches with dim3(16,16), not BLOCK threads.
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) o[j * rows + i] = a[i * cols + j];
}

__global__ void transpose_bwd_k(const float *g, float *da, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) da[j * rows + i] += g[i * cols + j];
}

void cuda_transpose_fwd(const float *a, float *o, int rows, int cols) {
    dim3 blk(16, 16);
    dim3 grd((cols+15)/16, (rows+15)/16);
    transpose_k<<<grd,blk>>>(a, o, rows, cols);
}
void cuda_transpose_bwd(const float *g, float *da, int out_rows, int out_cols) {
    dim3 blk(16, 16);
    dim3 grd((out_cols+15)/16, (out_rows+15)/16);
    transpose_bwd_k<<<grd,blk>>>(g, da, out_rows, out_cols);
}

// ── Reductions ────────────────────────────────────────────────────────────────

__global__ void sum_fwd_k(const float *a, float *out, int n) {
    __shared__ float smem[BLOCK];
    int i = blockIdx.x * BLOCK + threadIdx.x;
    smem[threadIdx.x] = (i < n) ? a[i] : 0.0f;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, smem[0]);
}

__global__ void broadcast_bwd_k(const float *g, float *da, int n, float scale) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) atomicAdd(&da[i], g[0] * scale);
}

__global__ void mean_rows_fwd_k(const float *a, float *out, int rows, int cols) {
    int j = blockIdx.x * BLOCK + threadIdx.x;
    if (j >= cols) return;
    float s = 0.0f;
    for (int i = 0; i < rows; i++) s += a[i * cols + j];
    out[j] = s / (float)rows;
}

__global__ void mean_rows_bwd_k(const float *g, float *da, int rows, int cols) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= rows * cols) return;
    int j = idx % cols;
    atomicAdd(&da[idx], g[j] / (float)rows);
}

void cuda_sum_fwd(const float *a, float *out, int n)
    { sum_fwd_k<<<blocks(n),BLOCK>>>(a, out, n); }
void cuda_sum_bwd(const float *g, float *da, int n)
    { broadcast_bwd_k<<<blocks(n),BLOCK>>>(g, da, n, 1.0f); }
void cuda_mean_bwd(const float *g, float *da, int n)
    { broadcast_bwd_k<<<blocks(n),BLOCK>>>(g, da, n, 1.0f / (float)n); }

void cuda_mean_rows_fwd(const float *a, float *out, int rows, int cols)
    { mean_rows_fwd_k<<<blocks(cols),BLOCK>>>(a, out, rows, cols); }
void cuda_mean_rows_bwd(const float *g, float *da, int rows, int cols)
    { mean_rows_bwd_k<<<blocks(rows*cols),BLOCK>>>(g, da, rows, cols); }

// ── Row-wise ops (one block per row, 256 threads) ────────────────────────────

__global__ void softmax_rows_fwd_k(const float *a, float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *in = a   + row * cols;
    float       *res = out + row * cols;

    __shared__ float smem[BLOCK];

    // max
    float mx = -1e38f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) mx = fmaxf(mx, in[j]);
    smem[threadIdx.x] = mx;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+s]);
        __syncthreads();
    }
    mx = smem[0];

    // exp and partial sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) {
        float e = expf(in[j] - mx);
        res[j] = e;
        sum += e;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    sum = smem[0];

    for (int j = threadIdx.x; j < cols; j += BLOCK) res[j] /= sum;
}

__global__ void softmax_rows_bwd_k(const float *s, const float *dy, float *da,
                                    int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *sr  = s  + row * cols;
    const float *dyr = dy + row * cols;
    float       *dxr = da + row * cols;

    __shared__ float smem[BLOCK];

    float dot = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) dot += dyr[j] * sr[j];
    smem[threadIdx.x] = dot;
    __syncthreads();
    for (int s2 = BLOCK/2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) smem[threadIdx.x] += smem[threadIdx.x+s2];
        __syncthreads();
    }
    dot = smem[0];

    for (int j = threadIdx.x; j < cols; j += BLOCK)
        atomicAdd(&dxr[j], sr[j] * (dyr[j] - dot));
}

__global__ void layer_norm_rows_fwd_k(const float *a, float eps, float *out,
                                       float *inv_std, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = a   + row * cols;
    float       *yr = out + row * cols;

    __shared__ float smem[BLOCK];

    // mean
    float mean = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) mean += xr[j];
    smem[threadIdx.x] = mean;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    mean = smem[0] / (float)cols;

    // variance
    float var = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) {
        float c = xr[j] - mean;
        var += c * c;
    }
    smem[threadIdx.x] = var;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    float is = 1.0f / sqrtf(smem[0] / (float)cols + eps);
    if (threadIdx.x == 0) inv_std[row] = is;

    for (int j = threadIdx.x; j < cols; j += BLOCK)
        yr[j] = (xr[j] - mean) * is;
}

__global__ void layer_norm_rows_bwd_k(const float *y, const float *dy,
                                       const float *inv_std, float *dx,
                                       int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *yr  = y  + row * cols;
    const float *dyr = dy + row * cols;
    float       *dxr = dx + row * cols;
    float is = inv_std[row];
    float invC = 1.0f / (float)cols;

    __shared__ float smem[BLOCK];

    // mean_dy
    float mean_dy = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) mean_dy += dyr[j];
    smem[threadIdx.x] = mean_dy;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    mean_dy = smem[0] * invC;

    // mean_dy_y
    float mean_dy_y = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) mean_dy_y += dyr[j] * yr[j];
    smem[threadIdx.x] = mean_dy_y;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    mean_dy_y = smem[0] * invC;

    for (int j = threadIdx.x; j < cols; j += BLOCK)
        atomicAdd(&dxr[j], is * (dyr[j] - mean_dy - yr[j] * mean_dy_y));
}

__global__ void layer_norm_rows_affine_fwd_k(const float *a, const float *gamma,
                                              const float *beta, float eps,
                                              float *out, float *cache,
                                              int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = a + row * cols;
    float *yr = out + row * cols;
    float *inv_std = cache;
    float *xhat = cache + rows + row * cols;

    __shared__ float smem[BLOCK];

    float mean = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) mean += xr[j];
    smem[threadIdx.x] = mean;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    mean = smem[0] / (float)cols;

    float var = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) {
        float c = xr[j] - mean;
        var += c * c;
    }
    smem[threadIdx.x] = var;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    float is = 1.0f / sqrtf(smem[0] / (float)cols + eps);
    if (threadIdx.x == 0) inv_std[row] = is;

    for (int j = threadIdx.x; j < cols; j += BLOCK) {
        float norm = (xr[j] - mean) * is;
        xhat[j] = norm;
        yr[j] = norm * gamma[j] + beta[j];
    }
}

__global__ void layer_norm_rows_affine_bwd_k(const float *xhat, const float *dy,
                                              const float *gamma,
                                              const float *inv_std, float *dx,
                                              float *dgamma, float *dbeta,
                                              int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xhr = xhat + row * cols;
    const float *dyr = dy + row * cols;
    float *dxr = dx + row * cols;
    float is = inv_std[row];
    float invC = 1.0f / (float)cols;

    __shared__ float smem[BLOCK];

    float mean_dxhat = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK)
        mean_dxhat += dyr[j] * gamma[j];
    smem[threadIdx.x] = mean_dxhat;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    mean_dxhat = smem[0] * invC;

    float mean_dxhat_xhat = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK)
        mean_dxhat_xhat += dyr[j] * gamma[j] * xhr[j];
    smem[threadIdx.x] = mean_dxhat_xhat;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    mean_dxhat_xhat = smem[0] * invC;

    for (int j = threadIdx.x; j < cols; j += BLOCK) {
        float g = dyr[j];
        float dxhat = g * gamma[j];
        atomicAdd(&dgamma[j], g * xhr[j]);
        atomicAdd(&dbeta[j], g);
        atomicAdd(&dxr[j], is * (dxhat - mean_dxhat - xhr[j] * mean_dxhat_xhat));
    }
}

void cuda_softmax_rows_fwd(const float *a, float *out, int rows, int cols)
    { softmax_rows_fwd_k<<<rows,BLOCK>>>(a, out, rows, cols); }
void cuda_softmax_rows_bwd(const float *out_data, const float *g, float *da,
                           int rows, int cols)
    { softmax_rows_bwd_k<<<rows,BLOCK>>>(out_data, g, da, rows, cols); }

void cuda_layer_norm_rows_fwd(const float *a, float eps, float *out,
                               float *inv_std, int rows, int cols)
    { layer_norm_rows_fwd_k<<<rows,BLOCK>>>(a, eps, out, inv_std, rows, cols); }
void cuda_layer_norm_rows_bwd(const float *y, const float *dy,
                               const float *inv_std, float *dx, int rows, int cols)
    { layer_norm_rows_bwd_k<<<rows,BLOCK>>>(y, dy, inv_std, dx, rows, cols); }

void cuda_layer_norm_rows_affine_fwd(const float *a, const float *gamma, const float *beta,
                                      float eps, float *out, float *cache,
                                      int rows, int cols)
    { layer_norm_rows_affine_fwd_k<<<rows,BLOCK>>>(a, gamma, beta, eps, out, cache, rows, cols); }
void cuda_layer_norm_rows_affine_bwd(const float *xhat, const float *dy,
                                      const float *gamma, const float *inv_std,
                                      float *dx, float *dgamma, float *dbeta,
                                      int rows, int cols)
    { layer_norm_rows_affine_bwd_k<<<rows,BLOCK>>>(xhat, dy, gamma, inv_std, dx, dgamma, dbeta, rows, cols); }

// ── Special ops ───────────────────────────────────────────────────────────────

__global__ void causal_mask_fwd_k(const float *a, float *out, int T) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T && j < T)
        out[i * T + j] = (j <= i) ? a[i * T + j] : -1.0e9f;
}
__global__ void causal_mask_bwd_k(const float *g, float *da, int T) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T && j < T && j <= i)
        atomicAdd(&da[i * T + j], g[i * T + j]);
}

void cuda_causal_mask_fwd(const float *a, float *out, int T) {
    dim3 blk(16,16); dim3 grd((T+15)/16,(T+15)/16);
    causal_mask_fwd_k<<<grd,blk>>>(a,out,T);
}
void cuda_causal_mask_bwd(const float *g, float *da, int T) {
    dim3 blk(16,16); dim3 grd((T+15)/16,(T+15)/16);
    causal_mask_bwd_k<<<grd,blk>>>(g,da,T);
}

// slice_cols: extract columns [col_start, col_start+out_cols) from a
__global__ void slice_cols_fwd_k(const float *a, float *out,
                                  int rows, int a_cols, int col_start, int out_cols) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= rows * out_cols) return;
    int i = idx / out_cols, j = idx % out_cols;
    out[i * out_cols + j] = a[i * a_cols + col_start + j];
}
__global__ void slice_cols_bwd_k(const float *g, float *da,
                                  int rows, int a_cols, int col_start, int out_cols) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= rows * out_cols) return;
    int i = idx / out_cols, j = idx % out_cols;
    atomicAdd(&da[i * a_cols + col_start + j], g[i * out_cols + j]);
}

void cuda_slice_cols_fwd(const float *a, float *out,
                         int rows, int a_cols, int col_start, int out_cols) {
    int n = rows * out_cols;
    slice_cols_fwd_k<<<blocks(n),BLOCK>>>(a, out, rows, a_cols, col_start, out_cols);
}
void cuda_slice_cols_bwd(const float *g, float *da,
                         int rows, int a_cols, int col_start, int out_cols) {
    int n = rows * out_cols;
    slice_cols_bwd_k<<<blocks(n),BLOCK>>>(g, da, rows, a_cols, col_start, out_cols);
}

// concat_cols: copy one part into the concatenated buffer
__global__ void concat_fwd_k(const float *src, float *dst,
                              int rows, int src_cols, int total_cols, int col_off) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= rows * src_cols) return;
    int i = idx / src_cols, j = idx % src_cols;
    dst[i * total_cols + col_off + j] = src[i * src_cols + j];
}
__global__ void concat_bwd_k(const float *g, float *da,
                              int rows, int src_cols, int total_cols, int col_off) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= rows * src_cols) return;
    int i = idx / src_cols, j = idx % src_cols;
    atomicAdd(&da[i * src_cols + j], g[i * total_cols + col_off + j]);
}

void cuda_concat_cols_fwd(const float *src, float *dst,
                          int rows, int src_cols, int total_cols, int col_offset) {
    int n = rows * src_cols;
    concat_fwd_k<<<blocks(n),BLOCK>>>(src, dst, rows, src_cols, total_cols, col_offset);
}
void cuda_concat_cols_bwd(const float *g, float *da,
                          int rows, int src_cols, int total_cols, int col_offset) {
    int n = rows * src_cols;
    concat_bwd_k<<<blocks(n),BLOCK>>>(g, da, rows, src_cols, total_cols, col_offset);
}

// slice_rows: extract rows [row_start, row_start+out_rows) from a
__global__ void slice_rows_fwd_k(const float *a, float *out,
                                  int a_rows, int cols, int row_start, int out_rows) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= out_rows * cols) return;
    int i = idx / cols, j = idx % cols;
    out[i * cols + j] = a[(row_start + i) * cols + j];
}
__global__ void slice_rows_bwd_k(const float *g, float *da,
                                  int a_rows, int cols, int row_start, int out_rows) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= out_rows * cols) return;
    int i = idx / cols, j = idx % cols;
    atomicAdd(&da[(row_start + i) * cols + j], g[i * cols + j]);
}

void cuda_slice_rows_fwd(const float *a, float *out,
                         int a_rows, int cols, int row_start, int out_rows) {
    int n = out_rows * cols;
    slice_rows_fwd_k<<<blocks(n),BLOCK>>>(a, out, a_rows, cols, row_start, out_rows);
}
void cuda_slice_rows_bwd(const float *g, float *da,
                         int a_rows, int cols, int row_start, int out_rows) {
    int n = out_rows * cols;
    slice_rows_bwd_k<<<blocks(n),BLOCK>>>(g, da, a_rows, cols, row_start, out_rows);
}

// concat_rows: copy one part into the concatenated row buffer
__global__ void concat_rows_fwd_k(const float *src, float *dst,
                                   int src_rows, int cols, int total_rows, int row_off) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= src_rows * cols) return;
    int i = idx / cols, j = idx % cols;
    dst[(row_off + i) * cols + j] = src[i * cols + j];
}
__global__ void concat_rows_bwd_k(const float *g, float *da,
                                   int src_rows, int cols, int total_rows, int row_off) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= src_rows * cols) return;
    int i = idx / cols, j = idx % cols;
    atomicAdd(&da[i * cols + j], g[(row_off + i) * cols + j]);
}

void cuda_concat_rows_fwd(const float *src, float *dst,
                          int src_rows, int cols, int total_rows, int row_offset) {
    int n = src_rows * cols;
    concat_rows_fwd_k<<<blocks(n),BLOCK>>>(src, dst, src_rows, cols, total_rows, row_offset);
}
void cuda_concat_rows_bwd(const float *g, float *da,
                          int src_rows, int cols, int total_rows, int row_offset) {
    int n = src_rows * cols;
    concat_rows_bwd_k<<<blocks(n),BLOCK>>>(g, da, src_rows, cols, total_rows, row_offset);
}

// cross_entropy: one block per row — computes softmax, then CE loss
__global__ void cross_entropy_fwd_k(const float *logits, const float *targets,
                                     float *probs, float *loss_acc,
                                     int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *row_in  = logits  + row * cols;
    const float *row_tgt = targets + row * cols;
    float       *row_prb = probs   + row * cols;

    __shared__ float smem[BLOCK];

    // max
    float mx = -1e38f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) mx = fmaxf(mx, row_in[j]);
    smem[threadIdx.x] = mx;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+s]);
        __syncthreads();
    }
    mx = smem[0];

    // exp + sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) {
        float e = expf(row_in[j] - mx);
        row_prb[j] = e;
        sum += e;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    sum = smem[0];
    for (int j = threadIdx.x; j < cols; j += BLOCK) row_prb[j] /= sum;
    __syncthreads();

    // CE loss for this row, accumulated by thread 0
    if (threadIdx.x == 0) {
        float loss = 0.0f;
        for (int j = 0; j < cols; j++)
            loss -= row_tgt[j] * logf(row_prb[j] + 1e-7f);
        atomicAdd(loss_acc, loss / (float)rows);
    }
}

__global__ void cross_entropy_bwd_k(const float *probs, const float *targets,
                                     const float *g, float *d_logits,
                                     int rows, int cols) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= rows * cols) return;
    float scale = g[0] / (float)rows;
    atomicAdd(&d_logits[idx], scale * (probs[idx] - targets[idx]));
}

void cuda_cross_entropy_fwd(const float *logits, const float *targets,
                             float *probs, float *loss_out,
                             int rows, int cols) {
    cross_entropy_fwd_k<<<rows,BLOCK>>>(logits, targets, probs, loss_out, rows, cols);
}
void cuda_cross_entropy_bwd(const float *probs, const float *targets,
                             const float *g, float *d_logits, int rows, int cols) {
    int n = rows * cols;
    cross_entropy_bwd_k<<<blocks(n),BLOCK>>>(probs, targets, g, d_logits, rows, cols);
}

__global__ void cross_entropy_sparse_fwd_k(const float *logits, const float *ids,
                                            float smoothing, float *probs,
                                            float *loss_acc, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *row_in = logits + row * cols;
    float *row_prb = probs + row * cols;
    int id = (int)ids[row];
    float off = cols > 1 ? smoothing / (float)(cols - 1) : 0.0f;

    __shared__ float smem[BLOCK];

    float mx = -1e38f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) mx = fmaxf(mx, row_in[j]);
    smem[threadIdx.x] = mx;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+s]);
        __syncthreads();
    }
    mx = smem[0];

    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += BLOCK) {
        float e = expf(row_in[j] - mx);
        row_prb[j] = e;
        sum += e;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    sum = smem[0];
    float log_sum_exp = logf(sum) + mx;
    for (int j = threadIdx.x; j < cols; j += BLOCK) row_prb[j] /= sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float loss = 0.0f;
        for (int j = 0; j < cols; j++) {
            float target = (j == id) ? (1.0f - smoothing) : off;
            loss -= target * (row_in[j] - log_sum_exp);
        }
        atomicAdd(loss_acc, loss / (float)rows);
    }
}

__global__ void cross_entropy_sparse_bwd_k(const float *probs, const float *ids,
                                            const float *g, float smoothing,
                                            float *d_logits, int rows, int cols) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= rows * cols) return;
    int row = idx / cols;
    int col = idx % cols;
    int id = (int)ids[row];
    float off = cols > 1 ? smoothing / (float)(cols - 1) : 0.0f;
    float target = (col == id) ? (1.0f - smoothing) : off;
    float scale = g[0] / (float)rows;
    atomicAdd(&d_logits[idx], scale * (probs[idx] - target));
}

void cuda_cross_entropy_sparse_fwd(const float *logits, const float *ids,
                                   float smoothing, float *probs, float *loss_out,
                                   int rows, int cols) {
    cross_entropy_sparse_fwd_k<<<rows,BLOCK>>>(logits, ids, smoothing, probs, loss_out, rows, cols);
}
void cuda_cross_entropy_sparse_bwd(const float *probs, const float *ids,
                                   const float *g, float smoothing, float *d_logits,
                                   int rows, int cols) {
    cross_entropy_sparse_bwd_k<<<blocks(rows * cols),BLOCK>>>(probs, ids, g, smoothing, d_logits, rows, cols);
}

// embed: weight[V×C], ids[T] (float-encoded ints) → out[T×C]
__global__ void embed_fwd_k(const float *weight, const float *ids, float *out, int T, int C) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= T * C) return;
    int t  = idx / C;
    int c  = idx % C;
    int id = (int)ids[t];
    out[idx] = weight[id * C + c];
}
__global__ void embed_bwd_k(const float *g, const float *ids, float *dw, int T, int C) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= T * C) return;
    int t  = idx / C;
    int c  = idx % C;
    int id = (int)ids[t];
    atomicAdd(&dw[id * C + c], g[idx]);
}

void cuda_embed_fwd(const float *weight, const float *ids, float *out, int T, int C)
    { embed_fwd_k<<<blocks(T * C),BLOCK>>>(weight, ids, out, T, C); }
void cuda_embed_bwd(const float *g, const float *ids, float *dw, int T, int C)
    { embed_bwd_k<<<blocks(T * C),BLOCK>>>(g, ids, dw, T, C); }

// dropout: apply pre-computed mask (generated on CPU, stored in cuda_cache)
__global__ void dropout_fwd_k(const float *a, const float *mask, float *out, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) out[i] = a[i] * mask[i];
}
__global__ void dropout_bwd_k(const float *mask, const float *g, float *da, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) atomicAdd(&da[i], mask[i] * g[i]);
}

void cuda_dropout_fwd(const float *a, const float *mask, float *out, int n)
    { dropout_fwd_k<<<blocks(n),BLOCK>>>(a, mask, out, n); }
void cuda_dropout_bwd(const float *mask, const float *g, float *da, int n)
    { dropout_bwd_k<<<blocks(n),BLOCK>>>(mask, g, da, n); }

// ── Adam optimizer ───────────────────────────────────────────────────────────

__global__ void adam_step_k(float *param, float *m, float *v, const float *grad,
                             int n, float lr, float bc1, float bc2,
                             float beta1, float beta2, float eps) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
    param[i] -= lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + eps);
}

void cuda_adam_step(float *param, float *m, float *v, const float *grad,
                    int n, float lr, float bc1, float bc2,
                    float beta1, float beta2, float eps) {
    adam_step_k<<<blocks(n),BLOCK>>>(param, m, v, grad, n, lr, bc1, bc2, beta1, beta2, eps);
}

__global__ void grad_sumsq_k(const float *grad, float *sum_out, int n) {
    __shared__ float smem[BLOCK];
    int i = blockIdx.x * BLOCK + threadIdx.x;
    float v = (i < n) ? grad[i] : 0.0f;
    smem[threadIdx.x] = v * v;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(sum_out, smem[0]);
}

__global__ void scale_grad_k(float *grad, float scale, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) grad[i] *= scale;
}

void cuda_grad_sumsq(const float *grad, float *sum_out, int n)
    { grad_sumsq_k<<<blocks(n),BLOCK>>>(grad, sum_out, n); }
void cuda_scale_grad(float *grad, float scale, int n)
    { scale_grad_k<<<blocks(n),BLOCK>>>(grad, scale, n); }

#endif // OVG_CUDA_ENABLED
