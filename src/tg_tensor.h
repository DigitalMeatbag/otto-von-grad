#ifndef TG_TENSOR_H
#define TG_TENSOR_H

#define TG_MAX_PARENTS 8
#define TG_MAX_GRAPH   8192
#define TG_MAX_DIMS    4

typedef enum { TG_DTYPE_F32 = 0, TG_DTYPE_BF16 = 1 } TgDtype;

typedef struct Tensor Tensor;
typedef void (*TgBackwardFn)(Tensor *self);

struct Tensor {
    int          ndim;
    int          shape[TG_MAX_DIMS];
    TgDtype      dtype;
    void        *data;   /* float * for TG_DTYPE_F32; __nv_bfloat16 * for TG_DTYPE_BF16 */
    float       *grad;   /* always FP32; NULL for non-differentiable tensors */
    float       *cache;  /* op-specific scratch space (freed by tg_free) */
    float        aux;
    Tensor      *parents[TG_MAX_PARENTS];
    int          n_parents;
    TgBackwardFn backward_fn;
    int          persistent; /* 1 = skip in tg_free_graph (params, pre-alloc inputs) */
    int          visited;    /* scratch flag used by topo_sort; always 0 at rest */
    /* GPU mirror (NULL when not on CUDA; managed by tg_cuda.cu) */
    float       *cuda_data;
    float       *cuda_grad;
    float       *cuda_cache; /* op-specific scratch on device */
    int          on_cuda;    /* 1 = cuda_data/cuda_grad are authoritative */
};

/* Cast data pointer to float* — valid when dtype == TG_DTYPE_F32 */
#define TG_DATAF(t) ((float *)(t)->data)

/* Total element count: product of shape[0..ndim-1] */
static inline int tg_numel(const Tensor *t) {
    int n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/* Factory: ndim must be in [2, TG_MAX_DIMS]; shape[] must be positive.
   All tensors are allocated TG_DTYPE_F32 and zeroed. */
Tensor *tg_new(int ndim, const int shape[]);
void    tg_free(Tensor *t);
void    tg_fill(Tensor *t, float val);
void    tg_fill_uniform(Tensor *t, float lo, float hi);
void    tg_fill_randn(Tensor *t, float scale);
void    tg_fill_xavier_uniform(Tensor *t);
void    tg_fill_xavier_normal(Tensor *t);
float   tg_scalar_value(Tensor *t);
void    tg_print(const Tensor *t, const char *name);
void    tg_print_grad(const Tensor *t, const char *name);

#endif
