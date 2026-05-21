#ifndef TG_TENSOR_H
#define TG_TENSOR_H

#define TG_MAX_PARENTS 8
#define TG_MAX_GRAPH   4096

typedef struct Tensor Tensor;
typedef void (*TgBackwardFn)(Tensor *self);

struct Tensor {
    float        *data;
    float        *grad;
    float        *cache;  // op-specific scratch space (freed by tg_free)
    int           rows;
    int           cols;
    float         aux;
    Tensor       *parents[TG_MAX_PARENTS];
    int           n_parents;
    TgBackwardFn  backward_fn;
    int           persistent; // 1 = skip in tg_free_graph (params, pre-alloc inputs)
    int           visited;    // scratch flag used by topo_sort; always 0 at rest
    // GPU mirror (NULL when not on CUDA; managed by tg_cuda.cu)
    float        *cuda_data;
    float        *cuda_grad;
    float        *cuda_cache; // op-specific scratch on device (e.g. inv_std, probs, mask)
    int           on_cuda;    // 1 = cuda_data/cuda_grad are authoritative
};

Tensor *tg_new(int rows, int cols);
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
