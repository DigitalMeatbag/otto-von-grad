#ifndef SMOKE_TESTS_H
#define SMOKE_TESTS_H

void attention_smoke_test(int T, int C, int n_heads);
void encoder_smoke_test(int T, int C);
void mean_rows_smoke_test(void);
void transpose_grad_accum_test(void);
void collect_params_capacity_test(void);
void embed_smoke_test(void);

#ifdef OVG_CUDA_ENABLED
void cuda_causal_mask_large_test(void);
#endif

#endif
