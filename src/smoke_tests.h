#ifndef SMOKE_TESTS_H
#define SMOKE_TESTS_H

void attention_smoke_test(int T, int C, int n_heads);
void encoder_smoke_test(int T, int C);
void mean_rows_smoke_test(void);

#endif
