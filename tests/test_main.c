#include "ovg_test.h"
#include "tg_rng.h"
#include <stdio.h>
#include <stdlib.h>

int ovg_test_failed = 0;

void run_ops_tests(int *passed, int *failed);
void run_train_tests(int *passed, int *failed);
void run_attention_tests(int *passed, int *failed);
void run_gpt_tests(int *passed, int *failed);

int main(void) {
    tg_seed(42);
    int passed = 0, failed = 0;

    printf("=== ops ===\n");
    run_ops_tests(&passed, &failed);

    printf("=== train ===\n");
    run_train_tests(&passed, &failed);

    printf("=== attention ===\n");
    run_attention_tests(&passed, &failed);

    printf("=== gpt ===\n");
    run_gpt_tests(&passed, &failed);

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
