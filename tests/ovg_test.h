#ifndef OVG_TEST_H
#define OVG_TEST_H

#include <math.h>
#include <stdio.h>

/* Defined in test_main.c; each test function reads/writes this. */
extern int ovg_test_failed;

/*
 * On failure: print file:line diagnostic, mark the current test failed,
 * and return from the calling (void) test function.  Subsequent checks in
 * the same test are skipped, preventing NULL-dereference cascades.
 */
#define OVG_CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "  %s:%d: FAIL — %s\n", __FILE__, __LINE__, #cond); \
        ovg_test_failed = 1; \
        return; \
    } \
} while (0)

#define OVG_CHECK_EQ(a, b) do { \
    if ((a) != (b)) { \
        fprintf(stderr, "  %s:%d: FAIL — %s (%d) != %s (%d)\n", \
                __FILE__, __LINE__, #a, (int)(a), #b, (int)(b)); \
        ovg_test_failed = 1; \
        return; \
    } \
} while (0)

#define OVG_CHECK_NEAR(a, b, tol) do { \
    float _a = (float)(a), _b = (float)(b), _t = (float)(tol); \
    if (fabsf(_a - _b) > _t) { \
        fprintf(stderr, "  %s:%d: FAIL — |%s - %s| = %.7f > %.7f\n", \
                __FILE__, __LINE__, #a, #b, fabsf(_a - _b), _t); \
        ovg_test_failed = 1; \
        return; \
    } \
} while (0)

#define OVG_CHECK_SHAPE(t, r, c) do { \
    if ((t)->rows != (r) || (t)->cols != (c)) { \
        fprintf(stderr, "  %s:%d: FAIL — shape [%dx%d] != [%dx%d]\n", \
                __FILE__, __LINE__, (t)->rows, (t)->cols, (r), (c)); \
        ovg_test_failed = 1; \
        return; \
    } \
} while (0)

/* Runs a void test function, resets ovg_test_failed before each call. */
#define RUN_TEST(fn, passed, failed) do { \
    ovg_test_failed = 0; \
    fn(); \
    if (ovg_test_failed) { \
        fprintf(stderr, "FAIL: " #fn "\n"); \
        (*(failed))++; \
    } else { \
        printf("pass: " #fn "\n"); \
        (*(passed))++; \
    } \
} while (0)

#endif /* OVG_TEST_H */
