#ifndef OVG_TEST_H
#define OVG_TEST_H

#include <math.h>
#include <stdio.h>

/* Defined in test_main.c; each test function reads/writes this. */
extern int ovg_test_failed;

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

/* N-D shape check: verifies ndim and each shape dimension */
#define OVG_CHECK_SHAPE_ND(t, nd, ...) do { \
    int _expected[] = {__VA_ARGS__}; \
    if ((t)->ndim != (nd)) { \
        fprintf(stderr, "  %s:%d: FAIL — ndim %d != %d\n", \
                __FILE__, __LINE__, (t)->ndim, (nd)); \
        ovg_test_failed = 1; \
        return; \
    } \
    for (int _i = 0; _i < (nd); _i++) { \
        if ((t)->shape[_i] != _expected[_i]) { \
            fprintf(stderr, "  %s:%d: FAIL — shape[%d] %d != %d\n", \
                    __FILE__, __LINE__, _i, (t)->shape[_i], _expected[_i]); \
            ovg_test_failed = 1; \
            return; \
        } \
    } \
} while (0)

/* Backward-compatible 2D shape check */
#define OVG_CHECK_SHAPE(t, r, c) do { \
    if ((t)->ndim != 2 || (t)->shape[0] != (r) || (t)->shape[1] != (c)) { \
        fprintf(stderr, "  %s:%d: FAIL — shape [%dx%d ndim=%d] != [%dx%d]\n", \
                __FILE__, __LINE__, (t)->shape[0], (t)->shape[1], (t)->ndim, (r), (c)); \
        ovg_test_failed = 1; \
        return; \
    } \
} while (0)

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
