#include "ovg_test.h"
#include "tg_tokenizer.h"
#include <string.h>
#include <stdlib.h>

static void test_vocab_build(void) {
    const char *text = "abcabc";
    TgVocab v = tg_vocab_build(text, (int)strlen(text));
    OVG_CHECK_EQ(v.size, 3);
    OVG_CHECK(v.ids[(unsigned char)'a'] >= 0);
    OVG_CHECK(v.ids[(unsigned char)'b'] >= 0);
    OVG_CHECK(v.ids[(unsigned char)'c'] >= 0);
    OVG_CHECK_EQ(v.ids[(unsigned char)'z'], -1);
}

static void test_vocab_roundtrip(void) {
    const char *text = "hello world";
    TgVocab v = tg_vocab_build(text, (int)strlen(text));
    for (int i = 0; i < v.size; i++) {
        char c  = v.chars[i];
        int  id = tg_vocab_encode(&v, c);
        OVG_CHECK_EQ(id, i);
        OVG_CHECK_EQ(tg_vocab_decode(&v, id), c);
    }
}

static void test_tokenize_values(void) {
    const char *text = "abc";
    TgVocab v    = tg_vocab_build(text, 3);
    int    *toks = tg_tokenize(text, 3, &v);
    OVG_CHECK_EQ(toks[0], tg_vocab_encode(&v, 'a'));
    OVG_CHECK_EQ(toks[1], tg_vocab_encode(&v, 'b'));
    OVG_CHECK_EQ(toks[2], tg_vocab_encode(&v, 'c'));
    free(toks);
}

void run_tokenizer_tests(int *passed, int *failed) {
    RUN_TEST(test_vocab_build,     passed, failed);
    RUN_TEST(test_vocab_roundtrip, passed, failed);
    RUN_TEST(test_tokenize_values, passed, failed);
}
