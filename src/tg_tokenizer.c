#include "tg_tokenizer.h"
#include "ovg_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *tg_read_file(const char *path, int *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) ovg_fatal("tg_read_file: could not open: %s", path);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char *buf = malloc((size_t)size + 1);
    if (!buf) ovg_fatal("tg_read_file: out of memory");
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        ovg_fatal("tg_read_file: read error: %s", path);
    }
    buf[size] = '\0';
    fclose(f);
    *out_len = (int)size;
    return buf;
}

TgVocab tg_vocab_build(const char *text, int len) {
    TgVocab v;
    memset(v.ids, -1, sizeof(v.ids));
    v.size = 0;
    unsigned char seen[256] = {0};
    for (int i = 0; i < len; i++) seen[(unsigned char)text[i]] = 1;
    for (int c = 0; c < 256; c++) {
        if (seen[c]) { v.chars[v.size] = (char)c; v.ids[c] = v.size++; }
    }
    return v;
}

TgVocab tg_vocab_from_chars(const char *chars) {
    TgVocab v;
    memset(v.ids, -1, sizeof(v.ids));
    v.size = 0;
    unsigned char seen[256] = {0};
    for (int i = 0; chars[i] != '\0'; i++) seen[(unsigned char)chars[i]] = 1;
    for (int c = 0; c < 256; c++) {
        if (seen[c]) { v.chars[v.size] = (char)c; v.ids[c] = v.size++; }
    }
    return v;
}

int tg_vocab_encode(const TgVocab *v, char c) {
    int id = v->ids[(unsigned char)c];
    if (id < 0) ovg_fatal("tg_vocab_encode: unknown char 0x%02x", (unsigned char)c);
    return id;
}

char tg_vocab_decode(const TgVocab *v, int id) {
    if (id < 0 || id >= v->size) ovg_fatal("tg_vocab_decode: bad id %d", id);
    return v->chars[id];
}

int *tg_tokenize(const char *text, int len, const TgVocab *v) {
    int *tokens = malloc((size_t)len * sizeof(int));
    if (!tokens) ovg_fatal("tg_tokenize: out of memory");
    for (int i = 0; i < len; i++) tokens[i] = tg_vocab_encode(v, text[i]);
    return tokens;
}
