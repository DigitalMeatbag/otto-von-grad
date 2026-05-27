#ifndef TG_TOKENIZER_H
#define TG_TOKENIZER_H

typedef struct {
    char chars[256];
    int  ids[256];
    int  size;
} TgVocab;

/* Read entire file into a malloc'd buffer; *out_len receives byte count.
   Caller must free the returned pointer. Calls exit(1) on error. */
char    *tg_read_file(const char *path, int *out_len);

/* Build a vocab from every unique byte in text[0..len). */
TgVocab  tg_vocab_build(const char *text, int len);

/* Encode a single character; calls ovg_fatal on unknown char. */
int      tg_vocab_encode(const TgVocab *v, char c);

/* Decode a token id back to a character; calls ovg_fatal on bad id. */
char     tg_vocab_decode(const TgVocab *v, int id);

/* Tokenize text[0..len). Returns a malloc'd int array of length len; caller must free. */
int     *tg_tokenize(const char *text, int len, const TgVocab *v);

#endif /* TG_TOKENIZER_H */
