/* bpe.c — byte-level BPE tokenizer implementation. See bpe.h. */
#include "bpe.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- byte <-> unicode (GPT-2 reversible map) ---- */
static int printable_byte(int b)
{
    return (b >= 0x21 && b <= 0x7e) || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff);
}

/* ---- UTF-8 ---- */
static int utf8_next(const char *s, int i, int n, int *cp)
{
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c >> 5) == 0x6 && i + 1 < n) { *cp = ((c & 0x1f) << 6) | (s[i+1] & 0x3f); return 2; }
    if ((c >> 4) == 0xe && i + 2 < n) { *cp = ((c & 0x0f) << 12) | ((s[i+1] & 0x3f) << 6) | (s[i+2] & 0x3f); return 3; }
    if ((c >> 3) == 0x1e && i + 3 < n) { *cp = ((c & 0x07) << 18) | ((s[i+1] & 0x3f) << 12) | ((s[i+2] & 0x3f) << 6) | (s[i+3] & 0x3f); return 4; }
    *cp = c; return 1;   /* malformed: treat as one byte */
}

/* character classes used by the pre-tokenizer regex. \p{L}/\p{N} are approximated:
 * ASCII is exact; any non-ASCII, non-space codepoint is treated as a letter, which
 * matches Qwen's regex for Latin/CJK text (English is exact). */
static int is_space_cp(int c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v' ||
           c == 0x85 || c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
           c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}
static int is_digit_cp(int c) { return c >= '0' && c <= '9'; }
static int is_alpha_ascii(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static int is_letter_cp(int c)
{
    if (c < 0x80) return is_alpha_ascii(c);
    return !is_space_cp(c);   /* non-ASCII, non-space -> letter (approx of \p{L}) */
}
static int is_punct_cp(int c) { return !is_space_cp(c) && !is_letter_cp(c) && !is_digit_cp(c); }
static int lower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

/* ---- merge hash (left_id, right_id) -> (rank, merged_id), open addressing ---- */
typedef struct { uint64_t *key; int *rank; int *merged; int cap; } MergeMap;

static uint64_t mkey(int l, int r) { return ((uint64_t)(uint32_t)l << 20) | (uint32_t)r; }

static void mm_init(MergeMap *m, int n)
{
    int cap = 1;
    while (cap < n * 2) cap <<= 1;
    if (cap < 16) cap = 16;
    m->cap = cap;
    m->key = malloc((size_t)cap * sizeof(uint64_t));
    m->rank = malloc((size_t)cap * sizeof(int));
    m->merged = malloc((size_t)cap * sizeof(int));
    for (int i = 0; i < cap; i++) m->key[i] = UINT64_MAX;
}
static void mm_put(MergeMap *m, int l, int r, int rank, int merged)
{
    uint64_t k = mkey(l, r);
    size_t h = (size_t)((k * 1099511628211ULL) & (m->cap - 1));
    while (m->key[h] != UINT64_MAX) { h = (h + 1) & (m->cap - 1); }
    m->key[h] = k; m->rank[h] = rank; m->merged[h] = merged;
}
/* returns rank (>=0) and sets *merged, or -1 if absent */
static int mm_get(const MergeMap *m, int l, int r, int *merged)
{
    uint64_t k = mkey(l, r);
    size_t h = (size_t)((k * 1099511628211ULL) & (m->cap - 1));
    while (m->key[h] != UINT64_MAX) {
        if (m->key[h] == k) { *merged = m->merged[h]; return m->rank[h]; }
        h = (h + 1) & (m->cap - 1);
    }
    return -1;
}

/* ---- tokenizer ---- */
typedef struct { char *str; int len; int id; } Special;

struct BPETokenizer {
    int vocab_size, eos_id, bos_id;
    int byte_to_id[256];
    /* decode: id -> raw bytes */
    char *rawbuf; int *rawoff;   /* rawoff[id]..rawoff[id+1] */
    MergeMap merges;
    Special *specials; int n_special;
    void *owned;   /* blob to free, if we own it */
};

/* build raw-byte decode table by reversing the byte encoder over each id's string */
static void build_decode(BPETokenizer *t, const uint8_t *id2str_off_data,
                         const uint32_t *off, int vocab)
{
    int cp2byte[0x400];
    for (int i = 0; i < 0x400; i++) cp2byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int cp = printable_byte(b) ? b : (256 + n++);
        if (cp < 0x400) cp2byte[cp] = b;
    }
    /* first pass: total raw size */
    size_t total = 0;
    for (int id = 0; id < vocab; id++) {
        const uint8_t *s = id2str_off_data + off[id];
        int slen = (int)(off[id + 1] - off[id]);
        for (int i = 0; i < slen; ) {
            int cp; i += utf8_next((const char *)s, i, slen, &cp);
            total += (cp < 0x400 && cp2byte[cp] >= 0) ? 1 : 0;
        }
    }
    t->rawbuf = malloc(total ? total : 1);
    t->rawoff = malloc((size_t)(vocab + 1) * sizeof(int));
    size_t w = 0;
    for (int id = 0; id < vocab; id++) {
        t->rawoff[id] = (int)w;
        const uint8_t *s = id2str_off_data + off[id];
        int slen = (int)(off[id + 1] - off[id]);
        for (int i = 0; i < slen; ) {
            int cp; i += utf8_next((const char *)s, i, slen, &cp);
            if (cp < 0x400 && cp2byte[cp] >= 0) t->rawbuf[w++] = (char)cp2byte[cp];
        }
    }
    t->rawoff[vocab] = (int)w;
}

static BPETokenizer *parse(const uint8_t *p, size_t size, void *owned)
{
    (void)size;
    if (memcmp(p, "MTOK", 4) != 0) { fprintf(stderr, "bpe: bad magic\n"); return NULL; }
    const uint8_t *q = p + 4;
    int hdr[6]; memcpy(hdr, q, sizeof(hdr)); q += sizeof(hdr);
    /* hdr: version, vocab, n_merges, n_special, eos, bos */
    int vocab = hdr[1], n_merges = hdr[2], n_special = hdr[3];

    BPETokenizer *t = calloc(1, sizeof(*t));
    t->vocab_size = vocab; t->eos_id = hdr[4]; t->bos_id = hdr[5];
    t->owned = owned;

    memcpy(t->byte_to_id, q, 256 * sizeof(int)); q += 256 * sizeof(int);

    /* id -> string table: build an offset index over it (strings stay in place) */
    uint32_t *off = malloc((size_t)(vocab + 1) * sizeof(uint32_t));
    /* copy the string bytes contiguously so we can index by offset */
    const uint8_t *strs_start = q;
    size_t strs_total = 0;
    const uint8_t *scan = q;
    for (int i = 0; i < vocab; i++) {
        uint16_t l; memcpy(&l, scan, 2); scan += 2 + l; strs_total += l;
    }
    uint8_t *strdata = malloc(strs_total ? strs_total : 1);
    scan = strs_start; size_t wo = 0;
    for (int i = 0; i < vocab; i++) {
        off[i] = (uint32_t)wo;
        uint16_t l; memcpy(&l, scan, 2); scan += 2;
        memcpy(strdata + wo, scan, l); scan += l; wo += l;
    }
    off[vocab] = (uint32_t)wo;
    q = scan;

    build_decode(t, strdata, off, vocab);
    free(off); free(strdata);

    /* merges */
    mm_init(&t->merges, n_merges);
    for (int r = 0; r < n_merges; r++) {
        int trip[3]; memcpy(trip, q, sizeof(trip)); q += sizeof(trip);
        mm_put(&t->merges, trip[0], trip[1], r, trip[2]);
    }

    /* specials */
    t->n_special = n_special;
    t->specials = malloc((size_t)n_special * sizeof(Special));
    for (int i = 0; i < n_special; i++) {
        uint16_t l; memcpy(&l, q, 2); q += 2;
        t->specials[i].str = malloc(l + 1);
        memcpy(t->specials[i].str, q, l); t->specials[i].str[l] = 0;
        t->specials[i].len = l; q += l;
        memcpy(&t->specials[i].id, q, 4); q += 4;
    }
    return t;
}

BPETokenizer *bpe_load_from_blob(const void *blob, size_t size, int own)
{
    return parse((const uint8_t *)blob, size, own ? (void *)blob : NULL);
}

BPETokenizer *bpe_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "bpe: can't open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    BPETokenizer *t = parse(buf, sz, buf);
    if (!t) free(buf);
    return t;
}

/* consume one pre-token starting at i; returns the end offset (> i) */
static int pretok(const char *s, int i, int n)
{
    int cp, nb; nb = utf8_next(s, i, n, &cp);

    /* 1. contractions (?i:'s|'t|'re|'ve|'m|'ll|'d) */
    if (cp == '\'' && i + 1 < n) {
        int c1 = lower((unsigned char)s[i + 1]);
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return i + 2;
        if (i + 2 < n) {
            int c2 = lower((unsigned char)s[i + 2]);
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))
                return i + 3;
        }
    }

    /* 2. [^\r\n\p{L}\p{N}]? \p{L}+ */
    if (is_letter_cp(cp)) {
        int j = i;
        while (j < n) { int c, b = utf8_next(s, j, n, &c); if (is_letter_cp(c)) j += b; else break; }
        return j;
    }
    if (cp != '\r' && cp != '\n' && !is_digit_cp(cp)) {
        /* cp is a valid optional prefix; require a letter right after */
        int after = i + nb, c2;
        if (after < n) {
            utf8_next(s, after, n, &c2);
            if (is_letter_cp(c2)) {
                int j = after;
                while (j < n) { int c, b = utf8_next(s, j, n, &c); if (is_letter_cp(c)) j += b; else break; }
                return j;
            }
        }
    }

    /* 3. \p{N} (single) */
    if (is_digit_cp(cp)) return i + nb;

    /* 4. ' ?[^\s\p{L}\p{N}]+[\r\n]* */
    {
        int j = i;
        if (cp == ' ' && i + 1 < n) {
            int c2; utf8_next(s, i + 1, n, &c2);
            if (is_punct_cp(c2)) j = i + 1;
        }
        int c2, b2 = utf8_next(s, j, n, &c2);
        if (j < n && is_punct_cp(c2)) {
            j += b2;
            while (j < n) { int c, b = utf8_next(s, j, n, &c); if (is_punct_cp(c)) j += b; else break; }
            while (j < n && (s[j] == '\r' || s[j] == '\n')) j++;
            return j;
        }
    }

    /* 5/6/7. whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+ */
    {
        int we = i, has_nl = 0, last_nl = -1;
        while (we < n) {
            int c, b = utf8_next(s, we, n, &c);
            if (!is_space_cp(c)) break;
            if (c == '\r' || c == '\n') { has_nl = 1; last_nl = we; }
            we += b;
        }
        if (has_nl) return last_nl + 1;                 /* \s*[\r\n]+ */
        if (we < n && we - i >= 2) return we - 1;       /* \s+(?!\S): leave one for next word */
        return we;                                      /* \s+ */
    }
}

/* BPE-merge a word given as a list of symbol ids (in place); returns new length */
static int bpe_word(const BPETokenizer *t, int *sym, int len)
{
    for (;;) {
        int best = -1, best_rank = 0x7fffffff, best_merged = -1;
        for (int i = 0; i + 1 < len; i++) {
            int merged, r = mm_get(&t->merges, sym[i], sym[i + 1], &merged);
            if (r >= 0 && r < best_rank) { best_rank = r; best = i; best_merged = merged; }
        }
        if (best < 0) break;
        sym[best] = best_merged;
        memmove(&sym[best + 1], &sym[best + 2], (size_t)(len - best - 2) * sizeof(int));
        len--;
    }
    return len;
}

/* longest special token matching at position i, or -1 (its length in *len) */
static int special_at(const BPETokenizer *t, const char *text, int i, int n, int *len)
{
    int sp = -1, sp_len = 0;
    for (int k = 0; k < t->n_special; k++) {
        int l = t->specials[k].len;
        if (l > sp_len && i + l <= n && memcmp(text + i, t->specials[k].str, l) == 0) {
            sp = t->specials[k].id; sp_len = l;
        }
    }
    *len = sp_len;
    return sp;
}

int bpe_encode(const BPETokenizer *t, const char *text, int *out, int max)
{
    int n = (int)strlen(text), i = 0, count = 0;
    int cap = 256; int *sym = malloc(cap * sizeof(int));
    while (i < n && count < max) {
        /* special tokens are split out first, exactly like the reference: they never
         * merge with surrounding text, and the regex never crosses one. */
        int sl, sid = special_at(t, text, i, n, &sl);
        if (sid >= 0) { out[count++] = sid; i += sl; continue; }

        /* normal region runs up to the next special-token boundary */
        int seg_end = n;
        for (int j = i; j < n; j++) {
            int l2; if (special_at(t, text, j, n, &l2) >= 0) { seg_end = j; break; }
        }

        while (i < seg_end && count < max) {
            int end = pretok(text, i, seg_end);
            if (end <= i) end = i + 1;   /* safety */
            int wlen = end - i;
            if (wlen > cap) { cap = wlen * 2; sym = realloc(sym, cap * sizeof(int)); }
            for (int k = 0; k < wlen; k++) sym[k] = t->byte_to_id[(unsigned char)text[i + k]];
            wlen = bpe_word(t, sym, wlen);
            for (int k = 0; k < wlen && count < max; k++) out[count++] = sym[k];
            i = end;
        }
    }
    free(sym);
    return count;
}

const char *bpe_decode(const BPETokenizer *t, int id, int *out_len)
{
    if (id < 0 || id >= t->vocab_size) { *out_len = 0; return ""; }
    *out_len = t->rawoff[id + 1] - t->rawoff[id];
    return t->rawbuf + t->rawoff[id];
}

int bpe_eos(const BPETokenizer *t) { return t->eos_id; }
int bpe_vocab_size(const BPETokenizer *t) { return t->vocab_size; }

void bpe_free(BPETokenizer *t)
{
    if (!t) return;
    free(t->rawbuf); free(t->rawoff);
    free(t->merges.key); free(t->merges.rank); free(t->merges.merged);
    for (int i = 0; i < t->n_special; i++) free(t->specials[i].str);
    free(t->specials);
    free(t->owned);
    free(t);
}
