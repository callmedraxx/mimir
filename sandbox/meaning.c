/*
 * meaning.c — can Mimir learn what words mean just by reading, with a
 * brain-style local rule?  Math first, code later (like reader_math.py).
 *
 * Rule (random indexing, Hebbian): every word has a fixed random sparse
 * ±1 "name" (its index vector). A word's meaning vector adds the names of
 * the words around it each time it is read. Frequent words are seen less
 * (habituation: an occurrence is skipped with probability 1 - sqrt(t/f)).
 * No backprop, nothing global, one pass over the text.
 *
 * Scored against people: WordSim-353 (how related) and SimLex-999 (how
 * alike), Spearman correlation of cosine similarity with the human ratings.
 *
 *   cc -O3 -o meaning sandbox/meaning.c -lm
 *   PPMI=1 MINC=40 SHARDS=8 BITS=24 OUT=meaning.mng ./meaning D W 0 books/...txt < pairs.txt
 *   (test pairs "a b" on stdin; OUT saves the meanings, most common word first)
 */
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXV (1 << 21)
#define KNZ 8                     /* +1s and -1s in each word's name */

static char *words[MAXV];
static uint32_t counts[MAXV], slot[MAXV * 2];
static int32_t dix[MAXV];      /* dense index of words seen MINC+ times, or -1 */
static int nv;
static uint32_t *text;
static size_t ntext, captext;

static uint32_t h32(const char *s) { uint32_t h = 2166136261u; for (; *s; s++) { h ^= (unsigned char)*s; h *= 16777619u; } return h; }

static int id_of(const char *w, int add) {
    uint32_t h = h32(w) & (MAXV * 2 - 1);
    for (; slot[h]; h = (h + 1) & (MAXV * 2 - 1))
        if (strcmp(words[slot[h] - 1], w) == 0) return (int)slot[h] - 1;
    if (!add || nv >= MAXV - 1) return -1;
    words[nv] = strdup(w);
    slot[h] = (uint32_t)++nv;
    return nv - 1;
}

static void read_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    char w[64];
    int len = 0, c;
    do {
        c = fgetc(f);
        if (c != EOF && (isalpha(c) || (c == '\'' && len))) { if (len < 63) w[len++] = (char)tolower(c); continue; }
        if (c == 0xE2) { /* curly apostrophe ’ inside a word */
            int c2 = fgetc(f), c3 = fgetc(f);
            if (c2 == 0x80 && c3 == 0x99 && len) { if (len < 63) w[len++] = '\''; continue; }
        }
        if (!len) continue;
        while (len && w[len - 1] == '\'') len--;
        w[len] = 0;
        len = 0;
        if (!w[0]) continue;
        int id = id_of(w, 1);
        if (id < 0) continue;
        counts[id]++;
        if (ntext == captext) { captext = captext ? captext * 2 : 1 << 20; text = realloc(text, captext * sizeof *text); }
        text[ntext++] = (uint32_t)id;
    } while (c != EOF);
    fclose(f);
}

static uint64_t rng = 88172645463325252ull;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: meaning D W T files...\n"); return 1; }
    int D = atoi(argv[1]), W = atoi(argv[2]);
    double T = atof(argv[3]);
    for (int i = 4; i < argc; i++) read_file(argv[i]);
    fprintf(stderr, "%zu tokens, %d different words\n", ntext, nv);

    /* names: KNZ +1 and KNZ -1 at random places (stored as positions) */
    int16_t (*name)[2 * KNZ] = malloc((size_t)nv * sizeof *name);
    for (int v = 0; v < nv; v++) for (int k = 0; k < 2 * KNZ; k++) name[v][k] = (int16_t)(xr() % (uint64_t)D);
    /* meanings only for words seen at least MINC times (memory) */
    unsigned minc0 = getenv("MINC") ? (unsigned)atoi(getenv("MINC")) : 1;
    int nd = 0;
    for (int v = 0; v < nv; v++) dix[v] = counts[v] >= minc0 ? nd++ : -1;
    float *mean = calloc((size_t)nd * D, sizeof(float));
    if (!mean) { fprintf(stderr, "out of memory\n"); return 1; }
    fprintf(stderr, "%d words get a meaning (seen %u+ times)\n", nd, minc0);
#define MEAN(v) (mean + (size_t)dix[v] * D)

    double total = (double)ntext;
    if (getenv("PPMI")) {
        /* Day: count which words occur together (Hebbian counts), 1/d weighted.
         * Night: strength = how much more together than chance,
         * PPMI = max(0, log(p(w,u) / (p(w) p_ctx(u)))), with the context
         * counts raised to 0.75 (rarer contexts count a bit more), then each
         * meaning = sum of strength x name of the other word.
         * Memory is bounded: the night runs in SHARDS passes, each over the
         * words w with w % SHARDS == pass (like several sleep cycles), and
         * words seen fewer than MINC times are left out. */
        int shards = getenv("SHARDS") ? atoi(getenv("SHARDS")) : 1;
        unsigned minc = getenv("MINC") ? (unsigned)atoi(getenv("MINC")) : 1;
        size_t cap = (size_t)1 << (getenv("BITS") ? atoi(getenv("BITS")) : 24), mask = cap - 1;
        uint64_t *key = malloc(cap * sizeof *key);
        float *val = malloc(cap * sizeof *val);
        double *rowsum = calloc((size_t)nv, sizeof(double));
        double *ctx = calloc((size_t)nv, sizeof(double)), all = 0;
        if (!key || !val || !rowsum || !ctx) { fprintf(stderr, "out of memory\n"); return 1; }
        for (size_t i = 0; i < ntext; i++) {                 /* the totals first */
            uint32_t w = text[i];
            if (counts[w] < minc) continue;
            for (int d = -W; d <= W; d++) {
                if (!d || (long)i + d < 0 || i + d >= ntext) continue;
                uint32_t u = text[i + d];
                if (counts[u] < minc) continue;
                float wt = (float)(1.0 / abs(d));
                rowsum[w] += wt; ctx[u] += wt; all += wt;
            }
        }
        double ctxall = 0;
        for (int v = 0; v < nv; v++) { ctx[v] = pow(ctx[v], 0.75); ctxall += ctx[v]; }
        size_t most = 0;
        for (int sh = 0; sh < shards; sh++) {
            memset(key, 0, cap * sizeof *key);
            memset(val, 0, cap * sizeof *val);
            size_t used = 0;
            for (size_t i = 0; i < ntext; i++) {
                uint32_t w = text[i];
                if (counts[w] < minc || (int)(w % (uint32_t)shards) != sh) continue;
                for (int d = -W; d <= W; d++) {
                    if (!d || (long)i + d < 0 || i + d >= ntext) continue;
                    uint32_t u = text[i + d];
                    if (counts[u] < minc) continue;
                    uint64_t k = ((uint64_t)w << 32 | u) + 1;
                    size_t h = (size_t)(k * 0x9E3779B97F4A7C15ull >> 40) & mask;
                    while (key[h] && key[h] != k) h = (h + 1) & mask;
                    if (!key[h]) { key[h] = k; if (++used > cap * 8 / 10) { fprintf(stderr, "pair table full: raise BITS or SHARDS\n"); return 1; } }
                    val[h] += (float)(1.0 / abs(d));
                }
            }
            if (used > most) most = used;
            for (size_t h = 0; h < cap; h++) {
                if (!key[h]) continue;
                uint32_t w = (uint32_t)((key[h] - 1) >> 32), u = (uint32_t)((key[h] - 1) & 0xFFFFFFFFu);
                double pmi = log((val[h] / all) / ((rowsum[w] / all) * (ctx[u] / ctxall)));
                if (pmi <= 0) continue;
                float *m = MEAN(w);
                for (int k = 0; k < KNZ; k++) m[name[u][k]] += (float)pmi;
                for (int k = KNZ; k < 2 * KNZ; k++) m[name[u][k]] -= (float)pmi;
            }
        }
        fprintf(stderr, "%d night passes, at most %zu different pairs in one\n", shards, most);
        free(key); free(val);
    } else
    for (size_t i = 0; i < ntext; i++) {
        uint32_t w = text[i];
        if (dix[w] < 0) continue;
        double f = counts[w] / total;
        if (T > 0 && f > T && (double)(xr() % 1000000) / 1e6 > sqrt(T / f)) continue;   /* habituation */
        float *m = MEAN(w);
        for (int d = -W; d <= W; d++) {
            if (!d || (long)i + d < 0 || i + d >= ntext) continue;
            uint32_t u = text[i + d];
            double fu = counts[u] / total;
            float wt = (float)(1.0 / abs(d)) * (float)(T > 0 && fu > T ? sqrt(T / fu) : 1.0);
            for (int k = 0; k < KNZ; k++) m[name[u][k]] += wt;
            for (int k = KNZ; k < 2 * KNZ; k++) m[name[u][k]] -= wt;
        }
    }
    /* lateral inhibition: take away what every meaning has in common */
    double *avg = calloc((size_t)D, sizeof(double));
    for (int v = 0; v < nd; v++) { float *m = mean + (size_t)v * D; double n = 0; for (int d = 0; d < D; d++) n += m[d] * m[d]; n = sqrt(n) + 1e-9; for (int d = 0; d < D; d++) avg[d] += m[d] / n / nd; }
    for (int v = 0; v < nd; v++) { float *m = mean + (size_t)v * D; double n = 0; for (int d = 0; d < D; d++) n += m[d] * m[d]; n = sqrt(n) + 1e-9; for (int d = 0; d < D; d++) m[d] = (float)(m[d] / n - avg[d]); }

    /* save the meanings: "MMNG" | int D | int n | per word: u8 len, chars,
     * float scale, int8[D]   (words seen at least MINC times) */
    if (getenv("OUT")) {
        unsigned minc = getenv("MINC") ? (unsigned)atoi(getenv("MINC")) : 1;
        FILE *o = fopen(getenv("OUT"), "wb");
        int n = 0;
        for (int v = 0; v < nv; v++) n += counts[v] >= minc;
        fwrite("MMNG", 1, 4, o); fwrite(&D, sizeof D, 1, o); fwrite(&n, sizeof n, 1, o);
        int8_t *q = malloc((size_t)D);
        /* most common first: a word's place in the file is its frequency rank,
         * so a reader can tell function words (the, and, of) from the rest */
        int *ord = malloc((size_t)nv * sizeof *ord), no = 0;
        for (int v = 0; v < nv; v++) if (counts[v] >= minc) ord[no++] = v;
        {
            int gap = 1;
            while (gap < no / 3) gap = gap * 3 + 1;
            for (; gap > 0; gap /= 3)
                for (int i = gap; i < no; i++) {
                    int t = ord[i], j = i;
                    while (j >= gap && counts[ord[j - gap]] < counts[t]) { ord[j] = ord[j - gap]; j -= gap; }
                    ord[j] = t;
                }
        }
        for (int oi = 0; oi < no; oi++) {
            int v = ord[oi];
            float *m = MEAN(v), mx = 1e-9f;
            for (int d = 0; d < D; d++) if (fabsf(m[d]) > mx) mx = fabsf(m[d]);
            float sc = mx / 127.0f;
            for (int d = 0; d < D; d++) q[d] = (int8_t)lrintf(m[d] / sc);
            unsigned char l = (unsigned char)strlen(words[v]);
            fwrite(&l, 1, 1, o); fwrite(words[v], 1, l, o); fwrite(&sc, sizeof sc, 1, o); fwrite(q, 1, (size_t)D, o);
        }
        fclose(o);
        fprintf(stderr, "saved %d meanings to %s\n", n, getenv("OUT"));
    }

    /* similarities for the test pairs on stdin: "a b" → "a b cos" or "a b NA" */
    char a[64], b[64];
    while (scanf("%63s %63s", a, b) == 2) {
        int x = id_of(a, 0), y = id_of(b, 0);
        if (x < 0 || y < 0 || counts[x] < 5 || counts[y] < 5 || dix[x] < 0 || dix[y] < 0) { printf("%s %s NA\n", a, b); continue; }
        double dot = 0, nx = 0, ny = 0;
        float *p = MEAN(x), *q = MEAN(y);
        for (int d = 0; d < D; d++) { dot += p[d] * q[d]; nx += p[d] * p[d]; ny += q[d] * q[d]; }
        printf("%s %s %.5f\n", a, b, dot / (sqrt(nx * ny) + 1e-12));
    }
    /* nearest neighbours for a few words (stderr) */
    const char *probe[] = { "doctor", "king", "happy", "dog", "money", "night", "ship", "mother" };
    for (int pi = 0; pi < 8; pi++) {
        int x = id_of(probe[pi], 0);
        if (x < 0) continue;
        int best[6] = { -1, -1, -1, -1, -1, -1 };
        double bs[6] = { -9, -9, -9, -9, -9, -9 };
        if (dix[x] < 0) continue;
        float *p = MEAN(x);
        double nx = 0; for (int d = 0; d < D; d++) nx += p[d] * p[d];
        for (int v = 0; v < nv; v++) {
            if (v == x || counts[v] < 20 || dix[v] < 0) continue;
            float *q = MEAN(v);
            double dot = 0, ny = 0;
            for (int d = 0; d < D; d++) { dot += p[d] * q[d]; ny += q[d] * q[d]; }
            double s = dot / (sqrt(nx * ny) + 1e-12);
            for (int k = 0; k < 6; k++) if (s > bs[k]) { for (int j = 5; j > k; j--) { bs[j] = bs[j - 1]; best[j] = best[j - 1]; } bs[k] = s; best[k] = v; break; }
        }
        fprintf(stderr, "  %-8s →", probe[pi]);
        for (int k = 0; k < 6; k++) if (best[k] >= 0) fprintf(stderr, " %s", words[best[k]]);
        fprintf(stderr, "\n");
    }
    return 0;
}
