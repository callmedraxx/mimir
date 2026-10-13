/*
 * topic_split.c — docs/TOPIC_COMPARE.md: split English Wikipedia into on-topic (science) and general articles.
 * A word's weight is log p(w | ARC Corpus) − log p(w | Simple Wikipedia) (counts + 1, words seen >= 20 times);
 * an article's score is the mean weight of its counted words (fewer than 20 counted words: general). The top 20%
 * of enwiki_00's articles set the cut-off, used unchanged for every file.
 *
 *   gcc -O2 -o topic_split tools/topic_split.c -lm
 *   ./topic_split OUTDIR data/shelf/enwiki_*.txt      (run from the repo root; writes OUTDIR/sci_NN.txt, gen_NN.txt)
 */
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TB (1u << 24)
static uint64_t *key; static uint32_t *c1, *c2;
static uint64_t N1, N2;
static uint64_t whash(const char *s, int n) { uint64_t h = 1469598103934665603ull; for (int i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ull; } return h | 1; }
static uint32_t slot(uint64_t h, int add) {
    uint32_t i = (uint32_t)(h >> 7) & (TB - 1);
    while (key[i] && key[i] != h) i = (i + 1) & (TB - 1);
    if (!key[i] && add) key[i] = h;
    return i;
}
/* calls fn on each lowercase run of letters of b[0..n) */
#define WORDS(b, n, BODY) do { int _l = 0; char _w[64]; for (size_t _i = 0; _i <= (n); _i++) { int _c = _i < (n) ? (unsigned char)(b)[_i] : 0; \
    if (isalpha(_c)) { if (_l < 63) _w[_l++] = (char)tolower(_c); } else if (_l) { const char *w = _w; int wl = _l; BODY; _l = 0; } } } while (0)

static void count(const char *path, uint32_t *c, uint64_t *N) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    static char buf[1 << 20]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) {
        size_t cut = n; if (n == sizeof buf) while (cut > 0 && isalpha((unsigned char)buf[cut - 1])) cut--;
        WORDS(buf, cut, { uint32_t i = slot(whash(w, wl), 1); c[i]++; (*N)++; });
        if (cut < n) fseek(f, -(long)(n - cut), SEEK_CUR);
    }
    fclose(f);
}
static double weight(const char *w, int wl, int *ok) {
    uint32_t i = slot(whash(w, wl), 0);
    if (!key[i] || c1[i] + c2[i] < 20) { *ok = 0; return 0; }
    *ok = 1; return log((c1[i] + 1.0) / N1) - log((c2[i] + 1.0) / N2);
}
static double score(const char *a, size_t n, int *nw) {
    double s = 0; int k = 0; WORDS(a, n, { int ok; double x = weight(w, wl, &ok); if (ok) { s += x; k++; } });
    *nw = k; return k >= 20 ? s / k : -1e9;
}
/* calls BODY for each article (blank-line separated) of a file, with a, an */
static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); } fseeko(f, 0, SEEK_END); *n = (size_t)ftello(f); fseeko(f, 0, SEEK_SET);
    char *b = malloc(*n + 1); if (fread(b, 1, *n, f) != *n) { perror(path); exit(1); } b[*n] = 0; fclose(f); return b;
}
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? 1 : x > y ? -1 : 0; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: topic_split OUTDIR enwiki_NN.txt...\n"); return 1; }
    key = calloc(TB, 8); c1 = calloc(TB, 4); c2 = calloc(TB, 4);
    count("data/shelf/arc_corpus.txt", c1, &N1); count("data/shelf/simplewiki.txt", c2, &N2);
    fprintf(stderr, "science %llu words, general %llu words\n", (unsigned long long)N1, (unsigned long long)N2);
    double cut = 0;
    for (int fi = 2; fi < argc; fi++) {
        size_t n; char *b = slurp(argv[fi], &n);
        if (fi == 2) {                     /* the cut-off: top 20% of the first file's articles */
            double *sc = NULL; size_t ns = 0, cap = 0;
            for (char *a = b, *e; a < b + n; a = e + 2) { e = strstr(a, "\n\n"); if (!e) e = b + n; int nw; double s = score(a, e - a, &nw);
                if (ns == cap) { cap = cap ? cap * 2 : 1 << 16; sc = realloc(sc, cap * sizeof *sc); } sc[ns++] = s; if (e == b + n) break; }
            qsort(sc, ns, sizeof *sc, cmpd); cut = sc[ns / 5]; free(sc);
            fprintf(stderr, "cut-off score %.4f (top 20%% of %zu articles in %s)\n", cut, ns, argv[fi]);
        }
        const char *base = strrchr(argv[fi], '_'); char ps[512], pg[512]; snprintf(ps, sizeof ps, "%s/sci%s", argv[1], base); snprintf(pg, sizeof pg, "%s/gen%s", argv[1], base);
        FILE *fs = fopen(ps, "wb"), *fg = fopen(pg, "wb"); if (!fs || !fg) { perror(ps); return 1; }
        size_t na = 0, nsci = 0, bs = 0;
        for (char *a = b, *e; a < b + n; a = e + 2) { e = strstr(a, "\n\n"); if (!e) e = b + n; int nw; double s = score(a, e - a, &nw); na++;
            FILE *o = s > cut ? fs : fg; if (o == fs) { nsci++; bs += e - a + 2; } fwrite(a, 1, e - a, o); fwrite("\n\n", 1, 2, o); if (e == b + n) break; }
        fclose(fs); fclose(fg); free(b);
        fprintf(stderr, "%s: %zu articles, %zu on topic (%.1f%%), %.0f MB on topic\n", argv[fi], na, nsci, 100.0 * nsci / na, bs / 1e6);
    }
    return 0;
}
