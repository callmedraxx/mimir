/*
 * MIMIR — Meaning (semantic memory)
 *
 * What words mean, learned only from reading, the way children learn most
 * of their words: from the words around them.
 *
 * ════════════════════════════════════════════════════════════════════
 * HOW A WORD GETS ITS MEANING
 * ════════════════════════════════════════════════════════════════════
 *
 * Every word has a fixed random "name": MEANING_NAME_BITS +1s and as many
 * -1s at places picked from a hash of the word (nothing stored, the same
 * on every device).
 *
 *   Day   (reading):  count which words occur near each other, closer
 *                     words counting more (1/distance), up to 8 apart.
 *                     Hebbian: what fires together is counted together.
 *   Night (sleep):    turn the counts into "more together than chance":
 *                     PPMI = max(0, log p(w,u) / (p(w) p(u)^0.75)).
 *                     A word's meaning = the sum of its partners' names,
 *                     each times that strength. Then the part every
 *                     meaning shares is taken away (lateral inhibition).
 *
 * No backprop and nothing global while reading; the night pass is local to
 * each word's counts. Words near in meaning end up with meaning vectors
 * pointing the same way: dog → dogs, breed, pet; money → pay, loans.
 *
 * Measured (sandbox/meaning.c, 32M words: the books + Simple Wikipedia):
 * WordSim-353 0.55, SimLex-999 0.18 at 1024 numbers a word; 0.47 / 0.15
 * at 256 (the small profile shipped in data/meaning/meaning.mng).
 *
 * CAPACITY (a device setting, not a constant): how many numbers per word
 * (MEANING dims) and how many words. A phone or PC takes the large
 * profile, a browser the small one. Reading cost per token is the same.
 *
 * The file lists words most common first, so a word's place is its
 * frequency rank: the MEANING_FUNCTION_WORDS commonest (the, of, and...)
 * are the glue of sentences, not their content.
 * ════════════════════════════════════════════════════════════════════
 */

#include "mimir.h"
#include <ctype.h>
#include <time.h>

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint32_t mhash(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 16777619u; }
    return h;
}

void meaning_init(Meaning *m) { memset(m, 0, sizeof *m); }

void meaning_free(Meaning *m) {
    free(m->word);
    free(m->vec);
    free(m->scale);
    free(m->slot);
    memset(m, 0, sizeof *m);
}

/* The word's id (its frequency rank), or -1. */
int meaning_id(const Meaning *m, const char *w) {
    if (!m->n) return -1;
    for (uint32_t h = mhash(w) & m->mask; m->slot[h]; h = (h + 1) & m->mask)
        if (strcmp(m->word[m->slot[h] - 1], w) == 0) return (int)m->slot[h] - 1;
    return -1;
}

/* Is this word glue (the, of, and...) rather than content? */
int meaning_is_function(const Meaning *m, const char *w) {
    int id = meaning_id(m, w);
    return id >= 0 && id < MEANING_FUNCTION_WORDS;
}

static int build_slots(Meaning *m) {
    m->mask = 1;
    while (m->mask < (uint32_t)m->n * 2) m->mask <<= 1;
    m->slot = calloc(m->mask, sizeof *m->slot);
    if (!m->slot) return -1;
    m->mask -= 1;
    for (int i = 0; i < m->n; i++) {
        uint32_t h = mhash(m->word[i]) & m->mask;
        while (m->slot[h]) h = (h + 1) & m->mask;
        m->slot[h] = (uint32_t)i + 1;
    }
    return 0;
}

/* "MMNG" | int dims | int n | per word: u8 len, chars, float scale, int8[dims] */
int meaning_load(Meaning *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    int dims = 0, n = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MMNG", 4) ||
        fread(&dims, sizeof dims, 1, f) != 1 || fread(&n, sizeof n, 1, f) != 1 ||
        dims <= 0 || dims > MEANING_MAX_DIMS || n <= 0) { fclose(f); return -1; }
    meaning_free(m);
    m->dims = dims;
    m->word = calloc((size_t)n, sizeof *m->word);
    m->vec = malloc((size_t)n * (size_t)dims);
    m->scale = malloc((size_t)n * sizeof *m->scale);
    if (!m->word || !m->vec || !m->scale) { fclose(f); meaning_free(m); return -1; }
    for (int i = 0; i < n; i++) {
        unsigned char l;
        if (fread(&l, 1, 1, f) != 1 || l >= sizeof m->word[0] ||
            fread(m->word[i], 1, l, f) != l || fread(&m->scale[i], sizeof(float), 1, f) != 1 ||
            fread(m->vec + (size_t)i * dims, 1, (size_t)dims, f) != (size_t)dims) { fclose(f); meaning_free(m); return -1; }
        m->word[i][l] = 0;
        m->n = i + 1;
    }
    fclose(f);
    if (build_slots(m)) { meaning_free(m); return -1; }
    return n;
}

int meaning_save(const Meaning *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite("MMNG", 1, 4, f);
    fwrite(&m->dims, sizeof m->dims, 1, f);
    fwrite(&m->n, sizeof m->n, 1, f);
    for (int i = 0; i < m->n; i++) {
        unsigned char l = (unsigned char)strlen(m->word[i]);
        fwrite(&l, 1, 1, f);
        fwrite(m->word[i], 1, l, f);
        fwrite(&m->scale[i], sizeof(float), 1, f);
        fwrite(m->vec + (size_t)i * m->dims, 1, (size_t)m->dims, f);
    }
    return fclose(f) == 0 ? 0 : -1;
}

/* Add a word's meaning (unit length) into acc[dims]. Returns 1 if known. */
int meaning_add(const Meaning *m, const char *w, float *acc) {
    int id = meaning_id(m, w);
    if (id < 0) return 0;
    const int8_t *v = m->vec + (size_t)id * m->dims;
    float n = 0;
    for (int d = 0; d < m->dims; d++) n += (float)v[d] * v[d];
    n = 1.0f / (sqrtf(n) + 1e-9f);
    for (int d = 0; d < m->dims; d++) acc[d] += v[d] * n;
    return 1;
}

static float cosine_ids(const Meaning *m, int a, int b) {
    const int8_t *p = m->vec + (size_t)a * m->dims, *q = m->vec + (size_t)b * m->dims;
    long dot = 0, np = 0, nq = 0;
    for (int d = 0; d < m->dims; d++) { dot += p[d] * q[d]; np += p[d] * p[d]; nq += q[d] * q[d]; }
    return (float)(dot / (sqrt((double)np * (double)nq) + 1e-12));
}

/* How close two words are in meaning, -1..1 (or -2 if either is unknown). */
float meaning_similarity(const Meaning *m, const char *a, const char *b) {
    int x = meaning_id(m, a), y = meaning_id(m, b);
    if (x < 0 || y < 0) return -2;
    return cosine_ids(m, x, y);
}

/* `meaning <word>`: the words closest to it in meaning. */
void meaning_show(const Meaning *m, const char *w) {
    int x = meaning_id(m, w);
    if (x < 0) { printf("  I have no meaning for '%s' yet.\n", w); return; }
    enum { K = 10 };
    int best[K];
    float bs[K];
    for (int k = 0; k < K; k++) { best[k] = -1; bs[k] = -9; }
    for (int v = MEANING_FUNCTION_WORDS; v < m->n; v++) {
        if (v == x) continue;
        float s = cosine_ids(m, x, v);
        for (int k = 0; k < K; k++)
            if (s > bs[k]) {
                for (int j = K - 1; j > k; j--) { bs[j] = bs[j - 1]; best[j] = best[j - 1]; }
                bs[k] = s;
                best[k] = v;
                break;
            }
    }
    printf("  '%s' (number %d of the %d words I know, by how common) is closest to:\n   ", w, x + 1, m->n);
    for (int k = 0; k < K && best[k] >= 0; k++) printf(" %s (%.2f)", m->word[best[k]], bs[k]);
    printf("\n");
}

/* ── the word tests: how close Mimir's sense of meaning is to people's ──
 * Spearman correlation between its similarities and human ratings. */

static void ranks(const double *x, int n, double *r) {
    int *o = malloc((size_t)n * sizeof *o);
    for (int i = 0; i < n; i++) o[i] = i;
    for (int i = 1; i < n; i++) {                    /* insertion sort: n ≤ 1,000 */
        int t = o[i], j = i;
        while (j > 0 && x[o[j - 1]] > x[t]) { o[j] = o[j - 1]; j--; }
        o[j] = t;
    }
    for (int i = 0; i < n;) {
        int j = i;
        while (j + 1 < n && x[o[j + 1]] == x[o[i]]) j++;
        for (int k = i; k <= j; k++) r[o[k]] = (i + j) / 2.0;
        i = j + 1;
    }
    free(o);
}

static double spearman(const double *a, const double *b, int n) {
    double *ra = malloc((size_t)n * sizeof *ra), *rb = malloc((size_t)n * sizeof *rb);
    ranks(a, n, ra);
    ranks(b, n, rb);
    double ma = 0, mb = 0, num = 0, da = 0, db = 0;
    for (int i = 0; i < n; i++) { ma += ra[i]; mb += rb[i]; }
    ma /= n; mb /= n;
    for (int i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - mb); da += (ra[i] - ma) * (ra[i] - ma); db += (rb[i] - mb) * (rb[i] - mb); }
    free(ra);
    free(rb);
    return num / (sqrt(da * db) + 1e-12);
}

/* path: one pair a line, "word1<sep>word2<sep>...rating in column col" */
static void word_test(const Meaning *m, const char *name, const char *path, char sep, int col) {
    FILE *f = fopen(path, "r");
    if (!f) { printf("  %s: could not open %s\n", name, path); return; }
    double mine[1200], human[1200];
    int n = 0, total = 0;
    char line[512];
    if (!fgets(line, sizeof line, f)) { fclose(f); return; }   /* header */
    while (fgets(line, sizeof line, f) && n < 1200) {
        char *field[8];
        int nf = 0;
        for (char *p = line; nf < 8;) {
            field[nf++] = p;
            char *e = strchr(p, sep);
            if (!e) break;
            *e = 0;
            p = e + 1;
        }
        if (nf <= col) continue;
        total++;
        for (char *c = field[0]; *c; c++) *c = (char)tolower((unsigned char)*c);
        for (char *c = field[1]; *c; c++) *c = (char)tolower((unsigned char)*c);
        float s = meaning_similarity(m, field[0], field[1]);
        if (s < -1.5f) continue;
        mine[n] = s;
        human[n] = atof(field[col]);
        n++;
    }
    fclose(f);
    printf("  %-10s %.3f  (on the %d of %d pairs where I know both words)\n", name, n > 2 ? spearman(mine, human, n) : 0.0, n, total);
}

void meaning_test(const Meaning *m) {
    printf("  How well my sense of meaning matches people's (Spearman, 1.0 = perfectly):\n");
    word_test(m, "WordSim", "data/meaning/wordsim353.csv", ',', 2);
    word_test(m, "SimLex", "data/meaning/simlex999.txt", '\t', 3);
    printf("  (WordSim: how related two words are; SimLex: how alike. %d numbers a word, %d words.)\n",
           m->dims, m->n);
}

/* ════════════════════════════════════════════════════════════════════
 * LEARNING: a day of reading, then a night
 * ════════════════════════════════════════════════════════════════════ */

#define WINDOW 8

typedef struct {
    char   **words;              /* all words read, id → text */
    uint32_t *count, *slot, mask;
    int      n, cap;
    uint32_t *text;              /* the words read, in order (ids) */
    size_t   ntext, captext;
} Day;

static int day_id(Day *d, const char *w) {
    if ((uint32_t)d->n * 2 >= d->mask) {
        uint32_t nm = d->mask ? (d->mask + 1) * 2 : 1 << 16;
        uint32_t *ns = calloc(nm, sizeof *ns);
        if (!ns) return -1;
        for (int i = 0; i < d->n; i++) {
            uint32_t h = mhash(d->words[i]) & (nm - 1);
            while (ns[h]) h = (h + 1) & (nm - 1);
            ns[h] = (uint32_t)i + 1;
        }
        free(d->slot);
        d->slot = ns;
        d->mask = nm - 1;
    }
    uint32_t h = mhash(w) & d->mask;
    for (; d->slot[h]; h = (h + 1) & d->mask)
        if (strcmp(d->words[d->slot[h] - 1], w) == 0) return (int)d->slot[h] - 1;
    if (d->n == d->cap) {
        int cap = d->cap ? d->cap * 2 : 1 << 15;
        char **gw = realloc(d->words, (size_t)cap * sizeof *gw);
        uint32_t *gc = realloc(d->count, (size_t)cap * sizeof *gc);
        if (!gw || !gc) return -1;
        d->words = gw;
        d->count = gc;
        d->cap = cap;
    }
    d->words[d->n] = strdup(w);
    d->count[d->n] = 0;
    d->slot[h] = (uint32_t)d->n + 1;
    return d->n++;
}

/* Read a file into the day: every word, in order. */
static long day_read(Day *d, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char w[32];
    int len = 0, c;
    long n = 0;
    do {
        c = fgetc(f);
        if (c != EOF && (isalpha(c) || (c == '\'' && len))) { if (len < 31) w[len++] = (char)tolower(c); continue; }
        if (c == 0xE2) {                              /* ’ inside a word */
            int c2 = fgetc(f), c3 = fgetc(f);
            if (c2 == 0x80 && c3 == 0x99 && len) { if (len < 31) w[len++] = '\''; continue; }
        }
        while (len && w[len - 1] == '\'') len--;
        if (!len) continue;
        w[len] = 0;
        len = 0;
        int id = day_id(d, w);
        if (id < 0) break;
        d->count[id]++;
        if (d->ntext == d->captext) {
            size_t cap = d->captext ? d->captext * 2 : (size_t)1 << 20;
            uint32_t *g = realloc(d->text, cap * sizeof *g);
            if (!g) break;
            d->text = g;
            d->captext = cap;
        }
        d->text[d->ntext++] = (uint32_t)id;
        n++;
    } while (c != EOF);
    fclose(f);
    return n;
}

/* A word's random name: MEANING_NAME_BITS places to add, as many to take away. */
static void name_of(const char *w, int dims, int *plus, int *minus) {
    uint64_t s = mhash(w) * 0x9E3779B97F4A7C15ull + 1;
    for (int k = 0; k < 2 * MEANING_NAME_BITS; k++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        int place = (int)(s % (uint64_t)dims);
        if (k < MEANING_NAME_BITS) plus[k] = place; else minus[k - MEANING_NAME_BITS] = place;
    }
}

/* `meaning learn <dims> <min count> <file>...`: a day of reading the files,
 * then a night that turns the counts into meanings. Replaces the meanings. */
int meaning_learn(Meaning *m, int dims, int minc, char **files, int nfiles) {
    if (dims < 16 || dims > MEANING_MAX_DIMS) { printf("  Numbers a word must be 16 to %d.\n", MEANING_MAX_DIMS); return -1; }
    double t0 = seconds();
    Day d = {0};
    for (int i = 0; i < nfiles; i++) {
        long got = day_read(&d, files[i]);
        printf("  Day: read %s (%ld words)\n", files[i], got < 0 ? 0 : got);
    }
    if (!d.ntext) { printf("  Nothing read.\n"); return -1; }
    /* words that get a meaning, most common first */
    int *ord = malloc((size_t)d.n * sizeof *ord), no = 0;
    int *dix = malloc((size_t)d.n * sizeof *dix);
    for (int v = 0; v < d.n; v++) { dix[v] = -1; if (d.count[v] >= (uint32_t)minc) ord[no++] = v; }
    int gap = 1;
    while (gap < no / 3) gap = gap * 3 + 1;
    for (; gap > 0; gap /= 3)
        for (int i = gap; i < no; i++) {
            int t = ord[i], j = i;
            while (j >= gap && d.count[ord[j - gap]] < d.count[t]) { ord[j] = ord[j - gap]; j -= gap; }
            ord[j] = t;
        }
    for (int i = 0; i < no; i++) dix[ord[i]] = i;
    printf("  %zu words read, %d different; %d seen %d+ times get a meaning\n", d.ntext, d.n, no, minc);

    float *acc = calloc((size_t)no * dims, sizeof(float));
    double *row = calloc((size_t)no, sizeof(double)), *ctx = calloc((size_t)no, sizeof(double)), all = 0;
    size_t cap = (size_t)1 << MEANING_PAIR_BITS, mask = cap - 1;
    uint64_t *key = malloc(cap * sizeof *key);
    float *val = malloc(cap * sizeof *val);
    if (!acc || !row || !ctx || !key || !val) { printf("  Not enough memory for that many words.\n"); return -1; }
    for (size_t i = 0; i < d.ntext; i++) {               /* totals */
        int w = dix[d.text[i]];
        if (w < 0) continue;
        for (int o = -WINDOW; o <= WINDOW; o++) {
            if (!o || (long)i + o < 0 || i + o >= d.ntext) continue;
            int u = dix[d.text[i + o]];
            if (u < 0) continue;
            double wt = 1.0 / abs(o);
            row[w] += wt; ctx[u] += wt; all += wt;
        }
    }
    double ctxall = 0;
    for (int v = 0; v < no; v++) { ctx[v] = pow(ctx[v], 0.75); ctxall += ctx[v]; }
    /* the night, in passes over slices of the words, so it fits in memory */
    int passes = 1;
    for (;;) {
        int ok = 1;
        for (int sh = 0; sh < passes && ok; sh++) {
            memset(key, 0, cap * sizeof *key);
            memset(val, 0, cap * sizeof *val);
            size_t used = 0;
            for (size_t i = 0; i < d.ntext && ok; i++) {
                int w = dix[d.text[i]];
                if (w < 0 || w % passes != sh) continue;
                for (int o = -WINDOW; o <= WINDOW; o++) {
                    if (!o || (long)i + o < 0 || i + o >= d.ntext) continue;
                    int u = dix[d.text[i + o]];
                    if (u < 0) continue;
                    uint64_t k = ((uint64_t)w << 32 | (uint32_t)u) + 1;
                    size_t h = (size_t)(k * 0x9E3779B97F4A7C15ull >> 40) & mask;
                    while (key[h] && key[h] != k) h = (h + 1) & mask;
                    if (!key[h]) { key[h] = k; if (++used > cap * 8 / 10) { ok = 0; break; } }
                    val[h] += (float)(1.0 / abs(o));
                }
            }
            if (!ok) break;
            int plus[MEANING_NAME_BITS], minus[MEANING_NAME_BITS];
            for (size_t h = 0; h < cap; h++) {
                if (!key[h]) continue;
                int w = (int)((key[h] - 1) >> 32), u = (int)((key[h] - 1) & 0xFFFFFFFFu);
                double pmi = log((val[h] / all) / ((row[w] / all) * (ctx[u] / ctxall)));
                if (pmi <= 0) continue;
                float *a = acc + (size_t)w * dims;
                name_of(d.words[ord[u]], dims, plus, minus);
                for (int k = 0; k < MEANING_NAME_BITS; k++) { a[plus[k]] += (float)pmi; a[minus[k]] -= (float)pmi; }
            }
        }
        if (ok) break;
        memset(acc, 0, (size_t)no * dims * sizeof(float));   /* table full: more, smaller passes */
        passes *= 2;
    }
    printf("  Night: %d pass%s\n", passes, passes == 1 ? "" : "es");
    /* lateral inhibition: take away what every meaning has in common */
    double *avg = calloc((size_t)dims, sizeof(double));
    for (int v = 0; v < no; v++) {
        float *a = acc + (size_t)v * dims;
        double n = 0;
        for (int k = 0; k < dims; k++) n += (double)a[k] * a[k];
        n = sqrt(n) + 1e-9;
        for (int k = 0; k < dims; k++) { a[k] = (float)(a[k] / n); avg[k] += a[k] / no; }
    }
    meaning_free(m);
    m->dims = dims;
    m->n = no;
    m->word = calloc((size_t)no, sizeof *m->word);
    m->vec = malloc((size_t)no * dims);
    m->scale = malloc((size_t)no * sizeof *m->scale);
    for (int v = 0; v < no; v++) {
        float *a = acc + (size_t)v * dims, mx = 1e-9f;
        for (int k = 0; k < dims; k++) { a[k] -= (float)avg[k]; if (fabsf(a[k]) > mx) mx = fabsf(a[k]); }
        m->scale[v] = mx / 127.0f;
        for (int k = 0; k < dims; k++) m->vec[(size_t)v * dims + k] = (int8_t)lrintf(a[k] / m->scale[v]);
        snprintf(m->word[v], sizeof m->word[v], "%s", d.words[ord[v]]);
    }
    build_slots(m);
    free(avg); free(acc); free(row); free(ctx); free(key); free(val); free(ord); free(dix);
    for (int i = 0; i < d.n; i++) free(d.words[i]);
    free(d.words); free(d.count); free(d.slot); free(d.text);
    printf("  Learned %d meanings, %d numbers each, in %.1f s\n", m->n, dims, seconds() - t0);
    MIMIR_PROBE("meaning.words", m->n);
    return 0;
}
