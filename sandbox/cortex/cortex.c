/* Mimir's cortex, v0 (docs/CORTEX.md): words as sparse patterns over N units, learned with local rules.
 *
 *   cortex vocab  VOCAB TEXT...                      the V most frequent words (held-out paragraphs not counted)
 *   cortex items  VOCAB DEV TEST TEXT...             T2 fill-the-blank items from the held-out paragraphs
 *   cortex train  CELL VOCAB MODEL TEXT...           CELL = K0 (random), K1 (Hebbian), K2 (predictive)
 *   cortex eval   MODEL VOCAB                        T1 (WS=files), T2 (T2=items), T3 (STORIES=tsv SPLIT=dev|test)
 *   cortex near   MODEL VOCAB word...                the nearest words (to look at)
 *
 * Held out: every 200th line (paragraph) of the text, counted over all files in the order given; never trained on.
 * Settings (env): V N K WIN ETA ETA2 BETA SUB MAXTOK SEED SIM (see train()).
 */
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int V = 50000, N = 2048, K = 40, WIN = 2, M = 128;
static double ETA = 0.05, ETA2 = 0.01, BETA = 2.0, SUB = 1e-4, IDW = 0.5;
static long MAXTOK = 0; static uint64_t SEED = 12345;
static int envi(const char *n, int d) { return getenv(n) ? atoi(getenv(n)) : d; }
static double envd(const char *n, double d) { return getenv(n) ? atof(getenv(n)) : d; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double rndf(void) { return (rnd() >> 11) * (1.0 / 9007199254740992.0); }

/* ---- words */
typedef struct { char *w; uint64_t n; } Ent;
static Ent *H; static uint64_t hcap, hn;
static uint64_t fnv(const char *s) { uint64_t h = 1469598103934665603ull; while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ull; } return h; }
static Ent *hget(const char *w, int add) {
    if (add && hn * 2 >= hcap) { Ent *o = H; uint64_t oc = hcap; hcap = hcap ? hcap * 2 : 1 << 20; H = calloc(hcap, sizeof *H); hn = 0;
        for (uint64_t i = 0; i < oc; i++) if (o[i].w) { Ent *e = hget(o[i].w, 1); e->w = o[i].w; e->n = o[i].n; } free(o); }
    if (!hcap) return 0;
    uint64_t h = fnv(w) & (hcap - 1);
    while (H[h].w) { if (!strcmp(H[h].w, w)) return &H[h]; h = (h + 1) & (hcap - 1); }
    if (!add) return 0;
    H[h].w = strdup(w); H[h].n = 0; hn++; return &H[h];
}
static char **voc; static int nvoc; static uint64_t *vcnt, vtot;
static int wid(const char *w) { Ent *e = hget(w, 0); return e ? (int)e->n : -1; }   /* after vocab_load, n holds the id */
static void vocab_load(const char *p) {
    FILE *f = fopen(p, "r"); if (!f) { perror(p); exit(1); } char l[512];
    voc = malloc(V * sizeof *voc); vcnt = malloc(V * sizeof *vcnt); nvoc = 0; vtot = 0;
    while (nvoc < V && fgets(l, sizeof l, f)) { char *t = strchr(l, '\t'); if (!t) continue; *t = 0;
        voc[nvoc] = strdup(l); vcnt[nvoc] = strtoull(t + 1, 0, 10); vtot += vcnt[nvoc]; Ent *e = hget(l, 1); e->n = nvoc++; }
    fclose(f); V = nvoc;
}

/* ---- text: lines (paragraphs) in order; every 200th is held out */
typedef void (*LineFn)(char *line, int held);
static long lineno;
static void each_line(int nf, char **files, LineFn fn) {
    static char buf[1 << 20];
    for (int i = 0; i < nf; i++) { FILE *f = fopen(files[i], "r"); if (!f) { perror(files[i]); exit(1); }
        while (fgets(buf, sizeof buf, f)) { int held = lineno++ % 200 == 0; fn(buf, held); if (MAXTOK < 0) break; }
        fclose(f); if (MAXTOK < 0) break; }
}
/* words of a line, lower case letters only; sentence ends marked by -1 in ids when wanted */
static int words(const char *s, char w[][32], int max) {
    int n = 0, l = 0;
    for (;; s++) { if (isalpha((unsigned char)*s)) { if (l < 31) w[n][l++] = (char)tolower((unsigned char)*s); continue; }
        if (l) { w[n][l] = 0; if (++n == max) return n; l = 0; } if (!*s) break; }
    return n;
}

/* ---- vocab */
static void count_line(char *line, int held) { if (held) return; static char w[4096][32]; int n = words(line, w, 4096); for (int i = 0; i < n; i++) hget(w[i], 1)->n++; }
static int cmp_ent(const void *a, const void *b) { const Ent *x = *(Ent *const *)a, *y = *(Ent *const *)b; return x->n < y->n ? 1 : x->n > y->n ? -1 : strcmp(x->w, y->w); }
static int run_vocab(const char *out, int nf, char **files) {
    each_line(nf, files, count_line);
    Ent **a = malloc(hn * sizeof *a); uint64_t k = 0; for (uint64_t i = 0; i < hcap; i++) if (H[i].w) a[k++] = &H[i];
    qsort(a, k, sizeof *a, cmp_ent);
    FILE *f = fopen(out, "w"); for (uint64_t i = 0; i < k && i < (uint64_t)V; i++) fprintf(f, "%s\t%llu\n", a[i]->w, (unsigned long long)a[i]->n); fclose(f);
    fprintf(stderr, "%llu word types, %d kept\n", (unsigned long long)k, V < (int)k ? V : (int)k); return 0;
}

/* ---- T2 items: in held-out paragraphs, one sentence each (8-40 words, all but at most 2 known), one content word hidden
   (known, not among the 100 most frequent words, 3+ letters) */
static char **items; static int nitems, capitems;
static void item_line(char *line, int held) {
    if (!held) return;
    for (char *s = line; *s;) { char *e = s; while (*e && !(strchr(".!?", *e) && (e[1] == ' ' || e[1] == '\n' || !e[1]))) e++;
        char sent[2048]; int L = (int)(e - s) + (*e ? 1 : 0); if (L > 2000) L = 2000; memcpy(sent, s, L); sent[L] = 0; s = *e ? e + 1 : e;
        static char w[64][32]; int n = words(sent, w, 64); if (n < 8 || n > 40) continue;
        int unk = 0, cand[64], nc = 0; for (int i = 0; i < n; i++) { int id = wid(w[i]); if (id < 0) unk++; else if (id >= 100 && strlen(w[i]) >= 3) cand[nc++] = i; }
        if (unk > 2 || !nc) continue;
        int pick = cand[rnd() % nc];
        /* the sentence with that word (its pick-th word) replaced by ___ */
        char out[2200]; int o = 0, wi = -1, inw = 0;
        for (char *p = sent; *p; p++) { int a = isalpha((unsigned char)*p); if (a && !inw) wi++; inw = a;
            if (a && wi == pick) { if (!(p > sent && isalpha((unsigned char)p[-1]))) { memcpy(out + o, "___", 3); o += 3; } continue; }
            if (*p == '\t' || *p == '\n') continue; out[o++] = *p; }
        out[o] = 0;
        if (nitems == capitems) { capitems = capitems ? capitems * 2 : 4096; items = realloc(items, capitems * sizeof *items); }
        char *it = malloc(o + 40); sprintf(it, "%s\t%s", out, w[pick]); items[nitems++] = it;
        break;   /* one sentence per paragraph */
    }
}
static int run_items(const char *dev, const char *test, int nf, char **files) {
    rs ^= SEED; each_line(nf, files, item_line);
    for (int i = nitems - 1; i > 0; i--) { int j = rnd() % (i + 1); char *t = items[i]; items[i] = items[j]; items[j] = t; }
    FILE *a = fopen(dev, "w"), *b = fopen(test, "w");
    for (int i = 0; i < nitems && i < 3000; i++) fprintf(i < 1000 ? a : b, "%s\n", items[i]);
    fclose(a); fclose(b); fprintf(stderr, "%d items found, 1000 dev + %d test written\n", nitems, nitems < 3000 ? (nitems > 1000 ? nitems - 1000 : 0) : 2000); return 0;
}

/* ---- the cortex
   r[w]  a fixed random pattern per word (K active of N): its identity, as first seen
   m[w]  its meaning: up to M (unit, strength) pairs; its pattern (code) = the K strongest
   WT    K2's association weights, WT[j][i] = from input unit j to unit i
   duty  how often each unit wins (homeostasis)                                                              */
static int CELL; static uint16_t *R, *mi; static float *mv, *WT, *duty, *dense; static uint8_t *mn;
static void rpat(int w, uint16_t *out) { uint64_t s = rs; rs = SEED * 0x9E3779B97F4A7C15ull + (uint64_t)w * 0xBF58476D1CE4E5B9ull + 1;
    for (int k = 0; k < K; k++) { again: { uint16_t u = (uint16_t)(rnd() % N); for (int x = 0; x < k; x++) if (out[x] == u) goto again; out[k] = u; } } rs = s; }
static int code(int w, uint16_t *out) {   /* the K strongest units of m[w] */
    int n = mn[w]; uint16_t *I = mi + (size_t)w * M; float *Vv = mv + (size_t)w * M; int idx[256]; for (int i = 0; i < n; i++) idx[i] = i;
    int k = n < K ? n : K;
    for (int a = 0; a < k; a++) { int b = a; for (int c = a + 1; c < n; c++) if (Vv[idx[c]] > Vv[idx[b]]) b = c; int t = idx[a]; idx[a] = idx[b]; idx[b] = t; out[a] = I[idx[a]]; }
    return k;
}
static uint16_t *CD; static uint8_t *CN; static float *BOOST;   /* every word's pattern, kept (units in rising order); homeostasis factors */
static int cmp_u16(const void *a, const void *b) { return (int)*(const uint16_t *)a - (int)*(const uint16_t *)b; }
static void code_refresh(int w) { uint16_t *o = CD + (size_t)w * K; CN[w] = (uint8_t)code(w, o); qsort(o, CN[w], 2, cmp_u16); }
static void codes_all(void) { if (!CD) { CD = malloc((size_t)V * K * 2); CN = malloc(V); } for (int w = 0; w < V; w++) code_refresh(w); }
static void boost_refresh(void) { if (!BOOST) BOOST = malloc(N * 4); for (int i = 0; i < N; i++) BOOST[i] = expf((float)(BETA * ((double)K / N - duty[i]) * N / K)); }
static void m_add(int w, uint16_t u, float v) {   /* m[w][u] += v; when full, the weakest entry gives way */
    uint16_t *I = mi + (size_t)w * M; float *Vv = mv + (size_t)w * M; int n = mn[w];
    for (int i = 0; i < n; i++) if (I[i] == u) { Vv[i] += v; return; }
    if (n < M) { I[n] = u; Vv[n] = v; mn[w]++; return; }
    int lo = 0; for (int i = 1; i < M; i++) if (Vv[i] < Vv[lo]) lo = i; if (v > Vv[lo]) { I[lo] = u; Vv[lo] = v; }
}
static void init_cells(void) {
    R = malloc((size_t)V * K * 2); mi = calloc((size_t)V * M, 2); mv = calloc((size_t)V * M, 4); mn = calloc(V, 1);
    for (int w = 0; w < V; w++) { rpat(w, R + (size_t)w * K); for (int k = 0; k < K; k++) m_add(w, R[(size_t)w * K + k], 1.0f); }
    duty = malloc(N * 4); for (int i = 0; i < N; i++) duty[i] = (float)K / N;
    if (CELL == 2) { WT = calloc((size_t)N * N, 4); if (!WT) { fprintf(stderr, "no memory for W\n"); exit(1); } }
    if (CELL == 1) { dense = calloc((size_t)V * N, 4); if (!dense) { fprintf(stderr, "no memory (V x N)\n"); exit(1); } }
    codes_all(); boost_refresh();
}

/* one replay step: target word t, its context ctx[0..nc) */
static long ntok, nskip; static double tstart;
static float *P; static uint16_t *Y, *Yh;
static int topk(const float *p, uint16_t *out, int boost) {
    /* the K largest (boosted by homeostasis when asked) */
    static float best[256]; int n = 0;
    for (int i = 0; i < N; i++) { float v = boost ? p[i] * BOOST[i] : p[i];
        if (n < K) { int j = n++; while (j > 0 && best[j - 1] < v) { best[j] = best[j - 1]; out[j] = out[j - 1]; j--; } best[j] = v; out[j] = (uint16_t)i; }
        else if (v > best[K - 1]) { int j = K - 1; while (j > 0 && best[j - 1] < v) { best[j] = best[j - 1]; out[j] = out[j - 1]; j--; } best[j] = v; out[j] = (uint16_t)i; } }
    return n;
}
static void step(int t, const int *ctx, int nc) {
    if (CELL == 1) { float *d = dense + (size_t)t * N; for (int c = 0; c < nc; c++) { const uint16_t *r = R + (size_t)ctx[c] * K; for (int k = 0; k < K; k++) d[r[k]] += 1.0f; } return; }
    if (CELL != 2) return;
    /* input: the context words' patterns added up (sparse), prediction P = W·input, inhibition: K winners */
    static uint16_t cu[1024]; static float cvv[1024]; int ncu = 0;
    memset(P, 0, N * 4);
    for (int c = 0; c < nc; c++) { int k = CN[ctx[c]]; const uint16_t *cd = CD + (size_t)ctx[c] * K;
        for (int x = 0; x < k; x++) { int j = cd[x], f = -1; for (int y = 0; y < ncu; y++) if (cu[y] == j) { f = y; break; } if (f < 0) { cu[ncu] = (uint16_t)j; cvv[ncu++] = 1; } else cvv[f] += 1; } }
    for (int y = 0; y < ncu; y++) { const float *row = WT + (size_t)cu[y] * N; float a = cvv[y]; for (int i = 0; i < N; i++) P[i] += a * row[i]; }
    int ny = topk(P, Yh, 1), nt = CN[t]; memcpy(Y, CD + (size_t)t * K, nt * 2);
    /* local learning: a unit that should have fired and did not strengthens its inputs; one that fired and should not weakens them */
    for (int a = 0; a < nt; a++) { int hit = 0; for (int b = 0; b < ny; b++) hit |= Yh[b] == Y[a]; if (hit) continue;
        for (int y = 0; y < ncu; y++) WT[(size_t)cu[y] * N + Y[a]] += (float)ETA * cvv[y]; }
    for (int b = 0; b < ny; b++) { int hit = 0; for (int a = 0; a < nt; a++) hit |= Yh[b] == Y[a]; if (hit) continue;
        for (int y = 0; y < ncu; y++) WT[(size_t)cu[y] * N + Yh[b]] -= (float)ETA * cvv[y]; }
    /* meaning drifts toward what its contexts predict; homeostasis counts the winners */
    float *Vv = mv + (size_t)t * M; for (int i = 0; i < mn[t]; i++) Vv[i] *= (float)(1 - ETA2);
    for (int b = 0; b < ny; b++) m_add(t, Yh[b], (float)ETA2);
    { const uint16_t *r = R + (size_t)t * K; for (int k = 0; k < K; k++) m_add(t, r[k], (float)(ETA2 * IDW)); }   /* its own identity stays, at strength IDW */
    code_refresh(t);
    static long nstep; for (int b = 0; b < ny; b++) duty[Yh[b]] += 0.0001f;
    if (++nstep % 256 == 0) { float d = powf(0.9999f, 256); for (int i = 0; i < N; i++) duty[i] *= d; boost_refresh(); }
}
static int *seq; static int nseq, capseq;
static void train_line(char *line, int held) {
    if (held) return;
    static char w[4096][32]; int n = words(line, w, 4096); nseq = 0;
    for (int i = 0; i < n; i++) { int id = wid(w[i]); if (id < 0) continue;
        if (SUB > 0) { double f = (double)vcnt[id] / vtot; double keep = (sqrt(f / SUB) + 1) * SUB / f; if (keep < 1 && rndf() > keep) { nskip++; continue; } }
        if (nseq == capseq) { capseq = capseq ? capseq * 2 : 4096; seq = realloc(seq, capseq * sizeof *seq); } seq[nseq++] = id; }
    for (int i = 0; i < nseq; i++) { int ctx[64], nc = 0; for (int d = -WIN; d <= WIN; d++) if (d && i + d >= 0 && i + d < nseq) ctx[nc++] = seq[i + d];
        if (!nc) continue; step(seq[i], ctx, nc);
        if (++ntok % 2000000 == 0) fprintf(stderr, "  %ld M words replayed, %.0f words/s\n", ntok / 1000000, ntok / (now() - tstart));
        if (MAXTOK > 0 && ntok >= MAXTOK) { MAXTOK = -1; return; } }
}


/* ---- K3 (v0.1, docs/CORTEX.md): Hebbian + anti-Hebbian. Each word has D meaning cells (MU) and D context cells (MC).
   A word t next to a neighbour c: reward, the links between t's meaning cells and c's context cells grow by how much the
   pair was not yet expected (1 - sigma(agreement)); penalty, for NEG words drawn at random (by how common they are) the
   links shrink by how much they were expected (sigma(agreement)). Each change uses only the two cells it joins and the
   reward/penalty signal. Two threads share the cells (Hogwild). */
static int D = 300, NEG = 5; static float *MU, *MC; static int *UT; static const long UTN = 10000000; static double ETA3 = 0.025, TOTAL; static long k3done;
static float sigm(float x) { return x > 8 ? 1.0f : x < -8 ? 0.0f : 1.0f / (1.0f + expf(-x)); }
static uint64_t xr(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static void k3_init(void) {
    MU = malloc((size_t)V * D * 4); MC = calloc((size_t)V * D, 4); uint64_t st = SEED * 2654435761u + 7;
    for (size_t i = 0; i < (size_t)V * D; i++) MU[i] = (float)(((xr(&st) >> 11) * (1.0 / 9007199254740992.0)) - 0.5) / D;
    UT = malloc(UTN * sizeof *UT); double z = 0; for (int w = 0; w < V; w++) z += pow((double)vcnt[w], 0.75);
    int w = 0; double c = pow((double)vcnt[0], 0.75) / z;
    for (long i = 0; i < UTN; i++) { UT[i] = w; if ((double)(i + 1) / UTN > c && w < V - 1) { w++; c += pow((double)vcnt[w], 0.75) / z; } }
    if (MAXTOK > 0) TOTAL = (double)MAXTOK;
    else { for (int x = 0; x < V; x++) { double f = (double)vcnt[x] / vtot, keep = SUB > 0 ? (sqrt(f / SUB) + 1) * SUB / f : 1; TOTAL += vcnt[x] * (keep < 1 ? keep : 1); } }
}
static void k3_line(const char *line, uint64_t *st, float *neu) {
    static __thread char (*w)[32]; static __thread int *seq; if (!w) { w = malloc(4096 * 32); seq = malloc(4096 * sizeof *seq); }
    int n = words(line, w, 4096), ns = 0;
    for (int i = 0; i < n; i++) { int id = wid(w[i]); if (id < 0) continue;
        if (SUB > 0) { double f = (double)vcnt[id] / vtot, keep = (sqrt(f / SUB) + 1) * SUB / f; if (keep < 1 && (xr(st) >> 11) * (1.0 / 9007199254740992.0) > keep) continue; }
        seq[ns++] = id; }
    for (int i = 0; i < ns; i++) {
        double prog = (double)k3done / (TOTAL > 1 ? TOTAL : 1); float eta = (float)(ETA3 * (prog < 0.9999 ? 1 - prog : 0.0001));
        int t = seq[i], b = 1 + (int)(xr(st) % (uint64_t)WIN); float *u = MU + (size_t)t * D;
        for (int d = -b; d <= b; d++) { int j = i + d; if (!d || j < 0 || j >= ns) continue;
            memset(neu, 0, D * 4);
            for (int k = 0; k <= NEG; k++) { int o; float label;
                if (k == 0) { o = seq[j]; label = 1; } else { o = UT[xr(st) % (uint64_t)UTN]; if (o == seq[j]) continue; label = 0; }
                float *v = MC + (size_t)o * D, dot = 0; for (int x = 0; x < D; x++) dot += u[x] * v[x];
                float g = eta * (label - sigm(dot));          /* reward (label 1) or penalty (label 0), times the surprise */
                for (int x = 0; x < D; x++) { neu[x] += g * v[x]; v[x] += g * u[x]; } }
            for (int x = 0; x < D; x++) u[x] += neu[x]; }
        long dn = __atomic_add_fetch(&k3done, 1, __ATOMIC_RELAXED);
        if (dn % 5000000 == 0) fprintf(stderr, "  %ld M words replayed, %.0f words/s\n", dn / 1000000, dn / (now() - tstart));
    }
}
static void k3_train(int nf, char **files) {
    enum { B = 20000 }; static char *L[B]; int nl = 0; static char buf[1 << 20];
    for (int fi = 0; fi < nf && (MAXTOK <= 0 || k3done < MAXTOK); fi++) { FILE *f = fopen(files[fi], "r"); if (!f) { perror(files[fi]); exit(1); }
        for (int more = 1; more && (MAXTOK <= 0 || k3done < MAXTOK);) {
            nl = 0; while (nl < B && (more = fgets(buf, sizeof buf, f) != NULL)) { int held = lineno++ % 200 == 0; if (!held) L[nl++] = strdup(buf); }
            #pragma omp parallel
            { uint64_t st = SEED * 0x9E3779B97F4A7C15ull + (uint64_t)lineno * 31 + 1; float *neu = malloc(D * 4);
              #ifdef _OPENMP
              st += 1000003ull * (uint64_t)omp_get_thread_num();
              #endif
              #pragma omp for schedule(dynamic, 64)
              for (int i = 0; i < nl; i++) if (MAXTOK <= 0 || k3done < MAXTOK) k3_line(L[i], &st, neu);
              free(neu); }
            for (int i = 0; i < nl; i++) free(L[i]); }
        fclose(f); }
    ntok = k3done;
}

/* ---- model file */
static void save(const char *p) {
    FILE *f = fopen(p, "wb"); fwrite("CTX0", 1, 4, f); int h[6] = { CELL, V, CELL == 3 ? D : N, K, M, WIN }; fwrite(h, 4, 6, f); fwrite(&SEED, 8, 1, f);
    if (CELL == 3) { fwrite(MU, 4, (size_t)V * D, f); fwrite(MC, 4, (size_t)V * D, f); fclose(f); return; }
    fwrite(mn, 1, V, f); fwrite(mi, 2, (size_t)V * M, f); fwrite(mv, 4, (size_t)V * M, f); fwrite(duty, 4, N, f);
    if (CELL == 2) fwrite(WT, 4, (size_t)N * N, f); fclose(f);
}
static int load(const char *p) {
    FILE *f = fopen(p, "rb"); char mg[4]; int h[6]; if (!f || fread(mg, 1, 4, f) != 4 || memcmp(mg, "CTX0", 4) || fread(h, 4, 6, f) != 6) { fprintf(stderr, "bad model %s\n", p); return 1; }
    CELL = h[0]; V = h[1]; N = h[2]; K = h[3]; M = h[4]; WIN = h[5]; if (fread(&SEED, 8, 1, f) != 1) return 1;
    if (CELL == 3) { D = N; MU = malloc((size_t)V * D * 4); MC = malloc((size_t)V * D * 4);
        if (fread(MU, 4, (size_t)V * D, f) != (size_t)V * D || fread(MC, 4, (size_t)V * D, f) != (size_t)V * D) return 1; fclose(f); return 0; }
    R = malloc((size_t)V * K * 2); for (int w = 0; w < V; w++) rpat(w, R + (size_t)w * K);
    mn = malloc(V); mi = malloc((size_t)V * M * 2); mv = malloc((size_t)V * M * 4); duty = malloc(N * 4);
    if (fread(mn, 1, V, f) != (size_t)V || fread(mi, 2, (size_t)V * M, f) != (size_t)V * M || fread(mv, 4, (size_t)V * M, f) != (size_t)V * M || fread(duty, 4, N, f) != (size_t)N) return 1;
    if (CELL == 2) { WT = malloc((size_t)N * N * 4); if (fread(WT, 4, (size_t)N * N, f) != (size_t)N * N) return 1; }
    fclose(f); codes_all(); boost_refresh(); return 0;
}
static int run_train(const char *cell, const char *model, int nf, char **files) {
    CELL = cell[1] - '0'; if (cell[0] != 'K' || CELL < 0 || CELL > 3) { fprintf(stderr, "cell is K0, K1, K2 or K3\n"); return 1; }
    if (CELL == 3) { k3_init(); tstart = now(); k3_train(nf, files); save(model);
        fprintf(stderr, "RESULT train cell=%s words=%ld seconds=%.0f D=%d WIN=%d NEG=%d ETA3=%g SUB=%g\n", cell, ntok, now() - tstart, D, WIN, NEG, ETA3, SUB); return 0; }
    init_cells(); P = malloc(N * 4); Y = malloc(256 * 2); Yh = malloc(256 * 2); rs ^= SEED;
    tstart = now(); if (CELL) each_line(nf, files, train_line);
    if (CELL == 1) {   /* meaning = the context patterns met, less what every word meets (the mean), kept as its M strongest units */
        double *mean = calloc(N, 8); for (int w = 0; w < V; w++) { float *d = dense + (size_t)w * N; double s = 0; for (int i = 0; i < N; i++) s += d[i]; if (s > 0) for (int i = 0; i < N; i++) mean[i] += d[i] / s; }
        for (int i = 0; i < N; i++) mean[i] /= V;
        for (int w = 0; w < V; w++) { float *d = dense + (size_t)w * N; double s = 0; for (int i = 0; i < N; i++) s += d[i]; if (s <= 0) continue;
            mn[w] = 0; for (int i = 0; i < N; i++) { float v = (float)(d[i] / s - mean[i]); if (v > 0) m_add(w, (uint16_t)i, v); } }
        free(mean); codes_all();
    }
    save(model);
    fprintf(stderr, "RESULT train cell=%s words=%ld skipped=%ld seconds=%.0f N=%d K=%d WIN=%d ETA=%g ETA2=%g BETA=%g SUB=%g IDW=%g\n", cell, ntok, nskip, now() - tstart, N, K, WIN, ETA, ETA2, BETA, SUB, IDW);
    return 0;
}

/* ---- using it */
static int SIMCOS;
static double sim(int a, int b) {
    if (CELL == 3) { const float *x = MU + (size_t)a * D, *y = MU + (size_t)b * D; double d = 0, na = 0, nb = 0; for (int i = 0; i < D; i++) { d += x[i] * y[i]; na += x[i] * x[i]; nb += y[i] * y[i]; } return na > 0 && nb > 0 ? d / sqrt(na * nb) : 0; }
    if (SIMCOS) { double d = 0, na = 0, nb = 0; uint16_t *Ia = mi + (size_t)a * M, *Ib = mi + (size_t)b * M; float *Va = mv + (size_t)a * M, *Vb = mv + (size_t)b * M;
        for (int i = 0; i < mn[a]; i++) { na += Va[i] * Va[i]; for (int j = 0; j < mn[b]; j++) if (Ia[i] == Ib[j]) d += Va[i] * Vb[j]; }
        for (int j = 0; j < mn[b]; j++) nb += Vb[j] * Vb[j]; return na > 0 && nb > 0 ? d / sqrt(na * nb) : 0; }
    const uint16_t *ca = CD + (size_t)a * K, *cb = CD + (size_t)b * K; int i = 0, j = 0, o = 0;
    while (i < CN[a] && j < CN[b]) { if (ca[i] == cb[j]) { o++; i++; j++; } else if (ca[i] < cb[j]) i++; else j++; } return (double)o / K;
}
/* score of every word as the filler for a context */
static double STORYW = 1;   /* T3: the story's words, all together, weigh as much as STORYW words next to the blank */
static void fill(const int *ctx, int nc, const int *ex, int ne, float *score) {
    if (CELL == 3) {   /* how well each word's meaning cells fit the context cells of the words around the blank */
        static float *q; if (!q) q = malloc(D * 4); memset(q, 0, D * 4);
        for (int c = 0; c < nc + ne; c++) { int w = c < nc ? ctx[c] : ex[c - nc]; float a = c < nc ? 1.0f : (float)(STORYW / ne); const float *v = MC + (size_t)w * D; for (int i = 0; i < D; i++) q[i] += a * v[i]; }
        /* the cells hold how much likelier a neighbour is next to a word than anywhere (log P(u|w)/P(u)); for the word in a
           blank, Bayes adds how common the word is itself: log P(w) */
        for (int w = 0; w < V; w++) { const float *u = MU + (size_t)w * D; double s = 0; for (int i = 0; i < D; i++) s += u[i] * q[i]; score[w] = (float)(s + log((double)vcnt[w] / vtot)); }
        for (int c = 0; c < nc; c++) score[ctx[c]] = -1e9f; return; }
    static float *in; if (!in) in = malloc(N * 4); memset(in, 0, N * 4);
    for (int c = 0; c < nc + ne; c++) { int w = c < nc ? ctx[c] : ex[c - nc]; float a = c < nc ? 1.0f : (float)(STORYW / ne);
        if (CELL == 1) { const uint16_t *r = R + (size_t)w * K; for (int k = 0; k < K; k++) in[r[k]] += a; }
        else { const uint16_t *cd = CD + (size_t)w * K; for (int x = 0; x < CN[w]; x++) in[cd[x]] += a; } }
    if (CELL == 2) { memset(P, 0, N * 4); for (int j = 0; j < N; j++) if (in[j] != 0) { const float *row = WT + (size_t)j * N; for (int i = 0; i < N; i++) P[i] += in[j] * row[i]; }
        for (int i = 0; i < N; i++) in[i] = P[i] * BOOST[i]; }
    for (int w = 0; w < V; w++) {
        if (CELL == 1) { double d = 0, n = 0; uint16_t *I = mi + (size_t)w * M; float *Vv = mv + (size_t)w * M; for (int i = 0; i < mn[w]; i++) { d += Vv[i] * in[I[i]]; n += Vv[i] * Vv[i]; } score[w] = n > 0 ? (float)(d / sqrt(n)) : -1e9f; }
        else { const uint16_t *cd = CD + (size_t)w * K; double s = 0; for (int x = 0; x < CN[w]; x++) s += in[cd[x]]; score[w] = (float)s; } }
    for (int c = 0; c < nc; c++) score[ctx[c]] = -1e9f;   /* the context's own words are not the answer */
}
static int rank_of(const float *score, int ans) { int r = 0; for (int w = 0; w < V; w++) if (w != ans && score[w] >= score[ans]) r++; return r; }   /* ties count against it */
/* the in-vocabulary words of a text, and where the blank is (its index in ids, or -1) */
static int text_ids(const char *s, int *ids, int max, int *blank) {
    int n = 0, l = 0; char w[32]; *blank = -1;
    for (;; s++) { if (s[0] == '_' && s[1] == '_' && s[2] == '_') { if (l) { w[l] = 0; int id = wid(w); if (id >= 0 && n < max) ids[n++] = id; l = 0; } *blank = n; s += 2; continue; }
        if (isalpha((unsigned char)*s)) { if (l < 31) w[l++] = (char)tolower((unsigned char)*s); continue; }
        if (l) { w[l] = 0; int id = wid(w); if (id >= 0 && n < max) ids[n++] = id; l = 0; } if (!*s) break; }
    return n;
}
static double spearman(const double *x, const double *y, int n) {
    double *rx = malloc(n * 8), *ry = malloc(n * 8);
    for (int pass = 0; pass < 2; pass++) { const double *a = pass ? y : x; double *r = pass ? ry : rx;
        for (int i = 0; i < n; i++) { int lt = 0, eq = 0; for (int j = 0; j < n; j++) { lt += a[j] < a[i]; eq += a[j] == a[i]; } r[i] = lt + (eq + 1) / 2.0; } }
    double mx = 0, my = 0; for (int i = 0; i < n; i++) { mx += rx[i]; my += ry[i]; } mx /= n; my /= n;
    double sxy = 0, sxx = 0, syy = 0; for (int i = 0; i < n; i++) { sxy += (rx[i] - mx) * (ry[i] - my); sxx += (rx[i] - mx) * (rx[i] - mx); syy += (ry[i] - my) * (ry[i] - my); }
    free(rx); free(ry); return sxy / sqrt(sxx * syy + 1e-30);
}
static int run_eval(void) {
    P = malloc(N * 4); float *score = malloc(V * 4);
    printf("cell K%d  N=%d K=%d WIN=%d V=%d  similarity=%s\n", CELL, N, K, WIN, V, SIMCOS ? "cosine" : "overlap");
    if (getenv("WS")) { char *list = strdup(getenv("WS"));   /* T1 */
        for (char *p = strtok(list, " "); p; p = strtok(0, " ")) { FILE *f = fopen(p, "r"); if (!f) { perror(p); continue; } char a[64], b[64]; double h; int n = 0, known = 0;
            double *x = malloc(4000 * 8), *y = malloc(4000 * 8);
            while (n < 4000 && fscanf(f, "%63s %63s %lf", a, b, &h) == 3) { for (char *c = a; *c; c++) *c = (char)tolower((unsigned char)*c); for (char *c = b; *c; c++) *c = (char)tolower((unsigned char)*c);
                int ia = wid(a), ib = wid(b); x[n] = ia >= 0 && ib >= 0 ? sim(ia, ib) : 0; known += ia >= 0 && ib >= 0; y[n++] = h; }
            fclose(f); const char *bn = strrchr(p, '/'); printf("T1 %-24s %4d pairs (%4d known)  spearman %.3f\n", bn ? bn + 1 : p, n, known, spearman(x, y, n)); free(x); free(y); } }
    if (getenv("T2")) { FILE *f = fopen(getenv("T2"), "r"); char l[4096]; int n = 0, hit = 0, oov = 0;   /* T2 */
        while (f && fgets(l, sizeof l, f)) { char *t = strchr(l, '\t'); if (!t) continue; *t = 0; char *ans = t + 1; ans[strcspn(ans, "\r\n")] = 0;
            int ids[256], bl, nn = text_ids(l, ids, 256, &bl), a = wid(ans); n++; if (a < 0 || bl < 0) { oov++; continue; }
            int ctx[64], nc = 0; for (int d = -WIN; d <= WIN; d++) if (d && bl + (d < 0 ? d : d - 1) >= 0 && bl + (d < 0 ? d : d - 1) < nn) ctx[nc++] = ids[bl + (d < 0 ? d : d - 1)];
            fill(ctx, nc, 0, 0, score); hit += rank_of(score, a) < 10; }
        if (f) { fclose(f); printf("T2 fill the blank          %4d items  hit@10 %.2f%%\n", n, n ? 100.0 * hit / n : 0); } }
    if (getenv("STORIES")) { FILE *f = fopen(getenv("STORIES"), "r"); char l[4096], dir[1024]; const char *sp = getenv("SPLIT") ? getenv("SPLIT") : "dev";   /* T3 */
        snprintf(dir, sizeof dir, "%s", getenv("STORIES")); char *sl = strrchr(dir, '/'); if (sl) sl[1] = 0; else dir[0] = 0;
        int n = 0, hit = 0;
        while (f && fgets(l, sizeof l, f)) { if (l[0] == '#') continue; char *fl[4]; int k = 0; for (char *p = strtok(l, "\t\r\n"); p && k < 4; p = strtok(0, "\t\r\n")) fl[k++] = p; if (k < 4 || strcmp(fl[0], sp)) continue;
            char sp2[1100]; snprintf(sp2, sizeof sp2, "%sstories/%s.txt", dir, fl[1]); FILE *g = fopen(sp2, "r"); static char story[1 << 16]; size_t sn = g ? fread(story, 1, sizeof story - 1, g) : 0; story[sn] = 0; if (g) fclose(g);
            int ids[256], bl, nn = text_ids(fl[2], ids, 256, &bl), a = wid(fl[3]); n++;
            /* context: the sentence's words around the blank, and the story's words (all of them, as one more input) */
            int ctx[64], nc = 0; for (int d = -WIN; d <= WIN; d++) if (d && bl + (d < 0 ? d : d - 1) >= 0 && bl + (d < 0 ? d : d - 1) < nn) ctx[nc++] = ids[bl + (d < 0 ? d : d - 1)];
            static int sids[4096]; int sb, ns = text_ids(story, sids, 4096, &sb);
            if (a < 0) { printf("T3 %-40s answer %-12s not a known word\n", fl[2], fl[3]); continue; }
            fill(ctx, nc, sids, ns, score); int r1 = rank_of(score, a);
            printf("T3 %-45s %-12s rank %d%s\n", fl[2], fl[3], r1 + 1, r1 < 10 ? "  (top 10)" : ""); hit += r1 < 10; }
        if (f) { fclose(f); printf("T3 stories (%s)             %4d items  hit@10 %.2f%%\n", sp, n, n ? 100.0 * hit / n : 0); } }
    return 0;
}
static int run_near(int nw, char **ws) {
    for (int i = 0; i < nw; i++) { int a = wid(ws[i]); if (a < 0) { printf("%s: not known\n", ws[i]); continue; }
        int best[10]; double bv[10]; for (int k = 0; k < 10; k++) { best[k] = -1; bv[k] = -1e9; }
        for (int w = 0; w < V; w++) { if (w == a) continue; double s = sim(a, w); for (int k = 0; k < 10; k++) if (s > bv[k]) { for (int x = 9; x > k; x--) { bv[x] = bv[x - 1]; best[x] = best[x - 1]; } bv[k] = s; best[k] = w; break; } }
        printf("%s:", ws[i]); for (int k = 0; k < 10; k++) if (best[k] >= 0) printf(" %s(%.2f)", voc[best[k]], bv[k]); printf("\n"); }
    return 0;
}

int main(int argc, char **argv) {
    V = envi("V", V); N = envi("N", N); K = envi("K", K); WIN = envi("WIN", WIN); M = envi("M", M);
    ETA = envd("ETA", ETA); D = envi("D", D); NEG = envi("NEG", NEG); ETA3 = envd("ETA3", ETA3); IDW = envd("IDW", IDW); ETA2 = envd("ETA2", ETA2); BETA = envd("BETA", BETA); SUB = envd("SUB", SUB);
    MAXTOK = getenv("MAXTOK") ? atol(getenv("MAXTOK")) : 0; SEED = getenv("SEED") ? strtoull(getenv("SEED"), 0, 10) : SEED; SIMCOS = envi("SIM", 0); STORYW = envd("STORYW", STORYW);
    if (K > 255 || M > 255 || N > 65535 || K > M) { fprintf(stderr, "K, M up to 255, K <= M, N up to 65535\n"); return 1; }
    if (argc >= 4 && !strcmp(argv[1], "vocab")) return run_vocab(argv[2], argc - 3, argv + 3);
    if (argc >= 6 && !strcmp(argv[1], "items")) { vocab_load(argv[2]); return run_items(argv[3], argv[4], argc - 5, argv + 5); }
    if (argc >= 6 && !strcmp(argv[1], "train")) { vocab_load(argv[3]); return run_train(argv[2], argv[4], argc - 5, argv + 5); }
    if (argc >= 4 && !strcmp(argv[1], "eval")) { if (load(argv[2])) return 1; int keepV = V; vocab_load(argv[3]); if (V != keepV) { fprintf(stderr, "vocab size differs from the model's\n"); return 1; } return run_eval(); }
    if (argc >= 5 && !strcmp(argv[1], "near")) { if (load(argv[2])) return 1; vocab_load(argv[3]); return run_near(argc - 4, argv + 4); }
    fprintf(stderr, "see the top of cortex.c\n"); return 1;
}
