/*
 * bg.c — background knowledge: Mimir reads a shelf of text first, and keeps what it
 * understood and where it read it, not the words. Rules fixed before this was
 * written: docs/BACKGROUND_COMPARE.md.
 *
 * Reading: each sentence is read into facts (know.c's reader). A fact already known
 * is strengthened (counted), not stored again; it keeps at most two pointers to where
 * it was read (shelf file, byte offset of the sentence). When the brain is full,
 * facts read only once are forgotten. Cue lists word → facts are how it knows where
 * to look. The brain is saved to one file; the shelf stays on disk.
 *
 *   bg read BRAIN SECONDS FILE...   read the shelf files in order for at most SECONDS, save BRAIN
 *   bg arc BRAIN|- FILE.tsv...      questions without a passage (tools/shelf_fetch.py); '-' = empty brain
 *   bg know BRAIN|- FILE...         MCTest-format (passage): K open (know.c) + background
 * Env: ROOM (facts, default 6000000), BW="a b c d" answer weights, PART=n read only every n-th block (dev brain),
 *      DUMP=file (per-option parts), SHOW=1.
 * Run from the repo root.
 */
#define KNOW_NO_MAIN
#include "know.c"

/* ------------------------------------------------------------------ */
/* the brain's words: one table for everything it ever read              */
/* ------------------------------------------------------------------ */

typedef struct { char **s; uint32_t n, cap, mask, *slot; int32_t *mid; } Lex;
static Lex LX;
static uint32_t lx_id(const char *s, int add) {
    if (!LX.slot) { LX.mask = (1 << 20) - 1; LX.slot = calloc(LX.mask + 1, 4); }
    uint32_t h = shash(s) & LX.mask;
    for (; LX.slot[h]; h = (h + 1) & LX.mask) if (!strcmp(LX.s[LX.slot[h] - 1], s)) return LX.slot[h] - 1;
    if (!add) return NONE;
    if (LX.n * 2 >= LX.mask) { uint32_t m = LX.mask * 2 + 1; uint32_t *sl = calloc(m + 1, 4);
        for (uint32_t i = 0; i < LX.n; i++) { uint32_t g = shash(LX.s[i]) & m; while (sl[g]) g = (g + 1) & m; sl[g] = i + 1; }
        free(LX.slot); LX.slot = sl; LX.mask = m; h = shash(s) & m; while (LX.slot[h]) h = (h + 1) & m; }
    if (LX.n == LX.cap) { LX.cap = LX.cap ? LX.cap * 2 : 1 << 16; LX.s = realloc(LX.s, LX.cap * sizeof *LX.s); LX.mid = realloc(LX.mid, LX.cap * 4); }
    LX.s[LX.n] = strdup(s); LX.mid[LX.n] = -2; LX.slot[h] = ++LX.n; return LX.n - 1;
}
static int gmid(uint32_t g) { if (g >= LX.n) return -1; if (LX.mid[g] == -2) LX.mid[g] = meaning_id(MM, LX.s[g]); return LX.mid[g]; }
static float mcos(int x, int y) {
    if (x < 0 || y < 0) return 0;
    const int8_t *u = MM->vec + (size_t)x * MM->dims, *v = MM->vec + (size_t)y * MM->dims; int d = 0;
    for (int i = 0; i < MM->dims; i++) d += u[i] * v[i];
    return d * inv_norm[x] * inv_norm[y];
}
/* a word asked about: its id in the brain (NONE if never read) and its meaning */
typedef struct { uint32_t g; int mid; } QW;
static float qsim(const QW *a, uint32_t g) {
    if (g == NONE) return 0; if (a->g == g) return 1;
    float c = mcos(a->mid, gmid(g)); return c >= 1 ? 1 : c > 0.25f ? (c - 0.25f) / 0.75f : 0;
}
static float qqsim(const QW *a, const QW *b) {
    if (a->g != NONE && a->g == b->g) return 1; if (a->mid >= 0 && a->mid == b->mid) return 1;
    float c = mcos(a->mid, b->mid); return c >= 1 ? 1 : c > 0.25f ? (c - 0.25f) / 0.75f : 0;
}

/* ------------------------------------------------------------------ */
/* known facts, merged                                                   */
/* ------------------------------------------------------------------ */

#define KA 3
typedef struct { uint32_t verb, head[KA], mod[KA], prep[KA], count; uint8_t role[KA], narg, flags; uint64_t ptr[2]; } KF;
static KF *kf; static uint32_t nkf, capkf, *kslot, kmask; static uint64_t forgotten; static int forget_th;
static uint32_t ROOM = 6000000;

static uint64_t kf_hash(const KF *f) {
    uint64_t h = 1469598103934665603ull;
#define MIX(x) do { h ^= (uint64_t)(x); h *= 1099511628211ull; } while (0)
    MIX(f->verb); MIX(f->flags & F_NEG); MIX(f->narg);
    for (int i = 0; i < f->narg; i++) { MIX(f->role[i]); MIX(f->prep[i]); MIX(f->head[i]); }
#undef MIX
    return h;
}
static int kf_same(const KF *a, const KF *b) {
    if (a->verb != b->verb || (a->flags & F_NEG) != (b->flags & F_NEG) || a->narg != b->narg) return 0;
    for (int i = 0; i < a->narg; i++) if (a->role[i] != b->role[i] || a->prep[i] != b->prep[i] || a->head[i] != b->head[i]) return 0;
    return 1;
}
static void kf_rehash(uint32_t slots) {
    free(kslot); kmask = slots - 1; kslot = calloc(slots, 4);
    for (uint32_t i = 0; i < nkf; i++) { uint32_t h = (uint32_t)kf_hash(&kf[i]) & kmask; while (kslot[h]) h = (h + 1) & kmask; kslot[h] = i + 1; }
}
/* the brain is full: what was read only once is forgotten (then twice, ...) until a quarter of the room is free */
static void forget(void) {
    while (nkf > ROOM * 3 / 4) {
        forget_th++; uint32_t j = 0;
        for (uint32_t i = 0; i < nkf; i++) if (kf[i].count > (uint32_t)forget_th) kf[j++] = kf[i]; else forgotten++;
        nkf = j;
    }
    kf_rehash(kmask + 1);
}
static void kf_add(const KF *f, uint64_t ptr) {
    if (!kslot) { capkf = 1 << 20; kf = malloc((size_t)capkf * sizeof *kf); kf_rehash(1u << 21); }
    uint32_t h = (uint32_t)kf_hash(f) & kmask;
    for (; kslot[h]; h = (h + 1) & kmask) { KF *e = &kf[kslot[h] - 1]; if (kf_same(e, f)) { if (e->count < 0xffffffffu) e->count++; if (e->count == 2) e->ptr[1] = ptr; return; } }
    if (nkf >= ROOM) { forget(); kf_add(f, ptr); return; }
    if (nkf == capkf) { capkf = capkf * 2 > ROOM ? ROOM : capkf * 2; kf = realloc(kf, (size_t)capkf * sizeof *kf); }
    if ((nkf + 1) * 2 > kmask) { kf[nkf] = *f; kf_rehash((kmask + 1) * 2); h = (uint32_t)kf_hash(f) & kmask; while (kslot[h]) h = (h + 1) & kmask; }
    KF *e = &kf[nkf]; *e = *f; e->count = 1; e->ptr[0] = ptr; e->ptr[1] = ~0ull; kslot[h] = ++nkf;
}

/* ------------------------------------------------------------------ */
/* reading the shelf                                                      */
/* ------------------------------------------------------------------ */

static uint32_t *l2g; static uint32_t nl2g;         /* the reading context's word ids → the brain's */
static uint32_t gid_of(Know *k, uint32_t w) {
    if (w >= OPEN) return NONE;
    if (w >= nl2g) { uint32_t n = nl2g ? nl2g : 1 << 16; while (n <= w) n *= 2; l2g = realloc(l2g, n * 4); for (uint32_t i = nl2g; i < n; i++) l2g[i] = NONE - 1; nl2g = n; }
    if (l2g[w] == NONE - 1) l2g[w] = lx_id(k->t.voc[w].word, 1);
    return l2g[w];
}
/* a frame becomes a fact the brain can keep; 0 if it says too little */
static int to_kf(Know *k, const Frame *f, KF *o) {
    memset(o, 0, sizeof *o); int said = 0;
    o->verb = f->verb == NONE ? NONE : gid_of(k, f->verb); o->flags = f->flags & F_NEG;
    if (f->verb != NONE && f->verb != k->R.w_be && f->verb != k->R.w_have && f->verb != k->R.w_do && content(k, f->verb)) said++;
    for (int i = 0; i < f->narg && o->narg < KA; i++) { const Arg *a = &f->a[i]; if (a->head >= OPEN) continue;
        int j = o->narg++; o->role[j] = a->role; o->prep[j] = a->prep < OPEN ? gid_of(k, a->prep) : NONE; o->head[j] = gid_of(k, a->head); o->mod[j] = a->mod < OPEN ? gid_of(k, a->mod) : NONE;
        said += content(k, a->head); }
    if (said < 2) return 0;
    for (int i = 1; i < o->narg; i++) for (int j = i; j > 0; j--) {          /* the same fact the same way whatever the order said */
        int sw = o->role[j] < o->role[j - 1] || (o->role[j] == o->role[j - 1] && (o->prep[j] < o->prep[j - 1] || (o->prep[j] == o->prep[j - 1] && o->head[j] < o->head[j - 1])));
        if (!sw) break;
        uint8_t r = o->role[j]; o->role[j] = o->role[j - 1]; o->role[j - 1] = r;
        uint32_t x; x = o->prep[j]; o->prep[j] = o->prep[j - 1]; o->prep[j - 1] = x; x = o->head[j]; o->head[j] = o->head[j - 1]; o->head[j - 1] = x; x = o->mod[j]; o->mod[j] = o->mod[j - 1]; o->mod[j - 1] = x; }
    return 1;
}
#define FID_SHIFT 40
static const char *SHELF[16]; static int NSHELF;
static uint64_t nsent_read, nfacts_read;

/* read one block of the shelf (whole lines); a line end ends a sentence */
static void read_block(Know *k, const char *b, size_t n, int fid, uint64_t off) {
    TextContext *t = &k->t; t->ntok = 0; t->src_len = 0; text_take(t, NULL, b, n);
    for (uint32_t i = 0; i + 1 < t->ntok; i++) {
        uint32_t e = t->tok[i].src + t->tok[i].len, s = t->tok[i + 1].src; int nl = 0;
        for (uint32_t x = e; x < s && x < t->src_len; x++) nl += t->src[x] == '\n';
        if (nl) t->tok[i].end = 1;
    }
    if (t->ntok) t->tok[t->ntok - 1].end = 1;
    static Frame fr[512]; uint32_t a = 0;
    for (uint32_t i = 0; i < t->ntok; i++) {
        if (!t->tok[i].end) continue;
        if (i - a > 120) { a = i + 1; continue; }                 /* not a sentence (a list, a table) */
        uint64_t ptr = ((uint64_t)fid << FID_SHIFT) | (off + t->tok[a].src);
        k->R.question = 0; int nf = read_span(&k->R, t->tok, a, i + 1, fr, 512, 0); nsent_read++;
        for (int j = 0; j < nf; j++) { KF f; if (to_kf(k, &fr[j], &f)) { kf_add(&f, ptr); nfacts_read++; } }
        a = i + 1;
    }
    t->ntok = 0; t->src_len = 0;
}

static int brain_save(const char *path, double secs, uint64_t bytes);
static int run_read(const char *brain, double seconds, int nfile, char **files) {
    Know k; memset(&k, 0, sizeof k); text_init(&k.t); r2_init(&k.R, &k.t, &k.wm);
    if (getenv("ROOM")) ROOM = (uint32_t)atol(getenv("ROOM"));
    int part = getenv("PART") ? atoi(getenv("PART")) : 1;
    double t0 = now(), next = t0 + 60; uint64_t bytes = 0; const size_t B = 1 << 22; char *buf = malloc(B + 1); int stop = 0;
    for (int fi = 0; fi < nfile && !stop; fi++) {
        SHELF[NSHELF++] = files[fi];
        FILE *f = fopen(files[fi], "rb"); if (!f) { perror(files[fi]); return 1; }
        uint64_t off = 0; size_t have = 0; long blk = 0;
        for (;;) {
            size_t got = fread(buf + have, 1, B - have, f); size_t n = have + got; if (!n) break;
            size_t cut = n; if (got) { while (cut > 0 && buf[cut - 1] != '\n') cut--; if (!cut) cut = n; }
            if (blk++ % part == 0) { read_block(&k, buf, cut, NSHELF - 1, off); bytes += cut; }
            memmove(buf, buf + cut, n - cut); have = n - cut; off += cut;
            double tn = now();
            if (tn > next) { next = tn + 60; fprintf(stderr, "  %4.0f s  %s  %.0f MB read (%.1f MB/s)  %llu sentences  %u facts kept, %llu forgotten (read once%s)\n",
                tn - t0, files[fi], bytes / 1e6, bytes / 1e6 / (tn - t0), (unsigned long long)nsent_read, nkf, (unsigned long long)forgotten, forget_th > 1 ? " or more" : ""); }
            if (tn - t0 > seconds) { stop = 1; break; }
            if (!got && !have) break;
            if (!got) { if (have) { read_block(&k, buf, have, NSHELF - 1, off); bytes += have; } break; }
        }
        fclose(f);
    }
    double secs = now() - t0;
    fprintf(stderr, "read %.0f MB in %.0f s (%.2f MB/s): %llu sentences, %llu facts read; %u kept, %llu forgotten%s\n", bytes / 1e6, secs, bytes / 1e6 / secs,
        (unsigned long long)nsent_read, (unsigned long long)nfacts_read, nkf, (unsigned long long)forgotten, stop ? " (time up)" : " (shelf finished)");
    return brain_save(brain, secs, bytes);
}

/* ------------------------------------------------------------------ */
/* the brain on disk                                                      */
/* ------------------------------------------------------------------ */

static int vput64(uint8_t *b, uint64_t v) { int n = 0; while (v >= 128) { b[n++] = (uint8_t)(v | 128); v >>= 7; } b[n++] = (uint8_t)v; return n; }
static uint64_t vget64(FILE *f) { uint64_t v = 0; int sh = 0, c; while ((c = fgetc(f)) >= 0) { v |= (uint64_t)(c & 127) << sh; if (c < 128) break; sh += 7; } return v; }
static int brain_save(const char *path, double secs, uint64_t bytes) {
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); return 1; }
    fwrite("MIMB2", 1, 5, f); fwrite(&NSHELF, 4, 1, f);
    for (int i = 0; i < NSHELF; i++) { uint16_t l = strlen(SHELF[i]); fwrite(&l, 2, 1, f); fwrite(SHELF[i], 1, l, f); }
    fwrite(&LX.n, 4, 1, f); for (uint32_t i = 0; i < LX.n; i++) { uint8_t l = strlen(LX.s[i]) > 255 ? 255 : strlen(LX.s[i]); fwrite(&l, 1, 1, f); fwrite(LX.s[i], 1, l, f); }
    fwrite(&nkf, 4, 1, f);                       /* facts as varints: ids + 1, a role byte, count, pointers */
    for (uint32_t i = 0; i < nkf; i++) { const KF *e = &kf[i]; uint8_t b[160]; int n = 0;
        n += vput(b + n, e->verb + 1); b[n++] = e->narg | (e->flags & F_NEG) << 4;
        for (int j = 0; j < e->narg; j++) { b[n++] = e->role[j]; n += vput(b + n, e->prep[j] + 1); n += vput(b + n, e->head[j] + 1); n += vput(b + n, e->mod[j] + 1); }
        n += vput(b + n, e->count); n += vput64(b + n, e->ptr[0] + 1); n += vput64(b + n, e->count > 1 ? e->ptr[1] + 1 : 0); fwrite(b, 1, n, f); }
    long sz = ftell(f); fclose(f);
    printf("RESULT bgread brain=%s bytes=%ld facts=%u words=%u forgotten=%llu read_bytes=%llu seconds=%.0f mb_s=%.2f\n", path, sz, nkf, LX.n,
        (unsigned long long)forgotten, (unsigned long long)bytes, secs, bytes / 1e6 / secs);
    return 0;
}
static char *SHELFP[16];
static uint32_t *cue_off, *cue;              /* word → facts that hold it, strongest first */
static int cmp_strong(const void *a, const void *b) { uint32_t x = kf[*(const uint32_t *)a].count, y = kf[*(const uint32_t *)b].count; return x < y ? 1 : x > y ? -1 : 0; }
static int kf_words(const KF *f, uint32_t *w) { int n = 0; if (f->verb != NONE) w[n++] = f->verb; for (int i = 0; i < f->narg; i++) { w[n++] = f->head[i]; if (f->mod[i] != NONE) w[n++] = f->mod[i]; } return n; }
static void build_cues(void) {
    cue_off = calloc(LX.n + 2, 4);
    for (uint32_t i = 0; i < nkf; i++) { uint32_t w[8]; int n = kf_words(&kf[i], w); for (int j = 0; j < n; j++) { int dup = 0; for (int x = 0; x < j; x++) dup |= w[x] == w[j]; if (!dup && w[j] < LX.n) cue_off[w[j] + 2]++; } }
    for (uint32_t g = 0; g < LX.n; g++) cue_off[g + 2] += cue_off[g + 1];
    cue = malloc(((size_t)cue_off[LX.n + 1] + 1) * 4);
    for (uint32_t i = 0; i < nkf; i++) { uint32_t w[8]; int n = kf_words(&kf[i], w); for (int j = 0; j < n; j++) { int dup = 0; for (int x = 0; x < j; x++) dup |= w[x] == w[j]; if (!dup && w[j] < LX.n) cue[cue_off[w[j] + 1]++] = i; } }
    for (uint32_t g = 0; g < LX.n; g++) qsort(cue + cue_off[g], cue_off[g + 1] - cue_off[g], 4, cmp_strong);
}
static uint32_t df_of(uint32_t g) { return g < LX.n && cue_off ? cue_off[g + 1] - cue_off[g] : 0; }
static double gidf(uint32_t g) { return log((nkf + 1.0) / (df_of(g) + 0.5)); }
static int brain_load(const char *path) {
    if (!strcmp(path, "-")) { build_cues(); return 0; }                  /* the empty brain: nothing read yet */
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return 1; }
    char mg[5]; if (fread(mg, 1, 5, f) != 5 || memcmp(mg, "MIMB2", 5)) { fprintf(stderr, "%s: not a brain\n", path); return 1; }
    if (fread(&NSHELF, 4, 1, f) != 1) return 1;
    for (int i = 0; i < NSHELF; i++) { uint16_t l; if (fread(&l, 2, 1, f) != 1) return 1; SHELFP[i] = calloc(l + 1, 1); if (fread(SHELFP[i], 1, l, f) != l) return 1; SHELF[i] = SHELFP[i]; }
    uint32_t nw; if (fread(&nw, 4, 1, f) != 1) return 1;
    for (uint32_t i = 0; i < nw; i++) { uint8_t l; char s[256]; if (fread(&l, 1, 1, f) != 1 || fread(s, 1, l, f) != l) return 1; s[l] = 0; lx_id(s, 1); }
    if (fread(&nkf, 4, 1, f) != 1) return 1;
    kf = calloc(nkf ? nkf : 1, sizeof *kf);
    for (uint32_t i = 0; i < nkf; i++) { KF *e = &kf[i]; int c;
        e->verb = (uint32_t)vget64(f) - 1; c = fgetc(f); if (c < 0) return 1; e->narg = c & 15; e->flags = (c >> 4) & F_NEG;
        for (int j = 0; j < e->narg; j++) { e->role[j] = fgetc(f); e->prep[j] = (uint32_t)vget64(f) - 1; e->head[j] = (uint32_t)vget64(f) - 1; e->mod[j] = (uint32_t)vget64(f) - 1; }
        e->count = (uint32_t)vget64(f); e->ptr[0] = vget64(f) - 1; e->ptr[1] = vget64(f) - 1; }
    fclose(f); build_cues(); return 0;
}

/* ------------------------------------------------------------------ */
/* answering from what it knows                                           */
/* ------------------------------------------------------------------ */

typedef struct { double fact, joint, open, mean; } BParts;
/* words of a statement read by the answering context, as the brain knows them */
static int qwords(Know *A, const uint32_t *w, int n, QW *out) {
    for (int i = 0; i < n; i++) { const char *s = A->t.voc[w[i]].word; out[i].g = lx_id(s, 0); out[i].mid = meaning_id(MM, s); } return n;
}
/* how well some words are covered in meaning by a known fact's words, rare words counting more */
static double kcover(const QW *q, int nq, const uint32_t *w, int nw) {
    double s = 0, d = 0; for (int i = 0; i < nq; i++) { double b = 0; for (int j = 0; j < nw; j++) { double y = qsim(&q[i], w[j]); if (y > b) b = y; } double wt = q[i].g == NONE ? log(nkf + 2.0) : gidf(q[i].g); s += wt * b; d += wt; }
    return d > 0 ? s / d : 0;
}
static double qcover(const QW *q, int nq, const QW *w, int nw) {
    double s = 0, d = 0; for (int i = 0; i < nq; i++) { double b = 0; for (int j = 0; j < nw; j++) { double y = qqsim(&q[i], &w[j]); if (y > b) b = y; } double wt = q[i].g == NONE ? log(nkf + 2.0) : gidf(q[i].g); s += wt * b; d += wt; }
    return d > 0 ? s / d : 0;
}
/* a statement's fact (answering context ids) against a known fact, 0..1 (as know.c fact_sim) */
static float kfact_sim(Know *A, const Frame *h, const KF *f) {
    double num = 0, den = 0;
    if (h->verb != NONE) { int light = h->verb == A->R.w_be || h->verb == A->R.w_have || h->verb == A->R.w_do; double w = light ? 0.3 : 1.0;
        QW v = { lx_id(A->t.voc[h->verb].word, 0), meaning_id(MM, A->t.voc[h->verb].word) }; num += w * (f->verb != NONE ? qsim(&v, f->verb) : 0); den += w; }
    for (int i = 0; i < h->narg; i++) { const Arg *a = &h->a[i]; if (a->head >= OPEN || !content(A, a->head)) continue;
        QW x = { lx_id(A->t.voc[a->head].word, 0), meaning_id(MM, A->t.voc[a->head].word) }; double best = 0;
        for (int j = 0; j < f->narg; j++) { Arg b = { f->role[j], 0, f->prep[j], 0, 0 }, aa = *a; aa.prep = a->prep < OPEN ? lx_id(A->t.voc[a->prep].word, 0) : NONE;
            double s = role_fit(&aa, &b) * fmax(qsim(&x, f->head[j]), f->mod[j] != NONE ? 0.6 * qsim(&x, f->mod[j]) : 0); if (s > best) best = s; }
        num += best; den += 1; }
    if (den <= 0) return 0;
    double s = num / den; if ((h->flags & F_NEG) != (f->flags & F_NEG) && s > 0.5) s *= 0.3;
    return (float)s;
}
static double conf(const KF *f) { return 1.0 - 0.5 / f->count; }

/* re-read one sentence from the shelf through a pointer */
static int reread(Know *A, uint64_t ptr, QW *out, int max) {
    if (ptr == ~0ull) return 0;
    int fid = (int)(ptr >> FID_SHIFT); uint64_t off = ptr & ((1ull << FID_SHIFT) - 1); if (fid >= NSHELF) return 0;
    static FILE *fp[16]; if (!fp[fid]) fp[fid] = fopen(SHELF[fid], "rb"); if (!fp[fid]) return 0;
    char s[1024]; fseeko(fp[fid], (off_t)off, SEEK_SET); size_t n = fread(s, 1, sizeof s - 1, fp[fid]); s[n] = 0;
    char *e = strchr(s, '\n'); if (e) *e = 0;
    for (char *p = s; *p; p++) if ((*p == '.' || *p == '!' || *p == '?') && (p[1] == ' ' || !p[1])) { p[1] = 0; break; }
    Frame fr[64]; uint32_t w[256]; int nw; read_statement(A, s, fr, 64, w, &nw, 256); if (nw > max) nw = max;
    return qwords(A, w, nw, out);
}

static float *mark; static uint32_t nmark;
static void bg_option(Know *A, const char *q, const char *opt, BParts *P) {
    char st[4096]; statement(q, opt, st, sizeof st);
    static Frame hf[64], qf[64], of[64]; uint32_t sw[256], qwl[256], owl[128]; int nsw, nq, now_;
    int nh = read_statement(A, st, hf, 64, sw, &nsw, 256); read_statement(A, q, qf, 64, qwl, &nq, 256); read_statement(A, opt, of, 64, owl, &now_, 128);
    QW Q[256], O[128]; qwords(A, qwl, nq, Q);
    int no = 0; for (int i = 0; i < now_; i++) { int inq = 0; for (int j = 0; j < nq; j++) inq |= qwl[j] == owl[i]; if (!inq) { const char *s = A->t.voc[owl[i]].word; O[no].g = lx_id(s, 0); O[no++].mid = meaning_id(MM, s); } }
    memset(P, 0, sizeof *P);
    /* (d) how close in meaning the option's own words are to the question's */
    P->mean = no && nq ? qcover(O, no, Q, nq) : 0;
    if (!nkf || !no) return;
    /* candidate facts: those holding one of the option's own words, ranked first by the question's words they also hold (exactly) */
    if (nmark < LX.n) { mark = realloc(mark, LX.n * sizeof *mark); memset(mark + nmark, 0, (LX.n - nmark) * sizeof *mark); nmark = LX.n; }
    for (int i = 0; i < nq; i++) if (Q[i].g != NONE) mark[Q[i].g] = (float)gidf(Q[i].g);
    enum { NC = 200 }; uint32_t cand[NC]; float cs[NC]; int nc = 0;
    for (int i = 0; i < no; i++) { if (O[i].g == NONE) continue;
        uint32_t a = cue_off[O[i].g], b = cue_off[O[i].g + 1]; if (b - a > 200000) b = a + 200000;
        for (uint32_t x = a; x < b; x++) { uint32_t fi = cue[x]; uint32_t w[8]; int n = kf_words(&kf[fi], w); float s = 0; for (int j = 0; j < n; j++) s += mark[w[j]];
            s = (float)(s * conf(&kf[fi])) + 1e-3f * (float)conf(&kf[fi]);
            if (nc < NC) { cand[nc] = fi; cs[nc++] = s; } else { int m = 0; for (int j = 1; j < NC; j++) if (cs[j] < cs[m]) m = j; if (s > cs[m]) { cand[m] = fi; cs[m] = s; } } } }
    for (int i = 0; i < nq; i++) if (Q[i].g != NONE) mark[Q[i].g] = 0;
    /* (a) fact agreement, (b) one known fact covering the question and the option */
    int focus = 0; for (int i = 0; i < nh; i++) { uint32_t w[24]; int m; fact_words(&hf[i], w, &m); int has = 0; for (int j = 0; j < m; j++) for (int x = 0; x < now_; x++) has |= w[j] == owl[x]; hf[i].tc = has; focus += has; }
    double fs = 0; int nfs = 0;
    for (int i = 0; i < nh; i++) { if (focus && !hf[i].tc) continue; double b = 0; for (int c = 0; c < nc; c++) { double s = kfact_sim(A, &hf[i], &kf[cand[c]]) * conf(&kf[cand[c]]); if (s > b) b = s; } fs += b; nfs++; }
    P->fact = nfs ? fs / nfs : 0;
    int top[3] = { -1, -1, -1 }; double tv[3] = { -1, -1, -1 };
    for (int c = 0; c < nc; c++) { uint32_t w[8]; int n = kf_words(&kf[cand[c]], w); double j = conf(&kf[cand[c]]) * (kcover(Q, nq, w, n) + kcover(O, no, w, n));
        if (j > P->joint) P->joint = j;
        for (int r = 0; r < 3; r++) if (j > tv[r]) { for (int x = 2; x > r; x--) { tv[x] = tv[x - 1]; top[x] = top[x - 1]; } tv[r] = j; top[r] = cand[c]; break; } }
    /* (c) open book: those facts' pointers lead back to the shelf; the sentences re-read */
    for (int r = 0; r < 3; r++) { if (top[r] < 0) continue;
        for (int p = 0; p < 2; p++) { QW S[256]; int ns = reread(A, kf[top[r]].ptr[p], S, 256); if (!ns) continue;
            double j = qcover(Q, nq, S, ns) + qcover(O, no, S, ns); if (j > P->open) P->open = j; } }
}

static double BW[4] = { 1, 0.5, 2, 4 };      /* chosen on ARC train + dev with a dev brain (5% of the shelf) */
static double bscore(const BParts *p) { return BW[0] * p->fact + BW[1] * p->joint + BW[2] * p->open + BW[3] * p->mean; }
static void answer_ctx(Know *A) { memset(A, 0, sizeof *A); text_init(&A->t); r2_init(&A->R, &A->t, &A->wm); }

/* questions without a passage: id, question, n, options..., answer */
static int run_arc(int nfile, char **files) {
    Know A; answer_ctx(&A);
    for (int fi = 0; fi < nfile; fi++) {
        FILE *f = fopen(files[fi], "r"); if (!f) { perror(files[fi]); return 1; }
        static char line[1 << 16]; double ok = 0, okp[4] = { 0 }; int n = 0; double t0 = now();
        static FILE *dump; if (!dump && getenv("DUMP")) dump = fopen(getenv("DUMP"), "a");
        while (fgets(line, sizeof line, f)) {
            size_t L = strlen(line); while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
            char *fld[12]; int nf = 0; for (char *p = strtok(line, "\t"); p && nf < 12; p = strtok(0, "\t")) fld[nf++] = p;
            if (nf < 4) continue; int no = atoi(fld[2]); if (no < 2 || no > 8 || nf < 4 + no) continue; int want = atoi(fld[3 + no]);
            BParts P[8]; double sc[8];
            for (int o = 0; o < no; o++) { bg_option(&A, fld[1], fld[3 + o], &P[o]); sc[o] = bscore(&P[o]); A.t.ntok = 0; }
            /* ties share the credit */
            double best = -1e9; int nb = 0; for (int o = 0; o < no; o++) if (sc[o] > best + 1e-9) { best = sc[o]; nb = 1; } else if (sc[o] > best - 1e-9) nb++;
            if (sc[want] > best - 1e-9) ok += 1.0 / nb;
            for (int c = 0; c < 4; c++) { double b = -1e9; int k = 0; for (int o = 0; o < no; o++) { double v = c == 0 ? P[o].fact : c == 1 ? P[o].joint : c == 2 ? P[o].open : P[o].mean; if (v > b + 1e-9) { b = v; k = 1; } else if (v > b - 1e-9) k++; }
                double v = c == 0 ? P[want].fact : c == 1 ? P[want].joint : c == 2 ? P[want].open : P[want].mean; if (v > b - 1e-9) okp[c] += 1.0 / k; }
            if (dump) { const char *bn = strrchr(files[fi], '/'); fprintf(dump, "%s\t%d\t%d", bn ? bn + 1 : files[fi], want, no); for (int o = 0; o < no; o++) fprintf(dump, "\t%.5f %.5f %.5f %.5f", P[o].fact, P[o].joint, P[o].open, P[o].mean); fprintf(dump, "\n"); }
            if (getenv("SHOW")) printf("  %s  want %d  scores %.2f %.2f %.2f %.2f\n", fld[1], want, sc[0], sc[1], no > 2 ? sc[2] : 0, no > 3 ? sc[3] : 0);
            n++;
        }
        fclose(f); if (dump) fflush(dump);
        const char *b = strrchr(files[fi], '/'); b = b ? b + 1 : files[fi];
        printf("%-22s %5d q   %6.2f%%   (alone: fact %.2f  joint %.2f  open %.2f  meaning %.2f)  %.1f ms/q\n", b, n, 100 * ok / n, 100 * okp[0] / n, 100 * okp[1] / n, 100 * okp[2] / n, 100 * okp[3] / n, 1000 * (now() - t0) / n);
        printf("RESULT bgarc file=%s n=%d acc=%.2f facts=%u\n", b, n, 100 * ok / n, nkf);
    }
    return 0;
}

/* with a passage: K open (know.c, its frozen weights) plus the background score */
static int run_know_bg(int nfile, char **files) {
    double wa = 2, wg = 2, wj = 2, wo = 2, owb = 0, owj = 0.5, wbg = getenv("WBG") ? atof(getenv("WBG")) : 0;   /* chosen on RACE-M dev + MCTest train/dev: background hurt there at every weight tried */
    Know A; answer_ctx(&A);
    for (int fi = 0; fi < nfile; fi++) {
        FILE *f = fopen(files[fi], "r"); if (!f) { perror(files[fi]); return 1; }
        static char line[1 << 17]; static char text[1 << 17]; Know k; int open = 0; int n = 0, okk = 0, okb = 0;
        static FILE *dump; if (!dump && getenv("DUMP")) dump = fopen(getenv("DUMP"), "a");
        while (fgets(line, sizeof line, f)) {
            size_t L = strlen(line); while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
            if (line[0] == 'S') { if (open) know_free(&k); memset(&k, 0, sizeof k); text_init(&k.t); r2_init(&k.R, &k.t, &k.wm); open = 1; continue; }
            if (line[0] == 'T') { snprintf(text, sizeof text, "%s", line + 2); know_read(&k, text, strlen(text)); continue; }
            if (line[0] != 'Q') continue;
            char *fld[9]; int nfld = 0; for (char *p = strtok(line, "\t"); p && nfld < 9; p = strtok(0, "\t")) fld[nfld++] = p;
            if (nfld < 8) continue;
            const char *q = fld[2]; int want = fld[7][0] - 'A';
            uint32_t qw[256]; int nq; static Frame qf[64]; read_statement(&k, q, qf, 64, qw, &nq, 256);
            int top[8], ntop = 0; float ts[8];
            for (int j = 0; j < k.nf; j++) { float a = about(&k, &k.fa[j], qw, nq); if (ntop < 8) { top[ntop] = j; ts[ntop++] = a; } else if (a > ts[7]) { top[7] = j; ts[7] = a; } else continue;
                for (int x = ntop - 1; x > 0 && ts[x] > ts[x - 1]; x--) { float tf = ts[x]; ts[x] = ts[x - 1]; ts[x - 1] = tf; int ti = top[x]; top[x] = top[x - 1]; top[x - 1] = ti; } }
            int ntop5 = ntop < 5 ? ntop : 5;
            double bk = -1e9, bb = -1e9; int ck = 0, cb = 0; double so[4], sb[4];
            for (int o = 0; o < 4; o++) {
                Parts P; score_option(&k, q, fld[3 + o], qw, nq, top, ntop5, &P);
                so[o] = wa * P.fact + owb * P.evid + owj * P.jc + wg * P.open + wo * P.jo;
                BParts B; bg_option(&A, q, fld[3 + o], &B); A.t.ntok = 0; sb[o] = bscore(&B);
                if (so[o] > bk) { bk = so[o]; ck = o; } if (so[o] + wbg * sb[o] > bb) { bb = so[o] + wbg * sb[o]; cb = o; }
            }
            if (dump) { const char *bn = strrchr(files[fi], '/'); fprintf(dump, "%s\t%d", bn ? bn + 1 : files[fi], want); for (int o = 0; o < 4; o++) fprintf(dump, "\t%.5f %.5f", so[o], sb[o]); fprintf(dump, "\n"); }
            okk += ck == want; okb += cb == want; n++;
        }
        if (open) know_free(&k); fclose(f); if (dump) fflush(dump);
        const char *b = strrchr(files[fi], '/'); b = b ? b + 1 : files[fi];
        printf("%-18s %5d q   K open %5.2f%%   K open + background %5.2f%%\n", b, n, 100.0 * okk / n, 100.0 * okb / n);
        printf("RESULT bgknow file=%s n=%d kopen=%.2f withbg=%.2f facts=%u\n", b, n, 100.0 * okk / n, 100.0 * okb / n, nkf);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "bg read BRAIN SECONDS FILE... | bg arc BRAIN|- FILE... | bg know BRAIN|- FILE...\n"); return 1; }
    lex_load(getenv("WORDCLASS") ? getenv("WORDCLASS") : "data/understand/wordclass.tsv");
    static Meaning m; meaning_init(&m);
    if (meaning_load(&m, getenv("MEANING") ? getenv("MEANING") : MEANING_PATH) <= 0) { fprintf(stderr, "no meanings\n"); return 1; }
    meaning_norms(&m); MM = &m; RV = 2;
    if (getenv("BW")) sscanf(getenv("BW"), "%lf %lf %lf %lf", &BW[0], &BW[1], &BW[2], &BW[3]);
    if (!strcmp(argv[1], "read") && argc >= 5) return run_read(argv[2], atof(argv[3]), argc - 4, argv + 4);
    double t0 = now(); if (brain_load(argv[2])) return 1;
    fprintf(stderr, "brain %s: %u facts, %u words (loaded in %.1f s)\n", argv[2], nkf, LX.n, now() - t0);
    if (!strcmp(argv[1], "arc")) return run_arc(argc - 3, argv + 3);
    if (!strcmp(argv[1], "know")) return run_know_bg(argc - 3, argv + 3);
    return 1;
}
