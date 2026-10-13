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
 *   bg serve BRAIN                  one question a line on stdin, JSON answers on stdout (tools/versus/)
 * Env: ROOM (facts, default 6000000), BW="a b c d" answer weights, PART=n read only every n-th block (dev brain),
 *      DUMP=file (per-option parts), SHOW=1.
 *      ROOMS="a b": a subject room of a facts and a general room of b, each with its own clock; files named gen_* are
 *      general reading, the rest the subject (docs/TOPIC_COMPARE.md).
 *      KEEP=1: every fact of the FROM brain is learned for good (docs/KEEP_COMPARE.md).
 *      BW="a b c d e": e weighs the chain part, two facts linked by a word in neither question nor option (docs/CHAIN_COMPARE.md).
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
typedef struct { uint32_t verb, head[KA], mod[KA], prep[KA], count; uint8_t role[KA], narg, flags; uint64_t ptr[2]; float s; uint32_t L; uint16_t round; uint8_t uses, perm, room; } KF;
/* strength (docs/STRENGTH_COMPARE.md): s and the time last met L; the clock moves one step a sentence read or a question practised */
typedef struct { double fact, joint, open, mean, chain; uint32_t used[16]; int nused; uint32_t cq, co, cw; uint64_t optr; } BParts;
static int STRENGTH; static uint64_t CLK; static const double H0 = 1e6;
/* rooms (docs/TOPIC_COMPARE.md): room 0 the subject, room 1 general reading; each forgets on its own and has its own clock */
static int NROOM = 1, CUR; static uint64_t CLK1; static uint32_t RROOM[2], nroom[2];
#define CLOCK(r) ((r) ? CLK1 : CLK)
/* consolidation: a fact that helped answer right in three different practice sessions is learned for good: it no longer fades and is never forgotten */
static int ROUND;
static double eff(const KF *f) { if (!STRENGTH) return f->count; if (f->perm) return f->s; double age = (double)(CLOCK(f->room) - f->L); return f->s * exp2(-age / (H0 * f->s)); }
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
static int cmp_f(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
static void forget(void) {
    if (STRENGTH) {                       /* the weakest now (strength faded by time) go, until a quarter of the room is free */
        if (NROOM > 1) {                  /* only the full room forgets, by its own clock */
            uint32_t m = nroom[CUR]; float *e = malloc((size_t)nkf * sizeof *e), *c = malloc((size_t)m * sizeof *c); uint32_t k = 0;
            for (uint32_t i = 0; i < nkf; i++) if (kf[i].room == CUR) c[k++] = e[i] = kf[i].perm ? 1e30f : (float)eff(&kf[i]);
            qsort(c, m, sizeof *c, cmp_f); uint32_t drop = m - RROOM[CUR] * 3 / 4; float th = c[drop]; uint32_t j = 0, dropped = 0;
            for (uint32_t i = 0; i < nkf; i++) if (kf[i].room != CUR || e[i] > th || (e[i] == th && (dropped >= drop || kf[i].perm))) kf[j++] = kf[i]; else { dropped++; forgotten++; }
            nkf = j; nroom[CUR] -= dropped; free(e); free(c); forget_th++; kf_rehash(kmask + 1); return;
        }
        float *e = malloc((size_t)nkf * sizeof *e); for (uint32_t i = 0; i < nkf; i++) e[i] = kf[i].perm ? 1e30f : (float)eff(&kf[i]);
        float *c = malloc((size_t)nkf * sizeof *c); memcpy(c, e, (size_t)nkf * sizeof *c); qsort(c, nkf, sizeof *c, cmp_f);
        uint32_t drop = nkf - ROOM * 3 / 4; float th = c[drop]; uint32_t j = 0, dropped = 0;
        for (uint32_t i = 0; i < nkf; i++) if (e[i] > th || (e[i] == th && (dropped >= drop || kf[i].perm))) kf[j++] = kf[i]; else { dropped++; forgotten++; }
        nkf = j; free(e); free(c); forget_th++; kf_rehash(kmask + 1); return;
    }
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
    for (; kslot[h]; h = (h + 1) & kmask) { KF *e = &kf[kslot[h] - 1]; if (kf_same(e, f)) { if (e->count < 0xffffffffu) e->count++; if (e->count == 2) e->ptr[1] = ptr; if (STRENGTH) { e->s = (float)(eff(e) + 1); e->L = (uint32_t)CLOCK(e->room); } return; } }
#define FULL (NROOM > 1 ? nroom[CUR] >= RROOM[CUR] : nkf >= ROOM)
    if (FULL) { forget(); if (FULL) return; kf_add(f, ptr); return; }   /* a room full of facts learned for good: the new one is not kept */
    if (nkf == capkf) { capkf = capkf * 2 > ROOM ? ROOM : capkf * 2; kf = realloc(kf, (size_t)capkf * sizeof *kf); }
    if ((nkf + 1) * 2 > kmask) { kf[nkf] = *f; kf_rehash((kmask + 1) * 2); h = (uint32_t)kf_hash(f) & kmask; while (kslot[h]) h = (h + 1) & kmask; }
    KF *e = &kf[nkf]; *e = *f; e->count = 1; e->ptr[0] = ptr; e->ptr[1] = ~0ull; e->s = 1; e->L = (uint32_t)CLOCK(CUR); e->round = 0; e->uses = 0; e->perm = 0; e->room = (uint8_t)CUR; nroom[CUR]++; kslot[h] = ++nkf;
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
static const char *SHELF[64]; static int NSHELF;
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
        k->R.question = 0; int nf = read_span(&k->R, t->tok, a, i + 1, fr, 512, 0); nsent_read++; if (CUR) CLK1++; else CLK++;
        for (int j = 0; j < nf; j++) { KF f; if (to_kf(k, &fr[j], &f)) { kf_add(&f, ptr); nfacts_read++; } }
        a = i + 1;
    }
    t->ntok = 0; t->src_len = 0;
}

static int brain_save(const char *path, double secs, uint64_t bytes);
static int brain_load(const char *path);
static void build_cues(void);
typedef struct { char *line; char *q, *opt[8]; int no, want; } PQ;
static PQ *PQS; static int NPQ;
static void practice_load(const char *list) {
    char *l = strdup(list);
    for (char *fn = strtok(l, " "); fn; fn = strtok(0, " ")) {
        FILE *f = fopen(fn, "r"); if (!f) { perror(fn); continue; } char line[1 << 16];
        while (fgets(line, sizeof line, f)) { size_t L = strlen(line); while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
            PQ p; memset(&p, 0, sizeof p); p.line = strdup(line); char *fld[12]; int nf = 0; char *sv;
            for (char *x = strtok_r(p.line, "\t", &sv); x && nf < 12; x = strtok_r(0, "\t", &sv)) fld[nf++] = x;
            if (nf < 4) { free(p.line); continue; } p.no = atoi(fld[2]); if (p.no < 2 || p.no > 8 || nf < 4 + p.no) { free(p.line); continue; }
            p.q = fld[1]; for (int o = 0; o < p.no; o++) p.opt[o] = fld[3 + o]; p.want = atoi(fld[3 + p.no]);
            PQS = realloc(PQS, (NPQ + 1) * sizeof *PQS); PQS[NPQ++] = p; }
        fclose(f); }
    free(l);
    uint64_t r = 12345; for (int i = NPQ - 1; i > 0; i--) { r = r * 6364136223846793005ull + 1442695040888963407ull; int j = (int)((r >> 33) % (uint64_t)(i + 1)); PQ t = PQS[i]; PQS[i] = PQS[j]; PQS[j] = t; }   /* the sets mixed, the same way every time */
}
static void bg_option(Know *A, const char *q, const char *opt, BParts *P);
static void answer_ctx(Know *A);
static double bscore(const BParts *p);
static uint64_t npractised, nperm; static double practice_ok;
/* practise questions [a, b): answer, then be shown the right answer; the facts that support it are strengthened, those behind a wrong choice weakened */
static void practise(Know *A, int a, int b, const char *what) {
    if (a < 0) a = 0; if (b > NPQ) b = NPQ; if (a >= b) return;
    double t0 = now(); build_cues(); double ok = 0; ROUND++;
    for (int i = a; i < b; i++) { PQ *p = &PQS[i]; BParts P[8]; int best = 0;
        for (int o = 0; o < p->no; o++) { bg_option(A, p->q, p->opt[o], &P[o]); A->t.ntok = 0; if (bscore(&P[o]) > bscore(&P[best])) best = o; }
        ok += best == p->want; CLK++;
        for (int x = 0; x < P[p->want].nused; x++) { KF *e = &kf[P[p->want].used[x]]; e->s = (float)(eff(e) + 2); e->L = (uint32_t)CLOCK(e->room);
            if (best == p->want && e->round != ROUND) { e->round = (uint16_t)ROUND; if (e->uses < 255) e->uses++; if (e->uses >= 3 && !e->perm) { e->perm = 1; nperm++; } } }
        if (best != p->want) for (int x = 0; x < P[best].nused; x++) { KF *e = &kf[P[best].used[x]]; int also = 0; for (int y = 0; y < P[p->want].nused; y++) also |= P[p->want].used[y] == P[best].used[x]; if (!also) { e->s = (float)(eff(e) * 0.5); e->L = (uint32_t)CLOCK(e->room); } } }
    npractised += b - a; practice_ok += ok;
    fprintf(stderr, "    practised %s: %d questions, %.1f%% right before being shown the answer, %llu facts learned for good so far (%.0f s)\n", what, b - a, 100 * ok / (b - a), (unsigned long long)nperm, now() - t0);
}
static int is_general(const char *fn) { const char *b = strrchr(fn, '/'); b = b ? b + 1 : fn; return NROOM > 1 && !strncmp(b, "gen_", 4); }
static int run_read(const char *brain, double seconds, int nfile, char **files) {
    Know k; memset(&k, 0, sizeof k); text_init(&k.t); r2_init(&k.R, &k.t, &k.wm);
    if (getenv("ROOM")) ROOM = (uint32_t)atol(getenv("ROOM"));
    if (getenv("ROOMS")) { NROOM = 2; if (sscanf(getenv("ROOMS"), "%u %u", &RROOM[0], &RROOM[1]) != 2) return 1; ROOM = RROOM[0] + RROOM[1]; }
    int part = getenv("PART") ? atoi(getenv("PART")) : 1;
    STRENGTH = getenv("STRENGTH") ? atoi(getenv("STRENGTH")) : 0;
    if (getenv("FROM")) { setenv("NOCUES", "1", 1); if (brain_load(getenv("FROM"))) return 1; unsetenv("NOCUES");
        capkf = nkf > (1u << 20) ? nkf : 1u << 20; kf = realloc(kf, (size_t)capkf * sizeof *kf); uint32_t sl = 1u << 21; while (sl < 2 * nkf + 2) sl *= 2; kf_rehash(sl); for (uint32_t i = 0; i < nkf; i++) nroom[kf[i].room]++;
        if (getenv("KEEP") && atoi(getenv("KEEP"))) { for (uint32_t i = 0; i < nkf; i++) kf[i].perm = 1; fprintf(stderr, "what it knew: %u facts learned for good\n", nkf); }
        fprintf(stderr, "continuing %s: %u facts, %u words, clock %llu\n", getenv("FROM"), nkf, LX.n, (unsigned long long)CLK); }
    uint64_t stopbytes = getenv("STOPBYTES") ? strtoull(getenv("STOPBYTES"), 0, 10) : 0;
    Know A; int nslice = 0;
    if (getenv("PRACTICE") && STRENGTH) { answer_ctx(&A); practice_load(getenv("PRACTICE"));
        uint64_t sum = 0; for (int fi = 0; fi < nfile; fi++) { FILE *f = fopen(files[fi], "rb"); if (!f) break; fseeko(f, 0, SEEK_END); sum += (uint64_t)ftello(f); fclose(f); if (!is_general(files[fi])) nslice++; if (stopbytes && sum >= stopbytes) break; }
        fprintf(stderr, "practice: %d questions in %d slices, one after each %sshelf file, each reviewed three files later\n", NPQ, nslice, NROOM > 1 ? "subject " : ""); }
    int si = 0;                       /* subject files read so far: practice follows them */
    double t0 = now(), next = t0 + 60; uint64_t bytes = 0; const size_t B = 1 << 22; char *buf = malloc(B + 1); int stop = 0;
    for (int fi = 0; fi < nfile && !stop; fi++) {
        SHELF[NSHELF++] = files[fi]; CUR = is_general(files[fi]);
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
            if (tn - t0 > seconds || (stopbytes && bytes >= stopbytes)) { stop = 1; break; }
            if (!got && !have) break;
            if (!got) { if (have) { read_block(&k, buf, have, NSHELF - 1, off); bytes += have; } break; }
        }
        fclose(f);
        if (!CUR && nslice && si < nslice) { int per = (NPQ + nslice - 1) / nslice; char w[64]; CUR = 0;
            snprintf(w, sizeof w, "slice %d (new)", si); practise(&A, si * per, (si + 1) * per, w);
            if (si >= 3) { snprintf(w, sizeof w, "slice %d (review)", si - 3); practise(&A, (si - 3) * per, (si - 2) * per, w); } }
        if (!CUR) si++;
    }
    double secs = now() - t0;
    if (npractised) fprintf(stderr, "practised %llu questions in all, %.1f%% right before being shown the answer; %llu facts learned for good\n", (unsigned long long)npractised, 100 * practice_ok / npractised, (unsigned long long)nperm);
    fprintf(stderr, "read %.0f MB in %.0f s (%.2f MB/s): %llu sentences, %llu facts read; %u kept, %llu forgotten%s\n", bytes / 1e6, secs, bytes / 1e6 / secs,
        (unsigned long long)nsent_read, (unsigned long long)nfacts_read, nkf, (unsigned long long)forgotten, stop ? " (time up)" : " (shelf finished)");
    if (NROOM > 1) fprintf(stderr, "rooms: subject %u facts (clock %llu), general %u facts (clock %llu)\n", nroom[0], (unsigned long long)CLK, nroom[1], (unsigned long long)CLK1);
    return brain_save(brain, secs, bytes);
}

/* ------------------------------------------------------------------ */
/* the brain on disk                                                      */
/* ------------------------------------------------------------------ */

static int vput64(uint8_t *b, uint64_t v) { int n = 0; while (v >= 128) { b[n++] = (uint8_t)(v | 128); v >>= 7; } b[n++] = (uint8_t)v; return n; }
static uint64_t vget64(FILE *f) { uint64_t v = 0; int sh = 0, c; while ((c = fgetc(f)) >= 0) { v |= (uint64_t)(c & 127) << sh; if (c < 128) break; sh += 7; } return v; }
static int brain_save(const char *path, double secs, uint64_t bytes) {
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); return 1; }
    fwrite(NROOM > 1 ? "MIMB5" : "MIMB4", 1, 5, f); fwrite(&CLK, 8, 1, f); if (NROOM > 1) fwrite(&CLK1, 8, 1, f); fwrite(&NSHELF, 4, 1, f);
    for (int i = 0; i < NSHELF; i++) { uint16_t l = strlen(SHELF[i]); fwrite(&l, 2, 1, f); fwrite(SHELF[i], 1, l, f); }
    fwrite(&LX.n, 4, 1, f); for (uint32_t i = 0; i < LX.n; i++) { uint8_t l = strlen(LX.s[i]) > 255 ? 255 : strlen(LX.s[i]); fwrite(&l, 1, 1, f); fwrite(LX.s[i], 1, l, f); }
    fwrite(&nkf, 4, 1, f);                       /* facts as varints: ids + 1, a role byte, count, pointers */
    for (uint32_t i = 0; i < nkf; i++) { const KF *e = &kf[i]; uint8_t b[160]; int n = 0;
        n += vput(b + n, e->verb + 1); b[n++] = e->narg | (e->flags & F_NEG) << 4;
        for (int j = 0; j < e->narg; j++) { b[n++] = e->role[j]; n += vput(b + n, e->prep[j] + 1); n += vput(b + n, e->head[j] + 1); n += vput(b + n, e->mod[j] + 1); }
        n += vput(b + n, e->count); n += vput64(b + n, e->ptr[0] + 1); n += vput64(b + n, e->count > 1 ? e->ptr[1] + 1 : 0);
        memcpy(b + n, &e->s, 4); n += 4; n += vput(b + n, e->L); b[n++] = e->perm; if (NROOM > 1) b[n++] = e->room; fwrite(b, 1, n, f); }
    long sz = ftell(f); fclose(f);
    printf("RESULT bgread brain=%s bytes=%ld facts=%u words=%u forgotten=%llu read_bytes=%llu seconds=%.0f mb_s=%.2f\n", path, sz, nkf, LX.n,
        (unsigned long long)forgotten, (unsigned long long)bytes, secs, bytes / 1e6 / secs);
    return 0;
}
static char *SHELFP[64];
static uint32_t *cue_off, *cue;              /* word → facts that hold it, strongest first */
static float *EF;
static int cmp_strong(const void *a, const void *b) { float x = EF[*(const uint32_t *)a], y = EF[*(const uint32_t *)b]; return x < y ? 1 : x > y ? -1 : 0; }
static int kf_words(const KF *f, uint32_t *w) { int n = 0; if (f->verb != NONE) w[n++] = f->verb; for (int i = 0; i < f->narg; i++) { w[n++] = f->head[i]; if (f->mod[i] != NONE) w[n++] = f->mod[i]; } return n; }
static void build_cues(void) {
    free(cue_off); free(cue); free(EF); EF = malloc(((size_t)nkf + 1) * sizeof *EF); for (uint32_t i = 0; i < nkf; i++) EF[i] = (float)eff(&kf[i]);
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
    char mg[5]; if (fread(mg, 1, 5, f) != 5 || (memcmp(mg, "MIMB2", 5) && memcmp(mg, "MIMB3", 5) && memcmp(mg, "MIMB4", 5) && memcmp(mg, "MIMB5", 5))) { fprintf(stderr, "%s: not a brain\n", path); return 1; }
    int v3 = mg[4] >= '3', v4 = mg[4] >= '4', v5 = mg[4] >= '5'; CLK = 0; if (v3 && fread(&CLK, 8, 1, f) != 1) return 1; if (v5 && fread(&CLK1, 8, 1, f) != 1) return 1;
    if (fread(&NSHELF, 4, 1, f) != 1) return 1;
    for (int i = 0; i < NSHELF; i++) { uint16_t l; if (fread(&l, 2, 1, f) != 1) return 1; SHELFP[i] = calloc(l + 1, 1); if (fread(SHELFP[i], 1, l, f) != l) return 1; SHELF[i] = SHELFP[i]; }
    uint32_t nw; if (fread(&nw, 4, 1, f) != 1) return 1;
    for (uint32_t i = 0; i < nw; i++) { uint8_t l; char s[256]; if (fread(&l, 1, 1, f) != 1 || fread(s, 1, l, f) != l) return 1; s[l] = 0; lx_id(s, 1); }
    if (fread(&nkf, 4, 1, f) != 1) return 1;
    kf = calloc(nkf ? nkf : 1, sizeof *kf);
    for (uint32_t i = 0; i < nkf; i++) { KF *e = &kf[i]; int c;
        e->verb = (uint32_t)vget64(f) - 1; c = fgetc(f); if (c < 0) return 1; e->narg = c & 15; e->flags = (c >> 4) & F_NEG;
        for (int j = 0; j < e->narg; j++) { e->role[j] = fgetc(f); e->prep[j] = (uint32_t)vget64(f) - 1; e->head[j] = (uint32_t)vget64(f) - 1; e->mod[j] = (uint32_t)vget64(f) - 1; }
        e->count = (uint32_t)vget64(f); e->ptr[0] = vget64(f) - 1; e->ptr[1] = vget64(f) - 1;
        if (v3) { if (fread(&e->s, 4, 1, f) != 1) return 1; e->L = (uint32_t)vget64(f); if (v4) { int c2 = fgetc(f); if (c2 < 0) return 1; e->perm = (uint8_t)c2; nperm += e->perm; } if (v5) { int c3 = fgetc(f); if (c3 < 0) return 1; e->room = (uint8_t)(c3 & 1); } } else { e->s = (float)e->count; e->L = 0; } }   /* an older brain: strength = read count */
    fclose(f); if (!getenv("NOCUES")) build_cues(); return 0;
}

/* ------------------------------------------------------------------ */
/* answering from what it knows                                           */
/* ------------------------------------------------------------------ */

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
static double conf(const KF *f) { if (!STRENGTH) return 1.0 - 0.5 / f->count; double e = EF ? EF[f - kf] : eff(f); return 1.0 - 0.5 / (e > 1 ? e : 1); }

/* re-read one sentence from the shelf through a pointer */
static int reread(Know *A, uint64_t ptr, QW *out, int max) {
    if (ptr == ~0ull) return 0;
    int fid = (int)(ptr >> FID_SHIFT); uint64_t off = ptr & ((1ull << FID_SHIFT) - 1); if (fid >= NSHELF) return 0;
    static FILE *fp[64]; if (!fp[fid]) fp[fid] = fopen(SHELF[fid], "rb"); if (!fp[fid]) return 0;
    char s[1024]; fseeko(fp[fid], (off_t)off, SEEK_SET); size_t n = fread(s, 1, sizeof s - 1, fp[fid]); s[n] = 0;
    char *e = strchr(s, '\n'); if (e) *e = 0;
    for (char *p = s; *p; p++) if ((*p == '.' || *p == '!' || *p == '?') && (p[1] == ' ' || !p[1])) { p[1] = 0; break; }
    Frame fr[64]; uint32_t w[256]; int nw; read_statement(A, s, fr, 64, w, &nw, 256); if (nw > max) nw = max;
    return qwords(A, w, nw, out);
}

static float *mark; static uint32_t nmark;
static double BW[5] = { 1, 0.5, 2, 4, 0 };   /* chosen on ARC train + dev with a dev brain (5% of the shelf); the chain's on dev (CHAIN_COMPARE.md) */
/* (e) chain: a fact about the question and a fact about the option, linked by a word in neither (not a common one) */
static int NOPEN = 3;   /* facts whose shelf sentences are re-read when answering (LAYA_LEVEL.md stage 1) */
static int CHAINV = 1; static double LINKIDF = 5;   /* chosen on dev (CHAIN_COMPARE.md): both facts must fit (product), the link word rarer than 1 fact in 150 */
static int is_link(uint32_t g, const QW *Q, int nq, const QW *O, int no) {
    if (g == NONE || g >= LX.n || gidf(g) < LINKIDF) return 0;
    for (int i = 0; i < nq; i++) if (Q[i].g == g) return 0;
    for (int i = 0; i < no; i++) if (O[i].g == g) return 0;
    return 1;
}
static void chain_part(const QW *Q, int nq, const QW *O, int no, const char *q, const uint32_t *cand, int nc, BParts *P) {
    enum { NQC = 200, LS = 4096, PER = 4 };
    static char lastq[4096]; static uint32_t qc[NQC]; static float qv[NQC]; static int nqc = -1;
    if (nqc < 0 || strcmp(lastq, q)) {               /* the question's facts, once per question (mark holds its words' rarity) */
        float cs[NQC]; nqc = 0; snprintf(lastq, sizeof lastq, "%s", q);
        for (int i = 0; i < nq; i++) { if (Q[i].g == NONE) continue; int dup = 0; for (int k = 0; k < i; k++) dup |= Q[k].g == Q[i].g; if (dup) continue;
            uint32_t a = cue_off[Q[i].g], b = cue_off[Q[i].g + 1]; if (b - a > 200000) b = a + 200000;
            for (uint32_t x = a; x < b; x++) { uint32_t fi = cue[x]; int seen = 0; for (int k = 0; k < nqc; k++) seen |= qc[k] == fi; if (seen) continue;
                uint32_t w[8]; int n = kf_words(&kf[fi], w); float s = 0; for (int j = 0; j < n; j++) s += mark[w[j]];
                s = (float)(s * conf(&kf[fi])) + 1e-3f * (float)conf(&kf[fi]);
                if (nqc < NQC) { qc[nqc] = fi; cs[nqc++] = s; } else { int m = 0; for (int j = 1; j < NQC; j++) if (cs[j] < cs[m]) m = j; if (s > cs[m]) { qc[m] = fi; cs[m] = s; } } } }
        for (int k = 0; k < nqc; k++) { uint32_t w[8]; int n = kf_words(&kf[qc[k]], w); qv[k] = (float)kcover(Q, nq, w, n); }
    }
    /* link words of the option's facts → those facts (a few per word) */
    static uint32_t lw[LS], lf[LS][PER]; static float lc[LS][PER]; static uint8_t ln[LS]; memset(ln, 0, sizeof ln);
    for (int c = 0; c < nc; c++) { uint32_t w[8]; int n = kf_words(&kf[cand[c]], w); double co = -1;
        for (int j = 0; j < n; j++) { if (!is_link(w[j], Q, nq, O, no)) continue; if (co < 0) co = kcover(O, no, w, n);
            uint32_t h = (w[j] * 2654435761u) & (LS - 1); while (ln[h] && lw[h] != w[j]) h = (h + 1) & (LS - 1);
            lw[h] = w[j]; int m = ln[h] < PER ? ln[h]++ : 0; if (m == 0 && ln[h] == PER) { for (int k = 1; k < PER; k++) if (lc[h][k] < lc[h][m]) m = k; if (co <= lc[h][m]) continue; }
            lf[h][m] = cand[c]; lc[h][m] = (float)co; } }
    for (int k = 0; k < nqc; k++) { uint32_t w[8]; int n = kf_words(&kf[qc[k]], w); double cq = conf(&kf[qc[k]]);
        for (int j = 0; j < n; j++) { if (!is_link(w[j], Q, nq, O, no)) continue;
            uint32_t h = (w[j] * 2654435761u) & (LS - 1); while (ln[h] && lw[h] != w[j]) h = (h + 1) & (LS - 1); if (!ln[h]) continue;
            for (int m = 0; m < ln[h]; m++) { if (lf[h][m] == qc[k]) continue;
                double v = CHAINV == 1 ? cq * conf(&kf[lf[h][m]]) * qv[k] * lc[h][m] : cq * conf(&kf[lf[h][m]]) * (qv[k] + lc[h][m]) / 2; if (v > P->chain) { P->chain = v; P->cq = qc[k]; P->co = lf[h][m]; P->cw = w[j]; } } } }
}
static void bg_option(Know *A, const char *q, const char *opt, BParts *P) {
    char st[4096]; statement(q, opt, st, sizeof st);
    static Frame hf[64], qf[64], of[64]; uint32_t sw[256], qwl[256], owl[128]; int nsw, nq, now_;
    int nh = read_statement(A, st, hf, 64, sw, &nsw, 256); read_statement(A, q, qf, 64, qwl, &nq, 256); read_statement(A, opt, of, 64, owl, &now_, 128);
    QW Q[256], O[128]; qwords(A, qwl, nq, Q);
    int no = 0; for (int i = 0; i < now_; i++) { int inq = 0; for (int j = 0; j < nq; j++) inq |= qwl[j] == owl[i]; if (!inq) { const char *s = A->t.voc[owl[i]].word; O[no].g = lx_id(s, 0); O[no++].mid = meaning_id(MM, s); } }
    memset(P, 0, sizeof *P); P->optr = ~0ull;
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
    if (BW[4] != 0 || getenv("DUMP")) chain_part(Q, nq, O, no, q, cand, nc, P);
    for (int i = 0; i < nq; i++) if (Q[i].g != NONE) mark[Q[i].g] = 0;
    /* (a) fact agreement, (b) one known fact covering the question and the option */
    int focus = 0; for (int i = 0; i < nh; i++) { uint32_t w[24]; int m; fact_words(&hf[i], w, &m); int has = 0; for (int j = 0; j < m; j++) for (int x = 0; x < now_; x++) has |= w[j] == owl[x]; hf[i].tc = has; focus += has; }
    double fs = 0; int nfs = 0;
    for (int i = 0; i < nh; i++) { if (focus && !hf[i].tc) continue; double b = 0; int bc = -1; for (int c = 0; c < nc; c++) { double s = kfact_sim(A, &hf[i], &kf[cand[c]]) * conf(&kf[cand[c]]); if (s > b) { b = s; bc = c; } } fs += b; nfs++;
        if (bc >= 0 && b > 0.5 && P->nused < 16) P->used[P->nused++] = cand[bc]; }
    P->fact = nfs ? fs / nfs : 0;
    enum { TOPMAX = 64 }; int top[TOPMAX]; double tv[TOPMAX]; int K = NOPEN < 3 ? 3 : NOPEN > TOPMAX ? TOPMAX : NOPEN; for (int r = 0; r < K; r++) { top[r] = -1; tv[r] = -1; }
    for (int c = 0; c < nc; c++) { uint32_t w[8]; int n = kf_words(&kf[cand[c]], w); double j = conf(&kf[cand[c]]) * (kcover(Q, nq, w, n) + kcover(O, no, w, n));
        if (j > P->joint) P->joint = j;
        for (int r = 0; r < K; r++) if (j > tv[r]) { for (int x = K - 1; x > r; x--) { tv[x] = tv[x - 1]; top[x] = top[x - 1]; } tv[r] = j; top[r] = cand[c]; break; } }
    for (int r = 0; r < 3; r++) if (top[r] >= 0 && P->nused < 16) { int dup = 0; for (int x = 0; x < P->nused; x++) dup |= P->used[x] == (uint32_t)top[r]; if (!dup) P->used[P->nused++] = top[r]; }
    /* (c) open book: those facts' pointers lead back to the shelf; the sentences re-read */
    for (int r = 0; r < K; r++) { if (top[r] < 0) continue;
        for (int p = 0; p < 2; p++) { QW S[256]; int ns = reread(A, kf[top[r]].ptr[p], S, 256); if (!ns) continue;
            double j = qcover(Q, nq, S, ns) + qcover(O, no, S, ns); if (j > P->open) { P->open = j; P->optr = kf[top[r]].ptr[p]; } } }
}

static double bscore(const BParts *p) { return BW[0] * p->fact + BW[1] * p->joint + BW[2] * p->open + BW[3] * p->mean + BW[4] * p->chain; }
/* the question asks for the option that is NOT so ("not", "least", "except", ...): support is turned around (LAYA_LEVEL.md stage 2) */
static int POLARITY;
static int asks_opposite(const char *q) {
    static const char *W[] = { "not", "least", "except", "false", "unlikely", "cannot", "never", "incorrect", "untrue", "isn't", "doesn't", "can't", "won't", "wouldn't", "aren't", 0 };
    char w[32]; int n = 0;
    for (const char *s = q;; s++) { if (isalpha((unsigned char)*s) || *s == '\'') { if (n < 31) w[n++] = (char)tolower((unsigned char)*s); continue; }
        if (n) { w[n] = 0; for (int i = 0; W[i]; i++) if (!strcmp(w, W[i])) return 1; n = 0; } if (!*s) break; }
    return 0;
}
static double qscore(const char *q, const BParts *p) { double s = bscore(p); return POLARITY && asks_opposite(q) ? -s : s; }
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
            for (int o = 0; o < no; o++) { bg_option(&A, fld[1], fld[3 + o], &P[o]); sc[o] = qscore(fld[1], &P[o]); A.t.ntok = 0; }
            /* ties share the credit */
            double best = -1e9; int nb = 0; for (int o = 0; o < no; o++) if (sc[o] > best + 1e-9) { best = sc[o]; nb = 1; } else if (sc[o] > best - 1e-9) nb++;
            if (sc[want] > best - 1e-9) ok += 1.0 / nb;
            for (int c = 0; c < 4; c++) { double b = -1e9; int k = 0; for (int o = 0; o < no; o++) { double v = c == 0 ? P[o].fact : c == 1 ? P[o].joint : c == 2 ? P[o].open : P[o].mean; if (v > b + 1e-9) { b = v; k = 1; } else if (v > b - 1e-9) k++; }
                double v = c == 0 ? P[want].fact : c == 1 ? P[want].joint : c == 2 ? P[want].open : P[want].mean; if (v > b - 1e-9) okp[c] += 1.0 / k; }
            if (dump) { const char *bn = strrchr(files[fi], '/'); fprintf(dump, "%s\t%d\t%d", bn ? bn + 1 : files[fi], want, no); for (int o = 0; o < no; o++) fprintf(dump, "\t%.5f %.5f %.5f %.5f %.5f", P[o].fact, P[o].joint, P[o].open, P[o].mean, P[o].chain); fprintf(dump, "\n"); }
            if (getenv("CHAINSHOW")) { printf("Q %s  [want %d]\n", fld[1], want); for (int o = 0; o < no; o++) { printf("  %c %-28.28s chain %.3f  total %.3f", o == want ? '*' : ' ', fld[3 + o], P[o].chain, sc[o]);
                if (P[o].chain > 0) { uint32_t w[8]; int n; printf("  via %s: (", LX.s[P[o].cw]); n = kf_words(&kf[P[o].cq], w); for (int j = 0; j < n; j++) printf(" %s", w[j] < LX.n ? LX.s[w[j]] : "?"); printf(" ) + ("); n = kf_words(&kf[P[o].co], w); for (int j = 0; j < n; j++) printf(" %s", w[j] < LX.n ? LX.s[w[j]] : "?"); printf(" )"); }
                printf("\n"); } }
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

/* one question at a time (tools/versus/): "question<TAB>option<TAB>option..." in, one JSON line out:
   each option's score and the shelf sentence its open-book part came from */
static void jstr(const char *s) { putchar('"'); for (; *s; s++) { unsigned char c = (unsigned char)*s; if (c == '"' || c == '\\') printf("\\%c", c); else if (c < 32) putchar(' '); else putchar(c); } putchar('"'); }
static int run_serve(void) {
    Know A; answer_ctx(&A); char line[1 << 16];
    printf("{\"ready\":true,\"facts\":%u}\n", nkf); fflush(stdout);
    while (fgets(line, sizeof line, stdin)) {
        size_t L = strlen(line); while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        char *fld[10]; int nf = 0; for (char *p = strtok(line, "\t"); p && nf < 10; p = strtok(0, "\t")) fld[nf++] = p;
        if (nf < 3) { printf("{\"error\":\"need a question and at least two options\"}\n"); fflush(stdout); continue; }
        double t0 = now(); int no = nf - 1, best = 0; BParts P[9]; double sc[9];
        for (int o = 0; o < no; o++) { bg_option(&A, fld[0], fld[1 + o], &P[o]); sc[o] = qscore(fld[0], &P[o]); A.t.ntok = 0; if (sc[o] > sc[best] + 1e-9) best = o; }
        printf("{\"pick\":%d,\"ms\":%.0f,\"options\":[", best, 1000 * (now() - t0));
        for (int o = 0; o < no; o++) { char s[1024] = ""; uint64_t ptr = P[o].optr;
            if (ptr != ~0ull) { int fid = (int)(ptr >> FID_SHIFT); FILE *f = fid < NSHELF ? fopen(SHELF[fid], "rb") : NULL;
                if (f) { fseeko(f, (off_t)(ptr & ((1ull << FID_SHIFT) - 1)), SEEK_SET); size_t n = fread(s, 1, sizeof s - 1, f); s[n] = 0; fclose(f);
                    char *e = strchr(s, '\n'); if (e) *e = 0; for (char *x = s; *x; x++) if ((*x == '.' || *x == '!' || *x == '?') && (x[1] == ' ' || !x[1])) { x[1] = 0; break; } } }
            printf("%s{\"score\":%.4f,\"fact\":%.4f,\"joint\":%.4f,\"open\":%.4f,\"meaning\":%.4f,\"from\":", o ? "," : "", sc[o], P[o].fact, P[o].joint, P[o].open, P[o].mean); jstr(s);
            const char *src = ""; if (ptr != ~0ull && (int)(ptr >> FID_SHIFT) < NSHELF) { src = strrchr(SHELF[ptr >> FID_SHIFT], '/'); src = src ? src + 1 : SHELF[ptr >> FID_SHIFT]; }
            printf(",\"book\":"); jstr(src); printf("}"); }
        printf("]}\n"); fflush(stdout);
    }
    return 0;
}

/* pack: the brain with only the shelf sentences its pointers lead to, in one file (to carry the brain elsewhere, e.g.
   Colab); each sentence is cut exactly as reread() cuts it, so answers do not change */
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static int run_pack(const char *outbrain, const char *outshelf) {
    uint64_t n = 0, *P = malloc((size_t)nkf * 2 * sizeof *P); for (uint32_t i = 0; i < nkf; i++) for (int p = 0; p < 2; p++) if (kf[i].ptr[p] != ~0ull) P[n++] = kf[i].ptr[p];
    qsort(P, n, sizeof *P, cmp_u64); uint64_t m = 0; for (uint64_t i = 0; i < n; i++) if (!m || P[i] != P[m - 1]) P[m++] = P[i];
    uint64_t *NP = malloc(m * sizeof *NP); FILE *o = fopen(outshelf, "wb"); if (!o) { perror(outshelf); return 1; } uint64_t off = 0; FILE *f = NULL; int cur = -1;
    for (uint64_t i = 0; i < m; i++) { int fid = (int)(P[i] >> FID_SHIFT); char s[1024] = "";
        if (fid != cur) { if (f) fclose(f); f = fid < NSHELF ? fopen(SHELF[fid], "rb") : NULL; cur = fid; if (!f) { perror(SHELF[fid]); return 1; } }
        fseeko(f, (off_t)(P[i] & ((1ull << FID_SHIFT) - 1)), SEEK_SET); size_t k = fread(s, 1, sizeof s - 1, f); s[k] = 0;
        char *e = strchr(s, '\n'); if (e) *e = 0; for (char *x = s; *x; x++) if ((*x == '.' || *x == '!' || *x == '?') && (x[1] == ' ' || !x[1])) { x[1] = 0; break; }
        NP[i] = off; size_t L = strlen(s); fwrite(s, 1, L, o); fputc('\n', o); off += L + 1; }
    if (f) fclose(f); fclose(o);
    for (uint32_t i = 0; i < nkf; i++) for (int p = 0; p < 2; p++) if (kf[i].ptr[p] != ~0ull) { uint64_t *h = bsearch(&kf[i].ptr[p], P, m, sizeof *P, cmp_u64); kf[i].ptr[p] = NP[h - P]; }
    NSHELF = 1; SHELF[0] = outshelf;   /* the packed shelf, by the path given (relative to where Mimir runs) */
    fprintf(stderr, "packed %llu sentences, %.1f MB\n", (unsigned long long)m, off / 1e6);
    return brain_save(outbrain, 0, 0);
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
    if (getenv("BW")) sscanf(getenv("BW"), "%lf %lf %lf %lf %lf", &BW[0], &BW[1], &BW[2], &BW[3], &BW[4]);
    if (getenv("POLARITY")) POLARITY = atoi(getenv("POLARITY"));
    if (getenv("NOPEN")) NOPEN = atoi(getenv("NOPEN"));
    if (getenv("CHAINV")) CHAINV = atoi(getenv("CHAINV")); if (getenv("LINKIDF")) LINKIDF = atof(getenv("LINKIDF"));
    if (!strcmp(argv[1], "read") && argc >= 5) return run_read(argv[2], atof(argv[3]), argc - 4, argv + 4);
    STRENGTH = getenv("STRENGTH") ? atoi(getenv("STRENGTH")) : 0;
    double t0 = now(); if (brain_load(argv[2])) return 1;
    fprintf(stderr, "brain %s: %u facts, %u words (loaded in %.1f s)\n", argv[2], nkf, LX.n, now() - t0);
    if (!strcmp(argv[1], "arc")) return run_arc(argc - 3, argv + 3);
    if (!strcmp(argv[1], "know")) return run_know_bg(argc - 3, argv + 3);
    if (!strcmp(argv[1], "serve")) return run_serve();
    if (!strcmp(argv[1], "pack") && argc >= 5) return run_pack(argv[3], argv[4]);
    return 1;
}
