/*
 * bg.c — Mimir's brain: it reads a shelf of text and keeps what it understood (facts) and where it read it
 * (pointers), not the words; then answers questions from what it keeps. docs/JOURNAL.md says what is in use.
 *
 * Reading: each sentence is read into facts (know.c's reader). A fact already known is strengthened, not stored
 * again; it keeps at most two pointers to where it was read (shelf file, byte offset of the sentence). Strength
 * fades with time unless the fact is met again (docs/past/STRENGTH_COMPARE.md); with no room limit nothing is forgotten
 * unless memory runs low. Cue lists word → facts are how it knows where to look. The shelf stays on disk.
 *
 *   bg read BRAIN SECONDS FILE...   read the shelf files in order for at most SECONDS, save BRAIN
 *   bg arc BRAIN FILE.tsv...        questions without a passage: id, question, n, options..., answer
 *   bg serve BRAIN                  one question a line on stdin, JSON answers on stdout (tools/versus/)
 *   bg passages BRAIN FILE...       passages and their questions (MCTest layout), each passage read as the episode
 *                                   (docs/EPISODE.md; EPW its weight over the shelf) and put back after its questions
 *   bg words BRAIN FILE...          the cortex on word-pair files (Spearman against people's ratings)
 *   bg near BRAIN WORD...           the words the cortex puts nearest
 *   bg facts -                      the facts the reader makes of each line on stdin (no brain)
 * Env: ROOM (facts, default 0 = no limit), FROM=brain (read on from it), PART=n (read only every n-th block: a
 *      dev brain), DUMP=file (each option's parts).
 *      The cortex (docs/CORTEX_LINK.md) learns from the facts while reading (BRAIN.cx beside the brain) and compares
 *      words when answering; CXTH the cosine a link counts from (0.40), CXFIT the fit weight (1).
 * Run from the repo root.
 */
#include "facts.c"
#include <pthread.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* the brain's words: one table for everything it ever read              */
/* ------------------------------------------------------------------ */

typedef struct { char **s; uint32_t n, cap, mask, *slot; } Lex;
static Lex LX;
static uint32_t lx_id(const char *s, int add) {
    if (!LX.slot) { LX.mask = (1 << 20) - 1; LX.slot = calloc(LX.mask + 1, 4); }
    uint32_t h = shash(s) & LX.mask;
    for (; LX.slot[h]; h = (h + 1) & LX.mask) if (!strcmp(LX.s[LX.slot[h] - 1], s)) return LX.slot[h] - 1;
    if (!add) return NONE;
    if (LX.n * 2 >= LX.mask) { uint32_t m = LX.mask * 2 + 1; uint32_t *sl = calloc(m + 1, 4);
        for (uint32_t i = 0; i < LX.n; i++) { uint32_t g = shash(LX.s[i]) & m; while (sl[g]) g = (g + 1) & m; sl[g] = i + 1; }
        free(LX.slot); LX.slot = sl; LX.mask = m; h = shash(s) & m; while (LX.slot[h]) h = (h + 1) & m; }
    if (LX.n == LX.cap) { LX.cap = LX.cap ? LX.cap * 2 : 1 << 16; LX.s = realloc(LX.s, LX.cap * sizeof *LX.s); }
    LX.s[LX.n] = strdup(s); LX.slot[h] = ++LX.n; return LX.n - 1;
}

/* ------------------------------------------------------------------ */
/* the cortex (docs/CORTEX_LINK.md): each word has D meaning cells and D context cells, learned from the facts as   */
/* they are read, with K3b's rule (docs/past/CORTEX.md v0.1). Words come in blocks of 65,536, made when first met.  */
/* ------------------------------------------------------------------ */

enum { CXD = 300, CXNEG = 5, CXB = 16, CXBLK = 1 << CXB, CXMAXB = 256 };
static float *cxu[CXMAXB], *cxc[CXMAXB]; static uint32_t *cxn[CXMAXB]; static int cxnb;   /* meaning cells, context cells, times met */
static float *cxinv;                              /* 1 / |meaning cells|, for answering */
static double CXTH = 0.40; static int CXMIN = 5;  /* a link counts from this cosine (chosen on dev); a word met fewer times is matched only exactly */
#define CXU(g) (cxu[(g) >> CXB] + (size_t)((g) & (CXBLK - 1)) * CXD)
#define CXC(g) (cxc[(g) >> CXB] + (size_t)((g) & (CXBLK - 1)) * CXD)
#define CXN(g) (cxn[(g) >> CXB][(g) & (CXBLK - 1)])
static uint64_t cx_rnd(uint64_t *st) { *st ^= *st << 13; *st ^= *st >> 7; *st ^= *st << 17; return *st; }
static void cx_block(void) {                      /* room for 65,536 more words: meaning cells small and random, context cells zero */
    int b = cxnb; uint64_t st = 0x9E3779B97F4A7C15ull * (uint64_t)(b + 1);
    cxu[b] = malloc((size_t)CXBLK * CXD * 4); cxc[b] = calloc((size_t)CXBLK * CXD, 4); cxn[b] = calloc(CXBLK, 4);
    for (size_t i = 0; i < (size_t)CXBLK * CXD; i++) cxu[b][i] = (float)(((cx_rnd(&st) >> 11) * (1.0 / 9007199254740992.0)) - 0.5) / CXD;
    __atomic_store_n(&cxnb, b + 1, __ATOMIC_RELEASE);
}
static int cx_has(uint32_t g) { return g != NONE && (g >> CXB) < (uint32_t)cxnb && CXN(g) >= (uint32_t)CXMIN; }
static float cx_dot(const float *x, const float *y) { float d = 0; for (int i = 0; i < CXD; i++) d += x[i] * y[i]; return d; }
static void cx_norms(void) {
    free(cxinv); cxinv = calloc((size_t)cxnb * CXBLK, sizeof *cxinv);
    for (uint32_t g = 0; g < (uint32_t)cxnb * CXBLK; g++) { float n = cx_dot(CXU(g), CXU(g)); cxinv[g] = n > 0 ? 1 / sqrtf(n) : 0; }
}

/* a word's meaning: the brain's own word in the cortex, if met often enough (-1: matched only exactly) */
static int wmid(const char *s) { uint32_t g = lx_id(s, 0); return cx_has(g) ? (int)g : -1; }
static int gmid(uint32_t g) { return g < LX.n && cx_has(g) ? (int)g : -1; }
static float mcos(int x, int y) { return x < 0 || y < 0 ? 0 : cx_dot(CXU((uint32_t)x), CXU((uint32_t)y)) * cxinv[x] * cxinv[y]; }
/* a word asked about: its id in the brain (NONE if never read) and its meaning */
typedef struct { uint32_t g; int mid; } QW;
static float qsim(const QW *a, uint32_t g) {
    if (g == NONE) return 0; if (a->g == g) return 1;
    float c = mcos(a->mid, gmid(g)), th = (float)CXTH; return c >= 1 ? 1 : c > th ? (c - th) / (1 - th) : 0;
}
static float qqsim(const QW *a, const QW *b) {
    if (a->g != NONE && a->g == b->g) return 1; if (a->mid >= 0 && a->mid == b->mid) return 1;
    float c = mcos(a->mid, b->mid), th = (float)CXTH; return c >= 1 ? 1 : c > th ? (c - th) / (1 - th) : 0;
}

/* ------------------------------------------------------------------ */
/* known facts, merged                                                   */
/* ------------------------------------------------------------------ */

#define KA 3
typedef struct { uint32_t verb, head[KA], mod[KA], prep[KA], count; uint8_t role[KA], narg, flags; uint64_t ptr[2]; float s; uint32_t L; } KF;
/* strength (docs/past/STRENGTH_COMPARE.md): s and the time last met L; the clock moves one step a sentence read */
typedef struct { double fact, joint, open, mean, fit, efact, ejoint, eopen; int active; uint64_t optr, eptr; } BParts;
static uint64_t CLK; static const double H0 = 1e6;
static double eff(const KF *f) { double age = (double)(CLK - f->L); return f->s * exp2(-age / (H0 * f->s)); }
static KF *kf; static uint32_t nkf, capkf, *kslot, kmask; static uint64_t forgotten; static int forget_th;
static uint32_t ROOM = 0;           /* 0 = no limit: keep every fact; only low memory makes it forget (docs/past/SCHOOL_BRAIN.md) */
static uint64_t mem_free_mb(void) { FILE *f = fopen("/proc/meminfo", "r"); char k[64]; unsigned long long v; uint64_t r = ~0ull;
    if (!f) return r; while (fscanf(f, "%63s %llu kB", k, &v) == 2) if (!strcmp(k, "MemAvailable:")) { r = v / 1024; break; } fclose(f); return r; }
static uint64_t nmemforget;

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
static void forget(void) {               /* the weakest now (strength faded by time) go, until a quarter of the room is free */
    float *e = malloc((size_t)nkf * sizeof *e); for (uint32_t i = 0; i < nkf; i++) e[i] = (float)eff(&kf[i]);
    float *c = malloc((size_t)nkf * sizeof *c); memcpy(c, e, (size_t)nkf * sizeof *c); qsort(c, nkf, sizeof *c, cmp_f);
    uint32_t drop = nkf - ROOM * 3 / 4; float th = c[drop]; uint32_t j = 0, dropped = 0;
    for (uint32_t i = 0; i < nkf; i++) if (e[i] > th || (e[i] == th && dropped >= drop)) kf[j++] = kf[i]; else { dropped++; forgotten++; }
    nkf = j; free(e); free(c); forget_th++; kf_rehash(kmask + 1);
}
static void kf_add(const KF *f, uint64_t ptr) {
    if (!kslot) { capkf = 1 << 20; kf = malloc((size_t)capkf * sizeof *kf); kf_rehash(1u << 21); }
    uint32_t h = (uint32_t)kf_hash(f) & kmask;
    for (; kslot[h]; h = (h + 1) & kmask) { KF *e = &kf[kslot[h] - 1]; if (kf_same(e, f)) { if (e->count < 0xffffffffu) e->count++; if (e->count == 2) e->ptr[1] = ptr; e->s = (float)(eff(e) + 1); e->L = (uint32_t)CLK; return; } }
#define FULL (ROOM ? nkf >= ROOM : 0)
    if (!ROOM && (nkf & 0xfffff) == 0 && nkf && mem_free_mb() < 1536) { ROOM = nkf; nmemforget++; fprintf(stderr, "  low memory at %u facts: forgetting the weakest quarter\n", nkf); forget(); ROOM = 0; }
    if (FULL) { forget(); kf_add(f, ptr); return; }
    if (nkf == capkf) { capkf = ROOM && capkf * 2 > ROOM ? ROOM : capkf * 2; kf = realloc(kf, (size_t)capkf * sizeof *kf); }
    if ((nkf + 1) * 2 > kmask) { kf[nkf] = *f; kf_rehash((kmask + 1) * 2); h = (uint32_t)kf_hash(f) & kmask; while (kslot[h]) h = (h + 1) & kmask; }
    KF *e = &kf[nkf]; *e = *f; e->count = 1; e->ptr[0] = ptr; e->ptr[1] = ~0ull; e->s = 1; e->L = (uint32_t)CLK; kslot[h] = ++nkf;
}

/* ------------------------------------------------------------------ */
/* the episode (docs/EPISODE.md): the text being read now, apart from the shelf's facts      */
/* ------------------------------------------------------------------ */

/* The text being read is kept apart from the shelf's facts and is active while it is asked about: when the question or an
   option speaks of it, its facts answer first, its fit weighted EPW (4, chosen on dev) over the shelf's. */
static int EPREAD; static double EPW = 4;
enum { EPFID = 63 };                                  /* pointers into the episode's own text (in memory) */
static KF *ep; static uint32_t nep, capep; static char *epbuf; static size_t neb, capeb;
static uint8_t *epword; static uint32_t nepword;     /* words the episode holds */
/* docs/STATE.md: each episode fact's order of saying (ep_ord) and whether a later fact of the same state supersedes it (ep_sup) */
static uint8_t *ep_sup, *ep_der; static uint32_t *ep_ord, ep_seq, ep_derive_ord; static double SUPW = 0.25; static int ep_deriving;   /* ep_der: a fact derived from another (S2, S3), not said by the text */
static void ep_add(const KF *f, uint64_t ptr) {
    for (uint32_t i = 0; i < nep; i++) if (kf_same(&ep[i], f)) { ep[i].count++; ep[i].s += 1; if (ep[i].count == 2) ep[i].ptr[1] = ptr; ep_ord[i] = ep_deriving && s5(8192) ? (ep_derive_ord > ep_ord[i] ? ep_derive_ord : ep_ord[i]) : ++ep_seq; return; }   /* Y13: a derived fact said again keeps its source's order, not a fresh one (it was marked latest of all and superseded the real latest state) */
    if (nep == capep) { capep = capep ? capep * 2 : 256; ep = realloc(ep, capep * sizeof *ep); ep_sup = realloc(ep_sup, capep); ep_der = realloc(ep_der, capep); ep_ord = realloc(ep_ord, capep * 4); }
    KF *e = &ep[nep++]; *e = *f; e->count = 1; e->s = 1; e->L = (uint32_t)CLK; e->ptr[0] = ptr; e->ptr[1] = ~0ull; ep_sup[nep - 1] = 0; ep_der[nep - 1] = (uint8_t)ep_deriving; ep_ord[nep - 1] = ++ep_seq;
}
static int is_ep(const KF *f) { return f >= ep && f < ep + nep; }

/* ---- docs/STATE.md: the state now ------------------------------------------------------------------------------- */
/* S2: a motion with "to" is also a place ("Sandra went to the kitchen": Sandra is in the kitchen); taking, giving, dropping and
   putting are also having (or no longer having) and, for a put, a place. S3: a thing held is where its holder is, until it is
   put down. S1: of the facts filling one state (someone's place, a thing's holder, what a thing is made of) the latest is
   current and the earlier ones are superseded: they count SUPW of themselves for a question about the present. */
static uint32_t G_be, G_have, G_in, G_to, G_of, G_make, G_leave, G_motion[32], G_take[32], G_give[32], G_drop[32], G_put[32], G_loc[16];
static int state_ready;
static void gset(uint32_t *out, const char *words) { char b[512]; snprintf(b, sizeof b, "%s", words); int n = 0; for (char *w = strtok(b, " "); w && n < 31; w = strtok(0, " ")) out[n++] = lx_id(w, 1); out[n] = NONE; }
static int gin(const uint32_t *set, uint32_t g) { for (int i = 0; set[i] != NONE; i++) if (set[i] == g) return 1; return 0; }
static void state_init(void) {              /* before any episode, so the words stay in the table */
    if (state_ready) return; state_ready = 1;
    G_be = lx_id("be", 1); G_have = lx_id("have", 1); G_in = lx_id("in", 1); G_to = lx_id("to", 1); G_of = lx_id("of", 1); G_make = lx_id("make", 1); G_leave = lx_id("leave", 1);
    gset(G_motion, "go move travel journey walk run return come head drive fly ride hurry wander sail swim march enter");
    gset(G_take, "take get grab pick hold carry keep receive borrow steal fetch catch find buy collect");
    gset(G_give, "give hand pass lend send bring sell");
    gset(G_drop, "drop discard lose throw");
    gset(G_put, "put place set lay leave");
    gset(G_loc, "in at on inside near under behind beside into onto");
}
static int kf_arg(const KF *f, int role, uint32_t prep) { for (int i = 0; i < f->narg; i++) if (f->role[i] == role && (prep == NONE || f->prep[i] == prep || (prep == G_in && gin(G_loc, f->prep[i])))) return i; return -1; }
static void ep_derive(uint32_t ord, uint32_t verb, uint8_t flags, uint32_t who, uint32_t prep, uint32_t what, uint64_t ptr) {
    KF f; memset(&f, 0, sizeof f); f.verb = verb; f.flags = flags; f.narg = 2;
    f.role[0] = A_AGENT; f.prep[0] = NONE; f.head[0] = who; f.mod[0] = NONE;
    f.role[1] = prep == NONE ? A_PAT : A_PP; f.prep[1] = prep; f.head[1] = what; f.mod[1] = NONE;
    if (f.role[1] < f.role[0]) { f.role[0] = f.role[1]; f.prep[0] = prep; f.head[0] = what; f.role[1] = A_AGENT; f.prep[1] = NONE; f.head[1] = who; }   /* roles in order, as to_kf keeps them */
    ep_derive_ord = ord;
    uint32_t n0 = nep; ep_deriving = 1; ep_add(&f, ptr); ep_deriving = 0; if (nep > n0) ep_ord[nep - 1] = ord;   /* said when its source was said */
    if (nep == n0) for (uint32_t i = 0; i < nep; i++) if (kf_same(&ep[i], &f)) { if (ep_ord[i] < ord) ep_ord[i] = ord; break; }
}
/* the state a fact fills: kind 1 a place (subject, where), 2 a holder (thing, who), 3 made of (thing, what); 0 none */
static uint32_t tracked[64]; static int ntracked;        /* things taken, given, dropped or put somewhere in this episode: their holder is a state */
static int is_tracked(uint32_t g) { for (int i = 0; i < ntracked; i++) if (tracked[i] == g) return 1; return 0; }
static int is_take(const KF *f) { return gin(G_take, f->verb) && kf_arg(f, A_PP, G_to) < 0; }   /* "got the football", not "get to Lab" */
static int state_key(const KF *f, uint32_t *subj, uint32_t *val) {
    int ag = kf_arg(f, A_AGENT, NONE), pa = kf_arg(f, A_PAT, NONE);
    if (gin(G_motion, f->verb) && ag >= 0) { int to = kf_arg(f, A_PP, G_to); if (to >= 0) { *subj = f->head[ag]; *val = f->head[to]; return 1; } }
    if (f->verb == G_be && ag >= 0) { int in = kf_arg(f, A_PP, G_in); if (in >= 0) { *subj = f->head[ag]; *val = f->head[in]; return 1; } }
    if ((f->verb == G_have || is_take(f)) && ag >= 0 && pa >= 0 && is_tracked(f->head[pa])) { *subj = f->head[pa]; *val = f->flags & F_NEG ? NONE : f->head[ag]; return 2; }
    if (f->verb == G_make && (pa >= 0 || ag >= 0)) { int of = kf_arg(f, A_PP, G_of); if (of >= 0) { *subj = f->head[pa >= 0 ? pa : ag]; *val = f->head[of]; return 3; } }
    return 0;
}
/* a statement's frame as a fact in the brain's words (lookup only: nothing added, nothing mapped), for state_key */
static void frame_kf(Know *A, const Frame *h, KF *o) {
    memset(o, 0, sizeof *o); o->verb = h->verb < OPEN ? lx_id(A->t.voc[h->verb].word, 0) : NONE; o->flags = h->flags;
    for (int i = 0; i < h->narg && o->narg < KA; i++) { const Arg *a = &h->a[i]; if (a->head >= OPEN) continue; int j = o->narg++;
        o->role[j] = a->role; o->prep[j] = a->prep < OPEN ? lx_id(A->t.voc[a->prep].word, 0) : NONE; o->head[j] = lx_id(A->t.voc[a->head].word, 0); o->mod[j] = NONE; }
}
/* a statement about a state as such ("Mary is in the kitchen", "Ruth has the key", "the stall is made of bamboo"), not about the
   event that made it ("who took the basket home"): only these meet derived and superseded facts differently */
static int state_stmt(const KF *h) { uint32_t sj, v; int k = state_key(h, &sj, &v); return k == 1 ? h->verb == G_be : k == 2 ? h->verb == G_have : k; }
static void ep_states(void) {
    state_init(); uint32_t n0 = nep; ntracked = 0;
    for (uint32_t i = 0; i < n0; i++) { const KF *f = &ep[i]; int pa = kf_arg(f, A_PAT, NONE); if (pa < 0 || ntracked >= 64 || is_tracked(f->head[pa])) continue;
        if (is_take(f) || gin(G_give, f->verb) || gin(G_drop, f->verb) || (gin(G_put, f->verb) && kf_arg(f, A_PP, G_in) >= 0)) tracked[ntracked++] = f->head[pa]; }
    for (uint32_t i = 0; i < n0; i++) { const KF *f = &ep[i]; if (f->flags & F_NEG) continue;                     /* S2: what an event also says of a state */ int ag = kf_arg(f, A_AGENT, NONE), pa = kf_arg(f, A_PAT, NONE); uint32_t ord = ep_ord[i];
        if (gin(G_motion, f->verb) && ag >= 0) { int to = kf_arg(f, A_PP, G_to); if (to >= 0) ep_derive(ord, G_be, 0, f->head[ag], G_in, f->head[to], ep[i].ptr[0]); }
        if (is_take(f) && ag >= 0 && pa >= 0) ep_derive(ord, G_have, 0, f->head[ag], NONE, f->head[pa], ep[i].ptr[0]);
        if (gin(G_give, f->verb) && ag >= 0 && pa >= 0) { int to = kf_arg(f, A_PP, G_to); if (to >= 0) { ep_derive(ord, G_have, 0, f->head[to], NONE, f->head[pa], ep[i].ptr[0]); ep_derive(ord, G_have, F_NEG, f->head[ag], NONE, f->head[pa], ep[i].ptr[0]); } }
        if (ag >= 0 && pa >= 0 && (gin(G_drop, f->verb) || (s5(262144) && ((gin(G_put, f->verb) && kf_arg(f, A_PP, G_in) < 0) || (f->verb == G_leave && is_tracked(f->head[pa])))))) ep_derive(ord, G_have, F_NEG, f->head[ag], NONE, f->head[pa], ep[i].ptr[0]);   /* Y17: "put down the apple", "left the milk there": no longer held */
        if (gin(G_put, f->verb) && ag >= 0 && pa >= 0) { int in = kf_arg(f, A_PP, G_in); if (in >= 0) { ep_derive(ord, G_be, 0, f->head[pa], G_in, f->head[in], ep[i].ptr[0]); ep_derive(ord, G_have, F_NEG, f->head[ag], NONE, f->head[pa], ep[i].ptr[0]); } } }
    {                                                   /* S3: a thing is where its holder is */
        uint32_t n1 = nep; enum { MAXT = 64 }; uint32_t things[MAXT]; int nt = 0;
        for (uint32_t i = 0; i < n1; i++) { uint32_t sj, v; if (state_key(&ep[i], &sj, &v) == 2) { int dup = 0; for (int t = 0; t < nt; t++) dup |= things[t] == sj; if (!dup && nt < MAXT) things[nt++] = sj; } }
        for (int t = 0; t < nt; t++) { uint32_t T = things[t], holder = NONE;
            for (uint32_t step = 0, guard = 0; guard < 100000; guard++) {           /* the facts in the order said */
                uint32_t best = NONE, bo = ~0u; for (uint32_t i = 0; i < n1; i++) if (ep_ord[i] > step && ep_ord[i] < bo) { bo = ep_ord[i]; best = i; }
                if (best == NONE) break; step = bo; const KF *f = &ep[best]; uint32_t sj, v; int k = state_key(f, &sj, &v);
                if (k == 2 && sj == T) { holder = v;
                    if (holder != NONE) { uint32_t pl = NONE, po = 0; for (uint32_t j = 0; j < n1; j++) { uint32_t s2, v2; if (state_key(&ep[j], &s2, &v2) == 1 && s2 == holder && ep_ord[j] <= bo && ep_ord[j] >= po) { po = ep_ord[j]; pl = v2; } }
                        if (pl != NONE) ep_derive(bo, G_be, 0, T, G_in, pl, f->ptr[0]); } }
                else if (k == 1 && holder != NONE && sj == holder) ep_derive(bo, G_be, 0, T, G_in, v, f->ptr[0]); } } }
    {                                                   /* S1: the latest fact of a state is current, the others superseded */
        for (uint32_t i = 0; i < nep; i++) { uint32_t si, vi; int ki = state_key(&ep[i], &si, &vi); if (!ki) continue;
            for (uint32_t j = 0; j < nep; j++) { if (j == i) continue; uint32_t sj, vj; if (state_key(&ep[j], &sj, &vj) != ki || sj != si || vj == vi) continue;
                if ((ep[j].flags & F_PAST) && (ep[i].flags & F_PRES)) continue;     /* "last year it was wood", said after "now it is bamboo", does not supersede it */
                if (ep_ord[j] > ep_ord[i]) { ep_sup[i] = 1; break; } } } }
}

/* the cortex while a passage is read: it learns as always, and what it changed is put back after the passage, so one
   passage cannot help the next (for measuring) */
static uint32_t *cxring;                              /* (below: words met lately) */
static int cxj; static uint64_t cxj_tot, cxj_nring;
typedef struct { uint32_t g; uint8_t which; float *row; } CxRow;
static CxRow *cxjr; static uint32_t ncxjr, capcxjr; static uint32_t *cxjn_g, *cxjn_v, ncxjn, capcxjn; static uint64_t *cxjring_i; static uint32_t *cxjring_v, ncxjring, capcxjring;
static void cxj_row(uint32_t g, int which) {
    for (uint32_t i = 0; i < ncxjr; i++) if (cxjr[i].g == g && cxjr[i].which == which) return;
    if (ncxjr == capcxjr) { capcxjr = capcxjr ? capcxjr * 2 : 1024; cxjr = realloc(cxjr, capcxjr * sizeof *cxjr); }
    float *r = malloc(CXD * 4); memcpy(r, which ? CXC(g) : CXU(g), CXD * 4); cxjr[ncxjr++] = (CxRow){ g, (uint8_t)which, r };
}
static void cxj_count(uint32_t g) {
    for (uint32_t i = 0; i < ncxjn; i++) if (cxjn_g[i] == g) return;
    if (ncxjn == capcxjn) { capcxjn = capcxjn ? capcxjn * 2 : 1024; cxjn_g = realloc(cxjn_g, capcxjn * 4); cxjn_v = realloc(cxjn_v, capcxjn * 4); }
    cxjn_g[ncxjn] = g; cxjn_v[ncxjn++] = CXN(g);
}
static void cxj_ring(uint64_t i) {
    if (ncxjring == capcxjring) { capcxjring = capcxjring ? capcxjring * 2 : 1024; cxjring_i = realloc(cxjring_i, capcxjring * 8); cxjring_v = realloc(cxjring_v, capcxjring * 4); }
    cxjring_i[ncxjring] = i; cxjring_v[ncxjring++] = cxring[i];
}

/* ------------------------------------------------------------------ */
/* the cortex learning while Mimir reads                                */
/* ------------------------------------------------------------------ */

/* The reader hands each fact's words to the cortex (a queue) as it stores the fact; the cortex learns from them in its
   own thread. Every two words of one fact: reward, the links between one's meaning cells and the other's context cells
   grow by how much the pair was not yet expected (1 - sigma); NEG words drawn from what it met lately: penalty, their
   links shrink by how much they were expected (sigma). Very common words are thinned, as K3b. Each word's rate falls as
   it is met more often, 1/sqrt(1 + n/CXT), not below CXFLOOR of the start: a new word learns fast, a known one slowly
   (L2, chosen on dev over a rate falling with the reading clock). */
enum { CXQ = 1 << 22, CXR = 1 << 24 };
static uint32_t *cxq; static uint64_t cxqh, cxqt; static int cxdone; static pthread_t cxthr;
static uint32_t *cxring; static uint64_t cxnring;       /* words met lately (after thinning), for the penalties */
static uint64_t cxtot, cxpairs;                         /* words met in facts; pairs learned */
static double CXETA = 0.025, CXSUB = 1e-4, CXFLOOR = 0.05, CXT = 10000;
static float cx_sig(float x) { return x > 8 ? 1 : x < -8 ? 0 : 1 / (1 + expf(-x)); }
static void cx_learn(const uint32_t *w, int n, uint64_t *st, float *neu) {
    uint32_t k[16]; int m = 0;
    for (int i = 0; i < n; i++) { uint32_t g = w[i]; if (cxj) cxj_count(g); CXN(g)++; cxtot++;
        double f = (double)CXN(g) / cxtot, keep = (sqrt(f / CXSUB) + 1) * CXSUB / f;
        if (keep < 1 && (cx_rnd(st) >> 11) * (1.0 / 9007199254740992.0) > keep) continue;
        int dup = 0; for (int j = 0; j < m; j++) dup |= k[j] == g; if (dup) continue;
        k[m++] = g; if (cxj) cxj_ring(cxnring & (CXR - 1)); cxring[cxnring++ & (CXR - 1)] = g; }
    if (m < 2 || cxnring < 10000) return;
    uint64_t nr = cxnring < CXR ? cxnring : CXR;
    for (int a = 0; a < m; a++) { uint32_t t = k[a]; float *u = CXU(t); if (cxj) cxj_row(t, 0);
        double r = 1 / sqrt(1 + CXN(t) / CXT);
        float eta = (float)(CXETA * (r > CXFLOOR ? r : CXFLOOR));
        for (int b = 0; b < m; b++) { if (b == a) continue; memset(neu, 0, CXD * 4);
            for (int x = 0; x <= CXNEG; x++) { uint32_t o; float label;
                if (!x) { o = k[b]; label = 1; } else { o = cxring[(cx_rnd(st) >> 11) % nr]; if (o == k[b]) continue; label = 0; }
                if (cxj) cxj_row(o, 1);
                float *v = CXC(o), g = eta * (label - cx_sig(cx_dot(u, v)));   /* reward (label 1) or penalty (0), times the surprise */
                for (int i = 0; i < CXD; i++) { neu[i] += g * v[i]; v[i] += g * u[i]; } }
            for (int i = 0; i < CXD; i++) u[i] += neu[i];
            cxpairs++; } }
}
static void *cx_thread(void *arg) {
    (void)arg; uint64_t st = 88172645463325252ull; float *neu = malloc(CXD * 4); uint32_t w[16];
    for (;;) { uint64_t h = __atomic_load_n(&cxqh, __ATOMIC_ACQUIRE);
        if (cxqt == h) { if (__atomic_load_n(&cxdone, __ATOMIC_ACQUIRE) && cxqt == __atomic_load_n(&cxqh, __ATOMIC_ACQUIRE)) break; usleep(200); continue; }
        uint32_t n = cxq[cxqt & (CXQ - 1)];
        for (uint32_t i = 0; i < n; i++) w[i] = cxq[(cxqt + 1 + i) & (CXQ - 1)];
        __atomic_store_n(&cxqt, cxqt + 1 + n, __ATOMIC_RELEASE);
        cx_learn(w, (int)n, &st, neu); }
    free(neu); return 0;
}
static int kf_words(const KF *f, uint32_t *w);
static void cx_fact(const KF *f) {                      /* the reader's side: a fact was stored (or met again) */
    uint32_t w[8]; int n = kf_words(f, w); if (n < 2) return;
    for (int i = 0; i < n; i++) while ((w[i] >> CXB) >= (uint32_t)cxnb) { if (cxnb >= CXMAXB) return; cx_block(); }
    while (cxqh + 1 + n - __atomic_load_n(&cxqt, __ATOMIC_ACQUIRE) > CXQ) usleep(50);
    uint64_t h = cxqh; cxq[h & (CXQ - 1)] = (uint32_t)n;
    for (int i = 0; i < n; i++) cxq[(h + 1 + i) & (CXQ - 1)] = w[i];
    __atomic_store_n(&cxqh, h + 1 + n, __ATOMIC_RELEASE);
}
static uint64_t cx_epst; static float *cx_epneu;
static void cx_fact_now(const KF *f) {                  /* the episode's side: learn from the fact now, in this thread */
    uint32_t w[8]; int n = kf_words(f, w); if (n < 2) return;
    for (int i = 0; i < n; i++) while ((w[i] >> CXB) >= (uint32_t)cxnb) { if (cxnb >= CXMAXB) return; cx_block(); }
    if (!cx_epneu) cx_epneu = malloc(CXD * 4);
    cx_learn(w, n, &cx_epst, cx_epneu);
}
static void cx_start(void) {
    cxq = malloc((size_t)CXQ * 4); if (!cxring) cxring = malloc((size_t)CXR * 4);
    if (getenv("CXETA")) CXETA = atof(getenv("CXETA"));
    if (getenv("CXFLOOR")) CXFLOOR = atof(getenv("CXFLOOR")); if (getenv("CXT")) CXT = atof(getenv("CXT"));
    pthread_create(&cxthr, 0, cx_thread, 0);
}
static void cx_finish(void) { __atomic_store_n(&cxdone, 1, __ATOMIC_RELEASE); pthread_join(cxthr, 0);
    fprintf(stderr, "cortex: %llu words met in facts, %llu pairs learned, %d blocks (%.1f GB)\n", (unsigned long long)cxtot, (unsigned long long)cxpairs, cxnb, cxnb * (double)CXBLK * CXD * 8 / 1e9); }
/* the cortex on disk, beside the brain (BRAIN.cx): words met, then meaning and context cells, word by word */
static int cx_save(const char *brain) {
    char p[4096], tmp[4200]; snprintf(p, sizeof p, "%s.cx", brain); snprintf(tmp, sizeof tmp, "%s.part", p);
    FILE *f = fopen(tmp, "wb"); if (!f) { perror(tmp); return 1; }
    uint32_t nw = LX.n < (uint32_t)cxnb * CXBLK ? LX.n : (uint32_t)cxnb * CXBLK, d = CXD;
    fwrite("MCX1", 1, 4, f); fwrite(&d, 4, 1, f); fwrite(&nw, 4, 1, f); fwrite(&cxtot, 8, 1, f); fwrite(&cxpairs, 8, 1, f);
    for (uint32_t g = 0; g < nw; g += CXBLK) { uint32_t k = nw - g < CXBLK ? nw - g : CXBLK; fwrite(&CXN(g), 4, k, f); }
    for (uint32_t g = 0; g < nw; g += CXBLK) { uint32_t k = nw - g < CXBLK ? nw - g : CXBLK; fwrite(CXU(g), 4, (size_t)k * CXD, f); }
    for (uint32_t g = 0; g < nw; g += CXBLK) { uint32_t k = nw - g < CXBLK ? nw - g : CXBLK; fwrite(CXC(g), 4, (size_t)k * CXD, f); }
    if (fflush(f) || fsync(fileno(f))) { perror(tmp); fclose(f); return 1; } fclose(f);
    if (rename(tmp, p)) { perror(p); return 1; }
    fprintf(stderr, "cortex saved: %s (%u words)\n", p, nw); return 0;
}
static int cx_load(const char *brain) {
    char p[4096]; snprintf(p, sizeof p, "%s.cx", brain); FILE *f = fopen(p, "rb"); if (!f) { perror(p); return 1; }
    char mg[4]; uint32_t d, nw; if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "MCX1", 4) || fread(&d, 4, 1, f) != 1 || d != CXD || fread(&nw, 4, 1, f) != 1) { fprintf(stderr, "%s: not a cortex\n", p); return 1; }
    if (fread(&cxtot, 8, 1, f) != 1 || fread(&cxpairs, 8, 1, f) != 1) return 1;
    while ((uint32_t)cxnb * CXBLK < nw) cx_block();
    for (uint32_t g = 0; g < nw; g += CXBLK) { uint32_t k = nw - g < CXBLK ? nw - g : CXBLK; if (fread(&CXN(g), 4, k, f) != k) return 1; }
    for (uint32_t g = 0; g < nw; g += CXBLK) { uint32_t k = nw - g < CXBLK ? nw - g : CXBLK; if (fread(CXU(g), 4, (size_t)k * CXD, f) != (size_t)k * CXD) return 1; }
    for (uint32_t g = 0; g < nw; g += CXBLK) { uint32_t k = nw - g < CXBLK ? nw - g : CXBLK; if (fread(CXC(g), 4, (size_t)k * CXD, f) != (size_t)k * CXD) return 1; }
    fclose(f);
    /* what it met lately, for the penalties when it reads on: words drawn by how often met (^0.75) */
    if (!cxring) cxring = malloc((size_t)CXR * 4);
    double z = 0; for (uint32_t g = 0; g < nw; g++) z += pow(CXN(g), 0.75);
    double c = 0; uint32_t g = 0; for (uint64_t i = 0; i < CXR && z > 0; i++) { while (g < nw - 1 && (c + pow(CXN(g), 0.75)) / z < (i + 0.5) / CXR) { c += pow(CXN(g), 0.75); g++; } cxring[i] = g; }
    cxnring = CXR; fprintf(stderr, "cortex %s: %u words, %llu met in facts\n", p, nw, (unsigned long long)cxtot); return 0;
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
    o->verb = f->verb == NONE ? NONE : gid_of(k, f->verb); o->flags = f->flags & (F_NEG | F_TENSE);   /* F2: the tense rides in the flags byte */
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
        k->R.question = 0; int nf = read_span(&k->R, t->tok, a, i + 1, fr, 512, 0); nsent_read++; CLK++;
        for (int j = 0; j < nf; j++) { KF f; if (to_kf(k, &fr[j], &f)) { if (!EPREAD) kf_add(&f, ptr); nfacts_read++; if (EPREAD) { ep_add(&f, ptr); cx_fact_now(&f); } else cx_fact(&f); } }
        a = i + 1;
    }
    t->ntok = 0; t->src_len = 0;
}

static int brain_save(const char *path, double secs, uint64_t bytes);
static int brain_load(const char *path);
static void build_cues(void);
static int run_read(const char *brain, double seconds, int nfile, char **files) {
    Know k; memset(&k, 0, sizeof k); text_init(&k.t); r2_init(&k.R, &k.t, &k.wm);
    if (getenv("ROOM")) ROOM = (uint32_t)atol(getenv("ROOM"));
    int part = getenv("PART") ? atoi(getenv("PART")) : 1;
    if (getenv("FROM")) { setenv("NOCUES", "1", 1); if (brain_load(getenv("FROM"))) return 1; unsetenv("NOCUES");
        capkf = nkf > (1u << 20) ? nkf : 1u << 20; kf = realloc(kf, (size_t)capkf * sizeof *kf); uint32_t sl = 1u << 21; while (sl < 2 * nkf + 2) sl *= 2; kf_rehash(sl);
        fprintf(stderr, "continuing %s: %u facts, %u words, clock %llu\n", getenv("FROM"), nkf, LX.n, (unsigned long long)CLK); }
    cx_start();
    double t0 = now(), next = t0 + 60; uint64_t bytes = 0; const size_t B = 1 << 22; char *buf = malloc(B + 1); int stop = 0;
    for (int fi = 0; fi < nfile && !stop; fi++) {
        { int done = 0; for (int s = 0; s < NSHELF; s++) done |= !strcmp(SHELF[s], files[fi]);       /* read already (continuing from a brain) */
          if (done) { fprintf(stderr, "  %s: read already, skipped\n", files[fi]); continue; } }
        if (NSHELF >= 64) { fprintf(stderr, "too many shelf files (64)\n"); return 1; }
        SHELF[NSHELF++] = files[fi];
        FILE *f = fopen(files[fi], "rb"); if (!f) { perror(files[fi]); return 1; }
        uint64_t off = 0; size_t have = 0; long blk = 0;
        for (;;) {
            size_t got = fread(buf + have, 1, B - have, f); size_t n = have + got; if (!n) break;
            size_t cut = n; if (got) { while (cut > 0 && buf[cut - 1] != '\n') cut--; if (!cut) cut = n; }
            if (blk++ % part == 0) { read_block(&k, buf, cut, NSHELF - 1, off); bytes += cut; }
            memmove(buf, buf + cut, n - cut); have = n - cut; off += cut;
            double tn = now();
            if (tn > next) { next = tn + 60; fprintf(stderr, "  %4.0f s  %s  %.0f MB read (%.1f MB/s)  %llu sentences  %u facts kept, %llu forgotten\n",
                tn - t0, files[fi], bytes / 1e6, bytes / 1e6 / (tn - t0), (unsigned long long)nsent_read, nkf, (unsigned long long)forgotten); }
            if (tn - t0 > seconds) { stop = 1; break; }
            if (!got && !have) break;
            if (!got) { if (have) { read_block(&k, buf, have, NSHELF - 1, off); bytes += have; } break; }
        }
        fclose(f);
    }
    double secs = now() - t0;
    fprintf(stderr, "read %.0f MB in %.0f s (%.2f MB/s): %llu sentences, %llu facts read; %u kept, %llu forgotten%s\n", bytes / 1e6, secs, bytes / 1e6 / secs,
        (unsigned long long)nsent_read, (unsigned long long)nfacts_read, nkf, (unsigned long long)forgotten, stop ? " (time up)" : " (shelf finished)");
    if (nmemforget) fprintf(stderr, "low memory made it forget %llu times\n", (unsigned long long)nmemforget);
    cx_finish(); if (cx_save(brain)) return 1;
    return brain_save(brain, secs, bytes);
}

/* ------------------------------------------------------------------ */
/* the brain on disk                                                      */
/* ------------------------------------------------------------------ */

static int vput64(uint8_t *b, uint64_t v) { int n = 0; while (v >= 128) { b[n++] = (uint8_t)(v | 128); v >>= 7; } b[n++] = (uint8_t)v; return n; }
static uint64_t vget64(FILE *f) { uint64_t v = 0; int sh = 0, c; while ((c = fgetc(f)) >= 0) { v |= (uint64_t)(c & 127) << sh; if (c < 128) break; sh += 7; } return v; }
static int brain_save(const char *path, double secs, uint64_t bytes) {
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s.part", path);   /* written whole, then renamed: a save cut off never spoils the last good one */
    FILE *f = fopen(tmp, "wb"); if (!f) { perror(tmp); return 1; }
    fwrite("MIMB4", 1, 5, f); fwrite(&CLK, 8, 1, f); fwrite(&NSHELF, 4, 1, f);
    for (int i = 0; i < NSHELF; i++) { uint16_t l = strlen(SHELF[i]); fwrite(&l, 2, 1, f); fwrite(SHELF[i], 1, l, f); }
    fwrite(&LX.n, 4, 1, f); for (uint32_t i = 0; i < LX.n; i++) { uint8_t l = strlen(LX.s[i]) > 255 ? 255 : strlen(LX.s[i]); fwrite(&l, 1, 1, f); fwrite(LX.s[i], 1, l, f); }
    fwrite(&nkf, 4, 1, f);                       /* facts as varints: ids + 1, a role byte, count, pointers */
    for (uint32_t i = 0; i < nkf; i++) { const KF *e = &kf[i]; uint8_t b[160]; int n = 0;
        n += vput(b + n, e->verb + 1); b[n++] = e->narg | (e->flags & F_NEG) << 4;
        for (int j = 0; j < e->narg; j++) { b[n++] = e->role[j]; n += vput(b + n, e->prep[j] + 1); n += vput(b + n, e->head[j] + 1); n += vput(b + n, e->mod[j] + 1); }
        n += vput(b + n, e->count); n += vput64(b + n, e->ptr[0] + 1); n += vput64(b + n, e->count > 1 ? e->ptr[1] + 1 : 0);
        memcpy(b + n, &e->s, 4); n += 4; n += vput(b + n, e->L); b[n++] = 0; fwrite(b, 1, n, f); }   /* 0: the old "learned for good" byte, kept so the file layout stays */
    long sz = ftell(f); if (fflush(f) || fsync(fileno(f))) { perror(tmp); fclose(f); return 1; } fclose(f);
    if (rename(tmp, path)) { perror(path); return 1; }
    printf("RESULT bgread brain=%s bytes=%ld facts=%u words=%u forgotten=%llu read_bytes=%llu seconds=%.0f mb_s=%.2f\n", path, sz, nkf, LX.n,
        (unsigned long long)forgotten, (unsigned long long)bytes, secs, bytes / 1e6 / secs);
    return 0;
}
static char *SHELFP[64];
static uint32_t *cue_off, *cue;              /* word → facts that hold it, strongest first */
static uint32_t ncuew;                       /* words when the cue lists were built (words met since have none) */
static float *EF;
static int cmp_strong(const void *a, const void *b) { float x = EF[*(const uint32_t *)a], y = EF[*(const uint32_t *)b]; return x < y ? 1 : x > y ? -1 : 0; }
static int kf_words(const KF *f, uint32_t *w) { int n = 0; if (f->verb != NONE) w[n++] = f->verb; for (int i = 0; i < f->narg; i++) { w[n++] = f->head[i]; if (f->mod[i] != NONE) w[n++] = f->mod[i]; } return n; }
static void build_cues(void) {
    free(cue_off); free(cue); free(EF); EF = malloc(((size_t)nkf + 1) * sizeof *EF); for (uint32_t i = 0; i < nkf; i++) EF[i] = (float)eff(&kf[i]);
    cue_off = calloc(LX.n + 2, 4); ncuew = LX.n;
    for (uint32_t i = 0; i < nkf; i++) { uint32_t w[8]; int n = kf_words(&kf[i], w); for (int j = 0; j < n; j++) { int dup = 0; for (int x = 0; x < j; x++) dup |= w[x] == w[j]; if (!dup && w[j] < LX.n) cue_off[w[j] + 2]++; } }
    for (uint32_t g = 0; g < LX.n; g++) cue_off[g + 2] += cue_off[g + 1];
    cue = malloc(((size_t)cue_off[LX.n + 1] + 1) * 4);
    for (uint32_t i = 0; i < nkf; i++) { uint32_t w[8]; int n = kf_words(&kf[i], w); for (int j = 0; j < n; j++) { int dup = 0; for (int x = 0; x < j; x++) dup |= w[x] == w[j]; if (!dup && w[j] < LX.n) cue[cue_off[w[j] + 1]++] = i; } }
    for (uint32_t g = 0; g < LX.n; g++) qsort(cue + cue_off[g], cue_off[g + 1] - cue_off[g], 4, cmp_strong);
}
static uint32_t df_of(uint32_t g) { return g < ncuew && cue_off ? cue_off[g + 1] - cue_off[g] : 0; }
static double gidf(uint32_t g) { return log((nkf + 1.0) / (df_of(g) + 0.5)); }
static int brain_load(const char *path) {
    if (!strcmp(path, "-")) { build_cues(); return 0; }                  /* the empty brain: nothing read yet */
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return 1; }
    char mg[5]; if (fread(mg, 1, 5, f) != 5 || memcmp(mg, "MIMB4", 5)) { fprintf(stderr, "%s: not a brain\n", path); return 1; }
    if (fread(&CLK, 8, 1, f) != 1) return 1;
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
        if (fread(&e->s, 4, 1, f) != 1) return 1; e->L = (uint32_t)vget64(f); if (fgetc(f) < 0) return 1; }
    fclose(f);
    if (cx_load(path)) { if (!getenv("NOCUES")) return 1; fprintf(stderr, "no cortex beside %s: a new one starts\n", path); }
    if (!getenv("NOCUES")) build_cues(); return 0;
}

/* ------------------------------------------------------------------ */
/* answering from what it knows                                           */
/* ------------------------------------------------------------------ */

/* words of a statement read by the answering context, as the brain knows them */
static int qwords(Know *A, const uint32_t *w, int n, QW *out) {
    for (int i = 0; i < n; i++) { const char *s = A->t.voc[w[i]].word; out[i].g = lx_id(s, 0); out[i].mid = wmid(s); } return n;
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
/* docs/COMPREHEND.md: a fact or sentence that says the opposite ("not") of the option counts against it, weighted NW
   (chosen on dev from 0.5, 1, 2) */
static const double NW = 2;
static double TENSEW = 0.5;      /* docs/FOLLOW.md F2: a fact in the other tense counts this much of itself (dev) */
static double FX = 0.85, FO = 0.25;   /* F5: an option matched by an episode fact at least FX of the most one can give, alone: the open book counts FO (dev) */
static int direct_neg;           /* F1: this question's "not" is matched directly, not turned around */
/* a statement's fact (answering context ids) against a known fact, 0..1 (as know.c fact_sim); below 0 when it matches
   but says the opposite */
static int yn_mode;   /* docs/YESNO.md: the option being scored is a bare yes or no */
static float kfact_sim(Know *A, const Frame *h, const KF *f) {
    double num = 0, den = 0, vs = 1;
    if (h->verb != NONE) { int light = h->verb == A->R.w_be || h->verb == A->R.w_have || h->verb == A->R.w_do; double w = light ? 0.3 : 1.0;
        QW v = { lx_id(A->t.voc[h->verb].word, 0), wmid(A->t.voc[h->verb].word) }; vs = f->verb != NONE ? qsim(&v, f->verb) : 0; num += w * vs; den += w; }
    double subj = -1; int others = 0, matched = 0;   /* Y5b: the statement's subject and how many of its other things the fact holds */
    for (int i = 0; i < h->narg; i++) { const Arg *a = &h->a[i]; if (a->head >= OPEN || !content(A, a->head)) continue;
        QW x = { lx_id(A->t.voc[a->head].word, 0), wmid(A->t.voc[a->head].word) }; double best = 0;
        for (int j = 0; j < f->narg; j++) { Arg b = { f->role[j], 0, f->prep[j], 0, 0 }, aa = *a; aa.prep = a->prep < OPEN ? lx_id(A->t.voc[a->prep].word, 0) : NONE;
            double s = role_fit(&aa, &b) * fmax(qsim(&x, f->head[j]), f->mod[j] != NONE ? 0.6 * qsim(&x, f->mod[j]) : 0); if (s > best) best = s; }
        num += best; den += 1;
        if (subj < 0 && (a->role == A_AGENT || i == 0)) subj = best; else { others++; matched += best >= 0.5; } }
    if (den <= 0) return 0;
    double s = num / den; if ((h->flags & F_NEG) != (f->flags & F_NEG) && s > (s5(4) ? 0.35 : 0.5)) s = vs >= 0.5 ? -s : (yn_mode && s5(16)) ? -0.5 * s : s * 0.3;   /* it says the opposite (of the same doing); Y3: from 0.35; Y5: for a yes/no judgement a "not" about the same things counts against even with another verb */
    else if (yn_mode && s5(512) && (h->flags & F_NEG) != (f->flags & F_NEG) && vs < 0.5 && subj >= 0.8 && (others == 0 || matched > 0) && s > 0.3) s = -0.5 * s;   /* Y5b: "the certificate is not considered proof of identity" against "the certificate is a valid form of id": the same subject, one more thing in common, said with "not" */
    if ((h->flags & F_TENSE) && (f->flags & F_TENSE) && (h->flags & F_TENSE) != (f->flags & F_TENSE)) s *= TENSEW;   /* F2: said of another time */
    return (float)s;
}
static double conf(const KF *f) {
    if (is_ep(f)) return 1.0 - 0.5 / (f->s > 1 ? f->s : 1);
    double e = EF ? EF[f - kf] : eff(f); return 1.0 - 0.5 / (e > 1 ? e : 1); }

/* re-read one sentence from the shelf through a pointer (its facts kept in rr_fr) */
static Frame rr_fr[64]; static int rr_nf;
static int reread(Know *A, uint64_t ptr, QW *out, int max) {
    if (ptr == ~0ull) return 0;
    int fid = (int)(ptr >> FID_SHIFT); uint64_t off = ptr & ((1ull << FID_SHIFT) - 1); char s[1024]; size_t n;
    if (fid == EPFID) { if (off >= neb) return 0; n = neb - off < sizeof s - 1 ? neb - off : sizeof s - 1; memcpy(s, epbuf + off, n); s[n] = 0; }
    else { if (fid >= NSHELF) return 0;
        static FILE *fp[64]; if (!fp[fid]) fp[fid] = fopen(SHELF[fid], "rb"); if (!fp[fid]) return 0;
        fseeko(fp[fid], (off_t)off, SEEK_SET); n = fread(s, 1, sizeof s - 1, fp[fid]); s[n] = 0; }
    char *e = strchr(s, '\n'); if (e) *e = 0;
    for (char *p = s; *p; p++) if ((*p == '.' || *p == '!' || *p == '?') && (p[1] == ' ' || !p[1])) {
        if (*p == '.') { char *b = p; while (b > s && isalpha((unsigned char)b[-1])) b--; char w[8]; size_t L = (size_t)(p - b); if (L < sizeof w) { memcpy(w, b, L); w[L] = 0; if (abbrev(w)) continue; } }
        p[1] = 0; break; }
    uint32_t w[256]; int nw; rr_nf = read_statement(A, s, rr_fr, 64, w, &nw, 256); if (nw > max) nw = max;
    return qwords(A, w, nw, out);
}

static float *mark; static uint32_t nmark;
static double BW[5] = { 1, 0.5, 2, 4, 1 };   /* fact, joint, open, mean (chosen on ARC train + dev with a dev brain, 5% of the shelf); fit (chosen on dev, CORTEX_LINK) */
static int NOPEN = 20;  /* facts whose shelf sentences are re-read when answering (docs/past/LAYA_LEVEL.md stage 1, chosen on dev) */
/* docs/PASSAGE.md: inside an active episode the meaning and fit parts (association, which never look at the passage) are
   scaled by EA: 0, chosen on dev, so the passage decides */
static double EA = 0;
/* candidate facts: shelf facts by index, episode facts with the top bit set */
static const KF *kf_at(uint32_t i) { return i & 0x80000000u ? &ep[i & 0x7fffffffu] : &kf[i]; }
/* R1: does the sentence just re-read (rr_fr) say the opposite of the option's statement? Each statement fact against its
   best match among the sentence's facts that hold one of the option's own words: the sum of the signed matches below 0 */
static int opposes(Know *A, Frame *hf, int nh, int focus, const QW *O, int no) {
    double t = 0;
    for (int i = 0; i < nh; i++) { if (focus && !hf[i].tc) continue; double b = 0;
        for (int j = 0; j < rr_nf; j++) { const Frame *g = &rr_fr[j]; KF f; memset(&f, 0, sizeof f);
            f.verb = g->verb < OPEN ? lx_id(A->t.voc[g->verb].word, 0) : NONE; f.flags = g->flags & (F_NEG | F_TENSE);   /* F2: "was afraid" against "is no longer afraid" */
            for (int a = 0; a < g->narg && f.narg < KA; a++) { const Arg *x = &g->a[a]; if (x->head >= OPEN) continue; int n = f.narg++;
                f.role[n] = x->role; f.prep[n] = x->prep < OPEN ? lx_id(A->t.voc[x->prep].word, 0) : NONE; f.head[n] = lx_id(A->t.voc[x->head].word, 0); f.mod[n] = x->mod < OPEN ? lx_id(A->t.voc[x->mod].word, 0) : NONE; }
            int has = 0; uint32_t w[8]; int n = kf_words(&f, w); for (int x = 0; x < n; x++) for (int y = 0; y < no; y++) has |= w[x] != NONE && w[x] == O[y].g;
            if (!has) continue;
            double s = kfact_sim(A, &hf[i], &f); if (fabs(s) > fabs(b) || (fabs(s) == fabs(b) && s != b)) b = s; }   /* F2: equally close, the later clause */
        t += b; }
    return t < 0;
}
/* (a) fact agreement, (b) one known fact covering the question and the option, (c) open book, over some candidates */
static void score_cands(Know *A, Frame *hf, int nh, int focus, const QW *Q, int nq, const QW *O, int no, const uint32_t *cand, int nc,
                        double *fact, double *joint, double *open, uint64_t *optr) {
    *fact = *joint = *open = 0; *optr = ~0ull;
    double fs = 0; int nfs = 0; int stq = 0, stk[64];        /* S1: which statement frames state a place, a holder, a making-of (only those meet superseded facts differently) */
    for (int i = 0; i < nh && i < 64; i++) { stk[i] = 0; if (nep) { KF hk; frame_kf(A, &hf[i], &hk); stk[i] = state_stmt(&hk); } stq |= stk[i] && !(hf[i].flags & F_PAST); }
    for (int i = 0; i < nh; i++) { if (focus && !hf[i].tc) continue; double b = 0, bo = 0;
        for (int c = 0; c < nc; c++) { const KF *f = kf_at(cand[c]); if (is_ep(f) && ep_der[f - ep] && !(i < 64 && stk[i])) continue;   /* S2/S3: derived states answer state questions only */
            double s = kfact_sim(A, &hf[i], f) * conf(f);
            if (is_ep(f) && ep_sup[f - ep] && i < 64 && stk[i] && !(hf[i].flags & F_PAST)) {     /* S1: a superseded state, asked about the state now */
                if (hf[i].flags & F_PRES) { KF g = *f; g.flags ^= F_NEG; s = kfact_sim(A, &hf[i], &g) * conf(f); }   /* S1b: "is": it is no longer so, the fact says the opposite now */
                else s *= SUPW; }                                                         /* no tense said: it counts less */
            if (fabs(s) > fabs(b) || (is_ep(f) && fabs(s) == fabs(b) && s != b)) b = s;    /* F2: of two facts equally close, the one read later (the newer state) */
            if (fabs(s) > fabs(bo) || (is_ep(f) && fabs(s) == fabs(bo) && s != bo)) { uint32_t w[8]; int n = kf_words(f, w), has = 0; for (int x = 0; x < n; x++) for (int y = 0; y < no; y++) has |= w[x] == O[y].g && w[x] != NONE; if (has) bo = s; } }
        if (bo != 0 && !(b < 0 && fabs(b) > fabs(bo) && nc && is_ep(kf_at(cand[0])))) b = bo;       /* the closest fact decides, for or against; among those holding the option's own words first (F1: unless the closest says the opposite) */
        fs += b < 0 ? NW * b : b; nfs++; }
    *fact = nfs ? fs / nfs : 0;
    enum { TOPMAX = 64 }; int64_t top[TOPMAX]; double tv[TOPMAX]; int K = NOPEN < 3 ? 3 : NOPEN > TOPMAX ? TOPMAX : NOPEN; for (int r = 0; r < K; r++) { top[r] = -1; tv[r] = -1; }
    for (int c = 0; c < nc; c++) { uint32_t w[8]; int n = kf_words(kf_at(cand[c]), w); double j = conf(kf_at(cand[c])) * (kcover(Q, nq, w, n) + kcover(O, no, w, n));
        if (is_ep(kf_at(cand[c])) && ep_sup[cand[c] & 0x7fffffffu] && stq) j *= SUPW;   /* S1 */
        if (is_ep(kf_at(cand[c])) && ep_der[cand[c] & 0x7fffffffu]) continue;             /* S2/S3: a derived fact is not text: it has no sentence to re-read */
        if (j > *joint) *joint = j;
        for (int r = 0; r < K; r++) if (j > tv[r]) { for (int x = K - 1; x > r; x--) { tv[x] = tv[x - 1]; top[x] = top[x - 1]; } tv[r] = j; top[r] = cand[c]; break; } }
    /* (c) open book: those facts' pointers lead back to the shelf (or the episode's text); the sentences re-read */
    uint64_t seen[2 * TOPMAX]; int nseen = 0, bopp = 0;
    for (int r = 0; r < K; r++) { if (top[r] < 0) continue;
        for (int p = 0; p < 2; p++) { uint64_t pt = kf_at((uint32_t)top[r])->ptr[p]; int dup = 0; for (int x = 0; x < nseen; x++) dup |= seen[x] == pt; if (dup || pt == ~0ull) continue; seen[nseen++] = pt;
            QW S[256]; int ns = reread(A, pt, S, 256); if (!ns) continue;
            double j = qcover(Q, nq, S, ns) + qcover(O, no, S, ns);
            if (j > *open) { *open = j; *optr = pt; bopp = opposes(A, hf, nh, focus, O, no); } } }
    if (bopp) *open *= -NW;                  /* the sentence that covers the question and option best says the opposite */
}
static int asks_opposite(const char *q);
static int yesno_q(const char *q) { static const char *AUX[] = { "do", "does", "did", "is", "are", "was", "were", "can", "could", "will", "would", "has", "have", "had", 0 };
    for (int i = 0; AUX[i]; i++) if (is_word_at(q, AUX[i])) return 1; return 0; }
static int rel_verdict(Know *A, Frame *hf, int nh); static int ep_mod_exists(uint32_t x);
static int yn_slot;   /* docs/YESNO.md Y12: the episode's current state of the statement's slot says +1 the same value, -1 another value, 0 nothing */
static int state_verdict(Know *A, Frame *hf, int nh);
static int yn_bare(const char *o) { return (is_word_at(o, "yes") && !o[3]) || (is_word_at(o, "no") && !o[2]); }
static int wh_q(const char *q) { static const char *W[] = { "what", "which", "who", "whom", "where", "when", "why", "how", 0 }; for (int i = 0; W[i]; i++) if (is_word_at(q, W[i])) return 1; return 0; }   /* "What did the mayor say?" + yes / no: the word said, not a judgement */
static void bg_option(Know *A, const char *q, const char *opt, BParts *P) {
    yn_mode = yn_bare(opt) && (yesno_q(q) || (s5(32) && !wh_q(q)));   /* Y6: a bare yes/no answers a yes/no question whatever its first word */
    char st[4096]; statement(q, opt, st, sizeof st);
    static Frame hf[64], qf[64], of[64]; uint32_t sw[256], qwl[256], owl[128]; int nsw, nq, now_;
    int nh = read_statement(A, st, hf, 64, sw, &nsw, 256); read_statement(A, q, qf, 64, qwl, &nq, 256); read_statement(A, opt, of, 64, owl, &now_, 128);
    if (yn_mode && s5(4096) && is_word_at(opt, "yes") && nep) yn_slot = state_verdict(A, hf, nh);   /* Y12 */
    if (yn_mode && s5(65536) && is_word_at(opt, "yes") && nep && !yn_slot) yn_slot = rel_verdict(A, hf, nh);   /* Y16 */
    if (getenv("STDBG")) { fprintf(stderr, "  STMT[%s]: %s\n", opt, st); for (int i = 0; i < nh; i++) { fprintf(stderr, "    %s%s%s", hf[i].flags & F_NEG ? "NOT " : "", hf[i].flags & F_PAST ? "[past] " : hf[i].flags & F_PRES ? "[now] " : "", hf[i].verb < OPEN ? A->t.voc[hf[i].verb].word : "-");
        for (int a_ = 0; a_ < hf[i].narg; a_++) fprintf(stderr, " %s%s%s=%s", hf[i].a[a_].role == A_AGENT ? "who" : hf[i].a[a_].role == A_PAT ? "what" : hf[i].a[a_].role == A_ATTR ? "is" : "pp", hf[i].a[a_].prep < OPEN ? ":" : "", hf[i].a[a_].prep < OPEN ? A->t.voc[hf[i].a[a_].prep].word : "", hf[i].a[a_].head < OPEN ? A->t.voc[hf[i].a[a_].head].word : "?"); fprintf(stderr, "\n"); } }   /* docs/YESNO.md dev trace */
    QW Q[256], O[128]; qwords(A, qwl, nq, Q);
    int no = 0; for (int i = 0; i < now_; i++) { int inq = 0; for (int j = 0; j < nq; j++) inq |= qwl[j] == owl[i]; if (!inq) { const char *s = A->t.voc[owl[i]].word; O[no].g = lx_id(s, 0); O[no++].mid = wmid(s); } }
    memset(P, 0, sizeof *P); P->optr = ~0ull;
    /* a question that asks for what is NOT so ("What did Marta not take?") is read without its "not" and turned around
       (scores): the option the facts say least, or say the opposite of ("without a candle"), wins */
    direct_neg = 0;
    if (nep && asks_opposite(q)) {       /* F1: "Who did not laugh?": the episode has a NOT fact for that verb: matched as said, not turned around */
        for (int i = 0; i < nh && !direct_neg; i++) { if (!(hf[i].flags & F_NEG) || hf[i].verb >= OPEN) continue; uint32_t g = lx_id(A->t.voc[hf[i].verb].word, 0);
            for (uint32_t e = 0; e < nep && !direct_neg; e++) direct_neg = (ep[e].flags & F_NEG) && ep[e].verb == g; } }
    if (asks_opposite(q) && !direct_neg) for (int i = 0; i < nh; i++) hf[i].flags &= (uint8_t)~F_NEG;   /* (not "why ... not": a reason is asked) */
    if (s5(8) && no && yn_mode) no = 0;   /* docs/YESNO.md Y4: a bare "yes" is no content word of its own ("no" is glue and never was): both answers are about the question's words */
    if (!no && (is_word_at(opt, "no") || is_word_at(opt, "yes") || strstr(opt, "not") || strstr(opt, "n't"))) {   /* "no, she was not sad": by its yes or no */
        for (int i = 0; i < nq && i < 128; i++) O[no++] = Q[i]; }
    else if (nep && yesno_q(q)) for (int i = 0; i < nq && no < 128; i++) { int dup = 0; for (int j = 0; j < no; j++) dup |= O[j].g == Q[i].g; if (!dup) O[no++] = Q[i]; }   /* F1: a yes/no question is about its own words, not the option's extras ("the man") */
    /* (d) how close in meaning the option's own words are to the question's */
    P->mean = no && nq ? qcover(O, no, Q, nq) : 0;
    /* (e) fit: how well the option fits the question as the cortex expects (the option's meaning cells against the
       question words' context cells; K3's fill-the-blank score, docs/past/CORTEX.md) */
    { double s = 0; int k = 0; for (int i = 0; i < no; i++) { if (O[i].mid < 0) continue;
        for (int j = 0; j < nq; j++) if (Q[j].mid >= 0 && Q[j].g != O[i].g) { s += cx_dot(CXU((uint32_t)O[i].mid), CXC((uint32_t)Q[j].mid)); k++; } }
      P->fit = k ? s / k : 0; }
    if (!no || (!nkf && !nep)) return;
    /* candidate facts: those holding one of the option's own words, ranked first by the question's words they also hold (exactly) */
    if (nmark < LX.n) { mark = realloc(mark, LX.n * sizeof *mark); memset(mark + nmark, 0, (LX.n - nmark) * sizeof *mark); nmark = LX.n; }
    for (int i = 0; i < nq; i++) if (Q[i].g != NONE) mark[Q[i].g] = (float)gidf(Q[i].g);
    enum { NC = 200 }; static uint32_t cand[NC]; static float cs[NC]; int nc = 0;   /* docs/past/RANKING_COMPARE.md: more candidates were worse */
#define CAND_ADD(fi) do { uint32_t w_[8]; int n_ = kf_words(kf_at(fi), w_); float s_ = 0; for (int j_ = 0; j_ < n_; j_++) s_ += mark[w_[j_]]; \
        s_ = (float)(s_ * conf(kf_at(fi))) + 1e-3f * (float)conf(kf_at(fi)); int m_ = nc; \
        if (nc == NC) { m_ = 0; for (int j_ = 1; j_ < NC; j_++) if (cs[j_] < cs[m_]) m_ = j_; if (!(s_ > cs[m_])) break; } \
        cand[m_] = (fi); cs[m_] = s_; if (m_ == nc) nc++; } while (0)
    for (int i = 0; i < no; i++) { if (O[i].g == NONE || O[i].g >= ncuew) continue;
        uint32_t a = cue_off[O[i].g], b = cue_off[O[i].g + 1]; if (b - a > 200000) b = a + 200000;
        for (uint32_t x = a; x < b; x++) CAND_ADD(cue[x]); }
#undef CAND_ADD
    for (int i = 0; i < nq; i++) if (Q[i].g != NONE) mark[Q[i].g] = 0;
    int focus = 0; for (int i = 0; i < nh; i++) { uint32_t w[24]; int m; fact_words(&hf[i], w, &m); int has = 0; for (int j = 0; j < m; j++) for (int x = 0; x < now_; x++) has |= w[j] == owl[x]; hf[i].tc = has; focus += has; }
    if (yesno_q(q)) focus = 0;     /* docs/READ2.md: "Does Mr. Sola bake on Sundays?": every option's statement is judged whole, "only cakes" too */
    score_cands(A, hf, nh, focus, Q, nq, O, no, cand, nc, &P->fact, &P->joint, &P->open, &P->optr);
    /* the episode: when the question or the option speaks of it, its facts answer first, weighted EPW */
    if (nep) {
        int active = 0; for (int i = 0; i < nq; i++) active |= Q[i].g != NONE && Q[i].g < nepword && epword[Q[i].g];
        for (int i = 0; i < no; i++) active |= O[i].g != NONE && O[i].g < nepword && epword[O[i].g];
        if (active) { static uint32_t *ec; static uint32_t cap; if (cap < nep) { cap = nep; ec = realloc(ec, cap * 4); } for (uint32_t e = 0; e < nep; e++) ec[e] = e | 0x80000000u;
            score_cands(A, hf, nh, focus, Q, nq, O, no, ec, (int)nep, &P->efact, &P->ejoint, &P->eopen, &P->eptr);
            if (EPW * P->efact > P->fact || P->efact < 0) P->fact = EPW * P->efact;     /* the passage saying the opposite decides */
            if (EPW * P->ejoint > P->joint) P->joint = EPW * P->ejoint;
            if (EPW * P->eopen > P->open || P->eopen < 0) { P->open = EPW * P->eopen; P->optr = P->eptr; }
            P->active = 1; P->mean *= EA; P->fit *= EA; } }
}

static double bscore(const BParts *p) { return BW[0] * p->fact + BW[1] * p->joint + BW[2] * p->open + BW[3] * p->mean + BW[4] * p->fit; }
/* the question asks for the option that is NOT so ("not", "least", "except", ...): support is turned around (docs/past/LAYA_LEVEL.md stage 2) */
static int asks_opposite0(const char *q);
static int asks_opposite(const char *q) { return asks_opposite0(q) && !is_word_at(q, "why") && !is_word_at(q, "how"); }   /* "why ... not" asks a reason */
static int asks_opposite0(const char *q) {
    static const char *W[] = { "not", "least", "except", "false", "unlikely", "cannot", "never", "incorrect", "untrue", "isn't", "doesn't", "can't", "won't", "wouldn't", "aren't", 0 };
    char w[32]; int n = 0;
    for (const char *s = q;; s++) { if (isalpha((unsigned char)*s) || *s == '\'') { if (n < 31) w[n++] = (char)tolower((unsigned char)*s); continue; }
        if (n) { w[n] = 0; for (int i = 0; W[i]; i++) if (!strcmp(w, W[i])) return 1; n = 0; } if (!*s) break; }
    return 0;
}
static void scores(const char *q, const BParts *P, int no, double *sc) { int neg = asks_opposite(q) && !direct_neg; for (int o = 0; o < no; o++) sc[o] = neg ? -bscore(&P[o]) : bscore(&P[o]); }
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
            for (int o = 0; o < no; o++) { bg_option(&A, fld[1], fld[3 + o], &P[o]); A.t.ntok = 0; }
            scores(fld[1], P, no, sc);
            /* ties share the credit */
            double best = -1e9; int nb = 0; for (int o = 0; o < no; o++) if (sc[o] > best + 1e-9) { best = sc[o]; nb = 1; } else if (sc[o] > best - 1e-9) nb++;
            if (sc[want] > best - 1e-9) ok += 1.0 / nb;
            for (int c = 0; c < 4; c++) { double b = -1e9; int k = 0; for (int o = 0; o < no; o++) { double v = c == 0 ? P[o].fact : c == 1 ? P[o].joint : c == 2 ? P[o].open : P[o].mean; if (v > b + 1e-9) { b = v; k = 1; } else if (v > b - 1e-9) k++; }
                double v = c == 0 ? P[want].fact : c == 1 ? P[want].joint : c == 2 ? P[want].open : P[want].mean; if (v > b - 1e-9) okp[c] += 1.0 / k; }
            if (dump) { const char *bn = strrchr(files[fi], '/'); fprintf(dump, "%s\t%d\t%d\t%d", bn ? bn + 1 : files[fi], want, no, asks_opposite(fld[1])); for (int o = 0; o < no; o++) fprintf(dump, "\t%.5f %.5f %.5f %.5f %.5f", P[o].fact, P[o].joint, P[o].open, P[o].mean, P[o].fit); fprintf(dump, "\n"); }
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
        for (int o = 0; o < no; o++) { bg_option(&A, fld[0], fld[1 + o], &P[o]); A.t.ntok = 0; }
        scores(fld[0], P, no, sc); for (int o = 1; o < no; o++) if (sc[o] > sc[best] + 1e-9) best = o;
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

/* passages (docs/EPISODE.md): MCTest layout (S id / T text / Q type question 4 options letter). Each passage is read as
   the episode (the cortex learning from it), its questions answered, then everything it changed is put back. */
static uint32_t ep_lxn; static uint64_t ep_clk;
static void ep_begin(void) { cxj = 1; cxj_tot = cxtot; cxj_nring = cxnring; ep_lxn = LX.n; ep_clk = CLK; }
static void ep_end(void) {                                /* the passage is over: drop it, put the cortex back */
    for (uint32_t i = ncxjr; i-- > 0;) { memcpy(cxjr[i].which ? CXC(cxjr[i].g) : CXU(cxjr[i].g), cxjr[i].row, CXD * 4); free(cxjr[i].row); }
    for (uint32_t i = 0; i < ncxjr; i++) if (!cxjr[i].which) { uint32_t g = cxjr[i].g; float n = cx_dot(CXU(g), CXU(g)); cxinv[g] = n > 0 ? 1 / sqrtf(n) : 0; }
    for (uint32_t i = ncxjn; i-- > 0;) CXN(cxjn_g[i]) = cxjn_v[i];
    for (uint32_t i = ncxjring; i-- > 0;) cxring[cxjring_i[i]] = cxjring_v[i];
    cxtot = cxj_tot; cxnring = cxj_nring; ncxjr = ncxjn = ncxjring = 0; cxj = 0;
    for (uint32_t e = 0; e < nep; e++) { uint32_t w[8]; int n = kf_words(&ep[e], w); for (int j = 0; j < n; j++) if (w[j] < nepword) epword[w[j]] = 0; }
    nep = 0; neb = 0; CLK = ep_clk; ep_seq = 0;
    for (uint32_t g = LX.n; g-- > ep_lxn;) {            /* words first met in the passage are let go (newest first, so the table is as before) */
        uint32_t h = shash(LX.s[g]) & LX.mask; while (LX.slot[h] != g + 1) h = (h + 1) & LX.mask;
        LX.slot[h] = 0; free(LX.s[g]); }
    LX.n = ep_lxn;
    for (uint32_t i = 0; i < nl2g; i++) if (l2g[i] != NONE - 1 && l2g[i] >= ep_lxn) l2g[i] = NONE - 1;   /* the reader's word map, the same */
}
/* docs/YESNO.md Y15 (for docs/STATE.md): when the sentences carry time words ("Yesterday Julie went to the office. This morning she
   went to the school."), the states follow the times, not the telling: each fact's order becomes its sentence's time rank first
   (yesterday 1, last night 2, this morning 3, today 4, this afternoon 5, this evening 6, tonight 7; none: 4), its saying second. */
static int ep_time_rank(const char *s, size_t n) {
    static const struct { const char *w; int r; } T[] = { { "yesterday", 1 }, { "last night", 2 }, { "this morning", 3 }, { "today", 4 }, { "this afternoon", 5 }, { "this evening", 6 }, { "tonight", 7 }, { 0, 0 } };
    for (int i = 0; T[i].w; i++) { size_t L = strlen(T[i].w); for (size_t j = 0; j + L <= n; j++) if (!strncasecmp(s + j, T[i].w, L) && (j == 0 || !isalpha((unsigned char)s[j - 1])) && (j + L == n || !isalpha((unsigned char)s[j + L]))) return T[i].r; }
    return 0;
}
static void ep_time_order(void) {
    int any = 0; static int *rank; static uint32_t cap; if (cap < nep) { cap = nep; rank = realloc(rank, cap * sizeof *rank); }
    for (uint32_t e = 0; e < nep; e++) { rank[e] = 0; uint64_t p = ep[e].ptr[0]; if (p == ~0ull || (int)(p >> FID_SHIFT) != EPFID) continue;
        size_t off = (size_t)(p & ((1ull << FID_SHIFT) - 1)); if (off >= neb) continue;
        size_t a = off; while (a > 0 && epbuf[a - 1] != '.' && epbuf[a - 1] != '!' && epbuf[a - 1] != '?' && epbuf[a - 1] != '\n') a--;
        size_t b = off; while (b < neb && epbuf[b] != '.' && epbuf[b] != '!' && epbuf[b] != '?' && epbuf[b] != '\n') b++;
        rank[e] = ep_time_rank(epbuf + a, b - a); any |= rank[e] != 0; }
    if (!any) return;
    for (uint32_t e = 0; e < nep; e++) ep_ord[e] = (uint32_t)(rank[e] ? rank[e] : 4) * 100000u + ep_ord[e];
    ep_seq += 800000u;                                   /* later facts stay later than every timed one */
}
static void ep_read(Know *k, const char *text) {
    size_t L = strlen(text); if (neb + L + 2 > capeb) { capeb = (neb + L + 2) * 2; epbuf = realloc(epbuf, capeb); }
    memcpy(epbuf + neb, text, L); epbuf[neb + L] = '\n'; size_t off = neb; neb += L + 1; epbuf[neb] = 0;
    memset(&k->R.refs, 0, sizeof k->R.refs); k->R.refs.plthing = NONE; EPREAD = 1; read_block(k, epbuf + off, L + 1, EPFID, off); EPREAD = 0;
    if (s5(32768)) ep_time_order();                      /* Y15: "yesterday … this morning …": the order of saying is the order of the times said */
    ep_states();
    if (getenv("EPDBG")) for (uint32_t e = 0; e < nep; e++) { fprintf(stderr, "  ep%u ord=%u%s%s %s%s", e, ep_ord[e], ep_sup[e] ? " SUP" : "", ep_der[e] ? " der" : "", ep[e].flags & F_NEG ? "NOT " : "", ep[e].verb != NONE ? LX.s[ep[e].verb] : "-");
        for (int a = 0; a < ep[e].narg; a++) { fprintf(stderr, " %s%s%s=%s", ep[e].role[a] == A_AGENT ? "who" : ep[e].role[a] == A_PAT ? "what" : ep[e].role[a] == A_ATTR ? "is" : "pp", ep[e].prep[a] != NONE ? ":" : "", ep[e].prep[a] != NONE ? LX.s[ep[e].prep[a]] : "", LX.s[ep[e].head[a]]); if (ep[e].mod[a] != NONE) fprintf(stderr, "(%s)", LX.s[ep[e].mod[a]]); } fprintf(stderr, "\n"); }   /* dev trace */
    for (uint32_t i = 0; i < ncxjr; i++) if (!cxjr[i].which) { uint32_t g = cxjr[i].g; float n = cx_dot(CXU(g), CXU(g)); cxinv[g] = n > 0 ? 1 / sqrtf(n) : 0; }
    if (nepword < LX.n) { epword = realloc(epword, LX.n); memset(epword + nepword, 0, LX.n - nepword); nepword = LX.n; }
    for (uint32_t e = 0; e < nep; e++) { uint32_t w[8]; int n = kf_words(&ep[e], w); for (int j = 0; j < n; j++) if (w[j] < nepword) epword[w[j]] = 1; }
}
#define MAXO 24                                        /* options a passage question may have (docs/SURVEY.md: CBT 10, ReCoRD up to 24) */
/* ---- docs/YESNO.md: a yes/no question about a passage ------------------------------------------------------------ */
/* Y1: the passage judges it: when the passage is read (both statements active) the shelf's fact, joint and open-book parts are
   replaced by the episode's own (weighted EPW as always), unless the passage says nothing of either statement (both within
   YTHIN of zero), when the shelf stays. Y2: the judgement has three values: the evidence for "yes" less that for "no" (fact and
   open book, weighted as the score is) above YTHIN2 is true, below -YTHIN2 false, between them unknown, and an unknown yes/no
   question asked of a text is answered yes (BoolQ: 62% of the answers are yes; a question is asked when the asker leans to yes).
   yn_state is 1 true, 2 false, 3 unknown for SHOW (docs/SPEAK.md W3 will say "I don't know"). */
static int yn_state; static double YTHIN = 0.05, YTHIN2 = 0.1;
static int yn_opt(const char *o) { return is_word_at(o, "yes") ? 1 : is_word_at(o, "no") ? 2 : 0; }
/* Y12: "Is John in the kitchen?" when the episode's current state says John is in the garden: one state per slot (docs/STATE.md), so
   a current fact of the same slot with another value says no, with the same value yes. Only the statement's own state slot, only
   facts not superseded. */
static int state_verdict(Know *A, Frame *hf, int nh) {
    for (int i = 0; i < nh; i++) { KF hk; frame_kf(A, &hf[i], &hk); uint32_t sj, v; int k = state_key(&hk, &sj, &v); if (!k || !state_stmt(&hk) || v == NONE || sj == NONE) continue;
        int same = 0, other = 0, maybe = 0;
        for (uint32_t e = 0; e < nep; e++) { if (ep_sup[e] || (ep[e].flags & F_NEG)) continue; uint32_t s2, v2; if (state_key(&ep[e], &s2, &v2) != k || s2 != sj || v2 == NONE) continue;
            int npp = 0, hasv = 0; for (int a_ = 0; a_ < ep[e].narg; a_++) if (ep[e].role[a_] == A_PP) { npp++; hasv |= ep[e].head[a_] == v; }
            if (npp >= 2 && s5(262144)) { if (hasv) maybe = 1; else other = 1; continue; }     /* Y17: "either in the office or the kitchen": maybe */
            if (v2 == v) same = 1; else other = 1; }
        if (same) return 1; if (maybe) return 2; if (other) return -1; }
    return 0;
}
/* Y16: "Is the pink rectangle to the right of the blue square?", "Does the chest fit in the suitcase?": a relation said, its inverse
   ("below" is "above" the other way, "fits inside" is "bigger" the other way) and chains of it (above above = above). Each relation
   fact becomes (REL, a, b) in one direction; the statement's pair is looked for along chains; the pair the other way round says no. */
enum { R_ABOVE, R_LEFT, R_NORTH, R_EAST, R_BIGGER, R_NREL };
static int rel_of(const char *w, int *flip) { static const struct { const char *w; int r, f; } T[] = { { "above", R_ABOVE, 0 }, { "below", R_ABOVE, 1 }, { "over", R_ABOVE, 0 }, { "under", R_ABOVE, 1 }, { "left", R_LEFT, 0 }, { "right", R_LEFT, 1 },
        { "north", R_NORTH, 0 }, { "south", R_NORTH, 1 }, { "east", R_EAST, 0 }, { "west", R_EAST, 1 }, { "bigger", R_BIGGER, 0 }, { "larger", R_BIGGER, 0 }, { "smaller", R_BIGGER, 1 }, { "inside", R_BIGGER, 1 }, { "in", R_BIGGER, 1 }, { "into", R_BIGGER, 1 }, { 0, 0, 0 } };
    for (int i = 0; T[i].w; i++) if (!strcmp(w, T[i].w)) { *flip = T[i].f; return T[i].r; } return -1; }
/* an entity is its head with its modifier: "the blue square" is square(blue); a bare "the chocolate" is the thing modified by chocolate
   (the box of chocolates), and a bare "the box" is the box said without a modifier */
static int rel_same(const KF *f, int j, uint32_t x, uint32_t xm) {
    if (xm != NONE) return f->head[j] == x && (f->mod[j] == xm || f->mod[j] == NONE);
    return (f->head[j] == x && f->mod[j] == NONE) || f->mod[j] == x || (f->head[j] == x && !ep_mod_exists(x)); }
static int ep_mod_exists(uint32_t x) { for (uint32_t e = 0; e < nep; e++) for (int a = 0; a < ep[e].narg; a++) if (ep[e].head[a] == x && ep[e].mod[a] != NONE) return 1; return 0; }
static int rel_orth(int R) { return R == R_ABOVE ? R_LEFT : R == R_LEFT ? R_ABOVE : R == R_NORTH ? R_EAST : R == R_EAST ? R_NORTH : -1; }
/* from (x, xm) to (y, ym) along R edges; the orthogonal relation's edges may be crossed either way (left of something above you is
   left of you), at least one R edge used */
static int rel_reach(int R, uint32_t x, uint32_t xm, uint32_t y, uint32_t ym, int depth, int used) {
    if (depth > 5) return 0; int O = rel_orth(R);
    for (uint32_t e = 0; e < nep; e++) { const KF *f = &ep[e]; if ((f->flags & F_NEG) || f->narg < 2 || !(f->verb == G_be || !strcmp(LX.s[f->verb], "fit"))) continue;
        for (int j = 1; j < f->narg; j++) { if (f->role[j] != A_PP || f->prep[j] == NONE) continue; int flip, r = rel_of(LX.s[f->prep[j]], &flip); if (r < 0 || (r != R && r != O)) continue;
            if (!strcmp(LX.s[f->prep[j]], "in") && f->verb == G_be) continue;             /* "is in the kitchen" is a place, not a size */
            if (getenv("RELDBG")) fprintf(stderr, "    edge %s(%s) %s %s(%s) [r %d flip %d] from %s(%s) depth %d\n", LX.s[f->head[0]], f->mod[0] != NONE ? LX.s[f->mod[0]] : "-", LX.s[f->prep[j]], LX.s[f->head[j]], f->mod[j] != NONE ? LX.s[f->mod[j]] : "-", r, flip, LX.s[x], xm != NONE ? LX.s[xm] : "-", depth);
            for (int dir = 0; dir < 2; dir++) {                                           /* dir 1: the edge crossed backwards (orthogonal only) */
                if (dir == 1 && r != O) continue;
                int from_i = (flip ^ dir) ? j : 0, to_i = (flip ^ dir) ? 0 : j;
                if (!rel_same(f, from_i, x, xm)) continue;
                int u = used || r == R;
                if (u && rel_same(f, to_i, y, ym)) return 1;
                if (rel_reach(R, f->head[to_i], f->mod[to_i], y, ym, depth + 1, u)) return 1; } } }
    return 0;
}
static int rel_verdict(Know *A, Frame *hf, int nh) {
    for (int i = 0; i < nh; i++) { const Frame *F = &hf[i]; if (F->narg < 2 || F->a[0].head >= OPEN) continue;
        for (int j = 1; j < F->narg; j++) { if (F->a[j].role != A_PP || F->a[j].prep >= OPEN || F->a[j].head >= OPEN) continue; int flip, r = rel_of(A->t.voc[F->a[j].prep].word, &flip); if (r < 0) continue;
            if (!strcmp(A->t.voc[F->a[j].prep].word, "in") && F->verb == A->R.w_be) continue;
            uint32_t p = lx_id(A->t.voc[F->a[0].head].word, 0), pm = F->a[0].mod < OPEN ? lx_id(A->t.voc[F->a[0].mod].word, 0) : NONE;
            uint32_t q = lx_id(A->t.voc[F->a[j].head].word, 0), qm = F->a[j].mod < OPEN ? lx_id(A->t.voc[F->a[j].mod].word, 0) : NONE;
            if (flip) { uint32_t t = p; p = q; q = t; t = pm; pm = qm; qm = t; }
            int told = 0; for (uint32_t e = 0; e < nep && !told; e++) { const KF *f = &ep[e]; if ((f->flags & F_NEG) || f->narg < 2 || !(f->verb == G_be || !strcmp(LX.s[f->verb], "fit"))) continue;
                for (int k = 1; k < f->narg && !told; k++) { if (f->role[k] != A_PP || f->prep[k] == NONE) continue; int fl, rr = rel_of(LX.s[f->prep[k]], &fl); told = rr == r && !(!strcmp(LX.s[f->prep[k]], "in") && f->verb == G_be); } }
            if (!told) continue;                                      /* the text tells no such relations: nothing to judge by */
            int v = rel_reach(r, p, pm, q, qm, 0, 0) ? 1 : -1;   /* a relation asked that the text does not give holds not (positions and sizes are told whole) */
            if (getenv("RELDBG")) fprintf(stderr, "  REL %s: %s(%s) -> %s(%s) = %d\n", A->t.voc[F->a[j].prep].word, LX.s[p], pm != NONE ? LX.s[pm] : "-", LX.s[q], qm != NONE ? LX.s[qm] : "-", v);
            return v; } }
    return 0;
}
static void yesno_judge(BParts *P, int yes) {
    if (getenv("YTHIN")) YTHIN = atof(getenv("YTHIN")); if (getenv("YTHIN2")) YTHIN2 = atof(getenv("YTHIN2"));
    int no = 1 - yes;
    if (s5(1) && P[0].active && P[1].active) {
        int thin = fabs(P[0].efact) < YTHIN && fabs(P[0].eopen) < YTHIN && fabs(P[1].efact) < YTHIN && fabs(P[1].eopen) < YTHIN;
        if (!thin) for (int o = 0; o < 2; o++) { P[o].fact = EPW * P[o].efact; P[o].joint = EPW * P[o].ejoint; P[o].open = EPW * P[o].eopen; P[o].optr = P[o].eptr; } }
    if (s5(4096 | 65536) && yn_slot && yn_slot != 2) { yn_state = yn_slot > 0 ? 1 : 2; for (int o = 0; o < 2; o++) P[o].fact = P[o].joint = P[o].open = 0; P[yn_slot > 0 ? yes : no].fact = EPW; return; }   /* Y12: the state decides */
    if (s5(2)) {
        double ev = (BW[0] * P[yes].fact + BW[1] * P[yes].joint + BW[2] * P[yes].open) - (BW[0] * P[no].fact + BW[1] * P[no].joint + BW[2] * P[no].open);
        if (s5(2048) && P[0].active && P[1].active) ev = EPW * (P[yes].efact - P[no].efact);   /* Y10: the judgement is the passage's facts alone (open book and the shelf are at the prior's precision, docs/YESNO.md) */
        yn_state = ev > YTHIN2 ? 1 : ev < -YTHIN2 ? 2 : 3;
        if (yn_state == 3) { for (int o = 0; o < 2; o++) P[o].fact = P[o].joint = P[o].open = 0; P[yes].fact = 1e-6; } }
}
/* ---- docs/SPEAK.md W1 + W2: answer in words --------------------------------------------------------------------- */
/* The question is read as a frame with a hole (the reader marks the asked slot with an OPEN head: "Where is Mary?" → be who=mary
   pp:in=_). The episode fact closest to that frame which fills the slot gives the answer: the word in that slot, said as the
   text said it (its lemma for now). A yes/no question is judged as the choice reader judges "yes" against "no". */
/* SQuAD's word F1 between what was said and a right answer: words lower-cased, articles and punctuation dropped */
static int f1_words(const char *s, char w[32][48]) { int n = 0; const char *p = s; while (*p && n < 32) { while (*p && !isalnum((unsigned char)*p)) p++; if (!*p) break; int L = 0; while (*p && isalnum((unsigned char)*p) && L < 47) w[n][L++] = (char)tolower((unsigned char)*p++); w[n][L] = 0;
        if (strcmp(w[n], "a") && strcmp(w[n], "an") && strcmp(w[n], "the")) n++; } return n; }
static double word_f1(const char *said, const char *gold) {
    char a[32][48], b[32][48]; int na = f1_words(said, a), nb = f1_words(gold, b); if (!na || !nb) return na == nb;
    int used[32] = { 0 }, hit = 0; for (int i = 0; i < na; i++) for (int j = 0; j < nb; j++) if (!used[j] && !strcmp(a[i], b[j])) { used[j] = 1; hit++; break; }
    if (!hit) return 0; double pr = (double)hit / na, rc = (double)hit / nb; return 2 * pr * rc / (pr + rc);
}
static int numlike(const char *w);
/* docs/SPEAK.md W2: the filler said as the text said it. The fact points into the episode's text; in its sentence the word whose
   lemma is the filler's (or begins with it: panthers / panther) is found, and the capitalised or modifying words just before it
   are kept ("Carolina Panthers", "Levi Stadium"); a number word's digits are found by its value. */
static int ask_surface(const KF *f, int j, char *out, size_t cap) {
    uint64_t p = f->ptr[0]; if (p == ~0ull || (int)(p >> FID_SHIFT) != EPFID) return 0; size_t off = (size_t)(p & ((1ull << FID_SHIFT) - 1)); if (off >= neb) return 0;
    size_t a = off; while (a > 0 && epbuf[a - 1] != '.' && epbuf[a - 1] != '!' && epbuf[a - 1] != '?' && epbuf[a - 1] != '\n') a--;
    size_t b = off; while (b < neb && epbuf[b] != '.' && epbuf[b] != '!' && epbuf[b] != '?' && epbuf[b] != '\n') b++;
    const char *lem = LX.s[f->head[j]]; size_t L = strlen(lem); const char *mod = f->mod[j] != NONE ? LX.s[f->mod[j]] : 0;
    static const char *NUMW[] = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", 0 }; int numv = -1; for (int i = 0; NUMW[i]; i++) if (!strcmp(lem, NUMW[i])) numv = i;
    size_t ws[128], we[128]; int nw = 0;                   /* the sentence's words */
    for (size_t i = a; i < b && nw < 128; ) { while (i < b && !isalnum((unsigned char)epbuf[i])) i++; if (i >= b) break; size_t s0 = i; while (i < b && (isalnum((unsigned char)epbuf[i]) || epbuf[i] == '\'')) i++; ws[nw] = s0; we[nw++] = i; }
    int hit = -1;
    for (int w = 0; w < nw && hit < 0; w++) { size_t wl = we[w] - ws[w]; char t[64]; if (wl >= sizeof t) continue; for (size_t x = 0; x < wl; x++) t[x] = (char)tolower((unsigned char)epbuf[ws[w] + x]); t[wl] = 0;
        if (!strcmp(t, lem) || (wl > L && !strncmp(t, lem, L) && wl - L <= 2) || (numv >= 0 && atoi(t) == numv && isdigit((unsigned char)t[0]))) hit = w; }
    if (hit < 0 && numlike(lem)) {           /* a number said in figures ("24-10", "2015", "1,200"): the sentence's first run of digits */
        for (size_t i = a; i < b; i++) if (isdigit((unsigned char)epbuf[i]) && (i == a || !isalnum((unsigned char)epbuf[i - 1]))) { size_t e = i; while (e < b && (isdigit((unsigned char)epbuf[e]) || ((epbuf[e] == '-' || epbuf[e] == ',' || epbuf[e] == '.' || epbuf[e] == ':') && e + 1 < b && isdigit((unsigned char)epbuf[e + 1])))) e++;
            size_t len = e - i; if (len + 1 > cap) len = cap - 1; memcpy(out, epbuf + i, len); out[len] = 0; return 1; } }
    if (hit < 0) return 0;
    int st = hit;                                           /* the words before: the modifier, or capitalised words (a name) */
    while (st > 0) { size_t wl = we[st - 1] - ws[st - 1]; char t[64]; if (wl >= sizeof t) break; for (size_t x = 0; x < wl; x++) t[x] = (char)tolower((unsigned char)epbuf[ws[st - 1] + x]); t[wl] = 0;
        int ismod = mod && (!strcmp(t, mod) || (wl > strlen(mod) && !strncmp(t, mod, strlen(mod)) && wl - strlen(mod) <= 2));
        int cap_ = isupper((unsigned char)epbuf[ws[st - 1]]) && isupper((unsigned char)epbuf[ws[hit]]) && hit - st < 3;
        if (ismod || cap_) st--; else break; }
    size_t len = we[hit] - ws[st]; if (len + 1 > cap) len = cap - 1; memcpy(out, epbuf + ws[st], len); out[len] = 0; return 1;
}
static int numlike(const char *w) { static const char *N[] = { "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", 0 };
    for (int i = 0; N[i]; i++) { size_t L = strlen(N[i]); if (!strncmp(w, N[i], L) && (w[L] == 0 || w[L] == '-')) return 1; } return 0; }
static int timelike(const char *w) {                     /* a number word, a month, a day of the week, a time of day */
    static const char *X[] = { "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning", "afternoon", "evening", "night", "noon", "midnight", "yesterday", "today", "tomorrow", "week", "month", "year", "century", "summer", "winter", "spring", "autumn", "fall", 0 };
    static const char *N[] = { "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", 0 };
    for (int i = 0; X[i]; i++) if (!strcmp(w, X[i])) return 1; for (int i = 0; N[i]; i++) { size_t L = strlen(N[i]); if (!strncmp(w, N[i], L) && (w[L] == 0 || w[L] == '-')) return 1; } return 0; }
static int ask_answer(Know *A, const char *q, char *out, size_t cap) {
    if (yesno_q(q)) { BParts P[2]; yn_slot = 0; bg_option(A, q, "yes", &P[0]); A->t.ntok = 0; bg_option(A, q, "no", &P[1]); A->t.ntok = 0;
        yn_state = 0; if (yn_slot == 2) { snprintf(out, cap, "maybe"); return 1; } if (s5(3)) yesno_judge(P, 0); double sc[2]; scores(q, P, 2, sc); snprintf(out, cap, "%s", sc[0] >= sc[1] ? "yes" : "no"); return 1; }
    TextContext *t = &A->t; uint32_t n0 = t->ntok; Refs keep = A->R.refs; text_take(t, NULL, q, strlen(q)); A->R.question = 1;
    Frame hf[8]; int wh = -1; int nh = read_span(&A->R, t->tok, n0, t->ntok, hf, 8, &wh); t->ntok = n0; t->src_len = 0; A->R.refs = keep; A->R.question = 0;
    if (s5(2097152)) for (int i = 0; i < nh; i++) {        /* "What was the final score of the game?": the score is the subject, the hole is what it was */
        int ag = -1, at = -1; for (int a = 0; a < hf[i].narg; a++) { if (hf[i].a[a].role == A_AGENT && hf[i].a[a].head == OPEN) ag = a; if (hf[i].a[a].role == A_ATTR && hf[i].a[a].head != OPEN) at = a; }
        if (ag >= 0 && at >= 0 && hf[i].verb == A->R.w_be) { Arg t = hf[i].a[ag]; hf[i].a[ag] = hf[i].a[at]; hf[i].a[ag].role = A_AGENT; hf[i].a[at] = t; hf[i].a[at].role = A_ATTR; hf[i].a[at].mod = hf[i].a[ag].mod == NONE ? t.mod : NONE; } }
    int fi = -1, slot = -1; for (int i = 0; i < nh && fi < 0; i++) for (int a = 0; a < hf[i].narg; a++) if (hf[i].a[a].head == OPEN) { fi = i; slot = a; break; }
    if (fi < 0 && nh > 0 && wh >= 0 && wh < hf[0].narg) { fi = 0; slot = wh; }
    if (fi < 0) { snprintf(out, cap, "-"); return 0; }
    uint8_t role = hf[fi].a[slot].role; uint32_t prep = hf[fi].a[slot].prep < OPEN ? lx_id(A->t.voc[hf[fi].a[slot].prep].word, 0) : NONE;
    { int haswho = 0, haspp = 0; for (int a_ = 0; a_ < hf[fi].narg; a_++) { haswho |= hf[fi].a[a_].role == A_AGENT && hf[fi].a[a_].head != OPEN; haspp |= hf[fi].a[a_].role == A_PP; }
      if (role == A_ATTR && !haswho && haspp && hf[fi].verb == A->R.w_be) role = A_AGENT; }      /* "What is east of the hallway?": the thing asked is the subject */
    static const char *DIRS[] = { "north", "south", "east", "west", "above", "below", "left", "right", 0 }; int dirprep = 0;
    if (prep != NONE) for (int i = 0; DIRS[i]; i++) dirprep |= !strcmp(LX.s[prep], DIRS[i]);
    KF qk; frame_kf(A, &hf[fi], &qk); uint32_t qs = NONE, qv = NONE; int qkind = 0;     /* the question as a state with a hole: 1 "where is S" (place), 2 "what does S hold" (holder) */
    { uint32_t who = NONE; for (int a_ = 0; a_ < qk.narg && who == NONE; a_++) if (qk.role[a_] == A_AGENT || qk.role[a_] == A_PAT) who = qk.head[a_];
      if (who == NONE && qk.narg) who = qk.head[0];
      if (role == A_PP && (qk.verb == G_be || gin(G_motion, qk.verb))) { qkind = 1; qs = who; }
      else if (role == A_PAT && (gin(G_take, qk.verb) || (qk.verb == G_have && ntracked))) { qkind = 2; qs = NONE; qv = who; } }   /* "what is Mary carrying", "how many objects does she have": holding, when things are held in the text */
    if (getenv("ASKDBG")) { fprintf(stderr, "  ASKQ %s -> nh %d frame %d slot %d role %d kind %d subj %s val %s | verb %s", q, nh, fi, slot, role, qkind, qs < LX.n ? LX.s[qs] : "?", qv < LX.n ? LX.s[qv] : "?", qk.verb < LX.n ? LX.s[qk.verb] : "-");
        for (int a_ = 0; a_ < qk.narg; a_++) fprintf(stderr, " r%d:%s%s", qk.role[a_], qk.prep[a_] < LX.n ? LX.s[qk.prep[a_]] : "", qk.head[a_] < LX.n ? LX.s[qk.head[a_]] : "_"); fprintf(stderr, "\n"); }
    { /* W1b: "Where was the football before the bathroom?": the state before that one. The subject's state facts in the order said; the latest
         with the named value, and the last different value before it. */
        const char *bef = strcasestr(q, " before "); if (bef && qkind == 1 && qs != NONE) { char w[64]; const char *x = bef + 8; if (is_word_at(x, "the")) x += 4; int n = 0; while (*x && *x != ' ' && *x != '?' && n < 63) w[n++] = *x++; w[n] = 0;
            uint32_t X = lx_id(w, 0); uint32_t at = 0, pv = NONE, po = 0;
            for (uint32_t e = 0; e < nep; e++) { uint32_t s2, v2; if ((ep[e].flags & F_NEG) || state_key(&ep[e], &s2, &v2) != 1 || s2 != qs) continue; if (v2 == X && ep_ord[e] > at) at = ep_ord[e]; }
            if (at) for (uint32_t e = 0; e < nep; e++) { uint32_t s2, v2; if ((ep[e].flags & F_NEG) || state_key(&ep[e], &s2, &v2) != 1 || s2 != qs || v2 == X) continue; if (ep_ord[e] < at && ep_ord[e] > po) { po = ep_ord[e]; pv = v2; } }
            if (pv != NONE) { snprintf(out, cap, "%s", LX.s[pv]); return 1; } } }
    if (qkind == 2) {   /* W1c: what someone holds now: "What is Mary carrying?" the things, "How many objects is Mary carrying?" their count */
        uint32_t holder = qv; uint32_t things[32]; int nt = 0; static const char *NUM[] = { "none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten" };
        if (holder != NONE) for (uint32_t e = 0; e < nep; e++) { uint32_t s2, v2; if (ep_sup[e] || (ep[e].flags & F_NEG) || state_key(&ep[e], &s2, &v2) != 2 || v2 != holder) continue; int dup = 0; for (int i = 0; i < nt; i++) dup |= things[i] == s2; if (!dup && nt < 32) things[nt++] = s2; }
        if (is_word_at(q, "how many")) { snprintf(out, cap, "%s", NUM[nt > 10 ? 10 : nt]); return 1; }
        if (holder != NONE && role == A_PAT) { if (!nt) { snprintf(out, cap, "nothing"); return 1; } size_t L = 0; for (int i = 0; i < nt && L + 2 < cap; i++) L += (size_t)snprintf(out + L, cap - L, "%s%s", i ? " " : "", LX.s[things[i]]); return 1; } }
    uint32_t wtype = hf[fi].a[slot].mod < OPEN ? hf[fi].a[slot].mod : NONE; const char *wtw = wtype != NONE ? A->t.voc[wtype].word : "";
    int want_adj = !strcmp(wtw, "color") || !strcmp(wtw, "colour") || !strcmp(wtw, "size") || !strcmp(wtw, "shape");   /* "what colour is Greg": an adjective, not his kind */
    int want_time = !strcmp(wtw, "day") || !strcmp(wtw, "date") || !strcmp(wtw, "year") || !strcmp(wtw, "month") || !strcmp(wtw, "time") || is_word_at(q, "when");   /* "what day", "what year": a number, a month, a day */
#define FILLER_OK(f_, j_) ((!want_adj || wclass(LX.s[(f_)->head[j_]], strlen(LX.s[(f_)->head[j_]])) == 'J') && (!want_time || timelike(LX.s[(f_)->head[j_]]) || ((f_)->mod[j_] != NONE && timelike(LX.s[(f_)->mod[j_]]))))
    double best = 0; const KF *bf = 0;
    int stateq = qkind || qk.verb == G_be || qk.verb == G_have;      /* a derived state ("has", "is in") answers a question about a state, not about who gave or took */
    for (uint32_t e = 0; e < nep; e++) { const KF *f = &ep[e]; int has = 0; for (int j = 0; j < f->narg; j++) has |= f->role[j] == role && FILLER_OK(f, j) && (!dirprep || f->prep[j] == prep); if (!has || (f->flags & F_NEG) || (ep_der[e] && !stateq)) continue;   /* a direction asked is that direction, not another */
        double s = kfact_sim(A, &hf[fi], f) * conf(f); if (ep_sup[e]) s *= SUPW;
        if (s > best || (s == best && bf && ep_ord[e] > ep_ord[bf - ep])) { best = s; bf = f; } }   /* equally close: the later one */
    /* W1f: "How do you go from the kitchen to the garden?": a path over the places' relations (north / south / east / west), the
       moves said in order ("south east"). Each relation fact "A is north of B" is a step north from B to A and south from A to B. */
    if (is_word_at(q, "how") && strcasestr(q, " from ") && strcasestr(q, " to ")) {
        char xw[64] = "", yw[64] = ""; const char *fx = strcasestr(q, " from ") + 6, *fy = strcasestr(q, " to ") + 4; if (is_word_at(fx, "the")) fx += 4; if (is_word_at(fy, "the")) fy += 4;
        int n = 0; while (*fx && *fx != ' ' && *fx != '?' && n < 63) xw[n++] = *fx++; xw[n] = 0; n = 0; while (*fy && *fy != ' ' && *fy != '?' && n < 63) yw[n++] = *fy++; yw[n] = 0;
        uint32_t X = lx_id(xw, 0), Y = lx_id(yw, 0);
        if (X != NONE && Y != NONE && X != Y) { static const char *D[] = { "north", "south", "east", "west", 0 }; static const char *INVD[] = { "south", "north", "west", "east", 0 };
            uint32_t at[64]; int from[64], via[64], nq_ = 0, head = 0; at[nq_] = X; from[nq_] = -1; via[nq_++] = -1;
            while (head < nq_ && nq_ < 60) { uint32_t cur = at[head];
                for (uint32_t e = 0; e < nep && nq_ < 60; e++) { const KF *f = &ep[e]; if ((f->flags & F_NEG) || f->narg < 2 || f->verb != G_be) continue;
                    for (int j = 1; j < f->narg; j++) { if (f->role[j] != A_PP || f->prep[j] == NONE) continue; int d = -1; for (int i = 0; D[i]; i++) if (!strcmp(LX.s[f->prep[j]], D[i])) d = i; if (d < 0) continue;
                        uint32_t A_ = f->head[0], B_ = f->head[j]; uint32_t nxt = NONE; int mv = -1;
                        if (B_ == cur) { nxt = A_; mv = d; } else if (A_ == cur) { nxt = B_; mv = d ^ 1; }     /* from B north to A; from A the other way */
                        if (nxt == NONE) continue; int seen = 0; for (int k = 0; k < nq_; k++) seen |= at[k] == nxt; if (seen) continue;
                        at[nq_] = nxt; from[nq_] = head; via[nq_] = mv; nq_++;
                        if (nxt == Y) { int path[8], np = 0, k = nq_ - 1; while (from[k] >= 0 && np < 8) { path[np++] = via[k]; k = from[k]; }
                            size_t L = 0; for (int i = np - 1; i >= 0 && L + 8 < cap; i--) L += (size_t)snprintf(out + L, cap - L, "%s%s", i == np - 1 ? "" : " ", D[path[i]]); return 1; } } }
                head++; } } }
    /* W1e: "What is the kitchen south of?" when the text says "the office is north of the kitchen": the relation read the other way */
    if ((!bf || best < 0.3) && role == A_AGENT && qk.narg) {                   /* "What is east of the hallway?": the thing whose west is the hallway */
        for (int a_ = 0; a_ < qk.narg; a_++) if (qk.role[a_] == A_PP && qk.prep[a_] != NONE && qk.head[a_] != NONE) { static const char *INV2[][2] = { { "north", "south" }, { "east", "west" }, { "above", "below" }, { "left", "right" }, { 0, 0 } };
            const char *pw = LX.s[qk.prep[a_]]; const char *inv = 0; for (int i = 0; INV2[i][0]; i++) { if (!strcmp(pw, INV2[i][0])) inv = INV2[i][1]; if (!strcmp(pw, INV2[i][1])) inv = INV2[i][0]; }
            if (!inv) continue; uint32_t ip = lx_id(inv, 0), Y = qk.head[a_];
            for (uint32_t e = 0; e < nep; e++) { const KF *f = &ep[e]; if ((f->flags & F_NEG) || f->narg < 2 || f->head[0] != Y) continue;
                for (int j = 1; j < f->narg; j++) if (f->role[j] == A_PP && f->prep[j] == ip) { snprintf(out, cap, "%s", LX.s[f->head[j]]); return 1; } } } }
    if ((!bf || best < 0.3) && role == A_PP && prep != NONE && qk.narg) { static const char *INV[][2] = { { "north", "south" }, { "east", "west" }, { "above", "below" }, { "left", "right" }, { "inside", "outside" }, { 0, 0 } };
        const char *pw = LX.s[prep]; const char *inv = 0; for (int i = 0; INV[i][0]; i++) { if (!strcmp(pw, INV[i][0])) inv = INV[i][1]; if (!strcmp(pw, INV[i][1])) inv = INV[i][0]; }
        if (inv) { uint32_t ip = lx_id(inv, 0); uint32_t X = qk.head[0];
            for (uint32_t e = 0; e < nep; e++) { const KF *f = &ep[e]; if ((f->flags & F_NEG) || f->narg < 2) continue;
                for (int j = 1; j < f->narg; j++) if (f->role[j] == A_PP && f->prep[j] == ip && f->head[j] == X) { snprintf(out, cap, "%s", LX.s[f->head[0]]); return 1; } } } }
    /* W1d: nothing said of the subject itself: what is said of its kind ("Gertrude is a mouse. Mice are afraid of wolves": deduction), or of
       the other members of its kind ("Lily is a swan. Lily is green. Greg is a swan. What colour is Greg?": induction, the commonest value) */
    if ((!bf || best < 0.3) && qk.narg) { uint32_t X = qk.head[0]; uint32_t kinds[8]; int nk = 0;
        for (uint32_t e = 0; e < nep && nk < 8; e++) if (ep[e].verb == G_be && !(ep[e].flags & F_NEG) && ep[e].narg >= 2 && ep[e].head[0] == X && ep[e].role[1] == A_ATTR) { int dup = 0; for (int i = 0; i < nk; i++) dup |= kinds[i] == ep[e].head[1]; if (!dup) kinds[nk++] = ep[e].head[1]; }
        uint32_t vals[32]; int cnt[32], nv = 0;
        for (int ki = 0; ki < nk; ki++) { uint32_t K = kinds[ki];
            for (uint32_t e = 0; e < nep; e++) { const KF *f = &ep[e]; if ((f->flags & F_NEG) || ep_sup[e] || f->narg < 2) continue;
                int about_kind = f->head[0] == K, about_member = 0;
                if (!about_kind) { for (uint32_t e2 = 0; e2 < nep && !about_member; e2++) about_member = ep[e2].verb == G_be && ep[e2].narg >= 2 && ep[e2].head[0] == f->head[0] && ep[e2].role[1] == A_ATTR && ep[e2].head[1] == K && f->head[0] != X; }
                if (!about_kind && !about_member) continue;
                if (f->head[1] == K && f->verb == G_be) continue;                                   /* "Lily is a swan" itself */
                KF g = *f; g.head[0] = X;                                                          /* said of the kind, taken as said of X */
                double s = kfact_sim(A, &hf[fi], &g) * (about_kind ? 1.0 : 0.8); if (s < 0.3) continue;
                int j = -1; for (int a_ = 0; a_ < f->narg && j < 0; a_++) if (f->role[a_] == role && (role != A_PP || prep == NONE || f->prep[a_] == prep) && FILLER_OK(f, a_)) j = a_;
                if (j < 0) continue; uint32_t v = f->head[j]; int found = 0; for (int i = 0; i < nv; i++) if (vals[i] == v) { cnt[i] += about_kind ? 3 : 1; found = 1; } if (!found && nv < 32) { vals[nv] = v; cnt[nv++] = about_kind ? 3 : 1; } } }
        if (nv) { int bi = 0; for (int i = 1; i < nv; i++) if (cnt[i] > cnt[bi]) bi = i; if (getenv("ASKDBG")) fprintf(stderr, "  ASK kind: %d kinds, %d values, best %s\n", nk, nv, LX.s[vals[bi]]); snprintf(out, cap, "%s", LX.s[vals[bi]]); return 1; } }
    if (!bf || best < 0.3) { snprintf(out, cap, "-"); return 0; }
    int j0 = -1; for (int j = 0; j < bf->narg && j0 < 0; j++) if (bf->role[j] == role && (role != A_PP || prep == NONE || bf->prep[j] == prep) && FILLER_OK(bf, j)) j0 = j;
    for (int j = 0; j < bf->narg && j0 < 0; j++) if (bf->role[j] == role && FILLER_OK(bf, j)) j0 = j;
    if (j0 < 0) { snprintf(out, cap, "-"); return 0; }
#undef FILLER_OK
    if (s5(524288) && ask_surface(bf, j0, out, cap)) return 1;   /* W2: said as the text said it */
    if (getenv("ASKDBG")) fprintf(stderr, "  ASK %s -> frame %d slot %d role %d best %.3f fact %s\n", q, fi, slot, role, best, bf->verb != NONE ? LX.s[bf->verb] : "-");
    snprintf(out, cap, "%s", LX.s[bf->head[j0]]); return 1;
}
static int run_passages(int nfile, char **files) {
    Know A; answer_ctx(&A); Know k; memset(&k, 0, sizeof k); text_init(&k.t); r2_init(&k.R, &k.t, &k.wm);
    cx_epst = 0x2545F4914F6CDD1Dull; if (getenv("SUPW")) SUPW = atof(getenv("SUPW")); state_init();
    static FILE *dump; if (!dump && getenv("DUMP")) dump = fopen(getenv("DUMP"), "a");
    for (int fi = 0; fi < nfile; fi++) {
        FILE *f = fopen(files[fi], "r"); if (!f) { perror(files[fi]); return 1; }
        static char line[1 << 18]; double ok = 0, f1sum = 0; int n = 0, open = 0, nasked = 0; double t0 = now();
        const char *bn = strrchr(files[fi], '/'); bn = bn ? bn + 1 : files[fi];
        while (fgets(line, sizeof line, f)) {
            size_t L = strlen(line); while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
            if (line[0] == 'S') { if (open) ep_end(); ep_begin(); open = 1; cx_epst = 0x2545F4914F6CDD1Dull; continue; }
            if (line[0] == 'T') { ep_read(&k, line + 2); continue; }
            if (line[0] == 'A') {                                      /* A id question answer: the answer in words (docs/SPEAK.md) */
                char *fld[4]; int nf = 0; for (char *p = strtok(line, "\t"); p && nf < 4; p = strtok(0, "\t")) fld[nf++] = p; if (nf < 4) continue;
                char ans[256]; ask_answer(&A, fld[2], ans, sizeof ans); A.t.ntok = 0;
                int okq = 0; double f1 = 0;                               /* several right answers "a || b": exact match on any; word F1 the best (SQuAD's measures) */
                for (char *g = fld[3]; g; ) { char *sep = strstr(g, " || "); if (sep) *sep = 0; okq |= !strcasecmp(ans, g); double f = word_f1(ans, g); if (f > f1) f1 = f; if (sep) { *sep = ' '; g = sep + 4; } else break; }
                ok += okq; f1sum += f1; nasked++;
                if (getenv("SHOW")) printf("  %s\t%s\tsaid=%s\tright=%s\t%s\n", fld[1], fld[2], ans, fld[3], okq ? "OK" : "WRONG");
                n++; continue; }
            if (line[0] != 'Q') continue;
            char *fld[MAXO + 4]; int nf = 0; for (char *p = strtok(line, "\t"); p && nf < MAXO + 4; p = strtok(0, "\t")) fld[nf++] = p;
            if (nf < 6) continue;                                   /* Q id question option option… letters: 2 to MAXO options, one or more right letters ("AC") */
            int no = nf - 4; const char *ans = fld[nf - 1]; const char *q = fld[2]; int want = ans[0] - 'A'; BParts P[MAXO]; double sc[MAXO];
            int right[MAXO]; for (int o = 0; o < no; o++) right[o] = strchr(ans, 'A' + o) != 0;
            yn_slot = 0; for (int o = 0; o < no; o++) { bg_option(&A, q, fld[3 + o], &P[o]); A.t.ntok = 0; }
            yn_state = 0; if (s5(3) && no == 2 && (yesno_q(q) || (s5(32) && !wh_q(q))) && yn_opt(fld[3]) && yn_opt(fld[4])) yesno_judge(P, yn_opt(fld[3]) == 1 ? 0 : 1);   /* docs/YESNO.md */
            { int ex = 0; for (int o = 0; o < no; o++) ex += P[o].active && P[o].fact >= FX * EPW * 0.5;     /* F5: one option said almost whole by the passage: the fact decides */
                if (ex == 1) for (int o = 0; o < no; o++) { P[o].open *= FO; P[o].joint *= FO; } }
            scores(q, P, no, sc);
            double best = -1e9; int nb = 0, nbr = 0; for (int o = 0; o < no; o++) if (sc[o] > best + 1e-9) { best = sc[o]; nb = 1; nbr = right[o]; } else if (sc[o] > best - 1e-9) { nb++; nbr += right[o]; }
            ok += (double)nbr / nb;
            if (dump) { fprintf(dump, "%s\t%d\t%d\t%d", bn, want, no, asks_opposite(q));
                for (int o = 0; o < no; o++) fprintf(dump, "\t%.5f %.5f %.5f %.5f %.5f %d", P[o].fact, P[o].joint, P[o].open, P[o].mean, P[o].fit, P[o].active);
                if (getenv("EPDUMP")) for (int o = 0; o < no; o++) fprintf(dump, "\t%.5f %.5f %.5f", P[o].efact, P[o].ejoint, P[o].eopen);   /* docs/YESNO.md Y0: the episode's own parts */
                fprintf(dump, "\n"); }
            if (getenv("SHOW")) { int pick = 0; for (int o = 1; o < no; o++) if (sc[o] > sc[pick] + 1e-9) pick = o; printf("  %s\t%s\tpick=%s\tright=%s\t%s%s\n", fld[1], q, fld[3 + pick], want >= 0 && want < no ? fld[3 + want] : ans, right[pick] ? "OK" : "WRONG", yn_state == 1 ? "\ttrue" : yn_state == 2 ? "\tfalse" : yn_state == 3 ? "\tunknown" : ""); }
            n++; }
        if (open) ep_end();
        fclose(f); if (dump) fflush(dump);
        printf("%-22s %5d q   %6.2f%%   (episode weight %.2f)  %.1f ms/q\n", bn, n, 100 * ok / (n ? n : 1), EPW, 1000 * (now() - t0) / (n ? n : 1));
        if (nasked) printf("RESULT bgask file=%s n=%d acc=%.2f f1=%.2f epw=%.2f\n", bn, n, 100 * ok / (n ? n : 1), 100 * f1sum / (n ? n : 1), EPW);
        else printf("RESULT bgpass file=%s n=%d acc=%.2f epw=%.2f\n", bn, n, 100 * ok / (n ? n : 1), EPW);
    }
    return 0;
}

/* the cortex on word tests: Spearman between its cosines and people's ratings (a word it has not met scores 0) */
static void cx_ranks(const double *a, int n, double *r) {
    int *ix = malloc(n * sizeof *ix); for (int i = 0; i < n; i++) ix[i] = i;
    for (int i = 1; i < n; i++) for (int j = i; j > 0 && a[ix[j]] < a[ix[j - 1]]; j--) { int t = ix[j]; ix[j] = ix[j - 1]; ix[j - 1] = t; }
    for (int i = 0; i < n;) { int j = i; while (j + 1 < n && a[ix[j + 1]] == a[ix[i]]) j++; for (int k = i; k <= j; k++) r[ix[k]] = (i + j) / 2.0 + 1; i = j + 1; }
    free(ix);
}
static int run_words(int nfile, char **files) {
    for (int fi = 0; fi < nfile; fi++) {
        FILE *f = fopen(files[fi], "r"); if (!f) { perror(files[fi]); return 1; } char a[256], b[256]; double h; int n = 0, known = 0;
        double *x = malloc(20000 * sizeof *x), *y = malloc(20000 * sizeof *y), *rx = malloc(20000 * sizeof *rx), *ry = malloc(20000 * sizeof *ry);
        while (n < 20000 && fscanf(f, "%255s %255s %lf%*[^\n]", a, b, &h) == 3) {
            for (char *c = a; *c; c++) *c = (char)tolower((unsigned char)*c); for (char *c = b; *c; c++) *c = (char)tolower((unsigned char)*c);
            uint32_t ga = lx_id(a, 0), gb = lx_id(b, 0); int ok = cx_has(ga) && cx_has(gb);
            x[n] = ok ? mcos((int)ga, (int)gb) : 0; y[n++] = h; known += ok; }
        fclose(f); cx_ranks(x, n, rx); cx_ranks(y, n, ry);
        double mx = 0, my = 0, sxy = 0, sxx = 0, syy = 0; for (int i = 0; i < n; i++) { mx += rx[i]; my += ry[i]; } mx /= n; my /= n;
        for (int i = 0; i < n; i++) { sxy += (rx[i] - mx) * (ry[i] - my); sxx += (rx[i] - mx) * (rx[i] - mx); syy += (ry[i] - my) * (ry[i] - my); }
        const char *bn = strrchr(files[fi], '/'); bn = bn ? bn + 1 : files[fi];
        printf("%-24s %5d pairs (%5d known)  spearman %.3f\n", bn, n, known, sxy / sqrt(sxx * syy));
        printf("RESULT words file=%s n=%d known=%d spearman=%.4f\n", bn, n, known, sxy / sqrt(sxx * syy));
        free(x); free(y); free(rx); free(ry); }
    return 0;
}
/* the words nearest to some words, by the cortex (to look at) */
static int run_near(int nw, char **ws) {
    for (int i = 0; i < nw; i++) { uint32_t g = lx_id(ws[i], 0); if (!cx_has(g)) { printf("%s: not met often enough\n", ws[i]); continue; }
        enum { K = 10 }; uint32_t top[K]; float tv[K]; int nt = 0;
        for (uint32_t h = 0; h < LX.n && (h >> CXB) < (uint32_t)cxnb; h++) { if (h == g || CXN(h) < 20) continue; float c = mcos((int)g, (int)h);
            if (nt < K) { top[nt] = h; tv[nt++] = c; } else if (c > tv[K - 1]) { top[K - 1] = h; tv[K - 1] = c; } else continue;
            for (int x = nt - 1; x > 0 && tv[x] > tv[x - 1]; x--) { float t = tv[x]; tv[x] = tv[x - 1]; tv[x - 1] = t; uint32_t u = top[x]; top[x] = top[x - 1]; top[x - 1] = u; } }
        printf("%s (met %u):", ws[i], CXN(g)); for (int x = 0; x < nt; x++) printf(" %s(%.2f)", LX.s[top[x]], tv[x]); printf("\n"); }
    return 0;
}

/* what the reader makes of some text (stdin): one line per fact, for looking at the reader's work */
static int run_facts(void) {
    Know k; memset(&k, 0, sizeof k); text_init(&k.t); r2_init(&k.R, &k.t, &k.wm); static char line[1 << 16]; static Frame fr[512];
    static const char *RN[] = { "who", "what", "is", "pp" };
    while (fgets(line, sizeof line, stdin)) {
        k.t.ntok = 0; k.t.src_len = 0; text_take(&k.t, NULL, line, strlen(line)); if (k.t.ntok) k.t.tok[k.t.ntok - 1].end = 1;
        int n = read_span(&k.R, k.t.tok, 0, k.t.ntok, fr, 512, 0); printf("%s", line);
        for (int i = 0; i < n; i++) { const Frame *f = &fr[i]; printf("  %s%s%s", f->flags & F_NEG ? "NOT " : "", f->flags & F_PAST ? "[past] " : f->flags & F_PRES ? "[now] " : "", f->verb < OPEN ? k.t.voc[f->verb].word : "-");
            for (int a = 0; a < f->narg; a++) { const Arg *x = &f->a[a]; printf("  %s%s%s=%s", RN[x->role], x->prep < OPEN ? ":" : "", x->prep < OPEN ? k.t.voc[x->prep].word : "", x->head < OPEN ? k.t.voc[x->head].word : "?");
                if (x->mod < OPEN) printf("(%s)", k.t.voc[x->mod].word); if (x->k & K_KIND) printf("[kind]"); }
            printf("\n"); } }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "bg read BRAIN SECONDS FILE... | bg arc BRAIN FILE... | bg serve BRAIN | bg words BRAIN FILE... | bg near BRAIN WORD...\n"); return 1; }
    lex_load(getenv("WORDCLASS") ? getenv("WORDCLASS") : "data/understand/wordclass.tsv");
    static Meaning m; meaning_init(&m);
    if (meaning_load(&m, getenv("MEANING") ? getenv("MEANING") : MEANING_PATH) <= 0) { fprintf(stderr, "no meanings\n"); return 1; }
    MM = &m;                          /* the old meaning table: only its list of glue words (the most common) is used now */
    if (getenv("NOPEN")) NOPEN = atoi(getenv("NOPEN"));
    if (getenv("TENSEW")) TENSEW = atof(getenv("TENSEW")); if (getenv("FX")) FX = atof(getenv("FX")); if (getenv("FO")) FO = atof(getenv("FO"));
    if (getenv("CXTH")) CXTH = atof(getenv("CXTH")); if (getenv("CXMIN")) CXMIN = atoi(getenv("CXMIN")); if (getenv("CXFIT")) BW[4] = atof(getenv("CXFIT"));
    if (!strcmp(argv[1], "facts")) return run_facts();
    if (!strcmp(argv[1], "read") && argc >= 5) return run_read(argv[2], atof(argv[3]), argc - 4, argv + 4);
    double t0 = now(); if (brain_load(argv[2])) return 1;
    fprintf(stderr, "brain %s: %u facts, %u words (loaded in %.1f s)\n", argv[2], nkf, LX.n, now() - t0);
    cx_norms();
    if (!strcmp(argv[1], "words")) return run_words(argc - 3, argv + 3);
    if (!strcmp(argv[1], "passages")) { if (getenv("EPW")) EPW = atof(getenv("EPW"));
        if (getenv("EA")) EA = atof(getenv("EA")); return run_passages(argc - 3, argv + 3); }
    if (!strcmp(argv[1], "near")) return run_near(argc - 3, argv + 3);
    if (!strcmp(argv[1], "arc")) return run_arc(argc - 3, argv + 3);
    if (!strcmp(argv[1], "serve")) return run_serve();
    return 1;
}
