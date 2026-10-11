/*
 * compare.c — v0.1 vs v0.2 neurons, same memory, same tests.
 * Rules: docs/NEURON_COMPARE.md (fixed before this file was written).
 *
 *   compare DESIGN SETTING BUDGET_MB BOOK NOVEL_BOOK
 *     DESIGN 1 = v0.1 (neuron stores content, one neuron per thing)
 *     DESIGN 2 = v0.2 (neurons hold state only, knowledge in synapses)
 *     SETTING 0..3 (see setting1 / setting2)
 *
 * Prints one line of results. Both brains see the same events and answer
 * the same calls: store, recall, familiar, next.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t mix(uint64_t x) { x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31); }

/* ------------------------------------------------------------------ */
/* Text → events (identical for both designs)                          */
/* ------------------------------------------------------------------ */

#define MAXW 12
typedef struct { int n; int w[MAXW]; } Event;

typedef struct { char **str; int *count; int n, cap; int *slot; int nslot; } Vocab;
static Vocab V;

static uint64_t hstr(const char *s) { uint64_t h = 1469598103934665603ULL; while (*s) h = (h ^ (uint8_t)*s++) * 1099511628211ULL; return h; }
static int vocab_id(const char *s) {
    uint64_t h = hstr(s);
    for (int i = h & (V.nslot - 1);; i = (i + 1) & (V.nslot - 1)) {
        int id = V.slot[i];
        if (id < 0) {
            if (V.n == V.cap) { V.cap *= 2; V.str = realloc(V.str, V.cap * sizeof *V.str); V.count = realloc(V.count, V.cap * sizeof *V.count); }
            V.str[V.n] = strdup(s); V.count[V.n] = 0; V.slot[i] = V.n;
            return V.n++;
        }
        if (!strcmp(V.str[id], s)) return id;
    }
}

static char *load(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1); if (fread(b, 1, n, f) != (size_t)n) exit(1); b[n] = 0; fclose(f);
    char *s = strstr(b, "*** START OF"); if (s) { s = strchr(s, '\n'); if (s) memmove(b, s, strlen(s) + 1); }
    char *e = strstr(b, "*** END OF"); if (e) *e = 0;
    *len = strlen(b); return b;
}

/* Tokens: runs of letters/apostrophes, lowercased. Clause ends: , ; : . ! ?
   Each token becomes a word id; -1 marks a clause end. */
static int *tokenize(const char *text, int *ntok, long *nwords) {
    int cap = 1 << 20, n = 0; int *t = malloc(cap * sizeof *t); char w[64]; int wl = 0; long words = 0;
    for (const char *p = text;; p++) {
        char c = *p;
        if (c >= 'A' && c <= 'Z') c += 32;
        if ((c >= 'a' && c <= 'z') || (c == '\'' && wl)) { if (wl < 63) w[wl++] = c; continue; }
        if (wl) { while (wl && w[wl - 1] == '\'') wl--; w[wl] = 0; if (wl) { if (n + 2 > cap) t = realloc(t, (cap *= 2) * sizeof *t); t[n++] = vocab_id(w); words++; } wl = 0; }
        if (c == ',' || c == ';' || c == ':' || c == '.' || c == '!' || c == '?') { if (n + 2 > cap) t = realloc(t, (cap *= 2) * sizeof *t); t[n++] = -1; }
        if (!c) break;
    }
    *ntok = n; *nwords = words; return t;
}

static uint8_t *is_function;   /* by word id: top 150 of the stored book */
static int cmp_count_desc(const void *a, const void *b) { return V.count[*(const int *)b] - V.count[*(const int *)a]; }

static Event *make_events(const int *t, int n, int *nev) {
    Event *ev = malloc((n / 3 + 16) * sizeof *ev); int ne = 0; Event cur = {0};
    for (int i = 0; i <= n; i++) {
        if (i == n || t[i] < 0) { if (cur.n >= 3) ev[ne++] = cur; cur.n = 0; continue; }
        int w = t[i]; if (is_function[w] || cur.n == MAXW) continue;
        int dup = 0; for (int j = 0; j < cur.n; j++) dup |= cur.w[j] == w;
        if (!dup) cur.w[cur.n++] = w;
    }
    *nev = ne; return ev;
}

/* ------------------------------------------------------------------ */
/* Shared pieces: synapse (8 B, same record in both specs), axons      */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t to; int16_t w; uint8_t type, trace; } Syn;
typedef struct { Syn *s; uint32_t n, cap; } Axon;   /* block in the arena; bytes counted = cap × 8 */
static size_t syn_bytes;

static void ax_push(Axon *a, Syn s) {
    if (a->n == a->cap) { uint32_t c = a->cap ? a->cap * 2 : 4; syn_bytes += (size_t)(c - a->cap) * 8; a->s = realloc(a->s, c * sizeof(Syn)); a->cap = c; }
    a->s[a->n++] = s;
}
static void ax_free(Axon *a) { syn_bytes -= (size_t)a->cap * 8; free(a->s); a->s = 0; a->n = a->cap = 0; }
static void ax_shrink(Axon *a) {
    uint32_t c = 4; while (c < a->n) c *= 2;
    if (!a->n) { ax_free(a); return; }
    if (c < a->cap) { syn_bytes -= (size_t)(a->cap - c) * 8; a->s = realloc(a->s, c * sizeof(Syn)); a->cap = c; }
}
/* capped add: when full, drop the weakest, oldest first */
static void ax_push_capped(Axon *a, Syn s, uint32_t capmax) {
    if (a->n >= capmax) {
        uint32_t k = 0; for (uint32_t i = 1; i < a->n; i++) if (a->s[i].w < a->s[k].w) k = i;
        memmove(a->s + k, a->s + k + 1, (a->n - k - 1) * sizeof(Syn)); a->n--;
    }
    ax_push(a, s);
}

/* sparse accumulator */
typedef struct { int32_t *v; uint32_t *list; uint8_t *on; int n, cap; } Acc;
static void acc_grow(Acc *a, int cap) { if (cap <= a->cap) return; a->v = realloc(a->v, cap * 4); a->list = realloc(a->list, cap * 4); a->on = realloc(a->on, cap); memset(a->v + a->cap, 0, (cap - a->cap) * 4); memset(a->on + a->cap, 0, cap - a->cap); a->cap = cap; }
static inline void acc_add(Acc *a, uint32_t i, int32_t x) { if (!a->on[i]) { a->on[i] = 1; a->list[a->n++] = i; } a->v[i] += x; }
static void acc_clear(Acc *a) { for (int i = 0; i < a->n; i++) { a->v[a->list[i]] = 0; a->on[a->list[i]] = 0; } a->n = 0; }

/* ------------------------------------------------------------------ */
/* v0.1: neuron = one thing, with content bits (NEURON.md 0f6a119)     */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t label, syn; uint16_t nsyn, capsyn; uint8_t kind, state, maturity, flags; uint32_t born, last, uses, value; } Meta1; /* 32 B */
enum { K_WORD = 1, K_EVENT = 2 };
enum { S_FREE = 0, S_NEW = 1 };
enum { T_PART = 1, T_CUE, T_NEXT, T_FF, T_BACK, T_REC, T_ORD };

typedef struct {
    int D, W, C; size_t budget;
    Meta1 *meta; int16_t *act; uint64_t *content; Axon *ax; int nn, maxn;
    int *word_neuron; int nwordn;      /* label table: 8 B per word neuron counted */
    int *freel; int nfree;
    uint32_t tick; int prev; Acc acc;
    uint64_t *tmp;
} B1;

static size_t b1_bytes(B1 *b) { return (size_t)b->nn * (32 + 2 + b->D / 8) + (size_t)b->nwordn * 8 + syn_bytes; }

static int b1_new(B1 *b, int kind) {
    int i;
    if (b->nfree) i = b->freel[--b->nfree];
    else {
        if (b->nn == b->maxn) {
            int m = b->maxn * 2; b->meta = realloc(b->meta, m * sizeof(Meta1)); b->act = realloc(b->act, m * 2);
            b->content = realloc(b->content, (size_t)m * b->W * 8); b->ax = realloc(b->ax, m * sizeof(Axon));
            memset(b->ax + b->maxn, 0, (m - b->maxn) * sizeof(Axon)); b->freel = realloc(b->freel, m * sizeof(int)); b->maxn = m;
        }
        i = b->nn++;
    }
    memset(&b->meta[i], 0, sizeof(Meta1)); b->meta[i].kind = kind; b->meta[i].state = S_NEW; b->meta[i].born = b->tick; b->act[i] = 0;
    acc_grow(&b->acc, b->maxn);
    return i;
}

static int b1_word(B1 *b, int w, int create) {
    if (b->word_neuron[w] >= 0) return b->word_neuron[w];
    if (!create) return -1;
    int i = b1_new(b, K_WORD); b->meta[i].value = w; b->nwordn++;
    for (int k = 0; k < b->W; k++) b->content[(size_t)i * b->W + k] = mix((uint64_t)w * 1000003 + k);
    return b->word_neuron[w] = i;
}

static void bundle(B1 *b, const int *nid, int n, uint64_t seed, uint64_t *out) {
    for (int k = 0; k < b->W; k++) {
        uint64_t r = 0;
        for (int bit = 0; bit < 64; bit++) {
            int c = 0;
            for (int j = 0; j < n; j++) c += (b->content[(size_t)nid[j] * b->W + k] >> bit) & 1;
            int v = 2 * c - n; if (!v) v = (mix(seed + k * 64 + bit) & 1) ? 1 : -1;
            r |= (uint64_t)(v > 0) << bit;
        }
        out[k] = r;
    }
}
static int ham(B1 *b, const uint64_t *x, int i) { int d = 0; const uint64_t *y = b->content + (size_t)i * b->W; for (int k = 0; k < b->W; k++) d += __builtin_popcountll(x[k] ^ y[k]); return d; }

/* candidates through cue synapses of the words; returns best by (hits, closeness, recency) */
static int b1_best(B1 *b, const int *w, int n, const uint64_t *code, int minhits, int *bestham) {
    for (int i = 0; i < n; i++) { int wn = b1_word(b, w[i], 0); if (wn < 0) continue; Axon *a = &b->ax[wn];
        for (uint32_t s = 0; s < a->n; s++) acc_add(&b->acc, a->s[s].to, 1); }
    int best = -1, bh = 0, bd = 1 << 30;
    for (int t = 0; t < b->acc.n; t++) {
        int e = b->acc.list[t], h = b->acc.v[e]; if (h < minhits) continue;
        int d = ham(b, code, e);
        if (best < 0 || h > bh || (h == bh && (d < bd || (d == bd && b->meta[e].born > b->meta[best].born)))) { best = e; bh = h; bd = d; }
    }
    acc_clear(&b->acc); if (bestham) *bestham = bd; return best;
}

typedef struct { uint64_t k; int id; } PruneKey;
static int prune_cmp(const void *x, const void *y) { uint64_t a = ((const PruneKey *)x)->k, c = ((const PruneKey *)y)->k; return a < c ? -1 : a > c; }

static void b1_prune(B1 *b) {
    /* least used, then oldest; 10% of the events at a time */
    PruneKey *p = malloc(b->nn * sizeof(PruneKey)); int nl = 0;
    for (int i = 0; i < b->nn; i++) if (b->meta[i].state != S_FREE && b->meta[i].kind == K_EVENT) { p[nl].k = ((uint64_t)b->meta[i].uses << 32) | b->meta[i].born; p[nl++].id = i; }
    qsort(p, nl, sizeof(PruneKey), prune_cmp);
    int nk = nl / 10 + 1; if (nk > nl) nk = nl;
    for (int j = 0; j < nk; j++) { int e = p[j].id; b->meta[e].state = S_FREE; ax_free(&b->ax[e]); b->freel[b->nfree++] = e; if (b->prev == e) b->prev = -1; }
    /* sweep: synapses to freed neurons go */
    for (int i = 0; i < b->nn; i++) { if (b->meta[i].state == S_FREE) continue; Axon *a = &b->ax[i]; uint32_t k = 0;
        for (uint32_t s = 0; s < a->n; s++) if (b->meta[a->s[s].to].state != S_FREE) a->s[k++] = a->s[s];
        if (k != a->n) { a->n = k; ax_shrink(a); } }
    free(p);
}

static B1 *b1_make(int setting, size_t budget, int nvocab) {
    B1 *b = calloc(1, sizeof *b); b->D = (setting & 1) ? 1024 : 256; b->W = b->D / 64; b->C = (setting & 2) ? 512 : 64; b->budget = budget;
    b->maxn = 1024; b->meta = malloc(b->maxn * sizeof(Meta1)); b->act = malloc(b->maxn * 2); b->content = malloc((size_t)b->maxn * b->W * 8);
    b->ax = calloc(b->maxn, sizeof(Axon)); b->freel = malloc(b->maxn * sizeof(int));
    b->word_neuron = malloc(nvocab * sizeof(int)); for (int i = 0; i < nvocab; i++) b->word_neuron[i] = -1;
    b->prev = -1; b->tmp = malloc(b->W * 8 * 2); return b;
}

static void b1_store(B1 *b, const Event *e) {
    b->tick++;
    int nid[MAXW]; for (int i = 0; i < e->n; i++) nid[i] = b1_word(b, e->w[i], 1);
    uint64_t *code = b->tmp; bundle(b, nid, e->n, b->tick, code);
    int d, best = b1_best(b, e->w, e->n, code, (e->n + 1) / 2, &d);
    int ev;
    if (best >= 0 && d < b->D * 15 / 100) {     /* recognised: τ = 0.15 */
        ev = best; b->meta[ev].uses++; b->meta[ev].last = b->tick;
        for (int i = 0; i < e->n; i++) { Axon *a = &b->ax[nid[i]]; for (uint32_t s = 0; s < a->n; s++) if (a->s[s].to == (uint32_t)ev && a->s[s].w < 32767) a->s[s].w++; }
    } else {
        ev = b1_new(b, K_EVENT); b->meta[ev].uses = 1; b->meta[ev].last = b->tick;
        memcpy(b->content + (size_t)ev * b->W, code, b->W * 8);
        for (int i = 0; i < e->n; i++) { ax_push(&b->ax[ev], (Syn){ (uint32_t)nid[i], 1, T_PART, 0 }); ax_push_capped(&b->ax[nid[i]], (Syn){ (uint32_t)ev, 1, T_CUE, 0 }, b->C); }
    }
    if (b->prev >= 0 && b->prev != ev) {
        Axon *a = &b->ax[b->prev]; int found = 0;
        for (uint32_t s = 0; s < a->n; s++) if (a->s[s].type == T_NEXT && a->s[s].to == (uint32_t)ev) { if (a->s[s].w < 32767) a->s[s].w++; found = 1; }
        if (!found) ax_push(a, (Syn){ (uint32_t)ev, 1, T_NEXT, 0 });
    }
    b->prev = ev;
    while (b1_bytes(b) > b->budget) { size_t was = b1_bytes(b); b1_prune(b); if (b1_bytes(b) >= was) break; }
}

static int b1_out(B1 *b, int ev, int *out) { int n = 0; Axon *a = &b->ax[ev]; for (uint32_t s = 0; s < a->n; s++) if (a->s[s].type == T_PART) out[n++] = b->meta[a->s[s].to].value; return n; }

static int b1_find(B1 *b, const int *w, int n, int *d) {
    int nid[MAXW], m = 0; for (int i = 0; i < n; i++) { int x = b1_word(b, w[i], 0); if (x >= 0) nid[m++] = x; }
    if (!m) { *d = b->D; return -1; }
    bundle(b, nid, m, 7, b->tmp);
    return b1_best(b, w, n, b->tmp, 1, d);
}
static int b1_recall(B1 *b, const int *cue, int n, int *out) { int d, ev = b1_find(b, cue, n, &d); return ev < 0 ? 0 : b1_out(b, ev, out); }
static double b1_familiar(B1 *b, const int *w, int n) { int d, ev = b1_find(b, w, n, &d); return ev < 0 ? 0 : 1.0 - (double)d / b->D; }
static int b1_next(B1 *b, const int *w, int n, int *out) {
    int d, ev = b1_find(b, w, n, &d); if (ev < 0) return 0;
    Axon *a = &b->ax[ev]; int best = -1, bw = 0;
    for (uint32_t s = 0; s < a->n; s++) if (a->s[s].type == T_NEXT && a->s[s].w > bw) { bw = a->s[s].w; best = a->s[s].to; }
    return best < 0 ? 0 : b1_out(b, best, out);
}

/* ------------------------------------------------------------------ */
/* v0.2: neurons hold state only; knowledge in synapses (cf9e806)      */
/* ------------------------------------------------------------------ */

typedef struct { int16_t act, theta; uint32_t last, born, uses, syn; uint16_t nsyn; uint8_t area, life; } Neuron2; /* 24 B */
enum { A_WORD = 1, A_EVENT = 2 };
#define NW2 32768          /* word area size (fixed device setting) */
#define AXCAP2 512         /* outgoing synapses per neuron (device setting) */

typedef struct {
    int kw, ke; size_t budget;
    Neuron2 *n; Axon *ax; int nn, maxn; int *freel; int nfree;
    uint32_t tick; int prev[16]; int nprev;
    Acc acc, rec;
    /* electrode: word-area neuron → words (the test's read-out, not the brain's memory) */
    int *el_start, *el_word; uint32_t *stamp; uint32_t stampv; int *votes; int nvocab;
} B2;

static size_t b2_bytes(B2 *b) { return (size_t)b->nn * 24 + syn_bytes; }

static void b2_members(B2 *b, int w, int *m) {
    int c = 0;
    for (uint64_t r = 0; c < b->kw; r++) { int x = mix(((uint64_t)w << 20) + r) % NW2; int dup = 0; for (int j = 0; j < c; j++) dup |= m[j] == x; if (!dup) m[c++] = x; }
}

static B2 *b2_make(int setting, size_t budget, int nvocab) {
    B2 *b = calloc(1, sizeof *b); b->kw = (setting & 1) ? 32 : 16; b->ke = (setting & 2) ? 16 : 8; b->budget = budget;
    b->maxn = NW2 + 4096; b->n = calloc(b->maxn, sizeof(Neuron2)); b->ax = calloc(b->maxn, sizeof(Axon)); b->freel = malloc(b->maxn * sizeof(int));
    b->nn = NW2; for (int i = 0; i < NW2; i++) { b->n[i].area = A_WORD; b->n[i].life = S_NEW; }
    acc_grow(&b->acc, b->maxn); acc_grow(&b->rec, b->maxn);
    /* electrode */
    b->nvocab = nvocab; int *cnt = calloc(NW2 + 1, sizeof(int)); int m[64];
    for (int w = 0; w < nvocab; w++) { b2_members(b, w, m); for (int j = 0; j < b->kw; j++) cnt[m[j] + 1]++; }
    for (int i = 0; i < NW2; i++) cnt[i + 1] += cnt[i];
    b->el_start = cnt; b->el_word = malloc((size_t)nvocab * b->kw * sizeof(int)); int *fill = calloc(NW2, sizeof(int));
    for (int w = 0; w < nvocab; w++) { b2_members(b, w, m); for (int j = 0; j < b->kw; j++) b->el_word[cnt[m[j]] + fill[m[j]]++] = w; }
    free(fill); b->stamp = calloc(NW2, 4); b->votes = calloc(nvocab, sizeof(int));
    return b;
}

static int b2_new(B2 *b) {
    int i;
    if (b->nfree) i = b->freel[--b->nfree];
    else {
        if (b->nn == b->maxn) { int m = b->maxn * 2; b->n = realloc(b->n, m * sizeof(Neuron2)); b->ax = realloc(b->ax, m * sizeof(Axon)); memset(b->ax + b->maxn, 0, (m - b->maxn) * sizeof(Axon));
            b->freel = realloc(b->freel, m * sizeof(int)); b->maxn = m; acc_grow(&b->acc, m); acc_grow(&b->rec, m); }
        i = b->nn++;
    }
    memset(&b->n[i], 0, sizeof(Neuron2)); b->n[i].area = A_EVENT; b->n[i].life = S_NEW; b->n[i].born = b->tick; return i;
}

/* k winners among touched event neurons by score(i) = acc(i) (+ rec), ties: later born */
static int b2_winners(B2 *b, Acc *a, Acc *r, int *win) {
    int k = 0; int32_t sc[16];
    for (int t = 0; t < a->n; t++) {
        uint32_t i = a->list[t]; if (b->n[i].area != A_EVENT) continue;
        int32_t s = a->v[i] * 4 + (r ? r->v[i] : 0);
        if (k == b->ke && (s < sc[k - 1] || (s == sc[k - 1] && b->n[i].born <= b->n[win[k - 1]].born))) continue;
        int p = k < b->ke ? k++ : k - 1;
        while (p > 0 && (sc[p - 1] < s || (sc[p - 1] == s && b->n[win[p - 1]].born < b->n[i].born))) { sc[p] = sc[p - 1]; win[p] = win[p - 1]; p--; }
        sc[p] = s; win[p] = i;
    }
    if (r) for (int t = 0; t < r->n; t++) {   /* neurons reached only by recurrence */
        uint32_t i = r->list[t]; if (a->on[i]) continue;
        int32_t s = r->v[i];
        if (k == b->ke && (s < sc[k - 1] || (s == sc[k - 1] && b->n[i].born <= b->n[win[k - 1]].born))) continue;
        int p = k < b->ke ? k++ : k - 1;
        while (p > 0 && (sc[p - 1] < s || (sc[p - 1] == s && b->n[win[p - 1]].born < b->n[i].born))) { sc[p] = sc[p - 1]; win[p] = win[p - 1]; p--; }
        sc[p] = s; win[p] = i;
    }
    return k;
}

static void b2_spread_words(B2 *b, const int *w, int n) {
    int m[64];
    for (int i = 0; i < n; i++) { b2_members(b, w[i], m);
        for (int j = 0; j < b->kw; j++) { Axon *a = &b->ax[m[j]]; for (uint32_t s = 0; s < a->n; s++) acc_add(&b->acc, a->s[s].to, a->s[s].w); } }
}

/* pattern completion: winners excite their assembly; 2 rounds */
static int b2_complete(B2 *b, int *win, int k) {
    for (int round = 0; round < 2 && k; round++) {
        acc_clear(&b->rec);
        for (int j = 0; j < k; j++) { Axon *a = &b->ax[win[j]]; for (uint32_t s = 0; s < a->n; s++) if (a->s[s].type == T_REC) acc_add(&b->rec, a->s[s].to, a->s[s].w * 4); }
        k = b2_winners(b, &b->acc, &b->rec, win);
    }
    acc_clear(&b->rec); return k;
}

static void b2_prune(B2 *b) {
    PruneKey *p = malloc((b->nn - NW2) * sizeof(PruneKey)); int nl = 0;
    for (int i = NW2; i < b->nn; i++) if (b->n[i].life != S_FREE) { p[nl].k = ((uint64_t)b->n[i].uses << 32) | b->n[i].born; p[nl++].id = i; }
    qsort(p, nl, sizeof(PruneKey), prune_cmp);
    int nk = nl / 10 + 1; if (nk > nl) nk = nl;
    for (int j = 0; j < nk; j++) { int e = p[j].id; b->n[e].life = S_FREE; ax_free(&b->ax[e]); b->freel[b->nfree++] = e; }
    for (int j = 0; j < b->nprev; j++) if (b->n[b->prev[j]].life == S_FREE) { b->nprev = 0; break; }
    /* night sweep: synapses into dead neurons are removed */
    for (int i = 0; i < b->nn; i++) { if (b->n[i].life == S_FREE) continue; Axon *a = &b->ax[i]; uint32_t k = 0;
        for (uint32_t s = 0; s < a->n; s++) if (b->n[a->s[s].to].life != S_FREE) a->s[k++] = a->s[s];
        if (k != a->n) { a->n = k; ax_shrink(a); } }
    free(p);
}

static void b2_store(B2 *b, const Event *e) {
    b->tick++;
    int win[16], k, n = e->n;
    b2_spread_words(b, e->w, n);
    k = b2_winners(b, &b->acc, 0, win);
    int recog = k == b->ke;
    for (int j = 0; j < k && recog; j++) { int i = win[j]; int th = b->n[i].theta; if (b->acc.v[i] * 10 < th * 9 || th * 10 < 2 * n * 9) recog = 0; }
    int cur[16];
    if (recog) {
        for (int j = 0; j < k; j++) { b->n[win[j]].uses++; b->n[win[j]].last = b->tick; cur[j] = win[j]; }
    } else {
        k = b->ke; int m[64];
        for (int j = 0; j < k; j++) { cur[j] = b2_new(b); b->n[cur[j]].uses = 1; b->n[cur[j]].last = b->tick; b->n[cur[j]].theta = 2 * n; }
        for (int i = 0; i < n; i++) {
            b2_members(b, e->w[i], m);
            for (int j = 0; j < k; j++) {
                for (int r = 0; r < 2; r++) { int src = m[mix(((uint64_t)b->tick << 16) + i * 64 + j * 2 + r) % b->kw];   /* fan-in: 2 per word */
                    ax_push_capped(&b->ax[src], (Syn){ (uint32_t)cur[j], 1, T_FF, 0 }, AXCAP2); }
                for (int r = 0; r < 2; r++) ax_push(&b->ax[cur[j]], (Syn){ (uint32_t)m[(2 * j + r) % b->kw], 1, T_BACK, 0 });   /* back: 2 per word */
            }
        }
        for (int j = 0; j < k; j++) for (int q = 0; q < k; q++) if (q != j) ax_push(&b->ax[cur[j]], (Syn){ (uint32_t)cur[q], 1, T_REC, 0 });
    }
    /* order: each neuron of the event before → 2 neurons of this one */
    for (int j = 0; j < b->nprev; j++) for (int r = 0; r < 2; r++) {
        int to = cur[mix(((uint64_t)b->tick << 8) + j * 2 + r) % k]; Axon *a = &b->ax[b->prev[j]]; int found = 0;
        for (uint32_t s = 0; s < a->n; s++) if (a->s[s].type == T_ORD && a->s[s].to == (uint32_t)to) { if (a->s[s].w < 32767) a->s[s].w++; found = 1; break; }
        if (!found) ax_push_capped(a, (Syn){ (uint32_t)to, 1, T_ORD, 0 }, AXCAP2);
    }
    memcpy(b->prev, cur, k * sizeof(int)); b->nprev = k;
    acc_clear(&b->acc);
    while (b2_bytes(b) > b->budget) { size_t was = b2_bytes(b); b2_prune(b); if (b2_bytes(b) >= was) break; }
}

/* read-out: winners → back-projections → word area; words whose assemblies are ≥ half as covered as a full event covers them */
static int b2_readout(B2 *b, const int *win, int k, int *out) {
    b->stampv++; int touched[4096], nt = 0, nw[4096 * 4], nnw = 0;
    for (int j = 0; j < k; j++) { Axon *a = &b->ax[win[j]]; for (uint32_t s = 0; s < a->n; s++) if (a->s[s].type == T_BACK) {
        uint32_t x = a->s[s].to; if (b->stamp[x] == b->stampv) continue; b->stamp[x] = b->stampv; if (nt < 4096) touched[nt++] = x; } }
    for (int t = 0; t < nt; t++) for (int q = b->el_start[touched[t]]; q < b->el_start[touched[t] + 1]; q++) { int w = b->el_word[q]; if (!b->votes[w]++ && nnw < 4096 * 4) nw[nnw++] = w; }
    int full = 2 * b->ke < b->kw ? 2 * b->ke : b->kw, n = 0;
    for (int t = 0; t < nnw; t++) { int w = nw[t]; if (2 * b->votes[w] >= full && n < MAXW * 2) out[n++] = w; b->votes[w] = 0; }
    return n;
}

static int b2_find(B2 *b, const int *w, int n, int *win) {
    b2_spread_words(b, w, n);
    int k = b2_winners(b, &b->acc, 0, win);
    k = b2_complete(b, win, k);
    return k;
}
static int b2_recall(B2 *b, const int *cue, int n, int *out) { int win[16]; int k = b2_find(b, cue, n, win); acc_clear(&b->acc); return b2_readout(b, win, k, out); }
static double b2_familiar(B2 *b, const int *w, int n) {
    int win[16]; b2_spread_words(b, w, n); int k = b2_winners(b, &b->acc, 0, win); double f = 0;
    for (int j = 0; j < k; j++) { int th = b->n[win[j]].theta; int den = th > 2 * n ? th : 2 * n; f += (double)b->acc.v[win[j]] / (den ? den : 1); }
    acc_clear(&b->acc); return k ? f / b->ke : 0;
}
static int b2_next(B2 *b, const int *w, int n, int *out) {
    int win[16]; int k = b2_find(b, w, n, win); acc_clear(&b->acc); if (!k) return 0;
    for (int j = 0; j < k; j++) { Axon *a = &b->ax[win[j]]; for (uint32_t s = 0; s < a->n; s++) if (a->s[s].type == T_ORD) acc_add(&b->acc, a->s[s].to, a->s[s].w); }
    int nx[16]; k = b2_winners(b, &b->acc, 0, nx); k = b2_complete(b, nx, k); acc_clear(&b->acc);
    return b2_readout(b, nx, k, out);
}


/* ------------------------------------------------------------------ */
/* v0.1-lean: the winner cut down. Same behaviour, fewer bytes:         */
/*  - no content bits: recognise by synapse overlap (Jaccard ≥ 0.85)    */
/*  - word neuron = its vocabulary number: no meta, no label table      */
/*  - event = 8 B: start, n, uses, next as a signed step                */
/*  - part-of synapse = 3 B (word number), cue synapse = 4 B            */
/*    (24-bit event number + 8-bit strength)                            */
/*  - pruning compacts at night (ids stay in birth order)               */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t start; uint8_t n, uses; int16_t next; } Ev3;   /* 8 B */
typedef struct { uint32_t *v; uint32_t n, cap; } Cue3;                  /* header counted as 8 B (offset + count) */
typedef struct {
    int C; size_t budget;
    Ev3 *ev; int nev, evcap;
    uint8_t *pool; size_t npool, poolcap;
    Cue3 *cue; int nvocab; int nlists; size_t cuebytes;
    int prev; Acc acc;
} B3;

#define CHUNK 4096
static size_t up(size_t x) { return (x + CHUNK - 1) / CHUNK * CHUNK; }
static size_t b3_bytes(B3 *b) { return up(b->nev) * 8 + up(b->npool * 3) + (size_t)b->nlists * 8 + b->cuebytes; }
static inline int pw(B3 *b, size_t i) { const uint8_t *p = b->pool + i * 3; return p[0] | p[1] << 8 | p[2] << 16; }

static B3 *b3_make(size_t budget, int nvocab) {
    B3 *b = calloc(1, sizeof *b); b->C = 512; b->budget = budget; b->nvocab = nvocab;
    b->evcap = 1024; b->ev = malloc(b->evcap * sizeof(Ev3)); b->poolcap = 4096; b->pool = malloc(b->poolcap * 3);
    b->cue = calloc(nvocab, sizeof(Cue3)); b->prev = -1; acc_grow(&b->acc, b->evcap); return b;
}

static void cue_push(B3 *b, Cue3 *c, uint32_t x) {
    if (c->n >= (uint32_t)b->C) { uint32_t k = 0; for (uint32_t i = 1; i < c->n; i++) if ((c->v[i] & 255) < (c->v[k] & 255)) k = i;
        memmove(c->v + k, c->v + k + 1, (c->n - k - 1) * 4); c->n--; }
    if (c->n == c->cap) { uint32_t nc = c->cap ? c->cap + (c->cap >> 1) + 3 : 4; if (!c->cap) b->nlists++; b->cuebytes += (size_t)(nc - c->cap) * 4; c->v = realloc(c->v, nc * 4); c->cap = nc; }
    c->v[c->n++] = x;
}

/* best event for these words: most words shared, then fewest words (closest), then newest */
static int b3_best(B3 *b, const int *w, int n, int minhits, int *hits) {
    for (int i = 0; i < n; i++) { Cue3 *c = &b->cue[w[i]]; for (uint32_t s = 0; s < c->n; s++) acc_add(&b->acc, c->v[s] >> 8, 1); }
    int best = -1, bh = 0;
    for (int t = 0; t < b->acc.n; t++) { int e = b->acc.list[t], h = b->acc.v[e]; if (h < minhits) continue;
        if (best < 0 || h > bh || (h == bh && (b->ev[e].n < b->ev[best].n || (b->ev[e].n == b->ev[best].n && e > best)))) { best = e; bh = h; } }
    acc_clear(&b->acc); *hits = bh; return best;
}

static void b3_prune(B3 *b) {
    PruneKey *p = malloc(b->nev * sizeof(PruneKey));
    for (int i = 0; i < b->nev; i++) { p[i].k = ((uint64_t)b->ev[i].uses << 32) | (uint32_t)i; p[i].id = i; }
    qsort(p, b->nev, sizeof(PruneKey), prune_cmp);
    int nk = b->nev / 10 + 1; if (nk > b->nev) nk = b->nev;
    int *map = malloc(b->nev * sizeof(int)); for (int i = 0; i < b->nev; i++) map[i] = 0;
    for (int j = 0; j < nk; j++) map[p[j].id] = -1;
    int m = 0; for (int i = 0; i < b->nev; i++) if (!map[i]) map[i] = m++; else map[i] = -1;
    /* night: compact events and the word pool, renumber cues and next steps */
    Ev3 *ne = malloc((m + 1) * sizeof(Ev3)); uint8_t *np = malloc(b->npool * 3 + 3); size_t q = 0;
    for (int i = 0; i < b->nev; i++) { if (map[i] < 0) continue; Ev3 e = b->ev[i];
        memcpy(np + q * 3, b->pool + (size_t)e.start * 3, e.n * 3); e.start = q; q += e.n;
        int to = e.next ? i + e.next : -1; int nt = to >= 0 && to < b->nev ? map[to] : -1;
        e.next = nt >= 0 && nt - map[i] >= -32768 && nt - map[i] <= 32767 ? nt - map[i] : 0;
        ne[map[i]] = e; }
    free(b->ev); free(b->pool); b->ev = ne; b->evcap = m + 1; b->nev = m; b->pool = np; b->poolcap = b->npool + 1; b->npool = q;
    for (int w = 0; w < b->nvocab; w++) { Cue3 *c = &b->cue[w]; uint32_t k = 0;
        for (uint32_t s = 0; s < c->n; s++) { int t = map[c->v[s] >> 8]; if (t >= 0) c->v[k++] = (uint32_t)t << 8 | (c->v[s] & 255); }
        c->n = k;
        uint32_t nc = k ? (k + 3) & ~3u : 0;   /* night: give back unused room */
        if (nc < c->cap) { b->cuebytes -= (size_t)(c->cap - nc) * 4; if (!nc) { free(c->v); c->v = 0; b->nlists--; } else c->v = realloc(c->v, nc * 4); c->cap = nc; } }
    if (b->prev >= 0) b->prev = map[b->prev];
    free(map); free(p);
}

static void b3_store(B3 *b, const Event *e) {
    int h, best = b3_best(b, e->w, e->n, (e->n + 1) / 2, &h), ev;
    if (best >= 0 && h * 100 >= 85 * (e->n + b->ev[best].n - h)) {   /* recognised: Jaccard ≥ 0.85 */
        ev = best; if (b->ev[ev].uses < 255) b->ev[ev].uses++;
        for (int i = 0; i < e->n; i++) { Cue3 *c = &b->cue[e->w[i]]; for (uint32_t s = 0; s < c->n; s++) if ((int)(c->v[s] >> 8) == ev && (c->v[s] & 255) < 255) c->v[s]++; }
    } else {
        if (b->nev == b->evcap) { b->evcap *= 2; b->ev = realloc(b->ev, b->evcap * sizeof(Ev3)); acc_grow(&b->acc, b->evcap); }
        if (b->npool + MAXW > b->poolcap) { b->poolcap *= 2; b->pool = realloc(b->pool, b->poolcap * 3); }
        ev = b->nev++; b->ev[ev] = (Ev3){ (uint32_t)b->npool, (uint8_t)e->n, 1, 0 };
        for (int i = 0; i < e->n; i++) { uint8_t *p = b->pool + (b->npool++) * 3; p[0] = e->w[i]; p[1] = e->w[i] >> 8; p[2] = e->w[i] >> 16;
            cue_push(b, &b->cue[e->w[i]], (uint32_t)ev << 8 | 1); }
    }
    if (b->prev >= 0 && b->prev != ev && !b->ev[b->prev].next) { int d = ev - b->prev; if (d >= -32768 && d <= 32767) b->ev[b->prev].next = d; }
    b->prev = ev;
    while (b3_bytes(b) > b->budget && b->nev) { size_t was = b3_bytes(b); b3_prune(b); if (b3_bytes(b) >= was) break; }
}
static int b3_out(B3 *b, int ev, int *out) { for (int i = 0; i < b->ev[ev].n; i++) out[i] = pw(b, b->ev[ev].start + i); return b->ev[ev].n; }
static int b3_recall(B3 *b, const int *c, int n, int *out) { int h, ev = b3_best(b, c, n, 1, &h); return ev < 0 ? 0 : b3_out(b, ev, out); }
static double b3_familiar(B3 *b, const int *w, int n) { int h, ev = b3_best(b, w, n, 1, &h); return ev < 0 ? 0 : (double)h / (n + b->ev[ev].n - h); }
static int b3_next(B3 *b, const int *w, int n, int *out) { int h, ev = b3_best(b, w, n, 1, &h); if (ev < 0 || !b->ev[ev].next) return 0; return b3_out(b, ev + b->ev[ev].next, out); }

/* ------------------------------------------------------------------ */
/* v0.1-lean2: lean, cut again.                                         */
/*  - event = 1 byte: words (4 bits), uses as a log counter (2 bits),   */
/*    next (2 bits: none / the event after / in the exception table)    */
/*  - where an event's words start: one u32 per 64 events               */
/*  - word numbers in the pool as varints (1–3 B)                       */
/*  - cue synapses: gaps between event numbers as varints, no strength  */
/*    (when a list is over its cap, the least used, oldest events go)   */
/*  - lists grow in quarter steps (4,5,6,7,8,10,12,14,16,20…)           */
/* ------------------------------------------------------------------ */

typedef struct { uint8_t *b; uint32_t off, last; uint16_t nb, cnt; } Cue4;   /* counted 12 B: offset, last, bytes, count */
typedef struct {
    int C; size_t budget;
    uint8_t *ev; int nev, evcap; uint32_t *chk;           /* chk[i] = pool offset of event 64·i */
    uint8_t *pool; size_t npool, poolcap;
    Cue4 *cue; int nvocab, nlists; size_t cuebytes;
    uint32_t *xk, *xv; int xn, xcap;                      /* next exceptions */
    int prev; uint32_t tick; Acc acc;
    int bitpool;                                          /* lean4: pool as exp-Golomb (k 11) bits; npool counts bits */
} B4;

static uint32_t qcap(uint32_t n) { if (n <= 4) return 4; uint32_t p = 4; while (p * 2 <= n) p *= 2; uint32_t q = p / 4; return (n + q - 1) / q * q; }
static inline int vput(uint8_t *p, uint32_t v) { int n = 0; while (v >= 128) { p[n++] = (v & 127) | 128; v >>= 7; } p[n++] = v; return n; }
static inline uint32_t vget(const uint8_t *p, size_t *i) { uint32_t v = 0; int sh = 0; for (;;) { uint8_t c = p[(*i)++]; v |= (uint32_t)(c & 127) << sh; if (c < 128) return v; sh += 7; } }
static inline uint32_t eg_get(const uint8_t *buf, uint32_t *pos, int k, uint32_t nbytes);
static inline void eg_put(uint8_t *buf, uint32_t *pos, uint32_t x, int k);
#define POOLK 11
static inline size_t pool_bytes(B4 *b) { return b->bitpool ? (b->npool + 7) / 8 : b->npool; }
static inline uint32_t pool_get(B4 *b, size_t *i) { if (!b->bitpool) return vget(b->pool, i); uint32_t p = *i, v = eg_get(b->pool, &p, POOLK, (uint32_t)pool_bytes(b)); *i = p; return v; }
static inline void pool_put(B4 *b, uint32_t w) { if (!b->bitpool) { b->npool += vput(b->pool + b->npool, w); return; } uint32_t p = b->npool; eg_put(b->pool, &p, w, POOLK); b->npool = p; }
static size_t b4_bytes(B4 *b) { return up(b->nev) + up((b->nev / 64 + 1) * 4) + up(pool_bytes(b)) + (size_t)b->nlists * 12 + b->cuebytes + (size_t)b->xcap * 8; }
#define EVN(x) ((x) & 15)
#define EVU(x) (((x) >> 4) & 3)
#define EVX(x) ((x) >> 6)


static void x_put(B4 *b, uint32_t k, uint32_t v);
static int x_get(B4 *b, uint32_t k) { if (!b->xcap) return -1; for (uint32_t i = mix(k) & (b->xcap - 1);; i = (i + 1) & (b->xcap - 1)) { if (b->xk[i] == 0xffffffffu) return -1; if (b->xk[i] == k) return b->xv[i]; } }
static void x_put(B4 *b, uint32_t k, uint32_t v) {
    if ((b->xn + 1) * 4 > b->xcap * 3) { int oc = b->xcap; uint32_t *ok = b->xk, *ov = b->xv; b->xcap = oc ? oc * 2 : 64;
        b->xk = malloc(b->xcap * 4); b->xv = malloc(b->xcap * 4); memset(b->xk, 255, b->xcap * 4); b->xn = 0;
        for (int i = 0; i < oc; i++) if (ok[i] != 0xffffffffu) x_put(b, ok[i], ov[i]); free(ok); free(ov); }
    for (uint32_t i = mix(k) & (b->xcap - 1);; i = (i + 1) & (b->xcap - 1)) { if (b->xk[i] == 0xffffffffu) { b->xk[i] = k; b->xv[i] = v; b->xn++; return; } if (b->xk[i] == k) { b->xv[i] = v; return; } }
}

static B4 *b4_make(size_t budget, int nvocab) {
    B4 *b = calloc(1, sizeof *b); b->C = 512; b->budget = budget; b->nvocab = nvocab;
    b->evcap = 1024; b->ev = malloc(b->evcap); b->chk = malloc((b->evcap / 64 + 1) * 4); b->poolcap = 1 << 16; b->pool = calloc(b->poolcap, 1);
    b->cue = calloc(nvocab, sizeof(Cue4)); b->prev = -1; acc_grow(&b->acc, b->evcap); return b;
}

static void cue4_set(B4 *b, Cue4 *c, const uint8_t *src, uint32_t nb) {
    uint32_t oc = c->nb ? qcap(c->nb) : 0, nc = nb ? qcap(nb) : 0;
    if (nc != oc) { b->cuebytes += nc; b->cuebytes -= oc; c->b = realloc(c->b, nc ? nc : 1); if (!oc && nc) b->nlists++; if (oc && !nc) b->nlists--; }
    if (src != c->b) memcpy(c->b, src, nb); c->nb = nb;
}
static void cue4_append(B4 *b, Cue4 *c, uint32_t e) {
    uint8_t tmp[8]; int k = vput(tmp, c->cnt ? e - c->last : e);
    uint32_t oc = c->nb ? qcap(c->nb) : 0, nc = qcap(c->nb + k);
    if (nc != oc) { b->cuebytes += nc - oc; c->b = realloc(c->b, nc); if (!oc) b->nlists++; }
    memcpy(c->b + c->nb, tmp, k); c->nb += k; c->last = e; c->cnt++;
}
static int b4_decode(Cue4 *c, uint32_t *out) { size_t i = 0; uint32_t e = 0; for (int j = 0; j < c->cnt; j++) { uint32_t d = vget(c->b, &i); e = j ? e + d : d; out[j] = e; } return c->cnt; }
static void cue4_encode(B4 *b, Cue4 *c, const uint32_t *ids, int n) {
    uint8_t *tmp = malloc(n * 5 + 1); uint32_t nb = 0;
    for (int j = 0; j < n; j++) nb += vput(tmp + nb, j ? ids[j] - ids[j - 1] : ids[j]);
    cue4_set(b, c, tmp, nb); c->cnt = n; c->last = n ? ids[n - 1] : 0; free(tmp);
}
/* over the cap by 32: drop the least used, oldest down to the cap */
static void cue4_trim(B4 *b, Cue4 *c) {
    uint32_t *ids = malloc(c->cnt * 4); int n = b4_decode(c, ids), drop = n - b->C;
    for (int u = 0; u < 4 && drop > 0; u++) for (int j = 0; j < n && drop > 0; j++) if (ids[j] != 0xffffffffu && EVU(b->ev[ids[j]]) == u) { ids[j] = 0xffffffffu; drop--; }
    int m = 0; for (int j = 0; j < n; j++) if (ids[j] != 0xffffffffu) ids[m++] = ids[j];
    cue4_encode(b, c, ids, m); free(ids);
}

static size_t b4_start(B4 *b, int e) { size_t i = b->chk[e / 64]; for (int k = e & ~63; k < e; k++) for (int j = 0; j < EVN(b->ev[k]); j++) pool_get(b, &i); return i; }

static int b4_best(B4 *b, const int *w, int n, int minhits, int *hits) {
    static uint32_t *ids; static int idcap; 
    for (int i = 0; i < n; i++) { Cue4 *c = &b->cue[w[i]]; if (c->cnt > idcap) { idcap = c->cnt * 2; ids = realloc(ids, idcap * 4); }
        int m = b4_decode(c, ids); for (int s = 0; s < m; s++) acc_add(&b->acc, ids[s], 1); }
    int best = -1, bh = 0;
    for (int t = 0; t < b->acc.n; t++) { int e = b->acc.list[t], h = b->acc.v[e]; if (h < minhits) continue;
        if (best < 0 || h > bh || (h == bh && (EVN(b->ev[e]) < EVN(b->ev[best]) || (EVN(b->ev[e]) == EVN(b->ev[best]) && e > best)))) { best = e; bh = h; } }
    acc_clear(&b->acc); *hits = bh; return best;
}

static int b4_nextof(B4 *b, int e) { int x = EVX(b->ev[e]); return x == 0 ? -1 : x == 1 ? e + 1 : x_get(b, e); }

static void b4_prune(B4 *b) {
    PruneKey *p = malloc(b->nev * sizeof(PruneKey));
    for (int i = 0; i < b->nev; i++) { p[i].k = ((uint64_t)EVU(b->ev[i]) << 32) | (uint32_t)i; p[i].id = i; }
    qsort(p, b->nev, sizeof(PruneKey), prune_cmp);
    int nk = b->nev / 10 + 1; if (nk > b->nev) nk = b->nev;
    int *map = calloc(b->nev, sizeof(int)); for (int j = 0; j < nk; j++) map[p[j].id] = -1;
    int m = 0; for (int i = 0; i < b->nev; i++) map[i] = map[i] ? -1 : m++;
    int *nxt = malloc(b->nev * sizeof(int)); for (int i = 0; i < b->nev; i++) nxt[i] = b4_nextof(b, i);
    /* night: rewrite events, pool, next, cues in the new numbering */
    uint8_t *ne = malloc(b->evcap), *np = calloc(b->poolcap, 1), *op = b->pool; uint32_t *nc = malloc((b->evcap / 64 + 1) * 4); size_t r = 0, oldn = b->npool; b->npool = 0;
    uint32_t *ok = b->xk, *ov = b->xv; int oc = b->xcap; b->xk = b->xv = 0; b->xcap = b->xn = 0; free(ok); free(ov); (void)oc;
    for (int i = 0; i < b->nev; i++) {
        int wd[MAXW], nw = EVN(b->ev[i]); b->pool = op; size_t cur = b->npool; b->npool = oldn;
        for (int j = 0; j < nw; j++) wd[j] = pool_get(b, &r);
        b->pool = np; b->npool = cur;
        if (map[i] < 0) continue; int me = map[i];
        if (!(me & 63)) nc[me / 64] = b->npool;
        for (int j = 0; j < nw; j++) pool_put(b, wd[j]);
        int t = nxt[i] >= 0 ? map[nxt[i]] : -1, x = t < 0 ? 0 : t == me + 1 ? 1 : 2;
        ne[me] = (b->ev[i] & 63) | x << 6; if (x == 2) x_put(b, me, t);
    }
    free(b->ev); free(op); free(b->chk); b->ev = ne; b->pool = np; b->chk = nc; b->nev = m;
    uint32_t *ids = malloc((b->C + 64) * 4);
    for (int w = 0; w < b->nvocab; w++) { Cue4 *c = &b->cue[w]; if (!c->cnt) continue; int n = b4_decode(c, ids), k = 0;
        for (int j = 0; j < n; j++) if (map[ids[j]] >= 0) ids[k++] = map[ids[j]]; cue4_encode(b, c, ids, k); }
    if (b->prev >= 0) b->prev = map[b->prev];
    free(ids); free(map); free(nxt); free(p);
}

static void b4_store(B4 *b, const Event *e) {
    b->tick++;
    int h, best = b4_best(b, e->w, e->n, (e->n + 1) / 2, &h), ev;
    if (best >= 0 && h * 100 >= 85 * (e->n + EVN(b->ev[best]) - h)) {   /* recognised: Jaccard ≥ 0.85 */
        ev = best; int u = EVU(b->ev[ev]);
        if (u < 3 && !(mix(((uint64_t)b->tick << 24) ^ ev) & ((1u << u) - 1))) b->ev[ev] += 16;   /* log counter: 1, 2, 4, 8+ */
    } else {
        if (b->nev == b->evcap) { b->evcap *= 2; b->ev = realloc(b->ev, b->evcap); b->chk = realloc(b->chk, (b->evcap / 64 + 1) * 4); acc_grow(&b->acc, b->evcap); }
        if (pool_bytes(b) + MAXW * 8 > b->poolcap) { b->pool = realloc(b->pool, b->poolcap * 2); memset(b->pool + b->poolcap, 0, b->poolcap); b->poolcap *= 2; }
        ev = b->nev++; if (!(ev & 63)) b->chk[ev / 64] = b->npool;
        b->ev[ev] = e->n;
        for (int i = 0; i < e->n; i++) { pool_put(b, e->w[i]);
            Cue4 *c = &b->cue[e->w[i]]; cue4_append(b, c, ev); if (c->cnt > b->C + 32) cue4_trim(b, c); }
    }
    if (b->prev >= 0 && b->prev != ev && !EVX(b->ev[b->prev])) { if (ev == b->prev + 1) b->ev[b->prev] |= 1 << 6; else { b->ev[b->prev] |= 2 << 6; x_put(b, b->prev, ev); } }
    b->prev = ev;
    while (b4_bytes(b) > b->budget && b->nev) { size_t was = b4_bytes(b); b4_prune(b); if (b4_bytes(b) >= was) break; }
}
static int b4_out(B4 *b, int ev, int *out) { size_t i = b4_start(b, ev); int n = EVN(b->ev[ev]); for (int j = 0; j < n; j++) out[j] = pool_get(b, &i); return n; }
static int b4_recall(B4 *b, const int *c, int n, int *out) { int h, ev = b4_best(b, c, n, 1, &h); return ev < 0 ? 0 : b4_out(b, ev, out); }
static double b4_familiar(B4 *b, const int *w, int n) { int h, ev = b4_best(b, w, n, 1, &h); return ev < 0 ? 0 : (double)h / (n + EVN(b->ev[ev]) - h); }
static int b4_next(B4 *b, const int *w, int n, int *out) { int h, ev = b4_best(b, w, n, 1, &h); if (ev < 0) return 0; int t = b4_nextof(b, ev); return t < 0 || t >= b->nev ? 0 : b4_out(b, t, out); }

/* ------------------------------------------------------------------ */
/* v0.1-lean3: lean2 with smaller cue lists.                            */
/*  - header 7 B: arena offset, bits used, code parameter k             */
/*  - a list of ≤ 32 bits lives inside the header (words seen once)     */
/*  - gaps as exp-Golomb codes of order k, bit-packed; k re-chosen per  */
/*    word whenever its list is rewritten (cap trims, nights)           */
/* ------------------------------------------------------------------ */

typedef struct { union { uint8_t *p; uint8_t in[8]; } u; uint16_t nbits; uint8_t k; } Cue5;
typedef struct { B4 base; Cue5 *cue; int nlists; size_t cuebytes; } B5;   /* events, pool, next as lean2 */

static inline uint8_t *c5_buf(Cue5 *c) { return c->nbits <= 32 ? c->u.in : c->u.p; }
static size_t b5_bytes(B5 *b) { B4 *a = &b->base; return up(a->nev) + up((a->nev / 64 + 1) * 4) + up(pool_bytes(a)) + (size_t)b->nlists * 7 + b->cuebytes + (size_t)a->xcap * 8; }
static inline void bput(uint8_t *buf, uint32_t *pos, uint32_t v, int n) { for (int i = n - 1; i >= 0; i--, (*pos)++) { if ((v >> i) & 1) buf[*pos >> 3] |= 1 << (*pos & 7); else buf[*pos >> 3] &= ~(1 << (*pos & 7)); } }
static inline uint32_t bget(const uint8_t *buf, uint32_t *pos, int n) { uint32_t v = 0; for (int i = 0; i < n; i++, (*pos)++) v = v << 1 | ((buf[*pos >> 3] >> (*pos & 7)) & 1); return v; }
static inline int eg_len(uint32_t x, int k) { uint64_t v = (uint64_t)x + (1u << k); int nb = 63 - __builtin_clzll(v); return 2 * nb - k + 1; }
/* code: (nb − k) zeros, a one, then the low nb bits of v least significant first (v = x + 2^k, nb = its top bit) */
static inline void eg_put(uint8_t *buf, uint32_t *pos, uint32_t x, int k) { uint64_t v = (uint64_t)x + (1u << k); int nb = 63 - __builtin_clzll(v);
    for (int i = 0; i < nb - k; i++, (*pos)++) buf[*pos >> 3] &= ~(1 << (*pos & 7));
    buf[*pos >> 3] |= 1 << (*pos & 7); (*pos)++;
    for (int i = 0; i < nb; i++, (*pos)++) { if ((v >> i) & 1) buf[*pos >> 3] |= 1 << (*pos & 7); else buf[*pos >> 3] &= ~(1 << (*pos & 7)); } }
static inline uint32_t eg_get(const uint8_t *buf, uint32_t *pos, int k, uint32_t nbytes) {
    uint64_t w = 0; uint32_t b0 = *pos >> 3, avail = nbytes - b0; memcpy(&w, buf + b0, avail < 8 ? avail : 8); w >>= *pos & 7;
    int z = __builtin_ctzll(w), nb = z + k; uint32_t v = (uint32_t)(((w >> (z + 1)) & ((1ull << nb) - 1)) | (1ull << nb));
    *pos += z + 1 + nb; return v - (1u << k); }

static void c5_resize(B5 *b, Cue5 *c, uint32_t nbits) {   /* keeps contents */
    uint32_t ob = c->nbits <= 32 ? 0 : qcap((c->nbits + 7) / 8), nb = nbits <= 32 ? 0 : qcap((nbits + 7) / 8);
    if (!c->nbits && nbits) b->nlists++; if (c->nbits && !nbits) b->nlists--;
    if (ob == nb) { c->nbits = nbits; return; }
    uint8_t tmp[8]; if (!ob) memcpy(tmp, c->u.in, 8);
    if (!nb) { uint8_t *old = c->u.p; memcpy(c->u.in, old, 4); free(old); }
    else if (!ob) { c->u.p = calloc(nb, 1); memcpy(c->u.p, tmp, 4); }
    else { c->u.p = realloc(c->u.p, nb); if (nb > ob) memset(c->u.p + ob, 0, nb - ob); }
    b->cuebytes += nb; b->cuebytes -= ob; c->nbits = nbits;
}
static int c5_decode(Cue5 *c, uint32_t *out, uint32_t *last) {
    const uint8_t *buf = c5_buf(c); uint32_t pos = 0, e = 0; int n = 0;
    uint32_t nby = (c->nbits + 7) / 8; while (pos < c->nbits) { uint32_t g = eg_get(buf, &pos, c->k, nby); e = n ? e + g + 1 : g; if (out) out[n] = e; n++; }
    if (last) *last = e; return n;
}
static void c5_encode(B5 *b, Cue5 *c, const uint32_t *ids, int n) {
    int k = 0; if (n) { double mean = n > 1 ? (double)(ids[n - 1] - ids[0]) / (n - 1) : ids[0] + 1.0; while (k < 20 && (double)(2u << k) <= mean) k++; }
    uint32_t bits = 0; for (int j = 0; j < n; j++) bits += eg_len(j ? ids[j] - ids[j - 1] - 1 : ids[j], k);
    c5_resize(b, c, 0); c->k = k; c5_resize(b, c, bits);
    uint8_t *buf = c5_buf(c); uint32_t pos = 0; for (int j = 0; j < n; j++) eg_put(buf, &pos, j ? ids[j] - ids[j - 1] - 1 : ids[j], k);
}
static void c5_append(B5 *b, Cue5 *c, uint32_t e, uint32_t last, int cnt) {
    if (!cnt) { c5_encode(b, c, &e, 1); return; }
    uint32_t x = e - last - 1, pos = c->nbits; c5_resize(b, c, c->nbits + eg_len(x, c->k)); eg_put(c5_buf(c), &pos, x, c->k);
}

static uint32_t *c5_ids; static int c5_idcap;
static void c5_room(int n) { if (n > c5_idcap) { c5_idcap = n * 2; c5_ids = realloc(c5_ids, c5_idcap * 4); } }

static int b5_best(B5 *b, const int *w, int n, int minhits, int *hits, int *cnt, uint32_t *last) {
    B4 *a = &b->base;
    for (int i = 0; i < n; i++) { Cue5 *c = &b->cue[w[i]]; c5_room(b->base.C + 64); int m = c5_decode(c, c5_ids, last ? &last[i] : 0); if (cnt) cnt[i] = m;
        for (int s = 0; s < m; s++) acc_add(&a->acc, c5_ids[s], 1); }
    int best = -1, bh = 0;
    for (int t = 0; t < a->acc.n; t++) { int e = a->acc.list[t], h = a->acc.v[e]; if (h < minhits) continue;
        if (best < 0 || h > bh || (h == bh && (EVN(a->ev[e]) < EVN(a->ev[best]) || (EVN(a->ev[e]) == EVN(a->ev[best]) && e > best)))) { best = e; bh = h; } }
    acc_clear(&a->acc); *hits = bh; return best;
}
static void c5_trim(B5 *b, Cue5 *c) {
    B4 *a = &b->base; c5_room(a->C + 64); int n = c5_decode(c, c5_ids, 0), drop = n - a->C;
    for (int u = 0; u < 4 && drop > 0; u++) for (int j = 0; j < n && drop > 0; j++) if (c5_ids[j] != 0xffffffffu && EVU(a->ev[c5_ids[j]]) == u) { c5_ids[j] = 0xffffffffu; drop--; }
    int m = 0; for (int j = 0; j < n; j++) if (c5_ids[j] != 0xffffffffu) c5_ids[m++] = c5_ids[j];
    c5_encode(b, c, c5_ids, m);
}

static void b5_prune(B5 *b) {
    /* events, pool and next exactly as lean2 (its cue lists are empty), then renumber our cues */
    B4 *a = &b->base; int old = a->nev;
    PruneKey *p = malloc(old * sizeof(PruneKey));
    for (int i = 0; i < old; i++) { p[i].k = ((uint64_t)EVU(a->ev[i]) << 32) | (uint32_t)i; p[i].id = i; }
    qsort(p, old, sizeof(PruneKey), prune_cmp);
    int nk = old / 10 + 1; if (nk > old) nk = old;
    int *map = calloc(old, sizeof(int)); for (int j = 0; j < nk; j++) map[p[j].id] = -1;
    int m = 0; for (int i = 0; i < old; i++) map[i] = map[i] ? -1 : m++;
    b4_prune(a);   /* same choice: (uses, id) order, same 10% */
    c5_room(a->C + 64);
    for (int w = 0; w < a->nvocab; w++) { Cue5 *c = &b->cue[w]; if (!c->nbits) continue; int n = c5_decode(c, c5_ids, 0), k = 0;
        for (int j = 0; j < n; j++) if (map[c5_ids[j]] >= 0) c5_ids[k++] = map[c5_ids[j]]; c5_encode(b, c, c5_ids, k); }
    free(map); free(p);
}

static B5 *b5_make(size_t budget, int nvocab) { B5 *b = calloc(1, sizeof *b); B4 *a = b4_make(budget, nvocab); b->base = *a; free(a); b->cue = calloc(nvocab, sizeof(Cue5)); return b; }

static void b5_store(B5 *b, const Event *e) {
    B4 *a = &b->base; a->tick++;
    int cnt[MAXW]; uint32_t last[MAXW];
    int h, best = b5_best(b, e->w, e->n, (e->n + 1) / 2, &h, cnt, last), ev;
    if (best >= 0 && h * 100 >= 85 * (e->n + EVN(a->ev[best]) - h)) {
        ev = best; int u = EVU(a->ev[ev]);
        if (u < 3 && !(mix(((uint64_t)a->tick << 24) ^ ev) & ((1u << u) - 1))) a->ev[ev] += 16;
    } else {
        if (a->nev == a->evcap) { a->evcap *= 2; a->ev = realloc(a->ev, a->evcap); a->chk = realloc(a->chk, (a->evcap / 64 + 1) * 4); acc_grow(&a->acc, a->evcap); }
        if (pool_bytes(a) + MAXW * 8 > a->poolcap) { a->pool = realloc(a->pool, a->poolcap * 2); memset(a->pool + a->poolcap, 0, a->poolcap); a->poolcap *= 2; }
        ev = a->nev++; if (!(ev & 63)) a->chk[ev / 64] = a->npool;
        a->ev[ev] = e->n;
        for (int i = 0; i < e->n; i++) { pool_put(a, e->w[i]);
            Cue5 *c = &b->cue[e->w[i]]; c5_append(b, c, ev, last[i], cnt[i]); if (cnt[i] + 1 > a->C + 32) c5_trim(b, c); }
    }
    if (a->prev >= 0 && a->prev != ev && !EVX(a->ev[a->prev])) { if (ev == a->prev + 1) a->ev[a->prev] |= 1 << 6; else { a->ev[a->prev] |= 2 << 6; x_put(a, a->prev, ev); } }
    a->prev = ev;
    while (b5_bytes(b) > a->budget && a->nev) { size_t was = b5_bytes(b); b5_prune(b); if (b5_bytes(b) >= was) break; }
}
static int b5_recall(B5 *b, const int *c, int n, int *out) { int h, ev = b5_best(b, c, n, 1, &h, 0, 0); return ev < 0 ? 0 : b4_out(&b->base, ev, out); }
static double b5_familiar(B5 *b, const int *w, int n) { int h, ev = b5_best(b, w, n, 1, &h, 0, 0); return ev < 0 ? 0 : (double)h / (n + EVN(b->base.ev[ev]) - h); }
static int b5_next(B5 *b, const int *w, int n, int *out) { int h, ev = b5_best(b, w, n, 1, &h, 0, 0); if (ev < 0) return 0; int t = b4_nextof(&b->base, ev); return t < 0 || t >= b->base.nev ? 0 : b4_out(&b->base, t, out); }

/* ------------------------------------------------------------------ */
/* Tests (identical for both)                                           */
/* ------------------------------------------------------------------ */

static int design; static B1 *g1; static B2 *g2; static B3 *g3; static B4 *g4; static B5 *g5;
static void store(const Event *e) { if (design == 1) b1_store(g1, e); else if (design == 3) b3_store(g3, e); else if (design == 4) b4_store(g4, e); else if (design >= 5) b5_store(g5, e); else b2_store(g2, e); }
static int recall(const int *c, int n, int *o) { return design == 1 ? b1_recall(g1, c, n, o) : design == 4 ? b4_recall(g4, c, n, o) : (design == 5 || design == 6) ? b5_recall(g5, c, n, o) : design == 3 ? b3_recall(g3, c, n, o) : b2_recall(g2, c, n, o); }
static double familiar(const int *w, int n) { return design == 1 ? b1_familiar(g1, w, n) : design == 4 ? b4_familiar(g4, w, n) : (design == 5 || design == 6) ? b5_familiar(g5, w, n) : design == 3 ? b3_familiar(g3, w, n) : b2_familiar(g2, w, n); }
static int next(const int *w, int n, int *o) { return design == 1 ? b1_next(g1, w, n, o) : design == 4 ? b4_next(g4, w, n, o) : (design == 5 || design == 6) ? b5_next(g5, w, n, o) : design == 3 ? b3_next(g3, w, n, o) : b2_next(g2, w, n, o); }
static size_t bytes(void) { return design == 1 ? b1_bytes(g1) : design == 4 ? b4_bytes(g4) : (design == 5 || design == 6) ? b5_bytes(g5) : design == 3 ? b3_bytes(g3) : b2_bytes(g2); }
static int alive(void) { int c = 0; if (design == 3) return g3->nev; if (design == 4) return g4->nev; if (design >= 5) return g5->base.nev; if (design == 1) { for (int i = 0; i < g1->nn; i++) c += g1->meta[i].kind == K_EVENT && g1->meta[i].state != S_FREE; }
    else { for (int i = NW2; i < g2->nn; i++) c += g2->n[i].life != S_FREE; c /= g2->ke; } return c; }

static double f1(const int *a, int na, const int *b, int nb) {
    if (!na || !nb) return 0; int hit = 0; for (int i = 0; i < na; i++) for (int j = 0; j < nb; j++) if (a[i] == b[j]) { hit++; break; }
    if (!hit) return 0; double p = (double)hit / na, r = (double)hit / nb; return 2 * p * r / (p + r);
}

static double t_recall(const Event *ev, int lo, int hi, int samples, double *secs) {
    double s = 0; int out[MAXW * 2]; double t0 = now();
    for (int q = 0; q < samples; q++) { const Event *e = &ev[lo + (long)(hi - lo) * q / samples]; int nc = (e->n + 1) / 2; if (nc < 2) nc = 2;
        int no = recall(e->w, nc, out); s += f1(out, no, e->w, e->n); }
    if (secs) *secs = now() - t0; return s / samples;
}

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double t_recognise(const Event *ev, int ne, const Event *nov, int nn, int samples) {
    double *pos = malloc(samples * 8), *neg = malloc(samples * 8);
    for (int q = 0; q < samples; q++) { const Event *e = &ev[(long)ne * q / samples]; pos[q] = familiar(e->w, e->n); }
    for (int q = 0; q < samples; q++) { const Event *e = &nov[(long)nn * q / samples]; neg[q] = familiar(e->w, e->n); }
    /* AUC = P(pos > neg) + ½ P(equal) */
    qsort(neg, samples, 8, cmpd); double a = 0;
    for (int q = 0; q < samples; q++) { int lo = 0, hi = samples; while (lo < hi) { int m = (lo + hi) / 2; if (neg[m] < pos[q]) lo = m + 1; else hi = m; }
        int below = lo; int eq = 0; while (lo + eq < samples && neg[lo + eq] == pos[q]) eq++; a += below + 0.5 * eq; }
    free(pos); free(neg); return a / ((double)samples * samples);
}

static double t_next(const Event *ev, int ne, int samples) {
    double s = 0; int out[MAXW * 2];
    for (int q = 0; q < samples; q++) { int i = (long)(ne - 1) * q / samples; int no = next(ev[i].w, ev[i].n, out); s += f1(out, no, ev[i + 1].w, ev[i + 1].n); }
    return s / samples;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "compare DESIGN SETTING BUDGET_MB BOOK NOVEL_BOOK\n"); return 1; }
    design = atoi(argv[1]); int setting = atoi(argv[2]); double mb = atof(argv[3]);
    size_t budget = (size_t)(mb * 1024 * 1024);
    V.cap = 1 << 16; V.str = malloc(V.cap * sizeof *V.str); V.count = malloc(V.cap * sizeof *V.count); V.nslot = 1 << 20; V.slot = malloc(V.nslot * sizeof(int)); memset(V.slot, -1, V.nslot * sizeof(int));

    size_t len; char *book = load(argv[4], &len), *nbook = load(argv[5], &len);
    int nt, nnt; long words, nwords;
    int *t = tokenize(book, &nt, &words);
    for (int i = 0; i < nt; i++) if (t[i] >= 0) V.count[t[i]]++;
    int nbase = V.n; int *byc = malloc(nbase * sizeof(int)); for (int i = 0; i < nbase; i++) byc[i] = i; qsort(byc, nbase, sizeof(int), cmp_count_desc);
    int *nt2 = tokenize(nbook, &nnt, &nwords);
    is_function = calloc(V.n, 1); for (int i = 0; i < 150 && i < nbase; i++) is_function[byc[i]] = 1;
    int ne, nne; Event *ev = make_events(t, nt, &ne), *nov = make_events(nt2, nnt, &nne);

    syn_bytes = 0;
    if (design == 1) g1 = b1_make(setting, budget, V.n); else if (design == 3) g3 = b3_make(budget, V.n); else if (design == 4) g4 = b4_make(budget, V.n); else if (design == 5 || design == 6) { g5 = b5_make(budget, V.n); g5->base.bitpool = design == 6; } else g2 = b2_make(setting, budget, V.n);

    double t0 = now();
    for (int i = 0; i < ne; i++) store(&ev[i]);
    double ts = now() - t0;

    int S = 2000; double rs;
    double T1 = t_recall(ev, 0, ne, S, &rs);
    double T2 = t_recall(ev, 0, ne / 10, S, 0);
    double T3 = t_recognise(ev, ne, nov, nne, S);
    double T4 = t_next(ev, ne, S);
    printf("design=%s setting=%d budget=%.0fMB events=%d tokens=%ld T1=%.4f T2=%.4f T3=%.4f T4=%.4f store_tok_s=%.0f recall_s=%.0f bytes=%zu stored=%d\n",
           design == 1 ? "v0.1" : design == 2 ? "v0.2" : design == 3 ? "lean" : design == 4 ? "lean2" : design == 5 ? "lean3" : "lean4", setting, mb, ne, words, T1, T2, T3, T4, words / ts, S / rs, bytes(), alive());
    return 0;
}
