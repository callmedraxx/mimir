/*
 * ctxwin.c — a context window built from lean4 neurons (B) against today's (A).
 * Rules fixed before this was written: docs/CONTEXT_COMPARE.md.
 *
 *   A  today's: every token and the page kept (src/text.c); the gist, when
 *      asked, cuts the whole context into events and runs construction-
 *      integration over them (src/gist.c). Used as it is.
 *   B  neurons: text streams through (tokens and the page are not kept); each
 *      event becomes a lean4 event neuron (at most 12 content words) and the
 *      gist runs while reading, on parts that grow and shrink:
 *        workspace  event → activation         the 4 carried events      8 B
 *        meaning    event → words + int8 meaning  while in mind or a     4+1+48+dims B
 *                                              candidate; gone 16 events after last use
 *        strength   event → strength, born, last touched   while it gathers strength;
 *                                              400 events untouched → offered to the gist  16 B
 *        gist       event → strength, born     the strongest 64           12 B
 *      Cued recall takes the last 8 events of each word's cue list (A: the
 *      last 8 events per word in its index). Links, spreading, carrying 4,
 *      recalling 4, fairness over 400: the same as A.
 *
 *   ctxwin gist                   32 Grimm tales, each retold in 60 content words, A and B
 *   ctxwin book BOOK BUDGET_MB    take in a book and give its gist: speed, memory,
 *                                 recall, find, quote, who it is about
 *   Run from the repo root (data/meaning, data/gist).
 */
#define vocab_id text_vocab_id
#include "../../src/text.c"
#undef vocab_id

/* Shared by A and B: a word's meaning norm worked out once, not on every use (same numbers). */
static float *inv_norm;
static void meaning_norms(const Meaning *m) {
    inv_norm = malloc((size_t)m->n * sizeof *inv_norm);
    for (int i = 0; i < m->n; i++) { const int8_t *v = m->vec + (size_t)i * m->dims; float n = 0;
        for (int d = 0; d < m->dims; d++) n += (float)v[d] * v[d]; inv_norm[i] = 1.0f / (sqrtf(n) + 1e-9f); }
}
static inline void meaning_add_id(const Meaning *m, int id, float *acc) {
    const int8_t *v = m->vec + (size_t)id * m->dims; float n = inv_norm[id];
    for (int d = 0; d < m->dims; d++) acc[d] += v[d] * n;
}
static int fast_meaning_add(const Meaning *m, const char *w, float *acc) { int id = meaning_id(m, w); if (id < 0) return 0; meaning_add_id(m, id, acc); return 1; }
#define meaning_add fast_meaning_add
#define Event GEvent
#define make_events g_make_events
#define free_events g_free_events
#define link g_link
#define cmp_u32 g_cmp_u32
#include "../../src/gist.c"
#undef Event
#undef make_events
#undef free_events
#undef link
#undef cmp_u32
#undef meaning_add
#define GROW_NO_MAIN
#include "grow.c"

#define BUDGET_CW 60          /* content words in a retelling, both */
#define GIST_K 64             /* events B keeps as its gist */
#define MEAN_LIFE 16          /* events a meaning part outlives its last use */
#define STREAM_BYTES 16384    /* B takes text in pieces of about this, cut at a blank line */

/* ------------------------------------------------------------------ */
/* B: the neuron context window                                        */
/* ------------------------------------------------------------------ */

typedef struct __attribute__((packed)) { uint32_t last; uint8_t n; uint32_t w[MAXW]; } MeanHead;   /* then int8 v[dims] */
typedef struct __attribute__((packed)) { float s; uint32_t born, touched, pad; } StrV;
typedef struct __attribute__((packed)) { float s; uint32_t born, pad; } GistV;

typedef struct {
    B5 *b; const Meaning *m; const TextContext *t; int dims;
    Part ws, mean, str, gist;
    uint32_t cycle;                       /* events read */
    size_t peak, stream_bytes;            /* bytes: peak of everything B holds; its intake buffers now */
    int *map, nmap, capmap;               /* test side only (not counted): event j of the text → neuron */
    int32_t *mid; uint32_t nmid;          /* word → its meaning (−1: none); 4 B a word, counted */
} CW;

typedef struct { int n; uint32_t w[MAXW]; int8_t v[MEANING_MAX_DIMS]; } Node;

static size_t cw_parts(CW *c) { return part_bytes(&c->ws) + part_bytes(&c->mean) + part_bytes(&c->str) + part_bytes(&c->gist); }
static size_t cw_bytes(CW *c) { return b5_bytes(c->b) + cw_parts(c) + c->stream_bytes + (size_t)c->nmid * 4; }
static int cw_mid(CW *c, uint32_t w) {
    if (w >= c->nmid) { uint32_t n = c->t->nvoc > w + 1 ? c->t->nvoc : w + 1; c->mid = realloc(c->mid, n * 4); for (uint32_t i = c->nmid; i < n; i++) c->mid[i] = meaning_id(c->m, c->t->voc[i].word); c->nmid = n; }
    return c->mid[w];
}

/* an event's words and meaning: from its meaning part, growing it if it has none */
static long node_calls, node_miss;
static void cw_node(CW *c, int ev, Node *o) {
    MeanHead *h = (MeanHead *)part_get(&c->mean, ev); node_calls++;
    if (!h) { node_miss++;
        int w[MAXW], n = b4_out(&c->b->base, ev, w);
        float acc[MEANING_MAX_DIMS] = {0};
        for (int i = 0; i < n; i++) { int id = cw_mid(c, w[i]); if (id >= 0) meaning_add_id(c->m, id, acc); }
        float nn = 0; for (int d = 0; d < c->dims; d++) nn += acc[d] * acc[d];
        nn = nn > 0 ? 127.0f / sqrtf(nn) : 0;
        h = (MeanHead *)part_grow(&c->mean, ev); h->n = n; for (int i = 0; i < n; i++) h->w[i] = w[i];
        int8_t *v = (int8_t *)(h + 1); for (int d = 0; d < c->dims; d++) { float x = acc[d] * nn; v[d] = (int8_t)(x >= 0 ? (int)(x + 0.5f) : -(int)(-x + 0.5f)); }
    }
    h->last = c->cycle; o->n = h->n; memcpy(o->w, h->w, sizeof o->w); memcpy(o->v, h + 1, c->dims);
}
static float cw_link(const Node *x, const Node *y, int dims) {
    int share = 0; for (int i = 0; i < x->n; i++) for (int j = 0; j < y->n; j++) share += x->w[i] == y->w[j];
    int dot = 0; for (int d = 0; d < dims; d++) dot += x->v[d] * y->v[d];
    float cos = dot / (127.0f * 127.0f);
    return (float)share + 2.0f * (cos > 0 ? cos : 0);
}

static void cw_offer(CW *c, uint32_t ev, float s, uint32_t born) {   /* into the gist part if strong enough */
    #define FAIR(s, born) ((s) / (float)(c->cycle - (born) + 1 < GIST_HORIZON ? c->cycle - (born) + 1 : GIST_HORIZON))
    if (c->gist.n >= GIST_K) {
        int wi = -1; float wf = 0;
        for (int i = 0; i < c->gist.cap; i++) if (c->gist.key[i] != EMPTY) { GistV *g = (GistV *)(c->gist.val + (size_t)i * 12); float f = FAIR(g->s, g->born); if (wi < 0 || f < wf) { wf = f; wi = i; } }
        if (FAIR(s, born) <= wf) return;
        part_shrink(&c->gist, c->gist.key[wi]);
    }
    GistV *g = (GistV *)part_grow(&c->gist, ev); g->s = s; g->born = born;
}

static void cw_strength(CW *c, uint32_t ev, float a, int is_new) {
    StrV *v = (StrV *)part_get(&c->str, ev);
    if (!v) {
        GistV *g = (GistV *)part_get(&c->gist, ev); float s0 = 0; uint32_t born = c->cycle;
        if (g) { s0 = g->s; born = g->born; part_shrink(&c->gist, ev); }          /* back from the gist: keep gathering */
        else if (!is_new) born = c->cycle;                                           /* its strength was let go: starts again */
        v = (StrV *)part_grow(&c->str, ev); v->s = s0; v->born = born;
    }
    v->s += a; v->touched = c->cycle;
}

static void cw_tidy(CW *c, int all) {
    uint32_t *k = malloc(((size_t)c->str.cap + c->mean.cap + 1) * 4); int m = 0;
    if (all || !(c->cycle % MEAN_LIFE)) {
        for (int i = 0; i < c->mean.cap; i++) if (c->mean.key[i] != EMPTY && (all || ((MeanHead *)(c->mean.val + (size_t)i * c->mean.vsize))->last + MEAN_LIFE < c->cycle) && (all || !part_get(&c->ws, c->mean.key[i]))) k[m++] = c->mean.key[i];
        for (int i = 0; i < m; i++) part_shrink(&c->mean, k[i]);
    }
    if (all || !(c->cycle % 64)) {
        m = 0;
        for (int i = 0; i < c->str.cap; i++) if (c->str.key[i] != EMPTY && (all || ((StrV *)(c->str.val + (size_t)i * 16))->touched + GIST_HORIZON < c->cycle)) k[m++] = c->str.key[i];
        for (int i = 0; i < m; i++) { StrV v = *(StrV *)part_get(&c->str, k[i]); part_shrink(&c->str, k[i]); cw_offer(c, k[i], v.s, v.born); }
    }
    if (all) part_clear(&c->ws);
    free(k);
}

static int cw_keep(void *ctx, int e) { CW *c = ctx; return part_get(&c->gist, e) || part_get(&c->ws, e); }
static void cw_remap(void *ctx, const int *map, int old) {
    CW *c = ctx; Part *ts[4] = { &c->ws, &c->mean, &c->str, &c->gist };
    for (int t = 0; t < 4; t++) {
        Part *p = ts[t]; int n = p->n; if (!n) continue;
        uint32_t *k = malloc(n * 4); uint8_t *v = malloc((size_t)n * p->vsize); int m = 0;
        for (int i = 0; i < p->cap; i++) if (p->key[i] != EMPTY && (int)p->key[i] < old && map[p->key[i]] >= 0) {
            k[m] = map[p->key[i]]; memcpy(v + (size_t)m * p->vsize, p->val + (size_t)i * p->vsize, p->vsize); m++; }
        part_clear(p);
        for (int i = 0; i < m; i++) memcpy(part_grow(p, k[i]), v + (size_t)i * p->vsize, p->vsize);
        free(k); free(v);
    }
    for (int j = 0; j < c->nmap; j++) if (c->map[j] >= 0) c->map[j] = c->map[j] < old ? map[c->map[j]] : -1;
}

/* one event read: store it as a neuron, then one cycle of the gist on the neurons */
static void cw_event(CW *c, const uint32_t *w, int n) {
    c->cycle++;
    if (c->nmap == c->capmap) { c->capmap = c->capmap ? c->capmap * 2 : 4096; c->map = realloc(c->map, c->capmap * sizeof(int)); }
    c->map[c->nmap++] = -1;
    if (!n) return;                                    /* no content word: nothing a neuron can hold */
    Event e = { .n = n }; for (int i = 0; i < n; i++) e.w[i] = (int)w[i];
    int was = c->b->base.nev; b5_store(c->b, &e); int ev = c->b->base.prev, is_new = c->b->base.nev > was;
    c->map[c->nmap - 1] = ev;

    enum { MAXN = GIST_WORKSPACE + GIST_RECALL + 1 };
    static Node node[MAXN], cn; int id[MAXN], nn = 0;
    for (int i = 0; i < c->ws.cap; i++) if (c->ws.key[i] != EMPTY && (int)c->ws.key[i] != ev) id[nn++] = c->ws.key[i];
    int nc = nn;
    static Node me; cw_node(c, ev, &me);
    /* cued recall: the last GIST_INDEX_DEPTH events on each word's cue list */
    int cand[GIST_INDEX_DEPTH * 16], ncand = 0;
    for (int q = 0; q < n && ncand < GIST_INDEX_DEPTH * 16 - GIST_INDEX_DEPTH; q++) {
        Cue5 *cl = &c->b->cue[w[q]]; c5_room(c->b->base.C + 64); int m = c5_decode(cl, c5_ids, 0);
        for (int k = m - 1, took = 0; k >= 0 && took < GIST_INDEX_DEPTH; k--) {
            int j = c5_ids[k]; if (j == ev) continue; took++;
            int dup = 0; for (int x = 0; x < ncand; x++) dup |= cand[x] == j; for (int x = 0; x < nc; x++) dup |= id[x] == j;
            if (!dup) cand[ncand++] = j;
        }
    }
    float cl[GIST_INDEX_DEPTH * 16];
    for (int x = 0; x < ncand; x++) { cw_node(c, cand[x], &cn); cl[x] = cw_link(&me, &cn, c->dims); }
    for (int r = 0; r < GIST_RECALL && ncand; r++) {
        int bi = 0; float bl = -1; for (int x = 0; x < ncand; x++) if (cl[x] > bl) { bl = cl[x]; bi = x; }
        if (bl <= 0) break;
        id[nn++] = cand[bi]; cand[bi] = cand[--ncand]; cl[bi] = cl[ncand];
    }
    id[nn++] = ev;
    for (int x = 0; x < nn - 1; x++) cw_node(c, id[x], &node[x]);
    node[nn - 1] = me;
    float W[MAXN][MAXN], a[MAXN], bb[MAXN];
    for (int x = 0; x < nn; x++) { a[x] = 1.0f / nn; for (int y = 0; y < nn; y++) W[x][y] = x == y ? 0 : cw_link(&node[x], &node[y], c->dims); }
    for (int it = 0; it < 30; it++) {
        float sum = 0, diff = 0;
        for (int x = 0; x < nn; x++) { bb[x] = a[x]; for (int y = 0; y < nn; y++) bb[x] += W[x][y] * a[y]; sum += bb[x]; }
        for (int x = 0; x < nn; x++) { bb[x] /= sum + 1e-12f; diff = fmaxf(diff, fabsf(bb[x] - a[x])); a[x] = bb[x]; }
        if (diff < 1e-4f) break;
    }
    for (int x = 0; x < nn; x++) cw_strength(c, id[x], a[x], id[x] == ev && is_new);
    /* carry the GIST_WORKSPACE most active: their workspace parts; the rest shrink */
    int keep[GIST_WORKSPACE], nk = 0; float ka[GIST_WORKSPACE];
    for (int r = 0; r < GIST_WORKSPACE && r < nn; r++) { int bx = -1; for (int x = 0; x < nn; x++) if (a[x] >= 0 && (bx < 0 || a[x] > a[bx])) bx = x; keep[nk] = id[bx]; ka[nk++] = a[bx]; a[bx] = -1; }
    for (int i = 0; i < nc; i++) { int still = 0; for (int k = 0; k < nk; k++) still |= keep[k] == id[i]; if (!still) part_shrink(&c->ws, id[i]); }
    for (int k = 0; k < nk; k++) memcpy(part_grow(&c->ws, keep[k]), &ka[k], 4);
    cw_tidy(c, 0);
    size_t now_b = cw_bytes(c); if (now_b > c->peak) c->peak = now_b;
}

static int cw_content(const TextContext *t, const Meaning *m, uint32_t id) { return is_content(t, m, id); }

/* cut the stream's tokens into events by gist.c's rules; carry what is unfinished */
static void cw_cut(CW *c, TextContext *t, int final) {
    uint32_t start = 0;
    for (uint32_t i = 0; i < t->ntok; i++) {
        const TextToken *k = &t->tok[i]; const char *w = t->voc[k->word].word;
        int cut = k->end || (k->kind == TEXT_PUNCT && (w[0] == ';' || w[0] == ':')) ||
                  (k->kind == TEXT_PUNCT && w[0] == ',' && i + 1 < t->ntok && joins(t->voc[t->tok[i + 1].word].word));
        if (!cut && i + 1 < t->ntok) continue;
        while (cut && i + 1 < t->ntok && t->tok[i + 1].kind == TEXT_PUNCT &&
               (t->voc[t->tok[i + 1].word].word[0] == '\'' || t->voc[t->tok[i + 1].word].word[0] == '"') &&
               t->tok[i + 1].src == t->tok[i].src + t->tok[i].len) i++;
        uint32_t a = start, b = i + 1; start = i + 1;
        int words = 0; for (uint32_t j = a; j < b; j++) words += t->tok[j].kind == TEXT_WORD;
        if (words < 3) { start = a; if (i + 1 < t->ntok) continue; if (!final || !words) break; }
        uint32_t ew[MAXW]; int en = 0;
        for (uint32_t j = a; j < b && en < MAXW; j++) { uint32_t id = t->tok[j].word; if (!cw_content(t, c->m, id)) continue;
            int seen = 0; for (int q = 0; q < en; q++) seen |= ew[q] == id; if (!seen) ew[en++] = id; }
        cw_event(c, ew, en);
    }
    if (start > t->ntok) start = t->ntok;
    memmove(t->tok, t->tok + start, (t->ntok - start) * sizeof *t->tok); t->ntok -= start;
}

/* B takes in a text: pieces cut after a blank line, each forgotten once its events are neurons */
static void cw_take(CW *c, TextContext *t, const char *s, size_t n) {
    for (size_t p = 0; p < n;) {
        size_t q = p + STREAM_BYTES < n ? p + STREAM_BYTES : n;
        while (q < n && !(s[q - 1] == '\n' && (s[q - 2] == '\n' || (s[q - 2] == '\r' && q >= 3 && s[q - 3] == '\n')))) q++;
        if (t->src_len) { t->src_len = 0; }                       /* the page is not kept */
        text_take(t, NULL, s + p, q - p);
        size_t fold = (q - p) * 5 + 5;                             /* text_take's folding buffers, freed again */
        c->stream_bytes = t->tok_cap * sizeof(TextToken) + t->src_cap;
        if (cw_bytes(c) + fold > c->peak) c->peak = cw_bytes(c) + fold;
        cw_cut(c, t, q == n);
        p = q;
    }
}

static CW *cw_make(const Meaning *m, const TextContext *t, size_t budget) {
    CW *c = calloc(1, sizeof *c); c->m = m; c->t = t; c->dims = m->dims;
    c->b = b5_make(budget, 1 << 20); c->b->base.bitpool = 1;
    c->b->base.keep = cw_keep; c->b->base.remap = cw_remap; c->b->base.ctx = c;
    c->ws = (Part){ .vsize = 4 }; c->mean = (Part){ .vsize = (int)sizeof(MeanHead) + m->dims }; c->str = (Part){ .vsize = 16 }; c->gist = (Part){ .vsize = 12 };
    return c;
}

/* B's gist: strongest gist-part events up to the budget, in story order, as words */
static int cw_retell(CW *c, int budget, char *out, size_t cap, uint32_t *top, int ktop) {
    int n = 0; uint32_t ks[GIST_K]; float fs[GIST_K];
    for (int i = 0; i < c->gist.cap; i++) if (c->gist.key[i] != EMPTY) { GistV *g = (GistV *)(c->gist.val + (size_t)i * 12); ks[n] = c->gist.key[i]; fs[n++] = FAIR(g->s, g->born); }
    for (int i = 1; i < n; i++) for (int j = i; j > 0 && fs[j] > fs[j - 1]; j--) { float tf = fs[j]; fs[j] = fs[j - 1]; fs[j - 1] = tf; uint32_t tk = ks[j]; ks[j] = ks[j - 1]; ks[j - 1] = tk; }
    uint32_t ch[GIST_K]; int np = 0, used = 0;
    for (int k = 0; k < n && used < budget; k++) { int w = EVN(c->b->base.ev[ks[k]]); if (used + w > budget && used) continue; ch[np++] = ks[k]; used += w; }
    for (int i = 1; i < np; i++) for (int j = i; j > 0 && ch[j] < ch[j - 1]; j--) { uint32_t tk = ch[j]; ch[j] = ch[j - 1]; ch[j - 1] = tk; }
    size_t u = 0; out[0] = 0;
    for (int k = 0; k < np; k++) { int w[MAXW], m = b4_out(&c->b->base, ch[k], w);
        for (int i = 0; i < m; i++) u += snprintf(out + u, u < cap ? cap - u : 0, "%s ", c->t->voc[w[i]].word);
        if (u < cap) u += snprintf(out + u, cap - u, "/ "); }
    if (top) {   /* who: content words with the most gist strength behind them */
        int *wid = malloc(GIST_K * MAXW * 4); float *ws = malloc(GIST_K * MAXW * 4); int nw = 0;
        for (int k = 0; k < n; k++) { int w[MAXW], m = b4_out(&c->b->base, ks[k], w);
            for (int i = 0; i < m; i++) { int f = -1; for (int j = 0; j < nw; j++) if (wid[j] == w[i]) f = j; if (f < 0) { f = nw++; wid[f] = w[i]; ws[f] = 0; } ws[f] += fs[k]; } }
        for (int r = 0; r < ktop; r++) { int b = -1; for (int j = 0; j < nw; j++) if (ws[j] >= 0 && (b < 0 || ws[j] > ws[b])) b = j; top[r] = b < 0 ? UINT32_MAX : (uint32_t)wid[b]; if (b >= 0) ws[b] = -1; }
        free(wid); free(ws);
    }
    return np;
}

/* ------------------------------------------------------------------ */
/* A: today's, with the content-word budget                             */
/* ------------------------------------------------------------------ */

static int a_cost(const TextContext *t, const Meaning *m, const GEvent *e, int i) {
    static char buf[16384]; static W32 sw[4096]; int one = i;
    retell(t, e, &one, 1, buf, sizeof buf); return score_words(m, buf, sw, 4096);
}
static int a_pick(const TextContext *t, const Meaning *m, GEvent *e, int n, int budget, int *chosen) {
    int *order = malloc((size_t)n * sizeof *order), np = 0, used = 0;
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 1; i < n; i++) { int x = order[i], j = i; while (j > 0 && e[order[j - 1]].strength < e[x].strength) { order[j] = order[j - 1]; j--; } order[j] = x; }
    for (int k = 0; k < n && used < budget; k++) { int w = a_cost(t, m, e, order[k]); if (used + w > budget && used) continue; chosen[np++] = order[k]; used += w; }
    free(order);
    qsort(chosen, (size_t)np, sizeof *chosen, (int (*)(const void *, const void *))g_cmp_u32);
    return np;
}
static size_t a_resident(const TextContext *t, int tight) { return tight ? t->src_len + (size_t)t->ntok * sizeof(TextToken) : t->src_cap + (size_t)t->tok_cap * sizeof(TextToken); }
static size_t a_gist_bytes(const TextContext *t, const GEvent *e, int n, int dims) {
    size_t s = (size_t)n * sizeof(GEvent) + (size_t)n * dims * 4 + (size_t)t->nvoc * GIST_INDEX_DEPTH * 4 + (size_t)n * 8 + 16384;
    for (int i = 0; i < n; i++) s += (size_t)(e[i].b - e[i].a) * 4;
    return s;
}

static char *slurp_all(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long z = ftell(f); fseek(f, 0, SEEK_SET);
    char *s = malloc(z + 1); *n = fread(s, 1, z, f); s[*n] = 0; fclose(f); return s;
}
static void stream_init(TextContext *t) { text_init(t); t->tok_cap = 4096; t->tok = malloc(t->tok_cap * sizeof *t->tok); }

/* ------------------------------------------------------------------ */
/* ctxwin gist: 32 tales                                                */
/* ------------------------------------------------------------------ */

static int run_gist(const Meaning *m) {
    FILE *ix = fopen("data/gist/index.tsv", "r"); if (!ix) { printf("no data/gist/index.tsv\n"); return 1; }
    char line[1024]; if (!fgets(line, sizeof line, ix)) return 1;
    double sA[3] = {0}, sB[3] = {0}, sA120[3] = {0}; int tales = 0, bwin = 0, awin = 0;
    printf("%-34s %7s %7s %9s\n", "tale (F1, 60 content words)", "A", "B", "A 120w");
    while (fgets(line, sizeof line, ix)) {
        char *tale = strtok(line, "\t"); if (!tale) continue;
        char sp[512], rp[512]; snprintf(sp, sizeof sp, "data/gist/%s.story.txt", tale); snprintf(rp, sizeof rp, "data/gist/%s.summary.txt", tale);
        char *ref = slurp(rp); if (!ref) continue;
        size_t len; char *story = slurp_all(sp, &len);
        /* A */
        TextContext t; text_init(&t); text_take(&t, NULL, story, len);
        GEvent *e; int n = g_make_events(&t, m, &e); integrate(e, n, &t, m->dims);
        int *ch = malloc((size_t)n * sizeof *ch); char *out = malloc(16384); double a[3], a120[3], bsc[3];
        int np = a_pick(&t, m, e, n, BUDGET_CW, ch); retell(&t, e, ch, np, out, 16384); rouge(m, out, ref, &a[0], &a[1], &a[2]);
        np = pick(&t, e, n, GIST_BUDGET, ch); retell(&t, e, ch, np, out, 16384); rouge(m, out, ref, &a120[0], &a120[1], &a120[2]);
        /* B */
        TextContext st; stream_init(&st); CW *c = cw_make(m, &st, (size_t)64 << 20);
        cw_take(c, &st, story, len); cw_tidy(c, 1);
        cw_retell(c, BUDGET_CW, out, 16384, 0, 0); rouge(m, out, ref, &bsc[0], &bsc[1], &bsc[2]);
        if (getenv("SHOW")) printf("   B: %s\n", out);
        for (int x = 0; x < 3; x++) { sA[x] += a[x]; sB[x] += bsc[x]; sA120[x] += a120[x]; }
        bwin += bsc[1] > a[1]; awin += a[1] > bsc[1]; tales++;
        printf("%-34.34s %6.1f%% %6.1f%% %8.1f%%   (events A %d, B neurons %d)\n", tale, 100 * a[1], 100 * bsc[1], 100 * a120[1], n, c->b->base.nev);
        free(ch); free(out); g_free_events(e, n); text_free(&t); text_free(&st); free(story); free(ref);
    }
    fclose(ix);
    printf("\nmean over %d tales      recall     F1   2-word recall\n", tales);
    printf("A (60 content words)  %6.1f%% %6.1f%% %8.1f%%\n", 100 * sA[0] / tales, 100 * sA[1] / tales, 100 * sA[2] / tales);
    printf("B (60 content words)  %6.1f%% %6.1f%% %8.1f%%\n", 100 * sB[0] / tales, 100 * sB[1] / tales, 100 * sB[2] / tales);
    printf("A (120 words of page) %6.1f%% %6.1f%% %8.1f%%   (reference only)\n", 100 * sA120[0] / tales, 100 * sA120[1] / tales, 100 * sA120[2] / tales);
    printf("B better on %d tales, A better on %d\n", bwin, awin);
    printf("G1 A=%.4f B=%.4f\n", sA[1] / tales, sB[1] / tales);
    return 0;
}

/* ------------------------------------------------------------------ */
/* ctxwin book: one book                                                */
/* ------------------------------------------------------------------ */

static int has(const GEvent *x, uint32_t w) { int lo = 0, hi = x->nw; while (lo < hi) { int mid = (lo + hi) / 2; if (x->w[mid] < w) lo = mid + 1; else hi = mid; } return lo < x->nw && x->w[lo] == w; }

static int run_book(const Meaning *m, const char *path, double mb) {
    size_t len; char *book = slurp_all(path, &len);
    /* A: take in, then the gist */
    double t0 = now();
    TextContext t; text_init(&t); text_take(&t, NULL, book, len);
    double ta = now() - t0;
    size_t foldA = len * 5 + 5;
    GEvent *e; int n = g_make_events(&t, m, &e); integrate(e, n, &t, m->dims);
    int *ch = malloc((size_t)n * sizeof *ch); char *outA = malloc(16384);
    int np = a_pick(&t, m, e, n, BUDGET_CW, ch); retell(&t, e, ch, np, outA, 16384);
    uint32_t topA[6]; who(&t, e, n, topA, 6);
    double tA = now() - t0;
    size_t residA = a_resident(&t, 1), residA_alloc = a_resident(&t, 0), gistA = a_gist_bytes(&t, e, n, m->dims);
    size_t peakA = residA + (foldA > gistA ? foldA : gistA), peakA_alloc = residA_alloc + (foldA > gistA ? foldA : gistA);

    /* B: take in, gist while reading */
    t0 = now();
    TextContext st; stream_init(&st); CW *c = cw_make(m, &st, (size_t)(mb * 1024 * 1024));
    cw_take(c, &st, book, len); size_t parts_last = cw_parts(c); cw_tidy(c, 1);
    char *outB = malloc(16384); uint32_t topB[6]; cw_retell(c, BUDGET_CW, outB, 16384, topB, 6);
    double tB = now() - t0;
    size_t residB = cw_bytes(c);

    /* the events must be the same ones */
    int same = c->nmap == n, sub = 0;
    for (int j = 0; j < n && j < c->nmap; j++) { if (c->map[j] < 0) continue; int w[MAXW], k = b4_out(&c->b->base, c->map[j], w), ok = 1; for (int i = 0; i < k; i++) ok &= has(&e[j], w[i]); sub += !ok; }
    printf("events: A %d, B read %d (%s), B neurons %d, B events not matching A's words: %d (merged repeats count here)\n", n, c->nmap, same ? "same cuts" : "DIFFERENT CUTS", c->b->base.nev, sub);
    { long over = 0, lost = 0, all = 0; for (int j = 0; j < n; j++) { all += e[j].nw; if (e[j].nw > MAXW) { over++; lost += e[j].nw - MAXW; } }
      printf("events with more than %d content words: %ld of %d (%.1f%%); content words past the limit: %.1f%%\n", MAXW, over, n, 100.0 * over / n, 100.0 * lost / all); }
    printf("vocabulary: A %u words, B %u words (shared, not counted)\n", t.nvoc, st.nvoc);

    /* R1 recall: half the content words of 2,000 events */
    int S = 2000; double rA = 0, rB = 0; int cand[4096];
    int nel = 0; for (int j = 0; j < n; j++) nel += e[j].nw >= 3;
    int *el = malloc(nel * sizeof(int)); nel = 0; for (int j = 0; j < n; j++) if (e[j].nw >= 3) el[nel++] = j;
    double qa = 0, qb = 0;
    for (int q = 0; q < S; q++) {
        GEvent *x = &e[el[(long)nel * q / S]]; int nc = (x->nw + 1) / 2; if (nc < 2) nc = 2;
        double s0 = now(); int best = -1, bh = 0;
        for (int j = 0; j < n; j++) { int h = 0; for (int i = 0; i < nc; i++) h += has(&e[j], x->w[i]);
            if (h && (best < 0 || h > bh || (h == bh && (e[j].nw < e[best].nw || (e[j].nw == e[best].nw && j > best))))) { best = j; bh = h; } }
        qa += now() - s0;
        int truth[1024], nt = 0; for (int i = 0; i < x->nw && nt < 1024; i++) truth[nt++] = x->w[i];
        int got[1024], ng = 0; if (best >= 0) for (int i = 0; i < e[best].nw && ng < 1024; i++) got[ng++] = e[best].w[i];
        rA += f1(got, ng, truth, nt);
        int cue[1024]; for (int i = 0; i < nc; i++) cue[i] = x->w[i];
        s0 = now(); int hb, bev = b5_best(c->b, cue, nc, 1, &hb, 0, 0); ng = bev < 0 ? 0 : b4_out(&c->b->base, bev, got); qb += now() - s0;
        rB += f1(got, ng, truth, nt);
    }
    rA /= S; rB /= S;

    /* F1 find: 250 words in ≤20 events, 250 in more */
    int *evc = calloc(t.nvoc, sizeof(int)); for (int j = 0; j < n; j++) for (int i = 0; i < e[j].nw; i++) evc[e[j].w[i]]++;
    int *rare = malloc(t.nvoc * 4), *com = malloc(t.nvoc * 4), nr = 0, nco = 0;
    for (uint32_t w = 0; w < t.nvoc; w++) if (evc[w]) { if (evc[w] <= 20) rare[nr++] = w; else com[nco++] = w; }
    double fA = 0, fB = 0, fBr = 0, fBc = 0; int nf = 0; double fa_t = 0, fb_t = 0;
    uint8_t *inB = calloc(c->b->base.nev + 1, 1);
    for (int g = 0; g < 2; g++) for (int q = 0; q < 250; q++) {
        int *L = g ? com : rare, nl = g ? nco : nr; if (!nl) continue; uint32_t w = L[(long)nl * q / 250];
        double s0 = now(); int found = 0; for (int j = 0; j < n; j++) found += has(&e[j], w); fa_t += now() - s0;   /* A: scans its context */
        fA += (double)found / evc[w];
        s0 = now(); c5_room(c->b->base.C + 64); int k = c5_decode(&c->b->cue[w], c5_ids, 0); fb_t += now() - s0;
        for (int i = 0; i < k; i++) inB[c5_ids[i]] = 1;
        int hit = 0; for (int j = 0; j < n; j++) if (has(&e[j], w) && c->map[j] >= 0 && inB[c->map[j]]) hit++;
        for (int i = 0; i < k; i++) inB[c5_ids[i]] = 0;
        double r = (double)hit / evc[w]; fB += r; if (g) fBc += r; else fBr += r; nf++;
    }
    fA /= nf; fB /= nf;

    /* Q quote: an event given back exactly as written */
    int qA = 0, qB = 0; char pa[16384], pb[16384];
    for (int q = 0; q < S; q++) { int j = el[(long)nel * q / S], one = j;
        retell(&t, e, &one, 1, pa, sizeof pa);
        size_t a0 = t.tok[e[j].a].src, b0 = t.tok[e[j].b - 1].src + t.tok[e[j].b - 1].len; int ok = 1; for (size_t i = a0; i < b0; i++) { char ch2 = book[i] == '\n' || book[i] == '\r' ? ' ' : book[i]; ok &= pa[i - a0] == ch2; }
        qA += ok;
        int w[MAXW], k = c->map[j] >= 0 ? b4_out(&c->b->base, c->map[j], w) : 0; size_t u = 0; pb[0] = 0;
        for (int i = 0; i < k; i++) u += snprintf(pb + u, sizeof pb - u, "%s%s", i ? " " : "", st.voc[w[i]].word);
        pa[strlen(pa) ? strlen(pa) - 1 : 0] = 0; qB += strcmp(pa, pb) == 0;
    }

    printf("\nA (today's): who:"); for (int k = 0; k < 6 && topA[k] != UINT32_MAX; k++) printf(" %s", t.voc[topA[k]].word);
    printf("\n  gist: %.600s\n", outA);
    printf("B (neurons): who:"); for (int k = 0; k < 6 && topB[k] != UINT32_MAX; k++) printf(" %s", st.voc[topB[k]].word);
    printf("\n  gist: %.600s\n", outB);
    printf("\nB parts: at the last page %zu B; at rest %zu B (gist %d events); peak of all B holds %zu B\n", parts_last, cw_parts(c), c->gist.n, c->peak);
    printf("recall query: A %.1f us (scans its events), B %.1f us; find: A %.1f us, B %.1f us\n", 1e6 * qa / S, 1e6 * qb / S, 1e6 * fa_t / nf, 1e6 * fb_t / nf);
    printf("meaning parts: %ld uses, %ld grown\n", node_calls, node_miss);
    printf("find B: rare words %.4f, common words %.4f\n", fBr / 250, fBc / 250);
    printf("RESULT budget=%.1fMB tokens=%u R1_A=%.4f R1_B=%.4f F1_A=%.4f F1_B=%.4f Q_A=%.4f Q_B=%.4f M1_A=%zu M1_A_alloc=%zu M1_B=%zu M2_A=%zu M2_A_alloc=%zu M2_B=%zu tok_s_A=%.0f tok_s_B=%.0f takeonly_s_A=%.3f\n",
           mb, t.ntok, rA, rB, fA, fB, (double)qA / S, (double)qB / S, residA, residA_alloc, residB, peakA, peakA_alloc, c->peak, t.ntok / tA, t.ntok / tB, ta);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "ctxwin gist | ctxwin book BOOK BUDGET_MB\n"); return 1; }
    Meaning m; meaning_init(&m);
    if (meaning_load(&m, getenv("MEANING") ? getenv("MEANING") : MEANING_PATH) <= 0) { fprintf(stderr, "no meanings\n"); return 1; }
    meaning_norms(&m);
    if (!strcmp(argv[1], "gist")) return run_gist(&m);
    if (!strcmp(argv[1], "book") && argc >= 4) return run_book(&m, argv[2], atof(argv[3]));
    return 1;
}
