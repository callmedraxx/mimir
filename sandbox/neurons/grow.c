/*
 * grow.c — lean4 neurons that grow parts when they need them and shrink back.
 *
 * A neuron is the lean4 core (an event: 1 byte + its connections; a word: its
 * cue list). Extra abilities are PARTS. Each kind of part is its own table
 * keyed by neuron number: a neuron without the part costs nothing, growing
 * it adds one entry, shrinking removes it, and the table itself gets smaller
 * when it empties. At night, when events are renumbered, the tables follow.
 *
 *   part        on          grows when                     shrinks when                         bytes
 *   firing      event       it is read or recalled          it fades (halves every 2 events,     10
 *                                                           gone after ~8) or is pushed out of
 *                                                           the workspace of 4
 *   firing      word        it is read                      it fades the same way                10
 *   committed   event       it has been used 8+ times       committed events would pass ¼ of     4
 *                           (read again, or rehearsed);     the stored events: oldest commitment
 *                           a committed event is never      goes first
 *                           pruned
 *
 *   grow MODE BUDGET_MB BOOK NOVEL_BOOK
 *     MODE 0 = lean4 (rehearsal only bumps the use counter), 1 = lean4 + parts
 */
#define main compare_main
#include "compare.c"
#undef main

/* ------------------------------------------------------------------ */
/* A part table: neuron number → small value. Linear probing with       */
/* backward-shift delete, so it can shrink.                             */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t *key; uint8_t *val; int n, cap, vsize; const char *name; } Part;
#define EMPTY 0xffffffffu
static size_t part_bytes(const Part *t) { return (size_t)t->cap * (4 + t->vsize); }

static void part_resize(Part *t, int cap) {
    uint32_t *ok = t->key; uint8_t *ov = t->val; int oc = t->cap;
    t->cap = cap; t->n = 0;
    t->key = cap ? malloc(cap * 4) : 0; t->val = cap && t->vsize ? malloc((size_t)cap * t->vsize) : 0;
    if (cap) memset(t->key, 255, cap * 4);
    for (int i = 0; i < oc; i++) if (ok[i] != EMPTY) {
        uint32_t h = mix(ok[i]) & (cap - 1); while (t->key[h] != EMPTY) h = (h + 1) & (cap - 1);
        t->key[h] = ok[i]; if (t->vsize) memcpy(t->val + (size_t)h * t->vsize, ov + (size_t)i * t->vsize, t->vsize); t->n++;
    }
    free(ok); free(ov);
}
static void part_clear(Part *t) { free(t->key); free(t->val); t->key = 0; t->val = 0; t->n = t->cap = 0; }
static uint8_t *part_get(Part *t, uint32_t k) {
    if (!t->cap) return 0;
    for (uint32_t h = mix(k) & (t->cap - 1);; h = (h + 1) & (t->cap - 1)) {
        if (t->key[h] == EMPTY) return 0;
        if (t->key[h] == k) return t->vsize ? t->val + (size_t)h * t->vsize : (uint8_t *)t->key;   /* non-null = present */
    }
}
static uint8_t *part_grow(Part *t, uint32_t k) {   /* returns the value, new or existing */
    uint8_t *v = part_get(t, k); if (v) return v;
    if ((t->n + 1) * 4 > t->cap * 3) part_resize(t, t->cap ? t->cap * 2 : 8);
    uint32_t h = mix(k) & (t->cap - 1); while (t->key[h] != EMPTY) h = (h + 1) & (t->cap - 1);
    t->key[h] = k; t->n++;
    if (t->vsize) { memset(t->val + (size_t)h * t->vsize, 0, t->vsize); return t->val + (size_t)h * t->vsize; }
    return (uint8_t *)t->key;
}
static void part_shrink(Part *t, uint32_t k) {
    if (!t->cap) return;
    uint32_t h = mix(k) & (t->cap - 1);
    while (t->key[h] != k) { if (t->key[h] == EMPTY) return; h = (h + 1) & (t->cap - 1); }
    for (uint32_t j = h;;) {               /* backward shift */
        j = (j + 1) & (t->cap - 1); if (t->key[j] == EMPTY) break;
        uint32_t home = mix(t->key[j]) & (t->cap - 1);
        if (((j - home) & (t->cap - 1)) >= ((j - h) & (t->cap - 1))) {
            t->key[h] = t->key[j]; if (t->vsize) memcpy(t->val + (size_t)h * t->vsize, t->val + (size_t)j * t->vsize, t->vsize); h = j; }
    }
    t->key[h] = EMPTY; t->n--;
    if (t->n == 0) part_clear(t);
    else if (t->cap > 8 && t->n * 8 < t->cap) part_resize(t, t->cap / 2);
}

/* ------------------------------------------------------------------ */
/* lean4 + parts                                                        */
/* ------------------------------------------------------------------ */

#define FIRE_MAX 32767
#define HALF 2              /* events per halving */
#define FIRE_MIN 2048       /* below this a firing part is dropped: 4 halvings */
#define WORKSPACE 4
#define COMMIT_LEVEL 3      /* use counter level 3 = used 8+ times */

typedef struct { int16_t act; uint32_t last; } __attribute__((packed)) Fire;   /* 6 B value */
typedef struct { B5 *b; int parts, auto_commit; Part fire_ev, fire_w, commit; int peak_fire_ev, peak_fire_w; size_t peak_parts; uint32_t now; } G;

static size_t parts_bytes(G *g) { return part_bytes(&g->fire_ev) + part_bytes(&g->fire_w) + part_bytes(&g->commit); }
static size_t g_bytes(G *g) { return b5_bytes(g->b) + parts_bytes(g); }

static int act_now(G *g, const Fire *f) { uint32_t d = (g->now - f->last) / HALF; return d >= 15 ? 0 : f->act >> d; }

/* shrink everything that has faded (tables are small: they hold what is in mind) */
static void fade(G *g, Part *t) {
    for (int i = 0; i < t->cap;) {
        if (t->key[i] != EMPTY && act_now(g, (Fire *)(t->val + (size_t)i * t->vsize)) < FIRE_MIN) { part_shrink(t, t->key[i]); i = 0; if (!t->cap) return; }
        else i++;
    }
}
static void fire(G *g, Part *t, uint32_t k) {
    Fire *f = (Fire *)part_grow(t, k); f->act = FIRE_MAX; f->last = g->now;
}
/* workspace: only the 4 most active events keep firing */
static void workspace_limit(G *g) {
    while (g->fire_ev.n > WORKSPACE) {
        int wi = -1, wa = 1 << 30; uint32_t wl = 0;
        for (int i = 0; i < g->fire_ev.cap; i++) if (g->fire_ev.key[i] != EMPTY) {
            Fire *f = (Fire *)(g->fire_ev.val + (size_t)i * 6); int a = act_now(g, f);
            if (a < wa || (a == wa && f->last < wl)) { wa = a; wl = f->last; wi = i; } }
        part_shrink(&g->fire_ev, g->fire_ev.key[wi]);
    }
}

static int drop_cap, drop_full, drop_auto;
static int g_keep(void *ctx, int e) { G *g = ctx; return part_get(&g->commit, e) != 0; }
static void g_remap(void *ctx, const int *map, int old) {
    G *g = ctx; Part *ts[2] = { &g->fire_ev, &g->commit };
    for (int t = 0; t < 2; t++) {
        Part *p = ts[t]; int n = p->n; if (!n) continue;
        uint32_t *k = malloc(n * 4); uint8_t *v = malloc((size_t)n * (p->vsize ? p->vsize : 1)); int m = 0;
        for (int i = 0; i < p->cap; i++) if (p->key[i] != EMPTY && (int)p->key[i] < old && map[p->key[i]] >= 0) {
            k[m] = map[p->key[i]]; if (p->vsize) memcpy(v + (size_t)m * p->vsize, p->val + (size_t)i * p->vsize, p->vsize); m++; }
        part_clear(p);
        for (int i = 0; i < m; i++) { uint8_t *d = part_grow(p, k[i]); if (p->vsize) memcpy(d, v + (size_t)i * p->vsize, p->vsize); }
        free(k); free(v);
    }
}

/* commit when used enough; keep commitments to ¼ of the stored events (oldest goes first) */
static void maybe_commit(G *g, int ev) {
    if (!g->parts || !g->auto_commit || EVU(g->b->base.ev[ev]) < COMMIT_LEVEL || part_get(&g->commit, ev)) return;
    part_grow(&g->commit, ev);
    while (g->commit.n * 4 > g->b->base.nev) {
        uint32_t old = EMPTY; for (int i = 0; i < g->commit.cap; i++) if (g->commit.key[i] < old) old = g->commit.key[i];
        part_shrink(&g->commit, old); drop_auto++;
    }
}

static void g_note_peaks(G *g) {
    if (g->fire_ev.n > g->peak_fire_ev) g->peak_fire_ev = g->fire_ev.n;
    if (g->fire_w.n > g->peak_fire_w) g->peak_fire_w = g->fire_w.n;
    if (parts_bytes(g) > g->peak_parts) g->peak_parts = parts_bytes(g);
}

static void g_store(G *g, const Event *e) {
    B4 *a = &g->b->base; g->now++;
    b5_store(g->b, e);
    int ev = a->prev;
    if (g->parts) {
        fade(g, &g->fire_ev); fade(g, &g->fire_w);
        if (ev >= 0) { fire(g, &g->fire_ev, ev); workspace_limit(g); maybe_commit(g, ev); }
        for (int i = 0; i < e->n; i++) fire(g, &g->fire_w, e->w[i]);
        while (g_bytes(g) > a->budget && a->nev) {          /* parts count toward the budget too */
            size_t was = g_bytes(g); b5_prune(g->b);
            if (g_bytes(g) >= was) { if (!g->commit.n) break; drop_full++; uint32_t old = EMPTY; for (int i = 0; i < g->commit.cap; i++) if (g->commit.key[i] < old) old = g->commit.key[i]; part_shrink(&g->commit, old); }
        }
        g_note_peaks(g);
    }
}

/* rehearse: recall it, count the use (same log counter as reading), fire it, maybe commit */
static void g_rehearse(G *g, const int *w, int n) {
    B4 *a = &g->b->base; int h, ev = b5_best(g->b, w, n, 1, &h, 0, 0); if (ev < 0) return;
    g->now++; a->tick++;
    int u = EVU(a->ev[ev]); if (u < 3 && !(mix(((uint64_t)a->tick << 24) ^ ev) & ((1u << u) - 1))) a->ev[ev] += 16;
    if (g->parts) { fade(g, &g->fire_ev); fire(g, &g->fire_ev, ev); workspace_limit(g); maybe_commit(g, ev); g_note_peaks(g); }
}

/* "remember this": said once. With parts it grows the committed part; without, it is one more use */
static void g_pin(G *g, const int *w, int n) {
    if (!g->parts) { g_rehearse(g, w, n); return; }
    int h, ev = b5_best(g->b, w, n, 1, &h, 0, 0); if (ev < 0) return;
    part_grow(&g->commit, ev);
    while (g->commit.n * 4 > g->b->base.nev) { drop_cap++; uint32_t old = EMPTY; for (int i = 0; i < g->commit.cap; i++) if (g->commit.key[i] < old) old = g->commit.key[i]; part_shrink(&g->commit, old); }
}

static void g_idle(G *g, int ticks) { for (int i = 0; i < ticks; i++) { g->now++; fade(g, &g->fire_ev); fade(g, &g->fire_w); } }

static double f1s(const int *a, int na, const int *b, int nb) { return f1(a, na, b, nb); }

#ifndef GROW_NO_MAIN
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "grow MODE BUDGET_MB BOOK NOVEL_BOOK\n"); return 1; }
    int mode = atoi(argv[1]); double mb = atof(argv[2]); size_t budget = (size_t)(mb * 1024 * 1024);
    V.cap = 1 << 16; V.str = malloc(V.cap * sizeof *V.str); V.count = malloc(V.cap * sizeof *V.count); V.nslot = 1 << 20; V.slot = malloc(V.nslot * sizeof(int)); memset(V.slot, -1, V.nslot * sizeof(int));
    size_t len; char *book = load(argv[3], &len), *nbook = load(argv[4], &len);
    int nt, nnt; long words, nwords;
    int *t = tokenize(book, &nt, &words);
    for (int i = 0; i < nt; i++) if (t[i] >= 0) V.count[t[i]]++;
    int nbase = V.n; int *byc = malloc(nbase * sizeof(int)); for (int i = 0; i < nbase; i++) byc[i] = i; qsort(byc, nbase, sizeof(int), cmp_count_desc);
    int *nt2 = tokenize(nbook, &nnt, &nwords);
    is_function = calloc(V.n, 1); for (int i = 0; i < 150 && i < nbase; i++) is_function[byc[i]] = 1;
    int ne, nne; Event *ev = make_events(t, nt, &ne), *nov = make_events(nt2, nnt, &nne);

    syn_bytes = 0;
    G g = {0}; g.parts = mode; g.auto_commit = getenv("AUTOCOMMIT") != 0; g.b = b5_make(budget, V.n); g.b->base.bitpool = 1;
    g.fire_ev = (Part){ .vsize = 6, .name = "firing (event)" }; g.fire_w = (Part){ .vsize = 6, .name = "firing (word)" }; g.commit = (Part){ .vsize = 0, .name = "committed" };
    if (mode) { g.b->base.keep = g_keep; g.b->base.remap = g_remap; g.b->base.ctx = &g; }
    design = 5; g5 = g.b;   /* the shared tests read through lean4's recall */

    /* read the first tenth, rehearse 200 of its events 12 times, read the rest */
    int first = ne / 10, R = 200, REP = getenv("REP") ? atoi(getenv("REP")) : 12, PIN = getenv("PIN") != 0;
    double t0 = now();
    for (int i = 0; i < first; i++) g_store(&g, &ev[i]);
    double ts = now() - t0;
    { int nst = g.b->base.nev, have = 0; uint64_t *sg = malloc((size_t)nst * 8);
      for (int i = 0; i < nst; i++) { int o[MAXW], n = b4_out(&g.b->base, i, o); uint64_t h = 0; for (int j = 0; j < n; j++) h += mix(o[j]); sg[i] = h ^ (uint64_t)n << 59; }
      for (int q = 0; q < R; q++) { const Event *e = &ev[(long)first * q / R]; uint64_t h = 0; for (int j = 0; j < e->n; j++) h += mix(e->w[j]); h ^= (uint64_t)e->n << 59;
          int f = 0; for (int i = 0; i < nst && !f; i++) f = sg[i] == h; have += f; }
      printf("still stored when told to keep them: %d of %d\n", have, R); free(sg); }
    if (PIN) for (int q = 0; q < R; q++) { const Event *e = &ev[(long)first * q / R]; g_pin(&g, e->w, e->n); }
    else for (int r = 0; r < REP; r++) for (int q = 0; q < R; q++) { const Event *e = &ev[(long)first * q / R]; g_rehearse(&g, e->w, e->n); }
    t0 = now();
    for (int i = first; i < ne; i++) g_store(&g, &ev[i]);
    ts += now() - t0;

    /* in mind at the end of the book */
    printf("in mind after the last page (%d events firing):\n", g.fire_ev.n);
    for (int i = 0; i < g.fire_ev.cap; i++) if (g.fire_ev.key[i] != EMPTY) { int out[MAXW]; int n = b4_out(&g.b->base, g.fire_ev.key[i], out);
        printf("   [%5d]", act_now(&g, (Fire *)(g.fire_ev.val + (size_t)i * 6))); for (int j = 0; j < n; j++) printf(" %s", V.str[out[j]]); printf("\n"); }
    size_t before = parts_bytes(&g), fw = g.fire_w.n;
    g_idle(&g, 20);
    printf("parts: peak %zu B (event firing peak %d, word firing peak %d); at the last page %zu B (%zu words firing); after resting %zu B; committed %d events = %zu B\n",
           g.peak_parts, g.peak_fire_ev, g.peak_fire_w, before, fw, parts_bytes(&g), g.commit.n, part_bytes(&g.commit));

    /* the shared tests, plus T5: the rehearsed events, cued with half their words */
    int S = 2000; double rs;
    double T1 = t_recall(ev, 0, ne, S, &rs), T2 = t_recall(ev, 0, ne / 10, S, 0), T3 = t_recognise(ev, ne, nov, nne, S), T4 = t_next(ev, ne, S);
    double T5 = 0; int out[MAXW * 2];
    for (int q = 0; q < R; q++) { const Event *e = &ev[(long)first * q / R]; int nc = (e->n + 1) / 2; if (nc < 2) nc = 2; int no = recall(e->w, nc, out); T5 += f1s(out, no, e->w, e->n); }
    T5 /= R;
    /* stored: some event has exactly these words; reachable: all its words still lead to it */
    int nst = g.b->base.nev; uint64_t *sig = malloc((size_t)nst * 8);
    for (int i = 0; i < nst; i++) { int o[MAXW], n = b4_out(&g.b->base, i, o); uint64_t h = 0; for (int j = 0; j < n; j++) h += mix(o[j]); sig[i] = h ^ (uint64_t)n << 59; }
    int stored_n = 0, reach = 0;
    for (int q = 0; q < R; q++) { const Event *e = &ev[(long)first * q / R]; uint64_t h = 0; for (int j = 0; j < e->n; j++) h += mix(e->w[j]); h ^= (uint64_t)e->n << 59;
        int found = 0; for (int i = 0; i < nst && !found; i++) found = sig[i] == h; stored_n += found;
        int hh, x = b5_best(g.b, e->w, e->n, 1, &hh, 0, 0); reach += found && x >= 0 && hh == e->n && EVN(g.b->base.ev[x]) == e->n; }
    printf("commitments dropped: cap at pin %d, cap at auto-commit %d, memory full %d\n", drop_cap, drop_auto, drop_full);
    printf("of %d: stored %d, reachable %d\n", R, stored_n, reach); free(sig);
    int committed_rehearsed = 0; if (mode) for (int q = 0; q < R; q++) { const Event *e = &ev[(long)first * q / R]; int h, x = b5_best(g.b, e->w, e->n, 1, &h, 0, 0); committed_rehearsed += x >= 0 && part_get(&g.commit, x); }
    printf("design=%s budget=%.1fMB T1=%.4f T2=%.4f T3=%.4f T4=%.4f T5_rehearsed=%.4f store_tok_s=%.0f bytes=%zu core=%zu parts=%zu stored=%d rehearsed_committed=%d\n",
           mode ? "lean4+parts" : "lean4", mb, T1, T2, T3, T4, T5, words / ts, g_bytes(&g), b5_bytes(g.b), parts_bytes(&g), g.b->base.nev, committed_rehearsed);
    return 0;
}
#endif
