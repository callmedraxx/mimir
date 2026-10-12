/*
 * decide.c — Mimir decides like Jev and Laya (Typed Decisions), on neurons.
 * Rules fixed before this was written: docs/DECIDE_COMPARE.md.
 *
 * Reading a case: the JSON state becomes feature neurons — a leaf's key path and
 * its value (a word, yes/no, null, or a number felt as sign and size on a doubling
 * scale); free text as content words (also as plain words, wherever they were);
 * a list as its count; two numbers under the same key name compared (same / a
 * little / very different, which is larger); a value seen twice ("seen before").
 *
 *   Z  zero-shot: an option's description against the case's words, in word
 *      meanings (data/meaning); closeness → probabilities. Nothing learned.
 *   T  learns from train, once, as experience:
 *        Hebbian links   feature neuron → answer neuron, strengthened by the
 *                        answer's gold probability; a new case's active features
 *                        vote with log(P(answer | feature) / P(answer))
 *        episodes        each training case kept as an event neuron; a new case
 *                        recalls the k most similar through its features' cue lists
 *                        (rare features count more: log(N / df)); their gold votes
 *      mixed; mix, k, smoothing and a temperature per question type chosen by
 *      4-fold cross-validation on train (accuracy first, then KL for temperatures).
 *
 *   decide cv TRAIN                       the 4-fold search, prints the choice
 *   decide test TRAIN TEST OUT.jsonl      T: learn train with the chosen settings, answer test
 *   decide zero TEST OUT.jsonl            Z: answer test without learning
 *   decide feats TRAIN N                  show the features of the first N cases
 *   Run from the repo root; score with tools/decide_score.py.
 */
#include "mimir.h"
#include <ctype.h>
#include <time.h>
#include <stdarg.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* ------------------------------------------------------------------ */
/* a small JSON reader                                                  */
/* ------------------------------------------------------------------ */

typedef struct J J;
struct J { char t; double num; char *s; int n, cap; char **k; J **v; };   /* t: o a s n b z (null) */
static void jskip(const char **p) { while (**p && isspace((unsigned char)**p)) (*p)++; }
static char *jstr(const char **p) {
    (*p)++; size_t cap = 64, n = 0; char *o = malloc(cap);
    while (**p && **p != '"') {
        char c = *(*p)++;
        if (c == '\\') { char e = *(*p)++;
            if (e == 'u') { unsigned u = 0; for (int i = 0; i < 4 && **p; i++) { char h = *(*p)++; u = u * 16 + (isdigit((unsigned char)h) ? h - '0' : (tolower((unsigned char)h) - 'a' + 10)); }
                if (u < 128) c = (char)u; else if (u == 0x2019 || u == 0x2018) c = '\''; else if (u == 0x201c || u == 0x201d) c = '"'; else if (u == 0x2013 || u == 0x2014) c = '-'; else c = ' '; }
            else c = e == 'n' ? '\n' : e == 't' ? '\t' : e; }
        if (n + 2 > cap) { cap *= 2; o = realloc(o, cap); } o[n++] = c;
    }
    if (**p) (*p)++; o[n] = 0; return o;
}
static J *jparse(const char **p) {
    jskip(p); J *j = calloc(1, sizeof *j);
    if (**p == '{' || **p == '[') {
        int obj = **p == '{'; j->t = obj ? 'o' : 'a'; (*p)++;
        for (;;) { jskip(p); if (**p == '}' || **p == ']') { (*p)++; break; }
            if (**p == ',') { (*p)++; continue; }
            char *key = 0; if (obj) { key = jstr(p); jskip(p); if (**p == ':') (*p)++; }
            if (j->n == j->cap) { j->cap = j->cap ? j->cap * 2 : 4; j->k = realloc(j->k, j->cap * sizeof *j->k); j->v = realloc(j->v, j->cap * sizeof *j->v); }
            j->k[j->n] = key; j->v[j->n++] = jparse(p); }
    } else if (**p == '"') { j->t = 's'; j->s = jstr(p); }
    else if (!strncmp(*p, "true", 4)) { j->t = 'b'; j->num = 1; *p += 4; }
    else if (!strncmp(*p, "false", 5)) { j->t = 'b'; j->num = 0; *p += 5; }
    else if (!strncmp(*p, "null", 4)) { j->t = 'z'; *p += 4; }
    else { j->t = 'n'; char *e; j->num = strtod(*p, &e); *p = e; }
    return j;
}
static J *jget(J *o, const char *k) { if (!o || o->t != 'o') return 0; for (int i = 0; i < o->n; i++) if (!strcmp(o->k[i], k)) return o->v[i]; return 0; }
static void jfree(J *j) { if (!j) return; for (int i = 0; i < j->n; i++) { free(j->k ? j->k[i] : 0); jfree(j->v[i]); } free(j->k); free(j->v); free(j->s); free(j); }

/* ------------------------------------------------------------------ */
/* feature neurons                                                      */
/* ------------------------------------------------------------------ */

typedef struct { char **s; int n, cap; uint32_t *slot, mask; size_t bytes; } Names;
static uint32_t fnv(const char *s) { uint32_t h = 2166136261u; for (; *s; s++) { h ^= (unsigned char)*s; h *= 16777619u; } return h; }
static int name_id(Names *t, const char *s, int add) {
    if (!t->slot) { t->mask = (1 << 16) - 1; t->slot = calloc(t->mask + 1, 4); }
    if (t->n * 2 >= (int)t->mask) {       /* grow the slot table */
        uint32_t m = t->mask * 2 + 1; uint32_t *sl = calloc(m + 1, 4);
        for (int i = 0; i < t->n; i++) { uint32_t h = fnv(t->s[i]) & m; while (sl[h]) h = (h + 1) & m; sl[h] = i + 1; }
        free(t->slot); t->slot = sl; t->mask = m; }
    uint32_t h = fnv(s) & t->mask;
    for (; t->slot[h]; h = (h + 1) & t->mask) if (!strcmp(t->s[t->slot[h] - 1], s)) return t->slot[h] - 1;
    if (!add) return -1;
    if (t->n == t->cap) { t->cap = t->cap ? t->cap * 2 : 1024; t->s = realloc(t->s, t->cap * sizeof *t->s); }
    t->s[t->n] = strdup(s); t->bytes += strlen(s) + 1; t->slot[h] = t->n + 1; return t->n++;
}

static const Meaning *MM;
static Names F;                       /* feature neurons (learned: counted) */
typedef struct { int *f; int n, cap; } FSet;
static void fs_add(FSet *s, int id) { if (id < 0) return; for (int i = 0; i < s->n; i++) if (s->f[i] == id) return; if (s->n == s->cap) { s->cap = s->cap ? s->cap * 2 : 64; s->f = realloc(s->f, s->cap * sizeof(int)); } s->f[s->n++] = id; }
static int ADD = 1;
static int QUALIFY = 1;            /* list items also read by their label and as the last one */                   /* while learning, new features become neurons; while answering, unknown ones are ignored */
static void feat(FSet *s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void feat(FSet *s, const char *fmt, ...) { char b[512]; va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap); fs_add(s, name_id(&F, b, ADD)); }

static void bin(double v, char *o) {      /* a number felt as sign and size */
    if (v == 0) { strcpy(o, "0"); return; }
    double a = fabs(v); int e = (int)floor(log2(a));
    if (a < 1) sprintf(o, "%s<1", v < 0 ? "-" : "+"); else sprintf(o, "%s2^%d", v < 0 ? "-" : "+", e);
}
static int glue(const char *w) { return meaning_is_function(MM, w); }
/* content words of a text: lower case, letters (and inner apostrophes) */
static int words(const char *s, char out[][32], int max) {
    int n = 0; char w[64]; int k = 0;
    for (const char *p = s;; p++) {
        int c = (unsigned char)*p;
        if (isalpha(c) || (c == '\'' && k && isalpha((unsigned char)p[1]))) { if (k < 31) w[k++] = (char)tolower(c); continue; }
        if (k) { w[k] = 0; if (k > 1 && !glue(w) && n < max) strcpy(out[n++], w); k = 0; }
        if (!c) break;
    }
    return n;
}

typedef struct { char path[160], key[64]; double v; } NumLeaf;
typedef struct { char path[160]; char *v; } StrLeaf;
static NumLeaf nl[512]; static int nnl; static StrLeaf sl[512]; static int nsl;
static int nwords_of(const char *s) { int n = 1; for (; *s; s++) n += *s == ' '; return n; }

static void walk(J *j, const char *path, const char *key, FSet *s) {
    char b[32];
    switch (j->t) {
    case 'o': for (int i = 0; i < j->n; i++) { char p[160]; snprintf(p, sizeof p, "%s%s%s", path, *path ? "." : "", j->k[i]); walk(j->v[i], p, j->k[i], s); } break;
    case 'a': { bin(j->n, b); feat(s, "%s#=%s", path, b); char p[160]; snprintf(p, sizeof p, "%s[]", path);
        for (int i = 0; i < j->n; i++) {
            walk(j->v[i], p, key, s);
            if (j->v[i]->t != 'o' || !QUALIFY) continue;
            /* an item of a list, also read as "the item whose role is customer" and "the last item" */
            J *o = j->v[i]; const char *lk = 0, *lv = 0;
            for (int k = 0; k < o->n; k++) if (o->v[k]->t == 's' && strlen(o->v[k]->s) <= 20 && !strchr(o->v[k]->s, ' ') && (!lk || strcmp(o->k[k], lk) < 0)) { lk = o->k[k]; lv = o->v[k]->s; }
            char q[200]; if (lk) { snprintf(q, sizeof q, "%s[%s=%s]", path, lk, lv); walk(o, q, key, s); }
            if (i == j->n - 1) { snprintf(q, sizeof q, "%s[last]", path); walk(o, q, key, s); }
        }
        break; }
    case 'b': feat(s, "%s=%s", path, j->num ? "true" : "false"); break;
    case 'z': feat(s, "%s=null", path); break;
    case 'n': bin(j->num, b); feat(s, "%s=%s", path, b);
        if (nnl < 512) { snprintf(nl[nnl].path, 160, "%s", path); snprintf(nl[nnl].key, 64, "%s", key ? key : ""); nl[nnl++].v = j->num; } break;
    case 's': {
        int short_v = strlen(j->s) <= 48 && nwords_of(j->s) <= 5;
        if (short_v) { char low[64]; int i = 0; for (; j->s[i] && i < 63; i++) low[i] = (char)tolower((unsigned char)j->s[i]); low[i] = 0; feat(s, "%s=%s", path, low); }
        static char w[512][32]; int n = words(j->s, w, 512);
        for (int i = 0; i < n; i++) { feat(s, "%s~%s", path, w[i]); feat(s, "w~%s", w[i]); }
        if (nsl < 512 && strlen(j->s) >= 4) { snprintf(sl[nsl].path, 160, "%s", path); sl[nsl++].v = j->s; }
        break; }
    }
}
static const char *suffix(const char *k) { const char *u = strrchr(k, '_'); return u ? u + 1 : k; }
static void read_case(J *state, FSet *s) {
    nnl = nsl = 0; s->n = 0;
    walk(state, "", 0, s);
    /* two numbers under the same key name, in different places */
    for (int i = 0; i < nnl; i++) for (int j = i + 1; j < nnl; j++) {
        int same = !strcmp(nl[i].key, nl[j].key), suf = !same && !strcmp(suffix(nl[i].key), suffix(nl[j].key));
        if ((!same && !suf) || !strcmp(nl[i].path, nl[j].path)) continue;
        double a = nl[i].v, b = nl[j].v, m = fmax(fabs(a), fabs(b)), d = m ? fabs(a - b) / m : 0;
        const char *how = d == 0 ? "same" : d < 0.02 ? "little" : "very";
        const char *p1 = strcmp(nl[i].path, nl[j].path) < 0 ? nl[i].path : nl[j].path, *p2 = p1 == nl[i].path ? nl[j].path : nl[i].path;
        double x = p1 == nl[i].path ? a : b, y = p1 == nl[i].path ? b : a;
        feat(s, "cmp %s|%s=%s%s", p1, p2, how, d == 0 ? "" : x > y ? ">" : "<");
        feat(s, "cmp %s=%s", same ? nl[i].key : suffix(nl[i].key), how);
    }
    /* a value seen twice */
    for (int i = 0; i < nsl; i++) for (int j = 0; j < nsl; j++) if (i != j && !strcmp(sl[i].v, sl[j].v)) { feat(s, "seen %s", sl[i].path); feat(s, "seen any"); }
}

/* ------------------------------------------------------------------ */
/* cases, questions, answers                                            */
/* ------------------------------------------------------------------ */

#define MAXO 8
typedef struct { char name[64]; int type, no; char opt[MAXO][48]; char text[MAXO][512]; char instr[512]; } Q;   /* type: 0 noul, 1 choice, 2 score */
static Q QS[64]; static int NQ;
typedef struct { char id[96], wf[48]; int nq; int q[8]; float gold[8][MAXO]; int label[8]; FSet f; J *state; } Case;

static int SHARE = 0;   /* G1: a question with the same name, type and options is the same answer neurons in every workflow */
static int q_of(const char *wf, const char *name, J *qj) {
    char key[160]; snprintf(key, sizeof key, "%s/%s", wf, name);
    for (int i = 0; i < NQ; i++) if (!strcmp(QS[i].name, key)) return i;
    if (SHARE) {
        Q t; memset(&t, 0, sizeof t); J *ty = jget(qj, "type"), *cr = jget(qj, "criteria");
        t.type = !strcmp(ty->s, "noul") ? 0 : !strcmp(ty->s, "choice") ? 1 : 2;
        for (int i = 0; i < NQ; i++) { Q *q = &QS[i]; const char *sh = strchr(q->name, '/') + 1; if (strcmp(sh, name) || q->type != t.type) continue;
            int same = 1;
            if (t.type == 0) { J *f = jget(cr, "false"), *tr = jget(cr, "true"); same = !strcmp(q->text[0], f ? f->s : "") && !strcmp(q->text[1], tr ? tr->s : q->instr); }
            else if (!cr || cr->n != q->no) same = 0;
            else for (int k = 0; k < cr->n && same; k++) same = !strcmp(q->text[k], cr->v[k]->s) && (t.type == 2 || !strcmp(q->opt[k], cr->k[k]));
            if (same) return i; }
    }
    Q *q = &QS[NQ]; memset(q, 0, sizeof *q); snprintf(q->name, sizeof q->name, "%s", key);
    J *t = jget(qj, "type"), *cr = jget(qj, "criteria"), *in = jget(qj, "instructions");
    snprintf(q->instr, sizeof q->instr, "%s", in && in->t == 's' ? in->s : "");
    q->type = !strcmp(t->s, "noul") ? 0 : !strcmp(t->s, "choice") ? 1 : 2;
    if (q->type == 0) { q->no = 2; strcpy(q->opt[0], "false"); strcpy(q->opt[1], "true");
        J *f = jget(cr, "false"), *tr = jget(cr, "true");
        snprintf(q->text[0], 512, "%s", f ? f->s : ""); snprintf(q->text[1], 512, "%s", tr ? tr->s : q->instr); }
    else if (q->type == 1) for (int i = 0; i < cr->n && i < MAXO; i++) { snprintf(q->opt[i], 48, "%s", cr->k[i]); snprintf(q->text[i], 512, "%s", cr->v[i]->s); q->no++; }
    else for (int i = 0; i < cr->n && i < MAXO; i++) { snprintf(q->opt[i], 48, "%d", i); snprintf(q->text[i], 512, "%s", cr->v[i]->s); q->no++; }
    return NQ++;
}
static int load(const char *path, Case **out) {
    FILE *f = fopen(path, "r"); if (!f) { perror(path); exit(1); }
    int n = 0, cap = 256; Case *c = calloc(cap, sizeof *c); static char line[1 << 20];
    while (fgets(line, sizeof line, f)) {
        if (n == cap) { cap *= 2; c = realloc(c, cap * sizeof *c); memset(c + n, 0, (cap - n) * sizeof *c); }
        const char *p = line; J *j = jparse(&p); Case *x = &c[n++];
        snprintf(x->id, sizeof x->id, "%s", jget(j, "id")->s);
        const char *wf = jget(j, "workflow")->s; snprintf(x->wf, sizeof x->wf, "%s", wf); J *qs = jget(j, "questions"), *gd = jget(j, "gold");
        for (int i = 0; i < qs->n && i < 8; i++) {
            int q = q_of(wf, qs->k[i], qs->v[i]); x->q[x->nq] = q;
            J *g = jget(gd, qs->k[i]), *pr = jget(g, "probabilities"), *lb = jget(g, "label");
            for (int o = 0; o < QS[q].no; o++) { J *v = jget(pr, QS[q].opt[o]); x->gold[x->nq][o] = v ? (float)v->num : 0; if (lb && !strcmp(lb->s, QS[q].opt[o])) x->label[x->nq] = o; }
            x->nq++;
        }
        x->state = jget(j, "state");
        /* keep only the state */
        for (int i = 0; i < j->n; i++) if (j->v[i] == x->state) j->v[i] = 0;
        jfree(j);
    }
    fclose(f); *out = c; return n;
}

/* ------------------------------------------------------------------ */
/* T: what is learned                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    int nq_seen[64]; double pq[64][MAXO];          /* per question: cases, gold mass per answer */
    /* Hebbian links: (feature, question) → gold mass per answer */
    uint64_t *hk; float (*hv)[MAXO]; int hn, hcap;
    /* episodes and cue lists */
    int ne; Case **ep; int *df; int **cue, *ncue, *capcue; int nf;
} Mem;
static uint64_t hkey(int f, int q) { return (uint64_t)(f + 1) << 8 | (unsigned)q; }
static float *heb(Mem *m, int f, int q, int add) {
    if (!m->hcap) { m->hcap = 1 << 16; m->hk = calloc(m->hcap, 8); m->hv = calloc(m->hcap, sizeof *m->hv); }
    if (add && m->hn * 2 >= m->hcap) {
        int oc = m->hcap; uint64_t *ok = m->hk; float (*ov)[MAXO] = m->hv; m->hcap *= 2; m->hk = calloc(m->hcap, 8); m->hv = calloc(m->hcap, sizeof *m->hv); m->hn = 0;
        for (int i = 0; i < oc; i++) if (ok[i]) { uint64_t h = (ok[i] * 0x9E3779B97F4A7C15ull) >> 20 & (m->hcap - 1); while (m->hk[h]) h = (h + 1) & (m->hcap - 1); m->hk[h] = ok[i]; memcpy(m->hv[h], ov[i], sizeof *ov); m->hn++; }
        free(ok); free(ov); }
    uint64_t k = hkey(f, q), h = (k * 0x9E3779B97F4A7C15ull) >> 20 & (m->hcap - 1);
    for (; m->hk[h]; h = (h + 1) & (m->hcap - 1)) if (m->hk[h] == k) return m->hv[h];
    if (!add) return 0;
    m->hk[h] = k; m->hn++; return m->hv[h];
}
static void learn(Mem *m, Case *c) {
    for (int i = 0; i < c->nq; i++) { int q = c->q[i]; m->nq_seen[q]++; for (int o = 0; o < QS[q].no; o++) m->pq[q][o] += c->gold[i][o]; }
    for (int k = 0; k < c->f.n; k++) for (int i = 0; i < c->nq; i++) { float *h = heb(m, c->f.f[k], c->q[i], 1); for (int o = 0; o < QS[c->q[i]].no; o++) h[o] += c->gold[i][o]; }
    int e = m->ne++; m->ep = realloc(m->ep, m->ne * sizeof *m->ep); m->ep[e] = c;
    if (F.n > m->nf) { m->df = realloc(m->df, F.n * sizeof(int)); m->cue = realloc(m->cue, F.n * sizeof(int *)); m->ncue = realloc(m->ncue, F.n * sizeof(int)); m->capcue = realloc(m->capcue, F.n * sizeof(int));
        for (int i = m->nf; i < F.n; i++) { m->df[i] = 0; m->cue[i] = 0; m->ncue[i] = m->capcue[i] = 0; } m->nf = F.n; }
    for (int k = 0; k < c->f.n; k++) { int f = c->f.f[k]; m->df[f]++;
        if (m->ncue[f] == m->capcue[f]) { m->capcue[f] = m->capcue[f] ? m->capcue[f] * 2 : 4; m->cue[f] = realloc(m->cue[f], m->capcue[f] * sizeof(int)); } m->cue[f][m->ncue[f]++] = e; }
}
static void mem_free(Mem *m) { free(m->hk); free(m->hv); free(m->ep); for (int i = 0; i < m->nf; i++) free(m->cue[i]); free(m->cue); free(m->ncue); free(m->capcue); free(m->df); memset(m, 0, sizeof *m); }
/* bytes the learned memory would take, packed: feature names, links (2 B per answer), episodes (features as
   ~2 B gaps + 1 B per gold answer), cue lists (~2 B per entry) */
static size_t mem_bytes(Mem *m) {
    size_t b = F.bytes + (size_t)F.n * 8, links = 0, eps = 0, cues = 0;
    for (int i = 0; i < m->hcap; i++) if (m->hk[i]) links += 4 + 2 * QS[m->hk[i] & 255].no;
    for (int e = 0; e < m->ne; e++) { eps += 2 * m->ep[e]->f.n; for (int i = 0; i < m->ep[e]->nq; i++) eps += QS[m->ep[e]->q[i]].no; }
    for (int f = 0; f < m->nf; f++) cues += 2 * m->ncue[f];
    return b + links + eps + cues;
}

/* the two votes for one case: Hebbian log-scores (by smoothing) and episodes' distribution (by k) */
#define NA 4
static const double ALPHAS[NA] = { 0.1, 0.5, 1, 2 };
#define NK 6
static const int KS[NK] = { 3, 5, 10, 20, 40, 80 };
#define NR 3   /* episode similarity: 0 rarity (log N/df), 1 relevance to the question, 2 both */
typedef struct { double h[NA][MAXO]; double e[NR][NK][MAXO]; } Votes;
/* how much a feature tells about a question's answer: the information its links carry,
   sum P(o|f) log(P(o|f)/P(o)) — learned from the same co-firing counts as the links */
static double relevance(Mem *m, int f, int q) {
    float *h = heb(m, f, q, 0); if (!h) return 0; int no = QS[q].no; double tot = 0; for (int o = 0; o < no; o++) tot += h[o]; if (tot < 2) return 0;
    double N = m->nq_seen[q], r = 0;
    for (int o = 0; o < no; o++) { double pf = (h[o] + 0.5) / (tot + 0.5 * no), p = (m->pq[q][o] + 0.5) / (N + 0.5 * no); r += pf * log(pf / p); }
    return r * tot / (tot + 5);       /* a feature seen a few times is trusted less */
}
static void votes(Mem *m, Case *c, int skip_self, Votes *V) {
    static double *sim; static int *touched, simcap; if (m->ne > simcap) { simcap = m->ne * 2; sim = realloc(sim, simcap * sizeof(double)); touched = realloc(touched, simcap * sizeof(int)); memset(sim, 0, simcap * sizeof(double)); }
    for (int i = 0; i < c->nq; i++) {
        int q = c->q[i], no = QS[q].no;
        for (int a = 0; a < NA; a++) { double al = ALPHAS[a], N = m->nq_seen[q];
            for (int o = 0; o < no; o++) { double pr = (m->pq[q][o] + al) / (N + al * no); V[i].h[a][o] = log(pr);
                for (int k = 0; k < c->f.n; k++) { float *h = c->f.f[k] < m->nf ? heb(m, c->f.f[k], q, 0) : 0; if (!h) continue;
                    double tot = 0; for (int x = 0; x < no; x++) tot += h[x]; if (tot < 2) continue;    /* a link seen once says little */
                    V[i].h[a][o] += log(((h[o] + al) / (tot + al * no)) / pr); } } }
        for (int r = 0; r < NR; r++) {
            int nt = 0;
            for (int k = 0; k < c->f.n; k++) { int f = c->f.f[k]; if (f >= m->nf || !m->df[f]) continue;
                double idf = log((double)m->ne / m->df[f]), rel = r ? relevance(m, f, q) : 0, wgt = r == 0 ? idf : r == 1 ? rel : idf * rel;
                if (wgt <= 0) continue;
                for (int j = 0; j < m->ncue[f]; j++) { int e = m->cue[f][j]; if (m->ep[e] == c && skip_self) continue; if (sim[e] == 0) touched[nt++] = e; sim[e] += wgt; } }
            int top[80], ntop = 0;
            for (int t = 0; t < nt; t++) { int e = touched[t]; if (ntop < 80) top[ntop++] = e; else if (sim[e] > sim[top[79]]) top[79] = e; else continue;
                for (int j = ntop - 1; j > 0 && sim[top[j]] > sim[top[j - 1]]; j--) { int x = top[j]; top[j] = top[j - 1]; top[j - 1] = x; } }
            for (int kk = 0; kk < NK; kk++) { double s = 0; for (int o = 0; o < no; o++) V[i].e[r][kk][o] = 0;
                for (int t = 0; t < ntop && t < KS[kk]; t++) { Case *x = m->ep[top[t]]; double w = sim[top[t]]; for (int y = 0; y < x->nq; y++) if (x->q[y] == q) { for (int o = 0; o < no; o++) V[i].e[r][kk][o] += w * x->gold[y][o]; s += w; } }
                for (int o = 0; o < no; o++) V[i].e[r][kk][o] = s > 0 ? V[i].e[r][kk][o] / s : 1.0 / no; }
            for (int t = 0; t < nt; t++) sim[touched[t]] = 0;
        }
    }
}
typedef struct { int a, k, r; double w, tau[3]; } Set;
static void mix(const Votes *v, int no, const Set *s, int type, double *p) {
    double mx = -1e300, z = 0, hs[MAXO];
    for (int o = 0; o < no; o++) { hs[o] = v->h[s->a][o] / s->tau[type]; if (hs[o] > mx) mx = hs[o]; }
    for (int o = 0; o < no; o++) { hs[o] = exp(hs[o] - mx); z += hs[o]; }
    for (int o = 0; o < no; o++) p[o] = s->w * hs[o] / z + (1 - s->w) * (0.98 * v->e[s->r][s->k][o] + 0.02 / no);
}

/* ------------------------------------------------------------------ */
/* scoring (as tools/decide_score.py)                                   */
/* ------------------------------------------------------------------ */

static void score1(const double *p, const float *g, int label, int no, double *acc, double *kl, double *br) {
    double top = -1; int nt = 0, hit = 0; for (int o = 0; o < no; o++) if (p[o] > top + 1e-12) top = p[o];
    for (int o = 0; o < no; o++) if (p[o] >= top - 1e-12) { nt++; hit |= o == label; }
    *acc += hit ? 1.0 / nt : 0;
    for (int o = 0; o < no; o++) { if (g[o] > 0) *kl += g[o] * log(g[o] / fmax(p[o], 1e-6)); *br += (p[o] - g[o]) * (p[o] - g[o]); }
}

/* ------------------------------------------------------------------ */
/* commands                                                             */
/* ------------------------------------------------------------------ */

static Case *TR; static int NTR;
static void read_all(Case *c, int n) { for (int i = 0; i < n; i++) read_case(c[i].state, &c[i].f); }

static Set best_set;
static int run_cv(void) {
    /* every held-out decision's votes, then search the settings */
    int nd = 0; for (int i = 0; i < NTR; i++) nd += TR[i].nq;
    Votes *V = malloc((size_t)NTR * 8 * sizeof(Votes)); int *fold = malloc(NTR * sizeof(int));
    for (int i = 0; i < NTR; i++) fold[i] = i % 4;
    for (int f = 0; f < 4; f++) { Mem m = {0}; for (int i = 0; i < NTR; i++) if (fold[i] != f) learn(&m, &TR[i]);
        for (int i = 0; i < NTR; i++) if (fold[i] == f) votes(&m, &TR[i], 0, &V[(size_t)i * 8]); mem_free(&m); }
    static const double TAUS[] = { 1, 2, 3, 5, 8, 12, 20, 35, 60 }; const int NT = 9;
    static const double WS[] = { 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1 }; const int NW = 7;
    double bacc = -1, bkl = 1e9; Set b = {0};
    for (int a = 0; a < NA; a++) for (int k = 0; k < NK; k++) for (int r = 0; r < NR; r++) for (int w = 0; w < NW; w++) for (int t = 0; t < NT; t++) {
        Set s = { a, k, r, WS[w], { TAUS[t], TAUS[t], TAUS[t] } }; double acc = 0, kl = 0, br = 0;
        for (int i = 0; i < NTR; i++) for (int j = 0; j < TR[i].nq; j++) { int q = TR[i].q[j]; double p[MAXO]; mix(&V[(size_t)i * 8 + j], QS[q].no, &s, QS[q].type, p); score1(p, TR[i].gold[j], TR[i].label[j], QS[q].no, &acc, &kl, &br); }
        if (acc > bacc + 1e-9 || (fabs(acc - bacc) < 1e-9 && kl < bkl)) { bacc = acc; bkl = kl; b = s; }
    }
    /* then a temperature per question type, by KL, the rest fixed */
    for (int ty = 0; ty < 3; ty++) { double bk = 1e18; double bt = b.tau[ty];
        for (int t = 0; t < NT; t++) { Set s = b; s.tau[ty] = TAUS[t]; double acc = 0, kl = 0, br = 0;
            for (int i = 0; i < NTR; i++) for (int j = 0; j < TR[i].nq; j++) { int q = TR[i].q[j]; if (QS[q].type != ty) continue; double p[MAXO]; mix(&V[(size_t)i * 8 + j], QS[q].no, &s, QS[q].type, p); score1(p, TR[i].gold[j], TR[i].label[j], QS[q].no, &acc, &kl, &br); }
            if (kl < bk) { bk = kl; bt = TAUS[t]; } }
        b.tau[ty] = bt; }
    double acc = 0, kl = 0, br = 0, ah = 0, ae = 0, k2 = 0, b2 = 0;
    for (int i = 0; i < NTR; i++) for (int j = 0; j < TR[i].nq; j++) { int q = TR[i].q[j]; double p[MAXO];
        mix(&V[(size_t)i * 8 + j], QS[q].no, &b, QS[q].type, p); score1(p, TR[i].gold[j], TR[i].label[j], QS[q].no, &acc, &kl, &br);
        Set h = b; h.w = 1; mix(&V[(size_t)i * 8 + j], QS[q].no, &h, QS[q].type, p); score1(p, TR[i].gold[j], TR[i].label[j], QS[q].no, &ah, &k2, &b2);
        Set e = b; e.w = 0; mix(&V[(size_t)i * 8 + j], QS[q].no, &e, QS[q].type, p); score1(p, TR[i].gold[j], TR[i].label[j], QS[q].no, &ae, &k2, &b2); }
    if (getenv("PERQ")) for (int q = 0; q < NQ; q++) { double a1 = 0, k1 = 0, b1 = 0, ap = 0; int n1 = 0;
        for (int i = 0; i < NTR; i++) for (int j = 0; j < TR[i].nq; j++) if (TR[i].q[j] == q) { double p[MAXO]; mix(&V[(size_t)i * 8 + j], QS[q].no, &b, QS[q].type, p); score1(p, TR[i].gold[j], TR[i].label[j], QS[q].no, &a1, &k1, &b1); n1++;
            double pr[MAXO], z = 0; for (int o = 0; o < QS[q].no; o++) { pr[o] = 0; for (int x = 0; x < NTR; x++) for (int y = 0; y < TR[x].nq; y++) if (TR[x].q[y] == q && TR[x].label[y] == o) pr[o]++; z += pr[o]; } double kk = 0, bb = 0; score1(pr, TR[i].gold[j], TR[i].label[j], QS[q].no, &ap, &kk, &bb); }
        printf("  %-45s %.3f  (prior %.3f)\n", QS[q].name, a1 / n1, ap / n1); }
    printf("4-fold on train (%d decisions): accuracy %.3f  KL %.3f  Brier %.3f   (Hebbian alone %.3f, episodes alone %.3f)\n", nd, acc / nd, kl / nd, br / nd, ah / nd, ae / nd);
    printf("chosen: smoothing %.1f, episodes k %d by %s, mix %.1f Hebbian, temperatures yes/no %.0f choice %.0f score %.0f\n", ALPHAS[b.a], KS[b.k], b.r == 0 ? "rarity" : b.r == 1 ? "relevance" : "rarity x relevance", b.w, b.tau[0], b.tau[1], b.tau[2]);
    printf("SET %d %d %d %.2f %.1f %.1f %.1f\n", b.a, b.k, b.r, b.w, b.tau[0], b.tau[1], b.tau[2]);
    best_set = b; free(V); free(fold); return 0;
}

static void write_answers(FILE *o, Case *c, double (*p)[MAXO]) {
    fprintf(o, "{\"id\": \"%s\", \"answers\": {", c->id);
    for (int j = 0; j < c->nq; j++) { Q *q = &QS[c->q[j]]; const char *nm = strchr(q->name, '/') + 1;
        fprintf(o, "%s\"%s\": {", j ? ", " : "", nm); for (int k = 0; k < q->no; k++) fprintf(o, "%s\"%s\": %.6f", k ? ", " : "", q->opt[k], p[j][k]); fprintf(o, "}"); }
    fprintf(o, "}}\n");
}

static int run_test(const char *testp, const char *outp) {
    run_cv(); Set s = best_set;
    Mem m = {0}; double t0 = now(); for (int i = 0; i < NTR; i++) learn(&m, &TR[i]); double tl = now() - t0;
    Case *T; ADD = 0; int nt = load(testp, &T);
    FILE *o = fopen(outp, "w"); double acc = 0, kl = 0, br = 0; int nd = 0;
    t0 = now();
    for (int i = 0; i < nt; i++) { read_case(T[i].state, &T[i].f); Votes V[8]; votes(&m, &T[i], 0, V); double p[8][MAXO];
        for (int j = 0; j < T[i].nq; j++) mix(&V[j], QS[T[i].q[j]].no, &s, QS[T[i].q[j]].type, p[j]);
        write_answers(o, &T[i], p);
        for (int j = 0; j < T[i].nq; j++) { score1(p[j], T[i].gold[j], T[i].label[j], QS[T[i].q[j]].no, &acc, &kl, &br); nd++; } }
    double td = now() - t0; fclose(o);
    printf("T on test: accuracy %.3f  KL %.3f  Brier %.3f  (%d decisions)\n", acc / nd, kl / nd, br / nd, nd);
    printf("learned memory: %d feature neurons, %d Hebbian links, %d episodes: %zu B packed; learning took %.2f s; answering %.3f ms per case\n", F.n, m.hn, m.ne, mem_bytes(&m), tl, 1000 * td / nt);
    printf("RESULT T acc=%.4f kl=%.4f brier=%.4f ms_per_case=%.4f mem_bytes=%zu\n", acc / nd, kl / nd, br / nd, 1000 * td / nt, mem_bytes(&m));
    return 0;
}

/* Z: word meanings only */
static void mvec(const char *s, float *v) { char w[512][32]; int n = words(s, w, 512); for (int i = 0; i < n; i++) meaning_add(MM, w[i], v); }
static void jwords(J *j, float *v) {
    if (j->t == 'o') for (int i = 0; i < j->n; i++) { char k[128]; snprintf(k, sizeof k, "%s", j->k[i]); for (char *p = k; *p; p++) if (*p == '_') *p = ' '; mvec(k, v); jwords(j->v[i], v); }
    else if (j->t == 'a') for (int i = 0; i < j->n; i++) jwords(j->v[i], v);
    else if (j->t == 's') mvec(j->s, v);
}
static double cosv(const float *a, const float *b, int d) { double x = 0, y = 0, z = 0; for (int i = 0; i < d; i++) { x += a[i] * b[i]; y += a[i] * a[i]; z += b[i] * b[i]; } return y > 0 && z > 0 ? x / sqrt(y * z) : 0; }
static void zero_q(Case *c, int j, const float *sv, double *p) {
    Q *q = &QS[c->q[j]]; double cc[MAXO], mx = -1e9, z = 0;
    for (int k = 0; k < q->no; k++) { float ov[MEANING_MAX_DIMS] = {0}; char t[700]; snprintf(t, sizeof t, "%s %s", q->opt[k], q->text[k]); for (char *x = t; *x; x++) if (*x == '_') *x = ' ';
        mvec(t, ov); cc[k] = cosv(sv, ov, MM->dims) / 0.1; if (cc[k] > mx) mx = cc[k]; }
    for (int k = 0; k < q->no; k++) { cc[k] = exp(cc[k] - mx); z += cc[k]; } for (int k = 0; k < q->no; k++) p[k] = cc[k] / z;
}
/* the settings chosen for T by 4-fold CV on train (afbf448): smoothing 0.1, k 40 by relevance, mix 0.2, temperatures 8 / 35 / 8 */
static const Set T_SET = { 0, 4, 1, 0.20, { 8, 35, 8 } };

/* G1: each workflow answered by a memory that learned only the other three */
static int run_lowo(const char *testp) {
    Case *T; ADD = 0; int nt = load(testp, &T); for (int i = 0; i < nt; i++) read_case(T[i].state, &T[i].f);
    const char *W[4] = { "agent_trace_observability", "customer_service", "invoice_processing", "security_incidents" };
    double A = 0, K = 0, B = 0; int N = 0, learned = 0;
    for (int w = 0; w < 4; w++) {
        Mem m = {0}; for (int i = 0; i < NTR; i++) if (strcmp(TR[i].wf, W[w])) learn(&m, &TR[i]);
        double a = 0, k = 0, b = 0; int n = 0, lw = 0;
        for (int i = 0; i < nt; i++) { if (strcmp(T[i].wf, W[w])) continue;
            Votes V[8]; votes(&m, &T[i], 0, V); float sv[MEANING_MAX_DIMS] = {0}; jwords(T[i].state, sv);
            for (int j = 0; j < T[i].nq; j++) { int q = T[i].q[j]; double p[MAXO];
                if (m.nq_seen[q]) { mix(&V[j], QS[q].no, &T_SET, QS[q].type, p); lw++; } else zero_q(&T[i], j, sv, p);
                score1(p, T[i].gold[j], T[i].label[j], QS[q].no, &a, &k, &b); n++; } }
        printf("  %-28s accuracy %.3f  KL %.3f  Brier %.3f   (%d of %d decisions from learned questions)\n", W[w], a / n, k / n, b / n, lw, n);
        A += a; K += k; B += b; N += n; learned += lw; mem_free(&m);
    }
    printf("G1 unseen workflow: accuracy %.3f  KL %.3f  Brier %.3f  (%d decisions, %d answered from questions learned elsewhere)\n", A / N, K / N, B / N, N, learned);
    printf("RESULT G1 acc=%.4f kl=%.4f brier=%.4f learned=%d\n", A / N, K / N, B / N, learned);
    return 0;
}

/* G2: learning from n cases per workflow */
static uint64_t rng = 88172645463325252ull;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int run_curve(const char *testp) {
    Case *T; ADD = 0; int nt = load(testp, &T); for (int i = 0; i < nt; i++) read_case(T[i].state, &T[i].f);
    const char *W[4] = { "agent_trace_observability", "customer_service", "invoice_processing", "security_incidents" };
    static const int NS[] = { 10, 25, 50, 100, 200, 300 };
    printf("n per workflow   Mimir mean (worst of draws)          Prior mean (worst)\n");
    for (int si = 0; si < 6; si++) {
        int n = NS[si], draws = n < 300 ? 5 : 1; double sa = 0, wa = 1, sk = 0, wk = 0, sb = 0, wb = 0, pa = 0, pwa = 1;
        for (int d = 0; d < draws; d++) {
            rng = 88172645463325252ull + 7919 * (uint64_t)(d + 1) * (uint64_t)(n + 1);
            Mem m = {0}; static int pick[2000]; int np = 0;
            for (int w = 0; w < 4; w++) { int idx[400], c = 0; for (int i = 0; i < NTR; i++) if (!strcmp(TR[i].wf, W[w])) idx[c++] = i;
                for (int i = c - 1; i > 0; i--) { int r = rnd() % (i + 1), t = idx[i]; idx[i] = idx[r]; idx[r] = t; }
                for (int i = 0; i < n && i < c; i++) pick[np++] = idx[i]; }
            for (int i = 0; i < np; i++) learn(&m, &TR[pick[i]]);
            double a = 0, k = 0, b = 0, xa = 0, xk = 0, xb = 0; int nd = 0;
            for (int i = 0; i < nt; i++) { Votes V[8]; votes(&m, &T[i], 0, V);
                for (int j = 0; j < T[i].nq; j++) { int q = T[i].q[j]; double p[MAXO], pr[MAXO], z = 0;
                    mix(&V[j], QS[q].no, &T_SET, QS[q].type, p); score1(p, T[i].gold[j], T[i].label[j], QS[q].no, &a, &k, &b);
                    for (int o = 0; o < QS[q].no; o++) { pr[o] = 0; for (int x = 0; x < np; x++) for (int y = 0; y < TR[pick[x]].nq; y++) if (TR[pick[x]].q[y] == q && TR[pick[x]].label[y] == o) pr[o]++; z += pr[o]; }
                    for (int o = 0; o < QS[q].no; o++) pr[o] = z ? pr[o] / z : 1.0 / QS[q].no;
                    score1(pr, T[i].gold[j], T[i].label[j], QS[q].no, &xa, &xk, &xb); nd++; } }
            a /= nd; k /= nd; b /= nd; xa /= nd;
            sa += a; sk += k; sb += b; if (a < wa) wa = a; if (k > wk) wk = k; if (b > wb) wb = b; pa += xa; if (xa < pwa) pwa = xa;
            mem_free(&m);
        }
        printf("%5d            %.3f (%.3f)  KL %.3f (%.3f)  Brier %.3f (%.3f)    %.3f (%.3f)\n", n, sa / draws, wa, sk / draws, wk, sb / draws, wb, pa / draws, pwa);
        printf("RESULT G2 n=%d acc_mean=%.4f acc_worst=%.4f kl_mean=%.4f brier_mean=%.4f prior_mean=%.4f prior_worst=%.4f\n", n, sa / draws, wa, sk / draws, sb / draws, pa / draws, pwa);
    }
    return 0;
}

static int run_zero(const char *testp, const char *outp) {
    Case *T; int nt = load(testp, &T); FILE *o = fopen(outp, "w"); double acc = 0, kl = 0, br = 0; int nd = 0; int d = MM->dims;
    for (int i = 0; i < nt; i++) {
        float sv[MEANING_MAX_DIMS] = {0}; jwords(T[i].state, sv); double p[8][MAXO];
        for (int j = 0; j < T[i].nq; j++) { Q *q = &QS[T[i].q[j]]; double c[MAXO], mx = -1e9, z = 0;
            for (int k = 0; k < q->no; k++) { float ov[MEANING_MAX_DIMS] = {0}; char t[700]; snprintf(t, sizeof t, "%s %s", q->opt[k], q->text[k]); for (char *x = t; *x; x++) if (*x == '_') *x = ' ';
                mvec(t, ov); c[k] = cosv(sv, ov, d) / 0.1; if (c[k] > mx) mx = c[k]; }
            for (int k = 0; k < q->no; k++) { c[k] = exp(c[k] - mx); z += c[k]; } for (int k = 0; k < q->no; k++) p[j][k] = c[k] / z;
            score1(p[j], T[i].gold[j], T[i].label[j], q->no, &acc, &kl, &br); nd++; }
        write_answers(o, &T[i], p); }
    fclose(o); printf("Z on %s: accuracy %.3f  KL %.3f  Brier %.3f  (%d decisions)\nRESULT Z acc=%.4f kl=%.4f brier=%.4f\n", testp, acc / nd, kl / nd, br / nd, nd, acc / nd, kl / nd, br / nd);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "decide cv TRAIN | test TRAIN TEST OUT | zero TEST OUT | feats TRAIN N\n"); return 1; }
    static Meaning m; meaning_init(&m); if (meaning_load(&m, getenv("MEANING") ? getenv("MEANING") : MEANING_PATH) <= 0) { fprintf(stderr, "no meanings\n"); return 1; } MM = &m;
    if (!strcmp(argv[1], "zero") && argc >= 4) return run_zero(argv[2], argv[3]);
    if (getenv("NOQUAL")) QUALIFY = 0;
    if (!strcmp(argv[1], "lowo")) SHARE = 1;
    if (!strcmp(argv[1], "lowo") && argc >= 4) { NTR = load(argv[2], &TR); read_all(TR, NTR); return run_lowo(argv[3]); }
    NTR = load(argv[2], &TR); read_all(TR, NTR);
    if (!strcmp(argv[1], "feats")) { int n = argc > 3 ? atoi(argv[3]) : 1; for (int i = 0; i < n && i < NTR; i++) { printf("%s (%d features)\n", TR[i].id, TR[i].f.n); for (int k = 0; k < TR[i].f.n; k++) printf("  %s\n", F.s[TR[i].f.f[k]]); } return 0; }
    if (!strcmp(argv[1], "cv")) return run_cv();
    if (!strcmp(argv[1], "curve") && argc >= 4) return run_curve(argv[3]);
    if (!strcmp(argv[1], "test") && argc >= 5) return run_test(argv[3], argv[4]);
    return 1;
}
