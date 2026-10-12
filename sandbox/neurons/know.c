/*
 * know.c — understanding what it reads: facts with pointers, not words.
 * Rules fixed before this was written: docs/KNOW_COMPARE.md.
 *
 * Reading: each sentence of a text is read (understand.c's reader) into facts —
 * a verb, who, what, is-what, where/when phrases, negation — and each fact keeps
 * a pointer to where it was read (which sentence: its byte offset in the text).
 * The words of the text are then let go. The text stays where it is, a file.
 *
 *   closed book   an option becomes a statement (the question with its blank or
 *                 its wh-word filled, or the option itself), read into facts; it is
 *                 scored by how well those facts agree in meaning with the facts
 *                 understood (fact agreement), and by how close in meaning the
 *                 option's own words are to the facts that are about the question
 *                 (evidence)
 *   open book     the same, then the pointers of the facts most about the question
 *                 lead back to the text: those sentences only are re-read and the
 *                 option's words are looked for in them
 *
 *   know FILE...      MCTest-format files (MCTest, RACE, SciQ via tools/*_fetch.py)
 *   Env: SHOW=1 each question; KW="a b g" weights (fact agreement, evidence, open book)
 *   Run from the repo root.
 */
#define UNDERSTAND_NO_MAIN
#include "understand.c"

/* ------------------------------------------------------------------ */
/* word meanings, by word id                                            */
/* ------------------------------------------------------------------ */

typedef struct { int32_t *mid; uint32_t n; } MidMap;
static int mid_of(MidMap *m, TextContext *t, uint32_t w) {
    if (w == NONE || w == OPEN) return -1;
    if (w >= m->n) { uint32_t n = t->nvoc > w + 1 ? t->nvoc : w + 1; m->mid = realloc(m->mid, n * 4); for (uint32_t i = m->n; i < n; i++) m->mid[i] = -2; m->n = n; }
    if (m->mid[w] == -2) m->mid[w] = meaning_id(MM, t->voc[w].word);
    return m->mid[w];
}
static float wcos(MidMap *m, TextContext *t, uint32_t a, uint32_t b) {
    if (a == b && a != NONE && a != OPEN) return 1;
    int x = mid_of(m, t, a), y = mid_of(m, t, b); if (x < 0 || y < 0) return 0;
    const int8_t *u = MM->vec + (size_t)x * MM->dims, *v = MM->vec + (size_t)y * MM->dims; int d = 0;
    for (int i = 0; i < MM->dims; i++) d += u[i] * v[i];
    return d * inv_norm[x] * inv_norm[y];
}
/* how alike two words are, 0..1: the same word 1; near in meaning counts from cos 0.25 up */
static float wsim(MidMap *m, TextContext *t, uint32_t a, uint32_t b) { float c = wcos(m, t, a, b); if (c >= 1) return 1; return c > 0.25f ? (c - 0.25f) / 0.75f : 0; }

/* ------------------------------------------------------------------ */
/* what K keeps: facts, their pointers, and cue lists word → facts      */
/* ------------------------------------------------------------------ */

typedef struct { char **s; int n, cap; uint32_t *slot, mask; } Names;
static void name_id_k(Names *t, const char *s) {
    if (!t->slot) { t->mask = (1 << 16) - 1; t->slot = calloc(t->mask + 1, 4); }
    if (t->n * 2 >= (int)t->mask) { uint32_t m = t->mask * 2 + 1; uint32_t *sl = calloc(m + 1, 4);
        for (int i = 0; i < t->n; i++) { uint32_t h = shash(t->s[i]) & m; while (sl[h]) h = (h + 1) & m; sl[h] = i + 1; } free(t->slot); t->slot = sl; t->mask = m; }
    uint32_t h = shash(s) & t->mask; for (; t->slot[h]; h = (h + 1) & t->mask) if (!strcmp(t->s[t->slot[h] - 1], s)) return;
    if (t->n == t->cap) { t->cap = t->cap ? t->cap * 2 : 4096; t->s = realloc(t->s, t->cap * sizeof *t->s); }
    t->s[t->n] = strdup(s); t->slot[h] = ++t->n;
}
typedef struct { Frame f; uint32_t sent; } Fact;
typedef struct {
    TextContext t; WMap wm; Reader2 R; MidMap mm;
    Fact *fa; int nf, cap;
    uint32_t *where; int nsent, capsent;          /* sentence → byte offset in the text: the pointer */
    const char *src; size_t srclen;               /* the text on the shelf (not counted) */
    int *df; uint32_t ndf;                        /* in how many facts each word is */
    uint32_t **bw; int *nbw;                      /* per sentence: the words of its facts (built from facts, not from the text) */
} Know;

static void fact_words(const Frame *f, uint32_t *w, int *n) {
    *n = 0; if (f->verb != NONE) w[(*n)++] = f->verb;
    for (int i = 0; i < f->narg; i++) { if (f->a[i].head < OPEN) w[(*n)++] = f->a[i].head; if (f->a[i].mod < OPEN) w[(*n)++] = f->a[i].mod; }
}
static int content(Know *k, uint32_t w) { if (w >= OPEN) return 0; const char *s = k->t.voc[w].word; return s[0] && s[1] && !glue(s) && special(s) == SP_NONE; }

static void know_read(Know *k, const char *src, size_t n) {
    k->src = src; k->srclen = n;
    TextContext *t = &k->t; t->ntok = 0; t->src_len = 0; text_take(t, NULL, src, n);
    static Frame fr[512]; uint32_t a = 0;
    for (uint32_t i = 0; i < t->ntok; i++) {
        if (!t->tok[i].end && i + 1 < t->ntok) continue;
        if (k->nsent == k->capsent) { k->capsent = k->capsent ? k->capsent * 2 : 64; k->where = realloc(k->where, k->capsent * 4); }
        uint32_t s = k->nsent++; k->where[s] = t->tok[a].src;
        k->R.question = 0; int nf = read_span(&k->R, t->tok, a, i + 1, fr, 512, 0);
        for (int j = 0; j < nf; j++) {
            if (fr[j].verb == NONE && !fr[j].narg) continue;
            if (k->nf == k->cap) { k->cap = k->cap ? k->cap * 2 : 256; k->fa = realloc(k->fa, k->cap * sizeof(Fact)); }
            k->fa[k->nf].f = fr[j]; k->fa[k->nf++].sent = s;
        }
        a = i + 1;
    }
    /* the words of the text are let go: only facts and pointers stay */
    t->ntok = 0; t->src_len = 0;
    k->bw = calloc(k->nsent + 1, sizeof *k->bw); k->nbw = calloc(k->nsent + 1, sizeof(int));
    for (int i = 0; i < k->nf; i++) { uint32_t w[24]; int m; fact_words(&k->fa[i].f, w, &m); uint32_t s = k->fa[i].sent;
        k->bw[s] = realloc(k->bw[s], (k->nbw[s] + m) * 4); for (int j = 0; j < m; j++) { int dup = 0; for (int x = 0; x < k->nbw[s]; x++) dup |= k->bw[s][x] == w[j]; if (!dup && content(k, w[j])) k->bw[s][k->nbw[s]++] = w[j]; } }
    k->ndf = t->nvoc; k->df = calloc(k->ndf + 1, sizeof(int));
    for (int i = 0; i < k->nf; i++) { uint32_t w[24]; int m; fact_words(&k->fa[i].f, w, &m); for (int j = 0; j < m; j++) { int dup = 0; for (int x = 0; x < j; x++) dup |= w[x] == w[j]; if (!dup && w[j] < k->ndf) k->df[w[j]]++; } }
}
static double idf(Know *k, uint32_t w) { int d = w < k->ndf ? k->df[w] : 0; return log((k->nf + 1.0) / (d + 0.5)); }
/* bytes of understanding: facts as varints (as the role pool), pointers as gaps, cue lists, the word map */
static size_t know_bytes(Know *k) {
    size_t b = 0; uint8_t tmp[8];
    for (int i = 0; i < k->nf; i++) { Frame *f = &k->fa[i].f; b += 3 + vput(tmp, f->verb + 1) + vput(tmp, f->part + 1);
        for (int j = 0; j < f->narg; j++) { b += 1 + vput(tmp, f->a[j].head + 1) + vput(tmp, f->a[j].mod + 1); if (f->a[j].role == A_PP) b += vput(tmp, f->a[j].prep + 1); }
        b += vput(tmp, i ? k->fa[i].sent - k->fa[i - 1].sent : k->fa[i].sent); }
    for (int s = 0; s < k->nsent; s++) b += vput(tmp, s ? k->where[s] - k->where[s - 1] : k->where[s]);
    for (uint32_t w = 0; w < k->ndf; w++) b += 2 * (size_t)k->df[w];
    return b;
}
static Names SEEN;      /* every word this brain has met, for its word map (counted once, 9 B a word) */
static void know_seen(Know *k) { for (uint32_t w = 0; w < k->t.nvoc; w++) if (k->t.voc[w].word[0] && w < k->wm.cap && k->wm.w[w].known) name_id_k(&SEEN, k->t.voc[w].word); }
static void know_free(Know *k) { for (int s = 0; s <= k->nsent; s++) free(k->bw ? k->bw[s] : 0); free(k->bw); free(k->nbw); free(k->fa); free(k->where); free(k->df); free(k->mm.mid); free(k->wm.w); text_free(&k->t); }

/* read a statement (not kept): its facts and its content words */
static int read_statement(Know *k, const char *s, Frame *out, int max, uint32_t *words, int *nw, int maxw) {
    TextContext *t = &k->t; uint32_t n0 = t->ntok; Refs keep = k->R.refs;
    text_take(t, NULL, s, strlen(s)); k->R.question = 0;
    int nf = read_span(&k->R, t->tok, n0, t->ntok, out, max, 0);
    *nw = 0;
    for (uint32_t i = n0; i < t->ntok && *nw < maxw; i++) if (t->tok[i].kind == TEXT_WORD) {
        uint32_t w = wc_of(&k->wm, t, t->tok[i].word)->nb; if (!content(k, w)) continue;
        int dup = 0; for (int j = 0; j < *nw; j++) dup |= words[j] == w; if (!dup) words[(*nw)++] = w; }
    t->ntok = n0; t->src_len = 0; k->R.refs = keep; return nf;
}

/* ------------------------------------------------------------------ */
/* agreement in meaning                                                 */
/* ------------------------------------------------------------------ */

static float role_fit(const Arg *a, const Arg *b) {
    if (a->role == b->role) return a->role == A_PP && a->prep != b->prep ? 0.8f : 1.0f;
    if ((a->role == A_PAT && b->role == A_ATTR) || (a->role == A_ATTR && b->role == A_PAT)) return 0.8f;
    if ((a->role == A_PAT && b->role == A_PP) || (a->role == A_PP && b->role == A_PAT)) return 0.6f;
    return 0.4f;
}
/* how well fact h (from an option's statement) agrees with fact f (understood), 0..1 */
static float fact_sim(Know *k, const Frame *h, const Frame *f) {
    TextContext *t = &k->t; double num = 0, den = 0;
    if (h->verb != NONE) { int light = h->verb == k->R.w_be || h->verb == k->R.w_have || h->verb == k->R.w_do; double w = light ? 0.3 : 1.0;
        num += w * (f->verb != NONE ? wsim(&k->mm, t, h->verb, f->verb) : 0); den += w; }
    for (int i = 0; i < h->narg; i++) { const Arg *a = &h->a[i]; if (a->head >= OPEN || !content(k, a->head)) continue;
        double w = 1, best = 0;          /* every part of a fact counts the same: rarity finds facts, it does not judge agreement */
        for (int j = 0; j < f->narg; j++) { const Arg *b = &f->a[j]; if (b->head >= OPEN) continue;
            double s = role_fit(a, b) * fmax(wsim(&k->mm, t, a->head, b->head), b->mod < OPEN ? 0.6 * wsim(&k->mm, t, a->head, b->mod) : 0);
            if (s > best) best = s; }
        num += w * best; den += w; }
    if (den <= 0) return 0;
    double s = num / den;
    if ((h->flags & F_NEG) != (f->flags & F_NEG) && s > 0.5) s *= 0.3;     /* says the opposite */
    return (float)s;
}

/* how much a fact is about the question: its words against the question's content words */
static float about(Know *k, const Fact *f, const uint32_t *qw, int nq) {
    uint32_t w[24]; int n; fact_words(&f->f, w, &n); double s = 0, den = 0;
    for (int i = 0; i < nq; i++) { double b = 0; for (int j = 0; j < n; j++) { double x = wsim(&k->mm, &k->t, qw[i], w[j]); if (x > b) b = x; } s += idf(k, qw[i]) * b; den += idf(k, qw[i]); }
    return den > 0 ? (float)(s / den) : 0;
}

/* the statement an option makes: the option put where the question asks */
static int is_word_at(const char *s, const char *w) { size_t n = strlen(w); return !strncasecmp(s, w, n) && !isalpha((unsigned char)s[n]); }
static void statement(const char *q, const char *opt, char *out, size_t cap) {
    static const char *WH[] = { "what", "which", "who", "whom", "where", "when", "why", "how", 0 };
    static const char *AUX[] = { "do", "does", "did", "is", "are", "was", "were", "can", "could", "will", "would", "has", "have", "had", "should", "must", 0 };
    const char *u = strchr(q, '_');
    if (u) { snprintf(out, cap, "%.*s%s%s", (int)(u - q), q, opt, u + 1); for (char *p = out; *p; p++) if (*p == '_') *p = ' '; goto done; }
    if (strcasestr(q, "of the following") || strcasestr(q, "which is true") || strcasestr(q, "which is not true")) { snprintf(out, cap, "%s", opt); goto done; }
    for (int w = 0; WH[w]; w++) {
        if (is_word_at(q, WH[w])) {                                   /* asked at the start */
            const char *r = q + strlen(WH[w]); while (*r == ' ') r++;
            if (!strncasecmp(WH[w], "how", 3) && (is_word_at(r, "many") || is_word_at(r, "much"))) { r += 4; while (*r && *r != ' ') r++; while (*r == ' ') r++; }
            for (int a = 0; AUX[a]; a++) if (is_word_at(r, AUX[a])) {     /* "what does X do", "where did Tom go": X do <option> */
                const char *x = r + strlen(AUX[a]); while (*x == ' ') x++; snprintf(out, cap, "%s %s", x, opt); goto done; }
            snprintf(out, cap, "%s %s", opt, r); goto done;              /* "who has an eraser": <option> has an eraser */
        }
        for (const char *p = q; *p; p++)                                /* asked in the middle: "are called what" */
            if ((p == q || p[-1] == ' ') && is_word_at(p, WH[w])) { snprintf(out, cap, "%.*s%s%s", (int)(p - q), q, opt, p + strlen(WH[w])); goto done; }
    }
    snprintf(out, cap, "%s. %s", q, opt);
done:
    for (char *p = out; *p; p++) if (*p == '?') *p = '.';
}

/* ------------------------------------------------------------------ */
/* answering                                                             */
/* ------------------------------------------------------------------ */

typedef struct { double fact, evid, open, jc, jo; } Parts;
/* how well a set of words (a sentence's facts, or a re-read sentence) covers some words in meaning, rare words counting more */
static double cover(Know *k, const uint32_t *w, int n, const uint32_t *x, int nx) {
    double s = 0, d = 0; for (int i = 0; i < n; i++) { double b = 0; for (int j = 0; j < nx; j++) { double y = wsim(&k->mm, &k->t, w[i], x[j]); if (y > b) b = y; } double wt = idf(k, w[i]); s += wt * b; d += wt; }
    return d > 0 ? s / d : 0;
}
static int JMODE = 0;   /* 0: question and option coverage added; 1: multiplied */
static void score_option(Know *k, const char *q, const char *opt, const uint32_t *qw, int nq, const int *topf, int ntop, Parts *P) {
    char st[4096]; statement(q, opt, st, sizeof st);
    static Frame hf[64]; uint32_t sw[256]; int nsw; int nh = read_statement(k, st, hf, 64, sw, &nsw, 256);
    static Frame of[64]; uint32_t ow[128]; int now; read_statement(k, opt, of, 64, ow, &now, 128);
    /* fact agreement: each fact of the statement against its best understood fact */
    double fs = 0; int nfs = 0, focus = 0;
    /* the facts of the statement that carry the option's words: those are what differs between options */
    for (int i = 0; i < nh; i++) { uint32_t w[24]; int m; fact_words(&hf[i], w, &m); int has = 0; for (int j = 0; j < m; j++) for (int x = 0; x < now; x++) has |= w[j] == ow[x]; hf[i].tc = has; focus += has; }
    for (int i = 0; i < nh; i++) { if (focus && !hf[i].tc) continue; double b = 0; for (int j = 0; j < k->nf; j++) { double s = fact_sim(k, &hf[i], &k->fa[j].f); if (s > b) b = s; } fs += b; nfs++; }
    P->fact = nfs ? fs / nfs : 0;
    /* evidence: the option's own words (not the question's) in the facts most about the question */
    double ev = 0, den = 0;
    for (int i = 0; i < now; i++) { int inq = 0; for (int j = 0; j < nq; j++) inq |= qw[j] == ow[i]; if (inq) continue;
        double b = 0; for (int r = 0; r < ntop; r++) { uint32_t w[24]; int n; fact_words(&k->fa[topf[r]].f, w, &n); for (int j = 0; j < n; j++) { double x = wsim(&k->mm, &k->t, ow[i], w[j]); if (x > b) b = x; } }
        double wt = idf(k, ow[i]); ev += wt * b; den += wt; }
    P->evid = den > 0 ? ev / den : 0;
    /* one place that is about the question and says what the option says */
    uint32_t onq[128]; int nonq = 0; for (int i = 0; i < now; i++) { int inq = 0; for (int j = 0; j < nq; j++) inq |= qw[j] == ow[i]; if (!inq) onq[nonq++] = ow[i]; }
    int bests[3] = { -1, -1, -1 }; double bsv[3] = { -1, -1, -1 }; P->jc = 0;
    for (int s = 0; s < k->nsent; s++) { if (!k->nbw[s]) continue;
        double cq = cover(k, qw, nq, k->bw[s], k->nbw[s]), co = nonq ? cover(k, onq, nonq, k->bw[s], k->nbw[s]) : 0, j = JMODE ? cq * co : cq + co;
        if (j > P->jc) P->jc = j;
        for (int r = 0; r < 3; r++) if (j > bsv[r]) { for (int x = 2; x > r; x--) { bsv[x] = bsv[x - 1]; bests[x] = bests[x - 1]; } bsv[r] = j; bests[r] = s; break; } }
    /* open book: those places re-read from the text (and the sentence after each), the same measure on all their words */
    P->jo = 0;
    for (int r = 0; r < 3; r++) { if (bests[r] < 0) continue;
        TextContext *t = &k->t; uint32_t n0 = t->ntok; uint32_t s = bests[r], e = s + 2 < (uint32_t)k->nsent ? s + 2 : k->nsent;
        size_t a = k->where[s], b = e < (uint32_t)k->nsent ? k->where[e] : k->srclen; text_take(t, NULL, k->src + a, b - a);
        static uint32_t xw[4096]; int nx = 0;
        for (uint32_t x = n0; x < t->ntok && nx < 4096; x++) if (t->tok[x].kind == TEXT_WORD) { uint32_t w = wc_of(&k->wm, t, t->tok[x].word)->nb; if (content(k, w)) xw[nx++] = w; }
        double cq = cover(k, qw, nq, xw, nx), co = nonq ? cover(k, onq, nonq, xw, nx) : 0, j = JMODE ? cq * co : cq + co;
        if (j > P->jo) P->jo = j;
        t->ntok = n0; t->src_len = 0; }
    /* open book: the sentences the top facts point to, re-read from the text; the option's words looked for there */
    P->open = 0;
    if (ntop) {
        TextContext *t = &k->t; uint32_t n0 = t->ntok; int seen[16], ns = 0;
        for (int r = 0; r < ntop && r < 4; r++) { uint32_t s = k->fa[topf[r]].sent; int dup = 0; for (int x = 0; x < ns; x++) dup |= seen[x] == (int)s; if (dup) continue; seen[ns++] = s;
            size_t a = k->where[s], b = s + 1 < (uint32_t)k->nsent ? k->where[s + 1] : k->srclen; text_take(t, NULL, k->src + a, b - a); }
        double o = 0, d = 0;
        for (int i = 0; i < now; i++) { int inq = 0; for (int j = 0; j < nq; j++) inq |= qw[j] == ow[i]; if (inq) continue;
            double b = 0; for (uint32_t x = n0; x < t->ntok; x++) if (t->tok[x].kind == TEXT_WORD) { uint32_t w = wc_of(&k->wm, t, t->tok[x].word)->nb; double y = wsim(&k->mm, t, ow[i], w); if (y > b) b = y; }
            double wt = idf(k, ow[i]); o += wt * b; d += wt; }
        P->open = d > 0 ? o / d : 0;
        t->ntok = n0; t->src_len = 0;
    }
}

static int run_know(int nfile, char **files) {
    /* weights chosen on dev/train (RACE-M dev, SciQ dev, MCTest train+dev), sum mode:
       closed = 2 fact + 0.5 evidence + 2 joint;  open = 2 fact + 0.5 joint + 2 open-evidence + 2 open-joint */
    double wa = 2, wb = 0.5, wg = 2, wj = 2, wo = 2, owb = 0, owj = 0.5; if (getenv("KW")) sscanf(getenv("KW"), "%lf %lf %lf %lf %lf", &wa, &wb, &wg, &wj, &wo);
    if (getenv("JMODE")) JMODE = atoi(getenv("JMODE"));
    for (int fi = 0; fi < nfile; fi++) {
        FILE *f = fopen(files[fi], "r"); if (!f) { perror(files[fi]); return 1; }
        static char line[1 << 17]; static char text[1 << 17]; Know k; int open = 0, qseen = 0; SEEN.n = 0; if (SEEN.slot) memset(SEEN.slot, 0, (SEEN.mask + 1) * 4);
        int n = 0, okc = 0, oko = 0, okf = 0, oke = 0; size_t kb = 0, tb = 0; int stories = 0;
        while (fgets(line, sizeof line, f)) {
            size_t L = strlen(line); while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
            if (line[0] == 'S') { qseen = 0; if (open) know_free(&k); memset(&k, 0, sizeof k); text_init(&k.t); r2_init(&k.R, &k.t, &k.wm); open = 1; continue; }
            if (line[0] == 'T') { snprintf(text, sizeof text, "%s", line + 2); know_read(&k, text, strlen(text)); kb += know_bytes(&k); tb += strlen(text); stories++; continue; }
            if (line[0] == 'Q' && !qseen) { know_seen(&k); qseen = 1; }
            if (line[0] != 'Q') continue;
            char *fld[9]; int nfld = 0; for (char *p = strtok(line, "\t"); p && nfld < 9; p = strtok(0, "\t")) fld[nfld++] = p;
            if (nfld < 8) continue;
            const char *q = fld[2]; int want = fld[7][0] - 'A';
            uint32_t qw[256]; int nq; static Frame qf[64]; read_statement(&k, q, qf, 64, qw, &nq, 256);
            /* the facts most about the question */
            int top[8], ntop = 0; float ts[8];
            for (int j = 0; j < k.nf; j++) { float a = about(&k, &k.fa[j], qw, nq); if (ntop < 8) { top[ntop] = j; ts[ntop++] = a; } else if (a > ts[7]) { top[7] = j; ts[7] = a; } else continue;
                for (int x = ntop - 1; x > 0 && ts[x] > ts[x - 1]; x--) { float tf = ts[x]; ts[x] = ts[x - 1]; ts[x - 1] = tf; int ti = top[x]; top[x] = top[x - 1]; top[x - 1] = ti; } }
            int ntop5 = ntop < 5 ? ntop : 5;
            Parts P[4]; double bc = -1e9, bo = -1e9, bf = -1e9, be = -1e9; int cc = 0, co = 0, cf = 0, ce = 0;
            for (int o = 0; o < 4; o++) {
                score_option(&k, q, fld[3 + o], qw, nq, top, ntop5, &P[o]);
                double sc = wa * P[o].fact + wb * P[o].evid + wj * P[o].jc, so = wa * P[o].fact + owb * P[o].evid + owj * P[o].jc + wg * P[o].open + wo * P[o].jo;
                if (sc > bc) { bc = sc; cc = o; } if (so > bo) { bo = so; co = o; } if (P[o].fact > bf) { bf = P[o].fact; cf = o; } if (P[o].evid > be) { be = P[o].evid; ce = o; }
            }
            { static FILE *dump; if (!dump && getenv("DUMP")) dump = fopen(getenv("DUMP"), "a");
              if (dump) { const char *bn = strrchr(files[fi], '/'); fprintf(dump, "%s\t%d", bn ? bn + 1 : files[fi], want);
                  for (int o = 0; o < 4; o++) fprintf(dump, "\t%.5f %.5f %.5f %.5f %.5f", P[o].fact, P[o].evid, P[o].jc, P[o].open, P[o].jo); fprintf(dump, "\n"); fflush(dump); } }
            okc += cc == want; oko += co == want; okf += cf == want; oke += ce == want; n++;
            if (getenv("SHOW")) printf("  %s  want %c | closed %c open %c   [fact %.2f %.2f %.2f %.2f | evid %.2f %.2f %.2f %.2f]\n", q, 'A' + want, 'A' + cc, 'A' + co,
                P[0].fact, P[1].fact, P[2].fact, P[3].fact, P[0].evid, P[1].evid, P[2].evid, P[3].evid);
        }
        if (open) know_free(&k); fclose(f);
        const char *b = strrchr(files[fi], '/'); b = b ? b + 1 : files[fi];
        size_t wm = (size_t)SEEN.n * 9;
        printf("%-18s %5d q   K closed %5.2f%%   K open %5.2f%%   (fact agreement alone %5.2f%%, evidence alone %5.2f%%)\n", b, n, 100.0 * okc / n, 100.0 * oko / n, 100.0 * okf / n, 100.0 * oke / n);
        printf("   understanding of all %d texts: facts + pointers %zu B, word map %zu B (%d words, once per brain) = %zu B; the texts %zu B (%.2f of them)\n", stories, kb, wm, SEEN.n, kb + wm, tb, (double)(kb + wm) / tb);
        printf("RESULT know file=%s n=%d closed=%.2f open=%.2f fact=%.2f evid=%.2f facts_bytes=%zu wordmap_bytes=%zu text_bytes=%zu\n", b, n, 100.0 * okc / n, 100.0 * oko / n, 100.0 * okf / n, 100.0 * oke / n, kb, wm, tb);
    }
    return 0;
}

static int run_book_k(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return 1; } fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *src = malloc(n + 1); n = (long)fread(src, 1, n, f); src[n] = 0; fclose(f);
    Know k; memset(&k, 0, sizeof k); text_init(&k.t); r2_init(&k.R, &k.t, &k.wm);
    double t0 = now(); know_read(&k, src, n); double dt = now() - t0;
    size_t kb = know_bytes(&k), wm = k.wm.counted;
    printf("%s: %ld B of text, %d sentences, %d facts; understanding: facts + pointers %zu B + word map %zu B = %zu B (%.1f%% of the text); read in %.2f s\n",
           path, n, k.nsent, k.nf, kb, wm, kb + wm, 100.0 * (kb + wm) / n, dt);
    printf("RESULT kbook text=%ld facts=%d bytes=%zu wordmap=%zu seconds=%.2f\n", n, k.nf, kb, wm, dt);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "know FILE... | know book BOOK\n"); return 1; }
    lex_load(getenv("WORDCLASS") ? getenv("WORDCLASS") : "data/understand/wordclass.tsv");
    static Meaning m; meaning_init(&m);
    if (meaning_load(&m, getenv("MEANING") ? getenv("MEANING") : MEANING_PATH) <= 0) { fprintf(stderr, "no meanings\n"); return 1; }
    meaning_norms(&m); MM = &m; RV = 2;
    if (!strcmp(argv[1], "book") && argc >= 3) return run_book_k(argv[2]);
    return run_know(argc - 1, argv + 1);
}
