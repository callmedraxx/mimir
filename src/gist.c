/*
 * MIMIR — Gist (understanding what was read, the way people do)
 *
 * People do not keep what they read word for word. They read in small
 * cycles, keep a few ideas in mind, link each new idea to those and to
 * whatever it reminds them of, and what stays active the longest becomes
 * what they remember: the gist. This is Kintsch's construction-integration
 * model, the standard account of how people remember stories, built here
 * from Mimir's own parts:
 *
 *   events      the context split into events: clauses, cut at . ! ? ; :
 *               and at ", and / but / so / then / when / while"
 *   workspace   GIST_WORKSPACE events carried from cycle to cycle (about 4,
 *               like human working memory); it never grows
 *   cued recall the new event's content words cue the GIST_RECALL most
 *               related earlier events back into the workspace, through an
 *               index from each word to the last events it was in (like the
 *               hippocampus: a cue goes straight to the memories, no scan)
 *   links       two events activate each other by the content words they
 *               share, and by how close their meanings are (semantic memory)
 *   spreading   activation spreads in the workspace until it settles; each
 *               event's settled activation adds to its memory strength
 *   fairness    an event read late had fewer cycles to gather strength:
 *               strength is divided by the cycles it could have had
 *
 * The gist: who the story is about (content words with the most strength
 * behind them) and a retelling (the strongest events, in story order, up to
 * a word budget). Measured against human plot summaries of 32 Grimm tales
 * (`gist test`). Work per event is fixed (a workspace of 4, 4 recalled),
 * so any length of text costs the same per word.
 */

#include "mimir.h"
#include <ctype.h>

typedef struct {
    uint32_t a, b;           /* tokens [a, b) of the context */
    uint32_t *w;             /* its content words (TextWord ids), sorted */
    int      nw;
    float   *v;              /* its meaning (sum of its words' meanings, unit length) */
    double   strength;
} Event;

static int cmp_u32(const void *x, const void *y) {
    uint32_t a = *(const uint32_t *)x, b = *(const uint32_t *)y;
    return a < b ? -1 : a > b;
}

static int is_content(const TextContext *t, const Meaning *m, uint32_t id) {
    const TextWord *v = &t->voc[id];
    if (v->kind != TEXT_WORD || v->word[0] == '^' || !v->word[1]) return 0;
    return !meaning_is_function(m, v->word);
}

static int joins(const char *w) {
    static const char *J[] = { "and", "but", "so", "then", "when", "while" };
    for (int i = 0; i < 6; i++) if (strcmp(w, J[i]) == 0) return 1;
    return 0;
}

/* Cut the context into events. Returns how many; *out is malloc'd. */
static int make_events(const TextContext *t, const Meaning *m, Event **out) {
    int cap = 1024, n = 0;
    Event *e = malloc((size_t)cap * sizeof *e);
    uint32_t start = 0;
    for (uint32_t i = 0; i < t->ntok && e; i++) {
        const TextToken *k = &t->tok[i];
        const char *w = t->voc[k->word].word;
        int cut = k->end ||
                  (k->kind == TEXT_PUNCT && (w[0] == ';' || w[0] == ':')) ||
                  (k->kind == TEXT_PUNCT && w[0] == ',' && i + 1 < t->ntok &&
                   joins(t->voc[t->tok[i + 1].word].word));
        if (!cut && i + 1 < t->ntok) continue;
        /* a closing quote right after the stop belongs to this event: .’ */
        while (cut && i + 1 < t->ntok && t->tok[i + 1].kind == TEXT_PUNCT &&
               (t->voc[t->tok[i + 1].word].word[0] == '\'' || t->voc[t->tok[i + 1].word].word[0] == '"') &&
               t->tok[i + 1].src == t->tok[i].src + t->tok[i].len)
            i++;
        uint32_t a = start, b = i + 1;
        start = i + 1;
        int words = 0;
        for (uint32_t j = a; j < b; j++) words += t->tok[j].kind == TEXT_WORD;
        if (words < 3) {                       /* too short to be an event: join the next */
            start = a;
            if (i + 1 < t->ntok) continue;
            if (!words) break;
        }
        if (n == cap) { cap *= 2; Event *g = realloc(e, (size_t)cap * sizeof *e); if (!g) break; e = g; }
        Event *x = &e[n++];
        memset(x, 0, sizeof *x);
        x->a = a;
        x->b = b;
        x->w = malloc((size_t)(b - a) * sizeof *x->w);
        x->v = calloc((size_t)(m->dims ? m->dims : 1), sizeof(float));
        for (uint32_t j = a; j < b; j++) {
            uint32_t id = t->tok[j].word;
            if (!is_content(t, m, id)) continue;
            int seen = 0;
            for (int q = 0; q < x->nw; q++) seen |= x->w[q] == id;
            if (seen) continue;
            x->w[x->nw++] = id;
            meaning_add(m, t->voc[id].word, x->v);
        }
        qsort(x->w, (size_t)x->nw, sizeof *x->w, cmp_u32);
        float nn = 0;
        for (int d = 0; d < m->dims; d++) nn += x->v[d] * x->v[d];
        nn = nn > 0 ? 1.0f / sqrtf(nn) : 0;
        for (int d = 0; d < m->dims; d++) x->v[d] *= nn;
    }
    *out = e;
    return n;
}

static void free_events(Event *e, int n) {
    for (int i = 0; i < n; i++) { free(e[i].w); free(e[i].v); }
    free(e);
}

/* How strongly two events activate each other. */
static float link(const Event *x, const Event *y, int dims) {
    int share = 0;
    for (int i = 0, j = 0; i < x->nw && j < y->nw;) {
        if (x->w[i] == y->w[j]) { share++; i++; j++; }
        else if (x->w[i] < y->w[j]) i++;
        else j++;
    }
    float cos = 0;
    for (int d = 0; d < dims; d++) cos += x->v[d] * y->v[d];
    return (float)share + 2.0f * (cos > 0 ? cos : 0);
}

/* Read the events in cycles; fills each event's strength. */
static void integrate(Event *e, int n, const TextContext *t, int dims) {
    /* the hippocampal index: word → the last GIST_INDEX_DEPTH events it was in */
    int32_t (*recent)[GIST_INDEX_DEPTH] = malloc((size_t)t->nvoc * sizeof *recent);
    if (!recent) return;
    for (uint32_t i = 0; i < t->nvoc; i++) for (int k = 0; k < GIST_INDEX_DEPTH; k++) recent[i][k] = -1;

    int carried[GIST_WORKSPACE], nc = 0;
    for (int i = 0; i < n; i++) {
        int nodes[GIST_WORKSPACE + GIST_RECALL + 1], nn = 0;
        for (int k = 0; k < nc; k++) nodes[nn++] = carried[k];
        /* cued recall: candidates are earlier events sharing a word with this one */
        int cand[GIST_INDEX_DEPTH * 16], ncand = 0;
        for (int q = 0; q < e[i].nw && ncand < GIST_INDEX_DEPTH * 16 - GIST_INDEX_DEPTH; q++)
            for (int k = 0; k < GIST_INDEX_DEPTH; k++) {
                int j = recent[e[i].w[q]][k];
                if (j < 0) break;
                int dup = 0;
                for (int c = 0; c < ncand; c++) dup |= cand[c] == j;
                for (int c = 0; c < nc; c++) dup |= carried[c] == j;
                if (!dup) cand[ncand++] = j;
            }
        for (int r = 0; r < GIST_RECALL && ncand; r++) {
            int bi = 0;
            float bl = -1;
            for (int c = 0; c < ncand; c++) {
                float l = link(&e[i], &e[cand[c]], dims);
                if (l > bl) { bl = l; bi = c; }
            }
            if (bl <= 0) break;
            nodes[nn++] = cand[bi];
            cand[bi] = cand[--ncand];
        }
        nodes[nn++] = i;
        /* spread activation until it settles */
        float W[GIST_WORKSPACE + GIST_RECALL + 1][GIST_WORKSPACE + GIST_RECALL + 1];
        float a[GIST_WORKSPACE + GIST_RECALL + 1], b[GIST_WORKSPACE + GIST_RECALL + 1];
        for (int x = 0; x < nn; x++) {
            a[x] = 1.0f / nn;
            for (int y = 0; y < nn; y++) W[x][y] = x == y ? 0 : link(&e[nodes[x]], &e[nodes[y]], dims);
        }
        for (int it = 0; it < 30; it++) {
            float sum = 0, diff = 0;
            for (int x = 0; x < nn; x++) {
                b[x] = a[x];
                for (int y = 0; y < nn; y++) b[x] += W[x][y] * a[y];
                sum += b[x];
            }
            for (int x = 0; x < nn; x++) { b[x] /= sum + 1e-12f; diff = fmaxf(diff, fabsf(b[x] - a[x])); a[x] = b[x]; }
            if (diff < 1e-4f) break;
        }
        for (int x = 0; x < nn; x++) e[nodes[x]].strength += a[x];
        /* carry the GIST_WORKSPACE most active on */
        nc = 0;
        for (int r = 0; r < GIST_WORKSPACE && r < nn; r++) {
            int bx = -1;
            for (int x = 0; x < nn; x++) if (a[x] >= 0 && (bx < 0 || a[x] > a[bx])) bx = x;
            carried[nc++] = nodes[bx];
            a[bx] = -1;
        }
        /* index this event under each of its words */
        for (int q = 0; q < e[i].nw; q++) {
            int32_t *r = recent[e[i].w[q]];
            memmove(r + 1, r, (GIST_INDEX_DEPTH - 1) * sizeof *r);
            r[0] = i;
        }
    }
    /* fairness: an event read late had fewer cycles to gather strength; count
     * the cycles it could have had, but no more than a chapter's worth
     * (GIST_HORIZON), or the last page of a long book outweighs the rest */
    for (int i = 0; i < n; i++) e[i].strength /= (double)(n - i < GIST_HORIZON ? n - i : GIST_HORIZON);
    free(recent);
}

static int span_words(const TextContext *t, const Event *x) {
    int w = 0;
    for (uint32_t j = x->a; j < x->b; j++) w += t->tok[j].kind == TEXT_WORD;
    return w;
}

/* Pick the strongest events up to `budget` words; returns how many, in story order. */
static int pick(const TextContext *t, Event *e, int n, int budget, int *chosen) {
    int *order = malloc((size_t)n * sizeof *order), np = 0, used = 0;
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 1; i < n; i++) {                       /* strongest first */
        int x = order[i], j = i;
        while (j > 0 && e[order[j - 1]].strength < e[x].strength) { order[j] = order[j - 1]; j--; }
        order[j] = x;
    }
    for (int k = 0; k < n && used < budget; k++) {
        int w = span_words(t, &e[order[k]]);
        if (used + w > budget && used) continue;
        chosen[np++] = order[k];
        used += w;
    }
    free(order);
    qsort(chosen, (size_t)np, sizeof *chosen, (int (*)(const void *, const void *))cmp_u32);
    return np;
}

/* Write the source text of the chosen events into out. */
static void retell(const TextContext *t, const Event *e, const int *chosen, int np, char *out, size_t cap) {
    size_t used = 0;
    out[0] = 0;
    for (int k = 0; k < np; k++) {
        const Event *x = &e[chosen[k]];
        size_t a = t->tok[x->a].src, b = t->tok[x->b - 1].src + t->tok[x->b - 1].len;
        for (size_t i = a; i < b && used + 2 < cap; i++) {
            char c = t->src[i];
            out[used++] = (c == '\n' || c == '\r') ? ' ' : c;
        }
        if (used + 2 < cap) out[used++] = ' ';
        out[used] = 0;
    }
}

/* Who it is about: content words with the most strength behind them. */
static void who(const TextContext *t, const Event *e, int n, uint32_t *top, int k) {
    double *score = calloc(t->nvoc, sizeof(double));
    for (int i = 0; i < n; i++) for (int q = 0; q < e[i].nw; q++) score[e[i].w[q]] += e[i].strength;
    for (int r = 0; r < k; r++) {
        uint32_t b = 0;
        for (uint32_t i = 1; i < t->nvoc; i++) if (score[i] > score[b]) b = i;
        top[r] = score[b] > 0 ? b : UINT32_MAX;
        score[b] = -1;
    }
    free(score);
}

void gist_show(const TextContext *t, const Meaning *m, int budget) {
    if (!t->ntok) { printf("  My context is empty. Give me a text first (take).\n"); return; }
    Event *e;
    int n = make_events(t, m, &e);
    integrate(e, n, t, m->dims);
    uint32_t top[6];
    who(t, e, n, top, 6);
    printf("  Read as %d events. It is about:", n);
    for (int k = 0; k < 6 && top[k] != UINT32_MAX; k++) printf("%s %s", k ? "," : "", t->voc[top[k]].word);
    int *chosen = malloc((size_t)n * sizeof *chosen);
    int np = pick(t, e, n, budget, chosen);
    char *out = malloc(16384);
    retell(t, e, chosen, np, out, 16384);
    printf("\n  The gist, in about %d words:\n", budget);
    /* wrap at 76 columns */
    int col = 0;
    printf("    ");
    for (char *w = strtok(out, " "); w; w = strtok(NULL, " ")) {
        int l = (int)strlen(w);
        if (col + l > 72) { printf("\n    "); col = 0; }
        printf("%s ", w);
        col += l + 1;
    }
    printf("\n");
    MIMIR_PROBE("gist.events", n);
    free(out);
    free(chosen);
    free_events(e, n);
}

/* ── the test: 32 Grimm tales against their human plot summaries ── */

typedef struct { char w[32]; } W32;

/* Content words of a text, for scoring (same tokens as reading). */
static int score_words(const Meaning *m, const char *s, W32 *out, int cap) {
    int n = 0, len = 0;
    char w[32];
    for (const char *p = s;; p++) {
        unsigned char c = (unsigned char)*p;
        if (c && (isalpha(c) || (c == '\'' && len))) { if (len < 31) w[len++] = (char)tolower(c); continue; }
        if (c == 0xE2 && (unsigned char)p[1] == 0x80 && (unsigned char)p[2] == 0x99 && len) { if (len < 31) w[len++] = '\''; p += 2; continue; }
        while (len && w[len - 1] == '\'') len--;
        if (len) {
            w[len] = 0;
            if (!meaning_is_function(m, w) && n < cap) snprintf(out[n++].w, 32, "%s", w);
            len = 0;
        }
        if (!c) break;
    }
    return n;
}

/* ROUGE-1 recall and F1, ROUGE-2 recall, of pred against ref. */
static void rouge(const Meaning *m, const char *pred, const char *ref, double *r1, double *f1, double *r2) {
    enum { CAP = 8192 };
    W32 *p = malloc(CAP * sizeof *p), *r = malloc(CAP * sizeof *r);
    int np = score_words(m, pred, p, CAP), nr = score_words(m, ref, r, CAP);
    char *used = calloc((size_t)np + 1, 1);
    int hit = 0;
    for (int i = 0; i < nr; i++)
        for (int j = 0; j < np; j++)
            if (!used[j] && strcmp(r[i].w, p[j].w) == 0) { used[j] = 1; hit++; break; }
    *r1 = nr ? (double)hit / nr : 0;
    double pr = np ? (double)hit / np : 0;
    *f1 = hit ? 2 * *r1 * pr / (*r1 + pr) : 0;
    memset(used, 0, (size_t)np + 1);
    int hit2 = 0;
    for (int i = 0; i + 1 < nr; i++)
        for (int j = 0; j + 1 < np; j++)
            if (!used[j] && strcmp(r[i].w, p[j].w) == 0 && strcmp(r[i + 1].w, p[j + 1].w) == 0) { used[j] = 1; hit2++; break; }
    *r2 = nr > 1 ? (double)hit2 / (nr - 1) : 0;
    free(used); free(p); free(r);
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = malloc((size_t)n + 1);
    if (s) s[fread(s, 1, (size_t)n, f)] = 0;
    fclose(f);
    return s;
}

void gist_test(Reader *r, const Meaning *m) {
    FILE *ix = fopen("data/gist/index.tsv", "r");
    if (!ix) { printf("  No test set (data/gist/index.tsv).\n"); return; }
    char line[1024];
    if (!fgets(line, sizeof line, ix)) { fclose(ix); return; }
    double sum[3][3] = {{0}};
    int tales = 0, beat_rand = 0, beat_lead = 0;
    const int budget = 120;
    printf("  Retelling each tale in %d words; scoring against its Wikipedia plot summary.\n", budget);
    printf("  %-40s %8s %8s %8s\n", "tale", "Mimir", "first120", "random");
    int was_self = r->self_learn;
    r->self_learn = 0;                                  /* score reading, not fixing */
    while (fgets(line, sizeof line, ix)) {
        char *tale = strtok(line, "\t");
        if (!tale) continue;
        char sp[512], rp[512];
        snprintf(sp, sizeof sp, "data/gist/%s.story.txt", tale);
        snprintf(rp, sizeof rp, "data/gist/%s.summary.txt", tale);
        char *ref = slurp(rp);
        TextContext t;
        text_init(&t);
        if (!ref || text_take_file(&t, r, sp) <= 0) { free(ref); text_free(&t); continue; }
        Event *e;
        int n = make_events(&t, m, &e);
        integrate(e, n, &t, m->dims);
        int *chosen = malloc((size_t)n * sizeof *chosen);
        char *out = malloc(16384);
        double s[3][3];
        int np = pick(&t, e, n, budget, chosen);
        retell(&t, e, chosen, np, out, 16384);
        rouge(m, out, ref, &s[0][0], &s[0][1], &s[0][2]);
        /* first 120 words */
        np = 0;
        for (int i = 0, used = 0; i < n && used < budget; i++) { chosen[np++] = i; used += span_words(&t, &e[i]); }
        retell(&t, e, chosen, np, out, 16384);
        rouge(m, out, ref, &s[1][0], &s[1][1], &s[1][2]);
        /* random events (average of 10 draws) */
        s[2][0] = s[2][1] = s[2][2] = 0;
        unsigned seed = 12345u + (unsigned)tales;
        for (int draw = 0; draw < 10; draw++) {
            for (int i = 0; i < n; i++) { seed = seed * 1103515245u + 12345u; e[i].strength = (seed >> 8) & 0xFFFF; }
            np = pick(&t, e, n, budget, chosen);
            retell(&t, e, chosen, np, out, 16384);
            double a, b, c;
            rouge(m, out, ref, &a, &b, &c);
            s[2][0] += a / 10; s[2][1] += b / 10; s[2][2] += c / 10;
        }
        for (int x = 0; x < 3; x++) for (int y = 0; y < 3; y++) sum[x][y] += s[x][y];
        beat_rand += s[0][1] > s[2][1];
        beat_lead += s[0][1] > s[1][1];
        tales++;
        printf("  %-40.40s %7.1f%% %7.1f%% %7.1f%%\n", tale, 100 * s[0][1], 100 * s[1][1], 100 * s[2][1]);
        free(out); free(chosen); free_events(e, n); free(ref); text_free(&t);
    }
    fclose(ix);
    r->self_learn = was_self;
    if (!tales) return;
    printf("  (F1 of content words shared with the human summary)\n\n");
    printf("  Average over %d tales       recall    F1   2-word recall\n", tales);
    const char *names[3] = { "Mimir (workspace 4 + recall)", "First 120 words", "Random events" };
    for (int x = 0; x < 3; x++)
        printf("  %-28s %6.1f%% %6.1f%% %8.1f%%\n", names[x], 100 * sum[x][0] / tales, 100 * sum[x][1] / tales, 100 * sum[x][2] / tales);
    printf("  Mimir beats random on %d of %d tales, the first 120 words on %d of %d.\n", beat_rand, tales, beat_lead, tales);
    MIMIR_PROBE("gist.f1", 100 * sum[0][1] / tales);
}
