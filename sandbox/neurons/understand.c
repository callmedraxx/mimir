/*
 * understand.c — do neurons that grow roles understand what they read?
 * Rules fixed before this was written: docs/UNDERSTAND_COMPARE.md.
 *
 *   A   today's: every token and the page (src/text.c). Answers by the sliding window +
 *       distance baseline of the MCTest paper (Richardson et al. 2013, Figure 4), on its page.
 *   B   the neuron context window (ctxwin.c, grown). The same answerer, on what B holds:
 *       its events' content words in the order the events were made; counts from cue lists.
 *   B+  B, plus parts grown while reading:
 *         role part   event → its frames (verb, who, what, to whom, place/relation phrases,
 *                     negation, either-or, time word) and the times each was read, in a
 *                     byte pool (varints); an event without a verb grows nothing
 *         cue links   a frame's word that is not among the event's content words (a pronoun
 *                     read as "Mary", "wolves" as "wolf") gets a cue link to the event
 *         word map    word → its class and base forms (9 B a word), as B's word → meaning map
 *       Glue words are used to read the roles, then forgotten with the page.
 *
 *   understand babi DIR SPLIT [TASK]    bAbI tasks (data/understand/babi), all three
 *   understand mctest FILE...           MCTest files (data/understand/mctest), all three
 *   understand parse FILE               B+'s frames for each sentence (to check the reader)
 *   understand book BOOK BUDGET_MB      ctxwin's book run with role parts growing (bytes, speed)
 *   Run from the repo root.  Env: SHOW=1 prints each question; LAMBDA=x role weight (MCTest).
 */
#include <limits.h>
#define CTXWIN_NO_MAIN
#include "ctxwin.c"

#define NONE 0xffffffffu
#define OPEN 0xfffffffeu

/* ------------------------------------------------------------------ */
/* the shared word-class table (data/understand/wordclass.tsv)         */
/* ------------------------------------------------------------------ */

typedef struct { char *w, *vb, *nb; char c, v; } LexE;
static LexE *lex; static uint32_t lexmask;
static uint32_t shash(const char *s) { uint32_t h = 2166136261u; for (; *s; s++) { h ^= (unsigned char)*s; h *= 16777619u; } return h; }
static void lex_load(const char *path) {
    FILE *f = fopen(path, "r"); if (!f) { perror(path); exit(1); }
    lexmask = (1 << 17) - 1; lex = calloc(lexmask + 1, sizeof *lex);
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *w = strtok(line, "\t\n"), *c = strtok(0, "\t\n"), *vb = strtok(0, "\t\n"), *nb = strtok(0, "\t\n");
        if (!w || !c || !vb || !nb) continue;
        uint32_t h = shash(w) & lexmask; while (lex[h].w) h = (h + 1) & lexmask;
        lex[h] = (LexE){ strdup(w), strdup(vb), strdup(nb), c[0], c[1] == 'v' };
    }
    fclose(f);
}
static const LexE *lex_get(const char *w) { uint32_t h = shash(w) & lexmask; for (; lex[h].w; h = (h + 1) & lexmask) if (!strcmp(lex[h].w, w)) return &lex[h]; return 0; }

/* special words the reader looks for (by spelling) */
enum { SP_NONE, SP_NOT, SP_NO, SP_LONGER, SP_EITHER, SP_OR, SP_AND, SP_BUT, SP_THEN, SP_THERE,
       SP_YESTERDAY, SP_MORNING, SP_AFTERNOON, SP_EVENING, SP_THIS,
       SP_WHO, SP_WHAT, SP_WHERE, SP_WHY, SP_HOW, SP_WHICH, SP_WHEN, SP_MANY,
       SP_HE, SP_THEY, SP_A, SP_THE, SP_POSS, SP_HER, SP_THAN, SP_OF, SP_TO, SP_WILL, SP_BEFORE, SP_AFTER,
       SP_SUB, SP_REL, SP_UP, SP_DOWN, SP_BACK, SP_I };
static int special(const char *w) {
    static const struct { const char *w; int sp; } T[] = {
        {"not",SP_NOT},{"never",SP_NOT},{"no",SP_NO},{"longer",SP_LONGER},{"either",SP_EITHER},{"or",SP_OR},{"and",SP_AND},{"but",SP_BUT},
        {"then",SP_THEN},{"afterwards",SP_THEN},{"following",SP_THEN},{"there",SP_THERE},{"yesterday",SP_YESTERDAY},{"morning",SP_MORNING},
        {"afternoon",SP_AFTERNOON},{"evening",SP_EVENING},{"this",SP_THIS},{"who",SP_WHO},{"whom",SP_WHO},{"what",SP_WHAT},
        {"where",SP_WHERE},{"why",SP_WHY},{"how",SP_HOW},{"which",SP_WHICH},{"when",SP_WHEN},{"many",SP_MANY},{"much",SP_MANY},
        {"he",SP_HE},{"she",SP_HE},{"him",SP_HE},{"it",SP_HE},{"they",SP_THEY},{"them",SP_THEY},{"a",SP_A},{"an",SP_A},{"the",SP_THE},
        {"his",SP_POSS},{"its",SP_POSS},{"their",SP_POSS},{"her",SP_HER},{"than",SP_THAN},{"of",SP_OF},{"to",SP_TO},
        {"will",SP_WILL},{"before",SP_BEFORE},{"after",SP_AFTER},{"because",SP_SUB},{"while",SP_SUB},{"if",SP_SUB},{"so",SP_SUB},
        {"until",SP_SUB},{"since",SP_SUB},{"that",SP_REL},{"up",SP_UP},{"down",SP_DOWN},{"back",SP_BACK},
        {"i",SP_I},{"you",SP_I},{"we",SP_I},{"me",SP_I},{"us",SP_I},{"my",SP_I},{"your",SP_I},{"our",SP_I}, {0,0} };
    for (int i = 0; T[i].w; i++) if (!strcmp(T[i].w, w)) return T[i].sp;
    return SP_NONE;
}

/* ------------------------------------------------------------------ */
/* word map: word id (in the memory's vocabulary) → class, base forms   */
/* ------------------------------------------------------------------ */

typedef struct { char c, v, sp, known; uint32_t vb, nb; } WC;
typedef struct { WC *w; uint32_t n, cap; size_t counted; } WMap;

static uint32_t vid(TextContext *t, const char *s) {          /* a word's id, added if new (count unchanged) */
    uint32_t id = text_vocab_id(t, NULL, s, TEXT_WORD); t->voc[id].count--; return id;
}
static int vfind(const TextContext *t, const char *s) {         /* a word's id, or −1 */
    if (!t->slot) return -1;
    for (uint32_t h = text_hash(s) & t->mask; t->slot[h]; h = (h + 1) & t->mask) if (!strcmp(t->voc[t->slot[h] - 1].word, s)) return t->slot[h] - 1;
    return -1;
}
static WC *wc_of(WMap *m, TextContext *t, uint32_t id) {
    if (id >= m->cap) { uint32_t n = m->cap ? m->cap : 4096; while (n <= id) n *= 2; m->w = realloc(m->w, n * sizeof *m->w); memset(m->w + m->cap, 0, (n - m->cap) * sizeof *m->w); m->cap = n; }
    WC *e = &m->w[id];
    if (!e->known) {
        char w[64]; snprintf(w, sizeof w, "%s", t->voc[id].word);
        const LexE *l = lex_get(w); size_t n = strlen(w);
        e->known = 1; e->sp = (char)special(w); m->counted += 9;
        if (l) { e->c = l->c; e->v = l->v; e->vb = vid(t, l->vb); e->nb = vid(t, l->nb); }
        else {                                             /* not in the table: by its ending */
            e->c = n > 3 && (!strcmp(w + n - 2, "ed") || !strcmp(w + n - 3, "ing")) ? 'V' : 'N';
            e->vb = e->nb = id; e->v = 0;
            if (e->c == 'N' && n > 3 && w[n - 1] == 's' && w[n - 2] != 's' && w[n - 2] != '\'') {   /* plural of a word the table lacks */
                char b[64]; snprintf(b, sizeof b, "%s", w);
                if (n > 4 && !strcmp(w + n - 3, "ies")) strcpy(b + n - 3, "y");
                else if (n > 4 && (!strcmp(w + n - 4, "ches") || !strcmp(w + n - 4, "shes") || !strcmp(w + n - 3, "xes"))) b[n - 2] = 0;
                else b[n - 1] = 0;
                uint32_t nb = vid(t, b); e = &m->w[id]; e->nb = nb;
            }
        }
        e = &m->w[id];                                     /* vid may have grown the vocabulary, not this map */
    }
    return e;
}

/* ------------------------------------------------------------------ */
/* the reader: tokens → frames                                          */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t id; char c; uint8_t v, sp, poss, neg; uint32_t vb, nb; } Tk;
enum { A_AGENT, A_PAT, A_ATTR, A_PP };
#define K_KIND 1          /* an attribute said with a/an: "Lily is a frog" */
typedef struct { uint8_t role, k; uint32_t prep, head, mod; } Arg;
#define MAXARG 8
enum { F_NEG = 1, F_EITHER = 2, F_FUT = 4, F_Q = 8 };
typedef struct { uint32_t verb, part; uint8_t flags, tc, narg; Arg a[MAXARG]; uint32_t when; int ev; } Frame;

typedef struct {                       /* what the reader carries from sentence to sentence (counted) */
    uint32_t single, group[4]; int ngroup;
} Refs;

typedef struct { uint32_t head, mod, grp[4]; int ngrp; uint8_t det, pron; } NP;
enum { CH_NP, CH_PP, CH_VG, CH_ADJ, CH_WH, CH_AND, CH_OR, CH_SUB, CH_REL, CH_TIME, CH_NEG, CH_EITHER, CH_COMMA, CH_BEFORE };
typedef struct { uint8_t type; NP np; uint32_t prep, verb, part, wtype; uint8_t aux, fut, neg, wh, dangling, tc; } Chunk;

typedef struct {
    TextContext *t; WMap *wm; Refs refs; int question;
    uint32_t w_be, w_have, w_do, w_not, w_you;
} Reader2;

static const char *W(TextContext *t, uint32_t id) { return id == NONE ? "-" : id == OPEN ? "?" : t->voc[id].word; }

/* tokens of one stretch of text → Tk, contractions opened */
static int expand(Reader2 *R, const TextToken *tok, uint32_t a, uint32_t b, Tk *out, int max, uint8_t *ends) {
    TextContext *t = R->t; int n = 0;
    for (uint32_t i = a; i < b && n < max - 2; i++) {
        const TextToken *k = &tok[i]; const char *w = t->voc[k->word].word;
        if (k->kind == TEXT_PUNCT) {
            if (w[0] == ',' && n) { out[n] = (Tk){ .id = NONE, .c = ',' }; ends[n++] = 0; }
            if (k->end && n) ends[n - 1] = 1;
            continue;
        }
        if (k->kind != TEXT_WORD && k->kind != TEXT_LETTER) { if (k->end && n) ends[n - 1] = 1; continue; }
        char s[64]; snprintf(s, sizeof s, "%s", w); size_t L = strlen(s);
        const char *second = 0; int poss = 0, neg = 0;
        if (L > 3 && !strcmp(s + L - 3, "n't")) {
            neg = 1;
            if (!strcmp(s, "can't")) strcpy(s, "can"); else if (!strcmp(s, "won't")) strcpy(s, "will"); else if (!strcmp(s, "shan't")) strcpy(s, "shall");
            else s[L - 3] = 0;
        } else if (L > 2 && s[L - 2] == '\'') {
            char e = s[L - 1]; s[L - 2] = 0;
            if (e == 's') { int sp = special(s); if (sp == SP_HE || sp == SP_REL || sp == SP_THERE || sp == SP_WHAT || sp == SP_WHO || sp == SP_WHERE || sp == SP_HOW || !strcmp(s, "here") || !strcmp(s, "let")) second = "is"; else poss = 1; }
            else if (e == 'm') second = "am"; else if (e == 'd') second = "would";
            else s[L - 2] = '\'';
        } else if (L > 3 && s[L - 3] == '\'') {
            if (!strcmp(s + L - 2, "re")) { s[L - 3] = 0; second = "are"; } else if (!strcmp(s + L - 2, "ve")) { s[L - 3] = 0; second = "have"; } else if (!strcmp(s + L - 2, "ll")) { s[L - 3] = 0; second = "will"; }
        } else if (L > 1 && s[L - 1] == '\'') { s[L - 1] = 0; poss = 1; }
        uint32_t id = vid(t, s); WC *c = wc_of(R->wm, t, id);
        out[n] = (Tk){ id, poss && c->c != 'P' ? 'N' : c->c, (uint8_t)c->v, (uint8_t)c->sp, (uint8_t)poss, 0, c->vb, c->nb }; ends[n++] = 0;
        if (second) { uint32_t i2 = vid(t, second); WC *c2 = wc_of(R->wm, t, i2); out[n] = (Tk){ i2, c2->c, (uint8_t)c2->v, (uint8_t)c2->sp, 0, 0, c2->vb, c2->nb }; ends[n++] = 0; }
        if (neg) { out[n] = (Tk){ R->w_not, 'G', 0, SP_NOT, 0, 0, R->w_not, R->w_not }; ends[n++] = 0; }
        if (k->end) ends[n - 1] = 1;
    }
    return n;
}

/* rule 8's relation words: never part of a compound noun ("the kitchen | south of") */
static int relword(TextContext *t, uint32_t id) { static const char *X[] = { "east", "west", "north", "south", "left", "right", "above", "below", 0 }; for (int i = 0; X[i]; i++) if (!strcmp(t->voc[id].word, X[i])) return 1; return 0; }
static int spw(TextContext *t, uint32_t id) { return id >= OPEN ? SP_NONE : special(t->voc[id].word); }
static int is_nounish(const Tk *k) { return k->c == 'N' || k->c == 'P' || k->c == 'Q'; }
static int is_verbish(const Tk *k) { return k->c == 'V' || k->c == 'X' || k->c == 'M'; }

/* one sentence's tokens → chunks */
static int chunk(Reader2 *R, Tk *k, int n, Chunk *ch, int max) {
    int m = 0; (void)R;
    /* class fixes from neighbours */
    for (int i = 0; i < n; i++) {
        if (k[i].sp == SP_HER) k[i].c = (i + 1 < n && (is_nounish(&k[i + 1]) || k[i + 1].c == 'J' || k[i + 1].v || (k[i + 1].c == 'V' && k[i + 1].nb != k[i + 1].id))) ? 'D' : 'O';
        if (k[i].c == 'D' && k[i].sp != SP_REL && i + 1 < n && (k[i + 1].c == 'V' || k[i + 1].c == 'X') && (k[i + 1].nb != k[i + 1].id || k[i + 1].v)) k[i + 1].c = 'N';
        if (k[i].sp == SP_POSS) k[i].c = 'D';
        if (k[i].sp == SP_THAN) k[i].c = 'I';
        if (k[i].sp == SP_NO && i + 1 < n && k[i + 1].sp == SP_LONGER) { k[i].c = 'G'; k[i].sp = SP_NOT; k[i + 1].c = 'R'; k[i + 1].sp = SP_NONE; }
        if ((k[i].sp == SP_UP || k[i].sp == SP_DOWN) && i && k[i - 1].c == 'V' && !(i + 1 < n && (k[i + 1].c == 'D' || is_nounish(&k[i + 1])) && 0)) k[i].c = 'T';
        if (k[i].sp == SP_BACK) k[i].c = 'R';
        if ((k[i].sp == SP_BEFORE || k[i].sp == SP_AFTER) && i + 1 < n && (k[i + 1].c == 'D' || is_nounish(&k[i + 1]))) k[i].c = 'I';
        if (k[i].sp == SP_YESTERDAY || ((k[i].sp == SP_MORNING || k[i].sp == SP_AFTERNOON || k[i].sp == SP_EVENING) && i && k[i - 1].sp == SP_THIS)) k[i].c = 'Y';
        if (k[i].sp == SP_THIS && i + 1 < n && (k[i + 1].sp == SP_MORNING || k[i + 1].sp == SP_AFTERNOON || k[i + 1].sp == SP_EVENING)) k[i].c = 'Y';
        /* "is tired", "was bored": a participle ending the clause after be is a state */
        if (k[i].c == 'V' && i && k[i - 1].c == 'X' && k[i - 1].vb == R->w_be && k[i].vb != k[i].id && (i + 1 >= n || k[i + 1].c == ',' || k[i + 1].sp == SP_AND || k[i + 1].c == 'S')) k[i].c = 'J';
        if (k[i].sp == SP_THERE && !(i + 1 < n && k[i + 1].c == 'X')) k[i].c = 'R';
        /* after a determiner, possessive or adjective, a verb or adjective that ends the phrase is a noun ("the left") */
        int art = i && (k[i - 1].sp == SP_THE || k[i - 1].sp == SP_A || k[i - 1].sp == SP_POSS || (k[i - 1].sp == SP_HER && k[i - 1].c == 'D') || k[i - 1].poss);
        if (art && (k[i].c == 'V' || k[i].c == 'R' || k[i].c == 'X') && !(i + 1 < n && is_nounish(&k[i + 1]))) k[i].c = 'N';
        if (i && (k[i - 1].c == 'D' || k[i - 1].poss || k[i - 1].c == 'J') && k[i].c == 'V' && i + 1 < n && is_nounish(&k[i + 1])) k[i].c = 'J';
    }
    /* a noun or adjective that can be a verb is the verb when a subject came before and no verb yet */
    { int sawverb = 0, sawnp = 0;
      for (int i = 0; i < n; i++) {
        if (k[i].c == ',' || k[i].sp == SP_AND || k[i].c == 'S' || k[i].sp == SP_REL || k[i].sp == SP_SUB) { sawverb = 0; sawnp = 0; continue; }
        if (is_verbish(&k[i])) { sawverb = 1; continue; }
        if (!sawverb && sawnp && k[i].v && (k[i].c == 'N' || k[i].c == 'J') && !(i && (k[i - 1].c == 'D' || k[i - 1].poss))) { k[i].c = 'V'; sawverb = 1; continue; }
        if (is_nounish(&k[i]) || k[i].c == 'O' || k[i].c == 'E') sawnp = 1;
      } }
    for (int i = 0; i < n && m < max;) {
        Tk *x = &k[i]; Chunk c; memset(&c, 0, sizeof c); c.np.head = c.np.mod = NONE; c.prep = c.verb = c.part = c.wtype = NONE;
        if (x->c == ',') { c.type = CH_COMMA; ch[m++] = c; i++; continue; }
        if (x->sp == SP_WHO || x->sp == SP_WHAT || x->sp == SP_WHERE || x->sp == SP_WHY || x->sp == SP_HOW || x->sp == SP_WHICH || x->sp == SP_WHEN) {
            if (R->question && m == 0) {
                c.type = CH_WH; c.wh = x->sp; i++;
                if ((x->sp == SP_WHAT || x->sp == SP_WHICH) && i < n && is_nounish(&k[i]) && !(i + 1 < n && k[i + 1].c == 'I')) { c.wtype = k[i].nb; i++; }
                else if (x->sp == SP_HOW && i < n && k[i].sp == SP_MANY) { c.wh = SP_MANY; i++; if (i < n && is_nounish(&k[i])) { c.wtype = k[i].nb; i++; } }
                else if (x->sp == SP_HOW && i < n && (k[i].c == 'J' || k[i].c == 'R')) { c.wtype = k[i].id; i++; }
                ch[m++] = c; continue;
            }
            c.type = CH_REL; ch[m++] = c; i++; continue;
        }
        if (x->sp == SP_EITHER) { c.type = CH_EITHER; ch[m++] = c; i++; continue; }
        if (x->sp == SP_OR) { c.type = CH_OR; ch[m++] = c; i++; continue; }
        if (x->sp == SP_AND || x->sp == SP_BUT) { c.type = CH_AND; ch[m++] = c; i++; continue; }
        if (x->sp == SP_THEN) { i++; if (i < n && k[i].sp == SP_REL) i++; continue; }
        if (x->sp == SP_YESTERDAY) { c.type = CH_TIME; c.tc = 1; ch[m++] = c; i++; continue; }
        if (x->sp == SP_THIS && i + 1 < n && (k[i + 1].sp == SP_MORNING || k[i + 1].sp == SP_AFTERNOON || k[i + 1].sp == SP_EVENING)) {
            c.type = CH_TIME; c.tc = k[i + 1].sp == SP_MORNING ? 2 : k[i + 1].sp == SP_AFTERNOON ? 3 : 4; ch[m++] = c; i += 2; continue; }
        if (x->c == 'G' || x->sp == SP_NOT) { c.type = CH_NEG; ch[m++] = c; i++; continue; }
        if ((x->sp == SP_BEFORE || x->sp == SP_AFTER) && !(i + 1 < n && (k[i + 1].c == 'D' || is_nounish(&k[i + 1])))) { c.type = CH_SUB; ch[m++] = c; i++; continue; }
        if (x->sp == SP_SUB || x->c == 'S') { c.type = CH_SUB; ch[m++] = c; i++; continue; }
        if (x->sp == SP_REL) { if (i + 1 < n && is_nounish(&k[i + 1])) x->c = 'D'; else { c.type = CH_REL; ch[m++] = c; i++; continue; } }
        if (x->sp == SP_TO && i + 1 < n && (k[i + 1].c == 'V' || k[i + 1].c == 'X')) {   /* to + verb: a purpose clause */
            c.type = CH_SUB; ch[m++] = c; i++; continue; }
        if (x->c == 'I') {                                 /* preposition + noun phrase, or dangling */
            c.type = CH_PP; c.prep = x->sp == SP_BEFORE || x->sp == SP_AFTER ? x->id : x->id; i++;
            int j = i; while (j < n && (k[j].c == 'D' || k[j].c == 'J' || k[j].c == 'Q' || k[j].poss || (k[j].c == 'R' && j + 1 < n && k[j + 1].c == 'J'))) j++;
            if (j < n && (is_nounish(&k[j]) || k[j].c == 'O' || k[j].sp == SP_I) && !(j == i && k[j].c == 'Q' && 0)) {
                /* phrase from i to its last noun */
                int det = 0; uint32_t mod = NONE; int h = j;
                for (int q = i; q < j; q++) { if (k[q].sp == SP_A) det = 1; if (k[q].c == 'J' || k[q].poss || k[q].c == 'Q') mod = k[q].poss ? k[q].nb : k[q].id; }
                while (h + 1 < n && (is_nounish(&k[h + 1])) && !k[h].poss && !relword(R->t, k[h + 1].id)) { mod = k[h].nb; h++; }
                c.np.head = k[h].c == 'O' || k[h].sp == SP_I ? k[h].id : k[h].nb; c.np.mod = mod; c.np.det = (uint8_t)det; c.np.pron = k[h].sp == SP_HE || k[h].sp == SP_THEY;
                if (c.np.pron) c.np.head = k[h].id;
                i = h + 1;
            } else if (j == i && (i >= n || k[i].c == ',' || k[i].c == 'S')) c.dangling = 1;
            else if (j > i && j <= n && k[j - 1].c == 'J') { c.np.head = k[j - 1].id; i = j; }
            else c.dangling = 1;
            if (c.dangling) c.np.head = OPEN;
            ch[m++] = c; continue;
        }
        if (is_verbish(x) || (x->c == 'R' && i + 1 < n && is_verbish(&k[i + 1]) && x->sp != SP_THERE)) {   /* verb group */
            c.type = CH_VG; uint32_t lastx = NONE, lastv = NONE;
            while (i < n && (is_verbish(&k[i]) || k[i].c == 'G' || (k[i].c == 'R' && k[i].sp != SP_THERE && i + 1 < n && (is_verbish(&k[i + 1]) || k[i + 1].c == 'G')))) {
                if (k[i].c == 'V' && lastv != NONE && k[i].nb != k[i].id) { k[i].c = 'N'; break; }   /* "brought presents": a verb, then a noun */
                if (k[i].c == 'G') c.neg = 1;
                else if (k[i].c == 'V') lastv = k[i].vb;
                else if (k[i].c == 'X' || k[i].c == 'M') { lastx = k[i].vb; if (k[i].sp == SP_WILL) c.fut = 1; }
                i++;
            }
            c.verb = lastv != NONE ? lastv : lastx; c.aux = lastv == NONE;
            if (c.aux && lastx == R->w_do) c.aux = 2;        /* do as an auxiliary: waits for its verb */
            while (i < n && (k[i].c == 'T' || (k[i].c == 'R' && k[i].sp == SP_BACK))) { if (k[i].c == 'T') c.part = k[i].id; i++; }
            ch[m++] = c; continue;
        }
        if (x->c == 'J' && !(i + 1 < n && is_nounish(&k[i + 1]))) { c.type = CH_ADJ; c.np.head = x->id; i++; ch[m++] = c; continue; }
        if (x->c == 'D' || x->c == 'J' || x->c == 'Q' || is_nounish(x) || x->c == 'O' || x->sp == SP_I || x->poss || x->c == 'E') {
            if (x->c == 'E') { i++; continue; }
            int det = 0; uint32_t mod = NONE; int j = i;
            while (j < n && (k[j].c == 'D' || k[j].c == 'J' || k[j].c == 'Q' || (k[j].poss && j + 1 < n && !k[j + 1].poss && k[j + 1].c != 'V'))) {
                if (k[j].sp == SP_A) det = 1; if (k[j].sp == SP_HER || k[j].sp == SP_POSS) mod = R->refs.single;
                if (k[j].c == 'J' || k[j].c == 'Q') mod = k[j].id; if (k[j].poss) mod = k[j].nb; j++; }
            if (j < n && (is_nounish(&k[j]) || k[j].c == 'O' || k[j].sp == SP_I)) {
                int h = j; while (h + 1 < n && is_nounish(&k[h + 1]) && !k[h].poss && k[h].c != 'O' && !relword(R->t, k[h + 1].id)) { mod = k[h].nb; h++; }
                if (k[h].poss && h + 1 < n && is_nounish(&k[h + 1])) { mod = k[h].nb; h++; }
                c.type = CH_NP; c.np.det = (uint8_t)det; c.np.mod = mod;
                c.np.pron = k[h].sp == SP_HE || k[h].sp == SP_THEY;
                c.np.head = c.np.pron || k[h].sp == SP_I ? k[h].id : k[h].nb;
                if (c.np.pron) c.np.head = k[h].id;
                i = h + 1; ch[m++] = c; continue;
            }
            if (j > i && j - 1 >= i && k[j - 1].c == 'J') { c.type = CH_ADJ; c.np.head = k[j - 1].id; c.np.det = det; i = j; ch[m++] = c; continue; }
            i = j > i ? j : i + 1; continue;
        }
        i++;                                               /* anything else is not used */
    }
    return m;
}

static int frame_arg(Frame *f, int role, uint32_t prep, uint32_t head, uint32_t mod, int kind) {
    if (f->narg >= MAXARG) return -1;
    f->a[f->narg] = (Arg){ (uint8_t)role, (uint8_t)kind, prep, head, mod }; return f->narg++;
}
static int find_arg(const Frame *f, int role) { for (int i = 0; i < f->narg; i++) if (f->a[i].role == role) return i; return -1; }

/* pronouns: he/she/it → the latest single agent; they → the latest group */
static int resolve(Reader2 *R, NP *np, uint32_t *out) {
    if (!np->pron) { if (np->ngrp) { for (int i = 0; i < np->ngrp; i++) out[i] = np->grp[i]; return np->ngrp; } out[0] = np->head; return 1; }
    int sp = special(R->t->voc[np->head].word);
    if (sp == SP_THEY && R->refs.ngroup) { for (int i = 0; i < R->refs.ngroup; i++) out[i] = R->refs.group[i]; return R->refs.ngroup; }
    if (R->refs.single != NONE) { out[0] = R->refs.single; return 1; }
    out[0] = np->head; return 1;
}

/* one sentence → frames. Returns how many written to out. */
static int read_sentence(Reader2 *R, Tk *k, int n, Frame *out, int max, int *whslot) {
    static Chunk ch[512]; int m = chunk(R, k, n, ch, 512), nf = 0;
    NP pend[4]; int npend = 0, pend_and = 0, cur = -1, curn = 0;    /* cur: first of the frames being filled (one per agent) */
    uint8_t tc = 0, either = 0, negnext = 0, wh = 0; uint32_t wtype = NONE; int inverted = 0, wh_open_agent = 0, sub = 0;
    for (int i = 0; i < m; i++) {
        Chunk *c = &ch[i];
        switch (c->type) {
        case CH_WH: wh = c->wh; wtype = c->wtype; break;
        case CH_TIME: tc = c->tc; for (int q = cur; cur >= 0 && q < cur + curn; q++) out[q].tc = tc; break;
        case CH_NEG: if (cur >= 0) for (int q = cur; q < cur + curn; q++) out[q].flags |= F_NEG; else negnext = 1; break;
        case CH_EITHER: either = 1; break;
        case CH_COMMA: case CH_AND:
            if (c->type == CH_AND) pend_and = 1;
            if (cur >= 0 && i + 2 < m && ch[i + 1].type == CH_NP && ch[i + 2].type == CH_VG) { cur = -1; curn = 0; npend = 0; pend_and = 0; }
            else if (cur >= 0 && i + 3 < m && ch[i + 1].type == CH_NP && ch[i + 2].type == CH_AND && ch[i + 3].type == CH_NP) { cur = -1; curn = 0; npend = 0; pend_and = 0; }
            break;
        case CH_OR:
            if (cur >= 0 && i + 1 < m && (ch[i + 1].type == CH_NP || ch[i + 1].type == CH_PP)) {
                Frame *f = &out[cur]; int last = f->narg - 1; Chunk *nx = &ch[i + 1];
                if (last >= 0 && (f->a[last].role == A_PP || f->a[last].role == A_ATTR)) {
                    for (int q = cur; q < cur + curn; q++) { frame_arg(&out[q], f->a[last].role, nx->type == CH_PP ? nx->prep : f->a[last].prep, nx->np.head, nx->np.mod, 0); if (either) out[q].flags |= F_EITHER; }
                    i++;
                }
            }
            break;
        case CH_SUB: case CH_REL:
            if (c->type == CH_REL && npend == 0 && cur >= 0) {   /* "the boy who ...": the last phrase is the subject */
                Frame *f = &out[cur]; int last = f->narg - 1;
                if (last >= 0 && f->a[last].role != A_ATTR) { pend[0] = (NP){ .head = f->a[last].head, .mod = f->a[last].mod }; npend = 1; }
            }
            cur = -1; curn = 0; pend_and = 0; sub = 1; break;
        case CH_NP: case CH_ADJ: {
            uint32_t who[4]; int nw = 1;
            if (c->type == CH_NP) nw = resolve(R, &c->np, who); else who[0] = c->np.head;
            if (cur < 0 || out[cur].verb == NONE || (inverted == 1)) {
                if (inverted == 1 && cur >= 0) {             /* "is John in the kitchen", "what did Fred give": the subject after the verb */
                    Frame base = out[cur]; int q0 = cur;
                    for (int w = 0; w < nw && cur + w < max; w++) { if (w) out[cur + w] = base; frame_arg(&out[cur + w], A_AGENT, NONE, who[w], c->np.mod, 0); }
                    curn = nw; nf = cur + nw > nf ? cur + nw : nf; inverted = 2; (void)q0; break;
                }
                if (c->type == CH_ADJ) break;
                if (pend_and && npend && npend < 4) { pend[npend++] = c->np; pend_and = 0; }
                else { pend[0] = c->np; npend = 1; pend_and = 0; }
                break;
            }
            for (int q = cur; q < cur + curn; q++) {
                Frame *f = &out[q];
                int isbe = f->verb == R->w_be;
                if (c->type == CH_ADJ || isbe) frame_arg(f, A_ATTR, NONE, who[0], c->np.mod, c->np.det == 1);
                else if (find_arg(f, A_PAT) < 0 || (i && ch[i - 1].type != CH_NP)) for (int w = 0; w < nw; w++) frame_arg(f, A_PAT, NONE, who[w], c->np.mod, 0);
                else { int p = find_arg(f, A_PAT); Arg old = f->a[p];   /* two objects: the first is the one it went to */
                       f->a[p] = (Arg){ A_PP, 0, vid(R->t, "to"), old.head, old.mod }; frame_arg(f, A_PAT, NONE, who[0], c->np.mod, 0); }
            }
            break; }
        case CH_PP: {
            uint32_t who[4]; int nw = c->np.head == OPEN || c->np.head == NONE ? (who[0] = c->np.head, 1) : resolve(R, &c->np, who);
            if (cur < 0 || out[cur].verb == NONE) {
                if (npend && spw(R->t, c->prep) == SP_OF) { pend[npend - 1].mod = who[0]; break; }   /* "the box of chocolates" */
                if (npend && !c->dangling) { pend[npend - 1].mod = pend[npend - 1].mod == NONE ? who[0] : pend[npend - 1].mod; break; }
                break;
            }
            for (int q = cur; q < cur + curn; q++) {
                Frame *f = &out[q]; int last = f->narg - 1, sp = spw(R->t, c->prep);
                /* "east of the hallway", "to the left of the square", "afraid of wolves", "bigger than the box": a relation */
                if (last >= 0 && sp == SP_OF && (f->a[last].role == A_AGENT || (f->a[last].role == A_PP && spw(R->t, f->a[last].prep) != SP_TO)) && f->a[last].mod == NONE) {
                    f->a[last].mod = who[0];                       /* "the box of chocolates" */
                } else if (last >= 0 && (sp == SP_OF || sp == SP_THAN) && (f->a[last].role == A_ATTR || f->a[last].role == A_PP) && f->a[last].mod == NONE && f->a[last].head < OPEN &&
                    !(f->a[last].k & K_KIND)) {
                    f->a[last].role = A_PP; f->a[last].prep = f->a[last].head; f->a[last].head = who[0]; f->a[last].mod = c->np.mod;
                } else frame_arg(f, A_PP, c->prep, who[0], c->np.mod, 0);
            }
            break; }
        case CH_VG: {
            if (c->aux == 2 && i + 1 < m) { /* "did" waits for its verb: what did Fred give */
                if (R->question || (i + 1 < m && ch[i + 1].type == CH_NP)) { if (cur < 0) { /* subject comes after */ } }
            }
            if (cur >= 0 && out[cur].verb != NONE && npend == 0 && !sub && (inverted == 2 || 1)) {
                /* "did Fred give", "is Mary carrying": the main verb after an inverted subject */
                if (inverted == 2 && (out[cur].verb == R->w_do || out[cur].verb == R->w_be || out[cur].verb == R->w_have || c->aux == 0) && find_arg(&out[cur], A_PAT) < 0 && find_arg(&out[cur], A_ATTR) < 0 && find_arg(&out[cur], A_PP) < 0) {
                    for (int q = cur; q < cur + curn; q++) { out[q].verb = c->verb; out[q].part = c->part; if (c->fut) out[q].flags |= F_FUT; if (c->neg) out[q].flags |= F_NEG; }
                    break;
                }
                /* "took the milk and went to the hall": the same agents, a new frame */
                int base = cur, bn = curn; cur = nf;
                for (int q = 0; q < bn && nf < max; q++) { Frame *f = &out[nf++]; memset(f, 0, sizeof *f); f->verb = c->verb; f->part = c->part; f->tc = tc;
                    int ag = find_arg(&out[base + q], A_AGENT); if (ag >= 0) frame_arg(f, A_AGENT, NONE, out[base + q].a[ag].head, out[base + q].a[ag].mod, 0);
                    if (c->neg) f->flags |= F_NEG; if (c->fut) f->flags |= F_FUT; }
                curn = bn; break;
            }
            uint32_t who[16]; int nw = 0;
            for (int p = 0; p < npend && nw < 12; p++) nw += resolve(R, &pend[p], who + nw);
            cur = nf; curn = 0; sub = 0; inverted = 0;
            int laterverb = 0; for (int j = i + 1; j < m; j++) laterverb |= ch[j].type == CH_VG;
            if (!nw) {
                if (R->question && wh == SP_MANY && wtype != NONE && !laterverb) { who[0] = OPEN; nw = 1; wh_open_agent = 1; }
                else if (R->question && wh && (wh == SP_WHO || wh == SP_WHAT || wh == SP_WHICH) && (!c->aux || (!laterverb && c->verb != R->w_be))) { who[0] = OPEN; nw = 1; wh_open_agent = 1; }
                else if (R->question && i + 1 < m && (ch[i + 1].type == CH_NP)) {
                    if (wh == SP_WHAT && c->verb == R->w_be && i + 2 < m && ch[i + 2].type == CH_PP && spw(R->t, ch[i + 2].prep) == SP_OF && i + 3 >= m) {
                        /* "what is west of the kitchen": the asked one is the subject */
                        Frame *f = &out[nf++]; memset(f, 0, sizeof *f); f->verb = c->verb; f->part = NONE; frame_arg(f, A_AGENT, NONE, OPEN, NONE, 0);
                        frame_arg(f, A_PP, ch[i + 1].np.head, ch[i + 2].np.head, ch[i + 2].np.mod, 0); curn = 1; i += 2; wh_open_agent = 1; break;
                    }
                    inverted = 1;
                    Frame *f = &out[nf++]; memset(f, 0, sizeof *f); f->verb = c->verb; f->part = c->part; f->tc = tc; f->narg = 0;
                    if (c->fut) f->flags |= F_FUT; if (c->neg || negnext) f->flags |= F_NEG; curn = 1; break;
                }
                else { who[0] = NONE; nw = 1; }
            }
            for (int w = 0; w < nw && nf < max; w++) {
                Frame *f = &out[nf++]; memset(f, 0, sizeof *f); f->verb = c->verb; f->part = c->part; f->tc = tc;
                if (who[w] != NONE) frame_arg(f, A_AGENT, NONE, who[w], npend == 1 && !pend[0].pron ? pend[0].mod : NONE, 0);
                if (c->neg || negnext) f->flags |= F_NEG; if (c->fut) f->flags |= F_FUT; curn++;
            }
            negnext = 0; npend = 0; pend_and = 0;
            break; }
        }
    }
    if (either) for (int q = 0; q < nf; q++) out[q].flags |= F_EITHER;
    for (int q = 0; q < nf; q++) { if (tc && !out[q].tc) out[q].tc = tc; if (out[q].part == 0) out[q].part = NONE; }
    /* the asked slot */
    if (whslot) {
        *whslot = -1;
        if (R->question && nf) {
            Frame *f = &out[0];
            for (int q = 0; q < f->narg; q++) if (f->a[q].head == OPEN) *whslot = q;
            if (*whslot < 0 && wh && !wh_open_agent) {
                if (wh == SP_WHERE) *whslot = frame_arg(f, A_PP, vid(R->t, "in"), OPEN, NONE, 0);
                else if (wh == SP_WHY) *whslot = frame_arg(f, A_ATTR, vid(R->t, "why"), OPEN, NONE, 0);
                else if (wh == SP_MANY) *whslot = frame_arg(f, A_PAT, vid(R->t, "many"), OPEN, wtype, 0);
                else if (f->verb == R->w_be && find_arg(f, A_ATTR) < 0) *whslot = frame_arg(f, A_ATTR, NONE, OPEN, wtype, 0);
                else if (find_arg(f, A_PAT) < 0) *whslot = frame_arg(f, A_PAT, NONE, OPEN, wtype, 0);
                else *whslot = frame_arg(f, A_ATTR, NONE, OPEN, wtype, 0);
            }
            if (*whslot >= 0 && wtype != NONE && f->a[*whslot].mod == NONE && wh != SP_MANY) f->a[*whslot].mod = wtype;
        }
    }
    /* what the next sentence's pronouns point to */
    for (int q = 0; q < nf; q++) {
        int ag = find_arg(&out[q], A_AGENT); if (ag < 0 || out[q].a[ag].head == OPEN || out[q].a[ag].head == NONE) continue;
        uint32_t h = out[q].a[ag].head; if (special(R->t->voc[h].word) != SP_NONE) continue;
        int grp = 0; for (int r = q + 1; r < nf && out[r].verb == out[q].verb && find_arg(&out[r], A_AGENT) >= 0; r++) grp++;
        if (grp) { R->refs.ngroup = 0; for (int r = q; r <= q + grp && R->refs.ngroup < 4; r++) R->refs.group[R->refs.ngroup++] = out[r].a[find_arg(&out[r], A_AGENT)].head; q += grp; }
        else R->refs.single = h;
    }
    return nf;
}

static void show_frame(TextContext *t, const Frame *f) {
    static const char *RN[] = { "who", "what", "is", "pp" };
    printf("  [%s%s%s%s%s%s", W(t, f->verb), f->part != NONE ? "+" : "", f->part != NONE ? W(t, f->part) : "", f->flags & F_NEG ? " NOT" : "", f->flags & F_EITHER ? " EITHER" : "", f->flags & F_FUT ? " WILL" : "");
    if (f->tc) printf(" t%d", f->tc);
    for (int i = 0; i < f->narg; i++) {
        const Arg *a = &f->a[i];
        printf(" %s%s%s=%s%s%s%s", RN[a->role], a->role == A_PP ? ":" : "", a->role == A_PP ? W(t, a->prep) : "", a->k & K_KIND ? "a " : "", W(t, a->head), a->mod != NONE ? "/" : "", a->mod != NONE ? W(t, a->mod) : "");
    }
    printf("]");
}

/* a whole stretch of tokens → frames (sentence by sentence) */
static int read_span(Reader2 *R, const TextToken *tok, uint32_t a, uint32_t b, Frame *out, int max, int *whslot) {
    static Tk k[4096]; static uint8_t ends[4096];
    int n = expand(R, tok, a, b, k, 4096, ends), nf = 0, s = 0;
    for (int i = 0; i < n; i++) if (ends[i] || i == n - 1) {
        nf += read_sentence(R, k + s, i + 1 - s, out + nf, max - nf, whslot && nf == 0 ? whslot : 0);
        s = i + 1;
    }
    return nf;
}

static void r2_init(Reader2 *R, TextContext *t, WMap *wm) {
    memset(R, 0, sizeof *R); R->t = t; R->wm = wm; R->refs.single = NONE;
    R->w_be = vid(t, "be"); R->w_have = vid(t, "have"); R->w_do = vid(t, "do"); R->w_not = vid(t, "not"); R->w_you = vid(t, "you");
}

/* ------------------------------------------------------------------ */
/* parse: show B+'s frames for each line of a file                     */
/* ------------------------------------------------------------------ */

static int run_parse(const char *path) {
    TextContext t; text_init(&t); WMap wm = {0}; Reader2 R; r2_init(&R, &t, &wm);
    FILE *f = fopen(path, "r"); if (!f) { perror(path); return 1; }
    char line[8192]; static Frame fr[256];
    while (fgets(line, sizeof line, f)) {
        char *s = line; while (*s >= '0' && *s <= '9') s++; while (*s == ' ') s++;
        char *tab = strchr(s, '\t'); if (tab) *tab = 0;
        size_t L = strlen(s); while (L && (s[L - 1] == '\n' || s[L - 1] == ' ')) s[--L] = 0;
        if (!L) continue;
        R.question = s[L - 1] == '?';
        t.ntok = 0; t.src_len = 0; text_take(&t, NULL, s, L);
        int wh = -1, n = read_span(&R, t.tok, 0, t.ntok, fr, 256, R.question ? &wh : 0);
        printf("%s\n", s); for (int i = 0; i < n; i++) show_frame(&t, &fr[i]); if (R.question) printf("  ask=%d", wh); printf("\n");
    }
    fclose(f); return 0;
}

/* ------------------------------------------------------------------ */
/* B+: the role part, its pool, and the cue links frames add           */
/* ------------------------------------------------------------------ */

typedef struct {
    CW *c; TextContext *t; Reader2 R; WMap wm;
    Part role;                         /* event → 1 + where its latest record starts in the pool */
    uint8_t *pool; size_t np, cap;
    long frames, repeats, links;
} Roles;
static Roles *RL;                       /* the B+ whose hooks are on */

static size_t roles_bytes(Roles *r) { return part_bytes(&r->role) + r->np + r->wm.counted + sizeof(Refs); }
static void pool_room(Roles *r, size_t n) { if (r->np + n > r->cap) { r->cap = r->cap ? r->cap * 2 : 4096; while (r->np + n > r->cap) r->cap *= 2; r->pool = realloc(r->pool, r->cap); } }
static void pv(Roles *r, uint32_t v) { pool_room(r, 5); r->np += vput(r->pool + r->np, v); }
static void pb(Roles *r, uint8_t v) { pool_room(r, 1); r->pool[r->np++] = v; }

static void put_body(Roles *r, const Frame *f) {
    pb(r, f->flags); pb(r, f->tc); pb(r, f->narg); pv(r, f->verb + 1); pv(r, f->part + 1);
    for (int i = 0; i < f->narg; i++) { const Arg *a = &f->a[i]; pb(r, (uint8_t)(a->role | a->k << 4)); if (a->role == A_PP) pv(r, a->prep + 1); pv(r, a->head + 1); pv(r, a->mod + 1); }
}
static size_t get_body(const uint8_t *p, size_t i, Frame *f) {
    f->flags = p[i++]; f->tc = p[i++]; f->narg = p[i++]; f->verb = vget(p, &i) - 1; f->part = vget(p, &i) - 1;
    for (int q = 0; q < f->narg; q++) { Arg *a = &f->a[q]; uint8_t b = p[i++]; a->role = b & 15; a->k = b >> 4; a->prep = a->role == A_PP ? vget(p, &i) - 1 : NONE; a->head = vget(p, &i) - 1; a->mod = vget(p, &i) - 1; }
    return i;
}
/* an event's frames, each with when it was read (a frame read twice comes back twice) */
static int event_frames(Roles *r, int ev, Frame *out, int max) {
    uint32_t *v = (uint32_t *)part_get(&r->role, ev); if (!v) return 0;
    int n = 0; size_t at = *v;
    while (at && n < max) {
        size_t i = at - 1; uint32_t back = vget(r->pool, &i), when = vget(r->pool, &i);
        if (r->pool[i] == 0xff) { i++; size_t body = vget(r->pool, &i); get_body(r->pool, body, &out[n]); }
        else get_body(r->pool, i, &out[n]);
        out[n].when = when; out[n].ev = ev; n++;
        at = back;
    }
    return n;
}
static int same_frame(const Frame *a, const Frame *b) {
    if (a->verb != b->verb || a->part != b->part || a->flags != b->flags || a->tc != b->tc || a->narg != b->narg) return 0;
    for (int i = 0; i < a->narg; i++) if (memcmp(&a->a[i], &b->a[i], sizeof(Arg))) return 0;
    return 1;
}
static void cue_add(B5 *b, uint32_t w, uint32_t ev) {   /* a cue link word → event, kept in order */
    if ((int)w >= b->base.nvocab) return;
    Cue5 *cl = &b->cue[w]; c5_room(c5_decode(cl, 0, 0) + 2); int n = c5_decode(cl, c5_ids, 0);
    int j = n; while (j > 0 && c5_ids[j - 1] > ev) j--;
    if (j > 0 && c5_ids[j - 1] == ev) return;
    memmove(c5_ids + j + 1, c5_ids + j, (n - j) * 4); c5_ids[j] = ev; c5_encode(b, cl, c5_ids, n + 1); RL->links++;
}
static int linkable(Roles *r, uint32_t w) { if (w == NONE || w == OPEN) return 0; int sp = special(r->t->voc[w].word); return sp == SP_NONE || sp == SP_MORNING; }

static void roles_read(CW *c, TextContext *t, uint32_t a, uint32_t b, int ev) {
    Roles *r = RL; static Frame fr[512], old[512];
    r->R.question = 0;
    int n = read_span(&r->R, t->tok, a, b, fr, 512, 0);
    if (ev < 0 || !n) return;
    int no = event_frames(r, ev, old, 512);
    for (int i = 0; i < n; i++) {
        Frame *f = &fr[i]; int same = -1;
        size_t bodypos = 0;
        for (int j = 0; j < no && same < 0; j++) if (same_frame(f, &old[j])) same = j;
        uint32_t *v = (uint32_t *)part_get(&r->role, ev); uint32_t back = v ? *v : 0;
        size_t start = r->np;
        if (same >= 0) {                           /* read again: a short record pointing at the first */
            size_t at = back; Frame tmp;           /* find the body of that frame */
            while (at) { size_t q = at - 1; uint32_t bk = vget(r->pool, &q); vget(r->pool, &q);
                if (r->pool[q] == 0xff) { q++; size_t body = vget(r->pool, &q); get_body(r->pool, body, &tmp); if (same_frame(&tmp, f)) { bodypos = body; break; } }
                else { size_t body = q; get_body(r->pool, q, &tmp); if (same_frame(&tmp, f)) { bodypos = body; break; } }
                at = bk; }
            pv(r, back); pv(r, c->cycle); pb(r, 0xff); pv(r, (uint32_t)bodypos); r->repeats++;
        } else { pv(r, back); pv(r, c->cycle); put_body(r, f); r->frames++; }
        *(uint32_t *)part_grow(&r->role, ev) = (uint32_t)start + 1;
        old[no < 511 ? no++ : no] = *f;
        for (int q = 0; q < f->narg; q++) { if (linkable(r, f->a[q].head)) cue_add(c->b, f->a[q].head, ev); if (linkable(r, f->a[q].mod)) cue_add(c->b, f->a[q].mod, ev); }
    }
}
static size_t roles_more_bytes(CW *c) { (void)c; return RL ? roles_bytes(RL) : 0; }
static void roles_remap(CW *c, const int *map, int old) {
    (void)c; Roles *r = RL; Part *p = &r->role; int n = p->n; if (!n) return;
    uint32_t *k = malloc(n * 4), *v = malloc(n * 4); int m = 0;
    for (int i = 0; i < p->cap; i++) if (p->key[i] != EMPTY && (int)p->key[i] < old && map[p->key[i]] >= 0) { k[m] = map[p->key[i]]; v[m++] = *(uint32_t *)(p->val + (size_t)i * 4); }
    part_clear(p); for (int i = 0; i < m; i++) *(uint32_t *)part_grow(p, k[i]) = v[i];
    free(k); free(v);
}

/* one story's memories: A (page), B (neuron window), B+ (its own window, the same plus role parts) */
typedef struct { TextContext ta, tb, tp; CW *c, *c2; Roles r; } Story;
static const Meaning *MM;
static void story_open(Story *s, int roles) {
    memset(s, 0, sizeof *s); text_init(&s->ta); text_init(&s->tb); text_init(&s->tp);
    cw_nvocab = 1 << 16; s->c = cw_make(MM, &s->tb, (size_t)64 << 20); s->c2 = cw_make(MM, &s->tp, (size_t)64 << 20);
    s->r.c = s->c2; s->r.t = &s->tp; s->r.role = (Part){ .vsize = 4 }; r2_init(&s->r.R, &s->tp, &s->r.wm);
    RL = &s->r; (void)roles;
}
static void cw_free(CW *c) {
    B5 *b = c->b; B4 *a = &b->base;
    for (int w = 0; w < a->nvocab; w++) if (b->cue[w].nbits > 32) free(b->cue[w].u.p);
    free(b->cue); free(a->ev); free(a->chk); free(a->pool); free(a->cue); free(a->xk); free(a->xv); free(a->acc.v); free(a->acc.list); free(a->acc.on); free(b);
    part_clear(&c->ws); part_clear(&c->mean); part_clear(&c->str); part_clear(&c->gist); part_clear(&c->extra); free(c->xp); free(c->map); free(c->mid); free(c);
}
static void story_close(Story *s) {
    cw_free(s->c); cw_free(s->c2);
    part_clear(&s->r.role); free(s->r.pool); free(s->r.wm.w);
    text_free(&s->ta); text_free(&s->tb); text_free(&s->tp); RL = 0; cw_read_hook = 0; cw_more_bytes = 0; cw_remap_hook = 0;
}
static void story_take(Story *s, const char *text) {
    size_t n = strlen(text);
    text_take(&s->ta, NULL, text, n); text_take(&s->ta, NULL, "\n", 1);
    cw_read_hook = 0; cw_more_bytes = 0; cw_remap_hook = 0; cw_take(s->c, &s->tb, text, n);
    cw_read_hook = roles_read; cw_more_bytes = roles_more_bytes; cw_remap_hook = roles_remap; cw_take(s->c2, &s->tp, text, n);
    cw_read_hook = 0; cw_more_bytes = 0; cw_remap_hook = 0;
}

/* ------------------------------------------------------------------ */
/* A and B: sliding window + distance (MCTest paper, Figure 4)          */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t *w; int n, cap; double *ic; int nic; int *unit; int nunit; } Passage;   /* unit: sentence (A) or event (B) of each word */
static int glue(const char *w) { return meaning_is_function(MM, w); }
static void norm_word(const char *in, char *out) { snprintf(out, 64, "%s", in); size_t L = strlen(out); if (L > 2 && !strcmp(out + L - 2, "'s")) out[L - 2] = 0; else if (L > 1 && out[L - 1] == '\'') out[L - 1] = 0; }
static void pass_push(Passage *p, uint32_t w, int unit) { if (p->n == p->cap) { p->cap = p->cap ? p->cap * 2 : 1024; p->w = realloc(p->w, p->cap * 4); p->unit = realloc(p->unit, p->cap * sizeof(int)); } p->unit[p->n] = unit; p->w[p->n++] = w; }
static void pass_free(Passage *p) { free(p->w); free(p->ic); free(p->unit); memset(p, 0, sizeof *p); }

/* A's passage: its page's words in order, counts from the page */
static void pass_A(TextContext *t, Passage *p) {
    memset(p, 0, sizeof *p); int unit = 0;
    for (uint32_t i = 0; i < t->ntok; i++) {
        const TextToken *k = &t->tok[i];
        if (k->kind == TEXT_WORD || k->kind == TEXT_LETTER) { char w[64]; norm_word(t->voc[k->word].word, w); pass_push(p, vid(t, w), unit); }
        if (k->end) unit++;
    }
    p->nunit = unit + 1; p->nic = t->nvoc; p->ic = calloc(p->nic, sizeof(double)); int *cnt = calloc(p->nic, sizeof(int));
    for (int i = 0; i < p->n; i++) cnt[p->w[i]]++;
    for (int i = 0; i < p->nic; i++) p->ic[i] = cnt[i] ? log(1.0 + 1.0 / cnt[i]) : 0;
    free(cnt);
}
/* B's passage: its events' content words, events in the order they were made; counts: events on each word's cue list */
static void pass_B(CW *c, Passage *p) {
    memset(p, 0, sizeof *p); TextContext *t = (TextContext *)c->t; static uint32_t *w; static int wcap;
    for (int ev = 0; ev < c->b->base.nev; ev++) { int m = cw_words(c, ev, &w, &wcap); for (int i = 0; i < m; i++) { char s[64]; norm_word(t->voc[w[i]].word, s); pass_push(p, vid(t, s), ev); } }
    p->nunit = c->b->base.nev; p->nic = t->nvoc; p->ic = calloc(p->nic, sizeof(double));
    for (int i = 0; i < p->n; i++) { uint32_t x = p->w[i]; if (p->ic[x] != 0) continue;
        int cnt = 0; char raw[64]; snprintf(raw, sizeof raw, "%s", t->voc[x].word);
        int id = vfind(t, raw); if (id >= 0 && id < c->b->base.nvocab) cnt = c5_decode(&c->b->cue[id], 0, 0);
        char ps[70]; snprintf(ps, sizeof ps, "%s's", raw); int id2 = vfind(t, ps); if (id2 >= 0 && id2 < c->b->base.nvocab) cnt += c5_decode(&c->b->cue[id2], 0, 0);
        p->ic[x] = log(1.0 + 1.0 / (cnt ? cnt : 1)); }
}
/* words of a question or answer, as ids in the passage's vocabulary (−1: not there) */
static int words_of(TextContext *t, const char *s, int32_t *out, int max) {
    TextContext q; text_init(&q); text_take(&q, NULL, s, strlen(s)); int n = 0;
    for (uint32_t i = 0; i < q.ntok && n < max; i++) if (q.tok[i].kind == TEXT_WORD || q.tok[i].kind == TEXT_LETTER) {
        char w[64]; norm_word(q.voc[q.tok[i].word].word, w); int id = vfind(t, w); int dup = 0;
        for (int j = 0; j < n; j++) dup |= out[j] == id && id >= 0;
        if (!dup) out[n++] = id; }
    text_free(&q); return n;
}
static int in_set(const int32_t *s, int n, int32_t x) { for (int i = 0; i < n; i++) if (s[i] == x) return 1; return 0; }
static double sw_score(const Passage *p, const int32_t *S, int ns) {
    double best = 0;
    for (int j = 0; j < p->n; j++) { double v = 0; for (int w = 0; w < ns && j + w < p->n; w++) if (in_set(S, ns, (int32_t)p->w[j + w])) v += p->ic[p->w[j + w]]; if (v > best) best = v; }
    return best;
}
static double d_score(const Passage *p, TextContext *t, const int32_t *Q, int nq, const int32_t *A, int na) {
    int32_t sq[64], sa[64]; int nsq = 0, nsa = 0;
    for (int i = 0; i < nq; i++) if (Q[i] >= 0 && !glue(t->voc[Q[i]].word) && nsq < 64) { int on = 0; for (int j = 0; j < p->n && !on; j++) on = p->w[j] == (uint32_t)Q[i]; if (on) sq[nsq++] = Q[i]; }
    for (int i = 0; i < na; i++) if (A[i] >= 0 && !glue(t->voc[A[i]].word) && !in_set(Q, nq, A[i]) && nsa < 64) { int on = 0; for (int j = 0; j < p->n && !on; j++) on = p->w[j] == (uint32_t)A[i]; if (on) sa[nsa++] = A[i]; }
    if (!nsq || !nsa || p->n < 2) return 1;
    int best = INT_MAX, lq = -1, la = -1;    /* nearest occurrences, one pass */
    for (int j = 0; j < p->n; j++) {
        if (in_set(sq, nsq, p->w[j])) { lq = j; if (la >= 0 && j - la < best) best = j - la; }
        if (in_set(sa, nsa, p->w[j])) { la = j; if (lq >= 0 && j - lq < best) best = j - lq; }
    }
    return (double)best / (p->n - 1);
}
static double swd(const Passage *p, TextContext *t, const int32_t *Q, int nq, const int32_t *A, int na) {
    int32_t S[128]; int ns = 0;
    for (int i = 0; i < nq && ns < 128; i++) if (!in_set(S, ns, Q[i]) || Q[i] < 0) S[ns++] = Q[i];
    for (int i = 0; i < na && ns < 128; i++) if (!in_set(S, ns, A[i]) || A[i] < 0) S[ns++] = A[i];
    return sw_score(p, S, ns) - d_score(p, t, Q, nq, A, na);
}

/* bAbI by retrieval (A and B): a word answer is the passage word that, with the question,
   scores best; ties go to the word seen last. Yes/no: the last sentence (A) / event (B) with
   the question's first content word says yes if it has all its other content words. */
static int yesno_q(const char *q) { static const char *Y[] = { "is ", "does ", "do ", "did ", "was ", "are ", "can ", 0 }; for (int i = 0; Y[i]; i++) if (!strncasecmp(q, Y[i], strlen(Y[i]))) return 1; return 0; }
static void retrieve_answer(const Passage *p, TextContext *t, const char *q, char *ans) {
    int32_t Q[64]; int nq = words_of(t, q, Q, 64);
    if (yesno_q(q)) {
        int32_t cw[64]; int nc = 0; for (int i = 1; i < nq; i++) if (Q[i] >= 0 && !glue(t->voc[Q[i]].word)) cw[nc++] = Q[i];
        if (!nc) { strcpy(ans, "no"); return; }
        int last = -1; for (int j = 0; j < p->n; j++) if ((int32_t)p->w[j] == cw[0]) last = p->unit[j];
        if (last < 0) { strcpy(ans, "no"); return; }
        int all = 1; for (int x = 1; x < nc; x++) { int on = 0; for (int j = 0; j < p->n; j++) on |= p->unit[j] == last && (int32_t)p->w[j] == cw[x]; all &= on; }
        strcpy(ans, all ? "yes" : "no"); return;
    }
    double best = -1e9; int bpos = -1; uint32_t bw = NONE;
    for (int j = 0; j < p->n; j++) {
        uint32_t w = p->w[j]; if (glue(t->voc[w].word) || in_set(Q, nq, w)) continue;
        int seen = 0; for (int k = j + 1; k < p->n && !seen; k++) seen = p->w[k] == w; if (seen) continue;   /* each word once, at its last place */
        int32_t A1[1] = { (int32_t)w }; double v = swd(p, t, Q, nq, A1, 1);
        if (v > best || (v == best && j > bpos)) { best = v; bpos = j; bw = w; }
    }
    strcpy(ans, bw == NONE ? "nothing" : t->voc[bw].word);
}

/* the book run: attach the reader to the book's TextContext and window on the first event */
static void book_first(CW *c, TextContext *t, uint32_t a, uint32_t b, int ev) {
    Roles *r = RL; r->c = c; r->t = t; r2_init(&r->R, t, &r->wm); cw_read_hook = roles_read; roles_read(c, t, a, b, ev);
}
/* ------------------------------------------------------------------ */
/* B+: answering from frames (rules 1–9 of UNDERSTAND_COMPARE.md)       */
/* ------------------------------------------------------------------ */

typedef struct { Frame *f; int n, cap; } Fr;
static void fr_push(Fr *x, const Frame *f) { if (x->n == x->cap) { x->cap = x->cap ? x->cap * 2 : 256; x->f = realloc(x->f, x->cap * sizeof(Frame)); } x->f[x->n++] = *f; }
static uint64_t fkey(const Frame *f) { return (uint64_t)f->tc << 32 | f->when; }
static int fcmp(const void *a, const void *b) { uint64_t x = fkey(a), y = fkey(b); return x < y ? -1 : x > y; }

/* frames that mention a word, found through that word's cue list */
static void frames_about(Roles *r, uint32_t w, Fr *out) {
    out->n = 0; if (w == NONE || w == OPEN || (int)w >= r->c->b->base.nvocab) return;
    Cue5 *cl = &r->c->b->cue[w]; int m = c5_decode(cl, 0, 0); uint32_t *ids = malloc((m + 1) * 4); c5_decode(cl, ids, 0);
    static Frame tmp[1024];
    for (int i = 0; i < m; i++) { int n = event_frames(r, ids[i], tmp, 1024);
        for (int j = 0; j < n; j++) { int on = tmp[j].verb == w; for (int q = 0; q < tmp[j].narg; q++) on |= tmp[j].a[q].head == w || tmp[j].a[q].mod == w; if (on) fr_push(out, &tmp[j]); } }
    free(ids); qsort(out->f, out->n, sizeof(Frame), fcmp);
}
static void frames_all(Roles *r, Fr *out) {
    out->n = 0; static Frame tmp[1024];
    for (int i = 0; i < r->role.cap; i++) if (r->role.key[i] != EMPTY) { int n = event_frames(r, r->role.key[i], tmp, 1024); for (int j = 0; j < n; j++) fr_push(out, &tmp[j]); }
    qsort(out->f, out->n, sizeof(Frame), fcmp);
}

typedef struct { uint32_t h, m; } Ent;
static int ent_is(Ent e, uint32_t h, uint32_t m) { return e.h == h && (e.m == NONE || e.m == m); }
static uint32_t W_(Roles *r, const char *s) { return vid(r->t, s); }
static int is_w(Roles *r, uint32_t id, const char *s) { return id != NONE && id != OPEN && !strcmp(r->t->voc[id].word, s); }
static int any_w(Roles *r, uint32_t id, const char **set) { for (int i = 0; set[i]; i++) if (is_w(r, id, set[i])) return 1; return 0; }
static const char *PLACE_P[] = { "to", "in", "into", "at", 0 };
static const char *TAKE_V[] = { "take", "get", "grab", "pick", 0 };
static const char *DROP_V[] = { "drop", "leave", "discard", "put", 0 };

/* rule 3: where something is, as a list of places in time order (carried things go with their carrier) */
enum { L_AT, L_NOTIN, L_MAYBE, L_TAKE, L_DROP };
typedef struct { uint64_t key; int type; uint32_t a, b; } LFact;
typedef struct { uint64_t key; uint32_t place, other; int type; } Stop;
static int lf_cmp(const void *x, const void *y) { uint64_t a = ((const LFact *)x)->key, b = ((const LFact *)y)->key; return a < b ? -1 : a > b; }
static int loc_facts(Roles *r, Ent e, LFact *out, int max) {
    Fr fr = {0}; frames_about(r, e.h, &fr); int n = 0;
    for (int i = 0; i < fr.n && n < max; i++) {
        Frame *f = &fr.f[i]; int ag = find_arg(f, A_AGENT), pt = find_arg(f, A_PAT);
        if (ag >= 0 && ent_is(e, f->a[ag].head, f->a[ag].mod) && !is_w(r, f->verb, "fit")) {
            uint32_t pl[2]; int np = 0;
            for (int q = 0; q < f->narg && np < 2; q++) if (f->a[q].role == A_PP && any_w(r, f->a[q].prep, PLACE_P) && f->a[q].head != NONE) pl[np++] = f->a[q].head;
            if (np) {
                if ((f->flags & F_EITHER) && np == 2) out[n++] = (LFact){ fkey(f), L_MAYBE, pl[0], pl[1] };
                else if (f->flags & F_NEG) out[n++] = (LFact){ fkey(f), L_NOTIN, pl[0], NONE };
                else out[n++] = (LFact){ fkey(f), L_AT, pl[np - 1], NONE };
            }
        }
        if (pt >= 0 && ag >= 0 && ent_is(e, f->a[pt].head, f->a[pt].mod) && !(f->flags & F_NEG)) {
            int take = any_w(r, f->verb, TAKE_V) && (!is_w(r, f->verb, "pick") || is_w(r, f->part, "up"));
            int drop = any_w(r, f->verb, DROP_V) && (!is_w(r, f->verb, "put") || is_w(r, f->part, "down"));
            if (take) out[n++] = (LFact){ fkey(f), L_TAKE, f->a[ag].head, NONE };
            else if (drop) out[n++] = (LFact){ fkey(f), L_DROP, f->a[ag].head, NONE };
        }
    }
    free(fr.f); qsort(out, n, sizeof *out, lf_cmp); return n;
}
static int trajectory(Roles *r, Ent e, uint64_t upto, Stop *out, int max, int depth) {
    LFact f[512]; int nf = loc_facts(r, e, f, 512), n = 0; uint32_t carrier = NONE; uint64_t ck = 0;
    #define PUSH(k, p, o, t) do { if (n < max && !(n && out[n - 1].place == (p) && out[n - 1].type == (t) && (t) == L_AT)) out[n++] = (Stop){ k, p, o, t }; } while (0)
    for (int i = 0; i <= nf; i++) {
        uint64_t key = i < nf ? f[i].key : upto; if (key > upto) key = upto;
        if (carrier != NONE && depth < 3) {      /* the carrier's moves while it carried this */
            Stop cs[256]; int nc = trajectory(r, (Ent){ carrier, NONE }, key, cs, 256, depth + 1);
            for (int j = 0; j < nc; j++) if (cs[j].key > ck && cs[j].type == L_AT) PUSH(cs[j].key, cs[j].place, NONE, L_AT);
        }
        if (i == nf || f[i].key > upto) break;
        switch (f[i].type) {
        case L_AT: carrier = NONE; PUSH(f[i].key, f[i].a, NONE, L_AT); break;
        case L_NOTIN: PUSH(f[i].key, f[i].a, NONE, L_NOTIN); break;
        case L_MAYBE: PUSH(f[i].key, f[i].a, f[i].b, L_MAYBE); break;
        case L_TAKE: case L_DROP: {
            if (depth < 3) { Stop cs[256]; int nc = trajectory(r, (Ent){ f[i].a, NONE }, f[i].key, cs, 256, depth + 1);
                if (nc && cs[nc - 1].type == L_AT) PUSH(f[i].key, cs[nc - 1].place, NONE, L_AT); }
            carrier = f[i].type == L_TAKE ? f[i].a : NONE; ck = f[i].key; break; }
        }
    }
    return n;
}

/* rule 5: what someone holds now */
static int holding(Roles *r, uint32_t who, uint32_t *out, int max) {
    Fr fr = {0}; frames_about(r, who, &fr); int n = 0;
    for (int i = 0; i < fr.n; i++) { Frame *f = &fr.f[i]; int ag = find_arg(f, A_AGENT), pt = find_arg(f, A_PAT);
        if (ag < 0 || pt < 0 || f->a[ag].head != who || !any_w(r, f->verb, TAKE_V)) continue;
        uint32_t o = f->a[pt].head; int dup = 0; for (int j = 0; j < n; j++) dup |= out[j] == o; if (dup || n >= max) continue;
        LFact lf[256]; int nl = loc_facts(r, (Ent){ o, NONE }, lf, 256); int held = 0;
        for (int j = 0; j < nl; j++) if (lf[j].type == L_TAKE) held = lf[j].a == who; else if (lf[j].type == L_DROP) held = 0;
        if (held) out[n++] = o; }
    free(fr.f); return n;
}

/* rule 8: relations. Each fact: a REL b. */
typedef struct { Ent a, b; int dx, dy, size; } Rel;   /* a is (dx,dy) from b; size: +1 a bigger than b */
static int rel_of(Roles *r, uint32_t w, int *dx, int *dy, int *size) {
    *dx = *dy = *size = 0; const char *s = r->t->voc[w].word;
    if (!strcmp(s, "east") || !strcmp(s, "right")) *dx = 1; else if (!strcmp(s, "west") || !strcmp(s, "left")) *dx = -1;
    else if (!strcmp(s, "north") || !strcmp(s, "above")) *dy = 1; else if (!strcmp(s, "south") || !strcmp(s, "below")) *dy = -1;
    else if (!strcmp(s, "bigger") || !strcmp(s, "larger")) *size = 1; else if (!strcmp(s, "smaller")) *size = -1;
    else return 0;
    return 1;
}
static int relations(Roles *r, Rel *out, int max) {
    Fr fr = {0}; frames_all(r, &fr); int n = 0;
    for (int i = 0; i < fr.n && n < max; i++) { Frame *f = &fr.f[i]; int ag = find_arg(f, A_AGENT); if (ag < 0 || (f->flags & F_NEG)) continue;
        for (int q = 0; q < f->narg && n < max; q++) { Arg *a = &f->a[q]; if (a->role != A_PP || a->head == NONE) continue;
            Ent A = { f->a[ag].head, f->a[ag].mod }, B = { a->head, a->mod }; int dx, dy, sz;
            if (is_w(r, f->verb, "fit") && (is_w(r, a->prep, "in") || is_w(r, a->prep, "inside") || is_w(r, a->prep, "into"))) out[n++] = (Rel){ A, B, 0, 0, -1 };
            else if (rel_of(r, a->prep, &dx, &dy, &sz)) out[n++] = (Rel){ A, B, dx, dy, sz };
        } }
    free(fr.f); return n;
}
static int ent_eq(Ent a, Ent b) { return a.h == b.h && a.m == b.m; }
/* where each thing is from `from` (dx, dy), by chaining relations; returns 0 if not reached */
static int position(Rel *R, int nr, Ent from, Ent to, int *x, int *y) {
    Ent q[128]; int px[128], py[128], n = 1, h = 0; q[0] = from; px[0] = py[0] = 0;
    while (h < n) { Ent e = q[h]; int ex = px[h], ey = py[h]; h++;
        if (ent_eq(e, to)) { *x = ex; *y = ey; return 1; }
        for (int i = 0; i < nr && n < 128; i++) { if (!R[i].dx && !R[i].dy) continue; Ent o; int ox, oy;
            if (ent_eq(R[i].b, e)) { o = R[i].a; ox = ex + R[i].dx; oy = ey + R[i].dy; } else if (ent_eq(R[i].a, e)) { o = R[i].b; ox = ex - R[i].dx; oy = ey - R[i].dy; } else continue;
            int seen = 0; for (int j = 0; j < n; j++) seen |= ent_eq(q[j], o); if (!seen) { q[n] = o; px[n] = ox; py[n] = oy; n++; } } }
    return 0;
}
static int bigger(Rel *R, int nr, Ent a, Ent b, int depth) {   /* is a bigger than b, by a chain */
    if (depth > 8) return 0;
    for (int i = 0; i < nr; i++) { if (!R[i].size) continue; Ent big = R[i].size > 0 ? R[i].a : R[i].b, small = R[i].size > 0 ? R[i].b : R[i].a;
        if (ent_eq(big, a) && (ent_eq(small, b) || bigger(R, nr, small, b, depth + 1))) return 1; }
    return 0;
}
static int path(Rel *R, int nr, Ent from, Ent to, char *out) {
    Ent q[128]; int prev[128], step[128], n = 1, h = 0; q[0] = from; prev[0] = -1;
    while (h < n) { int cur = h++; Ent e = q[cur];
        if (ent_eq(e, to)) { char st[64]; int k = 0; for (int c = cur; prev[c] >= 0; c = prev[c]) st[k++] = (char)step[c];
            out[0] = 0; for (int i = k - 1; i >= 0; i--) { size_t L = strlen(out); out[L] = st[i]; out[L + 1] = i ? ',' : 0; if (i) out[L + 2] = 0; } return 1; }
        for (int i = 0; i < nr && n < 128; i++) { if (!R[i].dx && !R[i].dy) continue; Ent o; int dx, dy;
            if (ent_eq(R[i].b, e)) { o = R[i].a; dx = R[i].dx; dy = R[i].dy; } else if (ent_eq(R[i].a, e)) { o = R[i].b; dx = -R[i].dx; dy = -R[i].dy; } else continue;
            int seen = 0; for (int j = 0; j < n; j++) seen |= ent_eq(q[j], o); if (seen) continue;
            q[n] = o; prev[n] = cur; step[n] = dx > 0 ? 'e' : dx < 0 ? 'w' : dy > 0 ? 'n' : 's'; n++; } }
    return 0;
}

/* rule 7: a thing's kind ("X is a Y") */
static uint32_t kind_of(Roles *r, uint32_t x) {
    Fr fr = {0}; frames_about(r, x, &fr); uint32_t k = NONE;
    for (int i = 0; i < fr.n; i++) { Frame *f = &fr.f[i]; int ag = find_arg(f, A_AGENT), at = find_arg(f, A_ATTR);
        if (ag >= 0 && at >= 0 && f->a[ag].head == x && (f->a[at].k & K_KIND) && !(f->flags & F_NEG)) k = f->a[at].head; }
    free(fr.f); return k;
}
/* the latest plain attribute (not a kind) of x, said before `upto` */
static uint32_t state_of(Roles *r, uint32_t x, uint64_t upto) {
    Fr fr = {0}; frames_about(r, x, &fr); uint32_t s = NONE;
    for (int i = 0; i < fr.n; i++) { Frame *f = &fr.f[i]; int ag = find_arg(f, A_AGENT), at = find_arg(f, A_ATTR);
        if (fkey(f) <= upto && ag >= 0 && at >= 0 && f->a[ag].head == x && !(f->a[at].k & K_KIND) && !(f->flags & F_NEG) && f->verb == r->R.w_be) s = f->a[at].head; }
    free(fr.f); return s;
}

/* rule 1: the latest frame matching the question's filled slots fills the asked one */
static int arg_match(const Arg *q, const Arg *a) {
    if (q->role != a->role) return 0;
    if (q->role == A_PP && q->prep != a->prep) return 0;
    if (q->head == OPEN) return 1;
    return q->head == a->head && (q->mod == NONE || q->mod == a->mod);
}
static uint32_t latest_fill(Roles *r, const Frame *q, int slot, Fr *cand, uint32_t subject_override) {
    uint32_t ans = NONE;
    for (int i = 0; i < cand->n; i++) { const Frame *f = &cand->f[i];
        if (f->verb != q->verb || (q->part != NONE && f->part != q->part) || (f->flags & F_NEG) != (q->flags & F_NEG)) continue;
        int ok = 1, fill = -1;
        for (int k = 0; k < q->narg && ok; k++) { Arg qa = q->a[k]; if (k == slot) { for (int j = 0; j < f->narg; j++) if (arg_match(&qa, &f->a[j]) && !(qa.role == A_ATTR && (f->a[j].k & K_KIND))) fill = j; continue; }
            if (subject_override != NONE && qa.role == A_AGENT) qa.head = subject_override;
            int on = 0; for (int j = 0; j < f->narg; j++) on |= arg_match(&qa, &f->a[j]); ok = on; }
        if (ok && fill >= 0 && f->a[fill].head != NONE) ans = f->a[fill].head;
    }
    (void)r; return ans;
}

static const char *NUMS[] = { "none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten" };

/* B+'s answer to a bAbI question; returns 0 if its frames give none */
static int plus_answer(Story *s, const char *q, char *ans) {
    Roles *r = &s->r; TextContext *t = &s->tp;
    uint32_t n0 = t->ntok; text_take(t, NULL, q, strlen(q));
    Refs keep = r->R.refs; r->R.question = 1; Frame fq[16]; int slot = -1;
    int nf = read_span(&r->R, t->tok, n0, t->ntok, fq, 16, &slot);
    r->R.refs = keep; r->R.question = 0; t->ntok = n0;
    if (!nf) return 0;
    Frame *f = &fq[0]; uint64_t now = UINT64_MAX;
    int ag = find_arg(f, A_AGENT); Ent subj = { ag >= 0 ? f->a[ag].head : NONE, ag >= 0 ? f->a[ag].mod : NONE };
    if (getenv("SHOW")) { printf("    B+ reads:"); show_frame(t, f); printf(" ask=%d\n", slot); }
    int from = -1, to = -1, before = -1;
    for (int k = 0; k < f->narg; k++) if (f->a[k].role == A_PP) { if (is_w(r, f->a[k].prep, "from")) from = k; if (is_w(r, f->a[k].prep, "to") && f->a[k].head != OPEN) to = k; if (is_w(r, f->a[k].prep, "before")) before = k; }
    /* path: how do you go from X to Y */
    if (from >= 0 && to >= 0 && !strncasecmp(q, "how", 3)) {
        Rel R[256]; int nr = relations(r, R, 256);
        return path(R, nr, (Ent){ f->a[from].head, f->a[from].mod }, (Ent){ f->a[to].head, f->a[to].mod }, ans);
    }
    if (slot < 0) {                                    /* yes / no */
        for (int k = 0; k < f->narg; k++) { Arg *a = &f->a[k]; int dx, dy, sz;
            int fit = is_w(r, f->verb, "fit") && a->role == A_PP;
            if (a->role == A_PP && (fit || rel_of(r, a->prep, &dx, &dy, &sz))) {
                Rel R[256]; int nr = relations(r, R, 256); Ent A = subj, B = { a->head, a->mod };
                if (fit) { strcpy(ans, bigger(R, nr, B, A, 0) ? "yes" : "no"); return 1; }
                if (sz) { strcpy(ans, (sz > 0 ? bigger(R, nr, A, B, 0) : bigger(R, nr, B, A, 0)) ? "yes" : "no"); return 1; }
                int x, y; if (!position(R, nr, B, A, &x, &y)) { strcpy(ans, "no"); return 1; }
                strcpy(ans, (dx > 0 && x > 0) || (dx < 0 && x < 0) || (dy > 0 && y > 0) || (dy < 0 && y < 0) ? "yes" : "no"); return 1;
            }
            if (a->role == A_PP && any_w(r, a->prep, PLACE_P)) {
                Stop st[512]; int n = trajectory(r, subj, now, st, 512, 0); if (!n) { strcpy(ans, "no"); return 1; }
                Stop *l = &st[n - 1];
                if (l->type == L_MAYBE) strcpy(ans, l->place == a->head || l->other == a->head ? "maybe" : "no");
                else if (l->type == L_NOTIN) strcpy(ans, "no");
                else strcpy(ans, l->place == a->head ? "yes" : "no");
                return 1;
            }
        }
        return 0;
    }
    Arg *open = &f->a[slot];
    if (open->role == A_PP && is_w(r, open->prep, "in") && !strncasecmp(q, "where", 5)) {
        if (f->flags & F_FUT) {                        /* rule 9: where will X go — the place nearest its state in meaning */
            uint32_t st = state_of(r, subj.h, now); if (st == NONE) return 0;
            Fr all = {0}; frames_all(r, &all); uint32_t best = NONE; float bs = -2;
            for (int i = 0; i < all.n; i++) for (int k = 0; k < all.f[i].narg; k++) { Arg *a = &all.f[i].a[k];
                if (a->role != A_PP || !any_w(r, a->prep, PLACE_P) || a->head == NONE) continue;
                float sim = meaning_similarity(MM, t->voc[st].word, t->voc[a->head].word); if (sim > bs) { bs = sim; best = a->head; } }
            free(all.f); if (best == NONE) return 0; strcpy(ans, t->voc[best].word); return 1;
        }
        Stop st[512]; int n = trajectory(r, subj, now, st, 512, 0);
        if (before >= 0) {                             /* rule 6: the place before */
            int at = -1; for (int i = 0; i < n; i++) if (st[i].type == L_AT && st[i].place == f->a[before].head) at = i;
            for (int i = at - 1; i >= 0; i--) if (st[i].type == L_AT && st[i].place != f->a[before].head) { strcpy(ans, t->voc[st[i].place].word); return 1; }
            return 0;
        }
        for (int i = n - 1; i >= 0; i--) if (st[i].type == L_AT) { strcpy(ans, t->voc[st[i].place].word); return 1; } else break;
        return 0;
    }
    if (open->role == A_ATTR && is_w(r, open->prep, "why")) {   /* rule 9: why — the latest state */
        uint32_t st = state_of(r, subj.h, now); if (st == NONE) return 0; strcpy(ans, t->voc[st].word); return 1;
    }
    if (open->role == A_PAT && is_w(r, f->verb, "carry")) {     /* rule 5 */
        uint32_t h[16]; int n = holding(r, subj.h, h, 16);
        if (is_w(r, open->prep, "many")) { strcpy(ans, NUMS[n < 10 ? n : 10]); return 1; }
        if (!n) { strcpy(ans, "nothing"); return 1; }
        ans[0] = 0; for (int i = 0; i < n; i++) { if (i) strcat(ans, ","); strcat(ans, t->voc[h[i]].word); } return 1;
    }
    { int dx, dy, sz;                                  /* rule 8: what is X east of / what is west of X */
      if (open->role == A_PP && rel_of(r, open->prep, &dx, &dy, &sz)) { Rel R[256]; int nr = relations(r, R, 256); uint32_t a = NONE;
          for (int i = 0; i < nr; i++) { if (ent_eq(R[i].a, subj) && R[i].dx == dx && R[i].dy == dy && (dx || dy)) a = R[i].b.h; if (ent_eq(R[i].b, subj) && R[i].dx == -dx && R[i].dy == -dy && (dx || dy)) a = R[i].a.h; }
          if (a == NONE) return 0; strcpy(ans, t->voc[a].word); return 1; }
      if (open->role == A_AGENT) for (int k = 0; k < f->narg; k++) if (f->a[k].role == A_PP && rel_of(r, f->a[k].prep, &dx, &dy, &sz) && (dx || dy)) {
          Rel R[256]; int nr = relations(r, R, 256); Ent B = { f->a[k].head, f->a[k].mod }; uint32_t a = NONE;
          for (int i = 0; i < nr; i++) { if (ent_eq(R[i].b, B) && R[i].dx == dx && R[i].dy == dy) a = R[i].a.h; if (ent_eq(R[i].a, B) && R[i].dx == -dx && R[i].dy == -dy) a = R[i].b.h; }
          if (a == NONE) return 0; strcpy(ans, t->voc[a].word); return 1; } }
    /* rule 1, then rule 7 (kinds) */
    Fr cand = {0}; uint32_t key = subj.h != NONE && subj.h != OPEN ? subj.h : NONE;
    for (int k = 0; key == NONE && k < f->narg; k++) if (k != slot && f->a[k].head != NONE && f->a[k].head != OPEN) key = f->a[k].head;
    if (key == NONE) return 0;
    frames_about(r, key, &cand);
    Frame qq = *f; if (open->role == A_ATTR && open->mod != NONE) qq.a[slot].mod = NONE;
    uint32_t a = latest_fill(r, &qq, slot, &cand, NONE);
    if (a == NONE && subj.h != NONE && subj.h != OPEN) {
        uint32_t k = kind_of(r, subj.h);
        if (k != NONE) {
            Fr kc = {0}; frames_about(r, k, &kc); a = latest_fill(r, &qq, slot, &kc, k);   /* what Ys have, X has */
            if (a == NONE) {                             /* what another Y has, X has */
                Fr all = {0}; frames_all(r, &all);
                for (int i = 0; i < all.n; i++) { Frame *g = &all.f[i]; int gag = find_arg(g, A_AGENT), gat = find_arg(g, A_ATTR);
                    if (gag < 0 || gat < 0 || !(g->a[gat].k & K_KIND) || g->a[gat].head != k || g->a[gag].head == subj.h) continue;
                    Fr oc = {0}; frames_about(r, g->a[gag].head, &oc); uint32_t b = latest_fill(r, &qq, slot, &oc, g->a[gag].head);
                    for (int j = 0; j < oc.n && b != NONE; j++) {}
                    if (b != NONE && b != k) a = b; free(oc.f); }
                free(all.f);
            }
            free(kc.f);
        }
    }
    free(cand.f);
    if (a == NONE) return 0;
    strcpy(ans, t->voc[a].word); return 1;
}
/* ------------------------------------------------------------------ */
/* bAbI                                                                 */
/* ------------------------------------------------------------------ */

static int same_answer(const char *got, const char *want) {
    if (!strchr(want, ',') || !strchr(got, ',')) return !strcasecmp(got, want);
    /* lists: the same things in any order (paths keep their order: they never get here as sets) */
    char a[256], b[256]; snprintf(a, sizeof a, "%s", got); snprintf(b, sizeof b, "%s", want);
    int na = 1, nb = 1; for (char *p = a; *p; p++) na += *p == ','; for (char *p = b; *p; p++) nb += *p == ',';
    if (na != nb) return 0;
    for (char *x = strtok(b, ","); x; x = strtok(0, ",")) { char c[260]; snprintf(c, sizeof c, ",%s,", x); char d[260]; snprintf(d, sizeof d, ",%s,", got); if (!strstr(d, c)) return 0; }
    return 1;
}

static int run_babi(const char *dir, const char *split, int only) {
    double sumA = 0, sumB = 0, sumP = 0; int passA = 0, passB = 0, passP = 0, ntask = 0; long fallback_all = 0;
    printf("task   A today's   B neurons   B+ roles   (B+ answered from frames)\n");
    for (int task = 1; task <= 20; task++) {
        if (only && task != only) continue;
        char path[512]; snprintf(path, sizeof path, "%s/qa%d_%s.txt", dir, task, split);
        FILE *f = fopen(path, "r"); if (!f) { perror(path); return 1; }
        char line[4096]; Story s; int open = 0, nq = 0, okA = 0, okB = 0, okP = 0, fromframes = 0;
        while (fgets(line, sizeof line, f)) {
            char *p = line; int num = atoi(p); while (*p >= '0' && *p <= '9') p++; while (*p == ' ') p++;
            if (num == 1) { if (open) story_close(&s); story_open(&s, 1); open = 1; }
            char *tab = strchr(p, '\t');
            if (!tab) { size_t L = strlen(p); while (L && (p[L - 1] == '\n' || p[L - 1] == '\r')) p[--L] = 0; story_take(&s, p); continue; }
            *tab = 0; char *want = tab + 1; char *t2 = strchr(want, '\t'); if (t2) *t2 = 0;
            char q[1024]; snprintf(q, sizeof q, "%s", p); size_t L = strlen(q); while (L && q[L - 1] == ' ') q[--L] = 0;
            char aA[256], aB[256], aP[256];
            Passage pa, pb; pass_A(&s.ta, &pa); pass_B(s.c, &pb);
            retrieve_answer(&pa, &s.ta, q, aA); retrieve_answer(&pb, &s.tb, q, aB);
            if (plus_answer(&s, q, aP)) fromframes++; else { Passage pp; pass_B(s.c2, &pp); retrieve_answer(&pp, &s.tp, q, aP); pass_free(&pp); }
            pass_free(&pa); pass_free(&pb);
            int a = same_answer(aA, want), b = same_answer(aB, want), c = same_answer(aP, want);
            okA += a; okB += b; okP += c; nq++;
            if (getenv("SHOW") && (!c || atoi(getenv("SHOW")) > 1)) printf("  qa%d %s  want %s | A %s | B %s | B+ %s%s\n", task, q, want, aA, aB, aP, c ? "" : "   <-- B+ wrong");
        }
        if (open) story_close(&s); fclose(f);
        double A = 100.0 * okA / nq, B = 100.0 * okB / nq, P = 100.0 * okP / nq;
        printf("qa%-3d  %8.1f%%  %8.1f%%  %8.1f%%   (%d of %d)\n", task, A, B, P, fromframes, nq);
        sumA += A; sumB += B; sumP += P; passA += A >= 95; passB += B >= 95; passP += P >= 95; ntask++; fallback_all += nq - fromframes;
    }
    printf("mean   %8.2f%%  %8.2f%%  %8.2f%%\npassed %8d    %8d    %8d     (of %d, at >= 95%%)\n", sumA / ntask, sumB / ntask, sumP / ntask, passA, passB, passP, ntask);
    printf("RESULT babi split=%s U1_A=%.2f U1_B=%.2f U1_P=%.2f U2_A=%d U2_B=%d U2_P=%d fallback=%ld\n", split, sumA / ntask, sumB / ntask, sumP / ntask, passA, passB, passP, fallback_all);
    return 0;
}

/* ------------------------------------------------------------------ */
/* MCTest                                                               */
/* ------------------------------------------------------------------ */

/* B+'s role score for one answer: the question read as a frame, the asked slot filled with the
   answer's words, against every frame of the story: the best share of its filled slots that match */
static double role_score(Story *s, const char *q, const char *opt) {
    Roles *r = &s->r; TextContext *t = &s->tp;
    uint32_t n0 = t->ntok; text_take(t, NULL, q, strlen(q));
    Refs keep = r->R.refs; r->R.question = 1; Frame fq[16]; int slot = -1;
    int nf = read_span(&r->R, t->tok, n0, t->ntok, fq, 16, &slot);
    t->ntok = n0;
    /* the answer: its frames if it is a clause (said by the question's subject), else its words fill the slot */
    uint32_t n1 = t->ntok; text_take(t, NULL, opt, strlen(opt));
    r->R.question = 0; Frame fo[16]; int no = read_span(&r->R, t->tok, n1, t->ntok, fo, 16, 0);
    static uint32_t ow[64]; int now = 0;
    for (uint32_t i = n1; i < t->ntok && now < 64; i++) if (t->tok[i].kind == TEXT_WORD && is_content(t, MM, t->tok[i].word)) ow[now++] = wc_of(&r->wm, t, t->tok[i].word)->nb;
    r->R.refs = keep; t->ntok = n0;
    if (!nf) return 0;
    Frame H[16]; int nh = 0;
    if (slot >= 0) for (int i = 0; i < now && nh < 16; i++) { H[nh] = fq[0]; H[nh].a[slot].head = ow[i]; H[nh].a[slot].mod = NONE; nh++; }
    if (slot < 0 || !now) { H[nh++] = fq[0]; }
    int ag = find_arg(&fq[0], A_AGENT);
    for (int i = 0; i < no && nh < 16; i++) if (fo[i].verb != NONE) { H[nh] = fo[i]; if (find_arg(&H[nh], A_AGENT) < 0 && ag >= 0) frame_arg(&H[nh], A_AGENT, NONE, fq[0].a[ag].head, fq[0].a[ag].mod, 0); nh++; }
    Fr all = {0}; frames_all(r, &all); double best = 0;
    for (int h = 0; h < nh; h++) { Frame *x = &H[h]; int slots = 1, hit = 0;
        for (int k = 0; k < x->narg; k++) if (x->a[k].head != OPEN && x->a[k].head != NONE) slots++;
        for (int i = 0; i < all.n; i++) { Frame *g = &all.f[i]; int m = g->verb == x->verb;
            for (int k = 0; k < x->narg; k++) { if (x->a[k].head == OPEN || x->a[k].head == NONE) continue; int on = 0;
                for (int j = 0; j < g->narg && !on; j++) on = g->a[j].role == x->a[k].role && g->a[j].head == x->a[k].head && (x->a[k].role != A_PP || g->a[j].prep == x->a[k].prep);
                m += on; }
            if (m > hit) hit = m; }
        double v = (double)hit / slots; if (v > best) best = v; }
    free(all.f); return best;
}

static int run_mctest(int nfile, char **files) {
    double lambda = getenv("LAMBDA") ? atof(getenv("LAMBDA")) : 2.0;   /* chosen on MC160 + MC500 train and dev */
    for (int fi = 0; fi < nfile; fi++) {
        FILE *f = fopen(files[fi], "r"); if (!f) { perror(files[fi]); return 1; }
        static char line[65536]; Story s; int open = 0; int n = 0, okA = 0, okB = 0, okP = 0, nS = 0, okAS = 0, okBS = 0, okPS = 0;
        Passage pa = {0}, pb = {0}, pp = {0}; int have = 0;
        while (fgets(line, sizeof line, f)) {
            size_t L = strlen(line); while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
            if (line[0] == 'S') { if (open) { story_close(&s); pass_free(&pa); pass_free(&pb); pass_free(&pp); } story_open(&s, 1); open = 1; have = 0; continue; }
            if (line[0] == 'T') { story_take(&s, line + 2); continue; }
            if (line[0] != 'Q') continue;
            if (!have) { pass_A(&s.ta, &pa); pass_B(s.c, &pb); pass_B(s.c2, &pp); have = 1; }
            char *fld[9]; int k = 0; for (char *p = strtok(line, "\t"); p && k < 9; p = strtok(0, "\t")) fld[k++] = p;
            if (k < 8) continue;
            int single = !strcmp(fld[1], "single"); const char *q = fld[2]; int want = fld[7][0] - 'A';
            int32_t QA[128], QB[128], QP[128]; int nqa = words_of(&s.ta, q, QA, 128), nqb = words_of(&s.tb, q, QB, 128), nqp = words_of(&s.tp, q, QP, 128);
            double bestA = -1e9, bestB = -1e9, bestP = -1e9; int cA = 0, cB = 0, cP = 0;
            for (int o = 0; o < 4; o++) {
                int32_t AA[128], AB[128], AP[128]; int naa = words_of(&s.ta, fld[3 + o], AA, 128), nab = words_of(&s.tb, fld[3 + o], AB, 128), nap = words_of(&s.tp, fld[3 + o], AP, 128);
                double va = swd(&pa, &s.ta, QA, nqa, AA, naa), vb = swd(&pb, &s.tb, QB, nqb, AB, nab);
                double vp = swd(&pp, &s.tp, QP, nqp, AP, nap) + lambda * role_score(&s, q, fld[3 + o]);
                if (va > bestA) { bestA = va; cA = o; } if (vb > bestB) { bestB = vb; cB = o; } if (vp > bestP) { bestP = vp; cP = o; }
            }
            okA += cA == want; okB += cB == want; okP += cP == want; n++;
            if (single) { nS++; okAS += cA == want; okBS += cB == want; okPS += cP == want; }
            if (getenv("SHOW")) printf("  %s  want %c | A %c | B %c | B+ %c\n", q, 'A' + want, 'A' + cA, 'A' + cB, 'A' + cP);
        }
        if (open) { story_close(&s); pass_free(&pa); pass_free(&pb); pass_free(&pp); }
        fclose(f);
        const char *b = strrchr(files[fi], '/'); b = b ? b + 1 : files[fi];
        printf("%-18s %4d questions   A %5.2f%%   B %5.2f%%   B+ %5.2f%%     single: A %5.2f B %5.2f B+ %5.2f   multiple: A %5.2f B %5.2f B+ %5.2f\n", b, n,
               100.0 * okA / n, 100.0 * okB / n, 100.0 * okP / n, 100.0 * okAS / nS, 100.0 * okBS / nS, 100.0 * okPS / nS,
               100.0 * (okA - okAS) / (n - nS), 100.0 * (okB - okBS) / (n - nS), 100.0 * (okP - okPS) / (n - nS));
        printf("RESULT mctest file=%s n=%d A=%.2f B=%.2f P=%.2f lambda=%.3f\n", b, n, 100.0 * okA / n, 100.0 * okB / n, 100.0 * okP / n, lambda);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* War and Peace with roles growing: ctxwin's book run                  */
/* ------------------------------------------------------------------ */

static Roles book_roles;
static int run_book_roles(const Meaning *m, const char *book, double mb) {
    /* the book's own TextContext is made inside run_book; the reader attaches to it on the first event */
    memset(&book_roles, 0, sizeof book_roles); book_roles.role = (Part){ .vsize = 4 }; RL = &book_roles;
    cw_read_hook = book_first; cw_more_bytes = roles_more_bytes; cw_remap_hook = roles_remap;
    int rc = run_book(m, book, mb);
    printf("B+ role parts: %d events grew one; frames %ld (+%ld read again); extra cue links %ld; role bytes %zu (part %zu, pool %zu, word map %zu)\n",
           book_roles.role.n, book_roles.frames, book_roles.repeats, book_roles.links, roles_bytes(&book_roles), part_bytes(&book_roles.role), book_roles.np, book_roles.wm.counted);
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "understand babi DIR SPLIT [TASK] | mctest FILE... | parse FILE | book BOOK MB\n"); return 1; }
    lex_load(getenv("WORDCLASS") ? getenv("WORDCLASS") : "data/understand/wordclass.tsv");
    if (!strcmp(argv[1], "parse") && argc >= 3) return run_parse(argv[2]);
    static Meaning m; meaning_init(&m);
    if (meaning_load(&m, getenv("MEANING") ? getenv("MEANING") : MEANING_PATH) <= 0) { fprintf(stderr, "no meanings\n"); return 1; }
    meaning_norms(&m); MM = &m;
    if (!strcmp(argv[1], "babi") && argc >= 4) return run_babi(argv[2], argv[3], argc >= 5 ? atoi(argv[4]) : 0);
    if (!strcmp(argv[1], "mctest") && argc >= 3) return run_mctest(argc - 2, argv + 2);
    if (!strcmp(argv[1], "book") && argc >= 4) return run_book_roles(&m, argv[2], atof(argv[3]));
    return 1;
}
