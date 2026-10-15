/*
 * facts.c — how Mimir reads a sentence into facts: each word's class (data/understand/wordclass.tsv),
 * the sentence cut into chunks, and its frames (verb, who, what, to whom, is-what, place/time
 * phrases, negation); a question's statement is read the same way when answering. Used by bg.c.
 *
 * Gathered on 2026-10-14 from the parts of compare.c, grow.c, ctxwin.c, understand.c and know.c
 * that the brain still uses (docs/JOURNAL.md); the rest of those experiments is in git history,
 * tag before-cleanup. Tokens come from the app's src/text.c.
 */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define vocab_id text_vocab_id
#include "../../src/text.c"
#undef vocab_id

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static inline int vput(uint8_t *p, uint32_t v) { int n = 0; while (v >= 128) { p[n++] = (v & 127) | 128; v >>= 7; } p[n++] = v; return n; }

/* docs/COMPREHEND.md R2 (RD=1): the reader reads names ("a boy named Teo", "is called", "the baker, Mr. Oduya,"),
   "made of", "without", "X, not Y" (the "not" only for Y), passives as actives, a capital mid-sentence as a name,
   "Mr." not as a sentence end; questions with "is" read as "X is <option>". RD=0: as before. */
static int RD = 0;
#define NONE 0xffffffffu
#define OPEN 0xfffffffeu

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

typedef struct { char c, v, sp, known; uint32_t vb, nb; } WC;
typedef struct { WC *w; uint32_t n, cap; size_t counted; } WMap;

static uint32_t vid(TextContext *t, const char *s) {          /* a word's id, added if new (count unchanged) */
    uint32_t id = text_vocab_id(t, NULL, s, TEXT_WORD); t->voc[id].count--; return id;
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

typedef struct { uint32_t id; char c; uint8_t v, sp, poss, neg; uint32_t vb, nb; uint8_t cap; } Tk;
enum { A_AGENT, A_PAT, A_ATTR, A_PP };
#define K_KIND 1          /* an attribute said with a/an: "Lily is a frog" */
typedef struct { uint8_t role, k; uint32_t prep, head, mod; } Arg;
#define MAXARG 8
enum { F_NEG = 1, F_EITHER = 2, F_FUT = 4, F_Q = 8, F_PASS = 16 };
typedef struct { uint32_t verb, part; uint8_t flags, tc, narg; Arg a[MAXARG]; uint32_t when; int ev; } Frame;

typedef struct {                       /* what the reader carries from sentence to sentence (counted) */
    uint32_t single, group[4]; int ngroup;
    uint32_t person, thing;            /* reader 2: the latest named person, the latest thing */
} Refs;

typedef struct { uint32_t head, mod, grp[4]; int ngrp; uint8_t det, pron; } NP;
typedef struct { uint32_t h, hm, k, km; } NameF;      /* RD: "h (hm) is a k (km)", read from a name */
static NameF nmf[16]; static int nnmf;
enum { CH_NP, CH_PP, CH_VG, CH_ADJ, CH_WH, CH_AND, CH_OR, CH_SUB, CH_REL, CH_TIME, CH_NEG, CH_EITHER, CH_COMMA, CH_BEFORE };
typedef struct { uint8_t type; NP np; uint32_t prep, verb, part, wtype; uint8_t aux, fut, neg, wh, dangling, tc, nt, be, infl; } Chunk;

typedef struct {
    TextContext *t; WMap *wm; Refs refs; int question;
    uint32_t w_be, w_have, w_do, w_not, w_you;
    uint32_t *name, nname, nsent;         /* RD: the sentence (+1) where each word was last read as a name */
} Reader2;
static void r2_name(Reader2 *R, uint32_t id) { if (id >= R->nname) { uint32_t n = R->nname ? R->nname : 4096; while (n <= id) n *= 2; R->name = realloc(R->name, n * 4); memset(R->name + R->nname, 0, (n - R->nname) * 4); R->nname = n; } R->name[id] = R->nsent + 1; }
static int r2_is_name(const Reader2 *R, uint32_t id) { return id < R->nname && R->name[id] && R->nsent + 1 - R->name[id] < 50; }   /* a name for the next 50 sentences */

/* RD: "Mr." and the like do not end a sentence when one is re-read from the text */
static int abbrev(const char *w) { static const char *A[] = { "mr", "mrs", "ms", "dr", "st", 0 }; for (int i = 0; A[i]; i++) if (!strcasecmp(w, A[i])) return 1; return 0; }
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
        const char *second = 0; int poss = 0, neg = 0, noend = 0;
        /* RD: "said no." is not the abbreviation "No." (number), and it ends the sentence */
        if (RD && !strcmp(s, "number") && k->len == 3 && k->src + 3 <= t->src_len && !strncasecmp(t->src + k->src, "no.", 3)) { strcpy(s, "no"); L = 2; noend = 1; }
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
        out[n] = (Tk){ id, poss && c->c != 'P' ? 'N' : c->c, (uint8_t)c->v, (uint8_t)c->sp, (uint8_t)poss, 0, c->vb, c->nb, 0 };
        if (RD) {
            if (!strcmp(s, "no")) { out[n].c = 'N'; out[n].vb = out[n].nb = id; }       /* "said no" */
            /* a name: a capital in the middle of a sentence ("Old Watch", "the Grey Heron"), or a capital word the table
               does not have ("Nadia"); a word the table calls a name but written small is not one ("van", "baker") */
            int cap = k->src < t->src_len && isupper((unsigned char)t->src[k->src]) && c->sp == SP_NONE && c->c != 'I' && c->c != 'X' && c->c != 'M' && c->c != 'D';
            int capnext = 0; for (uint32_t j = i + 1; j < b; j++) { if (tok[j].kind != TEXT_WORD) break; capnext = tok[j].src < t->src_len && isupper((unsigned char)t->src[tok[j].src]); break; }
            if (cap && ((n && !ends[n - 1]) || !lex_get(s) || c->c == 'P' || capnext)) { out[n].c = 'P'; if (!lex_get(s)) out[n].vb = out[n].nb = id; out[n].cap = 1; r2_name(R, id); }
            else if (c->c == 'P' && !cap) out[n].c = 'N';
        }
        ends[n++] = 0; if (noend) ends[n - 1] = 1;
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
    int m = 0;
    /* class fixes from neighbours */
    for (int i = 0; i < n; i++) {
        if (k[i].sp == SP_HER) k[i].c = (i + 1 < n && (is_nounish(&k[i + 1]) || k[i + 1].c == 'J' || k[i + 1].v || (k[i + 1].c == 'V' && k[i + 1].nb != k[i + 1].id))) ? 'D' : 'O';
        if (k[i].c == 'D' && k[i].sp != SP_REL && i + 1 < n && (k[i + 1].c == 'V' || k[i + 1].c == 'X') && (k[i + 1].nb != k[i + 1].id || k[i + 1].v)) k[i + 1].c = 'N';
        if (k[i].sp == SP_POSS) k[i].c = 'D';
        if (k[i].sp == SP_THAN) k[i].c = 'I';
        if (k[i].sp == SP_NO && i + 1 < n && k[i + 1].sp == SP_LONGER) { k[i].c = 'G'; k[i].sp = SP_NOT; k[i + 1].c = 'R'; k[i + 1].sp = SP_NONE; }
        if ((k[i].sp == SP_UP || k[i].sp == SP_DOWN) && i && k[i - 1].c == 'V' && !(i + 1 < n && (k[i + 1].c == 'D' || is_nounish(&k[i + 1])) && 0)) k[i].c = 'T';
        if (k[i].sp == SP_BACK) k[i].c = 'R';
        /* RD: after "did" / "did not", a word that can be a verb, or one the table lacks, is the verb ("did not purr") */
        if (RD && k[i].c == 'N' && i && ((k[i - 1].c == 'X' && k[i - 1].vb == R->w_do) || (k[i - 1].c == 'G' && i > 1 && k[i - 2].c == 'X' && k[i - 2].vb == R->w_do))
            && k[i].id < OPEN && (k[i].v || !lex_get(R->t->voc[k[i].id].word))) k[i].c = 'V';
        {
            static const char *PT[] = { "over", "out", "away", "off", "along", "around", "by", "in", "on", "down", "up", 0 };
            int pt = 0; for (int x = 0; PT[x] && k[i].id < OPEN; x++) pt |= !strcmp(R->t->voc[k[i].id].word, PT[x]);
            /* "came over", "ran away": a particle when no noun phrase follows */
            if (pt && i && k[i - 1].c == 'V' && !(i + 1 < n && (k[i + 1].c == 'D' || is_nounish(&k[i + 1]) || k[i + 1].c == 'J' || k[i + 1].c == 'O'))) k[i].c = 'T';
            /* "would like", "will love": a verb after a modal */
            if (i && k[i - 1].c == 'M' && k[i].c != 'V' && k[i].c != 'X' && k[i].v) k[i].c = 'V';
            if (i && k[i - 1].c == 'M' && k[i].id < OPEN && !strcmp(R->t->voc[k[i].id].word, "like")) k[i].c = 'V';
        }
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
            c.type = CH_VG; uint32_t lastx = NONE, lastv = NONE; int lastvi = -1;
            while (i < n && (is_verbish(&k[i]) || k[i].c == 'G' || (k[i].c == 'R' && k[i].sp != SP_THERE && i + 1 < n && (is_verbish(&k[i + 1]) || k[i + 1].c == 'G')))) {
                if (k[i].c == 'V' && lastv != NONE && k[i].nb != k[i].id) { k[i].c = 'N'; break; }   /* "brought presents": a verb, then a noun */
                if (k[i].c == 'G') c.neg = 1;
                else if (k[i].c == 'V') { lastv = k[i].vb; lastvi = i; }
                else if (k[i].c == 'X' || k[i].c == 'M') { lastx = k[i].vb; if (k[i].sp == SP_WILL) c.fut = 1; if (k[i].vb == R->w_be) c.be = 1; }
                c.nt++; i++;
            }
            if (lastvi >= 0) { const char *w = R->t->voc[k[lastvi].id].word; size_t L = strlen(w);     /* "made", "called": not "makes", "making" */
                c.infl = k[lastvi].id != k[lastvi].vb && !(L > 3 && !strcmp(w + L - 3, "ing")) && w[L - 1] != 's'; }
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
    if (sp == SP_HE) {          /* he / she / him / her: the latest person named; it: the latest thing */
        int it = !strcmp(R->t->voc[np->head].word, "it");
        if (!it && R->refs.person != NONE) { out[0] = R->refs.person; return 1; }
        if (it && R->refs.thing != NONE) { out[0] = R->refs.thing; return 1; }
    }
    if (R->refs.single != NONE) { out[0] = R->refs.single; return 1; }
    out[0] = np->head; return 1;
}

static int is_w(Reader2 *R, uint32_t id, const char *w) { return id < OPEN && !strcmp(R->t->voc[id].word, w); }
static int naming(Reader2 *R, uint32_t v) { return is_w(R, v, "name") || is_w(R, v, "call"); }
/* RD: y names x: one phrase (the name, with the kind as its modifier), and the fact "y is a x" */
static void name_merge(NP *x, const NP *y) {
    if (nnmf < 16) nmf[nnmf++] = (NameF){ y->head, y->mod, x->head, x->mod };
    x->mod = x->head; x->head = y->head; x->pron = 0; x->ngrp = 0;
}
static int names_pass(Reader2 *R, Chunk *ch, int m) {
    for (int j = 0; j + 2 < m; j++) {                    /* "a boy named Teo wanted", "keeps an owl named Pell" */
        Chunk *x = &ch[j], *v = &ch[j + 1], *y = &ch[j + 2];
        if (!((x->type == CH_NP || (x->type == CH_PP && !x->dangling)) && x->np.head < OPEN && !x->np.pron)) continue;
        if (!(v->type == CH_VG && v->nt == 1 && !v->aux && naming(R, v->verb)) || y->type != CH_NP || y->np.pron || y->np.head >= OPEN) continue;
        int before = 0, after = 0; for (int q = 0; q < j; q++) before |= ch[q].type == CH_VG; for (int q = j + 3; q < m; q++) after |= ch[q].type == CH_VG;
        if (!before && !after) continue;                 /* "Asha named the kite Ember": the verb of the sentence */
        name_merge(&x->np, &y->np); memmove(ch + j + 1, ch + j + 3, (size_t)(m - j - 3) * sizeof *ch); m -= 2;
    }
    for (int j = 0; j + 3 < m; j++) {                    /* "the baker, Mr. Oduya, fixed": apposition */
        if (ch[j].type != CH_NP || ch[j].np.pron || ch[j].np.head >= OPEN) continue;
        int a = j + 1; if (ch[a].type == CH_PP && spw(R->t, ch[a].prep) == SP_OF) a++;
        if (!(a + 3 < m && ch[a].type == CH_COMMA && ch[a + 1].type == CH_NP && !ch[a + 1].np.pron && ch[a + 1].np.head < OPEN && ch[a + 2].type == CH_COMMA && ch[a + 3].type == CH_VG)) continue;
        name_merge(&ch[j].np, &ch[a + 1].np); memmove(ch + a, ch + a + 3, (size_t)(m - a - 3) * sizeof *ch); m -= 3;
    }
    return m;
}

/* one sentence → frames. Returns how many written to out. */
static int read_sentence(Reader2 *R, Tk *k, int n, Frame *out, int max, int *whslot) {
    static Chunk ch[512]; int m = chunk(R, k, n, ch, 512), nf = 0;
    nnmf = 0; if (RD) { m = names_pass(R, ch, m); R->nsent++; }
    NP pend[4]; int npend = 0, pend_and = 0, cur = -1, curn = 0;    /* cur: first of the frames being filled (one per agent) */
    uint8_t tc = 0, either = 0, negnext = 0, wh = 0; uint32_t wtype = NONE; int inverted = 0, wh_open_agent = 0, sub = 0;
    for (int i = 0; i < m; i++) {
        Chunk *c = &ch[i];
        switch (c->type) {
        case CH_WH: wh = c->wh; wtype = c->wtype; break;
        case CH_TIME: tc = c->tc; for (int q = cur; cur >= 0 && q < cur + curn; q++) out[q].tc = tc; break;
        case CH_NEG:
            if (RD && cur >= 0 && out[cur].verb != NONE && i && (ch[i - 1].type == CH_COMMA || ch[i - 1].type == CH_AND) && i + 1 < m && (ch[i + 1].type == CH_NP || ch[i + 1].type == CH_PP)) {
                int base = cur, bn = curn; cur = nf;            /* "eats beetles, not mice": the "not" is for what follows only */
                for (int q = 0; q < bn && nf < max; q++) { Frame *f = &out[nf++]; memset(f, 0, sizeof *f); f->verb = out[base + q].verb; f->part = out[base + q].part; f->tc = out[base + q].tc; f->flags = F_NEG;
                    int ag = find_arg(&out[base + q], A_AGENT); if (ag >= 0) frame_arg(f, A_AGENT, NONE, out[base + q].a[ag].head, out[base + q].a[ag].mod, 0); }
                curn = bn; break;
            }
            if (cur >= 0) for (int q = cur; q < cur + curn; q++) out[q].flags |= F_NEG; else negnext = 1; break;
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
                else if (RD && naming(R, f->verb)) {                /* "call it Old Watch", "named the kite Ember" */
                    int p = find_arg(f, A_PAT); if (q == cur && nnmf < 16) nmf[nnmf++] = (NameF){ who[0], c->np.mod, f->a[p].head, f->a[p].mod }; }
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
            if (RD && is_w(R, c->prep, "without") && who[0] < OPEN) {   /* "got on without a ticket": did not have a ticket */
                for (int q = cur; q < cur + curn && nf < max; q++) { int ag = find_arg(&out[q], A_AGENT); Frame *f = &out[nf++]; memset(f, 0, sizeof *f); f->verb = R->w_have; f->part = NONE; f->tc = tc; f->flags = F_NEG;
                    if (ag >= 0) frame_arg(f, A_AGENT, NONE, out[q].a[ag].head, out[q].a[ag].mod, 0); frame_arg(f, A_PAT, NONE, who[0], c->np.mod, 0); }
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
            if (RD && c->nt == 1 && !c->aux && c->infl && cur >= 0 && out[cur].verb != NONE && npend == 0 && !sub && i && (ch[i - 1].type == CH_NP || ch[i - 1].type == CH_PP)
                && (i + 1 >= m || ch[i + 1].type == CH_PP || ch[i + 1].type == CH_COMMA) && out[cur].narg && out[cur].a[out[cur].narg - 1].head < OPEN) {
                Arg x = out[cur].a[out[cur].narg - 1];          /* "an owl made of stone", "a kite shaped like a dragon" */
                Frame *f = &out[nf]; memset(f, 0, sizeof *f); f->verb = c->verb; f->part = c->part; f->tc = tc; frame_arg(f, A_PAT, NONE, x.head, x.mod, 0);
                cur = nf++; curn = 1; break;
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
                    if (c->neg) f->flags |= F_NEG; if (c->fut) f->flags |= F_FUT; if (RD && c->be && c->infl) f->flags |= F_PASS; }
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
                if (c->neg || negnext) f->flags |= F_NEG; if (c->fut) f->flags |= F_FUT; if (RD && c->be && c->infl) f->flags |= F_PASS; curn++;
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
    for (int q = 0; q < nf; q++) for (int a = 0; a < out[q].narg; a++) {     /* the latest person named, the latest thing */
        uint32_t h = out[q].a[a].head; if (h >= OPEN || special(R->t->voc[h].word) != SP_NONE) continue;
        WC *c = wc_of(R->wm, R->t, h); const Arg *x = &out[q].a[a];
        if (c->c == 'P') R->refs.person = h;
        if (RD) { if ((c->c == 'N' || r2_is_name(R, h)) && (x->role == A_PAT || x->role == A_ATTR)) R->refs.thing = h; }   /* RD: a thing is what is done or is, not a place */
        else if (c->c == 'N' && x->role != A_AGENT) R->refs.thing = h;
    }
    if (RD) {
        for (int q = 0; q < nf; q++) { Frame *f = &out[q]; if (!(f->flags & F_PASS)) continue;
            int ag = find_arg(f, A_AGENT), pa = find_arg(f, A_PAT);
            if (naming(R, f->verb) && ag >= 0 && pa >= 0) {             /* "the train is called the Grey Heron": that fact instead */
                Arg x = f->a[ag], y = f->a[pa]; f->verb = R->w_be; f->flags &= (uint8_t)~F_PASS; f->narg = 0;
                frame_arg(f, A_AGENT, NONE, y.head, y.mod, 0); frame_arg(f, A_ATTR, NONE, x.head, x.mod, K_KIND); continue; }
            if (ag >= 0) f->a[ag].role = A_PAT;                          /* passive as active: "was built by a clockmaker" */
            for (int a = 0; a < f->narg; a++) if (f->a[a].role == A_PP && is_w(R, f->a[a].prep, "by")) { f->a[a].role = A_AGENT; f->a[a].prep = NONE; break; }
        }
        for (int x = 0; x < nnmf && nf < max; x++) { if (nmf[x].h >= OPEN || nmf[x].k >= OPEN) continue;
            Frame *f = &out[nf++]; memset(f, 0, sizeof *f); f->verb = R->w_be; f->part = NONE;
            frame_arg(f, A_AGENT, NONE, nmf[x].h, nmf[x].hm, 0); frame_arg(f, A_ATTR, NONE, nmf[x].k, nmf[x].km, K_KIND); }
    }
    return nf;
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
    memset(R, 0, sizeof *R); R->t = t; R->wm = wm; R->refs.single = NONE; R->refs.person = R->refs.thing = NONE;
    R->w_be = vid(t, "be"); R->w_have = vid(t, "have"); R->w_do = vid(t, "do"); R->w_not = vid(t, "not"); R->w_you = vid(t, "you");
}

static const Meaning *MM;

static int glue(const char *w) { return meaning_is_function(MM, w); }

typedef struct { int32_t *mid; uint32_t n; } MidMap;

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
static int content(Know *k, uint32_t w) { if (w >= OPEN) return 0; const char *s = k->t.voc[w].word;
    if (RD && (!strcmp(s, "no") || !strcmp(s, "yes"))) return 1;          /* "the mayor said no" */
    return s[0] && s[1] && !glue(s) && special(s) == SP_NONE; }

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

static float role_fit(const Arg *a, const Arg *b) {
    if (a->role == b->role) return a->role == A_PP && a->prep != b->prep ? 0.8f : 1.0f;
    if ((a->role == A_PAT && b->role == A_ATTR) || (a->role == A_ATTR && b->role == A_PAT)) return 0.8f;
    if ((a->role == A_PAT && b->role == A_PP) || (a->role == A_PP && b->role == A_PAT)) return 0.6f;
    return 0.4f;
}

static int is_word_at(const char *s, const char *w) { size_t n = strlen(w); return !strncasecmp(s, w, n) && !isalpha((unsigned char)s[n]); }
/* a word's class as the table has it (lower case; a word not in it: N) */
static char wclass(const char *w, size_t n) { char b[64]; if (n >= sizeof b) n = sizeof b - 1; for (size_t i = 0; i < n; i++) b[i] = (char)tolower((unsigned char)w[i]); b[n] = 0;
    if (special(b) == SP_NOT) return 'G'; const LexE *l = lex_get(b); if (l) return l->c;
    return n > 3 && (!strcmp(b + n - 2, "ed") || !strcmp(b + n - 3, "ing")) ? 'V' : 'N'; }
/* an option that is a clause of its own ("they snored", "Josh was late to class"): a verb after its first word */
static int clause(const char *o) { const char *p = o; int first = 1;
    while (*p) { while (*p == ' ') p++; const char *e = p; while (*e && *e != ' ') e++; if (e == p) break;
        char c = wclass(p, (size_t)(e - p)); if (!first && (c == 'V' || c == 'X' || c == 'M')) return 1; if (first && c == 'V') return 0; first = 0; p = e; }
    return 0; }
/* RD: "what is X" → "X is <option>", put where the verb or adjective begins ("Pepper is afraid of mice", "the lamp is made
   of glass"), or at the end ("Mr. Bayo is Suri's grandfather"); with no subject first ("stuck in the gears") the option is it */
static void be_statement(const char *x, const char *aux, const char *opt, char *out, size_t cap) {
    int noun = 0; const char *p = x;
    while (*p) { while (*p == ' ') p++; if (!*p) break; const char *e = p; while (*e && *e != ' ') e++;
        size_t L = (size_t)(e - p); char c = isupper((unsigned char)*p) ? 'P' : wclass(p, L);      /* a capital inside a question: a name */
        if (L > 4 && (!strncmp(e - 3, "ing", 3) || !strncmp(e - 2, "ed", 2)) && c != 'P') c = 'V';                     /* "having", "stuck" */
        if (c == 'V' || c == 'J' || c == 'R' || c == 'I' || c == 'G') {
            if (!noun) { snprintf(out, cap, "%s %s %s", opt, aux, x); return; }
            snprintf(out, cap, "%.*s%s %s %s", (int)(p - x), x, aux, p, opt); return; }
        if (c == 'N' || c == 'P' || c == 'O') noun = 1;
        p = e; }
    snprintf(out, cap, "%s %s %s", x, aux, opt);
}
/* R1 (docs/COMPREHEND.md, SYN=1): a yes/no question ("Was Nadia sad?") and an option that answers it ("no, she was not
   sad") → "Nadia was not sad. she was not sad"; other options follow the question said as a statement */
static int SYN = 0;
static int yesno_statement(const char *q, const char *opt, char *out, size_t cap) {
    static const char *AUX[] = { "do", "does", "did", "is", "are", "was", "were", "can", "could", "will", "would", "has", "have", "had", 0 };
    int a = -1; for (int i = 0; AUX[i] && a < 0; i++) if (is_word_at(q, AUX[i])) a = i; if (a < 0) return 0;
    const char *r = q + strlen(AUX[a]); while (*r == ' ') r++;
    const char *p = r; int first = 1;                    /* the subject: up to the first verb, adjective, adverb or preposition */
    while (*p) { const char *e = p; while (*e && *e != ' ' && *e != '?') e++; if (e == p) break;
        char c = isupper((unsigned char)*p) ? 'P' : wclass(p, (size_t)(e - p)); if (!first && (c == 'V' || c == 'J' || c == 'R' || c == 'I' || c == 'G')) break;
        first = 0; p = e; while (*p == ' ') p++; }
    if (!*p || *p == '?') return 0;
    int no = is_word_at(opt, "no"), yes = is_word_at(opt, "yes"); const char *rest = opt;
    if (no || yes) { rest += no ? 2 : 3; while (*rest == ',' || *rest == ' ') rest++; }
    snprintf(out, cap, "%.*s %s%s %s. %s", (int)(p - r), r, AUX[a], no ? " not" : "", p, rest);
    return 1;
}
/* the statement an option makes: the option put where the question asks */
static void statement(const char *q, const char *opt, char *out, size_t cap) {
    static const char *WH[] = { "what", "which", "who", "whom", "where", "when", "why", "how", 0 };
    static const char *AUX[] = { "do", "does", "did", "is", "are", "was", "were", "can", "could", "will", "would", "has", "have", "had", "should", "must", 0 };
    if (SYN && yesno_statement(q, opt, out, cap)) goto done;
    if (RD) {   /* "On which day does Nadia ride to school?" → "Nadia does ride to school on <option>" */
        const char *e = q; while (*e && *e != ' ') e++; const char *w2 = e; while (*w2 == ' ') w2++;
        if (e > q && wclass(q, (size_t)(e - q)) == 'I' && (is_word_at(w2, "which") || is_word_at(w2, "what"))) {
            const char *r = w2; while (*r && *r != ' ') r++; while (*r == ' ') r++;
            if (!(is_word_at(r, "do") || is_word_at(r, "does") || is_word_at(r, "did") || is_word_at(r, "is") || is_word_at(r, "was") || is_word_at(r, "are") || is_word_at(r, "were"))) { while (*r && *r != ' ') r++; while (*r == ' ') r++; }
            char tmp[1024]; snprintf(tmp, sizeof tmp, "%s", r); for (char *x = tmp + strlen(tmp); x > tmp && (x[-1] == '?' || x[-1] == ' '); ) *--x = 0;
            char decl[1024]; if (yesno_statement(tmp, "", decl, sizeof decl)) { char *d = decl + strlen(decl); while (d > decl && (d[-1] == ' ' || d[-1] == '.')) *--d = 0;
                snprintf(out, cap, "%s %.*s %s", decl, (int)(e - q), q, opt); goto done; } } }
    const char *u = strchr(q, '_');
    if (u) { snprintf(out, cap, "%.*s%s%s", (int)(u - q), q, opt, u + 1); for (char *p = out; *p; p++) if (*p == '_') *p = ' '; goto done; }
    if (strcasestr(q, "of the following") || strcasestr(q, "which is true") || strcasestr(q, "which is not true")) { snprintf(out, cap, "%s", opt); goto done; }
    for (int w = 0; WH[w]; w++) {
        if (is_word_at(q, WH[w])) {                                   /* asked at the start */
            const char *r = q + strlen(WH[w]); while (*r == ' ') r++;
            if (!strncasecmp(WH[w], "how", 3) && (is_word_at(r, "many") || is_word_at(r, "much"))) { r += 4; while (*r && *r != ' ') r++; while (*r == ' ') r++; }
            for (int a = 0; AUX[a]; a++) if (is_word_at(r, AUX[a])) {     /* "what does X do", "where did Tom go": X do <option> */
                const char *x = r + strlen(AUX[a]); while (*x == ' ') x++;
                if (RD) { char xx[1024]; snprintf(xx, sizeof xx, "%s", x); for (char *e = xx + strlen(xx); e > xx && (e[-1] == '?' || e[-1] == ' '); ) *--e = 0;
                    size_t L = strlen(xx);
                    if (a >= 3 && a <= 6 && !strncasecmp(xx, "the name of ", 12)) snprintf(out, cap, "%s %s named %s", xx + 12, AUX[a], opt);   /* "what was the name of the cat" */
                    else if (a >= 3 && a <= 6 && L > 7 && !strcasecmp(xx + L - 7, "'s name")) snprintf(out, cap, "%.*s %s named %s", (int)(L - 7), xx, AUX[a], opt);   /* "the cat's name" */
                    else if (a >= 3 && a <= 6 && !strcasestr(xx, "not") && !strstr(xx, "n't")) be_statement(xx, AUX[a], opt, out, cap);
                    else snprintf(out, cap, clause(opt) ? "%s. %s" : "%s %s", xx, opt);   /* "Pell eat mice", not "Pell eat? mice" (two sentences) */
                    goto done; }
                snprintf(out, cap, "%s %s", x, opt); goto done; }
            snprintf(out, cap, "%s %s", opt, r); goto done;              /* "who has an eraser": <option> has an eraser */
        }
        for (const char *p = q; *p; p++)                                /* asked in the middle: "are called what" */
            if ((p == q || p[-1] == ' ') && is_word_at(p, WH[w])) { snprintf(out, cap, "%.*s%s%s", (int)(p - q), q, opt, p + strlen(WH[w])); goto done; }
    }
    snprintf(out, cap, "%s. %s", q, opt);
done:
    for (char *p = out; *p; p++) if (*p == '?') *p = '.';
}

