/*
 * MIMIR — Text intake and context memory
 *
 * Takes in any text, a typed instruction or a whole book, the way a
 * language model takes in a file: every part of it is turned into tokens
 * and kept, in order, in a context memory. Mimir does not understand it
 * yet. It can say every word, knows which words are the same word, and
 * can give back any sentence or find any word.
 *
 * ════════════════════════════════════════════════════════════════════
 * THE PIPELINE
 * ════════════════════════════════════════════════════════════════════
 *
 *   raw bytes (UTF-8)
 *     → fold        curly quotes, dashes, accents → plain ASCII
 *     → pieces      words, numbers, punctuation, symbols
 *     → spoken      "12.5%"  → twelve point five percent
 *                   "3rd"    → third        "1999" → nineteen ninety nine
 *                   "5:30"   → five thirty  "$1,250" → one thousand ...
 *                   "Dr."    → doctor       "FBI"  → F B I (letter names)
 *                   "&" → and   "@" → at   "+" → plus
 *     → vocabulary  each spoken word gets an ID (like a token ID); a new
 *                   word is read once by the reader, checked against the
 *                   dictionary and fixed if wrong (learning on the fly)
 *     → context     token IDs in order, sentence ends marked, each token
 *                   pointing back to the text it came from
 *
 * A number or an abbreviation becomes several spoken words: each of them
 * is a token, all pointing at the same piece of the original text.
 * Punctuation is kept as tokens too (it is part of what was written),
 * but has no sound.
 * ════════════════════════════════════════════════════════════════════
 */

#include "mimir.h"
#include <ctype.h>
#include <strings.h>

/* ── vocabulary: spoken word → ID ── */

static uint32_t text_hash(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 16777619u; }
    return h;
}

static int vocab_grow(TextContext *t) {
    uint32_t cap = t->voc_cap ? t->voc_cap * 2 : 4096;
    TextWord *v = realloc(t->voc, cap * sizeof *v);
    if (!v) return -1;
    t->voc = v;
    t->voc_cap = cap;
    /* rebuild the slot table at twice the capacity (never more than half full) */
    free(t->slot);
    t->mask = cap * 2 - 1;
    t->slot = calloc(t->mask + 1, sizeof *t->slot);
    if (!t->slot) return -1;
    for (uint32_t i = 0; i < t->nvoc; i++) {
        uint32_t h = text_hash(t->voc[i].word) & t->mask;
        while (t->slot[h]) h = (h + 1) & t->mask;
        t->slot[h] = i + 1;
    }
    return 0;
}

/* Letter names, for things spelled out (FBI, U.S., an "x" on its own). */
static const char *LETTER_SOUNDS[26] = {
    "EY1", "B IY1", "S IY1", "D IY1", "IY1", "EH1 F", "JH IY1", "EY1 CH", "AY1",
    "JH EY1", "K EY1", "EH1 L", "EH1 M", "EH1 N", "OW1", "P IY1", "K Y UW1",
    "AA1 R", "EH1 S", "T IY1", "Y UW1", "V IY1", "D AH1 B AH0 L Y UW0", "EH1 K S",
    "W AY1", "Z IY1",
};

/* The ID of a spoken word, adding it (and reading it once) if new.
 * kind TEXT_LETTER: the name of a letter ("b" said "bee"). */
static uint32_t vocab_id(TextContext *t, Reader *r, const char *word, int kind) {
    char key[READER_MAX_WORD + 2];
    snprintf(key, sizeof key, "%s%s", kind == TEXT_LETTER ? "^" : "", word);
    if (t->nvoc * 2 >= t->voc_cap && vocab_grow(t)) return 0;
    uint32_t h = text_hash(key) & t->mask;
    for (; t->slot[h]; h = (h + 1) & t->mask) {
        TextWord *e = &t->voc[t->slot[h] - 1];
        if (strcmp(e->word, key) == 0) { e->count++; return t->slot[h] - 1; }
    }
    uint32_t id = t->nvoc++;
    t->slot[h] = id + 1;
    TextWord *e = &t->voc[id];
    memset(e, 0, sizeof *e);
    snprintf(e->word, sizeof e->word, "%s", key);
    e->count = 1;
    e->kind = (unsigned char)kind;
    if (kind == TEXT_SUFFIX) {                   /* the 's of FBI's: said "z" */
        snprintf(e->sounds, sizeof e->sounds, "Z");
        e->check = READ_RIGHT;
    } else if (kind == TEXT_LETTER) {
        int c = tolower((unsigned char)word[0]);
        snprintf(e->sounds, sizeof e->sounds, "%s", (c >= 'a' && c <= 'z') ? LETTER_SOUNDS[c - 'a'] : "");
        e->check = READ_RIGHT;
    } else if (kind == TEXT_WORD && r && r->ready) {
        /* A new word: read it once, check it, fix it if wrong. */
        e->check = (unsigned char)(r->self_learn ? reader_check(r, word, 0) : READ_UNCHECKED);
        reader_sounds(r, word, e->sounds, sizeof e->sounds);
        if (e->check == READ_FIXED) t->fixed++;
        if (e->check == READ_UNCHECKED) t->unchecked++;
    }
    return id;
}

/* ── context: tokens in order ── */

static void push(TextContext *t, Reader *r, const char *word, int kind, size_t src, size_t len) {
    if (t->ntok == t->tok_cap) {
        uint32_t cap = t->tok_cap ? t->tok_cap * 2 : 65536;
        TextToken *g = realloc(t->tok, cap * sizeof *g);
        if (!g) return;
        t->tok = g;
        t->tok_cap = cap;
    }
    TextToken *k = &t->tok[t->ntok++];
    k->word = vocab_id(t, r, word, kind);
    k->src = (uint32_t)src;
    k->len = (uint16_t)(len > 65535 ? 65535 : len);
    k->kind = (unsigned char)kind;
    k->end = 0;
    t->spoken += kind != TEXT_PUNCT;
}

static void end_sentence(TextContext *t) {
    if (t->ntok && !t->tok[t->ntok - 1].end) {
        t->tok[t->ntok - 1].end = 1;
        t->sentences++;
    }
}

/* ── numbers to words ── */

static const char *ONES[] = { "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen" };
static const char *TENS[] = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety" };
static const char *SCALES[] = { "", "thousand", "million", "billion", "trillion" };

typedef struct { char w[20][16]; int n; } Words;

static void add(Words *o, const char *w) {
    if (o->n < 20) snprintf(o->w[o->n++], 16, "%s", w);
}

static void under_thousand(Words *o, int v) {
    if (v >= 100) { add(o, ONES[v / 100]); add(o, "hundred"); v %= 100; if (!v) return; }
    if (v >= 20) { add(o, TENS[v / 10]); if (v % 10) add(o, ONES[v % 10]); }
    else add(o, ONES[v]);
}

/* 1250000 → one million two hundred fifty thousand */
static void cardinal(Words *o, unsigned long long v) {
    if (v == 0) { add(o, "zero"); return; }
    int part[5], n = 0;
    while (v && n < 5) { part[n++] = (int)(v % 1000); v /= 1000; }
    for (int i = n - 1; i >= 0; i--)
        if (part[i]) { under_thousand(o, part[i]); if (i) add(o, SCALES[i]); }
}

/* 1999 → nineteen ninety nine; 1900 → nineteen hundred; 2005 → two thousand five */
static void year(Words *o, int v) {
    if (v >= 2000 && v < 2010) { cardinal(o, (unsigned long long)v); return; }
    under_thousand(o, v / 100);
    if (v % 100 == 0) { add(o, "hundred"); return; }
    if (v % 100 < 10) add(o, "oh");
    under_thousand(o, v % 100);
}

/* Turn the last number word into its ordinal: three → third, twenty → twentieth. */
static void make_ordinal(Words *o) {
    static const char *from[] = { "one", "two", "three", "five", "eight", "nine", "twelve" };
    static const char *to[] = { "first", "second", "third", "fifth", "eighth", "ninth", "twelfth" };
    if (!o->n) return;
    char *w = o->w[o->n - 1];
    for (int i = 0; i < 7; i++)
        if (strcmp(w, from[i]) == 0) { snprintf(w, 16, "%s", to[i]); return; }
    size_t l = strlen(w);
    if (l && w[l - 1] == 'y') { w[l - 1] = 0; strncat(w, "ieth", 15 - strlen(w)); }
    else strncat(w, "th", 15 - l);
}

/* ── folding UTF-8 into plain ASCII ──
 * Writes the folded text and, for each output byte, where it came from.
 * With out == NULL it only counts, so the buffers can be sized exactly. */
static size_t fold_utf8(const char *in, size_t n, char *out, uint32_t *from, size_t base) {
    static const struct { const char *u, *a; } MAP[] = {
        {"\xe2\x80\x98", "'"}, {"\xe2\x80\x99", "'"}, {"\xe2\x80\x9c", "\""}, {"\xe2\x80\x9d", "\""},
        {"\xe2\x80\x93", " - "}, {"\xe2\x80\x94", " - "}, {"\xe2\x80\xa6", "..."}, {"\xc2\xa0", " "},
        {"\xc3\xa9", "e"}, {"\xc3\xa8", "e"}, {"\xc3\xaa", "e"}, {"\xc3\xab", "e"}, {"\xc3\xa1", "a"},
        {"\xc3\xa0", "a"}, {"\xc3\xa2", "a"}, {"\xc3\xa4", "a"}, {"\xc3\xa7", "c"}, {"\xc3\xb1", "n"},
        {"\xc3\xb3", "o"}, {"\xc3\xb4", "o"}, {"\xc3\xb6", "o"}, {"\xc3\xba", "u"}, {"\xc3\xbb", "u"},
        {"\xc3\xbc", "u"}, {"\xc3\xad", "i"}, {"\xc3\xae", "i"}, {"\xc3\xaf", "i"}, {"\xc3\x9f", "ss"},
        {"\xc3\x89", "E"}, {"\xc3\x80", "A"}, {"\xc2\xb0", " degrees"}, {"\xc2\xa3", "#GBP"},
        {"\xe2\x82\xac", "#EUR"}, {"\xc3\x97", " times "},
    };
    size_t o = 0;
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)in[i];
        if (c < 0x80) { if (out) { out[o] = (char)c; from[o] = (uint32_t)(base + i); } o++; i++; continue; }
        size_t len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        const char *a = " ";                         /* anything else: a gap */
        for (size_t k = 0; k < sizeof MAP / sizeof *MAP; k++)
            if (strlen(MAP[k].u) == len && i + len <= n && memcmp(in + i, MAP[k].u, len) == 0) { a = MAP[k].a; break; }
        for (; *a; a++) { if (out) { out[o] = *a; from[o] = (uint32_t)(base + i); } o++; }
        i += len;
    }
    return o;
}

/* ── abbreviations said in full ── */

static const char *abbreviation(const char *w) {
    static const struct { const char *a, *full; } T[] = {
        {"dr", "doctor"}, {"mr", "mister"}, {"mrs", "missus"}, {"ms", "miz"}, {"st", "saint"},
        {"jr", "junior"}, {"sr", "senior"}, {"vs", "versus"}, {"etc", "et cetera"}, {"no", "number"},
        {"prof", "professor"}, {"capt", "captain"}, {"gen", "general"}, {"col", "colonel"},
        {"lt", "lieutenant"}, {"mt", "mount"}, {"ft", "feet"}, {"approx", "approximately"},
        {"dept", "department"}, {"jan", "january"}, {"feb", "february"}, {"mar", "march"},
        {"apr", "april"}, {"aug", "august"}, {"sept", "september"}, {"oct", "october"},
        {"nov", "november"}, {"dec", "december"},
    };
    for (size_t i = 0; i < sizeof T / sizeof *T; i++)
        if (strcmp(w, T[i].a) == 0) return T[i].full;
    return NULL;
}

/* Is a word in capitals one to spell out (FBI) rather than say (NASA, STOP)? */
static int spell_out(const Reader *r, const char *caps, int len) {
    if (len < 2 || len > 5) return 0;
    char low[8];
    for (int i = 0; i < len; i++) low[i] = (char)tolower((unsigned char)caps[i]);
    low[len] = 0;
    int vowels = 0;
    for (int i = 0; i < len; i++) vowels += strchr("aeiouy", low[i]) != NULL;
    if (r && r->lex && reader_known_word(r, low)) return 0;   /* the dictionary says it */
    return vowels == 0 || len <= 3;
}

static void say_words(TextContext *t, Reader *r, const Words *w, size_t src, size_t len) {
    for (int i = 0; i < w->n; i++) push(t, r, w->w[i], TEXT_WORD, src, len);
}

static void say_phrase(TextContext *t, Reader *r, const char *phrase, size_t src, size_t len) {
    char tmp[64], *save = NULL;
    snprintf(tmp, sizeof tmp, "%s", phrase);
    for (char *w = strtok_r(tmp, " ", &save); w; w = strtok_r(NULL, " ", &save))
        push(t, r, w, TEXT_WORD, src, len);
}

/* ── taking in text ── */

int text_take(TextContext *t, Reader *r, const char *in, size_t n) {
    MIMIR_PROBE("text.bytes", n);
    /* keep the original text: tokens point into it */
    if (t->src_len + n + 1 > t->src_cap) {
        size_t cap = (t->src_len + n + 1) * 2;
        char *g = realloc(t->src, cap);
        if (!g) return -1;
        t->src = g;
        t->src_cap = cap;
    }
    size_t base = t->src_len;
    memcpy(t->src + base, in, n);
    t->src_len += n;
    t->src[t->src_len] = 0;

    size_t m = fold_utf8(in, n, NULL, NULL, base);     /* folded ASCII can be longer */
    char *a = malloc(m + 1);
    uint32_t *from = malloc((m + 1) * sizeof *from);
    if (!a || !from) { free(a); free(from); return -1; }
    fold_utf8(in, n, a, from, base);
    a[m] = 0;
    from[m] = (uint32_t)(base + n);
    uint32_t before = t->ntok;
    int newlines = 0;

    for (size_t i = 0; i < m;) {
        unsigned char c = (unsigned char)a[i];
        size_t s0 = from[i];

        if (c == '\n') {                               /* a blank line ends a paragraph */
            if (++newlines >= 2) end_sentence(t);           /* (a heading has no full stop) */
            i++;
            continue;
        }
        if (isspace(c)) { i++; continue; }
        newlines = 0;

        if (isalpha(c)) {                              /* a word: letters and inner apostrophes */
            size_t j = i;
            while (j < m && (isalpha((unsigned char)a[j]) ||
                   (a[j] == '\'' && j + 1 < m && isalpha((unsigned char)a[j + 1])))) j++;
            int len = (int)(j - i);
            char w[READER_MAX_WORD + 1], low[READER_MAX_WORD + 1];
            int keep = len < READER_MAX_WORD ? len : READER_MAX_WORD - 1;
            memcpy(w, a + i, (size_t)keep);
            w[keep] = 0;
            int caps = 1;
            for (int k = 0; k < keep; k++) {
                low[k] = (char)tolower((unsigned char)w[k]);
                if (isalpha((unsigned char)w[k]) && !isupper((unsigned char)w[k])) caps = 0;
            }
            low[keep] = 0;
            size_t span = from[j] - s0;
            /* initials and dotted letters: "U.S.", "J. K." */
            if (len == 1 && j < m && a[j] == '.' && isupper(c)) {
                push(t, r, low, TEXT_LETTER, s0, span + 1);
                i = j + 1;
                continue;
            }
            /* abbreviation with a full stop: "Dr." */
            const char *full = (j < m && a[j] == '.') ? abbreviation(low) : NULL;
            if (full) {
                say_phrase(t, r, full, s0, span + 1);
                i = j + 1;
                continue;
            }
            /* capitals to spell out: "FBI", "FBI's" */
            char *apos = strchr(w, '\'');
            int stem = apos ? (int)(apos - w) : keep;
            if (caps && spell_out(r, w, stem)) {
                for (int k = 0; k < stem; k++) {
                    char one[2] = { low[k], 0 };
                    push(t, r, one, TEXT_LETTER, s0, span);
                }
                if (apos && apos[1] == 's') push(t, r, "'s", TEXT_SUFFIX, s0, span);
                i = j;
                continue;
            }
            /* no vowels and not a word it knows ("txt", "mp"): spell it */
            int vowels = 0;
            for (int k = 0; k < keep; k++) vowels += strchr("aeiouy", low[k]) != NULL;
            if (!apos && len >= 2 && len <= 5 && !vowels && !(r && reader_known_word(r, low))) {
                for (int k = 0; k < keep; k++) {
                    char one[2] = { low[k], 0 };
                    push(t, r, one, TEXT_LETTER, s0, span);
                }
                i = j;
                continue;
            }
            /* a lone letter other than a / I is its name: "x" in "3x" */
            if (len == 1 && low[0] != 'a' && low[0] != 'i')
                push(t, r, low, TEXT_LETTER, s0, span);
            else
                push(t, r, low, TEXT_WORD, s0, span);
            i = j;
            continue;
        }

        if (isdigit(c) || (c == '$' && i + 1 < m && isdigit((unsigned char)a[i + 1])) ||
            (c == '#' && i + 4 <= m && (strncmp(a + i, "#GBP", 4) == 0 || strncmp(a + i, "#EUR", 4) == 0) &&
             i + 4 < m && isdigit((unsigned char)a[i + 4]))) {
            /* a number: money, commas, decimals, times, ordinals, percent, years */
            const char *money = NULL;
            size_t j = i;
            if (a[j] == '$') { money = "dollars"; j++; }
            else if (a[j] == '#') { money = a[j + 1] == 'G' ? "pounds" : "euros"; j += 4; }
            size_t ds = j;                             /* where the digits start */
            unsigned long long v = 0;
            int digits = 0, commas = 0;
            while (j < m && (isdigit((unsigned char)a[j]) ||
                   (a[j] == ',' && j + 3 < m && isdigit((unsigned char)a[j + 1]) && isdigit((unsigned char)a[j + 2]) && isdigit((unsigned char)a[j + 3])))) {
                if (a[j] == ',') commas++;
                else if (digits++ < 18) v = v * 10 + (unsigned long long)(a[j] - '0');
                j++;
            }
            Words w = { .n = 0 };
            int leading_zero = a[ds] == '0' && digits > 1;
            if (j + 2 < m && a[j] == ':' && isdigit((unsigned char)a[j + 1]) && isdigit((unsigned char)a[j + 2])) {
                /* a time: 5:30 → five thirty, 10:00 → ten o'clock, 7:05 → seven oh five */
                int mins = (a[j + 1] - '0') * 10 + (a[j + 2] - '0');
                cardinal(&w, v);
                if (mins == 0) add(&w, "o'clock");
                else { if (mins < 10) add(&w, "oh"); under_thousand(&w, mins); }
                j += 3;
            } else if (leading_zero) {
                /* digits said one by one: 010-0199, 007 */
                for (size_t k = ds; k < j; k++) if (isdigit((unsigned char)a[k])) add(&w, ONES[a[k] - '0']);
            } else if (!money && !commas && digits == 4 && v >= 1100 && v <= 2099 &&
                       !(j < m && (a[j] == '.' && j + 1 < m && isdigit((unsigned char)a[j + 1])))) {
                year(&w, (int)v);
            } else {
                cardinal(&w, v);
            }
            if (j + 1 < m && a[j] == '.' && isdigit((unsigned char)a[j + 1])) {   /* 12.5 */
                add(&w, "point");
                j++;
                while (j < m && isdigit((unsigned char)a[j])) { add(&w, ONES[a[j] - '0']); j++; }
            }
            /* 3rd, 21st, 4th */
            if (j + 1 < m && (strncasecmp(a + j, "st", 2) == 0 || strncasecmp(a + j, "nd", 2) == 0 ||
                              strncasecmp(a + j, "rd", 2) == 0 || strncasecmp(a + j, "th", 2) == 0) &&
                !(j + 2 < m && isalpha((unsigned char)a[j + 2]))) {
                make_ordinal(&w);
                j += 2;
            }
            if (money) {
                add(&w, v == 1 && money[0] == 'd' ? "dollar" : money);
            }
            if (j < m && a[j] == '%') { add(&w, "percent"); j++; }
            say_words(t, r, &w, s0, from[j] - s0);
            i = j;
            continue;
        }

        /* symbols that are said */
        const char *said = NULL;
        switch (c) {
        case '&': said = "and"; break;
        case '@': said = "at"; break;
        case '+': said = "plus"; break;
        case '=': said = "equals"; break;
        case '%': said = "percent"; break;
        case '/': said = (i > 0 && isalnum((unsigned char)a[i - 1]) && i + 1 < m && isalnum((unsigned char)a[i + 1])) ? "slash" : NULL; break;
        case '_': said = "underscore"; break;
        case '.':
            /* a dot inside a name (report.txt, example.com) is said */
            if (i > 0 && isalnum((unsigned char)a[i - 1]) && i + 1 < m && isalnum((unsigned char)a[i + 1])) said = "dot";
            break;
        }
        if (said) {
            say_phrase(t, r, said, s0, from[i + 1] - s0);
            i++;
            continue;
        }

        /* punctuation: kept, not said; . ! ? end a sentence */
        size_t j = i + 1;
        if (c == '.') while (j < m && a[j] == '.') j++;        /* ... */
        char p[8];
        size_t pl = j - i < sizeof p - 1 ? j - i : sizeof p - 1;
        memcpy(p, a + i, pl);
        p[pl] = 0;
        push(t, r, p, TEXT_PUNCT, s0, from[j] - s0);
        if (c == '!' || c == '?' || (c == '.' && j - i == 1)) end_sentence(t);
        i = j;
    }
    end_sentence(t);
    free(a);
    free(from);
    MIMIR_PROBE("text.tokens", t->ntok);
    return (int)(t->ntok - before);
}

int text_take_file(TextContext *t, Reader *r, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0) { fclose(f); return -1; }
    char *buf = malloc((size_t)size + 1);
    if (!buf) { fclose(f); return -1; }
    size_t got = fread(buf, 1, (size_t)size, f);
    fclose(f);
    int n = text_take(t, r, buf, got);
    free(buf);
    return n;
}

void text_init(TextContext *t) { memset(t, 0, sizeof *t); }

void text_free(TextContext *t) {
    free(t->src);
    free(t->tok);
    free(t->voc);
    free(t->slot);
    memset(t, 0, sizeof *t);
}

/* ── giving it back ── */

static uint32_t sentence_start(const TextContext *t, uint32_t s, uint32_t *end) {
    uint32_t k = 0, seen = 0;
    while (k < t->ntok && seen < s) { if (t->tok[k].end) seen++; k++; }
    uint32_t e = k;
    while (e < t->ntok && !t->tok[e].end) e++;
    *end = e < t->ntok ? e + 1 : e;
    return k;
}

void text_summary(const TextContext *t) {
    uint32_t words = 0, punct = 0;
    for (uint32_t i = 0; i < t->ntok; i++) t->tok[i].kind == TEXT_PUNCT ? punct++ : words++;
    printf("  In my context: %u tokens (%u spoken words, %u punctuation) in %u sentences,\n",
           t->ntok, words, punct, t->sentences);
    printf("  from %zu bytes of text. %u different tokens.\n", t->src_len, t->nvoc);
    printf("  New words I fixed while reading: %ld. Not in my dictionary (read, not checked): %ld.\n",
           t->fixed, t->unchecked);
}

/* Sentence s (from 1): the text as written, then the words as Mimir says them. */
void text_show(const TextContext *t, uint32_t s) {
    if (s < 1 || s > t->sentences) { printf("  There is no sentence %u (I have %u).\n", s, t->sentences); return; }
    uint32_t e, k = sentence_start(t, s - 1, &e);
    if (k >= e) return;
    size_t a = t->tok[k].src, b = t->tok[e - 1].src + t->tok[e - 1].len;
    printf("  Sentence %u of %u, as written:\n    %.*s\n", s, t->sentences, (int)(b - a), t->src + a);
    printf("  As I say it:\n    ");
    int col = 4;
    for (uint32_t i = k; i < e; i++) {
        const TextToken *tk = &t->tok[i];
        if (tk->kind == TEXT_PUNCT) continue;
        const TextWord *v = &t->voc[tk->word];
        char spoken[128];
        reader_respell(v->sounds, spoken, sizeof spoken);
        int l = (int)strlen(spoken) + 1;
        if (col + l > 76) { printf("\n    "); col = 4; }
        printf("%s ", spoken);
        col += l;
    }
    printf("\n  Tokens:\n    ");
    col = 4;
    for (uint32_t i = k; i < e; i++) {
        const TextWord *v = &t->voc[t->tok[i].word];
        char item[64];
        snprintf(item, sizeof item, "%s#%u", v->word[0] == '^' ? v->word + 1 : v->word, t->tok[i].word);
        int l = (int)strlen(item) + 1;
        if (col + l > 76) { printf("\n    "); col = 4; }
        printf("%s ", item);
        col += l;
    }
    printf("\n");
}

/* Every place a word was read, with the sentence it is in. */
void text_find(const TextContext *t, const char *word) {
    char key[READER_MAX_WORD], letters[READER_MAX_WORD];
    reader_word_key(word, key, letters);
    uint32_t hits = 0, s = 1, shown = 0, last = 0;
    for (uint32_t i = 0; i < t->ntok; i++) {
        const TextWord *v = &t->voc[t->tok[i].word];
        if (strcmp(v->word, key) == 0) {
            hits++;
            if (shown < 5 && s != last) {               /* each sentence once */
                last = s;
                uint32_t e, k = sentence_start(t, s - 1, &e);
                size_t a = t->tok[k].src, b = t->tok[e - 1].src + t->tok[e - 1].len;
                int len = (int)(b - a) > 160 ? 160 : (int)(b - a);
                printf("  sentence %u: %.*s%s\n", s, len, t->src + a, (int)(b - a) > 160 ? "..." : "");
                shown++;
            }
        }
        if (t->tok[i].end) s++;
    }
    printf("  \"%s\": %u time%s in my context.\n", key, hits, hits == 1 ? "" : "s");
}
