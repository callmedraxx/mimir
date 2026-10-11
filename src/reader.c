/*
 * MIMIR — Reader
 *
 * Sounds out English words the way a 12-year-old can: one letter at a
 * time, looking at the letters around it (NETtalk, 1987), and saying
 * which sound that letter makes.
 *
 *   "phone"   p→F  h→-  o→OW1  n→N  e→-       → F OW1 N   "FOHn"
 *
 * ════════════════════════════════════════════════════════════════════
 * HOW IT READS — HDC BINDING + ONE TRAINED LAYER, NO BACKPROP
 * ════════════════════════════════════════════════════════════════════
 *
 * For each letter, a window of 4 letters each side is cut out ('#' past
 * the ends of the word). Every group of letters in the window that
 * touches the centre letter is BOUND with its offset ("-1:ph", "0:ph",
 * "0:o") and hashed to one of READER_BUCKETS rows of a table. A letter
 * lights up READER_FEATS rows; the sound it makes is the column with the
 * largest sum over those rows.
 *
 * Learning is the delta rule on those rows only (softmax readout): the
 * rows that were lit move toward the right sound and away from the
 * guess. Nothing else changes, so a learning step is local and cheap,
 * and reading is a table lookup: READER_FEATS × classes int8 adds.
 *
 * The table is learned in float, then stored as int8 with one scale:
 * reading is integer adds only, which suits the phone's SDOT units.
 *
 * Data: data/reader/aligned.tsv (CMUdict, the 25,000 commonest words,
 * each letter aligned to its sound by sandbox/reader_data.py). One word
 * in ten is held out as "test" and is never learned from.
 * ════════════════════════════════════════════════════════════════════
 */

#include "mimir.h"
#include <ctype.h>
#include <stdint.h>
#include <time.h>

#define WIN   READER_WINDOW
#define SPAN  (2 * WIN + 1)

/* ── binding: the letter groups around the centre, hashed to rows ── */

static uint32_t fnv(uint32_t h, const char *s, int n) {
    for (int i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}

/* Fill feats[] with the rows lit by letter i of word w. Returns the count. */
int reader_features(const char *w, int len, int i, uint32_t *feats) {
    char s[SPAN];
    for (int k = 0; k < SPAN; k++) {
        int j = i - WIN + k;
        s[k] = (j >= 0 && j < len) ? w[j] : '#';
    }
    int n = 0;
    const int c = WIN;
    for (int a = 0; a <= c; a++)                 /* groups touching the centre */
        for (int b = c + 1; b <= SPAN; b++)
            if (b - a <= 6 || (a == c && b - a <= 4)) {
                char tag[2] = { 'g', (char)('0' + a) };
                feats[n++] = fnv(fnv(2166136261u, tag, 2), s + a, b - a) % READER_BUCKETS;
            }
    for (int a = 0; a < SPAN; a++) {             /* single letters, by position */
        char tag[2] = { 'u', (char)('0' + a) };
        feats[n++] = fnv(fnv(2166136261u, tag, 2), s + a, 1) % READER_BUCKETS;
    }
    return n;
}

/* ── the sound list ── */

static int class_of(Reader *r, const char *lab, int add) {
    for (int k = 0; k < r->classes; k++)
        if (strcmp(r->label[k], lab) == 0) return k;
    if (!add || r->classes >= READER_MAX_CLASSES) return -1;
    snprintf(r->label[r->classes], READER_LABEL_LEN, "%s", lab);
    return r->classes++;
}

void reader_init(Reader *r) {
    memset(r, 0, sizeof *r);
}

/* Frees the table only; the dictionary stays (see reader_close). */
void reader_free(Reader *r) {
    free(r->q);
    free(r->w);
    r->q = NULL;
    r->w = NULL;
    r->ready = 0;
}

/* ── reading one letter: sum the lit rows, pick the biggest column ── */

static int predict_letter(const Reader *r, const uint32_t *feats, int n) {
    /* int16 is enough: at most READER_FEATS × 127 = 4,064 */
    int16_t z[READER_MAX_CLASSES] = {0};
    const int K = r->classes;
    for (int f = 0; f < n; f++) {
        const int8_t *row = r->q + (size_t)feats[f] * K;
        for (int k = 0; k < K; k++) z[k] += row[k];
    }
    int best = 0;
    for (int k = 1; k < K; k++) if (z[k] > z[best]) best = k;
    return best;
}

/* ── sight words: mistakes it has fixed, remembered whole ──
 * Open addressing on the word; NULL when the word is not one. */
static uint32_t word_hash(const char *w) {
    uint32_t h = 2166136261u;
    for (; *w; w++) { h ^= (unsigned char)*w; h *= 16777619u; }
    return h;
}

static const ReaderMemory *sight_find(const Reader *r, const char *w) {
    if (!r->sight) return NULL;
    for (uint32_t h = word_hash(w) & r->sight_mask; r->sight[h].len; h = (h + 1) & r->sight_mask)
        if (strcmp(r->sight[h].word, w) == 0) return &r->sight[h];
    return NULL;
}

static int sight_put(Reader *r, const char *w, int len, const unsigned char *lab) {
    if (!r->sight) {
        r->sight_mask = READER_SIGHT_MAX * 2 - 1;        /* never more than half full */
        r->sight = calloc(r->sight_mask + 1, sizeof *r->sight);
        if (!r->sight) return -1;
    }
    uint32_t h = word_hash(w) & r->sight_mask;
    while (r->sight[h].len && strcmp(r->sight[h].word, w) != 0) h = (h + 1) & r->sight_mask;
    if (!r->sight[h].len) {
        if (r->nsight >= READER_SIGHT_MAX) return -1;
        r->nsight++;
    }
    snprintf(r->sight[h].word, READER_MAX_WORD, "%s", w);
    memcpy(r->sight[h].lab, lab, (size_t)len);
    r->sight[h].len = len;
    return 0;
}

/* Sound out a word: out[i] = the class index of letter i. Returns letters read. */
int reader_read_word(const Reader *r, const char *word, int *out) {
    char w[READER_MAX_WORD];
    int len = 0;
    for (const char *p = word; *p && len < READER_MAX_WORD - 1; p++)
        if (isalpha((unsigned char)*p)) w[len++] = (char)tolower((unsigned char)*p);
    w[len] = 0;
    const ReaderMemory *m = sight_find(r, w);
    if (m) {                                     /* a sight word: known by heart */
        for (int i = 0; i < len; i++) out[i] = m->lab[i];
        return len;
    }
    uint32_t feats[READER_FEATS];
    for (int i = 0; i < len; i++) {
        int n = reader_features(w, len, i, feats);
        out[i] = predict_letter(r, feats, n);
    }
    return len;
}

/* ── the data file ── */

typedef struct {
    char word[READER_MAX_WORD];
    unsigned char lab[READER_MAX_WORD];
    int len;
    int test;
} ReaderItem;

/* Load aligned.tsv. Classes are added in order of first sight. */
static ReaderItem *load_items(Reader *r, const char *path, int *count) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    int cap = 32768, n = 0;
    ReaderItem *it = malloc((size_t)cap * sizeof *it);
    char line[512];
    while (it && fgets(line, sizeof line, f)) {
        char *word = strtok(line, "\t");
        char *split = strtok(NULL, "\t");
        strtok(NULL, "\t");                      /* the whole pronunciation */
        char *labs = strtok(NULL, "\t\n");
        if (!word || !split || !labs) continue;
        int len = (int)strlen(word);
        if (len >= READER_MAX_WORD) continue;
        if (n == cap) {
            cap *= 2;
            ReaderItem *g = realloc(it, (size_t)cap * sizeof *it);
            if (!g) break;
            it = g;
        }
        ReaderItem *x = &it[n];
        memcpy(x->word, word, (size_t)len + 1);
        x->len = len;
        x->test = strcmp(split, "test") == 0;
        int k = 0, ok = 1;
        for (char *t = strtok(labs, " "); t; t = strtok(NULL, " ")) {
            int c = class_of(r, t, 1);
            if (c < 0 || k >= len) { ok = 0; break; }
            x->lab[k++] = (unsigned char)c;
        }
        if (ok && k == len) n++;
    }
    fclose(f);
    *count = n;
    return it;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── learning: the delta rule on the lit rows (softmax readout) ── */

static void learn_letter(float *w, int K, const uint32_t *feats, int n, int target, float lr) {
    float z[READER_MAX_CLASSES] = {0};
    for (int f = 0; f < n; f++) {
        const float *row = w + (size_t)feats[f] * K;
        for (int k = 0; k < K; k++) z[k] += row[k];
    }
    float m = z[0];
    for (int k = 1; k < K; k++) if (z[k] > m) m = z[k];
    float sum = 0;
    for (int k = 0; k < K; k++) { z[k] = expf(z[k] - m); sum += z[k]; }
    for (int k = 0; k < K; k++) z[k] = -lr * (z[k] / sum - (k == target));
    for (int f = 0; f < n; f++) {
        float *row = w + (size_t)feats[f] * K;
        for (int k = 0; k < K; k++) row[k] += z[k];
    }
}

/* Make int8 from the float table: one scale, so reading is integer adds. */
static int quantize(Reader *r) {
    size_t N = (size_t)READER_BUCKETS * r->classes;
    float mx = 1e-9f;
    for (size_t i = 0; i < N; i++) { float a = fabsf(r->w[i]); if (a > mx) mx = a; }
    /* no clipping: the rare big weights are the ones that decide (clipping
     * at half the largest cost 8 points of whole words) */
    int8_t *q = realloc(r->q, N);
    if (!q) return -1;
    r->q = q;
    r->scale = mx / 127.0f;
    for (size_t i = 0; i < N; i++) {
        float v = r->w[i] / r->scale;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        r->q[i] = (int8_t)lrintf(v);
    }
    return 0;
}

static void score_items(const Reader *r, const ReaderItem *it, int n, int test,
                        int *letters, int *letters_ok, int *words, int *words_ok) {
    int out[READER_MAX_WORD];
    *letters = *letters_ok = *words = *words_ok = 0;
    for (int i = 0; i < n; i++) {
        if (it[i].test != test) continue;
        int len = reader_read_word(r, it[i].word, out), wrong = 0;
        for (int j = 0; j < len; j++) wrong += out[j] != it[i].lab[j];
        *letters += len;
        *letters_ok += len - wrong;
        *words += 1;
        *words_ok += wrong == 0;
    }
}

int reader_learn(Reader *r, const char *tsv, int epochs) {
    reader_free(r);
    r->classes = 0;
    int n = 0;
    ReaderItem *it = load_items(r, tsv, &n);
    if (!it || n == 0) {
        free(it);
        printf("  Could not read %s\n", tsv);
        return -1;
    }
    r->w = calloc((size_t)READER_BUCKETS * r->classes, sizeof(float));
    int *order = malloc((size_t)n * sizeof(int));
    if (!r->w || !order) { free(it); free(order); return -1; }
    int ntrain = 0;
    for (int i = 0; i < n; i++) if (!it[i].test) order[ntrain++] = i;
    printf("  %d words to learn from, %d held back for the test, %d sounds\n",
           ntrain, n - ntrain, r->classes);
    uint32_t feats[READER_FEATS];
    unsigned seed = 12345;
    float lr = READER_LR;
    double t0 = now_s();
    for (int ep = 0; ep < epochs; ep++) {
        for (int i = ntrain - 1; i > 0; i--) {       /* shuffle */
            seed = seed * 1103515245u + 12345u;
            int j = (int)((seed >> 8) % (unsigned)(i + 1));
            int t = order[i]; order[i] = order[j]; order[j] = t;
        }
        for (int i = 0; i < ntrain; i++) {
            const ReaderItem *x = &it[order[i]];
            for (int j = 0; j < x->len; j++) {
                int nf = reader_features(x->word, x->len, j, feats);
                learn_letter(r->w, r->classes, feats, nf, x->lab[j], lr);
            }
        }
        lr *= READER_LR_DECAY;
        quantize(r);
        r->ready = 1;
        int L, Lok, Wn, Wok;
        score_items(r, it, n, 1, &L, &Lok, &Wn, &Wok);
        printf("  pass %2d: unseen words %5.1f%% right, sounds %5.1f%% right\n",
               ep + 1, 100.0 * Wok / Wn, 100.0 * Lok / L);
        MIMIR_PROBE("reader.words_right", 100.0 * Wok / Wn);
    }
    printf("  learned in %.1f s\n", now_s() - t0);
    free(order);
    free(it);
    free(r->w);
    r->w = NULL;
    return 0;
}

/* ── test and speed ── */

int reader_test(const Reader *r, const char *tsv) {
    Reader tmp;
    reader_init(&tmp);
    memcpy(tmp.label, r->label, sizeof r->label);
    tmp.classes = r->classes;
    int n = 0;
    ReaderItem *it = load_items(&tmp, tsv, &n);
    if (!it) { printf("  Could not read %s\n", tsv); return -1; }
    if (tmp.classes != r->classes) {
        printf("  %s has sounds this reader never learned; run 'reader learn'.\n", tsv);
        free(it);
        return -1;
    }
    int L, Lok, Wn, Wok;
    score_items(r, it, n, 1, &L, &Lok, &Wn, &Wok);
    printf("  Unseen words: %d of %d read fully right (%.1f%%)\n", Wok, Wn, 100.0 * Wok / Wn);
    printf("  Sounds:       %d of %d letters right (%.1f%%)\n", Lok, L, 100.0 * Lok / L);
    score_items(r, it, n, 0, &L, &Lok, &Wn, &Wok);
    printf("  Learned words: %.1f%% read fully right\n", 100.0 * Wok / Wn);
    free(it);
    return 0;
}

/* Read the same list of words over and over for about a second. */
void reader_speed(const Reader *r) {
    static const char *words[] = {
        "the", "phone", "because", "school", "friend", "night", "laugh", "though",
        "people", "water", "children", "beautiful", "different", "thought",
        "enough", "island", "knight", "photograph", "question", "together",
    };
    const int nw = (int)(sizeof words / sizeof *words);
    int out[READER_MAX_WORD];
    long letters = 0, nwords = 0;
    volatile int sink = 0;
    double t0 = now_s(), t;
    do {
        for (int i = 0; i < nw; i++) {
            int len = reader_read_word(r, words[i], out);
            sink += out[len - 1];
            letters += len;
        }
        nwords += nw;
        t = now_s() - t0;
    } while (t < 1.0);
    printf("  One thread: %.0f letters/s, %.0f words/s\n", letters / t, nwords / t);
    MIMIR_PROBE("reader.letters_per_s", letters / t);
}

/* ── speaking: sounds to a respelling a kid can say ── */

static const char *say(const char *ph, int *stressed) {
    static const struct { const char *ph, *say; } T[] = {
        {"AA","ah"}, {"AE","a"}, {"AH","uh"}, {"AO","aw"}, {"AW","ow"}, {"AY","igh"},
        {"EH","e"}, {"ER","ur"}, {"EY","ay"}, {"IH","i"}, {"IY","ee"}, {"OW","oh"},
        {"OY","oy"}, {"UH","oo"}, {"UW","oo"}, {"CH","ch"}, {"DH","th"}, {"HH","h"},
        {"JH","j"}, {"NG","ng"}, {"SH","sh"}, {"TH","th"}, {"ZH","zh"}, {"B","b"},
        {"D","d"}, {"F","f"}, {"G","g"}, {"K","k"}, {"L","l"}, {"M","m"}, {"N","n"},
        {"P","p"}, {"R","r"}, {"S","s"}, {"T","t"}, {"V","v"}, {"W","w"}, {"Y","y"},
        {"Z","z"},
    };
    char base[4] = {0};
    int n = 0;
    while (ph[n] && isalpha((unsigned char)ph[n]) && n < 3) { base[n] = ph[n]; n++; }
    *stressed = ph[n] == '1';
    for (size_t i = 0; i < sizeof T / sizeof *T; i++)
        if (strcmp(T[i].ph, base) == 0) return T[i].say;
    return "?";
}

/* "phone" → sounds "F OW1 N", say "FOHn" (the loud part in capitals) */
void reader_say(const Reader *r, const char *word) {
    int out[READER_MAX_WORD];
    int len = reader_read_word(r, word, out);
    char sounds[512] = "", spoken[256] = "", letters[1024] = "";
    size_t ns = 0, nsp = 0, nl = 0;
    int j = 0;
    for (int i = 0; i < len; i++) {
        while (word[j] && !isalpha((unsigned char)word[j])) j++;
        const char *lab = r->label[out[i]];
        nl += snprintf(letters + nl, sizeof letters - nl, "%s%c→%s", i ? "  " : "",
                       tolower((unsigned char)word[j++]), lab);
        if (nl >= sizeof letters) nl = sizeof letters - 1;
        if (strcmp(lab, "-") == 0) continue;     /* silent letter */
        char tmp[READER_LABEL_LEN];
        snprintf(tmp, sizeof tmp, "%s", lab);
        char *save = NULL;   /* strtok_r: the caller may be mid-strtok */
        for (char *t = strtok_r(tmp, "_", &save); t; t = strtok_r(NULL, "_", &save)) {
            int loud;
            const char *s = say(t, &loud);
            ns += snprintf(sounds + ns, sizeof sounds - ns, "%s%s", ns ? " " : "", t);
            if (ns >= sizeof sounds) ns = sizeof sounds - 1;
            /* a loud vowel takes the consonants just before it with it */
            if (loud) {
                size_t k = nsp;
                while (k > 0 && !strchr("aeiouy", spoken[k - 1]) && islower((unsigned char)spoken[k - 1])) k--;
                for (size_t m = k; m < nsp; m++) spoken[m] = (char)toupper((unsigned char)spoken[m]);
            }
            for (const char *c = s; *c && nsp < sizeof spoken - 1; c++)
                spoken[nsp++] = loud ? (char)toupper((unsigned char)*c) : *c;
            spoken[nsp] = 0;
        }
    }
    printf("  %s  →  %s\n", word, spoken);
    printf("  sounds:  %s\n", sounds);
    printf("  letters: %s\n", letters);
}

/* ── saving ──
 * "MRDR" | int version | int buckets | int classes | float scale |
 * classes × READER_LABEL_LEN labels | buckets × classes int8           */

int reader_save(const Reader *r, const char *path) {
    if (!r->ready) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int head[3] = { READER_VERSION, READER_BUCKETS, r->classes };
    int ok = fwrite("MRDR", 1, 4, f) == 4
          && fwrite(head, sizeof head, 1, f) == 1
          && fwrite(&r->scale, sizeof r->scale, 1, f) == 1
          && fwrite(r->label, READER_LABEL_LEN, (size_t)r->classes, f) == (size_t)r->classes
          && fwrite(r->q, 1, (size_t)READER_BUCKETS * r->classes, f) == (size_t)READER_BUCKETS * r->classes;
    return (fclose(f) == 0 && ok) ? 0 : -1;
}

int reader_load(Reader *r, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    int head[3];
    float scale;
    int ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "MRDR", 4) == 0
          && fread(head, sizeof head, 1, f) == 1
          && head[0] == READER_VERSION && head[1] == READER_BUCKETS
          && head[2] > 0 && head[2] <= READER_MAX_CLASSES
          && fread(&scale, sizeof scale, 1, f) == 1;
    if (!ok) { fclose(f); return -1; }
    reader_free(r);
    r->classes = head[2];
    r->scale = scale;
    size_t N = (size_t)READER_BUCKETS * r->classes;
    r->q = malloc(N);
    ok = r->q
      && fread(r->label, READER_LABEL_LEN, (size_t)r->classes, f) == (size_t)r->classes
      && fread(r->q, 1, N, f) == N;
    fclose(f);
    if (!ok) { reader_free(r); return -1; }
    r->ready = 1;
    return 0;
}

/* ════════════════════════════════════════════════════════════════════
 * LEARNING ON THE FLY — the reader checks itself and fixes its mistakes
 *
 * Nobody teaches it word by word. It carries a pronouncing dictionary
 * (CMUdict, ~105,000 words once the held-back ones are left out: what a
 * word sounds like when someone says it). Every word it reads, it
 * compares its own sounds with the dictionary. If they differ, it:
 *   1. lines the right sounds up with the letters itself, using its own
 *      beliefs to decide which letters are silent or make two sounds
 *      (the best path through its scores: reader_align);
 *   2. remembers the word whole, as a sight word, the way a child learns
 *      "said" or "colonel": next time it is read from memory, exactly.
 * The spelling-rule table is not changed: its rows are shared by many
 * letter groups (hashing), so nudging them for one word costs others.
 * Measured on Alice in Wonderland: nudging (READER_RULE_LR 0.02 to 0.1)
 * cost 0.4 to 1.6 points of held-back words; sight words cost none.
 * Words not in the dictionary it reads but cannot check.
 *
 * The 2,541 held-back test words are left out of the dictionary, so
 * `reader test` still measures words it has never been corrected on.
 * ════════════════════════════════════════════════════════════════════ */

/* ── the dictionary: word → one or more pronunciations ── */

struct ReaderLex {
    char    *text;       /* "word\0PH PH PH\0" ... */
    uint32_t *slot;      /* open addressing: offset+1 of the word, 0 = empty */
    uint32_t  mask;
    int       words;
};

/* FNV-1a of the word (reader_lex_load has the same loop in line). */
static uint32_t lex_hash(const char *w) { return fnv(2166136261u, w, (int)strlen(w)); }

/* Every pronunciation of word: calls back with each phoneme string. */
static int lex_find(const ReaderLex *x, const char *w, const char **prons, int max) {
    int n = 0;
    for (uint32_t h = lex_hash(w) & x->mask; x->slot[h] && n < max; h = (h + 1) & x->mask) {
        const char *e = x->text + x->slot[h] - 1;
        if (strcmp(e, w) == 0) prons[n++] = e + strlen(e) + 1;
    }
    return n;
}

int reader_lex_load(Reader *r, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    ReaderLex *x = calloc(1, sizeof *x);
    x->text = malloc((size_t)size + 2);
    x->mask = (1u << 18) - 1;                        /* 262,144 slots */
    x->slot = calloc(x->mask + 1, sizeof *x->slot);
    if (!x->text || !x->slot) { fclose(f); free(x->text); free(x->slot); free(x); return -1; }
    size_t used = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *hash = strchr(line, '#');
        if (hash) *hash = 0;
        char *sp = strchr(line, ' ');
        if (!sp) continue;
        *sp = 0;
        char *paren = strchr(line, '(');                 /* "word(2)": another way to say it */
        if (paren) *paren = 0;
        int ok = line[0] != 0 && strlen(line) < READER_MAX_WORD;
        for (char *c = line; *c && ok; c++) ok = *c >= 'a' && *c <= 'z';
        /* Held back for the test? The same one-in-ten rule as
         * sandbox/reader_data.py (zlib crc32 % 10 == 0), done here in the
         * loop so loading stays one step on the bench, not 100,000. */
        uint32_t crc = 0xFFFFFFFFu;
        for (const char *c = line; *c; c++) {
            crc ^= (unsigned char)*c;
            for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1)));
        }
        if (!ok || (~crc) % 10 == 0) continue;
        char *ph = sp + 1;
        size_t pl = strcspn(ph, "\r\n");
        while (pl && ph[pl - 1] == ' ') pl--;
        ph[pl] = 0;
        size_t wl = strlen(line);
        if (used + wl + pl + 2 > (size_t)size) break;
        uint32_t h = 2166136261u;                         /* lex_hash, in line */
        for (const char *c = line; *c; c++) { h ^= (unsigned char)*c; h *= 16777619u; }
        h &= x->mask;
        while (x->slot[h]) h = (h + 1) & x->mask;
        x->slot[h] = (uint32_t)used + 1;
        memcpy(x->text + used, line, wl + 1);
        memcpy(x->text + used + wl + 1, ph, pl + 1);
        used += wl + pl + 2;
        if (!paren) x->words++;
        if ((uint32_t)x->words * 2 > x->mask) break;     /* keep the table half empty */
    }
    fclose(f);
    r->lex = x;
    return x->words;
}

/* Turn letters' classes into the sounds they make: "-" is silent, "K_S" is two. */
static int sounds_of(const Reader *r, const int *cls, int len, char sounds[][8], int max) {
    int n = 0;
    for (int i = 0; i < len; i++) {
        const char *lab = r->label[cls[i]];
        if (strcmp(lab, "-") == 0) continue;
        char tmp[READER_LABEL_LEN], *save = NULL;
        snprintf(tmp, sizeof tmp, "%s", lab);
        for (char *t = strtok_r(tmp, "_", &save); t && n < max; t = strtok_r(NULL, "_", &save))
            snprintf(sounds[n++], 8, "%s", t);
    }
    return n;
}

static int split_sounds(const char *pron, char sounds[][8], int max) {
    char tmp[256], *save = NULL;
    snprintf(tmp, sizeof tmp, "%s", pron);
    int n = 0;
    for (char *t = strtok_r(tmp, " ", &save); t && n < max; t = strtok_r(NULL, " ", &save))
        snprintf(sounds[n++], 8, "%s", t);
    return n;
}

/* ── the float table, made on first need from the int8 one ── */

static int ensure_float(Reader *r) {
    if (r->w) return 0;
    size_t N = (size_t)READER_BUCKETS * r->classes;
    r->w = malloc(N * sizeof(float));
    if (!r->w) return -1;
    for (size_t i = 0; i < N; i++) r->w[i] = r->q[i] * r->scale;
    return 0;
}

/* After learning, copy the changed rows back to int8 (same scale). */
static void requantize_rows(Reader *r, const uint32_t *feats, int n) {
    const int K = r->classes;
    for (int f = 0; f < n; f++) {
        const float *w = r->w + (size_t)feats[f] * K;
        int8_t *q = r->q + (size_t)feats[f] * K;
        for (int k = 0; k < K; k++) {
            float v = w[k] / r->scale;
            q[k] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : lrintf(v));
        }
    }
}

/* ── lining the right sounds up with the letters, by its own beliefs ──
 * Each letter makes nothing ("-"), one sound, or two ("K_S"). Of all the
 * ways to split the sounds over the letters, take the one its table
 * likes best (sum of log-probabilities). Returns 0, or -1 if no split
 * uses only sounds it knows. */
int reader_align(Reader *r, const char *w, int len, char ph[][8], int np, unsigned char *lab) {
    enum { MAXP = READER_MAX_WORD * 2 };
    if (np > MAXP - 1) return -1;
    const int K = r->classes;
    int sil = class_of(r, "-", 0), one[MAXP], two[MAXP];
    for (int j = 0; j < np; j++) {
        one[j] = class_of(r, ph[j], 0);
        char pair[READER_LABEL_LEN * 2];
        snprintf(pair, sizeof pair, "%s_%s", ph[j], j + 1 < np ? ph[j + 1] : "");
        two[j] = j + 1 < np ? class_of(r, pair, 0) : -1;
    }
    static float best[READER_MAX_WORD + 1][MAXP + 1];
    static signed char step[READER_MAX_WORD + 1][MAXP + 1];
    for (int i = 0; i <= len; i++)
        for (int j = 0; j <= np; j++) best[i][j] = -1e30f;
    best[0][0] = 0;
    uint32_t feats[READER_FEATS];
    float lp[READER_MAX_CLASSES];
    for (int i = 0; i < len; i++) {
        int n = reader_features(w, len, i, feats);
        float m = -1e30f, sum = 0;
        for (int k = 0; k < K; k++) {
            float z = 0;
            for (int f = 0; f < n; f++) z += r->q[(size_t)feats[f] * K + k];
            lp[k] = z * r->scale;
            if (lp[k] > m) m = lp[k];
        }
        for (int k = 0; k < K; k++) sum += expf(lp[k] - m);
        float lz = m + logf(sum);
        for (int j = 0; j <= np; j++) {
            if (best[i][j] <= -1e29f) continue;
            if (sil >= 0 && best[i][j] + lp[sil] - lz > best[i + 1][j])
                best[i + 1][j] = best[i][j] + lp[sil] - lz, step[i + 1][j] = 0;
            if (j < np && one[j] >= 0 && best[i][j] + lp[one[j]] - lz > best[i + 1][j + 1])
                best[i + 1][j + 1] = best[i][j] + lp[one[j]] - lz, step[i + 1][j + 1] = 1;
            if (j + 1 < np && two[j] >= 0 && best[i][j] + lp[two[j]] - lz > best[i + 1][j + 2])
                best[i + 1][j + 2] = best[i][j] + lp[two[j]] - lz, step[i + 1][j + 2] = 2;
        }
    }
    if (best[len][np] <= -1e29f) return -1;
    for (int i = len, j = np; i > 0; i--) {
        int s = step[i][j];
        j -= s;
        lab[i - 1] = (unsigned char)(s == 0 ? sil : s == 1 ? one[j] : two[j]);
    }
    return 0;
}

/* One delta-rule pass over the letters it reads wrong; the right ones are left alone. */
static void learn_word(Reader *r, const char *w, int len, const unsigned char *lab, float lr) {
    uint32_t feats[READER_FEATS];
    int out[READER_MAX_WORD];
    reader_read_word(r, w, out);
    for (int i = 0; i < len; i++) {
        if (out[i] == lab[i]) continue;
        int n = reader_features(w, len, i, feats);
        learn_letter(r->w, r->classes, feats, n, lab[i], lr);
        requantize_rows(r, feats, n);
    }
}

/* Read a word, check it against the dictionary, fix it if wrong. */
ReadCheck reader_check(Reader *r, const char *word, int verbose) {
    char w[READER_MAX_WORD];
    int len = 0;
    for (const char *p = word; *p && len < READER_MAX_WORD - 1; p++)
        if (isalpha((unsigned char)*p)) w[len++] = (char)tolower((unsigned char)*p);
    w[len] = 0;
    r->seen++;
    const char *prons[8];
    int np = (r->lex && len) ? lex_find(r->lex, w, prons, 8) : 0;
    if (np == 0) { r->unchecked++; return READ_UNCHECKED; }

    int out[READER_MAX_WORD];
    char mine[READER_MAX_WORD * 2][8], right[READER_MAX_WORD * 2][8];
    reader_read_word(r, w, out);
    int nm = sounds_of(r, out, len, mine, READER_MAX_WORD * 2);
    for (int p = 0; p < np; p++) {                       /* any way of saying it counts */
        int nr = split_sounds(prons[p], right, READER_MAX_WORD * 2), same = nr == nm;
        for (int j = 0; j < nr && same; j++) same = strcmp(mine[j], right[j]) == 0;
        if (same) { r->right++; return READ_RIGHT; }
    }
    /* Wrong: line up the first way of saying it, then fix those rows. */
    unsigned char lab[READER_MAX_WORD];
    int nr = split_sounds(prons[0], right, READER_MAX_WORD * 2);
    if (reader_align(r, w, len, right, nr, lab) != 0) {
        r->stuck++;
        if (verbose) printf("  (cannot fix '%s': it needs a sound this reader has not got)\n", w);
        return READ_STUCK;
    }
    /* Remember it whole (a sight word), and, gently, let the spelling
     * rules learn from it too: one small delta step on the wrong letters. */
    if (READER_RULE_LR > 0 && ensure_float(r) == 0) learn_word(r, w, len, lab, READER_RULE_LR);
    int ok = sight_put(r, w, len, lab) == 0;
    r->dirty = 1;
    if (ok) r->fixed++; else r->stuck++;
    MIMIR_PROBE("reader.fixed", r->fixed);
    if (verbose) {
        printf("  ✗ I said it wrong. The dictionary says %s.\n", prons[0]);
        printf("    %s Now:\n", ok ? "Remembered as a sight word." : "Sight-word memory is full.");
        reader_say(r, w);
    }
    return ok ? READ_FIXED : READ_STUCK;
}

void reader_stats(const Reader *r) {
    long checked = r->seen - r->unchecked;
    printf("  Since start-up: read %ld words, checked %ld against the dictionary\n", r->seen, checked);
    if (checked)
        printf("  Right first time %ld (%.1f%%), fixed %ld, could not fix %ld; %d sight words\n",
               r->right, 100.0 * r->right / checked, r->fixed, r->stuck, r->nsight);
}

/* Read a text file from start to end, checking and fixing as it goes. */
void reader_book(Reader *r, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { printf("  Could not open %s\n", path); return; }
    long seen0 = r->seen, right0 = r->right, unch0 = r->unchecked;
    long blk_seen = 0, blk_right = 0, blk_unch = 0, n = 0;
    char w[READER_MAX_WORD];
    int len = 0, c;
    printf("  Reading %s, checking every word and fixing mistakes as I go.\n", path);
    printf("  %8s  %s\n", "words", "right first time (last 2,000 checked words)");
    do {
        c = fgetc(f);
        if (c != EOF && isalpha(c)) {
            if (len < READER_MAX_WORD - 1) w[len++] = (char)tolower(c);
            continue;
        }
        if (len == 0) continue;
        w[len] = 0;
        len = 0;
        if (strlen(w) == 1 && w[0] != 'a' && w[0] != 'i') continue;   /* the s of "Alice's" */
        long s0 = r->seen, r0 = r->right, u0 = r->unchecked;
        reader_check(r, w, 0);
        n++;
        blk_seen += r->seen - s0;
        blk_right += r->right - r0;
        blk_unch += r->unchecked - u0;
        if (blk_seen - blk_unch >= 2000) {
            printf("  %8ld  %5.1f%%\n", n, 100.0 * blk_right / (blk_seen - blk_unch));
            MIMIR_PROBE("reader.right_first_time", 100.0 * blk_right / (blk_seen - blk_unch));
            blk_seen = blk_right = blk_unch = 0;
        }
    } while (c != EOF);
    fclose(f);
    long checked = (r->seen - seen0) - (r->unchecked - unch0);
    printf("  Done: %ld words, %ld checked, %.1f%% right first time.\n", r->seen - seen0, checked,
           checked ? 100.0 * (r->right - right0) / checked : 0.0);
}

void reader_close(Reader *r) {
    reader_free(r);
    free(r->sight);
    r->sight = NULL;
    r->nsight = 0;
    if (r->lex) { free(r->lex->text); free(r->lex->slot); free(r->lex); r->lex = NULL; }
}

/* Sight words as text, one per line: "school\tS K - UW1 - L". */
int reader_sight_save(const Reader *r, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    for (uint32_t h = 0; r->sight && h <= r->sight_mask; h++) {
        const ReaderMemory *m = &r->sight[h];
        if (!m->len) continue;
        fprintf(f, "%s\t", m->word);
        for (int i = 0; i < m->len; i++) fprintf(f, "%s%s", i ? " " : "", r->label[m->lab[i]]);
        fprintf(f, "\n");
    }
    return fclose(f) == 0 ? 0 : -1;
}

int reader_sight_load(Reader *r, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[512];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        char *save = NULL, *w = strtok_r(line, "\t", &save), *labs = strtok_r(NULL, "\t\n", &save);
        if (!w || !labs || strlen(w) >= READER_MAX_WORD) continue;
        unsigned char lab[READER_MAX_WORD];
        int k = 0, len = (int)strlen(w), ok = 1;
        char *s2 = NULL;
        for (char *t = strtok_r(labs, " ", &s2); t && ok; t = strtok_r(NULL, " ", &s2)) {
            int c = class_of(r, t, 0);
            ok = c >= 0 && k < len;
            if (ok) lab[k++] = (unsigned char)c;
        }
        if (ok && k == len && sight_put(r, w, len, lab) == 0) n++;
    }
    fclose(f);
    return n;
}
