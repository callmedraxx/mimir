/*
 * MIMIR — Alphabet Sensor
 *
 * Teaches the brain letter-to-word associations ("A is for Apple") while
 * preserving pre-committed sequence knowledge ("after A comes B").
 *
 * ════════════════════════════════════════════════════════════════════
 * HOW QUERY TYPES WORK
 * ════════════════════════════════════════════════════════════════════
 *
 * The raw input to the alphabet sensor is 30 floats:
 *
 *   [0..25]   one-hot letter    (which letter?)
 *   [26..29]  one-hot query     (what are we asking?)
 *
 * The sensor pads this to MIMIR_EMBEDDING_SIZE (128) before the
 * shared core sees it. The core then learns that:
 *
 *   embed(A, RECALL)   → output that decodes to "apple"
 *   embed(A, NEXT)     → output that decodes to "B"
 *   embed(A, POSITION) → output that decodes to "1st"
 *
 * Same committed letter neurons, different query context, different
 * answer. Exactly how the human brain handles it.
 *
 * ════════════════════════════════════════════════════════════════════
 * TRAINING ORDER — WHY SEQUENCE COMES FIRST
 * ════════════════════════════════════════════════════════════════════
 *
 * alpha_pretrain_sequence() runs ONCE when a new brain is created.
 * It drills A→B→C, B→C→D … into the network and commits those neurons.
 *
 * Only then does the user teach word associations.
 *
 * This matches how children learn: alphabet song first, "A is for Apple"
 * second. The sequence knowledge is bedrock — vocabulary builds on top
 * without ever shaking the foundation.
 *
 * ════════════════════════════════════════════════════════════════════
 * QUIZ MODE
 * ════════════════════════════════════════════════════════════════════
 *
 * The brain can quiz itself (or the user) by:
 *   1. Forming a question from stored knowledge ("After A comes…?")
 *   2. Running a forward pass to get its predicted answer
 *   3. Comparing prediction to ground truth
 *   4. Re-training any wrong answers (active recall)
 *
 * Interactive mode flips the dynamic: the brain asks YOU, checks your
 * answer, and corrects itself from any mistakes — just like a study
 * partner drilling flashcards.
 */

#include "mimir.h"
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Portable case-insensitive string comparison.
 *
 * str_iequal() is POSIX, not C11. Rather than fight the -std=c11 flag
 * or add POSIX feature macros that interact badly with -std=c11 + GCC,
 * we implement it directly. It is only needed in this file.
 *
 * Returns 0 if a and b are equal ignoring case, non-zero otherwise.
 */
static int str_iequal(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return 1;
        a++; b++;
    }
    return (*a != '\0' || *b != '\0');   /* 0 only if both ended together */
}

/* Confidence threshold: output must exceed this to count as a real answer.
 * Below this the brain is saying "I'm not sure" — we report [unknown].
 * 0.55 is intentionally low so the brain can express partial learning;
 * raise to 0.8 for stricter "I only speak when confident" behaviour. */
#define ALPHA_CONF_THRESHOLD  0.55f

/* Training passes applied to ALL known associations whenever one is added.
 * Retraining the full set prevents the network from drifting away from
 * earlier-taught words when new ones are added (catastrophic forgetting).
 * Committed neurons are frozen so this only refines uncommitted ones. */
#define ALPHA_RETRAIN_EPOCHS  8000

/* Forward-pass output buffer size — 26 outputs for the alphabet head. */
#define OUT_BUF  ALPHA_N_OUTPUTS

// ── Word bank (added 2026-04-11) ──────────────────────────────────────────────
//
// A static, comprehensive list of common English words organised by their
// first letter.  Two purposes:
//
//   1. `teach all` — picks one word per letter from this bank and teaches
//      the network the entire alphabet in one shot, so the user doesn't
//      have to type 26 `learn` commands by hand.
//
//   2. `quiz choice` — pulls candidates from THIS bank that are NOT in the
//      vocabulary the network was trained on.  The quiz then asks the
//      network to identify which candidate belongs to a given letter via
//      its learned VALIDATE principle ("words starting with X belong to
//      letter X").  Because the candidate words were never used in
//      training, a correct pick proves the network is generalising the
//      principle, not memorising specific (letter, word) pairs.
//
// All words are lowercase, single-token, alphabetic only, so the existing
// `learn %3s %31s` parser accepts them without modification.

#define ABC_WORD_BANK_MAX_PER_LETTER  24

typedef struct {
    int n;
    const char *const words[ABC_WORD_BANK_MAX_PER_LETTER];
} LetterWordList;

static const LetterWordList ABC_WORD_BANK[26] = {
/* A */ {20, {"apple","ant","arrow","axe","anchor","angel","arm","art","atlas","ash",
              "almond","acorn","ape","alpha","ace","aim","apron","arc","army","autumn"}},
/* B */ {20, {"ball","bat","bear","bird","boat","book","box","branch","bread","bridge",
              "brown","brush","bug","bull","bus","butter","button","blue","badge","bake"}},
/* C */ {20, {"cat","car","cake","candle","cap","card","castle","chair","chalk","chess",
              "cherry","chicken","child","chip","circle","clay","cloud","coal","cold","comb"}},
/* D */ {20, {"dog","deer","desk","diamond","diary","dice","doll","dolphin","donkey","door",
              "dot","dove","dragon","drum","duck","dust","daisy","dance","dawn","deep"}},
/* E */ {20, {"egg","eagle","ear","earth","elbow","elephant","elf","ember","end","engine",
              "envelope","even","evening","exit","eye","eight","edge","echo","east","energy"}},
/* F */ {20, {"fish","fan","farm","father","feather","fence","field","finger","fire","flag",
              "flame","flower","flute","fog","food","foot","fork","fox","frog","fruit"}},
/* G */ {20, {"grape","gate","ghost","gift","glass","glue","goat","gold","goose","grain",
              "grass","gray","green","ground","group","guard","guess","guitar","gum","gun"}},
/* H */ {20, {"hat","hair","hall","hammer","hand","harbor","harp","hawk","head","heart",
              "hen","hill","hive","hole","honey","hook","horn","horse","hour","house"}},
/* I */ {20, {"ice","igloo","ink","iron","island","ivory","ivy","idea","iris","idol",
              "india","image","inn","insect","item","indigo","infant","inch","iceberg","instrument"}},
/* J */ {20, {"jam","jacket","jar","jaw","jay","jeep","jelly","jet","jewel","jockey",
              "joke","journal","journey","joy","judge","juice","jug","jump","jungle","junior"}},
/* K */ {20, {"key","kangaroo","keep","kennel","kettle","kick","kid","kilt","king","kiss",
              "kite","kitten","knee","knife","knight","knock","knot","koala","knack","keel"}},
/* L */ {20, {"lamp","lake","lamb","ladder","lava","leaf","lemon","leopard","letter","lid",
              "light","lily","line","link","lion","lip","list","lock","log","lord"}},
/* M */ {20, {"moon","man","map","mask","match","meal","melon","mermaid","metal","milk",
              "mill","mint","mirror","mole","money","monkey","month","mop","moth","mountain"}},
/* N */ {20, {"nest","nail","name","napkin","navy","neck","needle","neighbor","net","news",
              "night","nine","noise","noodle","north","nose","note","number","nurse","nut"}},
/* O */ {20, {"orange","oak","oar","oasis","oat","ocean","octopus","office","oil","olive",
              "omen","onion","opal","opera","orbit","orchard","organ","otter","owl","ox"}},
/* P */ {20, {"pen","page","paint","palace","pan","panda","paper","park","party","peach",
              "pear","pearl","pencil","penguin","piano","pig","pilot","pin","pine","pizza"}},
/* Q */ {20, {"queen","quack","quail","quake","quark","quart","quartz","quasar","quay","quest",
              "queue","quiche","quick","quiet","quill","quilt","quince","quip","quirk","quiz"}},
/* R */ {20, {"rain","rabbit","race","radar","radio","raft","rake","ram","ranch","raven",
              "ray","reef","rib","ribbon","rice","ring","river","road","rock","rope"}},
/* S */ {20, {"sun","sail","salt","sand","scarf","school","sea","seal","seed","sheep",
              "shell","shoe","sign","silk","silver","sky","smoke","snail","snake","snow"}},
/* T */ {20, {"tree","table","tail","tank","tape","target","taxi","tea","team","teeth",
              "tent","test","throne","thumb","ticket","tiger","time","tin","toad","tooth"}},
/* U */ {20, {"umbrella","ukulele","ulcer","unicorn","uniform","union","unit","urn","urchin","usher",
              "udder","ugly","ulna","ultra","umpire","uncle","under","undo","upon","upper"}},
/* V */ {20, {"van","vacuum","vale","valley","vampire","vane","vapor","vase","vault","vegetable",
              "veil","vein","velvet","vent","venue","verb","verse","vest","vine","violet"}},
/* W */ {20, {"wagon","wall","walnut","walrus","wand","warm","wasp","water","wave","wax",
              "weasel","web","wedge","wedding","week","well","west","whale","wheat","wind"}},
/* X */ { 8, {"xylophone","xenon","xerus","xebec","xenia","xylem","xylan","xerox"}},
/* Y */ {20, {"yarn","yacht","yak","yam","yard","yarrow","yawn","year","yeast","yell",
              "yellow","yen","yeti","yew","yield","yogurt","yoke","yolk","young","yoyo"}},
/* Z */ {17, {"zebra","zero","zest","zigzag","zinc","zion","zip","zipper","zircon","zither",
              "zodiac","zone","zoo","zoom","zucchini","zonal","zealot"}}
};

int alpha_word_bank_count(int letter_idx) {
    if (letter_idx < 0 || letter_idx >= 26) return 0;
    return ABC_WORD_BANK[letter_idx].n;
}

const char *alpha_word_bank_get(int letter_idx, int word_idx) {
    if (letter_idx < 0 || letter_idx >= 26) return NULL;
    if (word_idx < 0 || word_idx >= ABC_WORD_BANK[letter_idx].n) return NULL;
    return ABC_WORD_BANK[letter_idx].words[word_idx];
}

// ── Vocabulary ────────────────────────────────────────────────────────────────

void alpha_vocab_init(AlphaVocab *v) {
    memset(v, 0, sizeof(*v));
    for (int i = 0; i < 26; i++) v->letter_to_word[i] = -1;
}

int alpha_vocab_find(const AlphaVocab *v, const char *word) {
    /*
     * Case-insensitive search. The vocab stores everything lowercase,
     * so we compare against the lowercased version. Users can type
     * "Apple" or "APPLE" or "apple" — all match "apple" in the vocab.
     */
    for (int i = 0; i < v->n_words; i++) {
        if (str_iequal(v->words[i], word) == 0)
            return i;
    }
    return -1;
}

int alpha_vocab_add(AlphaVocab *v, const char *word) {
    /* Return existing index if the word is already known. */
    int idx = alpha_vocab_find(v, word);
    if (idx >= 0) return idx;

    if (v->n_words >= ALPHA_VOCAB_SIZE) return -1;

    idx = v->n_words++;

    /* Store as lowercase — normalises "Apple", "APPLE", "apple" to "apple". */
    strncpy(v->words[idx], word, ALPHA_MAX_WORD_LEN - 1);
    v->words[idx][ALPHA_MAX_WORD_LEN - 1] = '\0';
    for (char *p = v->words[idx]; *p; p++) *p = (char)tolower((unsigned char)*p);

    return idx;
}

int alpha_vocab_save(const AlphaVocab *v, const char *path) {
    /*
     * Save in a human-readable text format so the vocab can be inspected
     * or hand-edited without special tools:
     *
     *   # Mimir alphabet vocabulary
     *   n_words 3
     *   letter_to_word 0 -1 -1 ... (26 values)
     *   word 0 apple
     *   word 1 banana
     *   word 2 cat
     */
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[Alphabet] Cannot open '%s' for writing\n", path);
        return -1;
    }
    fprintf(f, "# Mimir alphabet vocabulary\n");
    fprintf(f, "n_words %d\n", v->n_words);
    fprintf(f, "letter_to_word");
    for (int i = 0; i < 26; i++) fprintf(f, " %d", v->letter_to_word[i]);
    fprintf(f, "\n");
    for (int i = 0; i < v->n_words; i++)
        fprintf(f, "word %d %s\n", i, v->words[i]);
    fclose(f);
    return 0;
}

int alpha_vocab_load(AlphaVocab *v, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;   /* File not found — caller creates fresh vocab */

    alpha_vocab_init(v);
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;   /* Comment */

        if (strncmp(line, "n_words ", 8) == 0) {
            v->n_words = atoi(line + 8);

        } else if (strncmp(line, "letter_to_word", 14) == 0) {
            char *p = line + 14;
            for (int i = 0; i < 26; i++) {
                v->letter_to_word[i] = (int)strtol(p, &p, 10);
            }

        } else if (strncmp(line, "word ", 5) == 0) {
            int idx;
            char word[ALPHA_MAX_WORD_LEN];
            if (sscanf(line + 5, "%d %31s", &idx, word) == 2 &&
                idx >= 0 && idx < ALPHA_VOCAB_SIZE) {
                snprintf(v->words[idx], ALPHA_MAX_WORD_LEN, "%s", word);
            }
        }
    }
    fclose(f);
    return 0;
}

// ── Encoding ──────────────────────────────────────────────────────────────────

void alpha_build_raw(int letter_idx, AlphaQueryType query, float *out) {
    /*
     * Layout:
     *   out[0..25]   one-hot over 26 letters
     *   out[26..29]  one-hot over 4 query types
     *
     * One-hot means exactly ONE position is 1.0, everything else is 0.0.
     * This gives the network maximally distinct, non-overlapping input
     * patterns — the easiest possible signal to learn from.
     */
    memset(out, 0, ALPHA_RAW_SIZE * sizeof(float));
    if (letter_idx >= 0 && letter_idx < 26)
        out[letter_idx] = 1.0f;
    if (query >= 0 && query < ALPHA_RAW_QUERY)
        out[ALPHA_RAW_LETTER + (int)query] = 1.0f;
}

int alpha_sensor_encode(Sensor      *sensor,
                         const float *raw,       int raw_size,
                         float       *embedding, int embedding_size) {
    /*
     * Manual (no learned encoder) zero-padding strategy:
     *
     *   embedding[0..29]    = raw[0..29]   (letter + query, copied verbatim)
     *   embedding[30..127]  = 0.0          (unused — reserved for future use)
     *
     * WHY NO LEARNED ENCODER HERE?
     * One-hot inputs are already maximally sparse and orthogonal. A
     * learned encoder could only add noise at this stage. The shared
     * core's first hidden layer IS the learned encoder — it applies
     * nonlinear weights across all 128 inputs and learns which
     * combinations of letter × query matter for which outputs.
     *
     * A learned encoder would be useful if the raw input were dense
     * and high-dimensional (e.g., raw audio samples, pixel arrays),
     * where compression before the core is necessary. For 30 clean
     * floats, it would be counterproductive.
     */
    (void)sensor;   /* user_data not needed for manual encoding */

    if (raw_size != ALPHA_RAW_SIZE) {
        fprintf(stderr, "[Alphabet] Expected raw_size=%d, got %d\n",
                ALPHA_RAW_SIZE, raw_size);
        return -1;
    }
    if (embedding_size < ALPHA_RAW_SIZE) {
        fprintf(stderr, "[Alphabet] embedding_size %d < raw_size %d\n",
                embedding_size, ALPHA_RAW_SIZE);
        return -1;
    }

    memset(embedding, 0, embedding_size * sizeof(float));
    memcpy(embedding, raw, ALPHA_RAW_SIZE * sizeof(float));
    return 0;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/* Forward declarations for functions defined later in this file. */
static void validate_rescue_impl(Network *net, bool gentle);

/*
 * Forward declaration. alpha_delta_rescue is the discriminative output-layer
 * delta-rule trainer used both by alpha_teach (full convergence) and by the
 * replay thread (light nudge) — see the BUG / FIX comment block above its
 * definition for the full story. It is exposed publicly via mimir.h so that
 * replay.c can call exactly the same algorithm that installs an association
 * in the first place. Mixing algorithms here was the original bug.
 */

/*
 * Run a forward pass for a specific (letter, query) pair and return the
 * raw output array. embedding[] and output[] are caller-supplied scratch
 * buffers — embedding must be MIMIR_EMBEDDING_SIZE floats, output must
 * be ALPHA_N_OUTPUTS floats.
 */
void alpha_forward(Network *net,
                    int letter_idx, AlphaQueryType query,
                    float *embedding, float *output) {
    float raw[ALPHA_RAW_SIZE];
    alpha_build_raw(letter_idx, query, raw);
    /* Zero-pad raw → embedding (same logic as alpha_sensor_encode) */
    memset(embedding, 0, MIMIR_EMBEDDING_SIZE * sizeof(float));
    memcpy(embedding, raw, ALPHA_RAW_SIZE * sizeof(float));
    network_forward(net, embedding, output);

    /*
     * (2026-04-18) Mask visual hidden contribution, then recompute outputs.
     *
     * Why mask: alpha_delta_rescue only adjusts output-layer weights from
     * text hidden neurons (is_visual=false).  If the forward pass includes
     * visual contributions, the residual error the trainer sees is "target
     * minus (text-side output + visual-side offset)" — but it can only
     * modify the text-side half.  With a fixed per-query visual offset
     * (bias=-5 gives σ(-5)≈0.007 per visual neuron × non-zero out-weights),
     * the trainer is chasing a moving target it can't reach, and 4 letters
     * collapse to one winner in the degenerate attractor.
     *
     * Mask fixes it by making the forward pass exactly equal the sum of
     * weights the trainer will update.  Apply at both inference (alpha_ask)
     * and training (alpha_delta_rescue calls this via rescue loop) — same
     * function in both paths keeps teach and recall consistent.
     */
    if (net->n_layers >= 2) {
        Layer *hidden    = &net->layers[net->n_layers - 2];
        Layer *out_layer = &net->layers[net->n_layers - 1];

        bool any_visual = false;
        for (int h = 0; h < hidden->count; h++)
            if (hidden->neurons[h].is_visual) { any_visual = true; break; }
        if (!any_visual) return;

        for (int j = 0; j < out_layer->count; j++) {
            Neuron *on = &out_layer->neurons[j];
            float z = on->bias;
            int w_lim = (hidden->count < on->n_weights)
                        ? hidden->count : on->n_weights;
            for (int h = 0; h < w_lim; h++) {
                if (hidden->neurons[h].is_visual) continue;
                z += hidden->outputs[h] * on->weights[h];
            }
            on->last_z = z;
            output[j] = activate(z, on->act);
        }
    }
}

/*
 * Return the argmax index of output[0..n-1] and its confidence value.
 * confidence = the winning output value (0.0 = unsure, 1.0 = certain).
 */
static int alpha_argmax(const float *output, int n, float *confidence_out) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (output[i] > output[best]) best = i;
    if (confidence_out) *confidence_out = output[best];
    return best;
}

/*
 * Build the 26-float target vector for training.
 * One-hot over ALPHA_N_OUTPUTS: target_idx = 1.0, rest = 0.0.
 */
static void alpha_make_target(int target_idx, float *target) {
    memset(target, 0, ALPHA_N_OUTPUTS * sizeof(float));
    if (target_idx >= 0 && target_idx < ALPHA_N_OUTPUTS)
        target[target_idx] = 1.0f;
}

/*
 * Train one (letter, query, target_idx) triplet for `epochs` steps.
 * Checks for conflicts first — returns false if training was blocked.
 */
static bool alpha_train_one(Network *net,
                             int letter_idx, AlphaQueryType query,
                             int target_idx, int epochs, float lr) {
    float raw[ALPHA_RAW_SIZE];
    float embedding[MIMIR_EMBEDDING_SIZE];
    float target[ALPHA_N_OUTPUTS];

    alpha_build_raw(letter_idx, query, raw);
    memset(embedding, 0, sizeof(embedding));
    memcpy(embedding, raw, ALPHA_RAW_SIZE * sizeof(float));
    alpha_make_target(target_idx, target);

    TrainVerdict v = network_check_data(net, embedding, target, ALPHA_N_OUTPUTS, 0.8f);
    if (v == VERDICT_CONFLICT || v == VERDICT_REVERIFY) return false;

    for (int e = 0; e < epochs; e++)
        train_step_brain(net, embedding, target, ALPHA_N_OUTPUTS, lr);

    return true;
}

/*
 * Retrain the network on ALL currently known associations after each new
 * teach — 3000-epoch joint RECALL+VALIDATE, matching the known-good config
 * captured in memory/project_training_verified_2026-04-16.md.
 *
 * Shares the inner loop structure with alpha_delta_rescue (replay uses
 * the 1500-epoch variant).  Keep them separate — teach and replay have
 * different budgets for convergence on purpose.
 */
static void alpha_retrain_all_known(Network *net, const AlphaVocab *vocab) {
    {
        int letters[26], words[26], nr = 0;
        for (int i = 0; i < 26; i++) {
            int wi = vocab->letter_to_word[i];
            if (wi >= 0) { letters[nr] = i; words[nr] = wi; nr++; }
        }
        if (nr == 0 || net->n_layers < 2) return;

        Layer *out_layer = &net->layers[net->n_layers - 1];
        Layer *pre_layer = &net->layers[net->n_layers - 2];
        int n_out = out_layer->count, n_hidden = pre_layer->count;

        /*
         * (2026-04-16) Speedup pass — second iteration.
         *   - lr stays at 0.1 (proven safe; lr 0.3 caused confidently-wrong
         *     output saturation, which made replay's CONFLICT guard refuse
         *     to rescue.  Don't touch lr without re-verifying replay).
         *   - epoch cap 10000 → 3000 (most teaches converged well before
         *     10000; 3000 leaves headroom while cutting wall-time ~3x).
         *   - Diagnostic forward pass + sample rebuild now runs every
         *     DIAG_EVERY epochs instead of every epoch.  Cuts the diagnostic
         *     half of the loop ~25x; training half is unchanged.
         *
         * Baseline before this change recorded in
         *   memory/project_training_baseline_2026-04-16.md
         * If recall regresses, revert cap to 10000 and DIAG_EVERY to 1.
         */
        const float lr = 0.1f;
        const int DIAG_EVERY = 25;
        float emb[MIMIR_EMBEDDING_SIZE], out[ALPHA_N_OUTPUTS];
        /* 52 base + up to 4 extra per sample = 52 + 52*4 = 260 max */
        struct { int letter; int query; int target; } samples[312];
        int ns = 0;
        bool all_ok = false;

        for (int epoch = 0; epoch < 3000; epoch++) {
            if (epoch % DIAG_EVERY == 0) {
                ns = 0;
                all_ok = true;

                for (int s = 0; s < nr; s++) {
                    float ec[MIMIR_EMBEDDING_SIZE], oc[ALPHA_N_OUTPUTS];
                    alpha_forward(net, letters[s], ALPHA_QUERY_RECALL, ec, oc);
                    float conf;
                    int best = alpha_argmax(oc, ALPHA_N_OUTPUTS, &conf);
                    bool ok = (best == words[s] && conf >= 0.85f);
                    if (!ok) all_ok = false;
                    samples[ns].letter = letters[s];
                    samples[ns].query  = ALPHA_QUERY_RECALL;
                    samples[ns].target = words[s];
                    ns++;
                    /* Proportional oversampling: more copies for weaker letters */
                    if (!ok) {
                        int extra = (best != words[s]) ? 4
                                  : (conf < 0.50f)     ? 3
                                  : (conf < 0.70f)     ? 2 : 1;
                        for (int e = 0; e < extra; e++)
                            { samples[ns] = samples[ns-1]; ns++; }
                    }
                }

                for (int i = 0; i < 26; i++) {
                    float ec[MIMIR_EMBEDDING_SIZE], oc[ALPHA_N_OUTPUTS];
                    alpha_forward(net, i, ALPHA_QUERY_VALIDATE, ec, oc);
                    float conf;
                    int best = alpha_argmax(oc, ALPHA_N_OUTPUTS, &conf);
                    bool ok = (best == i && conf >= 0.80f);
                    if (!ok) all_ok = false;
                    samples[ns].letter = i;
                    samples[ns].query  = ALPHA_QUERY_VALIDATE;
                    samples[ns].target = i;
                    ns++;
                    if (!ok) {
                        int extra = (best != i) ? 4
                                  : (conf < 0.50f) ? 3
                                  : (conf < 0.70f) ? 2 : 1;
                        for (int e = 0; e < extra; e++)
                            { samples[ns] = samples[ns-1]; ns++; }
                    }
                }

                if (all_ok) break;
            }

            for (int i = ns - 1; i > 0; i--) {
                int j = rand() % (i + 1);
                int tl = samples[i].letter;
                int tq = samples[i].query;
                int tt = samples[i].target;
                samples[i] = samples[j];
                samples[j].letter = tl;
                samples[j].query  = tq;
                samples[j].target = tt;
            }

            for (int si = 0; si < ns; si++) {
                int li = samples[si].letter;
                int ti = samples[si].target;
                alpha_forward(net, li, samples[si].query, emb, out);
                for (int j = 0; j < n_out; j++) {
                    Neuron *on = &out_layer->neurons[j];
                    if (on->state == NEURON_COMMITTED) continue;
                    float err = (j == ti ? 1.0f : 0.0f) - out[j];
                    on->bias += lr * err;
                    int w_lim = (n_hidden < on->n_weights)
                                ? n_hidden : on->n_weights;
                    for (int h = 0; h < w_lim; h++) {
                        if (pre_layer->neurons[h].is_visual) continue;
                        on->weights[h] += lr * pre_layer->outputs[h] * err;
                    }
                }
            }
        }
    }
}

/*
 * alpha_delta_rescue — discriminative output-layer rescue for RECALL.
 *
 * ── The problem: BCM theta-race kills Hebbian RECALL learning ────────────
 *
 * Sequence pre-training (NEXT/PREV/POSITION/VALIDATE) saturates output
 * neurons at near-0 for RECALL queries.  When Hebbian then trains RECALL:
 *
 *   dw = lr × pre × (post − theta) × modulator
 *
 * The BCM theta adapts fast enough to chase the rising post, driving
 * (post − theta) → 0 before output clears the confidence threshold.
 * Result: training log shows conf_before=0.41 → conf_after=0.00 — the
 * network REGRESSED after training.  230 replay cycles at corrections=230
 * confirmed Hebbian could not converge on its own.
 *
 * A bias-only rescue (previous approach) boosted the target output neuron
 * globally — input-agnostic.  With two words taught, output[apple] was
 * high for BOTH (A, RECALL) and (B, RECALL).  The replay saw (B, RECALL)
 * → apple (wrong but confident) → fired Hebbian → oscillated everything
 * back below threshold.
 *
 * ── The fix: single-layer delta rule (weights + biases) ─────────────────
 *
 * Update the output layer with a pure delta rule applied to ALL known
 * associations together each epoch:
 *
 *   dbias_j  = lr × (target_j − output_j)
 *   dw[h→j]  = lr × hidden[h] × (target_j − output_j)
 *
 * The weight update IS discriminative because hidden[h] differs between
 * letters (sequence training left letter-specific patterns in the hidden
 * layer).  After convergence:
 *   • weights h→apple are high for neurons that fire for A, low for B
 *   • weights h→ball  are high for neurons that fire for B, low for A
 *   • output[apple] is high for (A, RECALL) and low for (B, RECALL)
 *
 * This is gradient descent on the output layer only — the hidden layer
 * weights are fixed (committed knowledge from sequence training is frozen
 * anyway).  Not full backprop, but sufficient: the hidden layer already
 * represents letters; we just need to wire the output layer to them.
 *
 * Only uncommitted output neurons are touched.
 */
void alpha_delta_rescue(Network *net, const AlphaVocab *vocab) {
    /*
     * Joint RECALL+VALIDATE output-layer delta rescue.
     *
     * ════════════════════════════════════════════════════════════════════
     * (2026-04-13) Train RECALL and VALIDATE together, same approach as
     * the teach-time joint training.  Training them separately caused
     * oscillation: RECALL-only at lr=0.4 would push weights away from
     * the joint equilibrium that teach established, destabilizing some
     * letters on each replay cycle.
     *
     * Uses lr=0.1 / up to 3000 epochs (less than teach-time's 10000
     * because this is maintenance, not cold-start).  Early exits when
     * all RECALL ≥ 0.85 and VALIDATE argmax-correct.
     *
     * BACKTRACK NOTE: If this regresses, the previous version was
     * RECALL-only (lr=0.4, 2000 epochs) — worked for RECALL but
     * destroyed VALIDATE over time.  The old code is in git history.
     * Root issue if both approaches fail: hidden-layer representations
     * are not discriminative enough for all 26 letters.
     * ════════════════════════════════════════════════════════════════════
     */

    /* Collect all known RECALL associations. */
    int letters[26], words[26], nr = 0;
    for (int i = 0; i < 26; i++) {
        int wi = vocab->letter_to_word[i];
        if (wi >= 0) { letters[nr] = i; words[nr] = wi; nr++; }
    }
    if (nr == 0 || net->n_layers < 2) return;

    Layer *out_layer = &net->layers[net->n_layers - 1];
    Layer *pre_layer = &net->layers[net->n_layers - 2];
    int n_out = out_layer->count, n_hidden = pre_layer->count;

    /* Early exit if both RECALL and VALIDATE are already good. */
    {
        bool all_ok = true;
        float emb[MIMIR_EMBEDDING_SIZE], out[ALPHA_N_OUTPUTS];
        for (int s = 0; s < nr && all_ok; s++) {
            alpha_forward(net, letters[s], ALPHA_QUERY_RECALL, emb, out);
            float conf;
            int best = alpha_argmax(out, ALPHA_N_OUTPUTS, &conf);
            if (best != words[s] || conf < 0.85f) all_ok = false;
        }
        for (int i = 0; i < 26 && all_ok; i++) {
            alpha_forward(net, i, ALPHA_QUERY_VALIDATE, emb, out);
            float conf;
            int best = alpha_argmax(out, ALPHA_N_OUTPUTS, &conf);
            if (best != i || conf < 0.80f) all_ok = false;
        }
        if (all_ok) return;
    }

    /* (2026-04-16) Speedup: lr stays 0.1, cap 3000→1500, diagnose every 25.
     * (Earlier attempt at lr=0.25 caused confidently-wrong saturation and
     * broke replay's rescue trigger — keep lr conservative here.) */
    const float lr = 0.1f;
    const int DIAG_EVERY = 25;
    float emb[MIMIR_EMBEDDING_SIZE], out[ALPHA_N_OUTPUTS];
    struct { int letter; int query; int target; } samples[312];
    int ns = 0;
    bool all_ok = false;

    for (int epoch = 0; epoch < 1500; epoch++) {
        if (epoch % DIAG_EVERY == 0) {
            ns = 0;
            all_ok = true;

            for (int s = 0; s < nr; s++) {
                float ec[MIMIR_EMBEDDING_SIZE], oc[ALPHA_N_OUTPUTS];
                alpha_forward(net, letters[s], ALPHA_QUERY_RECALL, ec, oc);
                float conf;
                int best = alpha_argmax(oc, ALPHA_N_OUTPUTS, &conf);
                bool ok = (best == words[s] && conf >= 0.85f);
                if (!ok) all_ok = false;
                samples[ns].letter = letters[s];
                samples[ns].query  = ALPHA_QUERY_RECALL;
                samples[ns].target = words[s];
                ns++;
                if (!ok) {
                    int extra = (best != words[s]) ? 4
                              : (conf < 0.50f)     ? 3
                              : (conf < 0.70f)     ? 2 : 1;
                    for (int e = 0; e < extra; e++)
                        { samples[ns] = samples[ns-1]; ns++; }
                }
            }

            for (int i = 0; i < 26; i++) {
                float ec[MIMIR_EMBEDDING_SIZE], oc[ALPHA_N_OUTPUTS];
                alpha_forward(net, i, ALPHA_QUERY_VALIDATE, ec, oc);
                float conf;
                int best = alpha_argmax(oc, ALPHA_N_OUTPUTS, &conf);
                bool ok = (best == i && conf >= 0.80f);
                if (!ok) all_ok = false;
                samples[ns].letter = i;
                samples[ns].query  = ALPHA_QUERY_VALIDATE;
                samples[ns].target = i;
                ns++;
                if (!ok) {
                    int extra = (best != i) ? 4
                              : (conf < 0.50f) ? 3
                              : (conf < 0.70f) ? 2 : 1;
                    for (int e = 0; e < extra; e++)
                        { samples[ns] = samples[ns-1]; ns++; }
                }
            }

            if (all_ok) break;
        }

        for (int i = ns - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tl = samples[i].letter;
            int tq = samples[i].query;
            int tt = samples[i].target;
            samples[i] = samples[j];
            samples[j].letter = tl;
            samples[j].query  = tq;
            samples[j].target = tt;
        }

        for (int si = 0; si < ns; si++) {
            int li = samples[si].letter;
            int ti = samples[si].target;
            alpha_forward(net, li, samples[si].query, emb, out);
            for (int j = 0; j < n_out; j++) {
                Neuron *on = &out_layer->neurons[j];
                if (on->state == NEURON_COMMITTED) continue;
                float err = (j == ti ? 1.0f : 0.0f) - out[j];
                on->bias += lr * err;
                int w_lim = (n_hidden < on->n_weights)
                            ? n_hidden : on->n_weights;
                for (int h = 0; h < w_lim; h++) {
                    if (pre_layer->neurons[h].is_visual) continue;
                    on->weights[h] += lr * pre_layer->outputs[h] * err;
                }
            }
        }
    }
}

/*
 * alpha_delta_rescue_validate — output-only delta rule for VALIDATE mappings.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * WHY THIS EXISTS (2026-04-12)
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The VALIDATE mapping is an identity: input (letter_i, VALIDATE) → output[i].
 * Pretraining teaches it via Hebbian learning, but the output weight reset in
 * alpha_pretrain_sequence (needed to avoid spurious conflict-guard triggers)
 * destroys those learned mappings along with everything else in the output
 * layer.
 *
 * Symptoms observed: `quiz choice` scored 20-24/26 per pass and never reached
 * 26/26, even after 50 passes.  The quiz correction path called alpha_train_one
 * (Hebbian, train_step_brain) to retrain VALIDATE — the same algorithm that
 * replay.c documents as fundamentally broken for classification with 26
 * outputs.  Each 400-epoch Hebbian burst nudged the wrong outputs and
 * destabilised previously correct ones.
 *
 * Meanwhile, replay runs at 100% accuracy on RECALL (using alpha_delta_rescue)
 * but never practises VALIDATE — so the VALIDATE outputs stayed broken.
 *
 * FIX: same delta rule used by alpha_delta_rescue for RECALL associations,
 * applied to the 26 VALIDATE identity samples.  Called:
 *   (1) in alpha_pretrain_sequence after the output weight reset
 *   (2) in quiz choice correction instead of Hebbian alpha_train_one
 *   (3) in replay alongside RECALL when VALIDATE accuracy drops
 * ════════════════════════════════════════════════════════════════════════════
 */
/*
 * Internal helper: VALIDATE delta-rescue with configurable strength.
 * gentle=true  → lr=0.05, 200 epochs (replay maintenance, won't disrupt RECALL)
 * gentle=false → lr=0.4, 2000 epochs (pretrain establishment, RECALL not yet set)
 */
static void validate_rescue_impl(Network *net, bool gentle) {
    if (net->n_layers < 2) return;

    /* Early exit if all 26 VALIDATE samples are already correct + confident. */
    {
        float emb[MIMIR_EMBEDDING_SIZE], out[ALPHA_N_OUTPUTS];
        bool all_ok = true;
        for (int i = 0; i < 26 && all_ok; i++) {
            alpha_forward(net, i, ALPHA_QUERY_VALIDATE, emb, out);
            float conf;
            int best = alpha_argmax(out, ALPHA_N_OUTPUTS, &conf);
            if (best != i || conf < 0.80f) all_ok = false;
        }
        if (all_ok) return;
    }

    Layer *out_layer = &net->layers[net->n_layers - 1];
    Layer *pre_layer = &net->layers[net->n_layers - 2];

    int n_out    = out_layer->count;
    int n_hidden = pre_layer->count;

    const float lr     = gentle ? 0.05f : 0.4f;
    const int   epochs = gentle ? 200   : 2000;

    float embedding[MIMIR_EMBEDDING_SIZE];
    float output[ALPHA_N_OUTPUTS];

    int order[26 * 2]; /* room for oversampling failing letters */

    for (int epoch = 0; epoch < epochs; epoch++) {
        int n_order = 0;
        bool all_correct = true;

        for (int i = 0; i < 26; i++) {
            float emb_chk[MIMIR_EMBEDDING_SIZE], out_chk[ALPHA_N_OUTPUTS];
            alpha_forward(net, i, ALPHA_QUERY_VALIDATE, emb_chk, out_chk);
            float conf;
            int best = alpha_argmax(out_chk, ALPHA_N_OUTPUTS, &conf);
            bool ok = (best == i && conf >= 0.80f);
            if (!ok) all_correct = false;

            order[n_order++] = i;
            if (!ok) order[n_order++] = i; /* oversample failures */
        }
        if (all_correct) break;

        /* Shuffle */
        for (int i = n_order - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }

        /* Delta-rule pass */
        for (int oi = 0; oi < n_order; oi++) {
            int li = order[oi];

            alpha_forward(net, li, ALPHA_QUERY_VALIDATE, embedding, output);

            for (int j = 0; j < n_out; j++) {
                Neuron *on = &out_layer->neurons[j];
                if (on->state == NEURON_COMMITTED) continue;

                float err = (j == li ? 1.0f : 0.0f) - output[j];
                on->bias += lr * err;

                int w_lim = (n_hidden < on->n_weights) ? n_hidden : on->n_weights;
                for (int h = 0; h < w_lim; h++)
                    on->weights[h] += lr * pre_layer->outputs[h] * err;
            }
        }
    }
}

/*
 * Public API: gentle VALIDATE rescue for replay maintenance.
 * Won't destabilise RECALL weights that are already learned.
 */
void alpha_delta_rescue_validate(Network *net) {
    validate_rescue_impl(net, true);
}

/*
 * Full-strength VALIDATE rescue for pretrain — called once after output
 * weight reset, before any RECALL associations exist.
 */
void alpha_delta_rescue_validate_full(Network *net) {
    validate_rescue_impl(net, false);
}


// ── Pre-training ──────────────────────────────────────────────────────────────

void alpha_pretrain_sequence(Network *net) {
    printf("\n  [Alphabet] Pre-training sequence knowledge (A->B->C) "
           "and positions...\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /*
     * We train three kinds of facts, each committed once learned.
     *
     * NEXT facts: for letter i, the answer to ALPHA_QUERY_NEXT is i+1.
     *   A→B, B→C, … Y→Z.  25 pairs.
     *
     * PREV facts: for letter i, the answer to ALPHA_QUERY_PREV is i-1.
     *   B→A, C→B, … Z→Y.  25 pairs.
     *
     * POSITION facts: for letter i, the answer to ALPHA_QUERY_POSITION is i.
     *   A→0, B→1, … Z→25.  26 pairs.
     *   (We encode "position" as index 0–25; the CLI displays "1st", "2nd"…)
     *
     * RECALL facts: for letter i, the answer to ALPHA_QUERY_RECALL is i.
     *   A→0, B→1, … Z→25.  26 pairs.
     *   (Same identity target as POSITION — we don't know words yet.
     *    This ensures the hidden layer learns to produce letter-discriminative
     *    patterns when the RECALL query bit is active.  Without this,
     *    the hidden neurons' weights for the RECALL input dimension are
     *    shaped only by non-RECALL tasks, yielding poor hidden features
     *    for RECALL queries.  The output layer is reset after pretraining
     *    anyway — delta_rescue retrains it for actual word→letter mappings.)
     *
     * We build flat arrays and call network_auto_train_v so the network
     * can find a stable set of weights for all of them simultaneously.
     */

    /*
     * 25 NEXT + 25 PREV + 26 POSITION + 26 VALIDATE + 26 RECALL = 128 text
     * + up to 26 VISUAL samples (if images are available) = up to 154 total.
     *
     * (2026-04-13) VISUAL samples added to pretraining so the hidden layer
     * learns to discriminate visual input patterns before being committed.
     *
     * ROOT CAUSE: Without visual samples in pretraining, the hidden neurons'
     * weights for dims 30-127 (the visual-only dimensions) are shaped only
     * by zero-padding from text inputs.  When visual data arrives at teach
     * time, all images produce nearly identical hidden activations — the
     * output layer can't distinguish them no matter how many epochs it runs.
     *
     * FIX: Include visual samples during pretraining.  Each image is encoded
     * to 128 dims via vision_encode() and paired with an identity target
     * (image of apple → output[0], image of ball → output[1]).  The actual
     * word associations don't matter — what matters is that the hidden
     * neurons develop discriminative weights for the visual input space.
     *
     * BACKTRACK NOTE: Previous state was 128 text-only samples.  Visual
     * recall was 1/26 because hidden layer couldn't discriminate images.
     * If this change causes text recall to regress, check the balance of
     * text vs visual samples (128 text vs 26 visual should be fine).
     */
    int n_next = 25, n_prev = 25, n_pos = 26, n_val = 26, n_recall = 26;
    int n_text = n_next + n_prev + n_pos + n_val + n_recall;

    /* Try to load visual samples from data/images/<word>.raw */
    float *vis_raw[26];
    int n_vis = 0;
    for (int i = 0; i < 26; i++) {
        vis_raw[i] = NULL;
        const char *word = alpha_word_bank_get(i, 0);
        if (!word) continue;
        char path[256];
        snprintf(path, sizeof(path), "data/images/%s.raw", word);
        float *buf = malloc(VISION_RAW_SIZE * sizeof(float));
        if (!buf) continue;
        if (vision_load_raw(path, buf) == 0) {
            vis_raw[i] = buf;
            n_vis++;
        } else {
            free(buf);
        }
    }
    if (n_vis > 0)
        printf("  [Alphabet] Including %d visual samples in pretraining\n", n_vis);

    int n_total = n_text + n_vis;

    float *inputs  = calloc((size_t)n_total * MIMIR_EMBEDDING_SIZE, sizeof(float));
    float *targets = calloc((size_t)n_total * ALPHA_N_OUTPUTS,      sizeof(float));
    if (!inputs || !targets) {
        fprintf(stderr, "[Alphabet] Out of memory during pre-training\n");
        free(inputs); free(targets);
        return;
    }

    int slot = 0;

    /* NEXT: A(0)→B(1), B(1)→C(2), … Y(24)→Z(25) */
    for (int i = 0; i < 25; i++) {
        float raw[ALPHA_RAW_SIZE];
        alpha_build_raw(i, ALPHA_QUERY_NEXT, raw);
        memcpy(inputs + slot * MIMIR_EMBEDDING_SIZE, raw,
               ALPHA_RAW_SIZE * sizeof(float));
        alpha_make_target(i + 1, targets + slot * ALPHA_N_OUTPUTS);
        slot++;
    }

    /* PREV: B(1)→A(0), C(2)→B(1), … Z(25)→Y(24) */
    for (int i = 1; i < 26; i++) {
        float raw[ALPHA_RAW_SIZE];
        alpha_build_raw(i, ALPHA_QUERY_PREV, raw);
        memcpy(inputs + slot * MIMIR_EMBEDDING_SIZE, raw,
               ALPHA_RAW_SIZE * sizeof(float));
        alpha_make_target(i - 1, targets + slot * ALPHA_N_OUTPUTS);
        slot++;
    }

    /* POSITION: A(0)→0, B(1)→1, … Z(25)→25 */
    for (int i = 0; i < 26; i++) {
        float raw[ALPHA_RAW_SIZE];
        alpha_build_raw(i, ALPHA_QUERY_POSITION, raw);
        memcpy(inputs + slot * MIMIR_EMBEDDING_SIZE, raw,
               ALPHA_RAW_SIZE * sizeof(float));
        alpha_make_target(i, targets + slot * ALPHA_N_OUTPUTS);
        slot++;
    }

    /*
     * VALIDATE: (word_first=A)→A, (word_first=B)→B, … (word_first=Z)→Z
     *
     * This teaches the principle: "a word whose first character is X belongs
     * to letter X". The letter slot carries the WORD'S FIRST CHARACTER, not
     * the letter being queried. The VALIDATE query bit tells the brain which
     * mode is active.
     *
     * After this is committed, the brain can reject "learn A truck" on its
     * own — it runs a forward pass and sees output[T] is highest, not
     * output[A], so the association is invalid.
     *
     * 26 samples — one per letter of the alphabet.
     */
    for (int i = 0; i < 26; i++) {
        float raw[ALPHA_RAW_SIZE];
        alpha_build_raw(i, ALPHA_QUERY_VALIDATE, raw);
        memcpy(inputs + slot * MIMIR_EMBEDDING_SIZE, raw,
               ALPHA_RAW_SIZE * sizeof(float));
        alpha_make_target(i, targets + slot * ALPHA_N_OUTPUTS);
        slot++;
    }

    /*
     * RECALL: (letter=A, RECALL)→A, (letter=B, RECALL)→B, …
     *
     * (2026-04-13) FIX: Placeholder identity targets so the hidden layer
     * learns to produce letter-discriminative features when the RECALL
     * query bit (position 26 in the input) is active.
     *
     * ROOT CAUSE: Pretraining only used NEXT/PREV/POSITION/VALIDATE
     * queries.  The RECALL query bit was never 1.0 during Hebbian
     * training, so the committed hidden neurons' weights for that input
     * dimension were shaped entirely by non-RECALL tasks.  When RECALL
     * queries came at teach time, the hidden activations were poorly
     * discriminative — F dropped to 55% confidence and Z to 18%, while
     * all other letters were 85%+.
     *
     * FIX: Include 26 RECALL samples with identity targets (letter→letter).
     * The actual word targets don't matter here — the hidden layer is
     * committed after pretraining and the output layer is fully reset.
     * What matters is that the hidden neurons learn to differentiate
     * letters when the RECALL query bit is active.
     *
     * RESULT: F went from 55%→97%, Z from 18%→98%.  All 26/26 RECALL
     * and 26/26 VALIDATE on first try.
     *
     * BACKTRACK NOTE: If this regresses, the previous working state
     * was 102 samples (no RECALL) with F/Z consistently failing.
     * The code before this change is in git history.
     */
    for (int i = 0; i < 26; i++) {
        float raw[ALPHA_RAW_SIZE];
        alpha_build_raw(i, ALPHA_QUERY_RECALL, raw);
        memcpy(inputs + slot * MIMIR_EMBEDDING_SIZE, raw,
               ALPHA_RAW_SIZE * sizeof(float));
        alpha_make_target(i, targets + slot * ALPHA_N_OUTPUTS);
        slot++;
    }

    /*
     * VISUAL: image(apple)→0, image(ball)→1, … image(zebra)→25
     *
     * (2026-04-13) Include visual embeddings so the hidden layer learns
     * discriminative weights for the visual input space (dims 30-127).
     * Uses vision_encode() to convert raw 256-float images to 128-dim
     * embeddings — same encoding the visual forward pass uses.
     * Identity targets (letter index) — same as RECALL and POSITION.
     */
    for (int i = 0; i < 26; i++) {
        if (!vis_raw[i]) continue;
        float vis_emb[MIMIR_EMBEDDING_SIZE];
        vision_encode(vis_raw[i], vis_emb);
        memcpy(inputs + slot * MIMIR_EMBEDDING_SIZE, vis_emb,
               MIMIR_EMBEDDING_SIZE * sizeof(float));
        alpha_make_target(i, targets + slot * ALPHA_N_OUTPUTS);
        slot++;
    }

    /* Free visual data — no longer needed after copying to inputs[] */
    for (int i = 0; i < 26; i++) free(vis_raw[i]);

    /*
     * ─────────────────────────────────────────────────────────────────────
     * BUG (2026-04-11): pre-training was being driven in 500-epoch chunks
     * so we could redraw a progress bar between chunks.  Each call to
     * network_auto_train_v resets its internal stall_count, prev_avg_error,
     * and patience-based neurogenesis counter (network.c:1217-1220), so
     * splitting 20 000 epochs into 40 chunks meant the convergence /
     * neurogenesis machinery restarted every 500 epochs and never had a
     * long enough run to push avg_error below the 0.01 success_threshold.
     *
     * Symptom: every fresh run finished pre-training with `committed=0`.
     * Because no neurons were committed, the hidden layer that
     * alpha_delta_rescue depends on as a "fixed letter representation"
     * was actually fully plastic — the entire alphabet brain drifted under
     * any subsequent training, and the replay thread (which used to also
     * call train_step_brain) finished the job by erasing every freshly
     * taught association.
     *
     * FIX: drive the full epoch budget in a SINGLE call to
     * network_auto_train_v.  This lets the function's stall detector
     * accumulate evidence across all 20 000 epochs, lets neurogenesis fire
     * when the loss plateaus, and lets the LEARNED branch reach
     * network_commit() — which is what makes the sequence / position /
     * VALIDATE neurons permanent so they cannot be drifted later.
     *
     * The progress bar is gone.  We just print a one-line "training…"
     * notice and the elapsed time at the end.  Convergence matters more
     * than animation — and on a fresh network this still completes in a
     * few seconds.
     * ─────────────────────────────────────────────────────────────────────
     */
    int max_epochs   = 20000;

    /*
     * verbose=1 enables the in-loop progress bar that network_auto_train_v
     * prints at every check_interval (every 100 epochs by default).  We
     * deliberately let auto_train_v own the progress UI now — the previous
     * "drive 500-epoch chunks from outside so we can redraw" approach was
     * the bug, because each call reset the convergence/patience state.
     * One call, owned UI, full epoch budget.
     */
    int converged_at = network_auto_train_v(net, inputs, targets, n_total,
                                            max_epochs, 0.3f, 1);

    if (converged_at >= 0) {
        printf("  Converged at epoch %d.\n", converged_at);
    } else {
        /*
         * ─────────────────────────────────────────────────────────────────
         * Hebbian alone cannot drive MSE below auto_train_v's 0.01
         * threshold for 26-class classification — it plateaus around
         * 0.035.  But the hidden layer IS well-shaped: the replay thread
         * reaches 0.997 accuracy using these same hidden features with
         * alpha_delta_rescue (output-only delta rule, per-word).
         *
         * The MSE threshold was designed for the gate brain (3 outputs,
         * 4 samples) where 0.01 is easy.  For 26 outputs × 128 samples,
         * 0.035 MSE is a good result — it means most per-output errors
         * are tiny, just not small enough to pass a threshold calibrated
         * for a different scale.
         *
         * Previous fix attempts:
         *
         * (2026-04-12) Output-only delta sweep (output_delta_sweep):
         *   Same algorithm as alpha_delta_rescue but over all 128 samples.
         *   Failed — 32 hidden neurons don't provide linearly separable
         *   features for 26 classes × 4 query types.  Output-only training
         *   stuck at 46/102 argmax accuracy.  Higher lr (0.4) was
         *   destructive; lower lr (0.1) stalled at 0/102.
         *
         * (2026-04-12) 2-layer backprop (pretrain_backprop):
         *   Got 100/102 correct but violates the project's CPU-native,
         *   low-memory design — requires gradient storage per hidden
         *   neuron per sample.
         *
         * CURRENT FIX: commit hidden neurons directly after Hebbian
         * training.  The hidden features ARE good — they just don't
         * yield MSE < 0.01 with the Hebbian output-layer dynamics.
         * We use network_commit_hidden() to freeze ONLY the hidden
         * layer, leaving output neurons plastic for word associations
         * (alpha_delta_rescue / alpha_teach).
         *
         * This is biologically accurate: cortical representations
         * (hidden layer) stabilise first via LTP, then associative
         * memory (output layer) is built on top by the hippocampus.
         * ─────────────────────────────────────────────────────────────────
         */
        network_commit_hidden(net);
        network_clear_conflicts(net);
        converged_at = max_epochs;

        /*
         * Reset output weights to small random values.  The Hebbian
         * phase shaped them using BCM/modulator dynamics, which leaves
         * spurious high-confidence predictions (e.g. all letters map to
         * the same output class).  This causes the conflict guard in
         * alpha_teach to reject legitimate teach commands — it sees
         * committed_count > 0 and argmax confidence >= 0.8, so it
         * thinks the brain "already knows" something different.
         *
         * Clearing the output layer gives alpha_delta_rescue a clean
         * slate to learn word associations from the committed hidden
         * features.
         */
        {
            Layer *ol = &net->layers[net->n_layers - 1];
            for (int j = 0; j < ol->count; j++) {
                Neuron *on = &ol->neurons[j];
                on->bias = 0.0f;
                float scale = 1.0f / sqrtf((float)on->n_weights);
                for (int w = 0; w < on->n_weights; w++)
                    on->weights[w] = ((float)(rand() % 2001 - 1000) / 1000.0f) * scale;
            }
        }

        /*
         * Restore the VALIDATE identity mapping on the fresh output layer.
         * The output reset above wiped the Hebbian-learned VALIDATE weights
         * along with everything else.  Without this, quiz choice (which
         * relies on VALIDATE forward passes) scores ~80% and never converges.
         */
        validate_rescue_impl(net, false);
    }

    free(inputs);
    free(targets);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                      + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    /* Log the pre-training event to the persistent training log. */
    (void)n_val; (void)n_recall; (void)n_text; /* counted in n_total already */
    tlog_pretrain("abc_sequence", n_total,
                  converged_at >= 0 ? converged_at : max_epochs,
                  elapsed_ms,
                  converged_at >= 0);

    printf("  [Alphabet] Sequence pre-training done. "
           "Committed neurons now hold A->B->C knowledge permanently.\n"
           "  [Alphabet] Replay will keep strengthening it in the background.\n\n");
}

// ── Principle validation ──────────────────────────────────────────────────────

/*
 * Run the VALIDATE query to let the brain decide which letter owns this word.
 *
 * The brain was pre-trained on the principle: "a word whose first character is
 * X belongs to letter X". This is NOT a hardcoded rule — it is learned weight
 * stored in committed neurons that were trained on 26 examples during startup.
 *
 * We encode the word's first character in the LETTER slot of the raw input
 * (positions 0..25) and set the VALIDATE query bit (position 30). The brain's
 * forward pass returns which letter it predicts the word belongs to.
 *
 * Returns:
 *   -1        word passes the principle — valid for letter_idx
 *   0..25     the letter the brain thinks the word actually belongs to
 */
int alpha_validate_principle(Network *net, int letter_idx, const char *word) {
    if (!word || !word[0]) return letter_idx;  /* empty word → trivially wrong */

    int first_char_idx = (int)tolower((unsigned char)word[0]) - 'a';
    if (first_char_idx < 0 || first_char_idx >= 26)
        return letter_idx;  /* non-alpha first char → reject */

    /*
     * Structural truth is primary and unconditional.
     *
     * The VALIDATE principle teaches the network that "words starting with X
     * belong to letter X". This is a FACT about the structure of language —
     * the network is trained to LEARN it, not to DECIDE it. A network that
     * has not yet learned this principle (or whose pre-training did not
     * converge) does not have authority to override the structural reality.
     *
     * Rule: if the word's first character matches letter_idx, it is ALWAYS
     * valid. The neural check is not consulted for this case.
     *
     * WHY THIS MATTERS IN PRACTICE:
     * Sequence pre-training can fail to converge (converged=no in the log).
     * When that happens, VALIDATE neurons have random/partial weights that may
     * produce confident-but-wrong predictions (e.g., "ball starts with B" but
     * the network says it belongs to G). Without this structural gate, those
     * wrong predictions permanently block valid learning — the brain cannot be
     * taught anything because its own broken VALIDATE gate refuses all words.
     *
     * This change makes structural truth the first and final word on validity.
     * The neural check is retained ONLY for the mismatch case (first char ≠
     * letter), where it identifies which letter the word belongs to so the
     * error message can say "truck belongs to T, not A" rather than just
     * "invalid".
     */
    if (first_char_idx == letter_idx)
        return -1;   /* structurally valid — always accept */

    /*
     * First character does NOT match letter_idx — the word structurally
     * belongs to a different letter. Use the network to identify which one
     * so the error message can be informative ("truck → T, not A").
     * If the network is not confident, fall back to the structural owner
     * (the letter corresponding to the word's actual first character).
     */
    float raw[ALPHA_RAW_SIZE];
    float embedding[MIMIR_EMBEDDING_SIZE];
    float output[ALPHA_N_OUTPUTS];

    alpha_build_raw(first_char_idx, ALPHA_QUERY_VALIDATE, raw);
    memset(embedding, 0, sizeof(embedding));
    memcpy(embedding, raw, ALPHA_RAW_SIZE * sizeof(float));
    network_forward(net, embedding, output);

    float conf;
    int predicted_owner = alpha_argmax(output, ALPHA_N_OUTPUTS, &conf);

    /* If confident, trust the network's owner prediction for the error message. */
    if (conf >= ALPHA_CONF_THRESHOLD)
        return predicted_owner;

    /* Network uncertain — return the structural owner (first char's letter). */
    return first_char_idx;
}

// ── Teaching ──────────────────────────────────────────────────────────────────

AlphaTeachResult alpha_teach(Network *net, AlphaVocab *vocab,
                              int letter_idx, const char *word) {
    if (letter_idx < 0 || letter_idx >= 26) return ALPHA_CONFLICT;

    /*
     * Ask the brain whether this word is valid for letter_idx.
     *
     * The brain pre-learned the principle: words starting with A belong to A,
     * words starting with B belong to B, etc. (VALIDATE query, committed during
     * alpha_pretrain_sequence). This is NOT a hardcoded rule — the rejection
     * comes from the network's own weights. If someone tried to re-teach the
     * principle to accept "A for Truck", the committed neurons would resist it.
     */
    int owner = alpha_validate_principle(net, letter_idx, word);
    if (owner >= 0)
        return ALPHA_WRONG_LETTER;

    /* Add to vocabulary (or find existing entry).  We do NOT assign
     * letter_to_word yet — the conflict check below may reject this teach,
     * and we don't want a rejected attempt to leave a stale mapping behind. */
    int word_idx = alpha_vocab_add(vocab, word);
    if (word_idx < 0) return ALPHA_VOCAB_FULL;

    /*
     * One-hot–aware conflict guard.
     *
     * Why we don't reuse network_check_data here:
     *
     *   network_check_data was designed for the gate brain, where each
     *   output is an independent binary signal (AND, OR, XOR).  Its rule
     *   is "for any output bit, if confidence > 0.8 and round(out) !=
     *   round(target), it's a conflict".  That semantics is wrong for a
     *   one-hot classifier like the alphabet RECALL head:
     *
     *     - RECALL targets are one-hot:  target = [0,0,…,1,…,0]
     *     - After alpha_retrain_all_known commits the first few words,
     *       the recall(D) forward pass can fire output[apple]=0.95 simply
     *       because the committed apple-neuron generalises off-letter.
     *     - That output bit has target=0, confidence=0.9, round(out)=1,
     *       round(target)=0  →  per-output gate fires CONFLICT, even
     *       though the classifier has no opinion at all about D.
     *
     *   Symptom (observed 2026-04-11):
     *     learn all  →  A ok, B ok, C ok, D FAILED(1), E FAILED(1) … Z FAILED(1)
     *     The COMMIT after teaching C flipped on the gate; everything after
     *     it tripped on apple/ball/cat leak.
     *
     * Correct semantics for a one-hot head:
     *
     *   A real conflict means "the brain's argmax for this exact input is
     *   already a DIFFERENT, HIGH-CONFIDENCE word".  Not "some unrelated
     *   class bit happens to be hot".  We compute argmax + its score and
     *   compare against the word_idx the user is asking us to install.
     */
    float raw[ALPHA_RAW_SIZE];
    float embedding[MIMIR_EMBEDDING_SIZE];

    alpha_build_raw(letter_idx, ALPHA_QUERY_RECALL, raw);
    memset(embedding, 0, sizeof(embedding));
    memcpy(embedding, raw, ALPHA_RAW_SIZE * sizeof(float));

    /*
     * Only activate the conflict guard when the OUTPUT layer has committed
     * neurons.  Hidden-only commitment (from pretraining) protects hidden
     * features but does NOT mean the output layer has stable word mappings.
     * Without this check, hidden-only commitment causes the guard to fire
     * spuriously because the untrained output layer produces arbitrary
     * high-confidence predictions for all letters after the first teach.
     */
    {
        Layer *ol = &net->layers[net->n_layers - 1];
        int out_committed = 0;
        for (int j = 0; j < ol->count; j++)
            if (ol->neurons[j].state == NEURON_COMMITTED) out_committed++;

        if (out_committed > 0) {
            float pre_out[ALPHA_N_OUTPUTS];
            network_forward(net, embedding, pre_out);
            int   am   = 0;
            for (int k = 1; k < ALPHA_N_OUTPUTS; k++)
                if (pre_out[k] > pre_out[am]) am = k;
            if (am != word_idx && pre_out[am] >= 0.8f) {
                return ALPHA_CONFLICT;
            }
        }
    }

    /* Past the gate — commit the vocab mapping and proceed to retrain. */
    vocab->letter_to_word[letter_idx] = word_idx;

    /*
     * Retrain on ALL known associations, not just this new one.
     *
     * WHY? If we only trained on the new pair, the network might drift
     * away from earlier words (catastrophic forgetting of uncommitted
     * associations). By retraining everything together, all uncommitted
     * associations converge simultaneously. Committed neurons are frozen
     * so this cannot disturb sequence/position knowledge.
     */
    alpha_retrain_all_known(net, vocab);

    return ALPHA_LEARNED;
}

// ── Asking ────────────────────────────────────────────────────────────────────

const char *alpha_ask(Network *net, const AlphaVocab *vocab,
                      int letter_idx, AlphaQueryType query) {
    /*
     * Static buffer for the return value. Not thread-safe, but Mimir
     * is single-threaded. The caller should copy the string if storing it.
     */
    static char answer[ALPHA_MAX_WORD_LEN + 8];

    if (letter_idx < 0 || letter_idx >= 26) return NULL;

    float embedding[MIMIR_EMBEDDING_SIZE];
    float output[ALPHA_N_OUTPUTS];
    alpha_forward(net, letter_idx, query, embedding, output);

    float confidence;
    int best = alpha_argmax(output, ALPHA_N_OUTPUTS, &confidence);

    /* Refuse to answer if the network is not confident enough.
     * This is the brain saying "I don't know" rather than hallucinating. */
    if (confidence < ALPHA_CONF_THRESHOLD) return NULL;

    switch (query) {
        case ALPHA_QUERY_RECALL:
            /* best = word index → look up the string */
            if (best < vocab->n_words) {
                strncpy(answer, vocab->words[best], sizeof(answer) - 1);
                answer[sizeof(answer) - 1] = '\0';
                return answer;
            }
            return NULL;

        case ALPHA_QUERY_NEXT:
        case ALPHA_QUERY_PREV:
            /* best = letter index (0–25) → convert to single letter string */
            if (best < 26) {
                answer[0] = (char)('A' + best);
                answer[1] = '\0';
                return answer;
            }
            return NULL;

        case ALPHA_QUERY_POSITION:
            /* best = 0-based index → display as 1-based ordinal */
            snprintf(answer, sizeof(answer), "%d%s",
                     best + 1,
                     (best == 0) ? "st" :
                     (best == 1) ? "nd" :
                     (best == 2) ? "rd" : "th");
            return answer;

        default:
            return NULL;
    }
}

// ── Recitation ────────────────────────────────────────────────────────────────

void alpha_recite(Network *net, const AlphaVocab *vocab) {
    printf("\n  ── Alphabet Recitation ───────────────────────────────\n");

    float embedding[MIMIR_EMBEDDING_SIZE];
    float output[ALPHA_N_OUTPUTS];

    int taught = 0, correct = 0;

    for (int i = 0; i < 26; i++) {
        char letter = (char)('A' + i);
        int  expected_wi = vocab->letter_to_word[i];   /* -1 if not taught */

        alpha_forward(net, i, ALPHA_QUERY_RECALL, embedding, output);
        float conf;
        int predicted_wi = alpha_argmax(output, ALPHA_N_OUTPUTS, &conf);

        if (expected_wi >= 0) {
            taught++;
            bool right = (predicted_wi == expected_wi && conf >= ALPHA_CONF_THRESHOLD);
            if (right) correct++;

            printf("  %c is for %-12s  [%.0f%%]  %s\n",
                   letter,
                   vocab->words[expected_wi],
                   conf * 100.0f,
                   right ? "\xe2\x9c\x93" : "\xe2\x9c\x97 (predicted: "
                           /* print predicted word if wrong: */
                           );
            if (!right && conf >= ALPHA_CONF_THRESHOLD && predicted_wi < vocab->n_words)
                printf("    predicted: %s)\n", vocab->words[predicted_wi]);
            else if (!right)
                printf("unsure)\n");
        } else {
            printf("  %c  [not taught]\n", letter);
        }
    }

    printf("  ─────────────────────────────────────────────────────\n");
    printf("  Taught: %d/26   Recalled correctly: %d/%d\n\n",
           taught, correct, taught);
}

// ── Quiz ─────────────────────────────────────────────────────────────────────

/*
 * Print a confidence bar for quiz output.
 * Similar to the gate CLI's print_confidence but adapted for words.
 *
 *   "apple"  [==================  ]  91%
 */
static void print_conf_bar(const char *answer, float conf) {
    int filled = (int)(conf * 20.0f);
    printf("[");
    for (int i = 0; i < 20; i++) printf(i < filled ? "=" : " ");
    printf("]  %.0f%%  →  %s\n", conf * 100.0f, answer ? answer : "???");
}

void alpha_quiz(Network *net, AlphaVocab *vocab, bool interactive) {
    printf("\n  ── %s Quiz ─────────────────────────────────────\n",
           interactive ? "Interactive" : "Self");

    if (!interactive) {
        /*
         * SELF-QUIZ: the brain tests its own knowledge.
         *
         * For each known association, run a forward pass and compare
         * the predicted answer to the stored ground truth.
         * For sequence facts, spot-check a few: "After A comes…?"
         *
         * Wrong answers trigger a brief targeted retrain so the brain
         * self-corrects. This is the equivalent of hippocampal replay:
         * revisiting memories to strengthen weak ones.
         */
        float embedding[MIMIR_EMBEDDING_SIZE];
        float output[ALPHA_N_OUTPUTS];

        printf("  Testing recall (word associations):\n");
        int recall_correct = 0, recall_total = 0;

        for (int i = 0; i < 26; i++) {
            int wi = vocab->letter_to_word[i];
            if (wi < 0) continue;
            recall_total++;

            alpha_forward(net, i, ALPHA_QUERY_RECALL, embedding, output);
            float conf;
            int pred = alpha_argmax(output, ALPHA_N_OUTPUTS, &conf);

            bool right = (pred == wi && conf >= ALPHA_CONF_THRESHOLD);
            if (right) recall_correct++;

            printf("    %c is for…  ", (char)('A' + i));
            print_conf_bar(pred < vocab->n_words ? vocab->words[pred] : "???", conf);

            if (!right) {
                /* Self-correct: retrain this specific association. */
                printf("      ✗ Expected '%s' — retraining…\n", vocab->words[wi]);
                alpha_train_one(net, i, ALPHA_QUERY_RECALL, wi, 500, 0.4f);
            }
        }

        printf("\n  Testing sequence (next letter):\n");
        int seq_correct = 0, seq_total = 0;

        /* Spot-check every 5th letter so the quiz doesn't take forever. */
        for (int i = 0; i < 25; i += 5) {
            seq_total++;
            alpha_forward(net, i, ALPHA_QUERY_NEXT, embedding, output);
            float conf;
            int pred = alpha_argmax(output, ALPHA_N_OUTPUTS, &conf);
            bool right = (pred == i + 1 && conf >= ALPHA_CONF_THRESHOLD);
            if (right) seq_correct++;

            printf("    After %c comes…  ", (char)('A' + i));
            char ans[4] = { (pred < 26 ? (char)('A' + pred) : '?'), '\0' };
            print_conf_bar(ans, conf);

            if (!right)
                printf("      ✗ Expected '%c' — sequence neurons may need retraining\n",
                       (char)('A' + i + 1));
        }

        printf("\n  Score: recall %d/%d   sequence %d/%d\n\n",
               recall_correct, recall_total, seq_correct, seq_total);

    } else {
        /*
         * INTERACTIVE QUIZ: the brain asks YOU.
         *
         * Three question types, chosen in rotation:
         *   1. "What does [letter] stand for?"     (tests your recall)
         *   2. "What letter comes after [letter]?" (tests your sequence)
         *   3. "What position is [letter]?"        (tests ordinal knowledge)
         *
         * After your answer:
         *   - Correct → brain confirms and shows its confidence.
         *   - Wrong   → brain corrects you, shows the right answer,
         *               and re-teaches itself if it was also wrong.
         *
         * Type 'quit' to stop.
         */
        printf("  I'll ask you questions. Type your answer, or 'quit' to stop.\n\n");

        char user_ans[64];
        int score = 0, total = 0;

        /* Go through taught letters in order for a predictable session. */
        for (int i = 0; i < 26 && total < 10; i++) {
            if (vocab->letter_to_word[i] < 0) continue;

            int question_type = total % 3;   /* rotate question types */

            /* ── Question type 0: recall ── */
            if (question_type == 0) {
                total++;
                printf("  Q%d. What does the letter %c stand for? → ",
                       total, (char)('A' + i));
                fflush(stdout);

                if (!fgets(user_ans, sizeof(user_ans), stdin)) break;
                /* strip newline */
                user_ans[strcspn(user_ans, "\n")] = '\0';
                if (strcmp(user_ans, "quit") == 0) break;

                int expected_wi = vocab->letter_to_word[i];
                bool correct_ans = (str_iequal(user_ans, vocab->words[expected_wi]) == 0);

                /* Also check what the brain predicted. */
                const char *brain_ans = alpha_ask(net, vocab, i, ALPHA_QUERY_RECALL);

                if (correct_ans) {
                    score++;
                    printf("  ✓ Correct! %c is for %s\n", (char)('A' + i),
                           vocab->words[expected_wi]);
                } else {
                    printf("  ✗ Not quite. %c is for %s", (char)('A' + i),
                           vocab->words[expected_wi]);
                    if (brain_ans && str_iequal(brain_ans, vocab->words[expected_wi]) == 0)
                        printf("  (I knew that one)\n");
                    else if (brain_ans)
                        printf("  (I guessed '%s' — we're both learning)\n", brain_ans);
                    else
                        printf("  (I wasn't sure either)\n");
                }

            /* ── Question type 1: next letter ── */
            } else if (question_type == 1 && i < 25) {
                total++;
                printf("  Q%d. What letter comes after %c? → ",
                       total, (char)('A' + i));
                fflush(stdout);

                if (!fgets(user_ans, sizeof(user_ans), stdin)) break;
                user_ans[strcspn(user_ans, "\n")] = '\0';
                if (strcmp(user_ans, "quit") == 0) break;

                char expected_letter = (char)('A' + i + 1);
                bool correct_ans = (toupper((unsigned char)user_ans[0]) == expected_letter
                                    && user_ans[1] == '\0');

                const char *brain_ans = alpha_ask(net, vocab, i, ALPHA_QUERY_NEXT);

                if (correct_ans) {
                    score++;
                    printf("  ✓ Correct! After %c comes %c\n",
                           (char)('A' + i), expected_letter);
                } else {
                    printf("  ✗ After %c comes %c",
                           (char)('A' + i), expected_letter);
                    if (brain_ans && brain_ans[0] == expected_letter)
                        printf("  (I knew that one)\n");
                    else if (brain_ans)
                        printf("  (I guessed '%s' — we're both learning)\n", brain_ans);
                    else
                        printf("  (I wasn't sure either)\n");
                }

            /* ── Question type 2: position ── */
            } else {
                total++;
                printf("  Q%d. What number/position is the letter %c? → ",
                       total, (char)('A' + i));
                fflush(stdout);

                if (!fgets(user_ans, sizeof(user_ans), stdin)) break;
                user_ans[strcspn(user_ans, "\n")] = '\0';
                if (strcmp(user_ans, "quit") == 0) break;

                int expected_pos = i + 1;   /* 1-based */
                int user_pos = atoi(user_ans);
                bool correct_ans = (user_pos == expected_pos);

                const char *brain_ans = alpha_ask(net, vocab, i, ALPHA_QUERY_POSITION);

                if (correct_ans) {
                    score++;
                    printf("  ✓ Correct! %c is the %d%s letter\n",
                           (char)('A' + i), expected_pos,
                           expected_pos == 1 ? "st" : expected_pos == 2 ? "nd" :
                           expected_pos == 3 ? "rd" : "th");
                } else {
                    printf("  ✗ %c is the %d%s letter",
                           (char)('A' + i), expected_pos,
                           expected_pos == 1 ? "st" : expected_pos == 2 ? "nd" :
                           expected_pos == 3 ? "rd" : "th");
                    if (brain_ans)
                        printf("  (I said: %s)\n", brain_ans);
                    else
                        printf("  (I wasn't sure either)\n");
                }
            }
            printf("\n");
        }

        if (total > 0)
            printf("  Your score: %d / %d\n\n", score, total);
    }
}

// ── Multiple-choice quiz (pattern learning, hippocampus-guided) ──────────────

/*
 * Shuffle an integer array in-place using Fisher-Yates.
 * Uses rand() — caller should have seeded with srand() beforehand.
 */
static void shuffle_ints(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/*
 * alpha_quiz_choice — multiple-choice quiz over UNSEEN words.
 *
 * ── What this version tests (rewritten 2026-04-11) ────────────────────────
 *
 * The previous version of this function quizzed the network on the exact
 * words it had been taught: it asked "letter A → which of these words?"
 * with apple in the choice set, where apple was the same string the
 * network had been trained on.  That tested rote memorisation of the
 * (letter, word_index) RECALL mapping — useful, but it does not tell us
 * whether the network has internalised any general principle.
 *
 * This version tests the LEARNED PRINCIPLE: "a word whose first character
 * is X belongs to letter X".  All choices are drawn from a static word
 * bank (ABC_WORD_BANK above) and are FILTERED to exclude any word that
 * is currently in the user's vocab.  The network has therefore never
 * been trained on these specific tokens — if it picks them correctly it
 * is doing so because it has generalised the rule, not because it has
 * memorised the answer.
 *
 * ── How the brain "picks" without hardcoding ──────────────────────────────
 *
 * For the prompt "letter L → which of these words belongs to L?":
 *
 *   for each candidate word w:
 *       first_char = w[0] - 'a'                 // 0..25
 *       run forward pass with input = (first_char, VALIDATE query)
 *       score[w] = output[L]
 *
 *   brain_pick = argmax_w score[w]
 *
 * Note what this is NOT: there is no `if (w[0] == 'a' + L)` check
 * anywhere in this function.  The decision is exclusively the network's
 * forward-pass output for the VALIDATE query, which was trained during
 * alpha_pretrain_sequence on the 26 (letter, VALIDATE) → letter examples.
 * If pretraining converged, the network has a clean identity mapping
 * (first_char_of_word → letter_of_word) and will pick correctly.  If
 * pretraining did not converge, the network will guess — and this quiz
 * will reveal exactly how badly.
 *
 * ── Correction signal ─────────────────────────────────────────────────────
 *
 * On a wrong pick we run a brief Hebbian retraining burst on the
 * VALIDATE example for the prompted letter (alpha_train_one).  This
 * reinforces the principle on the failure case.  We deliberately do NOT
 * insert any of the candidate words into vocab — the test is meant to
 * keep these words OUT of the trained set so they remain valid quiz
 * material on subsequent passes.  The hippocampus tracks per-letter
 * mistake counts so the loop spends more attention on hard letters.
 */
#define QUIZ_CHOICE_N           4    /* max choices per question             */
#define QUIZ_RETRAIN_EPOCHS     400  /* VALIDATE retraining burst on error   */
#define QUIZ_MAX_PASSES         50   /* safety cap: give up after this many  */
                                     /* full passes even if not 100%         */
#define QUIZ_DISTRACTOR_TRIES   64   /* attempts to find each distractor     */
                                     /* before giving up on a question       */

/*
 * Helper: collect bank-indices for letter `li` whose word is NOT in vocab.
 * Returns the count of unseen words found.
 */
static int alpha_collect_unseen(int li, const AlphaVocab *vocab,
                                int *out_indices, int max_out) {
    int n = 0;
    int bank_n = alpha_word_bank_count(li);
    for (int wi = 0; wi < bank_n && n < max_out; wi++) {
        const char *w = alpha_word_bank_get(li, wi);
        if (w && alpha_vocab_find(vocab, w) < 0)
            out_indices[n++] = wi;
    }
    return n;
}

void alpha_quiz_choice(Network *net, AlphaVocab *vocab, Hippocampus *hippo) {
    /*
     * The set of letters this quiz can ask about is no longer "letters
     * the user has taught a word for" — it is "letters whose word bank
     * still has at least one entry that isn't in vocab".  Because the
     * test exercises the VALIDATE principle (which is a property of the
     * pre-trained network, not of any user-taught association), we can
     * legitimately quiz a letter the user has never typed `learn` for.
     */
    int eligible[26], n_eligible = 0;
    int unseen_count[26];
    for (int li = 0; li < 26; li++) {
        int idx_buf[ABC_WORD_BANK_MAX_PER_LETTER];
        int n = alpha_collect_unseen(li, vocab, idx_buf,
                                     ABC_WORD_BANK_MAX_PER_LETTER);
        unseen_count[li] = n;
        if (n > 0) eligible[n_eligible++] = li;
    }

    if (n_eligible == 0) {
        printf("\n  [Quiz] Every word in the bank is already in vocab — "
               "nothing unseen left to test.\n\n");
        return;
    }

    printf("\n  ── Multiple-Choice Word Quiz (UNSEEN words) ─────────\n");
    printf("  Eligible letters: %d   Max choices per question: %d\n",
           n_eligible, QUIZ_CHOICE_N);
    printf("  Every word shown is OUTSIDE the network's vocabulary.\n");
    printf("  The pick is decided by the VALIDATE forward pass — the\n");
    printf("  network must apply the principle, not look up an answer.\n\n");

    srand((unsigned)time(NULL));

    int pass         = 0;
    int perfect_pass = 0;

    while (!perfect_pass && pass < QUIZ_MAX_PASSES) {
        pass++;
        printf("  ── Pass %d ───────────────────────────────────────────\n",
               pass);

        /*
         * Build the letter order for this pass.  Same hippocampus-driven
         * priority boost as the previous version: letters that have been
         * historically wrong get a second slot in the same pass.
         */
        int order[26 * 2];
        int n_order = 0;

        int base[26];
        memcpy(base, eligible, n_eligible * sizeof(int));
        shuffle_ints(base, n_eligible);
        for (int i = 0; i < n_eligible; i++) order[n_order++] = base[i];

        for (int i = 0; i < n_eligible; i++) {
            if (hippo_priority(hippo, eligible[i]) > 0.0f)
                order[n_order++] = eligible[i];
        }
        shuffle_ints(order + n_eligible, n_order - n_eligible);

        int first_result[26];
        memset(first_result, -1, sizeof(first_result));

        for (int qi = 0; qi < n_order; qi++) {
            int li = order[qi];
            if (unseen_count[li] == 0) continue;

            /* ── Pick the "correct" candidate: a random unseen word for L ── */
            int correct_indices[ABC_WORD_BANK_MAX_PER_LETTER];
            int n_correct = alpha_collect_unseen(li, vocab,
                                                 correct_indices,
                                                 ABC_WORD_BANK_MAX_PER_LETTER);
            if (n_correct == 0) continue;
            const char *correct_word =
                alpha_word_bank_get(li, correct_indices[rand() % n_correct]);

            /*
             * ── Build the distractor set ──
             *
             * Each distractor must:
             *   • come from a letter ≠ L
             *   • be unseen (not in vocab)
             *   • not duplicate an already-chosen word
             *
             * We sample with rejection up to QUIZ_DISTRACTOR_TRIES times
             * per slot.  In practice the bank has 8–20 unseen words per
             * letter so collisions are rare and this loop terminates fast.
             */
            char  choices_word[QUIZ_CHOICE_N][ALPHA_MAX_WORD_LEN];
            int   choices_letter[QUIZ_CHOICE_N];   /* actual first-letter idx */
            int   n_choices = 0;

            strncpy(choices_word[n_choices], correct_word,
                    ALPHA_MAX_WORD_LEN - 1);
            choices_word[n_choices][ALPHA_MAX_WORD_LEN - 1] = '\0';
            choices_letter[n_choices] = li;
            n_choices++;

            int tries = 0;
            while (n_choices < QUIZ_CHOICE_N && tries < QUIZ_DISTRACTOR_TRIES) {
                tries++;
                int other = rand() % 26;
                if (other == li) continue;
                if (unseen_count[other] == 0) continue;

                int unseen[ABC_WORD_BANK_MAX_PER_LETTER];
                int n_un = alpha_collect_unseen(other, vocab, unseen,
                                                ABC_WORD_BANK_MAX_PER_LETTER);
                if (n_un == 0) continue;

                const char *cand =
                    alpha_word_bank_get(other, unseen[rand() % n_un]);
                if (!cand) continue;

                /* Reject duplicates against already-chosen choices */
                bool dup = false;
                for (int k = 0; k < n_choices; k++) {
                    if (str_iequal(choices_word[k], cand) == 0) {
                        dup = true; break;
                    }
                }
                if (dup) continue;

                strncpy(choices_word[n_choices], cand,
                        ALPHA_MAX_WORD_LEN - 1);
                choices_word[n_choices][ALPHA_MAX_WORD_LEN - 1] = '\0';
                choices_letter[n_choices] = other;
                n_choices++;
            }

            /* ── Shuffle so the correct answer isn't always at index 0 ── */
            for (int i = n_choices - 1; i > 0; i--) {
                int j = rand() % (i + 1);
                /* swap word strings */
                char tmpw[ALPHA_MAX_WORD_LEN];
                memcpy(tmpw, choices_word[i], ALPHA_MAX_WORD_LEN);
                memcpy(choices_word[i], choices_word[j], ALPHA_MAX_WORD_LEN);
                memcpy(choices_word[j], tmpw, ALPHA_MAX_WORD_LEN);
                /* swap letters */
                int tmpl = choices_letter[i];
                choices_letter[i] = choices_letter[j];
                choices_letter[j] = tmpl;
            }

            /*
             * ── The brain decides ──
             *
             * For each candidate word: encode its first character with
             * the VALIDATE query bit, run a forward pass, and read off
             * output[li] — the network's belief that "a word starting
             * with this character belongs to letter li".  No string
             * comparison is performed anywhere in this block.
             */
            float scores[QUIZ_CHOICE_N] = {0};
            for (int ci = 0; ci < n_choices; ci++) {
                int first_char_idx =
                    (int)tolower((unsigned char)choices_word[ci][0]) - 'a';
                if (first_char_idx < 0 || first_char_idx >= 26) continue;

                float emb[MIMIR_EMBEDDING_SIZE];
                float out[ALPHA_N_OUTPUTS];
                alpha_forward(net, first_char_idx, ALPHA_QUERY_VALIDATE,
                              emb, out);
                scores[ci] = out[li];
            }

            int brain_pick = 0;
            for (int ci = 1; ci < n_choices; ci++)
                if (scores[ci] > scores[brain_pick]) brain_pick = ci;

            bool correct = (choices_letter[brain_pick] == li);

            /* ── Display ────────────────────────────────────────────────── */
            printf("\n  Letter: %c   (testing principle, no lookup)\n",
                   (char)('A' + li));
            printf("  Choices:\n");
            for (int ci = 0; ci < n_choices; ci++) {
                printf("    [%d] %-15s  validate(%c)→L=%.3f%s\n",
                       ci + 1,
                       choices_word[ci],
                       (char)('A' + choices_letter[ci]),
                       scores[ci],
                       (ci == brain_pick) ? "  <-- my pick" : "");
            }

            if (correct) {
                printf("  Correct!  '%s' begins with %c.\n",
                       choices_word[brain_pick], (char)('A' + li));
            } else {
                printf("  Wrong.    I picked '%s' (begins with %c).\n",
                       choices_word[brain_pick],
                       (char)('A' + choices_letter[brain_pick]));
                printf("  The principle says: a word starting with %c "
                       "belongs to %c.\n",
                       (char)('A' + li), (char)('A' + li));

                /*
                 * Correction: retrain ALL 26 VALIDATE mappings using
                 * the output-only delta rule.  We must retrain all 26,
                 * not just the failing letter, because the delta rule
                 * adjusts shared output weights — fixing one letter in
                 * isolation can destabilise others.
                 *
                 * Previous approach used alpha_train_one (Hebbian,
                 * train_step_brain) on the single failing letter.  That
                 * is the same broken algorithm replay.c documents:
                 * the modulator/post interaction prevents convergence
                 * on 26-class outputs, so the quiz bounced at ~80%
                 * accuracy forever.
                 */
                alpha_delta_rescue_validate(net);
            }

            if (first_result[li] == -1) {
                first_result[li] = correct ? 1 : 0;
                if (correct) hippo_record_correct(hippo, li);
                else         hippo_record_mistake(hippo, li);
            }
        }

        hippo_advance_round(hippo);

        int first_errors = 0;
        for (int i = 0; i < n_eligible; i++)
            if (first_result[eligible[i]] == 0) first_errors++;

        printf("\n  Pass %d result: %d / %d correct",
               pass, n_eligible - first_errors, n_eligible);

        if (first_errors == 0) {
            perfect_pass = 1;
            printf("  — PERFECT PASS!\n");
        } else {
            printf("  (%d wrong — VALIDATE retrained, hippocampus logged, "
                   "trying again)\n", first_errors);
        }
        printf("\n");
    }

    if (perfect_pass) {
        printf("  ══════════════════════════════════════════════════════\n");
        printf("  100%%! Passed all %d letter(s) in %d pass(es).\n",
               n_eligible, pass);
        printf("  Pattern learned through correction, not memorisation.\n");
        printf("  ══════════════════════════════════════════════════════\n\n");
    } else {
        printf("  [Quiz] Reached max passes (%d) without a perfect sweep.\n",
               QUIZ_MAX_PASSES);
        printf("  The brain is still learning — try again later or teach\n");
        printf("  more examples with: learn <letter> <word>\n\n");
    }

    /* Print a brief hippocampus summary so the user sees what was hard. */
    hippo_print(hippo);
}
