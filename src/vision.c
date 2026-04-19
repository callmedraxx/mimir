/*
 * vision.c — Visual sensor for the MIMIR alphabet brain.
 *
 * (2026-04-13) First implementation: cross-modal letter-word associations.
 *
 * GOAL: Ground letter-word associations in visual features so the brain
 * can recognize objects by their appearance, not just their text symbol.
 * "A is for apple" gains meaning when the brain also knows what an apple
 * looks like.
 *
 * ARCHITECTURE:
 *   [16x16 grayscale image] → 256 floats → visual_encode() → 128-dim embedding
 *                                                                    ↓
 *                                              Shared hidden layer (committed)
 *                                                                    ↓
 *                                              Output layer → word class (0-25)
 *
 * The visual embedding occupies the FULL 128 dimensions (not just unused
 * slots).  This means a visual forward pass uses a completely different
 * input pattern from a text forward pass.  The committed hidden neurons
 * respond to both — their weights span all 128 input dimensions.  The
 * output layer (plastic) learns to map both text-activated and visually-
 * activated hidden patterns to the same word output.
 *
 * ENCODING: Simple average pooling — divide the 256 pixels into 128 pairs
 * and average each pair.  This is intentionally primitive.  A learned
 * encoder would be better but requires training infrastructure we don't
 * have yet.  The average-pool encoding still preserves spatial structure
 * (adjacent pixels are pooled, not random ones).
 *
 * BIOLOGICAL PARALLEL:
 * This is the ventral visual stream ("what pathway") — V1 → V2 → V4 →
 * inferotemporal cortex → hippocampus.  The encoding step is V1-V4
 * (feature extraction), the shared hidden layer is inferotemporal cortex
 * (object representation), and the output is the hippocampal association
 * (object → name).
 */

#include "mimir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* VISION_IMG_SIZE, VISION_RAW_SIZE, VISION_GABOR_ORIENTATIONS, VISION_POOL_GRID
 * are defined in mimir.h so the CLI and other callers see the same constants. */

/*
 * Load a raw float image from disk.
 * Returns 0 on success, -1 on failure.
 * out must point to VISION_RAW_SIZE floats.
 */
int vision_load_raw(const char *path, float *out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[Vision] Cannot open image: %s\n", path);
        return -1;
    }
    size_t n = fread(out, sizeof(float), VISION_RAW_SIZE, f);
    fclose(f);
    if ((int)n != VISION_RAW_SIZE) {
        fprintf(stderr, "[Vision] Expected %d floats, got %zu: %s\n",
                VISION_RAW_SIZE, n, path);
        return -1;
    }
    return 0;
}

/*
 * V1-analogue Gabor front-end.
 *
 * Pipeline (per image):
 *   1. Retina: input is already a grayscale 128×128 float image in [0,1].
 *   2. Local contrast normalization — subtract local mean, divide by local
 *      std.  Makes the features robust to overall lighting differences,
 *      important for webcam/screen input where exposure varies.
 *   3. V1: convolve with a bank of 8 oriented Gabor filters.  Each filter
 *      responds to edges at one orientation.  Take the absolute value of
 *      the response (like a simple-cell → complex-cell pool, phase-invariant).
 *   4. Spatial pooling: average each 4×4 region of each feature map,
 *      yielding 8 orientations × 4×4 grid = 128 features.
 *
 * Biological mapping: step 2 ≈ retinal ganglion / LGN centre-surround,
 * step 3 ≈ V1 simple+complex cells, step 4 ≈ V2/V4 coarser spatial codes.
 *
 * The goal is generalization: two photos of an apple differing in colour,
 * lighting, or position should produce similar 128-dim features because the
 * Gabor bank encodes edge orientation and coarse position, not raw pixels.
 */

#define VISION_GABOR_KSIZE 11          /* 11×11 filter kernel */
#define VISION_GABOR_RADIUS (VISION_GABOR_KSIZE / 2)
#define VISION_LCN_RADIUS  3           /* local contrast normalization window radius */

static float gabor_kernels[VISION_GABOR_ORIENTATIONS][VISION_GABOR_KSIZE * VISION_GABOR_KSIZE];
static int   gabor_init_done = 0;

static void vision_init_gabor(void) {
    if (gabor_init_done) return;
    const float sigma  = 2.5f;   /* Gaussian envelope */
    const float lambda = 5.5f;   /* wavelength of sinusoid */
    const float gamma  = 0.5f;   /* aspect ratio */
    for (int o = 0; o < VISION_GABOR_ORIENTATIONS; o++) {
        float theta = (float)o * 3.14159265f / (float)VISION_GABOR_ORIENTATIONS;
        float ct = cosf(theta), st = sinf(theta);
        float sum_sq = 0.0f;
        for (int y = -VISION_GABOR_RADIUS; y <= VISION_GABOR_RADIUS; y++) {
            for (int x = -VISION_GABOR_RADIUS; x <= VISION_GABOR_RADIUS; x++) {
                float xp =  x * ct + y * st;
                float yp = -x * st + y * ct;
                float env = expf(-(xp * xp + gamma * gamma * yp * yp) / (2.0f * sigma * sigma));
                float carrier = cosf(2.0f * 3.14159265f * xp / lambda);
                float v = env * carrier;
                int idx = (y + VISION_GABOR_RADIUS) * VISION_GABOR_KSIZE + (x + VISION_GABOR_RADIUS);
                gabor_kernels[o][idx] = v;
                sum_sq += v * v;
            }
        }
        /* L2-normalize so all orientations have equal response scale */
        float norm = 1.0f / sqrtf(sum_sq + 1e-8f);
        for (int i = 0; i < VISION_GABOR_KSIZE * VISION_GABOR_KSIZE; i++) {
            gabor_kernels[o][i] *= norm;
        }
    }
    gabor_init_done = 1;
}

/* Local contrast normalization: for each pixel, subtract the mean over
 * a (2r+1)×(2r+1) window and divide by local std.  Zero-padded at borders. */
static void vision_lcn(const float *in, float *out) {
    const int N = VISION_IMG_SIZE;
    const int r = VISION_LCN_RADIUS;
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            float sum = 0.0f, sum_sq = 0.0f;
            int count = 0;
            for (int dy = -r; dy <= r; dy++) {
                int yy = y + dy;
                if (yy < 0 || yy >= N) continue;
                for (int dx = -r; dx <= r; dx++) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= N) continue;
                    float v = in[yy * N + xx];
                    sum += v;
                    sum_sq += v * v;
                    count++;
                }
            }
            float mean = sum / (float)count;
            float var  = sum_sq / (float)count - mean * mean;
            float std  = sqrtf(var + 1e-3f);
            out[y * N + x] = (in[y * N + x] - mean) / std;
        }
    }
}

/* Convolve `in` with kernel `k` (VISION_GABOR_KSIZE square), write into `out`.
 * Valid region only for centre positions; borders zero-padded. */
static void vision_conv(const float *in, const float *k, float *out) {
    const int N = VISION_IMG_SIZE;
    const int R = VISION_GABOR_RADIUS;
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            float acc = 0.0f;
            for (int ky = -R; ky <= R; ky++) {
                int yy = y + ky;
                if (yy < 0 || yy >= N) continue;
                for (int kx = -R; kx <= R; kx++) {
                    int xx = x + kx;
                    if (xx < 0 || xx >= N) continue;
                    float kv = k[(ky + R) * VISION_GABOR_KSIZE + (kx + R)];
                    acc += in[yy * N + xx] * kv;
                }
            }
            out[y * N + x] = acc;
        }
    }
}

/*
 * Encode a VISION_RAW_SIZE grayscale image into a 128-dim feature vector.
 *
 *   raw  — pointer to VISION_IMG_SIZE * VISION_IMG_SIZE floats in [0,1]
 *   embedding — pointer to MIMIR_EMBEDDING_SIZE floats (= VISION_FEATURE_SIZE)
 *
 * Pipeline: LCN → 8 Gabor convolutions → |abs| → 4×4 average pool.
 * Deterministic; no randomness, no learning.  Same code path used for
 * training photos, webcam frames, and screen captures.
 */
void vision_encode(const float *raw, float *embedding) {
    vision_init_gabor();

    /* Stack-too-small for 128×128 on some systems, so use static scratch.
     * vision_encode is called serially inside a mutex (replay->lock) when
     * training, so static storage is safe. */
    static float normed[VISION_RAW_SIZE];
    static float resp[VISION_RAW_SIZE];
    vision_lcn(raw, normed);

    memset(embedding, 0, MIMIR_EMBEDDING_SIZE * sizeof(float));
    const int N = VISION_IMG_SIZE;
    const int G = VISION_POOL_GRID;
    const int cell = N / G;  /* 32 pixels per pool cell for 128/4 */

    for (int o = 0; o < VISION_GABOR_ORIENTATIONS; o++) {
        vision_conv(normed, gabor_kernels[o], resp);
        /* Pool |resp| into 4×4 grid, write into embedding */
        for (int gy = 0; gy < G; gy++) {
            for (int gx = 0; gx < G; gx++) {
                float sum = 0.0f;
                int count = 0;
                int y0 = gy * cell, y1 = y0 + cell;
                int x0 = gx * cell, x1 = x0 + cell;
                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {
                        sum += fabsf(resp[y * N + x]);
                        count++;
                    }
                }
                int feat_idx = o * (G * G) + gy * G + gx;
                embedding[feat_idx] = sum / (float)count;
            }
        }
    }

    /* Per-image z-score: remove the image-wide mean and divide by std.
     * Without this, mean-pooled Gabor magnitudes differ mostly by overall
     * edge energy (brightness/contrast), leaving encoded cosine similarity
     * ~0.80 across distinct images. After z-scoring, mean pairwise cosine
     * drops to ~0.13 — images become discriminable by the hidden layer.
     * Preserves shape information while discarding absolute scale. */
    float mean = 0.0f;
    for (int i = 0; i < MIMIR_EMBEDDING_SIZE; i++) mean += embedding[i];
    mean /= (float)MIMIR_EMBEDDING_SIZE;
    float var = 0.0f;
    for (int i = 0; i < MIMIR_EMBEDDING_SIZE; i++) {
        float d = embedding[i] - mean;
        var += d * d;
    }
    float inv_std = 1.0f / (sqrtf(var / (float)MIMIR_EMBEDDING_SIZE) + 1e-9f);
    for (int i = 0; i < MIMIR_EMBEDDING_SIZE; i++)
        embedding[i] = (embedding[i] - mean) * inv_std;
}

/*
 * Visual forward pass: load image, encode, run through the unified
 * HDC hidden layer.  No modality masking, no text-region zero-out —
 * the hidden layer produces a distinct binary signature for any
 * input vector, and the output layer has been trained to map both
 * text and visual signatures to their respective classes.
 */
void vision_forward(Network *net, const float *raw_image,
                    float *embedding, float *output) {
    vision_encode(raw_image, embedding);
    network_forward(net, embedding, output);
}

/*
 * (2026-04-18) HDC training — single shared random-projection hidden layer.
 *
 * Previous design grew modality-specific hidden neurons (is_visual flag)
 * and maintained a dual-bias on the output layer to keep text and vision
 * from disturbing each other.  Math analysis (sandbox/vision_math.py)
 * showed 8 visual neurons were not a linearly-separable code for 26
 * classes (7.7% strict argmax).  HDC replaces the modality-specific
 * scheme with a fixed random projection (256 hidden neurons, Gaussian
 * weights, step activation) that produces distinct binary signatures for
 * any input — text or visual — and delivers 100% strict on both
 * modalities with a single output-layer delta rule.
 *
 * No modality separation, no neurogenesis, no dual-bias.  Everything the
 * brain needs for cross-modal recognition lives in:
 *   (a) the fixed HDC projection (network_hdc_init_hidden, one-shot)
 *   (b) the output-layer delta rule (vision_train / alpha_retrain_all_known)
 *
 * img_data: array of 26 float pointers, img_data[i] points to
 *           VISION_RAW_SIZE floats for letter i, or NULL if no image.
 */
void vision_train(Network *net, const AlphaVocab *vocab,
                  float *img_data[26]) {
    if (net->n_layers < 2) return;

    /* HDC: no modality separation needed — hidden layer is a fixed random
     * projection that produces distinct binary signatures for text and
     * visual inputs alike.  Output layer delta rule learns the mapping. */

    Layer *out_layer = &net->layers[net->n_layers - 1];
    Layer *hidden    = &net->layers[0];
    int n_out    = out_layer->count;
    int n_hidden = hidden->count;

    /* Collect which letters have both a word and an image */
    int vis_letters[26], vis_words[26];
    float *vis_imgs[26];
    int n_vis = 0;
    for (int i = 0; i < 26; i++) {
        int wi = vocab->letter_to_word[i];
        if (wi >= 0 && img_data[i] != NULL) {
            vis_letters[n_vis] = i;
            vis_words[n_vis] = wi;
            vis_imgs[n_vis] = img_data[i];
            n_vis++;
        }
    }
    if (n_vis == 0) return;

    /* Collect text RECALL letters */
    int txt_letters[26], txt_words[26], n_txt = 0;
    for (int i = 0; i < 26; i++) {
        int wi = vocab->letter_to_word[i];
        if (wi >= 0) {
            txt_letters[n_txt] = i;
            txt_words[n_txt] = wi;
            n_txt++;
        }
    }

    const float lr = 0.1f;
    float emb[MIMIR_EMBEDDING_SIZE], out[ALPHA_N_OUTPUTS];

    /* Cache Gabor encodings — deterministic, ~150M ops each.
     * HDC: the hidden layer is a fixed random projection with ACT_STEP;
     * text inputs fire one subset of hidden neurons, visual inputs fire
     * a different subset.  No text-region zero-out is needed — the signs
     * of the projection separate modalities automatically. */
    static float cached_vis_emb[26][MIMIR_EMBEDDING_SIZE];
    for (int s = 0; s < n_vis; s++)
        vision_encode(vis_imgs[s], cached_vis_emb[s]);

    enum { STYPE_RECALL, STYPE_VALIDATE, STYPE_VISUAL };
    struct {
        int type;
        int letter;
        int target;
        int vis_idx;
    } samples[500];

    int best_vis_correct = -1, stale_epochs = 0;
    const int MAX_EPOCHS     = 3000;
    const int PLATEAU_EPOCHS = 300;
    const int DIAG_EVERY     = 25;

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        int ns = 0;
        int n_vis_correct = 0;

        /* --- Visual samples (the only thing we train) --- */
        for (int s = 0; s < n_vis; s++) {
            float ve[MIMIR_EMBEDDING_SIZE], vo[ALPHA_N_OUTPUTS];
            memcpy(ve, cached_vis_emb[s], MIMIR_EMBEDDING_SIZE * sizeof(float));
            network_forward(net, ve, vo);
            int best = -1; float best_val = -1.0f;
            for (int j = 0; j < ALPHA_N_OUTPUTS; j++)
                if (vo[j] > best_val) { best_val = vo[j]; best = j; }
            bool ok = (best == vis_words[s] && best_val >= 0.80f);
            if (ok) n_vis_correct++;

            samples[ns].type = STYPE_VISUAL;
            samples[ns].letter = vis_letters[s];
            samples[ns].target = vis_words[s];
            samples[ns].vis_idx = s;
            ns++;
            if (!ok) {
                int extra = (best != vis_words[s]) ? 4
                          : (best_val < 0.50f)     ? 3
                          : (best_val < 0.70f)     ? 2 : 1;
                for (int e = 0; e < extra; e++)
                    { samples[ns] = samples[ns-1]; ns++; }
            }
        }

        /* Diagnostic: check text health (log-only, no training).
         * With modality split, visual training can't affect text weights,
         * so text should remain as alpha_teach left it.  This check
         * verifies that assumption. */
        int n_txt_correct = 0;
        if (epoch % DIAG_EVERY == 0) {
            for (int s = 0; s < n_txt; s++) {
                float ec[MIMIR_EMBEDDING_SIZE], oc[ALPHA_N_OUTPUTS];
                alpha_forward(net, txt_letters[s], ALPHA_QUERY_RECALL, ec, oc);
                int best = -1; float best_val = -1.0f;
                for (int j = 0; j < ALPHA_N_OUTPUTS; j++)
                    if (oc[j] > best_val) { best_val = oc[j]; best = j; }
                if (best == txt_words[s] && best_val >= 0.55f)
                    n_txt_correct++;
            }
            printf("  [Vision] epoch %5d  text %d/%d  visual %d/%d\n",
                   epoch, n_txt_correct, n_txt, n_vis_correct, n_vis);
            fflush(stdout);
        }

        /* Convergence: all visual correct */
        if (n_vis_correct == n_vis) {
            printf("  [Vision] Converged at epoch %d (visual %d/%d)\n",
                   epoch, n_vis_correct, n_vis);
            return;
        }

        /* Plateau detection on visual only */
        if (n_vis_correct > best_vis_correct) {
            best_vis_correct = n_vis_correct;
            stale_epochs = 0;
        } else {
            stale_epochs++;
        }
        if (stale_epochs >= PLATEAU_EPOCHS) {
            printf("  [Vision] Plateau at epoch %d (visual %d/%d)\n",
                   epoch, n_vis_correct, n_vis);
            return;
        }

        /* Shuffle visual samples */
        for (int i = ns - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tt = samples[i].type, tl = samples[i].letter;
            int tg = samples[i].target, tv = samples[i].vis_idx;
            samples[i] = samples[j];
            samples[j].type = tt; samples[j].letter = tl;
            samples[j].target = tg; samples[j].vis_idx = tv;
        }

        /* --- Training loop: output + visual-hidden weight updates --- */
        for (int si = 0; si < ns; si++) {
            int ti = samples[si].target;

            /* Forward pass (HDC: unified hidden, no modality routing) */
            if (samples[si].type == STYPE_VISUAL) {
                memcpy(emb, cached_vis_emb[samples[si].vis_idx],
                       MIMIR_EMBEDDING_SIZE * sizeof(float));
                network_forward(net, emb, out);
            } else {
                int qtype = (samples[si].type == STYPE_RECALL)
                            ? ALPHA_QUERY_RECALL : ALPHA_QUERY_VALIDATE;
                alpha_forward(net, samples[si].letter, qtype, emb, out);
            }

            /* --- HDC output-layer delta rule ---
             *
             * The hidden layer is a fixed random projection with step
             * activation (see network_hdc_init_hidden).  Different input
             * vectors produce distinct binary hidden signatures, and the
             * Johnson–Lindenstrauss / Cover-hyperplane argument gives us
             * linear separability on the expanded hidden code.  So the
             * output layer just learns a single-layer delta rule over
             * ALL hidden neurons — no is_visual gating, no dual-bias. */
            for (int j = 0; j < n_out; j++) {
                Neuron *on = &out_layer->neurons[j];
                if (on->state == NEURON_COMMITTED) continue;
                float err = (j == ti ? 1.0f : 0.0f) - out[j];
                on->bias += lr * err;
                int w_lim = (n_hidden < on->n_weights)
                            ? n_hidden : on->n_weights;
                for (int h = 0; h < w_lim; h++)
                    on->weights[h] += lr * hidden->outputs[h] * err;
            }
        }
    }
    printf("  [Vision] Did not fully converge after %d epochs "
           "(%d visual samples)\n", MAX_EPOCHS, n_vis);
}

/*
 * vision_rescue — lightweight output-only visual weight repair.
 *
 * Called periodically by the replay thread to counteract bias drift.
 * When alpha_delta_rescue adjusts output biases for text recall, it
 * shifts the operating point of output neurons, degrading vision
 * predictions that were tuned to the old biases.
 *
 * Unlike full vision_train, this function:
 *   - Does NOT touch hidden-layer weights (visual neurons are stable)
 *   - Updates visual_bias and visual-side output weights only
 *     (dual-bias split since 2026-04-18 means text and vision no longer
 *      share the same bias, so rescue can now move it freely)
 *   - Caps at 500 epochs, early-exits when all visual correct at 80%
 *   - Returns the number of visual predictions currently correct
 *
 * This is the visual analog of alpha_delta_rescue: a focused output-layer
 * delta rule that maintains associations without disrupting the other
 * modality.
 */
int vision_rescue(Network *net, const AlphaVocab *vocab,
                  float *img_data[26]) {
    if (net->n_layers < 2) return 0;

    Layer *out_layer = &net->layers[net->n_layers - 1];
    Layer *hidden    = &net->layers[0];
    int n_out    = out_layer->count;
    int n_hidden = hidden->count;

    /* Collect visual samples */
    int vis_words[26];
    float *vis_imgs[26];
    int n_vis = 0;
    for (int i = 0; i < 26; i++) {
        int wi = vocab->letter_to_word[i];
        if (wi >= 0 && img_data[i] != NULL) {
            vis_words[n_vis] = wi;
            vis_imgs[n_vis] = img_data[i];
            n_vis++;
        }
    }
    if (n_vis == 0) return 0;

    /* Cache Gabor embeddings (HDC: no text-region zero-out needed) */
    float cached_emb[26][MIMIR_EMBEDDING_SIZE];
    for (int s = 0; s < n_vis; s++)
        vision_encode(vis_imgs[s], cached_emb[s]);

    const float lr = 0.1f;
    const int MAX_EPOCHS = 500;
    float emb[MIMIR_EMBEDDING_SIZE], out[ALPHA_N_OUTPUTS];

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        /* Check accuracy */
        int n_correct = 0;
        for (int s = 0; s < n_vis; s++) {
            memcpy(emb, cached_emb[s], sizeof(emb));
            network_forward(net, emb, out);
            int best = -1; float best_val = -1.0f;
            for (int j = 0; j < ALPHA_N_OUTPUTS; j++)
                if (out[j] > best_val) { best_val = out[j]; best = j; }
            if (best == vis_words[s] && best_val >= 0.80f)
                n_correct++;
        }
        if (n_correct == n_vis) return n_correct;

        /* HDC: single-layer delta rule over all hidden neurons */
        for (int s = 0; s < n_vis; s++) {
            memcpy(emb, cached_emb[s], sizeof(emb));
            network_forward(net, emb, out);
            int ti = vis_words[s];
            for (int j = 0; j < n_out; j++) {
                Neuron *on = &out_layer->neurons[j];
                if (on->state == NEURON_COMMITTED) continue;
                float err = (j == ti ? 1.0f : 0.0f) - out[j];
                on->bias += lr * err;
                int w_lim = (n_hidden < on->n_weights)
                            ? n_hidden : on->n_weights;
                for (int h = 0; h < w_lim; h++)
                    on->weights[h] += lr * hidden->outputs[h] * err;
            }
        }
    }

    /* Return final accuracy even if didn't fully converge */
    int n_correct = 0;
    for (int s = 0; s < n_vis; s++) {
        memcpy(emb, cached_emb[s], sizeof(emb));
        network_forward(net, emb, out);
        int best = -1; float best_val = -1.0f;
        for (int j = 0; j < ALPHA_N_OUTPUTS; j++)
            if (out[j] > best_val) { best_val = out[j]; best = j; }
        if (best == vis_words[s] && best_val >= 0.80f)
            n_correct++;
    }
    return n_correct;
}

/*
 * Test visual recall: for each letter with an image, run visual forward
 * and check if the network predicts the correct word.
 */
void vision_test(Network *net, const AlphaVocab *vocab,
                 float *img_data[26]) {
    int tested = 0, correct = 0;
    float emb[MIMIR_EMBEDDING_SIZE], out[ALPHA_N_OUTPUTS];

    printf("  -- VISUAL RECALL ----------------------------------\n");
    for (int i = 0; i < 26; i++) {
        int wi = vocab->letter_to_word[i];
        if (wi < 0 || img_data[i] == NULL) continue;

        vision_forward(net, img_data[i], emb, out);

        float conf = -1.0f;
        int best = 0;
        for (int j = 0; j < ALPHA_N_OUTPUTS; j++) {
            if (out[j] > conf) { conf = out[j]; best = j; }
        }

        tested++;
        bool right = (best == wi && conf >= 0.55f);
        if (right) correct++;

        if (right) {
            printf("  [IMG] %c -> %-12s  [%.0f%%]  \xe2\x9c\x93\n",
                   'A' + i, vocab->words[wi], conf * 100.0f);
        } else {
            const char *pred = (best >= 0 && best < ALPHA_VOCAB_SIZE
                                && vocab->words[best][0])
                               ? vocab->words[best] : "???";
            printf("  [IMG] %c -> %-12s  [%.0f%%]  \xe2\x9c\x97 (predicted: %s)\n",
                   'A' + i, vocab->words[wi], conf * 100.0f, pred);
        }
    }
    printf("  Visual recall: %d/%d correct\n", correct, tested);
}

/*
 * Load all 26 word images from data/images/<word>.raw.
 * Returns number of images loaded.
 * img_data[i] is allocated if letter i has a word with an image, else NULL.
 * Caller must free non-NULL entries.
 */
int vision_load_all(const AlphaVocab *vocab, float *img_data[26]) {
    int loaded = 0;
    for (int i = 0; i < 26; i++) {
        img_data[i] = NULL;
        int wi = vocab->letter_to_word[i];
        if (wi < 0) continue;

        char path[256];
        snprintf(path, sizeof(path), "data/images/%s.raw", vocab->words[wi]);

        float *buf = malloc(VISION_RAW_SIZE * sizeof(float));
        if (!buf) continue;

        if (vision_load_raw(path, buf) == 0) {
            img_data[i] = buf;
            loaded++;
        } else {
            free(buf);
        }
    }
    return loaded;
}

/*
 * Free all allocated image buffers.
 */
void vision_free_all(float *img_data[26]) {
    for (int i = 0; i < 26; i++) {
        free(img_data[i]);
        img_data[i] = NULL;
    }
}