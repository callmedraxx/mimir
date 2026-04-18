/*
 * MIMIR — Replay Engine (Background Self-Training)
 *
 * The brain that never stops practising.
 *
 * ════════════════════════════════════════════════════════════════════
 * WHAT THIS FILE DOES
 * ════════════════════════════════════════════════════════════════════
 *
 * A single background thread runs in an infinite loop. Each iteration:
 *
 *   1. Pick the next taught letter (cycles A → Z → A → …)
 *   2. Run a forward pass: "What do I think this letter stands for?"
 *   3. Compare prediction to the stored ground truth.
 *   4. If WRONG or UNCERTAIN: run 20 Hebbian training steps on that pair.
 *   5. Sleep 50 ms so the CPU isn't maxed, then repeat.
 *
 * This is exactly what children do when they recite the alphabet over
 * and over. Each repetition strengthens the correct associations and
 * weakens incorrect competing ones. After enough replays, the neurons
 * holding that association reach the committed threshold and become
 * permanent — the child no longer has to think about it.
 *
 * ════════════════════════════════════════════════════════════════════
 * CONCURRENCY
 * ════════════════════════════════════════════════════════════════════
 *
 * The replay thread and the CLI thread both access the Network and
 * AlphaVocab. Without coordination, one could corrupt the other's
 * weight update mid-stride.
 *
 * We use a single mutex (ReplayState.lock):
 *   - Replay thread: lock → forward pass → maybe train → unlock → sleep
 *   - CLI thread:    lock → teach / alpha_ask → unlock
 *
 * The sleep happens OUTSIDE the lock so the CLI is never blocked
 * waiting for the replay thread to finish sleeping.
 *
 * ════════════════════════════════════════════════════════════════════
 * AUTO-SAVE
 * ════════════════════════════════════════════════════════════════════
 *
 * Every REPLAY_SAVE_EVERY corrections the replay thread saves both the
 * network checkpoint and the vocab file. This means progress is never
 * lost if the process is killed — the brain picks up exactly where
 * it left off on the next run.
 */

#include "mimir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * usleep() is POSIX, not C11. GCC's -std=c11 hides it from <unistd.h>
 * even when _POSIX_C_SOURCE is set on the command line, because the
 * __STRICT_ANSI__ macro (set by -std=c11) causes glibc feature guards
 * to override our POSIX request. Rather than fight the macro system,
 * we declare the prototype directly — the symbol exists in glibc on
 * every Linux system Mimir targets, so this resolves correctly at link
 * time. This is the same technique used by many embedded and systems
 * projects that need POSIX utilities under strict-standard builds.
 */
extern int usleep(unsigned int usec);

/* How long to sleep between each letter-test (microseconds).
 * 50 000 µs = 50 ms = ~20 tests per second.
 * Raise to slow the brain down; lower to speed it up.
 * Below ~5 ms you will start saturating the mutex with lock contention. */
#define REPLAY_SLEEP_US       50000

/*
 * (Removed 2026-04-11) REPLAY_RETRAIN_STEPS used to gate a Hebbian retrain
 * burst here.  The Hebbian path is fundamentally wrong for RECALL queries —
 * see the long BUG/FIX comment in replay_thread_fn.  Replay now reuses
 * alpha_delta_rescue, which has its own internal early-exit and epoch cap.
 */

/* Save checkpoint every N corrections made by the replay thread.
 * Lower = safer against crashes but more I/O. 10 is a good balance. */
#define REPLAY_SAVE_EVERY     10

/*
 * Heartbeat: emit a tlog_replay line every N cycles regardless of whether
 * the brain needed correcting.  Before the 2026-04-11 fix, replay was
 * "correcting" almost every cycle so REPLAY events flooded the log; now
 * a healthy replay corrects rarely and would otherwise leave the log
 * silent.  We still want a periodic vital-signs entry so a human reading
 * mimir_training.log can see "yes the brain is alive and the answers are
 * still right at cycle N".  1000 cycles ≈ 50 seconds at 20 Hz.
 */
#define REPLAY_LOG_EVERY      1000

/* Confidence below which a "correct" argmax prediction still triggers
 * a retraining pass. Even if the brain got the right answer it should
 * express it confidently — a 56% correct answer is not good enough. */
#define REPLAY_CONF_MIN       0.85f

/* ── Private helpers ───────────────────────────────────────────────────── */

/* Argmax over output[0..n-1]. Returns the winning index.
 * Writes the winning value into *conf if conf is non-NULL. */
static int argmax_f(const float *output, int n, float *conf) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (output[i] > output[best]) best = i;
    if (conf) *conf = output[best];
    return best;
}

/* ── The background thread ─────────────────────────────────────────────── */

static void *replay_thread_fn(void *arg) {
    ReplayState *r = (ReplayState *)arg;

    float embedding[MIMIR_EMBEDDING_SIZE];
    float output[ALPHA_N_OUTPUTS];

    int   letter     = 0;   /* Next letter index to test (cycles 0–25) */
    int   save_countdown = REPLAY_SAVE_EVERY;
    unsigned long last_heartbeat_cycle = 0;

    while (r->running) {

        /* ── Lock: we own the network until unlock ── */
        pthread_mutex_lock(&r->lock);

        /*
         * Find the next letter that has been taught.
         * If no letters are taught yet, fall through to unlock+sleep.
         * We advance `letter` by one after each test so we cycle evenly
         * through the alphabet rather than always starting from A.
         */
        int found = -1;
        for (int tries = 0; tries < 26; tries++) {
            int candidate = (letter + tries) % 26;
            if (r->vocab->letter_to_word[candidate] >= 0) {
                found    = candidate;
                letter   = (candidate + 1) % 26;  /* next time, start one past */
                break;
            }
        }

        if (found < 0) {
            /* Nothing taught yet — brain has nothing to practise. */
            pthread_mutex_unlock(&r->lock);
            usleep(REPLAY_SLEEP_US * 10);   /* wait longer between checks */
            continue;
        }

        int li          = found;
        int expected_wi = r->vocab->letter_to_word[li];

        /* ── Forward pass: "What do I think?" ──
         *
         * (2026-04-18) Use alpha_forward (not raw network_forward) so the
         * replay check sees the SAME masked output that alpha_ask and
         * alpha_delta_rescue see.  Before this, replay used unmasked
         * network_forward which was polluted by visual-hidden-neuron
         * noise after vision training — every text query looked "wrong"
         * to replay even when alpha_ask answered correctly, so replay
         * hammered delta_rescue forever at 0 % reported accuracy. */
        alpha_forward(r->abc_net, li, ALPHA_QUERY_RECALL, embedding, output);

        float conf;
        int predicted_wi = argmax_f(output, ALPHA_N_OUTPUTS, &conf);

        r->cycles++;

        bool prediction_correct = (predicted_wi == expected_wi);
        bool confident_enough   = (conf >= REPLAY_CONF_MIN);

        if (prediction_correct && confident_enough) {
            /*
             * Got it right AND expressed it confidently.
             * No training needed — just acknowledge and move on.
             * This is how committed knowledge feels: effortless recall.
             */
            r->correct++;

            if (r->verbose) {
                printf("\r  [~] %c \xe2\x86\x92 %-12s \xe2\x9c\x93  %.0f%%\n",
                       'A' + li,
                       r->vocab->words[expected_wi],
                       conf * 100.0f);
                /* Reprint the prompt so typing is uninterrupted. */
                printf("mimir> ");
                fflush(stdout);
            }

        } else {
            /*
             * ─────────────────────────────────────────────────────────────
             * BUG (2026-04-11): Replay used to call train_step_brain here
             * (three-factor Hebbian) on RECALL associations.  That is the
             * exact training path that alpha_delta_rescue's header comment
             * documents as broken for RECALL — the per-neuron modulator
             * collapses, weights regress, and confident answers decay below
             * the threshold within seconds.
             *
             * Symptom observed in the wild:
             *   learn a apple   → recall a → "apple" 96 %   ✓
             *   learn b ball    → recall b → "[not learned yet]"   ✗
             *   recall a        → "[not learned yet]"   ✗   (also broken)
             * Training log showed cycles=8554, accuracy=0.029,
             * corrections=8310, committed=0 — replay was hammering away at
             * 20 Hz, "correcting" almost every cycle, achieving 3 %, and
             * silently destroying every fresh teach.
             *
             * FIX: reinforce RECALL associations using exactly the same
             * algorithm that alpha_teach uses to install them in the first
             * place — alpha_delta_rescue (single-layer discriminative delta
             * rule on the output neurons).  Same algorithm in both writers
             * means the network state is consistent: replay can only push
             * the weights toward the same fixed point that teach already
             * pushed them to, never away from it.
             *
             * WHY THIS WORKS WHERE HEBBIAN FAILS:
             * Three-factor Hebbian uses dw = lr · pre · post · modulator,
             * where modulator = error · surprise · (1 + alignment).  The
             * post and modulator factors both depend on the current output,
             * so when the output for a wrong-but-confident class is high,
             * the update is large and noisy across all hidden inputs and
             * keeps moving the wrong way.  The delta rule used by rescue is
             * dw = lr · pre · (target − post), with no post term and no
             * surprise scaling.  It is the closed-form gradient of squared
             * error on a single linear-in-weights output layer, so it
             * monotonically decreases the loss for every sample it sees.
             *
             * WHY DELTA RESCUE IS CHEAP HERE:
             * The function early-exits the moment every known association
             * is argmax-correct AND ≥ 0.80 confidence (alphabet.c).  In
             * steady state that is one forward-pass-per-sample sweep
             * (≤ 26 forwards) and we are gone before the next replay tick.
             * Only when something is genuinely wrong does it spend epochs.
             *
             * NEVER call train_step_brain on a RECALL embedding from this
             * file.  If you need to retrain RECALL, call delta_rescue.
             * ─────────────────────────────────────────────────────────────
             */
            /*
             * (2026-04-18) Dropped network_check_data verdict gate.
             *
             * The verdict check runs an UNMASKED network_forward for its
             * conflict logic.  After modality-separated vision training,
             * visual hidden neurons inject per-letter offsets into every
             * output, so unmasked forward looks "confidently wrong" for
             * most letters and returns VERDICT_CONFLICT on nearly every
             * cycle — silently blocking the retrain.
             *
             * alpha_delta_rescue already skips COMMITTED output neurons
             * (that's the actual protection we care about), and its
             * early-exit means it's a cheap no-op when the brain is
             * healthy.  Call it unconditionally when the masked check
             * said "wrong", and the real protection still holds.
             */
            {
                alpha_delta_rescue(r->abc_net, r->vocab);

                r->corrected++;

                if (r->verbose) {
                    /* Show what the brain said vs what it should have said */
                    const char *got = (predicted_wi < r->vocab->n_words)
                                      ? r->vocab->words[predicted_wi] : "???";
                    printf("\r  [~] %c \xe2\x86\x92 %-12s \xe2\x9c\x97  %.0f%%"
                           "  (said: %s) \xe2\x86\x92 retraining\n",
                           'A' + li,
                           r->vocab->words[expected_wi],
                           conf * 100.0f,
                           got);
                    printf("mimir> ");
                    fflush(stdout);
                }

                /* Periodically auto-save so no work is ever lost. */
                save_countdown--;
                if (save_countdown <= 0) {
                    save_countdown = REPLAY_SAVE_EVERY;
                    /* Save under the lock — network must not change mid-write */
                    checkpoint_mkdir(ALPHA_BRAIN_PATH);
                    network_save(r->abc_net,  ALPHA_BRAIN_PATH);
                    alpha_vocab_save(r->vocab, ALPHA_VOCAB_PATH);

                    /* Log the replay milestone to the persistent training log. */
                    int known_count = 0;
                    for (int _i = 0; _i < 26; _i++)
                        if (r->vocab->letter_to_word[_i] >= 0) known_count++;
                    double acc = (r->cycles > 0)
                                 ? (double)r->correct / (double)r->cycles
                                 : 0.0;
                    tlog_replay(r->cycles, known_count, acc,
                                r->corrected, network_committed_count(r->abc_net));

                    if (r->verbose) {
                        printf("\r  [~] [auto-saved]\n");
                        printf("mimir> ");
                        fflush(stdout);
                    }
                }
            }
            /* VERDICT_CONFLICT / VERDICT_REVERIFY: committed neurons protect
             * themselves — do not retrain, just move on. */
        }

        /*
         * (Added 2026-04-11) Heartbeat tlog_replay.
         *
         * The previous (broken) replay loop logged via the auto-save block
         * roughly every 10 corrections.  Because corrections fired ~97 % of
         * cycles, that produced a constant stream of log entries.  After
         * the alpha_delta_rescue fix, replay corrects ~1 % of cycles, so
         * that path almost never fires and the file looked silent even
         * though the brain was healthily running.
         *
         * To preserve observability we now write a periodic vital-signs
         * line every REPLAY_LOG_EVERY cycles regardless of correction
         * activity.  This is purely a log entry — no checkpoint write, no
         * weight update — so it costs nothing on the hot path.
         */
        if (r->cycles - last_heartbeat_cycle >= REPLAY_LOG_EVERY) {
            last_heartbeat_cycle = r->cycles;

            /*
             * (Added 2026-04-12) Periodic VALIDATE health check.
             *
             * RECALL associations are maintained by delta_rescue above,
             * but VALIDATE mappings (the identity: letter_i → output[i])
             * can drift when RECALL training adjusts shared output weights.
             * We piggyback on the heartbeat interval (~50 s) to run
             * delta_rescue_validate — it early-exits in one forward sweep
             * if everything is healthy, so the cost is negligible.
             */
            /*
             * (2026-04-12) VALIDATE rescue disabled in replay.
             *
             * Any VALIDATE training — even gentle (lr=0.05, 200 epochs) —
             * slowly destabilises RECALL because both share the same output
             * weights.  Over many heartbeats the cumulative drift degraded
             * RECALL from 26/26 → 6/26.
             *
             * VALIDATE is maintained via:
             *   - alpha_pretrain_sequence (full-strength, before RECALL exists)
             *   - quiz choice corrections (on-demand, user-initiated)
             * Replay focuses exclusively on RECALL stability.
             */

            int known_count = 0;
            for (int _i = 0; _i < 26; _i++)
                if (r->vocab->letter_to_word[_i] >= 0) known_count++;

            /*
             * (Added 2026-04-12) Output neuron commitment.
             *
             * Once all 26 letters are taught and every RECALL query is
             * argmax-correct at ≥ 90% confidence, commit the output
             * layer.  This freezes the word-association weights
             * permanently — the brain no longer needs the delta-rescue
             * tutor because it truly "knows" the answers.
             *
             * We check RECALL only (not VALIDATE) because VALIDATE
             * training cannot coexist with RECALL on the same weights
             * without oscillation.  RECALL is the primary association
             * task and the one replay maintains.
             *
             * We only attempt this when:
             *   1. All 26 letters have been taught (known_count == 26)
             *   2. Output neurons are still MATURE (not yet committed)
             *   3. Every RECALL forward pass is correct with confidence
             *      ≥ 0.85 and every VALIDATE is argmax-correct
             *
             * After commitment, delta_rescue skips committed neurons,
             * so the weights are frozen forever.
             */
            if (known_count == 26) {
                Layer *ol = &r->abc_net->layers[r->abc_net->n_layers - 1];
                bool output_already_committed = false;
                for (int _i = 0; _i < ol->count; _i++) {
                    if (ol->neurons[_i].state == NEURON_COMMITTED) {
                        output_already_committed = true;
                        break;
                    }
                }

                if (!output_already_committed) {
                    bool all_solid = true;
                    float emb[MIMIR_EMBEDDING_SIZE], out[ALPHA_N_OUTPUTS];

                    /* Check all 26 RECALL associations at ≥ 0.85 */
                    for (int _i = 0; _i < 26 && all_solid; _i++) {
                        int wi = r->vocab->letter_to_word[_i];
                        if (wi < 0) { all_solid = false; break; }
                        alpha_forward(r->abc_net, _i,
                                      ALPHA_QUERY_RECALL, emb, out);
                        float c; int best = argmax_f(out, ALPHA_N_OUTPUTS, &c);
                        if (best != wi || c < 0.85f) all_solid = false;
                    }

                    /* Also check VALIDATE is argmax-correct */
                    for (int _i = 0; _i < 26 && all_solid; _i++) {
                        alpha_forward(r->abc_net, _i,
                                      ALPHA_QUERY_VALIDATE, emb, out);
                        float c; int best = argmax_f(out, ALPHA_N_OUTPUTS, &c);
                        if (best != _i) all_solid = false;
                    }

                    if (all_solid) {
                        int n = network_commit_output(r->abc_net);
                        if (n > 0) {
                            /* Save immediately — this is a milestone */
                            checkpoint_mkdir(ALPHA_BRAIN_PATH);
                            network_save(r->abc_net, ALPHA_BRAIN_PATH);
                            alpha_vocab_save(r->vocab, ALPHA_VOCAB_PATH);

                            printf("\r  [Replay] OUTPUT COMMIT: %d output "
                                   "neurons now hold permanent knowledge.\n",
                                   n);
                            printf("  [Replay] The brain truly knows all "
                                   "26 letters — no more retraining needed.\n");
                            printf("mimir> ");
                            fflush(stdout);
                        }
                    }
                }
            }

            /*
             * (Added 2026-04-17) Vision health check.
             *
             * Text delta_rescue adjusts output biases, which shifts the
             * operating point of output neurons used by BOTH modalities.
             * Vision predictions drift between tests without this.
             *
             * vision_rescue updates only visual-side output weights
             * (skips text hidden neurons, no bias update, no hidden
             * backprop).  Early-exits in one forward sweep if all visual
             * predictions are correct at 80 %.  Cost is negligible when
             * the brain is healthy.
             */
            if (r->vis_loaded > 0) {
                int before_correct = 0;
                float emb[MIMIR_EMBEDDING_SIZE], out[ALPHA_N_OUTPUTS];
                for (int i = 0; i < 26; i++) {
                    int wi = r->vocab->letter_to_word[i];
                    if (wi < 0 || r->vis_images[i] == NULL) continue;
                    vision_forward(r->abc_net, r->vis_images[i], emb, out);
                    float c; int best = argmax_f(out, ALPHA_N_OUTPUTS, &c);
                    if (best == wi && c >= 0.80f) before_correct++;
                }
                int n_vis = r->vis_loaded;
                if (before_correct < n_vis) {
                    vision_rescue(r->abc_net, r->vocab, r->vis_images);
                }
            }

            double acc = (r->cycles > 0)
                         ? (double)r->correct / (double)r->cycles
                         : 0.0;
            tlog_replay(r->cycles, known_count, acc,
                        r->corrected, network_committed_count(r->abc_net));
        }

        pthread_mutex_unlock(&r->lock);

        /* Sleep OUTSIDE the lock so the CLI thread is not blocked
         * waiting for the replay thread to finish sleeping.            */
        usleep(REPLAY_SLEEP_US);
    }

    return NULL;
}

/* ── Public API ────────────────────────────────────────────────────────── */

void replay_init(ReplayState *r, Network *abc_net, AlphaVocab *vocab) {
    memset(r, 0, sizeof(*r));
    r->abc_net = abc_net;
    r->vocab   = vocab;
    r->running = 0;
    r->verbose = 0;
    pthread_mutex_init(&r->lock, NULL);

    /* Load word images for vision replay.
     * Done once at init — images don't change during a session. */
    r->vis_loaded = vision_load_all(vocab, r->vis_images);
}

int replay_start(ReplayState *r) {
    if (r->running) {
        printf("  [Replay] Already running.\n");
        return 0;
    }
    r->running = 1;
    int rc = pthread_create(&r->thread, NULL, replay_thread_fn, r);
    if (rc != 0) {
        fprintf(stderr, "[Replay] Failed to create thread (errno %d)\n", rc);
        r->running = 0;
        return -1;
    }
    printf("  [Replay] Background self-training started.\n");
    printf("  [Replay] The brain will keep practising even while you type.\n");
    printf("  [Replay] Type 'replay verbose' to watch it work.\n");
    return 0;
}

void replay_stop(ReplayState *r) {
    if (!r->running) return;
    r->running = 0;
    pthread_join(r->thread, NULL);   /* wait for thread to finish its current step */
    pthread_mutex_destroy(&r->lock);
    if (r->vis_loaded > 0) {
        vision_free_all(r->vis_images);
        r->vis_loaded = 0;
    }
    printf("  [Replay] Stopped. Brain has rested.\n");
}

void replay_status(const ReplayState *r) {
    unsigned long c = r->cycles;
    unsigned long ok = r->correct;
    double acc = (c > 0) ? (100.0 * ok / c) : 0.0;

    printf("\n  Replay status:\n");
    printf("    State:        %s\n", r->running ? "running" : "stopped");
    printf("    Verbose:      %s\n", r->verbose ? "on" : "off");
    printf("    Cycles:       %lu  (letter-tests performed)\n", c);
    printf("    Accuracy:     %.1f%%  (%lu / %lu correct)\n", acc, ok, c);
    printf("    Retrains:     %lu  (self-corrections made)\n", r->corrected);
    printf("    Sleep period: %d ms between tests\n\n", REPLAY_SLEEP_US / 1000);
}
