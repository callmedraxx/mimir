/*
 * MIMIR - Main entry point
 * Step 1: Prove the perceptron works by learning basic logic gates
 *
 * WHY LOGIC GATES?
 * Logic gates are the simplest possible classification problems:
 * - 2 binary inputs, 1 binary output
 * - Only 4 possible input combinations (the truth table)
 * - We know the exact answer
 *
 * This lets us verify our implementation is correct before scaling up.
 * It also demonstrates the fundamental LIMITATION of a single perceptron:
 * it can only learn linearly separable functions.
 *
 * LINEARLY SEPARABLE means you can draw a straight line (in 2D) or
 * hyperplane (in nD) that separates the positive from negative examples.
 * AND and OR are linearly separable. XOR is not.
 *
 * WHAT COMES NEXT (after we verify this works):
 * Step 2: Multi-layer perceptron (MLP) to solve XOR
 * Step 3: Backpropagation for training the MLP
 * Step 4: Matrix operations and SIMD optimization
 * Step 5: Attention mechanism
 * Step 6: Full transformer
 * Step 7: Tokenizer
 * Step 8: Training on text data
 */

#include "mimir.h"
#include <ctype.h>    /* tolower() for letter parsing in CLI commands */
#include <time.h>     /* clock_gettime() for learn-event timing */

/*
 * Test: Can a perceptron learn AND?
 *
 * AND truth table:
 *   0 AND 0 = 0
 *   0 AND 1 = 0
 *   1 AND 0 = 0
 *   1 AND 1 = 1   <- only case that outputs 1
 *
 * Geometrically: We need a line in the (x1, x2) plane such that
 * (1,1) is on one side and (0,0), (0,1), (1,0) are on the other.
 * This is easily possible—a line like x1 + x2 = 1.5 works.
 *
 * EXPECTED LEARNED WEIGHTS:
 * w1 ≈ w2 ≈ large positive (both inputs must contribute)
 * bias ≈ large negative (need both inputs to overcome it)
 * The decision boundary: w1*x1 + w2*x2 + bias = 0
 * So: ~5*x1 + ~5*x2 - ~8 = 0, i.e., x1 + x2 ≈ 1.6
 */
void test_and_gate(void) {
    printf("=== Learning AND gate ===\n");

    /*
     * Create a 2-input perceptron with sigmoid activation.
     * WHY SIGMOID? For binary classification, sigmoid's (0,1) output range
     * is perfect—we can interpret the output as P(output=1|inputs).
     * We threshold at 0.5 for the final decision.
     */
    Perceptron p = perceptron_create(2, ACT_SIGMOID);

    /*
     * Training data: all 4 combinations of 2 binary inputs.
     * This is TINY—real networks train on millions of examples.
     * But for a 2-input perceptron learning a known function,
     * 4 examples is the complete dataset.
     */
    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float targets[4]   = { 0,     0,     0,     1   };

    /*
     * Train for 1000 epochs. An "epoch" = one pass through all training data.
     *
     * WHY 1000? Empirically, sigmoid perceptron converges on AND in ~200-500
     * epochs. 1000 gives plenty of margin. In practice you'd monitor the loss
     * and stop when it plateaus (early stopping).
     *
     * WHY NOT 10? Might not converge. Sigmoid gradients are small when the
     * output is near 0 or 1, so late-stage fine-tuning is slow.
     *
     * WHY NOT 1,000,000? Wasting time. The solution doesn't improve after
     * convergence (no regularization needed for 3 parameters and 4 examples).
     *
     * HYPOTHETICAL: "Curriculum learning" — What if we trained on easy examples
     * first? For AND, start with (1,1)->1 and (0,0)->0 (the extremes) before
     * introducing the ambiguous cases. Bengio et al. (2009) showed this can
     * speed up convergence. WHY IT MIGHT WORK: The network builds a rough
     * decision boundary from easy cases, then refines it. WHY IT MIGHT NOT:
     * For 4 examples with a single perceptron, there's nothing to gain.
     * Curriculum learning shines on large, noisy datasets.
     */
    for (int epoch = 0; epoch < 1000; epoch++) {
        for (int i = 0; i < 4; i++) {
            perceptron_train(&p, inputs[i], targets[i], 1.0f);
        }
    }

    /* Test on all inputs and display results */
    printf("Results after training:\n");
    for (int i = 0; i < 4; i++) {
        float out = perceptron_forward(&p, inputs[i]);
        printf("  %d AND %d = %.3f (expected %d)\n",
               (int)inputs[i][0], (int)inputs[i][1], out, (int)targets[i]);
    }

    /* Show what the perceptron learned */
    perceptron_print(&p);

    /* Clean up */
    perceptron_free(&p);
    printf("\n");
}

/*
 * Test: Can a perceptron learn OR?
 *
 * OR truth table:
 *   0 OR 0 = 0   <- only case that outputs 0
 *   0 OR 1 = 1
 *   1 OR 0 = 1
 *   1 OR 1 = 1
 *
 * Geometrically: We need a line where (0,0) is on one side and
 * (0,1), (1,0), (1,1) are on the other. Even easier than AND.
 * A line like x1 + x2 = 0.5 works.
 *
 * EXPECTED LEARNED WEIGHTS:
 * w1 ≈ w2 ≈ large positive
 * bias ≈ small negative (only need one input to overcome it)
 */
void test_or_gate(void) {
    printf("=== Learning OR gate ===\n");
    Perceptron p = perceptron_create(2, ACT_SIGMOID);

    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float targets[4]   = { 0,     1,     1,     1   };

    for (int epoch = 0; epoch < 1000; epoch++) {
        for (int i = 0; i < 4; i++) {
            perceptron_train(&p, inputs[i], targets[i], 1.0f);
        }
    }

    printf("Results after training:\n");
    for (int i = 0; i < 4; i++) {
        float out = perceptron_forward(&p, inputs[i]);
        printf("  %d OR %d = %.3f (expected %d)\n",
               (int)inputs[i][0], (int)inputs[i][1], out, (int)targets[i]);
    }
    perceptron_print(&p);
    perceptron_free(&p);
    printf("\n");
}

/*
 * Test: XOR - the one a single perceptron CANNOT learn.
 *
 * XOR truth table:
 *   0 XOR 0 = 0
 *   0 XOR 1 = 1
 *   1 XOR 0 = 1
 *   1 XOR 1 = 0   <- this is what breaks it
 *
 * WHY IT'S IMPOSSIBLE:
 * Try to draw a single straight line separating {(0,1),(1,0)} from {(0,0),(1,1)}.
 * You can't. The positive examples are on OPPOSITE corners of the unit square.
 * Any line that puts (0,1) and (1,0) on one side must also include either
 * (0,0) or (1,1). Mathematically: no w1, w2, b exist such that
 *   sigmoid(w1*0 + w2*0 + b) < 0.5  AND
 *   sigmoid(w1*0 + w2*1 + b) > 0.5  AND
 *   sigmoid(w1*1 + w2*0 + b) > 0.5  AND
 *   sigmoid(w1*1 + w2*1 + b) < 0.5
 * (Proof: the first and fourth imply b < 0 and w1+w2+b < 0, i.e., w1+w2 < -b.
 * But second and third imply w2+b > 0 and w1+b > 0, so w1 > -b and w2 > -b,
 * meaning w1+w2 > -2b > -b. Contradiction.)
 *
 * THIS IS THE MOST IMPORTANT FAILURE IN AI HISTORY.
 * Minsky & Papert's proof of this limitation shut down neural network
 * research for over a decade. The solution—hidden layers—was known but
 * there was no efficient training algorithm until backpropagation.
 *
 * SOLUTION (next step): Two perceptrons in a hidden layer + one output
 * perceptron. Hidden neuron 1 learns OR, hidden neuron 2 learns NAND.
 * Output combines them: XOR = OR AND NAND. This is the multi-layer
 * perceptron (MLP), our next building block.
 *
 * HYPOTHETICAL: Can a single neuron learn XOR with a different activation?
 * What if we used activation = sin(x)? Periodic functions can create
 * multiple decision boundaries. sin(w1*x1 + w2*x2 + b) could potentially
 * map the XOR pattern. WHY IT MIGHT WORK: sin() creates wavy decision
 * boundaries that can separate non-convex regions. Papers on "periodic
 * activation functions" (SIREN, Sitzmann 2020) show they work for certain
 * tasks. WHY IT MIGHT NOT: The optimization landscape becomes highly
 * non-convex with many local minima. Gradient descent would struggle to
 * find the right frequency. And it doesn't generalize—you'd need to tune
 * the periodicity for each problem. The MLP is a more general solution.
 */
void test_xor_gate(void) {
    printf("=== Attempting XOR gate (should fail!) ===\n");
    Perceptron p = perceptron_create(2, ACT_SIGMOID);

    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float targets[4]   = { 0,     1,     1,     0   };

    /*
     * We train for 5000 epochs (5x more than AND/OR) to really prove
     * that no amount of training helps. The perceptron will oscillate
     * forever without converging.
     */
    for (int epoch = 0; epoch < 5000; epoch++) {
        for (int i = 0; i < 4; i++) {
            perceptron_train(&p, inputs[i], targets[i], 1.0f);
        }
    }

    printf("Results after training (a single perceptron can't solve XOR):\n");
    for (int i = 0; i < 4; i++) {
        float out = perceptron_forward(&p, inputs[i]);
        /*
         * Check if the output is far from the target (> 0.3 error).
         * For XOR, ALL outputs will hover around 0.5 (maximum uncertainty),
         * because the perceptron is equally confused about every input.
         * This 0.5 output means "I have no idea"—the sigmoid's way of shrugging.
         */
        printf("  %d XOR %d = %.3f (expected %d) %s\n",
               (int)inputs[i][0], (int)inputs[i][1], out, (int)targets[i],
               (fabsf(out - targets[i]) > 0.3f) ? "<-- WRONG" : "");
    }
    perceptron_print(&p);
    perceptron_free(&p);
    printf("\n");
    printf("^ This failure is WHY we need multi-layer networks.\n");
    printf("  XOR requires a hidden layer. That's our next step.\n");
}

// ============================================================
// STEP 2: NEUROGENESIS — ONE brain learns AND, OR, and XOR simultaneously
// ============================================================

/*
 * Train a single combined network with 3 outputs to learn all three
 * binary logic gates simultaneously.
 *
 * Output 0 = AND, Output 1 = OR, Output 2 = XOR
 *
 * This is more demanding than training three separate networks: the
 * hidden layer must discover representations that simultaneously support
 * all three output patterns. XOR in particular requires non-linear
 * features. By combining all three tasks, the hidden neurons must
 * find a basis that serves all outputs — forcing richer representations.
 *
 * Flat targets layout: targets[sample * 3 + output_idx]
 */
Network test_network_gates(void) {
    printf("=== One Brain: Learning AND, OR, and XOR simultaneously ===\n");
    random_seed(52);

    /* 8 active hidden + 12 dormant pool, 3 outputs (AND/OR/XOR).
     * Seed 52: gives enough initial weight diversity that WTA doesn't
     * collapse neurons to identical weights — see design history in training.c. */
    Network net = network_create_with_pool(2, 8, 12, 3, ACT_SIGMOID);
    printf("  Starting: %d active neurons, 12 in pool, 2 inputs, 3 outputs (AND/OR/XOR)\n",
           network_neuron_count(&net) - 12);

    /* Flat targets layout: targets[sample * 3 + output_idx]
     * Output 0=AND, 1=OR, 2=XOR */
    float inputs[]  = {0,0, 0,1, 1,0, 1,1};
    float targets[] = {
        0, 0, 0,   /* 0 op 0 */
        0, 1, 1,   /* 0 op 1 */
        0, 1, 1,   /* 1 op 0 */
        1, 1, 0,   /* 1 op 1 */
    };

    network_auto_train(&net, inputs, targets, 4, 50000, 0.3f);

    /* Show results for all 3 gates */
    printf("  Results (AND | OR | XOR):\n");
    int correct = 0;
    float out[3];
    for (int i = 0; i < 4; i++) {
        network_forward(&net, inputs + i*2, out);
        int a = (int)inputs[i*2], b = (int)inputs[i*2+1];
        int t_and = (int)targets[i*3+0], t_or = (int)targets[i*3+1], t_xor = (int)targets[i*3+2];
        bool ok_and = ((int)(out[0]>0.5f) == t_and);
        bool ok_or  = ((int)(out[1]>0.5f) == t_or);
        bool ok_xor = ((int)(out[2]>0.5f) == t_xor);
        if (ok_and && ok_or && ok_xor) correct++;
        printf("    %d,%d \xe2\x86\x92 AND=%.3f(%s) OR=%.3f(%s) XOR=%.3f(%s) %s\n",
               a, b,
               out[0], ok_and ? "\xe2\x9c\x93" : "\xe2\x9c\x97",
               out[1], ok_or  ? "\xe2\x9c\x93" : "\xe2\x9c\x97",
               out[2], ok_xor ? "\xe2\x9c\x93" : "\xe2\x9c\x97",
               (ok_and && ok_or && ok_xor) ? "" : "<-- some wrong");
    }
    printf("  All-correct rows: %d/4\n", correct);

    network_print(&net);
    return net;
}

// ============================================================
// STEP 3: INTERACTIVE CLI — Probe the trained brain
// ============================================================

/*
 * Print a confidence bar for a single output value.
 *
 * output=0.03 → very confident NO  : [                    ] 0.0300
 * output=0.50 → completely unsure  : [==========          ] 0.5000
 * output=0.97 → very confident YES : [====================] 0.9700
 */
static void print_confidence(float output) {
    int filled = (int)(output * 20.0f);
    printf("[");
    for (int i = 0; i < 20; i++) printf(i < filled ? "=" : " ");
    printf("] %.4f  \xe2\x86\x92  %s\n", output, output > 0.5f ? "1 (YES)" : "0 (NO)");
}

/*
 * Print all neurons in the network with their current state.
 */
static void print_neuron_detail(Network *net) {
    printf("\n  Neurons (last forward pass outputs):\n");
    for (int l = 0; l < net->n_layers; l++) {
        Layer *layer = &net->layers[l];
        const char *lname = (l == net->n_layers - 1) ? "output" : "hidden";
        printf("  Layer %d (%s): %d neurons\n", l, lname, layer->count);
        for (int j = 0; j < layer->count; j++) {
            Neuron *n = &layer->neurons[j];
            const char *state_str;
            switch (n->state) {
                case NEURON_DORMANT:   state_str = "DORMANT  "; break;
                case NEURON_IMMATURE:  state_str = "IMMATURE "; break;
                case NEURON_MATURE:    state_str = "MATURE   "; break;
                case NEURON_COMMITTED: state_str = "COMMITTED"; break;
                default:               state_str = "UNKNOWN  "; break;
            }
            printf("    #%-2d [%s] output=%.4f  activity=%.4f  maturity=%.2f\n",
                   n->id, state_str, n->last_output, n->activity, n->maturity);
        }
    }
    printf("\n");
}

/*
 * The interactive CLI — single brain, three outputs (AND / OR / XOR).
 *
 * COMMANDS:
 *   predict <x1> <x2>                   — show AND/OR/XOR outputs + bars
 *   train <x1> <x2> <and> <or> <xor>   — teach one sample (5 values)
 *   grow                                — manually trigger neurogenesis
 *   neurons                             — show all neuron states
 *   info                                — show network topology
 *   reset                               — wipe and retrain from scratch
 *   commit                              — lock in current knowledge
 *   help                                — list commands
 *   quit / exit                         — leave
 */

/* Gate index helpers — the brain has 3 outputs: 0=AND, 1=OR, 2=XOR */
static int gate_index(const char *name) {
    if (strcmp(name, "and") == 0) return 0;
    if (strcmp(name, "or")  == 0) return 1;
    if (strcmp(name, "xor") == 0) return 2;
    return -1;
}
static const char *gate_name(int idx) {
    if (idx == 0) return "AND";
    if (idx == 1) return "OR";
    if (idx == 2) return "XOR";
    return "?";
}

/*
 * The interactive CLI — one brain, three outputs (AND / OR / XOR).
 *
 * COMMANDS:
 *   predict <x1> <x2>               — show all three outputs + bars
 *   predict <gate> <x1> <x2>        — show only one gate (and/or/xor)
 *   train <x1> <x2> <and> <or> <xor>  — teach one sample (all 3 targets)
 *   train <gate> <x1> <x2> <target>   — teach one gate only
 *   grow                            — manually trigger neurogenesis
 *   neurons                         — show all neuron states
 *   info                            — show network topology summary
 *   commit                          — lock in current knowledge
 *   reset                           — wipe and retrain from scratch
 *   help                            — list commands
 *   quit / exit                     — leave
 *
 * NOTE on per-gate training:
 *   "train xor 1 1 0" sets targets = [AND=current, OR=current, XOR=0].
 *   The network's own current outputs are used for the other two gates,
 *   so their error is exactly 0 — only XOR's weights get corrected.
 *   This lets you fine-tune one gate without disturbing the others.
 */
/*
 * run_cli — the interactive command loop for the whole brain.
 *
 * Handles two networks simultaneously:
 *   gate_net  — the logic gate brain (AND / OR / XOR), trained at startup.
 *   abc_net   — the alphabet brain, taught interactively by the user.
 *
 * The replay thread runs in the background the entire time, continuously
 * rehearsing abc_net's known associations. All abc_net/vocab access in
 * this function holds replay->lock to stay thread-safe.
 */
static void run_cli(Network *gate_net, Network *abc_net,
                    AlphaVocab *vocab, ReplayState *replay,
                    Hippocampus *hippo) {
    printf("\n");
    printf("  +--------------------------------------------------+\n");
    printf("  |         MIMIR  Interactive CLI  v0.2.0          |\n");
    printf("  |   Type 'help' to see available commands         |\n");
    printf("  +--------------------------------------------------+\n");
    printf("\n");
    printf("  Gate brain: %d inputs -> %d neurons -> 3 outputs (AND/OR/XOR)\n",
           gate_net->n_inputs, network_neuron_count(gate_net));
    printf("  ABC  brain: %d inputs -> %d neurons -> %d outputs (A-Z)\n",
           abc_net->n_inputs, network_neuron_count(abc_net), abc_net->n_outputs);
    printf("  Replay: %s\n\n", replay->running ? "running" : "stopped");
    printf("  Try: learn a apple    recall a    recite    quiz\n\n");

    /* Gate training data used by 'reset' */
    static float reset_inputs[]  = {0,0, 0,1, 1,0, 1,1};
    static float reset_targets[] = {
        0,0,0,  0,1,1,  0,1,1,  1,1,0
    };

    char line[256];

    while (1) {
        printf("mimir> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) { printf("\n"); break; }

        int len = (int)strlen(line);
        if (len > 0 && line[len-1] == '\n') line[--len] = '\0';
        if (len == 0) continue;

        float x1, x2;
        char gname[16];

        /* ════════════════════════════════════════════════════
         * GATE COMMANDS  (logic gate brain)
         * ════════════════════════════════════════════════════ */

        /* ── predict <gate> <x1> <x2>  — single-gate output ── */
        if (sscanf(line, "predict %15s %f %f", gname, &x1, &x2) == 3) {
            int gi = gate_index(gname);
            if (gi < 0) {
                printf("  Unknown gate '%s'. Use: and, or, xor\n", gname);
                continue;
            }
            float inp[2] = {x1, x2};
            float out[MAX_OUTPUTS];
            network_forward(gate_net, inp, out);
            printf("\n  %s(%g, %g)  ", gate_name(gi), x1, x2);
            print_confidence(out[gi]);
            printf("\n");
            continue;
        }

        /* ── predict <x1> <x2>  — all three gate outputs ── */
        if (sscanf(line, "predict %f %f", &x1, &x2) == 2) {
            float inp[2] = {x1, x2};
            float out[MAX_OUTPUTS];
            network_forward(gate_net, inp, out);
            printf("\n  Input: [%.4f, %.4f]\n", x1, x2);
            printf("    AND  ");  print_confidence(out[0]);
            printf("    OR   ");  print_confidence(out[1]);
            printf("    XOR  ");  print_confidence(out[2]);
            printf("\n");
            continue;
        }

        /* ── train <gate> <x1> <x2> <target>  — per-gate teaching ── */
        float tgt;
        if (sscanf(line, "train %15s %f %f %f", gname, &x1, &x2, &tgt) == 4) {
            int gi = gate_index(gname);
            if (gi < 0) {
                printf("  Unknown gate '%s'. Use: and, or, xor\n", gname);
                continue;
            }
            float inp[2] = {x1, x2};
            float out_now[MAX_OUTPUTS];
            network_forward(gate_net, inp, out_now);
            float tgts[3] = { out_now[0], out_now[1], out_now[2] };
            tgts[gi] = tgt;

            TrainVerdict verdict = network_check_data(gate_net, inp, tgts, 3, 0.8f);
            if (verdict == VERDICT_REVERIFY) {
                printf("\n  [REVERIFY] Seen this %s conflict %d+ times.\n"
                       "  I believe %s(%g,%g) = %.3f  (%s)\n"
                       "  You keep saying it should be %.0f.\n"
                       "  Type 'reset' if the task has genuinely changed.\n\n",
                       gate_name(gi), CONFLICT_REVERIFY_AT,
                       gate_name(gi), x1, x2, out_now[gi],
                       out_now[gi] > 0.5f ? "YES" : "NO", tgt);
                continue;
            }
            if (verdict == VERDICT_CONFLICT) {
                printf("\n  [CONFLICT] I already know %s(%g,%g) = %.3f (%s)\n"
                       "  Teaching %.0f would corrupt committed knowledge.\n"
                       "  Rejected. (%d more time(s) -> REVERIFY)\n\n",
                       gate_name(gi), x1, x2, out_now[gi],
                       out_now[gi] > 0.5f ? "YES" : "NO",
                       tgt, CONFLICT_REVERIFY_AT - 1);
                continue;
            }
            float out_before[MAX_OUTPUTS];
            network_forward(gate_net, inp, out_before);
            train_step_brain(gate_net, inp, tgts, 3, 0.5f);
            float out_after[MAX_OUTPUTS];
            network_forward(gate_net, inp, out_after);
            printf("\n  Trained %s(%g,%g) -> %.0f\n", gate_name(gi), x1, x2, tgt);
            printf("  %s:  %.4f -> %.4f  (delta %.4f)\n",
                   gate_name(gi),
                   fabsf(tgt - out_before[gi]),
                   fabsf(tgt - out_after[gi]),
                   fabsf(tgt - out_after[gi]) - fabsf(tgt - out_before[gi]));
            printf("  %s  ", gate_name(gi));
            print_confidence(out_after[gi]);
            printf("\n");
            continue;
        }

        /* ── train <x1> <x2> <and> <or> <xor>  — all 3 targets ── */
        float t0, t1, t2;
        if (sscanf(line, "train %f %f %f %f %f", &x1, &x2, &t0, &t1, &t2) == 5) {
            float inp[2] = {x1, x2};
            float tgts[3] = {t0, t1, t2};
            TrainVerdict verdict = network_check_data(gate_net, inp, tgts, 3, 0.8f);
            if (verdict == VERDICT_REVERIFY) {
                float out_now[MAX_OUTPUTS];
                network_forward(gate_net, inp, out_now);
                printf("\n  [REVERIFY] Conflict seen %d+ times.\n"
                       "  My belief: AND=%.3f OR=%.3f XOR=%.3f\n"
                       "  Type 'reset' if the task has genuinely changed.\n\n",
                       CONFLICT_REVERIFY_AT, out_now[0], out_now[1], out_now[2]);
                continue;
            }
            if (verdict == VERDICT_CONFLICT) {
                float out_now[MAX_OUTPUTS];
                network_forward(gate_net, inp, out_now);
                printf("\n  [CONFLICT] Committed knowledge contradicts new targets.\n"
                       "  My belief: AND=%.3f OR=%.3f XOR=%.3f\n"
                       "  Rejected. (%d more time(s) -> REVERIFY)\n\n",
                       out_now[0], out_now[1], out_now[2], CONFLICT_REVERIFY_AT - 1);
                continue;
            }
            float out_before[MAX_OUTPUTS];
            network_forward(gate_net, inp, out_before);
            train_step_brain(gate_net, inp, tgts, 3, 0.5f);
            float out_after[MAX_OUTPUTS];
            network_forward(gate_net, inp, out_after);
            printf("\n  Trained [%g,%g] -> AND=%.0f OR=%.0f XOR=%.0f\n",
                   x1, x2, t0, t1, t2);
            printf("  AND  before=%.3f after=%.3f\n", out_before[0], out_after[0]);
            printf("  OR   before=%.3f after=%.3f\n", out_before[1], out_after[1]);
            printf("  XOR  before=%.3f after=%.3f\n", out_before[2], out_after[2]);
            printf("\n");
            continue;
        }

        /* ── grow ── */
        if (strcmp(line, "grow") == 0) {
            printf("\n  Triggering neurogenesis (gate brain)...\n  ");
            network_neurogenesis(gate_net);
            printf("  Gate brain: %d neurons (%d committed)\n\n",
                   network_neuron_count(gate_net), network_committed_count(gate_net));
            continue;
        }

        /* ── neurons ── */
        if (strcmp(line, "neurons") == 0) {
            print_neuron_detail(gate_net);
            continue;
        }

        /* ── info ── */
        if (strcmp(line, "info") == 0) {
            printf("\n  — Gate brain —\n");
            network_print(gate_net);
            pthread_mutex_lock(&replay->lock);
            printf("\n  — ABC brain —\n");
            network_print(abc_net);
            pthread_mutex_unlock(&replay->lock);
            printf("\n");
            continue;
        }

        /* ── commit ── */
        if (strcmp(line, "commit") == 0) {
            network_commit(gate_net);
            printf("  Gate brain: %d neurons (%d committed)\n\n",
                   network_neuron_count(gate_net), network_committed_count(gate_net));
            continue;
        }

        /* ── reset ── */
        if (strcmp(line, "reset") == 0) {
            printf("\n  Wiping gate brain and retraining AND+OR+XOR from scratch...\n\n");
            network_free(gate_net);
            random_seed(52);
            *gate_net = network_create_with_pool(2, 8, 12, 3, ACT_SIGMOID);
            network_auto_train(gate_net, reset_inputs, reset_targets, 4, 50000, 0.3f);
            network_clear_conflicts(gate_net);
            checkpoint_mkdir(CHECKPOINT_PATH);
            if (network_save(gate_net, CHECKPOINT_PATH) == 0)
                printf("  [Checkpoint updated: %s]\n", CHECKPOINT_PATH);
            printf("\n  Done. Gate brain: %d neurons (%d committed).\n\n",
                   network_neuron_count(gate_net), network_committed_count(gate_net));
            continue;
        }

        /* ════════════════════════════════════════════════════
         * ALPHABET COMMANDS  (ABC brain — runs under replay lock)
         * ════════════════════════════════════════════════════ */

        /*
         * learn <letter> <word>
         *
         * Teach the brain one association: "A is for apple".
         * Acquires the replay lock so the background thread cannot modify
         * the network while we are training. Releases it when done.
         * Auto-saves after a successful teach.
         *
         * Examples:
         *   learn a apple
         *   learn B Banana
         *   learn z zebra
         */
        /*
         * learn all
         *
         * Bulk-teach the brain one word for every letter A-Z, drawn from the
         * built-in word bank in alphabet.c (alpha_word_bank_get(li, 0)).
         * Letters that already have a vocab entry are skipped so existing
         * teach state is preserved.  This exists so a fresh checkpoint can
         * be populated in one shot — useful before running `quiz choice`,
         * which tests generalisation against UNSEEN bank words.
         */
        if (strcmp(line, "learn all") == 0) {
            printf("\n  Teaching one word per letter from the built-in bank "
                   "(text + vision per letter)...\n\n");
            /*
             * (2026-04-16) Truly per-letter: for each letter, run the same
             * alpha_teach + vision_train sequence the single-letter `learn`
             * branch runs.  Earlier bulk approach (alpha_teach_bulk → single
             * 26-image vision_train) left text at 0/26 because the cold
             * joint retrain on 26 letters didn't converge in one pass.
             * Going letter-by-letter gives each text retrain a warm start
             * (N-1 letters already bound) and keeps per-pass visual
             * gradient small.
             */
            int taught = 0, skipped = 0, failed = 0;
            float *img_data[26];
            for (int i = 0; i < 26; i++) img_data[i] = NULL;

            for (int li = 0; li < 26; li++) {
                if (vocab->letter_to_word[li] >= 0) {
                    skipped++;
                    continue;
                }
                const char *w = alpha_word_bank_get(li, 0);
                if (!w) {
                    printf("  %c -> (no bank entry)  FAILED\n", (char)('A' + li));
                    failed++;
                    continue;
                }

                printf("\n  [%d/26] Teaching %c -> %s\n",
                       li + 1, (char)('A' + li), w);
                fflush(stdout);

                pthread_mutex_lock(&replay->lock);
                AlphaTeachResult res = alpha_teach(abc_net, vocab, li, w);
                pthread_mutex_unlock(&replay->lock);

                if (res != ALPHA_LEARNED) {
                    printf("  %c -> %-12s  FAILED (%d)\n",
                           (char)('A' + li), w, (int)res);
                    failed++;
                    continue;
                }
                taught++;
                printf("  %c -> %-12s  ok\n", (char)('A' + li), w);
                fflush(stdout);

                /* Bind vision for letters taught so far, if image available. */
                pthread_mutex_lock(&replay->lock);
                char path[256];
                snprintf(path, sizeof(path), "data/images/%s.raw", w);
                float *buf = malloc(VISION_RAW_SIZE * sizeof(float));
                int have_image = 0;
                if (buf && vision_load_raw(path, buf) == 0) {
                    img_data[li] = buf;
                    have_image = 1;
                } else {
                    free(buf);
                }

                int n_img = 0;
                for (int i = 0; i < 26; i++) if (img_data[i]) n_img++;

                if (have_image && n_img > 0) {
                    printf("  Binding visual association (%d image%s so far)...\n",
                           n_img, n_img == 1 ? "" : "s");
                    fflush(stdout);
                    vision_train(abc_net, vocab, img_data);
                }
                network_save(abc_net, ALPHA_BRAIN_PATH);
                alpha_vocab_save(vocab, ALPHA_VOCAB_PATH);
                pthread_mutex_unlock(&replay->lock);
            }

            vision_free_all(img_data);

            printf("\n  Done. Taught %d, skipped %d (already known), failed %d.\n",
                   taught, skipped, failed);
            printf("  Run `quiz choice` to test generalisation on UNSEEN bank words.\n\n");
            continue;
        }

        char abc_letter[4], abc_word[ALPHA_MAX_WORD_LEN];
        if (sscanf(line, "learn %3s %31s", abc_letter, abc_word) == 2 &&
            strncmp(line, "learn", 5) == 0) {
            char lc = (char)tolower((unsigned char)abc_letter[0]);
            if (lc < 'a' || lc > 'z') {
                printf("  '%s' is not a letter (a–z).\n", abc_letter);
                continue;
            }
            int li = lc - 'a';
            /*
             * If the user typed "learn a for apple", abc_word captured "for"
             * and "apple" was silently dropped. Warn them so they know only
             * one word is stored — the brain associates ONE token per letter.
             * The token can be anything: "apple", "ant", "astronaut".
             */
            {
                char dummy[32];
                if (sscanf(line, "learn %3s %31s %31s", abc_letter, abc_word, dummy) == 3)
                    printf("  (Note: storing only '%s' — one word per letter. "
                           "Use: learn %c %s)\n", abc_word, lc, abc_word);
            }

            /* Measure confidence before teaching so we can log how much
             * the brain improved from this single association.
             * (2026-04-18) Use alpha_forward (masked) — raw network_forward
             * is polluted by visual-hidden-neuron noise on text queries. */
            float _emb[MIMIR_EMBEDDING_SIZE], _out[ALPHA_N_OUTPUTS];
            pthread_mutex_lock(&replay->lock);
            alpha_forward(abc_net, li, ALPHA_QUERY_RECALL, _emb, _out);
            pthread_mutex_unlock(&replay->lock);
            float conf_before = 0.0f;
            for (int _i = 0; _i < ALPHA_N_OUTPUTS; _i++)
                if (_out[_i] > conf_before) conf_before = _out[_i];

            struct timespec _t0, _t1;
            clock_gettime(CLOCK_MONOTONIC, &_t0);

            pthread_mutex_lock(&replay->lock);
            AlphaTeachResult result = alpha_teach(abc_net, vocab, li, abc_word);
            pthread_mutex_unlock(&replay->lock);

            clock_gettime(CLOCK_MONOTONIC, &_t1);
            double teach_ms = (_t1.tv_sec - _t0.tv_sec) * 1000.0
                            + (_t1.tv_nsec - _t0.tv_nsec) / 1e6;

            switch (result) {
                case ALPHA_LEARNED: {
                    printf("\n  Learned: %c is for %s\n", (char)('A' + li), abc_word);
                    printf("  (Replay thread will keep practising this now.)\n\n");
                    /* Save immediately so a crash cannot undo the new word */
                    checkpoint_mkdir(ALPHA_BRAIN_PATH);
                    pthread_mutex_lock(&replay->lock);
                    network_save(abc_net, ALPHA_BRAIN_PATH);
                    alpha_vocab_save(vocab, ALPHA_VOCAB_PATH);
                    /* Measure confidence after teaching (masked forward) */
                    alpha_forward(abc_net, li, ALPHA_QUERY_RECALL, _emb, _out);
                    pthread_mutex_unlock(&replay->lock);
                    float conf_after = 0.0f;
                    for (int _i = 0; _i < ALPHA_N_OUTPUTS; _i++)
                        if (_out[_i] > conf_after) conf_after = _out[_i];
                    tlog_learn((char)('A' + li), abc_word, teach_ms,
                               network_neuron_count(abc_net),
                               network_committed_count(abc_net),
                               conf_before, conf_after);

                    /*
                     * (2026-04-16) Inline visual binding: if there's an image
                     * for this word (and any other known word), bind the
                     * visual embedding to its word output now so `vision test`
                     * works immediately rather than only after a manual
                     * pretrain pass.  Joint with text RECALL+VALIDATE inside
                     * vision_train so existing text knowledge is preserved.
                     * Skipped silently when no images are present.
                     */
                    {
                        float *img_data[26];
                        pthread_mutex_lock(&replay->lock);
                        int n_img = vision_load_all(vocab, img_data);
                        if (n_img > 0) {
                            printf("  Binding visual association (%d image%s)...\n",
                                   n_img, n_img == 1 ? "" : "s");
                            vision_train(abc_net, vocab, img_data);
                            network_save(abc_net, ALPHA_BRAIN_PATH);
                        }
                        vision_free_all(img_data);
                        pthread_mutex_unlock(&replay->lock);
                    }
                    break;
                }
                case ALPHA_CONFLICT:
                    printf("\n  [CONFLICT] I already know something different for %c.\n"
                           "  My committed neurons are protecting that knowledge.\n\n",
                           (char)('A' + li));
                    break;
                case ALPHA_REVERIFY:
                    printf("\n  [REVERIFY] You keep insisting on this — "
                           "type 'abc reset' if things have genuinely changed.\n\n");
                    break;
                case ALPHA_VOCAB_FULL:
                    printf("\n  [FULL] Vocabulary is full (%d words max).\n\n",
                           ALPHA_VOCAB_SIZE);
                    break;
                case ALPHA_WRONG_LETTER:
                    printf("\n  [REJECTED] '%s' does not start with '%c'.\n"
                           "  I know the principle: words for a letter must begin with that letter.\n"
                           "  Try: learn %c %c...\n\n",
                           abc_word, (char)('A' + li),
                           lc, (char)('A' + li));
                    break;
            }
            continue;
        }

        /*
         * recall <letter>
         *
         * Ask the brain: "What does this letter stand for?"
         * Shows the answer with a confidence bar. If uncertain (<55%),
         * prompts you to correct it with: learn <letter> <word>
         *
         * Example:  recall a  →  "A is for apple  [89%]"
         */
        if (sscanf(line, "recall %3s", abc_letter) == 1 &&
            strncmp(line, "recall", 6) == 0) {
            char lc = (char)tolower((unsigned char)abc_letter[0]);
            if (lc < 'a' || lc > 'z') {
                printf("  '%s' is not a letter (a–z).\n", abc_letter);
                continue;
            }
            int li = lc - 'a';

            pthread_mutex_lock(&replay->lock);
            const char *ans = alpha_ask(abc_net, vocab, li, ALPHA_QUERY_RECALL);
            /* Confidence bar must match what alpha_ask saw.
             * (2026-04-18) Use alpha_forward (masked) — previously this
             * called raw network_forward, which is polluted by visual
             * hidden neuron noise and reported 0 % even when alpha_ask
             * correctly returned the word. */
            float emb[MIMIR_EMBEDDING_SIZE], out[ALPHA_N_OUTPUTS];
            alpha_forward(abc_net, li, ALPHA_QUERY_RECALL, emb, out);
            float conf = 0.0f;
            for (int i = 0; i < ALPHA_N_OUTPUTS; i++)
                if (out[i] > conf) conf = out[i];
            pthread_mutex_unlock(&replay->lock);

            printf("\n  %c is for…  ", (char)('A' + li));
            if (ans) {
                int filled = (int)(conf * 20.0f);
                printf("[");
                for (int i = 0; i < 20; i++) printf(i < filled ? "=" : " ");
                printf("]  %.0f%%  →  %s\n", conf * 100.0f, ans);
                if (conf < 0.75f)
                    printf("  (not very confident yet — replay is practising)\n");
            } else {
                printf("[not learned yet]\n");
                printf("  Teach me: learn %c <word>\n", (char)('A' + li));
            }
            printf("\n");
            continue;
        }

        /*
         * correct <letter> <word>
         *
         * Ask the brain: "Is this word a valid association for this letter?"
         * The brain runs its VALIDATE principle — a learned, committed rule
         * that says words for a letter must begin with that letter's sound.
         * This is not a hardcoded check: the brain was pre-trained on all 26
         * examples and those neurons are committed (they can never be changed).
         *
         * Examples:
         *   correct a apple   → "Yes, 'apple' starts with A — valid for A"
         *   correct a truck   → "No, 'truck' starts with T — it belongs to T"
         */
        if (sscanf(line, "correct %3s %31s", abc_letter, abc_word) == 2 &&
            strncmp(line, "correct", 7) == 0) {
            char lc = (char)tolower((unsigned char)abc_letter[0]);
            if (lc < 'a' || lc > 'z') {
                printf("  '%s' is not a letter (a-z).\n", abc_letter);
                continue;
            }
            int li = lc - 'a';
            pthread_mutex_lock(&replay->lock);
            int owner = alpha_validate_principle(abc_net, li, abc_word);
            pthread_mutex_unlock(&replay->lock);
            if (owner < 0) {
                printf("\n  Yes — '%s' starts with '%c', valid for %c.\n\n",
                       abc_word, (char)('A' + li), (char)('A' + li));
            } else {
                printf("\n  No — '%s' starts with '%c', so it belongs to %c, not %c.\n\n",
                       abc_word,
                       (char)toupper((unsigned char)abc_word[0]),
                       (char)('A' + owner),
                       (char)('A' + li));
            }
            continue;
        }

        /*
         * next <letter>   — "What letter comes after X?"
         * prev <letter>   — "What letter comes before X?"
         * pos  <letter>   — "What position is X in the alphabet?"
         *
         * These use the pre-trained (committed) sequence knowledge.
         * They should always be correct — committed neurons never forget.
         */
        if (sscanf(line, "next %3s", abc_letter) == 1 &&
            strncmp(line, "next", 4) == 0) {
            char lc = (char)tolower((unsigned char)abc_letter[0]);
            if (lc < 'a' || lc > 'z') { printf("  Not a letter.\n"); continue; }
            int li = lc - 'a';
            pthread_mutex_lock(&replay->lock);
            const char *ans = alpha_ask(abc_net, vocab, li, ALPHA_QUERY_NEXT);
            pthread_mutex_unlock(&replay->lock);
            printf("\n  After %c comes…  %s\n\n",
                   (char)('A' + li), ans ? ans : "[unknown]");
            continue;
        }

        if (sscanf(line, "prev %3s", abc_letter) == 1 &&
            strncmp(line, "prev", 4) == 0) {
            char lc = (char)tolower((unsigned char)abc_letter[0]);
            if (lc < 'a' || lc > 'z') { printf("  Not a letter.\n"); continue; }
            int li = lc - 'a';
            pthread_mutex_lock(&replay->lock);
            const char *ans = alpha_ask(abc_net, vocab, li, ALPHA_QUERY_PREV);
            pthread_mutex_unlock(&replay->lock);
            printf("\n  Before %c comes…  %s\n\n",
                   (char)('A' + li), ans ? ans : "[unknown]");
            continue;
        }

        if (sscanf(line, "pos %3s", abc_letter) == 1 &&
            strncmp(line, "pos", 3) == 0) {
            char lc = (char)tolower((unsigned char)abc_letter[0]);
            if (lc < 'a' || lc > 'z') { printf("  Not a letter.\n"); continue; }
            int li = lc - 'a';
            pthread_mutex_lock(&replay->lock);
            const char *ans = alpha_ask(abc_net, vocab, li, ALPHA_QUERY_POSITION);
            pthread_mutex_unlock(&replay->lock);
            printf("\n  %c is the %s letter.\n\n",
                   (char)('A' + li), ans ? ans : "[unknown]");
            continue;
        }

        /*
         * recite — show everything the brain knows about A–Z.
         * Prints each letter's association with confidence.
         */
        if (strcmp(line, "recite") == 0) {
            pthread_mutex_lock(&replay->lock);
            alpha_recite(abc_net, vocab);
            pthread_mutex_unlock(&replay->lock);
            continue;
        }

        /* Debug: dump VALIDATE accuracy for all 26 letters */
        if (strcmp(line, "validate") == 0) {
            pthread_mutex_lock(&replay->lock);
            printf("\n  ── VALIDATE accuracy ──────────────────────────\n");
            float emb[MIMIR_EMBEDDING_SIZE], out[ALPHA_N_OUTPUTS];
            int v_correct = 0;
            for (int i = 0; i < 26; i++) {
                alpha_forward(abc_net, i, ALPHA_QUERY_VALIDATE, emb, out);
                int best = 0;
                float best_val = out[0];
                for (int j = 1; j < ALPHA_N_OUTPUTS; j++)
                    if (out[j] > best_val) { best_val = out[j]; best = j; }
                bool ok = (best == i);
                if (ok) v_correct++;
                printf("  %c → predict %c [%3.0f%%]  %s\n",
                       'A'+i, 'A'+best, best_val*100, ok ? "OK" : "WRONG");
            }
            printf("  ───────────────────────────────────────────────\n");
            printf("  VALIDATE: %d/26 correct\n\n", v_correct);
            pthread_mutex_unlock(&replay->lock);
            continue;
        }

        /*
         * vision train — load images and train visual associations
         * vision test  — test visual recall (show image → predict word)
         */
        if (strncmp(line, "vision", 6) == 0) {
            char subcmd[32] = "train";
            sscanf(line + 6, " %31s", subcmd);

            /* Load all available word images */
            float *img_data[26];
            pthread_mutex_lock(&replay->lock);
            int n_loaded = vision_load_all(vocab, img_data);

            if (n_loaded == 0) {
                printf("  No images found in data/images/. "
                       "Run: python3 tools/fetch_images.py\n");
            } else {
                printf("  Loaded %d word images.\n", n_loaded);
                if (strcmp(subcmd, "train") == 0) {
                    vision_train(abc_net, vocab, img_data);
                    network_save(abc_net, ALPHA_BRAIN_PATH);
                    printf("  Visual training complete. Brain saved.\n");
                }
                vision_test(abc_net, vocab, img_data);
            }

            vision_free_all(img_data);
            pthread_mutex_unlock(&replay->lock);
            continue;
        }

        /*
         * quiz      — brain tests itself silently (self-quiz / replay check)
         * quiz live — brain asks YOU questions (interactive flashcard mode)
         */
        if (strcmp(line, "quiz") == 0) {
            pthread_mutex_lock(&replay->lock);
            alpha_quiz(abc_net, vocab, false);
            pthread_mutex_unlock(&replay->lock);
            continue;
        }
        if (strcmp(line, "quiz live") == 0) {
            /* Interactive quiz — replay thread prints would interleave with
             * user typing, so pause verbose output during the session.    */
            int was_verbose = replay->verbose;
            replay->verbose = 0;
            pthread_mutex_lock(&replay->lock);
            alpha_quiz(abc_net, vocab, true);
            pthread_mutex_unlock(&replay->lock);
            replay->verbose = was_verbose;
            continue;
        }

        /*
         * quiz choice — multiple-choice word quiz with hippocampus guidance.
         *
         * Presents each taught letter with N word choices (correct + distractors).
         * The brain pattern-matches and picks the highest-scoring candidate.
         * After every pick the correct answer is shown immediately.
         * Wrong answers trigger a retraining burst and are logged to the
         * hippocampus (error count + round — no word content stored).
         * The loop repeats — prioritising historically hard letters — until
         * the brain gets 100% on a full clean pass.
         *
         * Saves both the hippocampus and the ABC brain afterwards so that
         * mistake history and updated weights survive the session.
         */
        if (strcmp(line, "quiz choice") == 0) {
            int was_verbose = replay->verbose;
            replay->verbose = 0;   /* silence replay during the quiz */
            pthread_mutex_lock(&replay->lock);
            alpha_quiz_choice(abc_net, vocab, hippo);
            /* Save updated brain weights and hippocampus after the quiz */
            checkpoint_mkdir(ALPHA_BRAIN_PATH);
            network_save(abc_net, ALPHA_BRAIN_PATH);
            alpha_vocab_save(vocab, ALPHA_VOCAB_PATH);
            pthread_mutex_unlock(&replay->lock);
            checkpoint_mkdir(HIPPO_PATH);
            hippo_save(hippo, HIPPO_PATH);
            replay->verbose = was_verbose;
            continue;
        }

        /*
         * hippo — show the hippocampus mistake log.
         *
         * Displays per-letter error counts, correct counts, current streaks,
         * and the most recent mistakes from the episodic ring buffer.
         * No word content is shown — only which letters were hard and when.
         */
        if (strcmp(line, "hippo") == 0) {
            hippo_print(hippo);
            continue;
        }

        /* ════════════════════════════════════════════════════
         * REPLAY COMMANDS
         * ════════════════════════════════════════════════════ */

        /*
         * replay on      — start background self-training
         * replay off     — stop it
         * replay verbose — print every test the brain runs
         * replay quiet   — go back to silent mode
         * replay status  — show statistics
         */
        if (strcmp(line, "replay on") == 0) {
            replay_start(replay);
            continue;
        }
        if (strcmp(line, "replay off") == 0) {
            replay_stop(replay);
            /* Reinitialise the mutex so replay_start works again later */
            pthread_mutex_init(&replay->lock, NULL);
            continue;
        }
        if (strcmp(line, "replay verbose") == 0) {
            replay->verbose = 1;
            printf("  [Replay] Verbose on — you'll see every test.\n");
            continue;
        }
        if (strcmp(line, "replay quiet") == 0) {
            replay->verbose = 0;
            printf("  [Replay] Quiet mode — background training is silent.\n");
            continue;
        }
        if (strcmp(line, "replay status") == 0) {
            replay_status(replay);
            continue;
        }

        /* abc save — manual checkpoint save */
        if (strcmp(line, "abc save") == 0) {
            checkpoint_mkdir(ALPHA_BRAIN_PATH);
            pthread_mutex_lock(&replay->lock);
            int r1 = network_save(abc_net, ALPHA_BRAIN_PATH);
            int r2 = alpha_vocab_save(vocab, ALPHA_VOCAB_PATH);
            pthread_mutex_unlock(&replay->lock);
            if (r1 == 0 && r2 == 0)
                printf("  Saved: %s  %s\n\n", ALPHA_BRAIN_PATH, ALPHA_VOCAB_PATH);
            else
                printf("  Save failed.\n\n");
            continue;
        }

        /* ── help ── */
        if (strcmp(line, "help") == 0) {
            printf("\n");
            printf("  ── Alphabet commands ───────────────────────────────\n");
            printf("    learn <letter> <word>    Teach: 'learn a apple'\n");
            printf("    learn all                Bulk-teach one bank word per letter\n");
            printf("    correct <letter> <word>  Ask brain if word is valid for letter\n");
            printf("    recall <letter>          Ask: 'recall a' → apple\n");
            printf("    next <letter>            'next a' → B\n");
            printf("    prev <letter>            'prev b' → A\n");
            printf("    pos  <letter>            'pos a'  → 1st\n");
            printf("    recite                   Show all A-Z knowledge\n");
            printf("    quiz                     Brain tests itself\n");
            printf("    quiz live                Brain quizzes YOU\n");
            printf("    quiz choice              Multi-choice from UNSEEN bank words\n");
            printf("    hippo                    Show hippocampus mistake log\n");
            printf("    vision train              Train visual associations from images\n");
            printf("    vision test               Test visual recall (image → word)\n");
            printf("    abc save                 Save ABC brain now\n");
            printf("\n");
            printf("  ── Replay commands ─────────────────────────────────\n");
            printf("    replay on/off            Start/stop background training\n");
            printf("    replay verbose/quiet     Show/hide what brain is practising\n");
            printf("    replay status            Show training statistics\n");
            printf("\n");
            printf("  ── Gate commands ───────────────────────────────────\n");
            printf("    predict <x1> <x2>              Show AND/OR/XOR outputs\n");
            printf("    predict <and|or|xor> <x1> <x2> Show one gate\n");
            printf("    train <x1> <x2> <a> <o> <x>    Teach all 3 gates\n");
            printf("    train <gate> <x1> <x2> <t>      Teach one gate\n");
            printf("    grow / neurons / info / commit / reset\n");
            printf("\n");
            printf("    quit / exit              Save and leave\n");
            printf("\n");
            continue;
        }

        /* ── quit / exit ── */
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) {
            printf("\n  Saving and saying goodbye...\n");
            checkpoint_mkdir(ALPHA_BRAIN_PATH);
            pthread_mutex_lock(&replay->lock);
            network_save(abc_net, ALPHA_BRAIN_PATH);
            alpha_vocab_save(vocab, ALPHA_VOCAB_PATH);
            pthread_mutex_unlock(&replay->lock);
            checkpoint_mkdir(HIPPO_PATH);
            hippo_save(hippo, HIPPO_PATH);
            printf("  ABC brain saved to %s\n", ALPHA_BRAIN_PATH);
            printf("  Hippocampus saved to %s\n\n", HIPPO_PATH);
            break;
        }

        printf("  Unknown command: '%s'  (type 'help' for list)\n", line);
    }
}

/* Declared in training.c */
extern void run_xor_benchmark(void);

int main(int argc, char *argv[]) {
    /* Parse flags: --force-train retrains even if a checkpoint exists */
    bool force_train = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--force-train") == 0)
            force_train = true;
    }

    printf("\n");
    printf("  +--------------------------------------+\n");
    printf("  |          M I M I R  v0.1.0           |\n");
    printf("  |   The Well of Wisdom, from scratch   |\n");
    printf("  +--------------------------------------+\n");
    printf("\n");

    /* Open the persistent training log for this session.
     * All learn, replay, benchmark, and shutdown events are appended here. */
    tlog_open();

    /* --- Step 1: Neurogenesis (or load from checkpoint) --- */
    printf("Step 1: Neurogenesis -- The Brain Grows\n");
    printf("----------------------------------------\n\n");

    Network brain;
    bool from_checkpoint = false;

    if (!force_train) {
        Network *saved = network_load(CHECKPOINT_PATH);
        if (saved) {
            brain = *saved;
            free(saved);   /* free the outer shell only; network_free owns the rest */
            from_checkpoint = true;
            printf("  Loaded trained brain from checkpoint: %s\n", CHECKPOINT_PATH);
            printf("  (%d neurons, %d committed — skipping training)\n",
                   network_neuron_count(&brain), network_committed_count(&brain));
            printf("  Run with --force-train to retrain from scratch.\n\n");
            network_print(&brain);
        }
    }

    if (!from_checkpoint) {
        if (force_train)
            printf("  --force-train: ignoring checkpoint, retraining from scratch.\n\n");
        brain = test_network_gates();
        /* Persist so the next run skips training */
        checkpoint_mkdir(CHECKPOINT_PATH);
        if (network_save(&brain, CHECKPOINT_PATH) == 0)
            printf("\n  [Brain saved to %s — future runs will skip training]\n", CHECKPOINT_PATH);
        else
            printf("\n  [Warning: could not save checkpoint to %s]\n", CHECKPOINT_PATH);
    }

    /* --- Step 2: Training method benchmark --- */
    /*
     * Runs every startup. Trains fresh single-output networks for AND, OR,
     * and XOR and records epochs, wall time, memory, and accuracy for both
     * Brain-native and Backprop. The numbers are repeatable measurements you
     * can use to track how the learning algorithm improves over time.
     *
     * The benchmark networks are SEPARATE from the main gate brain — they
     * do not affect the committed checkpoint in data/mimir.brain.
     */
    printf("\nStep 2: Training Method Benchmark -- How Should We Learn?\n");
    printf("----------------------------------------------------------\n\n");
    run_xor_benchmark();

    /* --- Step 3: Alphabet Brain --- */
    printf("\nStep 3: Alphabet Brain -- Learning A-Z\n");
    printf("---------------------------------------\n\n");

    /*
     * The ABC brain takes MIMIR_EMBEDDING_SIZE inputs (the sensor embedding)
     * and produces ALPHA_N_OUTPUTS (26) outputs — one per possible answer.
     *
     * 32 active hidden neurons + 32 dormant pool gives plenty of capacity
     * for 26 letter associations, plus the pre-trained sequence/position
     * knowledge. The pool neurons are recruited by neurogenesis if needed.
     */
    Network abc_brain;
    AlphaVocab vocab;
    Hippocampus hippo;
    bool abc_from_checkpoint = false;

    if (!force_train) {
        Network *saved_abc = network_load(ALPHA_BRAIN_PATH);
        if (saved_abc && alpha_vocab_load(&vocab, ALPHA_VOCAB_PATH) == 0) {
            abc_brain = *saved_abc;
            free(saved_abc);
            abc_from_checkpoint = true;
            printf("  Loaded ABC brain from checkpoint: %s\n", ALPHA_BRAIN_PATH);
            printf("  Vocabulary: %d words known.\n\n", vocab.n_words);
        } else {
            if (saved_abc) { network_free(saved_abc); free(saved_abc); }
        }
    }

    if (!abc_from_checkpoint) {
        printf("  Creating new ABC brain (n_inputs=%d, n_outputs=%d)...\n",
               MIMIR_EMBEDDING_SIZE, ALPHA_N_OUTPUTS);
        random_seed(99);
        /*
         * HDC hidden: MIMIR_HDC_HIDDEN (=256) neurons with ACT_STEP and
         * fixed random Gaussian weights across all 128 input dims.  Unified
         * text+vision representation — each (letter, query) and each Gabor
         * image produces a distinct binary signature; the output layer
         * delta rule maps signatures → word class in a few hundred epochs.
         * No modality separation, no dual bias, no neurogenesis needed.
         * Math-verified in sandbox/vision_math_2.py (100% strict, conf 1.00
         * on both text and vision with a single shared hidden layer).
         */
        abc_brain = network_create_with_pool(
            MIMIR_EMBEDDING_SIZE, MIMIR_HDC_HIDDEN, 0, ALPHA_N_OUTPUTS, ACT_SIGMOID);
        network_hdc_init_hidden(&abc_brain);
        alpha_vocab_init(&vocab);

        /*
         * HDC brain: no Hebbian pretraining.
         *
         * The hidden layer is a fixed random Gaussian projection with step
         * activation — it is ALREADY a complete feature extractor.  Running
         * alpha_pretrain_sequence (Hebbian + BCM + WTA through train_step_brain)
         * would overwrite the projection and collapse the representation back
         * to 8 usable dimensions, recreating the exact capacity bottleneck HDC
         * was introduced to fix (see sandbox/vision_math_2.py).
         *
         * Sequence / position / recall / validate facts are learned instead
         * by the output-layer delta rule inside alpha_retrain_all_known when
         * the user teaches words (or by the replay thread's alpha_delta_rescue).
         * The HDC projection supplies enough linear separability that a
         * single-layer delta rule converges in a few hundred epochs.
         */

        checkpoint_mkdir(ALPHA_BRAIN_PATH);
        network_save(&abc_brain, ALPHA_BRAIN_PATH);
        alpha_vocab_save(&vocab, ALPHA_VOCAB_PATH);
        printf("  ABC brain saved. Ready to learn words.\n\n");
    }

    /* Load or create the hippocampus — episodic mistake memory.
     * The hippocampus persists across sessions so the brain remembers
     * which letters it historically struggled with, even after a restart.
     * It contains ONLY error metadata (letter indices + round numbers),
     * never word content — the network must re-derive semantics itself. */
    if (hippo_load(&hippo, HIPPO_PATH) == 0) {
        printf("  Loaded hippocampus from %s  "
               "(total rounds: %d)\n\n", HIPPO_PATH, hippo.total_rounds);
    } else {
        hippo_init(&hippo);
        printf("  No hippocampus found — starting fresh error memory.\n\n");
    }

    /* Record startup state: how many neurons each brain has and how many
     * are already committed (from prior sessions or just-completed pre-training). */
    tlog_startup(network_neuron_count(&brain),    network_committed_count(&brain),
                 network_neuron_count(&abc_brain), network_committed_count(&abc_brain));

    /*
     * (Added 2026-04-11) Loud warning when the ABC brain has zero committed
     * neurons.  This is the precondition that produced the "learn b breaks
     * recall a" bug: with nothing committed, the hidden layer is fully
     * plastic, alpha_delta_rescue's "letters live in the hidden layer"
     * assumption stops holding, and word associations decay under any
     * subsequent training.  If you see this message after a fresh start,
     * pre-training failed to converge — investigate alpha_pretrain_sequence
     * before teaching anything.  If you see it after loading an old
     * checkpoint, the checkpoint pre-dates the convergence fix and you
     * should `abc reset` (or delete data/mimir_abc.brain) to rebuild.
     */
    if (network_committed_count(&abc_brain) == 0) {
        printf("\n  ⚠  ABC brain has 0 committed neurons.\n");
        printf("     Word associations WILL drift under replay.\n");
        printf("     If this is a fresh run, pre-training failed to converge.\n");
        printf("     If this is a loaded checkpoint, run `abc reset` to rebuild.\n\n");
    }

    /* --- Step 4: Replay thread --- */
    printf("Step 4: Starting Replay (Background Self-Training)\n");
    printf("---------------------------------------------------\n\n");

    /*
     * The replay thread runs for the entire lifetime of the CLI session.
     * It continuously rehearses known associations, strengthening weak
     * ones and leaving strong (committed) ones alone.
     */
    ReplayState replay;
    replay_init(&replay, &abc_brain, &vocab);
    replay_start(&replay);

    /* --- Step 5: Interactive CLI --- */
    printf("\nStep 5: Interactive CLI -- Talk to the Brain\n");
    printf("----------------------------------------------\n");

    run_cli(&brain, &abc_brain, &vocab, &replay, &hippo);

    /* Clean shutdown: stop replay thread before freeing memory */
    replay_stop(&replay);

    /* Write final session summary to the training log then close the file. */
    tlog_shutdown(replay.cycles,
                  replay.cycles > 0
                      ? (double)replay.correct / (double)replay.cycles
                      : 0.0,
                  replay.corrected);
    tlog_close();

    network_free(&brain);
    network_free(&abc_brain);
    return 0;
}
