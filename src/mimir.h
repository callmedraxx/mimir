/*
 * MIMIR - A CPU-native language model built from scratch
 * Named after the Norse keeper of the Well of Wisdom
 *
 * "Odin gave his eye for wisdom. We give our cycles."
 *
 * ════════════════════════════════════════════════════════════════════
 * THE GOAL — Read this before optimising anything.
 * ════════════════════════════════════════════════════════════════════
 *
 * We are NOT trying to beat backprop at speed or accuracy on a fixed
 * benchmark. Backprop wins that race on purpose-built hardware.
 *
 * We are building a brain that:
 *
 *   1. ADAPTS IN REAL TIME.
 *      Not batch training over a frozen dataset — single experiences
 *      update the network immediately, the way a human adjusts mid-
 *      conversation. The network is always live, always changing.
 *
 *   2. QUESTIONS ITSELF.
 *      When the network is wrong it must know it is wrong. Stall
 *      detection is the seed of this: "I have been trying for 1000
 *      epochs and I am still failing — I need to grow." Over time
 *      this self-awareness should extend to confidence estimation,
 *      uncertainty signalling, and the ability to say "I do not know"
 *      rather than hallucinating a confident wrong answer.
 *
 *   3. NEVER FORGETS.
 *      When training converges, neurons auto-commit — no manual step.
 *      Committed weights are frozen (train_step_brain skips them).
 *      New contradictory data is rejected by network_check_data.
 *      New tasks get new neurons on top; old circuits are untouched.
 *
 *   4. GETS EASIER TO TEACH OVER TIME.
 *      A baby learns to walk in months. An adult learns a new dance
 *      in minutes — not because their neurons fire faster, but because
 *      they already have millions of committed circuits to build on.
 *      Task 1 is slow. Task 5 should be fast because tasks 1-4 built
 *      the foundation. This is the metric that matters — not epoch
 *      count on a single isolated benchmark.
 *
 * Backprop cannot do this. It trains on a fixed architecture for a
 * fixed task. Teach it something new and you either retrain from
 * scratch (forgetting everything) or freeze and add adapters (manual
 * surgery). It has no concept of a neuron maturing, committing, or
 * contributing its circuits to a later task. It cannot question itself
 * or decide to grow. It does exactly what you tell it, nothing more.
 *
 * The benchmark exists only to verify we are not wildly inefficient.
 * If brain-native needs 10× more epochs on task 1 but solves task 5
 * in 1/10th the epochs because the foundation is already laid — and
 * does it without forgetting task 1 — that is the win.
 * ════════════════════════════════════════════════════════════════════
 *
 * Step 1: The Perceptron — the single neuron that starts it all
 */

#ifndef MIMIR_H
#define MIMIR_H

/*
 * Request POSIX.1-2008 extensions before any system header is included.
 * This unlocks strcasecmp(), strdup(), and other POSIX functions that
 * are not part of the C11 standard but are available on Linux/macOS.
 * Must appear before the first #include — the C library reads this flag
 * during header preprocessing.
 */
#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>   /* ReplayState uses pthread_mutex_t and pthread_t */
#include <time.h>

/*
 * MAX_OUTPUTS — largest output layer any Network in this process will have.
 * Stack-allocated output buffers in the CLI and tests are sized by this.
 * 32 covers the current gate network (3 outputs) and the planned letter
 * network (26 outputs) with room to spare. Raise it if you add an output
 * head larger than 32 — it is just a stack-buffer ceiling, not a limit
 * baked into the Network struct.
 */
#define MAX_OUTPUTS 32

// ============================================================
// THE PERCEPTRON - Where it all begins
// ============================================================

typedef enum {
    ACT_STEP,      // Classic step function (Rosenblatt's original)
    ACT_SIGMOID,   // Smooth, differentiable
    ACT_RELU,      // Modern default
    ACT_HDC_BIT,   /* Bit-packed HDC: weights are ±1 stored as 1 bit each.
                    * z = sum_{d}(bit?+1:-1) * input[d];  output = z>0 ? 1 : 0.
                    * 32× less memory than ACT_STEP with float weights.
                    * When set, n->weights points to ((n_weights+31)/32) uint32_t
                    * words, cast through float* for struct compatibility. */
} Activation;

typedef struct {
    float *weights;    // one weight per input
    float bias;        // the bias term
    int n_inputs;      // number of inputs
    Activation act;    // activation function
} Perceptron;

// --- Perceptron API ---
Perceptron perceptron_create(int n_inputs, Activation act);
float perceptron_forward(Perceptron *p, const float *inputs);
void perceptron_train(Perceptron *p, const float *inputs, float target, float lr);
void perceptron_free(Perceptron *p);
void perceptron_print(Perceptron *p);

// --- Activation functions ---
float activate(float x, Activation act);
float activate_derivative(float x, Activation act);

// --- Random init ---
void random_init(float *data, int n, float scale);

// --- Tests ---
int run_tests(void);

// ============================================================
// NEUROGENESIS - Dynamic network that grows like a biological brain
// ============================================================
//
// In biology, neurogenesis is the process of creating new neurons.
// The brain doesn't start fully wired — it grows, prunes, and
// reorganizes throughout life. We mimic this:
//
//   1. Neurons are BORN when the network struggles (high error)
//   2. Newborn neurons are IMMATURE — they contribute almost nothing
//   3. Over epochs, they MATURE and integrate into the network
//   4. Neurons that helped learn something become COMMITTED — permanent
//   5. Growth is EXPONENTIAL: 1, 2, 4, 8, 16... neurons per generation
//      because learning goes from simple → harder → complex
//
// KEY RULE: Committed neurons NEVER die. They hold knowledge.
// Even when inactive (the network isn't using that knowledge right now),
// they persist. New challenges get NEW neurons — old knowledge stays.

/*
 * Neuron lifecycle states — mirrors biological neuron development.
 *
 * In the brain:
 *   Stem cell → Neuroblast → Immature neuron → Mature neuron
 *   (and ~50% die before reaching maturity — apoptosis)
 *
 * In our model:
 *   DORMANT → IMMATURE → MATURE → COMMITTED
 *   (neurons that never prove useful can be recycled)
 */
typedef enum {
    NEURON_DORMANT,      /* Unborn. Exists only as free memory in the pool. */
    NEURON_IMMATURE,     /* Just born. Weights exist but output is scaled   */
                         /* near-zero by maturity. Gradually integrates.    */
    NEURON_MATURE,       /* Fully active. Contributing to network output.   */
    NEURON_COMMITTED,    /* Has learned knowledge. PERMANENT. Never dies.   */
                         /* Like a neuron that's part of a stable circuit.  */
} NeuronState;

/*
 * A single neuron in the dynamic network.
 *
 * Unlike our original Perceptron struct (which was a standalone unit),
 * this Neuron is designed to live INSIDE a Layer, connected to other
 * neurons. It carries its own lifecycle state for neurogenesis.
 *
 * BIOLOGICAL PARALLEL:
 * - weights  → synaptic strengths (how strongly it listens to each input)
 * - bias     → intrinsic excitability (how easily it fires on its own)
 * - maturity → developmental stage (immature neurons in the brain have
 *              weaker synapses and are less excitable)
 * - activity → how often/strongly it fires (active neurons survive,
 *              inactive ones are candidates for pruning)
 */
typedef struct {
    float *weights;       /* Connection weights to previous layer neurons    */
    float bias;           /* Bias term                                       */
    int n_weights;        /* Number of weights (= previous layer size)       */

    /* --- Neurogenesis state --- */
    float maturity;       /* 0.0 (just born) → 1.0 (fully mature).          */
                          /* Output is multiplied by this, so immature       */
                          /* neurons contribute almost nothing at first.     */
    float activity;       /* Exponential moving average of |output|.         */
                          /* Tracks how much this neuron contributes.        */
                          /* High activity = useful. Low = candidate to die. */
    int age;              /* Epochs since birth. Used to enforce minimum     */
                          /* maturation time before judging usefulness.      */
    int id;               /* Unique lifetime ID from the pool. Never reused. */
                          /* Like a serial number — even dead neurons keep   */
                          /* their ID in the historical record.              */
    NeuronState state;    /* Current lifecycle state                         */
    Activation act;       /* Activation function                             */

    /* --- Cached forward pass values (needed for training) --- */
    float last_z;         /* Pre-activation value from last forward pass.    */
                          /* z = sum(w_i * x_i) + bias. Cached so we don't  */
                          /* recompute during backward pass.                 */
    float last_output;    /* Post-activation output from last forward pass.  */

    /* --- Brain-native learning state (persistent, zero per-step cost) --- */
    float theta;          /* BCM sliding modification threshold.             */
                          /* Rises when neuron is overactive (LTD zone),     */
                          /* falls when underactive (LTP zone).              */
                          /* dw = pre * (post - theta) * modulator           */
                          /* This replaces fixed weight decay with a         */
                          /* per-neuron adaptive homeostatic mechanism.      */
                          /* Bio: Bienenstock-Cooper-Munro 1982.             */
    float mean_out;       /* Exponential moving average of signed output.    */
                          /* Used with activity (EMA of |output|) to        */
                          /* estimate output variance:                       */
                          /*   variance ≈ activity - |mean_out|              */
                          /* High variance = neuron responds differently     */
                          /* to different inputs = informative = learns fast */
                          /* Low variance = neuron always fires same = slow  */
                          /* Bio: acetylcholine attention gate (ACh).        */

    /* --- Modality tag (visual neurogenesis) --- */
    uint8_t is_visual;    /* 1 = this hidden neuron is dedicated to visual    */
                          /* input and has zeroed text-region weights.        */
                          /* Text neurons have is_visual=0 and zeroed         */
                          /* visual-region weights. The weight structure       */
                          /* itself enforces modality gating — text neurons    */
                          /* are silent on visual input and vice versa.        */

    /* --- Dual bias for modality-separated output heads (2026-04-18) ---
     * `bias` is the text/default bias (used by alpha_forward and by any
     * non-visual training path).  `visual_bias` is used by vision forward
     * and vision training.  Previously both modalities shared `bias`, so
     * every text rescue shifted the operating point for vision predictions
     * (and vice versa).  Giving each modality its own output-layer bias
     * is the minimal "separate heads" change that eliminates the cross-
     * modal drift without duplicating the whole output layer.             */
    float visual_bias;
} Neuron;

/*
 * A layer of neurons.
 *
 * Neurons within a layer are stored contiguously in memory (array, not
 * linked list) for cache efficiency. Each neuron in this layer receives
 * input from ALL neurons in the previous layer (fully connected).
 *
 * The layer can GROW dynamically — when neurogenesis adds neurons,
 * we realloc the arrays and initialize the new neurons as IMMATURE.
 *
 * BIOLOGICAL PARALLEL:
 * A layer is like a cortical region — a collection of neurons that
 * process information at the same level of abstraction. Layer 1 might
 * detect edges, Layer 2 detects shapes, Layer 3 detects objects, etc.
 */
typedef struct {
    Neuron *neurons;      /* Contiguous array of neurons in this layer       */
    float *outputs;       /* Cached outputs from last forward pass.          */
                          /* outputs[i] = neurons[i].last_output.            */
                          /* Stored separately for fast access by next layer */
    int count;            /* Number of active neurons in this layer          */
    int capacity;         /* Allocated slots (count <= capacity). We over-   */
                          /* allocate to avoid realloc on every single add.  */
    int input_size;       /* Number of inputs each neuron receives           */
                          /* (= previous layer's neuron count, or n_inputs   */
                          /* for the first layer)                            */
} Layer;

/*
 * NeuronPool — the "stem cell niche" of our network.
 *
 * In the brain, the subventricular zone and hippocampal dentate gyrus
 * contain neural stem cells that can produce new neurons on demand.
 * Our pool serves the same role: it's the source of all new neurons.
 *
 * For now this is a statistics tracker. The actual memory allocation
 * happens in the Layer (realloc). When we scale to billions of neurons,
 * this becomes a proper chunk-based pool allocator to avoid per-neuron
 * malloc overhead (~40 bytes wasted per malloc call on most platforms).
 *
 * SCALING NOTE: At 1 billion neurons, individual malloc = ~40 GB overhead
 * just from allocation headers. Pool allocator with 1024-neuron chunks =
 * ~40 MB overhead. That's a 1000x difference. We'll upgrade when needed.
 */
typedef struct {
    int total_created;    /* Lifetime neuron births. Only goes up.           */
    int total_active;     /* Currently alive (IMMATURE + MATURE + COMMITTED) */
    int total_committed;  /* Knowledge-holding neurons. The "permanent" ones */
    int total_dead;       /* Neurons that were born but later recycled       */
} NeuronPool;

/* ── Conflict log types (used inside Network struct below) ─────────── */

/* How many times the same conflict must appear before escalating to REVERIFY */
#define CONFLICT_REVERIFY_AT  3
/* Max unique conflict patterns tracked per network at one time */
#define CONFLICT_LOG_SIZE     8

/*
 * A single entry in the per-network conflict log.
 * Tracks how many times a specific (input, target) pair has
 * conflicted with committed knowledge.
 */
typedef struct {
    uint64_t input_hash;   /* FNV-1a hash of the input float array          */
    uint64_t target_hash;  /* FNV-1a hash of the targets float array        */
    int      count;        /* times this exact conflict was observed         */
} ConflictRecord;

/*
 * The Network — a living, growing brain.
 *
 * This is the central struct. It manages layers, the neuron pool,
 * and neurogenesis parameters. Unlike a traditional neural network
 * with fixed architecture, this network discovers its own shape.
 *
 * STRUCTURE (evolves during training):
 *
 *   Initially:      [Input] → [Output]         (just 1 output neuron)
 *
 *   After growth:   [Input] → [Hidden] → [Output]    (hidden layer added)
 *
 *   After more:     [Input] → [Hidden(8)] → [Output]  (hidden layer grew)
 *
 * The output layer is always the LAST layer. Hidden layers are inserted
 * before it. When hidden layers grow, the output layer's weight arrays
 * are reallocated to match the new input dimension.
 *
 * EXPONENTIAL GROWTH:
 * Each neurogenesis event adds 2^generation neurons:
 *   Gen 0: +1 neuron    (simple problems need few neurons)
 *   Gen 1: +2 neurons
 *   Gen 2: +4 neurons
 *   Gen 3: +8 neurons
 *   ...
 *   Gen 20: +1,048,576 neurons  (complex problems need many)
 *   Gen 30: +1,073,741,824 neurons (~1 billion)
 *
 * This mirrors how learning complexity scales: learning the alphabet
 * is simple (few neurons), learning physics requires massive networks.
 */
typedef struct {
    Layer *layers;        /* Dynamic array of layers                         */
    int n_layers;         /* Current number of layers                        */
    int n_layers_cap;     /* Capacity of layers array                        */

    NeuronPool pool;      /* Neuron birth/death statistics                   */

    /* --- Network topology --- */
    int n_inputs;         /* Input dimension (fixed at creation)             */
    int n_outputs;        /* Output dimension (fixed at creation)            */
    Activation default_act; /* Default activation for new neurons            */

    /* --- Neurogenesis parameters --- */
    int growth_gen;       /* Exponential growth generation counter.          */
                          /* Each neurogenesis event adds 2^growth_gen       */
                          /* neurons, then increments this counter.          */
    float maturation_rate;/* How fast neurons mature per epoch.              */
                          /* 0.05 means 20 epochs to full maturity.          */
                          /* Biological parallel: real neurons take weeks    */
                          /* to months to fully integrate into circuits.     */

    /* --- Brain-native training state --- */
    float rpe_baseline;   /* Running average of squared error.               */
                          /* Used to compute Reward Prediction Error (RPE):  */
                          /*   modulator = error - expected_error            */
                          /* where expected_error = sign(err)*sqrt(baseline) */
                          /* Stored in Network (not static in fn) so each    */
                          /* Network instance has its own independent state. */

    /* --- Conflict / reverification log --- */
    ConflictRecord conflict_log[CONFLICT_LOG_SIZE]; /* ring of recent conflicts */
    int            n_conflict_records;  /* entries currently in log (0..CONFLICT_LOG_SIZE) */
} Network;

// --- Network API ---
Network network_create(int n_inputs, int n_outputs, Activation act);
void    network_free(Network *net);
void    network_forward(Network *net, const float *inputs, float *outputs);
float   network_train_step(Network *net, const float *inputs, float target, float lr);
void    network_auto_train(Network *net, const float *inputs, const float *targets,
                           int n_samples, int max_epochs, float lr);
int     network_auto_train_v(Network *net, const float *inputs, const float *targets,
                              int n_samples, int max_epochs, float lr, int verbose);

// --- Neurogenesis API ---
int     network_neurogenesis(Network *net);            /* Trigger growth (verbose)          */
int     network_neurogenesis_v(Network *net, int v);   /* Trigger growth (v=0: silent)      */
void    network_mature(Network *net);           /* Age all neurons by one epoch          */
void    network_commit(Network *net);           /* Commit all mature neurons (permanent) */
void    network_commit_hidden(Network *net);    /* Commit hidden layers only (output stays plastic) */
int     network_commit_output(Network *net);    /* Commit output layer (returns count of newly committed) */

// --- Network info ---
int     network_neuron_count(Network *net);     /* Total active neurons                  */
int     network_committed_count(Network *net);  /* Neurons holding knowledge             */
void    network_print(Network *net);            /* Display network state                 */

// --- Confidence and conflict detection ---
/*
 * How confident is the network in its current prediction for these inputs?
 * Returns [0.0, 1.0]:
 *   0.0 = output exactly on the decision boundary (0.5) — completely uncertain
 *   1.0 = output fully saturated (0.0 or 1.0) — maximally confident
 * Formula: |output - 0.5| * 2
 */
float network_confidence(Network *net, const float *inputs);

/*
 * Verdict returned by network_check_data: should we learn this sample?
 */
typedef enum {
    VERDICT_LEARN,     /* Data is consistent with committed knowledge, or network   */
                       /* has no committed neurons yet — safe to learn.             */
    VERDICT_CONFLICT,  /* New label contradicts a high-confidence committed output. */
                       /* Sample rejected to protect committed knowledge.           */
                       /* Likely a bad/mislabelled data point.                      */
    VERDICT_REVERIFY,  /* Same conflict has appeared CONFLICT_REVERIFY_AT times.   */
                       /* Too consistent to be noise — possible concept drift       */
                       /* (the world changed). Network flags for human review.      */
                       /* Currently: warns loudly but still rejects. Future: allow  */
                       /* selective uncommit + retrain path.                        */
} TrainVerdict;

/*
 * Check whether a new (inputs, target) sample conflicts with committed knowledge.
 *
 * conflict_threshold: confidence level above which a contradiction is flagged.
 *   0.7 → flag if output is above 0.85 or below 0.15 and label disagrees.
 *   0.9 → only flag near-saturated (very high confidence) disagreements.
 *   Recommended default: 0.8
 *
 * Returns VERDICT_CONFLICT when ALL THREE hold:
 *   1. Network has at least one committed neuron (knowledge is locked in).
 *      Without this, random initial weights that happen to be confident
 *      would block valid training data before any learning has happened.
 *      network_auto_train_v auto-commits on LEARNED, so by the time real
 *      user interaction happens, committed neurons always exist.
 *   2. Confidence > conflict_threshold (output > 0.9 or < 0.1 at 0.8).
 *   3. Network's prediction disagrees with the new target.
 *
 * Escalates to VERDICT_REVERIFY when the SAME (input, target) pair has
 * triggered CONFLICT_REVERIFY_AT conflicts — systematic contradiction,
 * not noise.
 */
TrainVerdict network_check_data(Network *net, const float *inputs,
                                const float *targets, int n_targets,
                                float conflict_threshold);

/* Clear the conflict log (call after intentional reset+retrain) */
void network_clear_conflicts(Network *net);

// ============================================================
// CHECKPOINT — Persist trained network state to disk
// ============================================================
//
// Saved to CHECKPOINT_PATH (default: data/mimir.brain).
// make clean  → removes build/ only (knowledge survives)
// make clean-all → also removes the checkpoint (full reset)
// --force-train flag → retrain even if checkpoint exists

#define CHECKPOINT_PATH "data/mimir.brain"

/*
 * Save net to path. Returns 0 on success, -1 on I/O error.
 * Creates the parent directory if needed (one level only).
 */
int      network_save(const Network *net, const char *path);

/*
 * Load a network from path. Returns a heap-allocated Network * on success
 * (caller must network_free() then free() it), or NULL if the file does not
 * exist, is corrupt, or has wrong magic.
 */
Network *network_load(const char *path);

/* Create the parent directory of path (silently ignores EEXIST). */
void     checkpoint_mkdir(const char *path);

// --- Pool API ---
void    pool_init(NeuronPool *pool);
int     pool_birth(NeuronPool *pool);           /* Register a birth, return unique ID    */
void    pool_kill(NeuronPool *pool);            /* Register a death                      */
void    pool_print(const NeuronPool *pool);

// --- Random utilities (exposed for training methods) ---
void    random_seed(uint64_t seed);             /* Set RNG state for reproducibility     */
float   random_uniform_f(void);                 /* Random float in [0, 1)                */

// --- Network factory for benchmarking ---
Network network_create_with_hidden(int n_inputs, int n_hidden, int n_outputs, Activation act);

// --- Network factory: pre-built hidden layer + dormant pool ---
// n_active: neurons ready to train from epoch 1 (MATURE)
// n_pool:   silent pre-allocated neurons recruited by neurogenesis
Network network_create_with_pool(int n_inputs, int n_active, int n_pool,
                                  int n_outputs, Activation act);

// --- HDC (hyperdimensional) hidden init ---
// Freeze hidden layer as a random binary projection: h_i = step(w · x),
// w ~ N(0,1), bias=0, MATURE.  Output layer unchanged, learned via delta
// rule on top of the {0,1} signatures.  Unified across text and vision.
#define MIMIR_HDC_HIDDEN 256
void network_hdc_init_hidden(Network *net);

// ============================================================
// TRAINING METHODS — Five ways to train a neural network
// ============================================================
//
// Each function trains the network on ONE input-target pair and
// returns the squared error. All methods use the same Network
// struct so they can be benchmarked fairly on identical architectures.

/* Method 1: Brain-native — Three-factor Hebbian with global modulator.
 * How the actual brain does it. Each synapse updates using:
 *   dw = lr * pre * post * modulator
 * where modulator is a global reward/punishment signal (like dopamine).
 * Multi-output version: takes a targets array and n_targets.
 * Returns mean squared error across all outputs. */
float train_step_brain(Network *net, const float *inputs, const float *targets,
                       int n_targets, float lr);

/* Method 2: Backpropagation — The industry standard.
 * Exact error gradients flow backward through every layer via chain rule.
 * Fastest convergence, highest memory cost. */
float train_step_backprop(Network *net, const float *inputs, float target, float lr);

/* Benchmark result for comparing training methods */
typedef struct {
    const char *name;
    int epochs_to_solve;    /* -1 if not solved within max_epochs             */
    double wall_time_ms;    /* Wall clock time to solve (or to max_epochs)    */
    int memory_bytes;       /* Extra memory beyond weights (per-method cost)  */
    int total_neurons;      /* Neurons used                                   */
    float final_error;      /* Average squared error at end of training       */
    int accuracy;           /* Correct out of 4 samples                       */
    bool solved;            /* True if all 4 samples correct                  */
} BenchmarkResult;

/* Run the logic gate benchmark: Brain-native vs Backprop on AND, OR, XOR */
void run_xor_benchmark(void);

/*
 * Forward declaration of Sensor so the Alphabet API below can reference
 * Sensor* in alpha_sensor_encode's signature. The full struct definition
 * appears in the Sensor Architecture section later in this header.
 * C allows pointers to incomplete types — the size of the struct is not
 * needed here, only its name.
 */
typedef struct Sensor Sensor;

// ============================================================
// ALPHABET — Letter-to-word associations with smart querying
// ============================================================
//
// The alphabet module teaches the brain letter associations ("A is for
// Apple") while simultaneously knowing the alphabet sequence ("after A
// comes B") and ordinal positions ("A is the 1st letter").
//
// HOW QUERY TYPES WORK:
//
// The human brain stores a letter in multiple overlapping circuits at
// once. Asking "what does A stand for?" activates different synaptic
// pathways than asking "what comes after A?" — but both use the SAME
// committed neurons, steered by the context of the question.
//
// We implement this by including the QUERY TYPE in the raw sensor input.
// The raw input for the alphabet sensor is:
//
//   [0..25]  one-hot letter      (which letter are we asking about?)
//   [26..29] one-hot query type  (what are we asking?)
//   Total: 30 floats → encoded to MIMIR_EMBEDDING_SIZE by the sensor
//
// (A + RECALL) and (A + NEXT) produce DIFFERENT embeddings and therefore
// DIFFERENT outputs from the same core network. The network learns that
// "A in recall context" → "Apple" and "A in next context" → "B".
//
// TRAINING ORDER matters — mirrors how humans learn:
//   1. Sequence is pre-trained first (A→B→C…) and committed.
//      A child learns the alphabet song before "A is for Apple".
//   2. Associations are taught on top without disturbing the sequence.
//      Committed sequence neurons are frozen — teaching "apple" cannot
//      accidentally break "after A comes B".
//
// QUIZ MODE:
//   The model tests itself by forming questions from its own knowledge,
//   running a forward pass to get its predicted answer, and comparing
//   to what it was taught. Correct answers reinforce committed circuits;
//   wrong answers trigger retraining on that association.
//   Interactive quiz also lets the USER answer — the model checks and
//   teaches itself from any mistakes the user corrects.

/* Maximum characters in any word the alphabet brain can learn. */
#define ALPHA_MAX_WORD_LEN  32

/*
 * Maximum distinct words in the vocabulary.
 * 26 covers one word per letter. Raise this if you want to teach
 * multiple words per letter (e.g., A → Ant, Apple, Alligator).
 */
#define ALPHA_VOCAB_SIZE    26

/*
 * Raw input size for the alphabet sensor.
 *
 * 26 (one-hot letter) + 4 (one-hot query type) = 30 floats.
 * The sensor's encode() pads this to MIMIR_EMBEDDING_SIZE.
 */
#define ALPHA_RAW_LETTER    26
#define ALPHA_RAW_QUERY      5   /* RECALL, NEXT, PREV, POSITION, VALIDATE */
#define ALPHA_RAW_SIZE      (ALPHA_RAW_LETTER + ALPHA_RAW_QUERY)

/* Output neurons for the alphabet network head — one per possible answer. */
#define ALPHA_N_OUTPUTS     26

/* Where the trained alphabet brain and its vocabulary are saved. */
#define ALPHA_BRAIN_PATH    "data/mimir_abc.brain"
#define ALPHA_VOCAB_PATH    "data/mimir_abc.vocab"

/*
 * AlphaQueryType — what kind of question are we asking about this letter?
 *
 * These four query types cover the core knowledge a literate brain holds
 * about each letter. Each is a distinct one-hot position in the raw input.
 *
 * BIOLOGICAL PARALLEL:
 * Different query types activate different hippocampal-cortical circuits.
 * "What does A stand for?" fires associative memory (temporal lobe).
 * "What comes after A?" fires sequential/procedural memory (striatum, cerebellum).
 * The same letter neuron participates in both — the query context routes it.
 */
typedef enum {
    ALPHA_QUERY_RECALL   = 0,  /* "What does A stand for?" → Apple             */
    ALPHA_QUERY_NEXT     = 1,  /* "What comes after A?"    → B                 */
    ALPHA_QUERY_PREV     = 2,  /* "What comes before B?"   → A                 */
    ALPHA_QUERY_POSITION = 3,  /* "What position is A?"    → 1st (index 0)     */
    ALPHA_QUERY_VALIDATE = 4,  /* "Which letter does a word starting with [X]  */
                               /*  belong to?" — encodes the word's FIRST CHAR */
                               /*  in the letter slot, not the letter itself.  */
                               /*  The brain pre-learns this principle:         */
                               /*  first-char=A → belongs to A, etc.           */
                               /*  Output is the same 26-letter space, so the  */
                               /*  argmax gives the "owner" letter of the word. */
} AlphaQueryType;

/*
 * AlphaVocab — the brain's vocabulary table for letter associations.
 *
 * Separate from the network weights because the weights encode a
 * WORD INDEX (0..25), not the string itself. This table maps index → string.
 * Decoding a network output means: argmax(output[]) → index → vocab.words[index].
 *
 * Persisted to ALPHA_VOCAB_PATH alongside the network checkpoint so that
 * word meanings survive restarts.
 */
typedef struct {
    char words[ALPHA_VOCAB_SIZE][ALPHA_MAX_WORD_LEN]; /* word index → string   */
    int  n_words;                                      /* how many words known  */
    int  letter_to_word[26];  /* letter index → word index; -1 = not yet taught */
} AlphaVocab;

/* --- Vocabulary management --- */

/* Zero the vocab table. Call before first use. */
void alpha_vocab_init(AlphaVocab *v);

/*
 * Find a word in the vocab by string (case-insensitive).
 * Returns its index, or -1 if not found.
 */
int  alpha_vocab_find(const AlphaVocab *v, const char *word);

/*
 * Add a word to the vocab if it is not already there.
 * Returns its index (existing or newly assigned).
 * Returns -1 if the vocab is full (ALPHA_VOCAB_SIZE reached).
 */
int  alpha_vocab_add(AlphaVocab *v, const char *word);

/* Save the vocabulary table to path (binary format). Returns 0 on success. */
int  alpha_vocab_save(const AlphaVocab *v, const char *path);

/* Load a vocabulary table from path. Returns 0 on success, -1 if not found. */
int  alpha_vocab_load(AlphaVocab *v, const char *path);

/* --- Encoding --- */

/*
 * Build the 30-float raw input for the alphabet sensor.
 *
 *   out[0..25]  = one-hot letter  (out[letter_idx] = 1.0, rest 0.0)
 *   out[26..29] = one-hot query   (out[26 + query] = 1.0, rest 0.0)
 *
 * This is the raw[] that gets passed to the sensor's encode() function.
 */
void alpha_build_raw(int letter_idx, AlphaQueryType query, float *out);

/*
 * Sensor encode function — implements the SensorEncodeFn contract.
 *
 * Copies the 30-float raw input into embedding[0..29] and zero-pads
 * embedding[30..127]. No learned encoder needed: one-hot inputs are
 * already maximally sparse and clean. The shared core learns the
 * linear and nonlinear combinations it needs.
 *
 * Register this with:
 *   sensor_register(reg, "letter", ALPHA_RAW_SIZE, NULL,
 *                   alpha_sensor_encode, vocab);
 */
int  alpha_sensor_encode(Sensor      *sensor,
                          const float *raw,       int raw_size,
                          float       *embedding, int embedding_size);

/* --- Pre-training: sequence and position knowledge --- */

/*
 * Pre-train the alphabet brain on sequential and positional knowledge.
 *
 * Teaches (and commits) three sets of facts BEFORE the user teaches
 * any word associations:
 *
 *   NEXT:     A→B, B→C, C→D, … Y→Z       (25 pairs)
 *   PREV:     B→A, C→B, D→C, … Z→Y       (25 pairs)
 *   POSITION: A→0, B→1, C→2, … Z→25      (26 pairs)
 *
 * WHY COMMIT FIRST?
 * Committed neurons are frozen — user-taught associations (RECALL)
 * cannot overwrite them. This mirrors how the alphabet sequence is
 * drilled into children BEFORE reading comprehension begins. "After A
 * comes B" is bedrock knowledge that vocabulary learning builds on top of.
 *
 * Called automatically when a new alphabet brain is created (no saved
 * checkpoint). Should NOT be called again after first commit.
 */
void alpha_pretrain_sequence(Network *net);

/* --- High-level teach / ask / quiz --- */

/*
 * Return values for alpha_teach — mirrors TrainVerdict semantics.
 */
typedef enum {
    ALPHA_LEARNED      = 0,  /* Taught successfully                              */
    ALPHA_CONFLICT     = 1,  /* Contradicts committed knowledge — rejected       */
    ALPHA_REVERIFY     = 2,  /* Same conflict seen multiple times — needs review */
    ALPHA_VOCAB_FULL   = 3,  /* Vocabulary is full — cannot add new word         */
    ALPHA_WRONG_LETTER = 4,  /* Word does not begin with the taught letter       */
} AlphaTeachResult;

/*
 * Teach the brain that letter_idx is associated with word.
 *
 *   alpha_teach(net, vocab, 0, "apple")  →  A is for Apple
 *
 * Internally:
 *   1. Adds word to vocab (or finds existing entry).
 *   2. Sets letter_to_word[letter_idx] = word_index.
 *   3. Trains on ALL currently known associations (not just this one) so
 *      that earlier taught words are reinforced alongside the new one.
 *      Committed sequence neurons are untouched (frozen weights).
 *   4. Returns ALPHA_LEARNED on success, or a conflict/error code.
 *
 * The caller should save ALPHA_BRAIN_PATH and ALPHA_VOCAB_PATH after
 * a successful teach so the knowledge survives a restart.
 */
AlphaTeachResult alpha_teach(Network *net, AlphaVocab *vocab,
                              int letter_idx, const char *word);

/*
 * Word bank accessors (alphabet.c).
 *
 * The bank holds ~20 common English words per letter, baked into the
 * binary so no file I/O is needed.  Two callers use it:
 *   - `teach all`  → uses the FIRST word per letter to train the network.
 *   - `quiz choice` → uses ALL words and filters the ones currently in
 *                     vocab so the quiz exclusively presents words the
 *                     network has never been trained on.
 *
 * Returns 0 / NULL when the indices are out of range or when the letter
 * has no entry.
 */
int         alpha_word_bank_count(int letter_idx);
const char *alpha_word_bank_get(int letter_idx, int word_idx);

/*
 * alpha_delta_rescue — discriminative single-layer delta-rule trainer for
 * RECALL associations. Public so the replay thread can reuse the SAME
 * algorithm used by alpha_teach. Three-factor Hebbian (train_step_brain)
 * cannot teach RECALL — see the long comment block in alphabet.c above the
 * function definition for the full bug history. If you find yourself
 * reinforcing a RECALL association from a new caller, ALWAYS use this
 * function. Never use train_step_brain on a RECALL pair.
 *
 * Cost: early-exits when every known association is correct and ≥ 0.80
 * confidence (single forward-pass-per-sample sweep). Worst case ~2000
 * epochs × n_known samples when the network is wildly off — at most a
 * handful of milliseconds for 1–26 letters.
 */
/*
 * Forward pass for a (letter, query) pair.  Builds the raw encoding,
 * zero-pads into embedding, and runs network_forward.
 */
void alpha_forward(Network *net, int letter_idx, AlphaQueryType query,
                   float *embedding, float *output);

void alpha_delta_rescue(Network *net, const AlphaVocab *vocab);

/*
 * Output-only delta rule for the 26 VALIDATE identity mappings.
 * Same algorithm as alpha_delta_rescue but for VALIDATE queries
 * (input letter_i → output[i]).  Called after output weight reset
 * in pretrain and from quiz/replay to maintain VALIDATE accuracy.
 */
void alpha_delta_rescue_validate(Network *net);

/*
 * Run the brain's LEARNED PRINCIPLE to decide whether word is a valid
 * association for letter_idx.
 *
 * The brain was pre-trained to know: "words starting with A belong to A,
 * words starting with B belong to B, etc." (identity principle over first
 * character). This function runs a VALIDATE forward pass using the word's
 * first character as the input, and returns:
 *
 *   -1          if the principle says word belongs to letter_idx (valid)
 *   0..25       the letter the brain thinks the word actually belongs to
 *
 * Example:
 *   alpha_validate_principle(net, 0, "apple")  → -1   (apple starts with A ✓)
 *   alpha_validate_principle(net, 0, "truck")  → 19   (truck starts with T, not A)
 */
int alpha_validate_principle(Network *net, int letter_idx, const char *word);

/*
 * Ask the brain a question about letter_idx using the given query type.
 *
 * Returns a pointer to a static string — do not free, copy if you need
 * to store it. Returns NULL if the brain has no confident answer.
 *
 * Examples:
 *   alpha_ask(net, vocab, 0, ALPHA_QUERY_RECALL)    → "apple"
 *   alpha_ask(net, vocab, 0, ALPHA_QUERY_NEXT)      → "B"
 *   alpha_ask(net, vocab, 1, ALPHA_QUERY_PREV)      → "A"
 *   alpha_ask(net, vocab, 0, ALPHA_QUERY_POSITION)  → "1st"
 */
const char *alpha_ask(Network *net, const AlphaVocab *vocab,
                      int letter_idx, AlphaQueryType query);

/*
 * Recite everything the brain knows about A through Z.
 *
 * For each letter, shows:
 *   - The association (A is for Apple) with confidence bar
 *   - Whether the expected word matches the network's actual output
 *
 * Letters not yet taught show as [not taught].
 */
void alpha_recite(Network *net, const AlphaVocab *vocab);

/*
 * Quiz the brain — tests and reinforces its own knowledge.
 *
 * Two modes:
 *
 *   Self-quiz (interactive=false):
 *     Iterates over all known associations and sequence facts.
 *     Runs forward pass to get predicted answer.
 *     Prints: "After A comes... B ✓" or "A is for... Banana ✗ (expected: Apple)"
 *     Wrong answers trigger a brief retraining burst on that pair.
 *
 *   Interactive quiz (interactive=true):
 *     Model asks YOU the question. You type an answer.
 *     Model checks correctness, explains, and re-teaches itself if wrong.
 *     This is active recall — the most effective form of learning.
 *
 * BIOLOGICAL PARALLEL:
 * Hippocampal replay during sleep is the brain doing its own self-quiz.
 * It replays experiences, strengthens correct circuits, weakens wrong ones.
 * The interactive mode adds a teacher signal — like a parent correcting
 * a child's recitation.
 */
void alpha_quiz(Network *net, AlphaVocab *vocab, bool interactive);

// ============================================================
// VISION — Visual sensor for cross-modal letter-word associations
// ============================================================
//
// (2026-04-13) Enables the brain to associate visual features with
// letter-word associations.  "A is for apple" gains grounding when
// the brain also knows what an apple looks like.
//
// Architecture: 16x16 grayscale images → 256 floats → average-pool
// encoding → 128-dim embedding → shared hidden layer → word output.
//
// The visual pathway shares the same hidden layer and output layer as
// the text pathway.  Cross-modal training teaches the output layer to
// respond to both text and visual input patterns.

#define VISION_IMG_SIZE   128
#define VISION_RAW_SIZE   (VISION_IMG_SIZE * VISION_IMG_SIZE)  /* 16384 grayscale floats */

/* V1-analogue Gabor front-end.
 * 8 orientations × 4×4 spatial pooling grid = 128 features → exactly fills
 * MIMIR_EMBEDDING_SIZE.  See src/vision.c for the pipeline rationale. */
#define VISION_GABOR_ORIENTATIONS 8
#define VISION_POOL_GRID          4
#define VISION_FEATURE_SIZE       (VISION_GABOR_ORIENTATIONS * VISION_POOL_GRID * VISION_POOL_GRID)

int  vision_load_raw(const char *path, float *out);
void vision_encode(const float *raw, float *embedding);
void vision_forward(Network *net, const float *raw_image,
                    float *embedding, float *output);
void vision_train(Network *net, const AlphaVocab *vocab,
                  float *img_data[26]);
int  vision_rescue(Network *net, const AlphaVocab *vocab,
                   float *img_data[26]);
void vision_test(Network *net, const AlphaVocab *vocab,
                 float *img_data[26]);
int  vision_load_all(const AlphaVocab *vocab, float *img_data[26]);
void vision_free_all(float *img_data[26]);

// ============================================================
// HIPPOCAMPUS — Episodic mistake memory (error metadata, no word content)
// ============================================================
//
// The hippocampus records WHICH letters were gotten wrong and WHEN,
// not what was said or what the correct word was. This distinction
// separates episodic memory (event tags) from semantic memory (meaning).
//
// Semantic memory lives in the network weights — the hippocampus only
// holds the index that says "letter A needs more practice", not "A=apple".
//
// Persisted to HIPPO_PATH so the brain's error history survives restarts.
// The file contains only integer arrays — no strings, no word content.

#define HIPPO_PATH            "data/mimir_hippo.dat"
#define HIPPO_RING_SIZE       64    /* remember last 64 individual mistakes    */
#define HIPPO_RECENCY_WINDOW   3    /* rounds within which a mistake is "recent" */
#define HIPPO_RECENCY_BOOST   3.0f  /* multiply error count when recent       */

/*
 * A single entry in the hippocampal mistake ring.
 * Stores ONLY the letter index and the round number — no word strings.
 * This forces the network to re-derive semantics from its own weights.
 */
typedef struct {
    int letter_idx;   /* which letter was gotten wrong (0=A … 25=Z) */
    int round;        /* which quiz round it happened in             */
} HippoMistake;

/*
 * The full hippocampus state.
 *
 * Per-letter arrays track aggregate statistics; the ring buffer tracks
 * the most recent HIPPO_RING_SIZE individual mistakes in time order.
 */
typedef struct {
    /* ── Aggregate per-letter statistics ── */
    int error_count[26];       /* total mistakes per letter across all rounds  */
    int correct_count[26];     /* total correct answers per letter             */
    int last_error_round[26];  /* round of most recent mistake (-1 = never)   */
    int streak_correct[26];    /* consecutive correct answers since last error */

    /* ── Episodic ring buffer ── */
    HippoMistake ring[HIPPO_RING_SIZE]; /* circular buffer of recent mistakes  */
    int ring_head;             /* next write slot (mod HIPPO_RING_SIZE)        */
    int ring_count;            /* entries written so far (caps at HIPPO_RING_SIZE) */

    int total_rounds;          /* how many quiz rounds have been completed     */
} Hippocampus;

/* --- Hippocampus API --- */

/* Zero all fields; set last_error_round[i] = -1 for all letters. */
void  hippo_init(Hippocampus *h);

/* Record a wrong answer for letter_idx on the current round. */
void  hippo_record_mistake(Hippocampus *h, int letter_idx);

/* Record a correct answer for letter_idx. */
void  hippo_record_correct(Hippocampus *h, int letter_idx);

/* Advance the round counter (call once at the end of each full quiz pass). */
void  hippo_advance_round(Hippocampus *h);

/*
 * Priority score for scheduling extra practice.
 * Higher = needs more rehearsal.
 * Letters never gotten wrong return 0.0 (but still appear in every pass).
 */
float hippo_priority(const Hippocampus *h, int letter_idx);

/* Print mistake history to stdout. */
void  hippo_print(const Hippocampus *h);

/* Save to path (binary, no strings). Returns 0 on success. */
int   hippo_save(const Hippocampus *h, const char *path);

/*
 * Load from path. Returns 0 on success, -1 if not found / corrupt.
 * On failure the caller should call hippo_init() to use a fresh state.
 */
int   hippo_load(Hippocampus *h, const char *path);

/*
 * Multiple-choice word quiz — the brain learns patterns, not lookups.
 *
 * For each taught letter, present the letter plus N_CHOICES candidate words
 * (the correct one plus distractors sampled from the vocabulary).
 * The brain runs a forward pass and picks whichever candidate scores
 * highest in the output layer — a pattern-matching decision, not a
 * table lookup.
 *
 * After every pick, show the correct answer regardless of whether the
 * brain was right or wrong. If wrong, immediately retrain on the correct
 * association and log the mistake to the hippocampus.
 *
 * The loop repeats — prioritising letters the hippocampus flagged as
 * error-prone — until the brain achieves 100% on a full clean pass
 * across all taught letters.
 *
 * hippo is updated in-place. Pass a zeroed Hippocampus if starting fresh.
 * The caller is responsible for saving hippo and the network afterwards.
 */
void alpha_quiz_choice(Network *net, AlphaVocab *vocab, Hippocampus *hippo);

// ============================================================
// SENSOR ARCHITECTURE — Pluggable multi-modal input system
// ============================================================
//
// The human brain has dedicated sensory cortices — V1 for vision,
// A1 for auditory, S1 for somatosensory (touch), OFC for smell —
// each one specialised to preprocess a single modality before the
// signal reaches shared association cortex for "thinking".
//
// We mirror that structure exactly:
//
//   Sensor (dedicated encoder)           Shared core
//   ─────────────────────────────        ─────────────────
//   Gate encoder    2 floats  ──┐
//   Letter encoder 26 floats  ──┤──→ 128-float embedding ──→ Neurogenesis brain
//   Audio encoder   N floats  ──┤
//   Camera encoder  M floats  ──┘
//
// Every sensor — no matter how different its raw data looks — speaks
// the same language before the core sees it: MIMIR_EMBEDDING_SIZE floats.
// Adding a new sense (sonar, gyroscope, language token) means writing
// one encode() function and calling sensor_register(). Nothing else
// in the system needs to change.
//
// WHY PLUGGABLE AT RUNTIME (not compile-time)?
// A robot may gain a camera after deployment. A person loses vision
// and the brain remaps. Hardcoding sensor types means recompiling for
// every new modality and makes the architecture fragile — removing one
// sensor would break every assumption downstream. Runtime registration
// treats sensors like USB devices: the hub doesn't care what's plugged
// in, only that it speaks the right protocol.
//
// THE PROTOCOL (the only contract a sensor must fulfil):
//
//   int encode(Sensor *self,
//              const float *raw,  int raw_size,
//              float *embedding,  int embedding_size);
//
//   Take raw_size floats in, write embedding_size floats out.
//   Return 0 on success, -1 on failure. That's it.
// ============================================================

// ── Embedding size ───────────────────────────────────────────────────────────

/*
 * MIMIR_EMBEDDING_SIZE — the shared "language" all sensors speak.
 *
 * WHY 128?
 *
 *   Cache alignment:
 *     128 floats = 512 bytes = exactly 8 cache lines (64 bytes each).
 *     The existing -mavx2 -mfma flags process 8 floats per cycle; 128
 *     divides evenly, so there is no scalar remainder loop.
 *
 *   Headroom for real sensors:
 *     A mel-spectrogram audio frame typically has 40–128 frequency bins.
 *     At 128 the audio encoder can copy directly without losing information.
 *     An image patch encoder (e.g. 8×8 pixels, 3 channels = 192 raw values)
 *     compresses into 128 without catastrophic loss.
 *
 *   Stability under Hebbian learning:
 *     Each neuron in the first hidden layer of the core receives
 *     MIMIR_EMBEDDING_SIZE inputs. With BCM homeostasis, the sliding
 *     threshold θ prevents any neuron from becoming permanently
 *     overactive. At 256+ inputs with dense activations, early θ
 *     saturation can stall learning; 128 sits below that threshold
 *     empirically for sigmoid activations in [0, 1].
 *
 *   Not a ceiling on raw sensor data:
 *     Sensors compress their own raw input DOWN to 128 before the
 *     core ever sees it. A camera producing 640×480 pixels is the
 *     sensor's problem to compress; the core always sees exactly 128.
 *
 *   Cannot change without full retrain:
 *     Every neuron in the core's first hidden layer has exactly
 *     MIMIR_EMBEDDING_SIZE weights. Changing this value after neurons
 *     are committed is not possible — committed weights are permanent.
 *     128 is chosen to be large enough that we never need to grow it.
 */
#define MIMIR_EMBEDDING_SIZE 128

/*
 * Maximum sensors that can be registered at one time.
 *
 * 16 is generous: the human body has roughly 5–9 named senses depending
 * on whether you count proprioception, vestibular, thermoception, etc.
 * A fixed-size array (not malloc) keeps the registry cache-contiguous
 * and makes save/restore trivial — no pointer chasing.
 */
#define SENSOR_MAX 16

// ── Sensor struct ────────────────────────────────────────────────────────────

/*
 * Forward declaration so SensorEncodeFn can take a Sensor* parameter
 * before the full struct definition appears below.
 */
typedef struct Sensor Sensor;

/*
 * SensorEncodeFn — the one function every sensor must provide.
 *
 * Parameters:
 *   sensor         — the Sensor that owns this function; use it to reach
 *                    sensor->encoder or sensor->user_data if needed.
 *   raw            — raw input from the physical sensor (microphone samples,
 *                    pixel values, one-hot letter vector, gate bits, …).
 *   raw_size       — number of floats in raw[]; must equal sensor->raw_size.
 *   embedding      — output buffer to fill; exactly embedding_size floats.
 *   embedding_size — always MIMIR_EMBEDDING_SIZE; passed explicitly so the
 *                    function signature is self-documenting.
 *
 * Returns 0 on success, -1 on any error (size mismatch, NULL pointer, …).
 *
 * BIOLOGICAL PARALLEL:
 * This is the "projection neuron" layer at the top of a sensory cortex —
 * the last stage of modality-specific preprocessing before the signal
 * enters the thalamic relay and from there the association cortex.
 */
typedef int (*SensorEncodeFn)(Sensor       *sensor,
                               const float  *raw,        int raw_size,
                               float        *embedding,  int embedding_size);

/*
 * Sensor — one pluggable sensory modality.
 *
 * BIOLOGICAL PARALLEL:
 * Each Sensor corresponds to one sensory cortex (V1, A1, S1, …).
 * It owns its own preprocessing network (the encoder), its own notion
 * of what "raw input" looks like, and its own private state (user_data —
 * e.g., a letter vocabulary, an audio sample rate, or a camera resolution).
 *
 * The Sensor does NOT know about the shared neurogenesis core.
 * It only knows: "I take raw input, I produce an embedding."
 * Separation of concerns: the sensor is responsible for its modality,
 * the core is responsible for thinking.
 */
struct Sensor {
    char           name[32];    /* Human-readable ID — "gate", "letter", "audio".
                                 * Used as a lookup key in the registry.          */

    int            raw_size;    /* Expected number of floats in the raw input.
                                 * sensor_encode() will reject calls that pass a
                                 * different size — fail fast, not silently wrong. */

    Network       *encoder;     /* Small learned network: raw_size → MIMIR_EMBEDDING_SIZE.
                                 * May be NULL if encode() builds the embedding
                                 * by hand (e.g., a one-hot letter vector padded
                                 * to 128 floats needs no learned encoder).
                                 * When non-NULL, the registry owns this pointer
                                 * and frees it on sensor_deregister().           */

    SensorEncodeFn encode;      /* Encode function — the sensor's only obligation
                                 * to the outside world. See SensorEncodeFn above.*/

    void          *user_data;   /* Sensor-specific state. Cast to the correct
                                 * type inside encode(). Examples:
                                 *   - AlphaVocab* for a letter sensor
                                 *   - int sample_rate for an audio sensor
                                 *   - struct CameraConfig* for a vision sensor
                                 * The registry never touches this pointer —
                                 * lifetime is the caller's responsibility.       */

    bool           active;      /* false = sensor is registered but switched off.
                                 * The core skips inactive sensors when building
                                 * the combined embedding. Lets you temporarily
                                 * disable a noisy or unavailable input (e.g.,
                                 * camera unplugged) without deregistering it.    */
};

// ── Sensor Registry ──────────────────────────────────────────────────────────

/*
 * SensorRegistry — the "thalamus" of Mimir's sensory system.
 *
 * BIOLOGICAL PARALLEL:
 * The thalamus is the brain's sensory relay station. Every signal from
 * every sense (except smell, interestingly) passes through it before
 * reaching the cortex. It doesn't process the signal — it routes it.
 * Our registry does the same: it doesn't care what a sensor IS, only
 * that it can produce MIMIR_EMBEDDING_SIZE floats on demand.
 *
 * Sensors register here at startup or at any later point. The shared
 * neurogenesis core reads from this registry on every forward pass —
 * it calls encode() on each active sensor and combines the results
 * into one input vector for the core network.
 *
 * Fixed-size array (not a linked list or malloc'd array) because:
 *   - Sensors rarely number more than a handful.
 *   - Contiguous memory means the iteration in the hot path (every
 *     forward pass) stays in L1 cache.
 *   - No pointer chasing = no branch mispredictions per element.
 */
typedef struct {
    Sensor sensors[SENSOR_MAX]; /* Registered sensors in registration order.
                                 * Active status checked per sensor at use time. */
    int    count;               /* Number of registered sensors (0..SENSOR_MAX). */
} SensorRegistry;

// ── Sensor API ───────────────────────────────────────────────────────────────

/*
 * Initialise an empty registry. Call exactly once before any other
 * sensor_* function. Zeroes all slots.
 */
void sensor_registry_init(SensorRegistry *reg);

/*
 * Register a new sensor.
 *
 * name      — unique identifier string (max 31 chars + NUL).
 * raw_size  — how many floats this sensor's raw input contains.
 * encoder   — learned encoder network (may be NULL for manual encoders).
 * encode_fn — the encode function this sensor implements.
 * user_data — opaque pointer forwarded to encode_fn unchanged.
 *
 * Returns the sensor's index (0..SENSOR_MAX-1) on success.
 * Returns -1 if the registry is full or the name is already taken.
 *
 * Ownership: when encoder is non-NULL, the registry takes ownership.
 * Do not network_free() it externally after calling this.
 */
int sensor_register(SensorRegistry *reg,
                    const char     *name,
                    int             raw_size,
                    Network        *encoder,
                    SensorEncodeFn  encode_fn,
                    void           *user_data);

/*
 * Deregister a sensor by name.
 * Frees its encoder network if present.
 * Compacts the sensors[] array so there are no gaps.
 * Returns 0 on success, -1 if the name is not found.
 */
int sensor_deregister(SensorRegistry *reg, const char *name);

/*
 * Find a registered sensor by name.
 * Returns a pointer into the registry (valid until next deregister),
 * or NULL if not found.
 */
Sensor *sensor_find(SensorRegistry *reg, const char *name);

/*
 * Run one sensor's encode function.
 * Validates raw_size against sensor->raw_size before calling encode().
 * Writes exactly MIMIR_EMBEDDING_SIZE floats into embedding[].
 * Returns 0 on success, -1 on any error.
 */
int sensor_encode(Sensor      *sensor,
                  const float *raw,       int raw_size,
                  float       *embedding);

/*
 * Print a human-readable summary of all registered sensors.
 * Shows name, raw_size, encoder type, and active status.
 */
void sensor_registry_print(const SensorRegistry *reg);

// ============================================================
// TRAINING LOG — Persistent measurement journal
// ============================================================
//
// Every real learning event the brain goes through is recorded here:
// letter associations taught, pre-training runs, replay cycle summaries,
// and benchmark results. The file is appended — it accumulates across
// restarts so you can track improvement over the lifetime of the brain.
//
// FORMAT (tab-separated, one line per event):
//   TIMESTAMP  EVENT  key=value  key=value  ...
//
// Example lines:
//   2026-04-10T14:23:01  STARTUP    gate_neurons=23  abc_neurons=64
//   2026-04-10T14:23:15  LEARN      letter=A  word=apple  time_ms=847  conf_before=0.12  conf_after=0.91
//   2026-04-10T14:23:45  REPLAY     cycles=50  known=3  accuracy=0.800  corrections=6
//   2026-04-10T14:24:00  BENCHMARK  gate=AND  method=brain_native  epochs=3200  time_ms=11.3  neurons=7  solved=yes
//
// Parse examples (shell):
//   grep LEARN   data/mimir_training.log              — all teach events
//   grep REPLAY  data/mimir_training.log              — replay accuracy over time
//   grep BENCHMARK data/mimir_training.log            — all benchmark results

#define TLOG_PATH  "data/mimir_training.log"

/*
 * Open (or create) the log file in append mode.
 * Call once at startup before any other tlog_* functions.
 */
void tlog_open(void);

/* Flush and close the log file. Call at shutdown. */
void tlog_close(void);

/* Startup event — records the brain state when the process begins. */
void tlog_startup(int gate_neurons, int gate_committed,
                  int abc_neurons,  int abc_committed);

/*
 * Pre-training event — records the sequence/position knowledge pre-train.
 *   task        — short name, e.g. "abc_sequence"
 *   samples     — number of training samples
 *   epochs_run  — epochs actually executed (may be < max if converged)
 *   time_ms     — wall clock time for the full pre-train
 *   converged   — true if it solved before hitting the epoch limit
 */
void tlog_pretrain(const char *task, int samples, int epochs_run,
                   double time_ms, bool converged);

/*
 * Learn event — records one user-taught association.
 *   letter      — 'A'..'Z'
 *   word        — the word being associated
 *   time_ms     — wall time for the training burst
 *   neurons     — total neurons in abc_net after teaching
 *   committed   — committed neurons after teaching
 *   conf_before — network's max output confidence before teaching
 *   conf_after  — network's max output confidence after teaching
 */
void tlog_learn(char letter, const char *word,
                double time_ms, int neurons, int committed,
                float conf_before, float conf_after);

/*
 * Replay event — periodic snapshot of the replay thread's progress.
 *   cycles      — total letter-tests run so far
 *   known       — how many letters are currently taught
 *   accuracy    — correct / cycles so far (0.0–1.0)
 *   corrections — total self-retrains performed
 *   committed   — total committed neurons in abc_net
 */
void tlog_replay(unsigned long cycles, int known,
                 double accuracy, unsigned long corrections,
                 int committed);

/*
 * Benchmark event — one result row from the benchmark table.
 *   gate        — "AND", "OR", "XOR"
 *   method      — "brain_native", "backprop"
 *   epochs      — epochs to solve (-1 if failed)
 *   time_ms     — wall time
 *   memory_bytes — extra memory beyond weights
 *   neurons     — neuron count at end
 *   correct     — correct predictions out of total
 *   total       — total samples (4 for logic gates)
 *   solved      — whether it reached the success threshold
 */
void tlog_benchmark(const char *gate, const char *method,
                    int epochs, double time_ms, int memory_bytes,
                    int neurons, int correct, int total, bool solved);

/* Shutdown event — records final replay stats before exit. */
void tlog_shutdown(unsigned long total_cycles, double accuracy,
                   unsigned long total_corrections);

// ============================================================
// REPLAY — Background self-training (hippocampal replay)
// ============================================================
//
// BIOLOGICAL PARALLEL:
// During sleep — especially REM and slow-wave sleep — the hippocampus
// replays sequences of the day's experiences back to the neocortex.
// Each replay cycle strengthens the synaptic weights that encode those
// memories. Repeated replay is how short-term memories become permanent.
// This is why a child who recites the alphabet before bed remembers it
// better the next morning than one who only heard it once.
//
// In Mimir, the replay thread does the same thing:
//   - It runs in the background continuously, even while you type.
//   - Each iteration picks a known association, tests it with a forward
//     pass, and retrains if the prediction was wrong or uncertain.
//   - Over time, weak associations become strong; strong ones get
//     committed and frozen — permanent long-term memory.
//   - The model gets smarter the longer it runs, even with no new input.
//
// CONCURRENCY:
// The main thread (CLI) and the replay thread both modify the network.
// They share a mutex — whoever holds the lock owns the network.
// The CLI locks before teaching; the replay thread locks before training.
// Neither can corrupt the other's weights.

/*
 * ReplayState — everything the background thread needs.
 *
 * Shared between the replay thread and the main CLI thread.
 * All network and vocab access must hold `lock`.
 */
typedef struct {
    Network         *abc_net;    /* The alphabet brain being replayed          */
    AlphaVocab      *vocab;      /* Word associations to test and reinforce    */
    pthread_mutex_t  lock;       /* Mutex — acquire before any net/vocab access*/
    pthread_t        thread;     /* Handle to the background thread            */
    volatile int     running;    /* 1 = thread is live, 0 = stop signal sent   */
    volatile int     verbose;    /* 1 = print each test, 0 = silent            */

    /* Visual replay — loaded once at init, reused by vision_rescue */
    float           *vis_images[26]; /* Gabor-ready raw images, NULL if absent */
    int              vis_loaded;     /* Number of images loaded (0 = disabled) */

    /* Statistics — updated by the replay thread, read by replay_status() */
    unsigned long    cycles;     /* Total individual letter-tests performed    */
    unsigned long    correct;    /* Tests where prediction matched expected     */
    unsigned long    corrected;  /* Tests that triggered a retraining burst    */
    unsigned long    committed;  /* Neurons committed since replay started     */
} ReplayState;

/*
 * Initialise a ReplayState. Call once before replay_start().
 * net and vocab must remain valid for the lifetime of the replay.
 */
void replay_init(ReplayState *r, Network *abc_net, AlphaVocab *vocab);

/*
 * Spawn the background replay thread.
 * Returns 0 on success, -1 if the thread could not be created.
 * After this call, the replay thread owns the mutex during training;
 * the caller must lock/unlock around any network or vocab modifications.
 */
int  replay_start(ReplayState *r);

/*
 * Signal the replay thread to stop and wait for it to exit.
 * After this returns, the mutex is no longer contended and the network
 * can be safely accessed without locking.
 */
void replay_stop(ReplayState *r);

/*
 * Print a human-readable status summary:
 *   cycles, accuracy %, neurons committed, retrain count.
 */
void replay_status(const ReplayState *r);

#endif // MIMIR_H
