/*
 * MIMIR - Dynamic Network with Neurogenesis
 *
 * This is the brain. Not a fixed blueprint of neurons wired at birth,
 * but a living system that grows new neurons when it needs to learn
 * something it can't currently handle.
 *
 * THE KEY IDEA:
 * Traditional neural networks: you decide the architecture, then train.
 * Our network: starts nearly empty, grows as challenges demand.
 *
 * HOW IT GROWS:
 * 1. Network attempts a task with its current neurons
 * 2. If error stays high (it's stuck), neurogenesis triggers
 * 3. New neurons are born IMMATURE — they contribute almost nothing
 * 4. Over epochs, they mature and integrate (maturity: 0.0 → 1.0)
 * 5. Once the task is learned, those neurons become COMMITTED
 * 6. Committed neurons are PERMANENT — they hold knowledge forever
 * 7. Next task? New neurons are added on top. Old knowledge stays.
 *
 * BIOLOGICAL PARALLELS:
 * - Neurogenesis: new neurons born in response to learning demand
 * - Maturation: newborn neurons take time to integrate into circuits
 * - Commitment: neurons that form stable circuits become permanent
 *
 * ════════════════════════════════════════════════════════════════════
 * DESIGN HISTORY — Neurogenesis architecture decisions.
 * ════════════════════════════════════════════════════════════════════
 *
 * ── EXPONENTIAL GROWTH SCHEME (2^gen) — REMOVED ──────────────────────
 *
 * Original design: each neurogenesis event adds 2^growth_gen neurons.
 * Gen 0: +1, Gen 1: +2, Gen 2: +4, Gen 3: +8 ...
 *
 * Motivation: learning complexity scales exponentially — simple tasks
 * need few neurons, complex tasks need many. The header still tracks
 * growth_gen for completeness but it no longer controls burst size.
 *
 * Why removed: for a stalled network with a bad seed, this created
 * runaway growth (Gen 0→1→2→... with no convergence between events).
 * After 10 events: 2^10 = 1024 neurons added. The output layer with
 * 1024 inputs became untrainable with brain-native's noisy modulator.
 *
 * ── TRICKLE RATE: 2 NEURONS PER EVENT ────────────────────────────────
 *
 * Current: every neurogenesis event activates exactly 2 dormant neurons.
 * Motivation: the hippocampus adds ~700 new neurons/day from a finite
 * stem-cell niche. Small controlled batches give the WTA + ACh mechanisms
 * time to specialise each recruit before the next wave arrives. 2 is the
 * minimum useful batch (odd counts cause WTA asymmetry on even-sample tasks).
 *
 * ── STARTING FROM 1 NEURON — REPLACED BY POOL ARCHITECTURE ──────────
 *
 * Original: network_create() → single output neuron, no hidden layer.
 * Neurogenesis inserts hidden layer when stuck (the "legacy path" in
 * network_neurogenesis_v still handles networks built this way).
 *
 * Problems observed:
 *   1. With RPE modulator: stall after epoch 1 because constant failure
 *      saturated baseline → modulator → 0 → no learning ever.
 *   2. Without RPE: slow to grow. XOR required many growth events
 *      (each adding only 1-2 neurons) and tens of thousands of epochs.
 *   3. Layer insertion (shifting layers array, rewiring output weights)
 *      was fragile and happened during training.
 *
 * Biological parallel: the human brain doesn't start from 1 neuron.
 * Peak fetal neurogenesis produces ~250,000 neurons/minute. By birth
 * the cortex is already overbuilt (~100 billion neurons). Postnatal
 * pruning then removes 50-70% of unused connections.
 *
 * Fix: network_create_with_pool(inputs, n_active, n_pool, outputs, act)
 * Pre-builds the architecture at creation time. n_active neurons are
 * MATURE and train immediately. n_pool neurons are DORMANT (id=-1,
 * contribute 0) and are activated 2 at a time when neurogenesis fires.
 * The output layer is pre-sized for all slots — no realloc needed.
 *
 * ── POOL AUTO-REFILL — TRIED AND REMOVED ─────────────────────────────
 *
 * Tried: whenever < 4 dormant remain after an activation, automatically
 * add 4 new dormant neurons and resize the output weights.
 *
 * Motivation: ensure neurogenesis always has dormant neurons available,
 * preventing "pool exhausted" failures mid-training.
 *
 * Why removed: caused runaway growth when a bad seed got the network stuck.
 * With XOR + seed=42: neurogenesis fired every 1000 epochs. Each event
 * activated 2 + immediately refilled 4 → net gain 4 per event → grew to
 * 109 neurons after 50000 epochs without converging. With 109 hidden
 * inputs, the output's Hebbian update (lr * pre * post * modulator) was
 * too diluted for any single weight to get a strong enough signal.
 * Final accuracy: 3/4, FAILED.
 *
 * Current: pool is finite. Max network size = n_active + n_pool at
 * creation. Once all dormant slots are consumed, neurogenesis events
 * do nothing (0 activated). The network either converges with what it
 * has or fails honestly. helpers layer_add_dormant() and
 * layer_resize_weights() remain in the codebase (marked unused) in case
 * we revisit controlled pool expansion with better convergence guards.
 *
 * ── MATURATION RATE ───────────────────────────────────────────────────
 *
 * Current: maturation_rate = 0.05 per epoch → 20 epochs to full maturity.
 * This was never a problem source — kept from the original design.
 * Biological parallel: hippocampal newborn neurons take 4-8 weeks to
 * fully integrate; we compress that to 20 training epochs.
 * ════════════════════════════════════════════════════════════════════
 */

#include "mimir.h"

// ============================================================
// INTERNAL HELPERS — Neuron and Layer management
// ============================================================

/*
 * Initialize a single neuron with random weights.
 *
 * New neurons are IMMATURE (maturity = 0.0). Their output is
 * multiplied by maturity, so they contribute ~nothing at birth.
 * This prevents a newly born neuron from disrupting the network's
 * existing learned behavior — it slides in gently.
 *
 * BIOLOGICAL PARALLEL: Newborn neurons in the hippocampus have
 * weak, unstable synapses. They're hyperexcitable but poorly
 * connected. Over days to weeks, they form stable synapses with
 * existing neurons and their excitability normalizes. Our maturity
 * scaling mimics this gradual integration.
 *
 * WEIGHT INITIALIZATION: Xavier/Glorot scaled by 1/sqrt(n_inputs).
 * Even though maturity will suppress the output initially, we want
 * good initial weights ready for when the neuron matures. Starting
 * with well-scaled random weights means the neuron can immediately
 * contribute meaningfully once maturity reaches ~0.5+.
 */
static void neuron_init(Neuron *n, int n_weights, Activation act, int id) {
    n->weights = (float *)malloc(n_weights * sizeof(float));
    if (!n->weights) {
        fprintf(stderr, "FATAL: Failed to allocate weights for neuron %d\n", id);
        exit(1);
    }
    n->n_weights = n_weights;
    n->bias = 0.0f;

    /* Neurogenesis state: born immature */
    n->maturity = 0.0f;
    n->activity = 0.0f;
    n->age = 0;
    n->id = id;
    n->state = NEURON_IMMATURE;
    n->act = act;

    /* Cached values: zeroed until first forward pass */
    n->last_z = 0.0f;
    n->last_output = 0.0f;

    /*
     * Brain-native learning state.
     * theta starts at 0.5 — the midpoint of sigmoid output range [0,1].
     * This means new neurons start neutral: they'll strengthen connections
     * that push them above 0.5 and weaken those that keep them below.
     * As training progresses, theta slides to each neuron's own natural
     * operating point based on its actual firing history.
     *
     * mean_out starts at 0 — no history yet.
     */
    n->theta = 0.5f;
    n->mean_out = 0.0f;
    n->is_visual = 0;

    /* Xavier initialization: scale = 1/sqrt(fan_in) */
    float scale = 1.0f / sqrtf((float)n_weights);
    random_init(n->weights, n_weights, scale);
}

/*
 * Initialize a neuron as already MATURE with maturity = 1.0.
 * Used for the initial output layer neuron — it doesn't need
 * to go through the maturation process since it's part of
 * the network's original structure, not a neurogenesis product.
 */
static void neuron_init_mature(Neuron *n, int n_weights, Activation act, int id) {
    neuron_init(n, n_weights, act, id);
    n->maturity = 1.0f;
    n->state = NEURON_MATURE;
}

/*
 * Initialize a neuron as DORMANT — allocated and weight-initialized,
 * but invisible to the network until explicitly activated.
 *
 * Dormant neurons live in the pool: pre-allocated in the hidden layer
 * array with weights already shaped by random init, waiting to be
 * recruited. They have no pool ID yet (id = -1); a real ID is assigned
 * by pool_birth() at the moment of activation.
 *
 * BIOLOGICAL PARALLEL: Neural stem cells in the subventricular zone
 * and hippocampal dentate gyrus. They exist as committed progenitors
 * — partially differentiated, with the molecular machinery for becoming
 * neurons, but not yet born into the circuit. Recruitment happens in
 * response to activity and learning signals.
 */
static void neuron_init_dormant(Neuron *n, int n_weights, Activation act) {
    n->weights = (float *)malloc(n_weights * sizeof(float));
    if (!n->weights) { fprintf(stderr, "FATAL: pool neuron alloc\n"); exit(1); }
    n->n_weights = n_weights;
    n->bias      = 0.0f;
    float scale  = 1.0f / sqrtf((float)n_weights);
    random_init(n->weights, n_weights, scale);

    n->maturity   = 0.0f;
    n->activity   = 0.0f;
    n->age        = 0;
    n->id         = -1;          /* unborn — ID assigned at activation */
    n->state      = NEURON_DORMANT;
    n->act        = act;
    n->last_z     = 0.0f;
    n->last_output = 0.0f;
    n->theta      = 0.5f;
    n->mean_out   = 0.0f;
    n->is_visual  = 0;
}

/*
 * Add dormant (pool) neurons to an existing layer.
 * They are pre-allocated with weights but stay DORMANT until activated
 * by neurogenesis. The output layer already has weights for them
 * (if using network_create_with_pool) so no resize is needed here.
 */
static void __attribute__((unused)) layer_add_dormant(Layer *layer, int count, Activation act) {
    int new_count = layer->count + count;
    if (new_count > layer->capacity) {
        int new_cap = layer->capacity * 2;
        while (new_cap < new_count) new_cap *= 2;
        layer->neurons = (Neuron *)realloc(layer->neurons, new_cap * sizeof(Neuron));
        layer->outputs = (float  *)realloc(layer->outputs, new_cap * sizeof(float));
        if (!layer->neurons || !layer->outputs) {
            fprintf(stderr, "FATAL: layer_add_dormant realloc\n"); exit(1);
        }
        layer->capacity = new_cap;
    }
    for (int i = layer->count; i < new_count; i++) {
        neuron_init_dormant(&layer->neurons[i], layer->input_size, act);
        layer->outputs[i] = 0.0f;
    }
    layer->count = new_count;
}

/*
 * Free a neuron's weight memory.
 * The neuron struct itself is freed when the layer is freed
 * (since neurons are stored in a contiguous layer array).
 */
static void neuron_free(Neuron *n) {
    free(n->weights);
    n->weights = NULL;
    n->state = NEURON_DORMANT;
}

/*
 * Forward pass for a single neuron.
 *
 * Computes: output = maturity * activation(sum(w_i * x_i) + bias)
 *
 * The maturity scaling is the key neurogenesis mechanic:
 * - maturity = 0.0: output is always 0 regardless of weights
 *   (the neuron exists but is invisible to the network)
 * - maturity = 0.5: output is half-strength (partially integrated)
 * - maturity = 1.0: full output (fully mature, like any normal neuron)
 *
 * This gradient from invisible → visible prevents new neurons from
 * causing sudden disruptions. The network smoothly adjusts to their
 * growing presence.
 *
 * ACTIVITY TRACKING: We maintain an exponential moving average of
 * the absolute output value. This tells us how much this neuron
 * "fires" over time. Consistently low activity = the neuron isn't
 * doing anything useful. Consistently high = it's important.
 *
 * BIOLOGICAL PARALLEL: Synaptic integration. New neurons in the
 * brain don't instantly fire at full strength. Their synaptic
 * currents are small and unreliable at first. As synapses
 * strengthen (long-term potentiation), the neuron's influence
 * on downstream neurons grows.
 */
static float neuron_forward(Neuron *n, const float *inputs) {
    /* Dead neurons (killed by apoptosis) output nothing. Skip computation. */
    if (n->state == NEURON_DORMANT) {
        n->last_z = 0.0f;
        n->last_output = 0.0f;
        return 0.0f;
    }

    /* Weighted sum: z = bias + sum(w_i * x_i) */
    float z = n->bias;
    for (int i = 0; i < n->n_weights; i++) {
        z += n->weights[i] * inputs[i];
    }

    /* Cache pre-activation value for training */
    n->last_z = z;

    /* Apply activation function */
    float raw_output = activate(z, n->act);

    /*
     * Scale by maturity. This is the neurogenesis mechanism:
     * immature neurons have near-zero output, mature ones have full output.
     *
     * NOTE: We scale the OUTPUT, not the weights. This means the
     * pre-activation value z is computed normally — so when we compute
     * derivatives for training, we get correct gradients. Only the
     * downstream visibility is affected.
     */
    float output = raw_output * n->maturity;

    /* Cache final output */
    n->last_output = output;

    /*
     * Update activity tracking (exponential moving average).
     * decay=0.9 means ~90% of the activity comes from history,
     * ~10% from the current output. This smooths out noise —
     * a neuron isn't judged on one bad epoch but on sustained
     * contribution over many epochs.
     */
    n->activity = 0.9f * n->activity + 0.1f * fabsf(output);

    return output;
}

/*
 * Create a layer with a given initial neuron count.
 *
 * We allocate with some headroom (capacity > count) to avoid
 * reallocating on every single neuron addition. Growth follows
 * a doubling strategy: when full, double the capacity.
 *
 * MEMORY LAYOUT: All neurons in a layer are contiguous in memory.
 * This is critical for cache performance — when the next layer
 * reads this layer's outputs, they're all in nearby cache lines.
 * A linked-list of neurons would scatter them across the heap,
 * causing constant cache misses. For large layers, the difference
 * is 10-100x in speed.
 */
static Layer layer_create(int initial_count, int input_size, Activation act,
                          NeuronPool *pool, bool mature) {
    Layer layer;
    /* Allocate at least 8 slots to avoid frequent early reallocs */
    layer.capacity = initial_count < 8 ? 8 : initial_count;
    layer.count = initial_count;
    layer.input_size = input_size;

    layer.neurons = (Neuron *)malloc(layer.capacity * sizeof(Neuron));
    layer.outputs = (float *)calloc(layer.capacity, sizeof(float));
    if (!layer.neurons || !layer.outputs) {
        fprintf(stderr, "FATAL: Failed to allocate layer\n");
        exit(1);
    }

    /* Initialize each neuron */
    for (int i = 0; i < initial_count; i++) {
        int id = pool_birth(pool);
        if (mature) {
            neuron_init_mature(&layer.neurons[i], input_size, act, id);
        } else {
            neuron_init(&layer.neurons[i], input_size, act, id);
        }
    }

    return layer;
}

/*
 * Free all memory owned by a layer.
 * First free each neuron's weight array, then the layer's arrays.
 */
static void layer_free(Layer *layer) {
    for (int i = 0; i < layer->count; i++) {
        neuron_free(&layer->neurons[i]);
    }
    free(layer->neurons);
    free(layer->outputs);
    layer->neurons = NULL;
    layer->outputs = NULL;
    layer->count = 0;
    layer->capacity = 0;
}

/*
 * Add neurons to an existing layer.
 *
 * New neurons are IMMATURE. They get random weights connecting them
 * to the previous layer (same input_size as existing neurons).
 *
 * If the layer is full (count == capacity), we double the capacity.
 * This amortized doubling means N insertions cost O(N) total,
 * not O(N^2) if we reallocated every time.
 *
 * RETURNS: Number of neurons actually added.
 */
static int __attribute__((unused))
layer_add_neurons(Layer *layer, int count, Activation act, NeuronPool *pool) {
    int new_count = layer->count + count;

    /* Grow capacity if needed (double until enough room) */
    if (new_count > layer->capacity) {
        int new_cap = layer->capacity;
        while (new_cap < new_count) {
            new_cap *= 2;
        }

        Neuron *new_neurons = (Neuron *)realloc(layer->neurons,
                                                 new_cap * sizeof(Neuron));
        float *new_outputs = (float *)realloc(layer->outputs,
                                               new_cap * sizeof(float));
        if (!new_neurons || !new_outputs) {
            fprintf(stderr, "FATAL: Failed to grow layer from %d to %d\n",
                    layer->capacity, new_cap);
            exit(1);
        }
        layer->neurons = new_neurons;
        layer->outputs = new_outputs;
        layer->capacity = new_cap;

        /* Zero out new output slots */
        for (int i = layer->count; i < new_cap; i++) {
            layer->outputs[i] = 0.0f;
        }
    }

    /* Initialize new neurons as IMMATURE */
    for (int i = 0; i < count; i++) {
        int idx = layer->count + i;
        int id = pool_birth(pool);
        neuron_init(&layer->neurons[idx], layer->input_size, act, id);
    }
    layer->count = new_count;

    return count;
}

/*
 * Resize the weights of all neurons in a layer to match a new input size.
 *
 * This is called when the PREVIOUS layer grows. If hidden layer goes
 * from 2 to 4 neurons, the output layer's neurons need to go from
 * 2 weights to 4 weights each.
 *
 * New weight slots (for the new connections) are initialized with
 * small random values. But since the new hidden neurons they connect
 * FROM are immature (maturity ~0), these new weights have ~zero effect
 * initially. As the hidden neurons mature, the connections come alive.
 *
 * BIOLOGICAL PARALLEL: When a new neuron migrates into a circuit,
 * existing neurons grow new dendritic spines to receive input from it.
 * These new synapses start weak (low weight) and strengthen over time
 * if the connection proves useful (Hebbian learning / LTP).
 */
static void __attribute__((unused)) layer_resize_weights(Layer *layer, int new_input_size) {
    for (int i = 0; i < layer->count; i++) {
        Neuron *n = &layer->neurons[i];
        int old_size = n->n_weights;

        if (new_input_size == old_size) continue;

        float *new_weights = (float *)realloc(n->weights,
                                               new_input_size * sizeof(float));
        if (!new_weights) {
            fprintf(stderr, "FATAL: Failed to resize neuron %d weights\n", n->id);
            exit(1);
        }

        /*
         * CRITICAL: Rescale existing weights when input dimension changes.
         *
         * WHY THIS MATTERS:
         * If a neuron had 3 hidden inputs with weights ~0.5 each, the
         * weighted sum is ~1.5. If we grow to 7 inputs and add 4 more
         * weights at ~0.5, the sum jumps to ~3.5. This pushes sigmoid
         * into saturation (output stuck near 0 or 1), where its
         * derivative is ~0, which kills learning completely.
         *
         * The fix: rescale all existing weights by sqrt(old_size/new_size).
         * This keeps the expected magnitude of the weighted sum constant
         * regardless of how many inputs there are.
         *
         * BIOLOGICAL PARALLEL: Synaptic scaling (Turrigiano 2008).
         * When a neuron receives too many excitatory inputs, it globally
         * scales DOWN all its synaptic strengths to maintain stable
         * firing rates. This is called "homeostatic plasticity" — the
         * neuron keeps itself in a healthy operating range despite
         * changes in its input count. Exactly what we're doing here.
         */
        if (new_input_size > old_size) {
            float rescale = sqrtf((float)old_size / (float)new_input_size);
            for (int w = 0; w < old_size; w++) {
                new_weights[w] *= rescale;
            }

            /* Initialize new weight slots with proper Xavier scaling */
            float scale = 1.0f / sqrtf((float)new_input_size);
            random_init(new_weights + old_size, new_input_size - old_size, scale);
        }

        n->weights = new_weights;
        n->n_weights = new_input_size;
    }
    layer->input_size = new_input_size;
}

// ============================================================
// NETWORK — Creation, destruction, forward pass
// ============================================================

/*
 * Create a new network.
 *
 * Starts with the MINIMUM viable structure: just one output layer.
 * No hidden layers. This is equivalent to our original single perceptron.
 *
 *   [Input (2)] → [Output (1)]
 *
 * The output neuron is created MATURE (maturity=1.0) since it's not
 * a product of neurogenesis — it's the original founding neuron.
 *
 * Hidden layers will be inserted dynamically when neurogenesis triggers.
 * The network discovers it needs more capacity by failing to learn,
 * then growing to meet the challenge.
 *
 * BIOLOGICAL PARALLEL: A developing embryo starts with a small number
 * of neurons in the neural plate. Through neurogenesis, this expands
 * into the full nervous system. Our network starts with 1 neuron and
 * grows from there.
 */
Network network_create(int n_inputs, int n_outputs, Activation act) {
    Network net;

    /* Start with capacity for a few layers (will grow if needed) */
    net.n_layers_cap = 8;
    net.n_layers = 1;  /* Just the output layer */
    net.layers = (Layer *)malloc(net.n_layers_cap * sizeof(Layer));
    if (!net.layers) {
        fprintf(stderr, "FATAL: Failed to allocate network layers\n");
        exit(1);
    }

    /* Initialize the pool (stem cell niche) */
    pool_init(&net.pool);

    /* Network topology */
    net.n_inputs = n_inputs;
    net.n_outputs = n_outputs;
    net.default_act = act;

    /* Neurogenesis parameters */
    net.growth_gen = 0;           /* Start at generation 0: first growth adds 2^0 = 1 neuron */
    net.maturation_rate = 0.05f;  /* 20 epochs to full maturity */
    net.rpe_baseline = 0.0f;
    net.n_conflict_records = 0;
    memset(net.conflict_log, 0, sizeof(net.conflict_log));

    /*
     * Create the output layer.
     * Input size = n_inputs (directly connected to raw input).
     * This neuron is born MATURE — it's the original, not a neurogenesis product.
     */
    net.layers[0] = layer_create(n_outputs, n_inputs, act, &net.pool, true);

    return net;
}

/*
 * Create a network with a pre-built hidden layer.
 *
 * Used for benchmarking: all training methods get the SAME architecture
 * (same number of hidden neurons, same initial weights if RNG is seeded
 * identically). This isolates the training method as the only variable.
 *
 * Architecture: [n_inputs] → [n_hidden] → [n_outputs]
 * All neurons start MATURE (maturity=1.0) to remove neurogenesis
 * as a variable in the benchmark.
 */
Network network_create_with_hidden(int n_inputs, int n_hidden, int n_outputs, Activation act) {
    Network net;

    net.n_layers_cap = 8;
    net.n_layers = 2;  /* hidden + output */
    net.layers = (Layer *)malloc(net.n_layers_cap * sizeof(Layer));
    if (!net.layers) {
        fprintf(stderr, "FATAL: Failed to allocate network layers\n");
        exit(1);
    }

    pool_init(&net.pool);

    net.n_inputs = n_inputs;
    net.n_outputs = n_outputs;
    net.default_act = act;
    net.growth_gen = 0;
    net.maturation_rate = 0.05f;
    net.rpe_baseline = 0.0f;
    net.n_conflict_records = 0;
    memset(net.conflict_log, 0, sizeof(net.conflict_log));

    /* Hidden layer: all neurons MATURE, reads raw input */
    net.layers[0] = layer_create(n_hidden, n_inputs, act, &net.pool, true);

    /* Output layer: all neurons MATURE, reads hidden layer output */
    net.layers[1] = layer_create(n_outputs, n_hidden, act, &net.pool, true);

    return net;
}

/*
 * Create a network with a pre-built hidden layer AND a dormant neuron pool.
 *
 * This is the biologically-motivated starting architecture:
 *   [Input] → [n_active MATURE | n_pool DORMANT] → [Output]
 *
 * The n_active neurons are ready to train from epoch 1 — no waiting for
 * neurogenesis to fire before the network has any capacity. The n_pool
 * dormant neurons sit silently in the hidden layer array, weights
 * pre-initialized, contributing exactly 0 to every forward pass.
 * When neurogenesis fires (stall detected), it wakes 2 dormant neurons
 * per event rather than malloc-ing brand-new ones.
 *
 * WHY THIS IS BETTER THAN STARTING FROM 1 NEURON:
 * - RPE modulator works: the network is competent from the start,
 *   so "surprise" fires on actual progress, not uniform failure.
 * - No exponential blowup: pool is finite and controlled.
 * - Faster convergence: hidden neurons learn alongside output from day 1.
 *
 * BIOLOGICAL PARALLEL: The brain starts overbuilt (fetal neurogenesis
 * produces billions of neurons before birth). The pool models the
 * "neural stem cell niche" — progenitor cells waiting to be recruited
 * when learning demand exceeds current circuit capacity.
 */
Network network_create_with_pool(int n_inputs, int n_active, int n_pool,
                                  int n_outputs, Activation act) {
    Network net;
    net.n_layers_cap  = 8;
    net.n_layers      = 2;
    net.layers        = (Layer *)malloc(net.n_layers_cap * sizeof(Layer));
    if (!net.layers) { fprintf(stderr, "FATAL: network_create_with_pool\n"); exit(1); }

    pool_init(&net.pool);
    net.n_inputs       = n_inputs;
    net.n_outputs      = n_outputs;
    net.default_act    = act;
    net.growth_gen     = 0;
    net.maturation_rate    = 0.05f;
    net.rpe_baseline       = 0.0f;
    net.n_conflict_records = 0;
    memset(net.conflict_log, 0, sizeof(net.conflict_log));

    int total_hidden = n_active + n_pool;

    /* Hidden layer: n_active MATURE neurons + n_pool DORMANT pool slots */
    Layer hidden;
    hidden.capacity   = total_hidden < 8 ? 8 : total_hidden;
    hidden.count      = total_hidden;
    hidden.input_size = n_inputs;
    hidden.neurons    = (Neuron *)malloc(hidden.capacity * sizeof(Neuron));
    hidden.outputs    = (float  *)calloc(hidden.capacity,  sizeof(float));
    if (!hidden.neurons || !hidden.outputs) {
        fprintf(stderr, "FATAL: hidden layer alloc\n"); exit(1);
    }

    for (int i = 0; i < n_active; i++) {
        int id = pool_birth(&net.pool);
        neuron_init_mature(&hidden.neurons[i], n_inputs, act, id);
    }
    for (int i = n_active; i < total_hidden; i++) {
        neuron_init_dormant(&hidden.neurons[i], n_inputs, act);
        /* Dormant neurons are not yet born — pool_birth called at activation */
    }
    net.layers[0] = hidden;

    /*
     * Output layer: weights for ALL hidden slots (active + pool).
     * Dormant neurons output 0.0, so their output weights are irrelevant
     * until activation. Having the weights pre-allocated means activation
     * needs zero resizing — just a state flip.
     */
    net.layers[1] = layer_create(n_outputs, total_hidden, act, &net.pool, true);

    return net;
}

/*
 * Free all network memory.
 * Layers → neurons → weights, then the layers array itself.
 */
void network_free(Network *net) {
    for (int l = 0; l < net->n_layers; l++) {
        layer_free(&net->layers[l]);
    }
    free(net->layers);
    net->layers = NULL;
    net->n_layers = 0;
}

/*
 * Forward pass through the entire network.
 *
 * Data flows from input → hidden layer(s) → output layer.
 * Each neuron in layer L reads outputs from layer L-1 (or raw
 * inputs for the first layer). This is standard feedforward
 * propagation — same as every neural network since the 1980s.
 *
 * The neurogenesis twist: immature neurons' outputs are scaled
 * by their maturity factor. So a newborn neuron at maturity=0.1
 * only contributes 10% of its normal output. The rest of the
 * network barely notices it. As it matures, its influence grows.
 *
 * FLOW:
 *   layer_inputs = raw_inputs
 *   for each layer:
 *     for each neuron in layer:
 *       neuron.output = maturity * activation(w . layer_inputs + bias)
 *     layer_inputs = layer.outputs   (feed to next layer)
 *   copy final layer outputs to result
 */
void network_forward(Network *net, const float *inputs, float *outputs) {
    const float *layer_input = inputs;

    for (int l = 0; l < net->n_layers; l++) {
        Layer *layer = &net->layers[l];

        for (int j = 0; j < layer->count; j++) {
            layer->outputs[j] = neuron_forward(&layer->neurons[j], layer_input);
        }

        /* This layer's outputs become next layer's inputs */
        layer_input = layer->outputs;
    }

    /* Copy final layer (output layer) results */
    Layer *last = &net->layers[net->n_layers - 1];
    for (int i = 0; i < last->count && i < net->n_outputs; i++) {
        outputs[i] = last->outputs[i];
    }
}

// ============================================================
// NEUROGENESIS — Birth, maturation, commitment
// ============================================================

/*
 * Trigger neurogenesis: grow the network.
 *
 * This is called when the network is struggling — error is high and
 * not decreasing. The network's response: GROW MORE BRAIN CELLS.
 *
 * GROWTH STRATEGY:
 * - Adds 2^growth_gen neurons (exponential: 1, 2, 4, 8, 16, ...)
 * - If no hidden layer exists: creates one (inserted before output)
 * - If hidden layer exists: adds neurons to it
 * - After adding hidden neurons, resizes output layer weights to match
 * - Increments growth_gen for next time
 *
 * WHY EXPONENTIAL?
 * Simple problems (AND, OR) need 0 extra neurons. They're solved by
 * the output neuron alone. XOR needs ~2-4 hidden neurons. A full
 * language model might need billions. The exponential curve means:
 * - Early growth is conservative (don't waste neurons on easy stuff)
 * - Later growth is aggressive (complex problems get the resources)
 *
 * This mirrors biological evolution: cortical neuron counts across
 * species roughly follow exponential scaling with brain size.
 * Mouse: ~70M neurons. Human: ~86B neurons. Not linear growth —
 * exponential jumps in capacity enabled exponential jumps in capability.
 *
 * RETURNS: Number of neurons added.
 */
int network_neurogenesis_v(Network *net, int verbose);

int network_neurogenesis(Network *net) {
    return network_neurogenesis_v(net, 1);
}

int network_neurogenesis_v(Network *net, int verbose) {
    /*
     * Pool-based neurogenesis: activate dormant neurons from the pre-allocated
     * pool rather than malloc-ing new ones on demand.
     *
     * ACTIVATE 2 AT A TIME — biological trickle rate.
     * The hippocampus adds ~700 neurons/day from a pool of stem cells.
     * Small controlled batches prevent the exponential blowups we saw
     * with the old 2^gen scheme and give the ACh/WTA mechanisms time
     * to specialise each recruit before the next wave arrives.
     *
     * POOL REFILL: if fewer than 4 dormant neurons remain after activation,
     * we top the pool back up with 4 new dormant neurons. They arrive
     * weight-initialised but invisible (DORMANT), ready for the next event.
     * If the output layer needs more weight slots, layer_resize_weights
     * handles that. This is the only path where a malloc happens — and
     * only when the pool genuinely runs dry, not every neurogenesis event.
     */
    Layer *hidden = NULL;

    /* Find or create the hidden layer */
    if (net->n_layers == 1) {
        /*
         * Legacy path: network was built with network_create (no hidden).
         * Insert a hidden layer the old way, then fall through to pool logic.
         */
        if (net->n_layers + 1 > net->n_layers_cap) {
            net->n_layers_cap *= 2;
            net->layers = (Layer *)realloc(net->layers,
                                            net->n_layers_cap * sizeof(Layer));
            if (!net->layers) { fprintf(stderr, "FATAL: grow layers\n"); exit(1); }
        }
        net->layers[1] = net->layers[0];
        net->n_layers  = 2;

        /* Create hidden layer with 2 IMMATURE neurons + 4 DORMANT pool */
        int n_active = 2, n_pool = 4;
        int total = n_active + n_pool;
        Layer h;
        h.capacity   = total < 8 ? 8 : total;
        h.count      = total;
        h.input_size = net->n_inputs;
        h.neurons    = (Neuron *)malloc(h.capacity * sizeof(Neuron));
        h.outputs    = (float  *)calloc(h.capacity, sizeof(float));
        for (int i = 0; i < n_active; i++) {
            int id = pool_birth(&net->pool);
            neuron_init(&h.neurons[i], net->n_inputs, net->default_act, id);
        }
        for (int i = n_active; i < total; i++) {
            neuron_init_dormant(&h.neurons[i], net->n_inputs, net->default_act);
        }
        net->layers[0] = h;

        /* Output layer: rewire to read from hidden instead of raw input */
        Layer *out = &net->layers[1];
        for (int i = 0; i < out->count; i++) {
            Neuron *n = &out->neurons[i];
            free(n->weights);
            n->n_weights = total;
            n->weights   = (float *)malloc(total * sizeof(float));
            float scale  = 1.0f / sqrtf((float)total);
            random_init(n->weights, total, scale);
        }
        out->input_size = total;

        if (verbose) printf("  NEUROGENESIS: created hidden layer "
                            "(%d active + %d in pool)\n", n_active, n_pool);

        net->rpe_baseline = 0.0f;
        net->growth_gen++;
        return n_active;
    }

    /* Pool path: hidden layer already exists */
    hidden = &net->layers[0];

    /* Count dormant neurons */
    int n_dormant = 0;
    for (int i = 0; i < hidden->count; i++)
        if (hidden->neurons[i].state == NEURON_DORMANT) n_dormant++;

    /* Activate up to 2 dormant neurons from the pool */
    int to_activate = 2;
    int activated   = 0;
    for (int i = 0; i < hidden->count && activated < to_activate; i++) {
        Neuron *n = &hidden->neurons[i];
        if (n->state == NEURON_DORMANT) {
            n->id       = pool_birth(&net->pool);
            n->state    = NEURON_IMMATURE;
            n->maturity = 0.0f;
            n->activity = 0.0f;
            n->age      = 0;
            activated++;
        }
    }

    int remaining = n_dormant - activated;

    if (verbose)
        printf("  NEUROGENESIS: activated %d from pool — %d dormant remaining\n",
               activated, remaining);

    /*
     * No auto-refill. The pool is finite.
     *
     * Biological parallel: hippocampal neurogenesis draws from a fixed
     * stem-cell niche. Once exhausted, no new neurons appear until the
     * niche itself regenerates (a much slower process). Infinite refill
     * caused runaway growth (100+ neurons) when a bad seed got stuck —
     * the network kept growing without ever converging.
     *
     * With a finite pool, the maximum network size is determined at
     * creation time (n_active + n_pool). When all dormant slots are
     * consumed, neurogenesis events become no-ops and training must
     * either converge with the current neurons or admit failure.
     *
     * If you need a larger network, create it with a bigger pool.
     */

    /* Fresh expectations: RPE fires strongly after structural change */
    net->rpe_baseline = 0.0f;
    net->growth_gen++;
    return activated;
}

/*
 * Mature all immature neurons by one epoch.
 *
 * Each epoch, every immature neuron's maturity increases by
 * maturation_rate. At maturation_rate=0.05, it takes 20 epochs
 * for a neuron to go from 0.0 to 1.0 (fully mature).
 *
 * When maturity reaches 1.0, the neuron's state changes from
 * IMMATURE to MATURE. It's now fully contributing to the network.
 *
 * BIOLOGICAL PARALLEL: In the adult hippocampus, newborn neurons
 * take 4-8 weeks to fully mature. During this time:
 * - Week 1-2: Neuron migrates to its destination, extends axon
 * - Week 2-3: Dendritic tree grows, first synapses form
 * - Week 3-4: Synapses strengthen, neuron becomes more excitable
 * - Week 4-8: Full integration, indistinguishable from old neurons
 *
 * Our maturity ramp from 0→1 compresses this timeline into epochs.
 */
void network_mature(Network *net) {
    for (int l = 0; l < net->n_layers; l++) {
        Layer *layer = &net->layers[l];
        for (int i = 0; i < layer->count; i++) {
            Neuron *n = &layer->neurons[i];

            if (n->state == NEURON_IMMATURE) {
                n->age++;
                n->maturity += net->maturation_rate;

                if (n->maturity >= 1.0f) {
                    n->maturity = 1.0f;
                    n->state = NEURON_MATURE;
                }
            } else if (n->state == NEURON_MATURE) {
                n->age++;
            }
            /* COMMITTED neurons don't age — they're timeless */
        }
    }
}

/*
 * Commit all mature neurons — make them permanent.
 *
 * Call this after the network has successfully learned a task.
 * All MATURE neurons become COMMITTED, meaning they can NEVER
 * be killed. They hold the knowledge of what was just learned.
 *
 * Even if the network later faces a different task and these
 * neurons go quiet (low activity), they persist. If the original
 * task is encountered again, they're ready.
 *
 * BIOLOGICAL PARALLEL: Long-term memory consolidation. When a
 * memory is formed, the synapses involved undergo structural
 * changes (new proteins are synthesized, dendritic spines grow
 * larger and more stable). This makes the memory resistant to
 * disruption. Our COMMITTED state is the computational equivalent:
 * "this neuron's knowledge is now permanent."
 *
 * WHY THIS PREVENTS CATASTROPHIC FORGETTING:
 * In traditional networks, learning task B overwrites the weights
 * learned for task A. Our approach: task A neurons are committed.
 * Task B gets NEW neurons. Old weights are untouched. The network
 * can solve both tasks simultaneously because the knowledge lives
 * in different (non-overlapping) neurons.
 */
void network_commit(Network *net) {
    /*
     * Activity-based commitment — biological apoptosis.
     *
     * In the real brain, ~50% of newborn neurons die before maturity.
     * Only neurons that integrate into active circuits survive.
     * We mimic this: only MATURE neurons with high enough activity
     * get promoted to COMMITTED. Low-activity neurons are killed.
     *
     * The threshold is the MEAN activity across all mature neurons.
     * Neurons below the mean are considered "not pulling their weight"
     * and are recycled. This is a simple but effective heuristic:
     * - In a well-trained network, useful neurons fire strongly → high activity
     * - Redundant/useless neurons barely fire → low activity → die
     *
     * IMPORTANT: We don't actually remove dead neurons from the layer
     * array (that would invalidate weight indices in downstream layers).
     * Instead, dead neurons keep their slot but their state is DORMANT
     * and their output is forced to 0 in neuron_forward(). Their weights
     * are freed to reclaim memory. The slot can be reused by future
     * neurogenesis events.
     */
    int newly_committed = 0;
    int newly_dead = 0;

    /* First pass: compute mean activity across all mature neurons */
    float total_activity = 0.0f;
    int mature_count = 0;
    for (int l = 0; l < net->n_layers; l++) {
        Layer *layer = &net->layers[l];
        for (int i = 0; i < layer->count; i++) {
            Neuron *n = &layer->neurons[i];
            if (n->state == NEURON_MATURE) {
                total_activity += n->activity;
                mature_count++;
            }
        }
    }

    if (mature_count == 0) return;  /* Nothing to commit */

    /*
     * Threshold = 10% of the most active neuron.
     *
     * WHY NOT THE MEAN?
     * Mean-based apoptosis kills the bottom half by definition — even
     * neurons with activity=0.22 die if the mean is 0.30.  In a small
     * trained network almost every neuron is contributing; killing half
     * of them by a statistical rule destroys knowledge the network
     * actually uses.
     *
     * WHY 10% OF MAX?
     * This targets truly idle neurons — ones that barely fired across
     * thousands of training steps.  A neuron at 10% of max is borderline
     * useless; a neuron at 50% of max is an active participant.
     * In the biological brain, apoptosis eliminates neurons that never
     * integrated into any circuit, not ones that fire less than average.
     */
    float max_activity = 0.0f;
    for (int l = 0; l < net->n_layers; l++) {
        Layer *layer = &net->layers[l];
        for (int i = 0; i < layer->count; i++) {
            if (layer->neurons[i].state == NEURON_MATURE &&
                layer->neurons[i].activity > max_activity) {
                max_activity = layer->neurons[i].activity;
            }
        }
    }
    float threshold = max_activity * 0.1f;

    /* Second pass: commit or kill, but guarantee one survivor per layer */
    for (int l = 0; l < net->n_layers; l++) {
        Layer *layer = &net->layers[l];

        /*
         * Safety net: find the most active MATURE neuron in this layer.
         * If ALL mature neurons fall below threshold (can happen when
         * activity is uniformly low), we must still keep the best one —
         * otherwise the layer is completely dead and the network loses
         * all representation at that depth. This mirrors the brain's
         * minimum circuit viability constraint: even a struggling circuit
         * retains its most-used pathway.
         */
        float best_activity = -1.0f;
        int   best_idx      = -1;
        for (int i = 0; i < layer->count; i++) {
            if (layer->neurons[i].state == NEURON_MATURE &&
                layer->neurons[i].activity > best_activity) {
                best_activity = layer->neurons[i].activity;
                best_idx      = i;
            }
        }

        for (int i = 0; i < layer->count; i++) {
            Neuron *n = &layer->neurons[i];
            if (n->state != NEURON_MATURE) continue;

            /* Always commit the most-active neuron in the layer, even if
             * it falls below the global mean. Without this, an unlucky
             * weight initialisation can kill every hidden neuron, leaving
             * the network unable to compute anything non-linear. */
            bool is_layer_champion = (i == best_idx);

            if (n->activity >= threshold || is_layer_champion) {
                n->state = NEURON_COMMITTED;
                net->pool.total_committed++;
                newly_committed++;
            } else {
                /*
                 * Apoptosis — programmed cell death.
                 * Neuron was not useful enough and is not the layer's best.
                 */
                n->state = NEURON_DORMANT;
                n->last_output = 0.0f;
                n->maturity = 0.0f;
                n->activity = 0.0f;
                pool_kill(&net->pool);
                newly_dead++;
            }
        }
    }

    if (newly_committed > 0 || newly_dead > 0) {
        printf("  COMMIT: %d neurons now hold permanent knowledge, "
               "%d neurons died (apoptosis, threshold=%.4f)\n",
               newly_committed, newly_dead, threshold);
    }
}

/*
 * network_commit_hidden — commit only hidden (non-output) layer neurons.
 *
 * Used by alphabet pretraining to freeze the hidden representations that
 * encode sequence/position knowledge while leaving output neurons plastic
 * for word associations.  Output neurons must stay uncommitted so that
 * alpha_delta_rescue can adjust them when new words are taught.
 *
 * Same activity-based apoptosis logic as network_commit(), just scoped
 * to layers 0..n_layers-2 (skipping the output layer).
 */
void network_commit_hidden(Network *net) {
    if (net->n_layers < 2) return;

    int newly_committed = 0;
    int newly_dead = 0;

    /* Compute max activity across hidden-layer mature neurons only */
    float max_activity = 0.0f;
    for (int l = 0; l < net->n_layers - 1; l++) {
        Layer *layer = &net->layers[l];
        for (int i = 0; i < layer->count; i++) {
            if (layer->neurons[i].state == NEURON_MATURE &&
                layer->neurons[i].activity > max_activity) {
                max_activity = layer->neurons[i].activity;
            }
        }
    }
    float threshold = max_activity * 0.1f;

    /* Commit or kill hidden neurons (skip output layer) */
    for (int l = 0; l < net->n_layers - 1; l++) {
        Layer *layer = &net->layers[l];
        float best_activity = -1.0f;
        int   best_idx      = -1;
        for (int i = 0; i < layer->count; i++) {
            if (layer->neurons[i].state == NEURON_MATURE &&
                layer->neurons[i].activity > best_activity) {
                best_activity = layer->neurons[i].activity;
                best_idx      = i;
            }
        }

        for (int i = 0; i < layer->count; i++) {
            Neuron *n = &layer->neurons[i];
            if (n->state != NEURON_MATURE) continue;
            bool is_layer_champion = (i == best_idx);

            if (n->activity >= threshold || is_layer_champion) {
                n->state = NEURON_COMMITTED;
                net->pool.total_committed++;
                newly_committed++;
            } else {
                n->state = NEURON_DORMANT;
                n->last_output = 0.0f;
                n->maturity = 0.0f;
                n->activity = 0.0f;
                pool_kill(&net->pool);
                newly_dead++;
            }
        }
    }

    if (newly_committed > 0 || newly_dead > 0) {
        printf("  COMMIT (hidden only): %d neurons committed, "
               "%d died (apoptosis, threshold=%.4f)\n",
               newly_committed, newly_dead, threshold);
    }
}

/*
 * network_commit_output — commit all MATURE output-layer neurons.
 *
 * Called by replay once all 26 RECALL + 26 VALIDATE associations are
 * correct and confident.  No apoptosis: every output neuron represents
 * a letter of the alphabet and is needed.  We simply promote MATURE → COMMITTED,
 * which freezes their weights so delta_rescue can no longer modify them.
 *
 * Returns the number of newly committed neurons (0 if already committed).
 */
int network_commit_output(Network *net) {
    if (net->n_layers < 1) return 0;

    Layer *ol = &net->layers[net->n_layers - 1];
    int newly = 0;

    for (int i = 0; i < ol->count; i++) {
        Neuron *n = &ol->neurons[i];
        if (n->state == NEURON_MATURE) {
            n->state = NEURON_COMMITTED;
            net->pool.total_committed++;
            newly++;
        }
    }

    return newly;
}

// ============================================================
// TRAINING — Simple output-layer delta rule
// ============================================================

/*
 * Train the network on a single input-target pair.
 *
 * IMPORTANT: Currently only updates the OUTPUT layer's weights.
 * Hidden layer neurons are NOT trained — they act as random feature
 * extractors. Their random projections transform the input into a
 * higher-dimensional space where the output neuron might find a
 * linear solution.
 *
 * THIS ACTUALLY WORKS for simple problems:
 * With enough random hidden neurons, the random projection is likely
 * to make even XOR linearly separable in the hidden feature space.
 * This is the principle behind "Extreme Learning Machines" (Huang 2006)
 * and "Random Kitchen Sinks" (Rahimi & Recht 2007).
 *
 * WHY NOT TRAIN HIDDEN LAYERS?
 * Training hidden layers requires credit assignment — figuring out
 * which hidden neuron is responsible for which part of the error.
 * Backpropagation solves this but we're not using backprop.
 * We'll implement our own training method for hidden layers later.
 *
 * RETURNS: Squared error for this sample (for tracking progress).
 */
float network_train_step(Network *net, const float *inputs, float target, float lr) {
    /* 1. Forward pass through entire network */
    float output;
    network_forward(net, inputs, &output);

    /* 2. Compute error */
    float error = target - output;

    /*
     * 3. Update OUTPUT layer only (delta rule).
     *
     * The output layer reads from the previous layer's outputs
     * (or raw inputs if no hidden layers). We apply the same
     * delta rule as our original perceptron:
     *   delta = error * f'(z)
     *   w_i += lr * delta * input_i
     *   bias += lr * delta
     */
    Layer *out_layer = &net->layers[net->n_layers - 1];
    Neuron *out_n = &out_layer->neurons[0];  /* Single output neuron */

    /* Get the inputs to the output layer */
    const float *prev_outputs;
    if (net->n_layers > 1) {
        /* Hidden layer(s) exist: output reads from last hidden layer */
        prev_outputs = net->layers[net->n_layers - 2].outputs;
    } else {
        /* No hidden layers: output reads raw inputs */
        prev_outputs = inputs;
    }

    /* Delta rule: delta = error * activation_derivative(z) */
    float deriv = activate_derivative(out_n->last_z, out_n->act);
    float delta = error * deriv;

    /* Update output neuron weights */
    for (int i = 0; i < out_n->n_weights; i++) {
        /*
         * Only update weights connected to ACTIVE inputs.
         * For committed neurons' outputs, always update (knowledge is stable).
         * For immature neurons' outputs, the value is small (maturity-scaled),
         * so the weight update is naturally small too — immature neurons
         * don't cause large weight changes in the output layer.
         */
        out_n->weights[i] += lr * delta * prev_outputs[i];
    }
    out_n->bias += lr * delta;

    return error * error;  /* Squared error */
}

/*
 * Automatic training with neurogenesis.
 *
 * This is the main training loop. It:
 * 1. Trains for up to max_epochs
 * 2. Every check_interval epochs, evaluates average error
 * 3. If error is low enough: task is learned, stop
 * 4. If error is stuck (hasn't improved enough): trigger neurogenesis
 * 5. Matures neurons every epoch
 *
 * The patience mechanism prevents premature growth:
 * - We track error improvement over patience_window epochs
 * - Only if error hasn't improved enough over that window do we grow
 * - This gives existing neurons a fair chance to learn before adding more
 *
 * NEUROGENESIS TRIGGER:
 * "Error stuck" = average error > threshold AND error hasn't decreased
 * by more than min_improvement over the patience window.
 * When triggered, new neurons are born immature and gradually integrate.
 *
 * PARAMETERS:
 * - inputs:     flat array of n_samples * net->n_inputs floats
 * - targets:    array of n_samples target values
 * - n_samples:  number of training examples
 * - max_epochs: maximum training epochs
 * - lr:         learning rate
 */
/*
 * Core training loop used by both network_auto_train (verbose) and the
 * benchmark (silent).  Returns the epoch at which the network solved the
 * task, or -1 if it never reached the success threshold within max_epochs.
 *
 * verbose=1 → print progress and neurogenesis events (CLI / interactive use)
 * verbose=0 → fully silent (benchmark use — keeps the result table clean)
 */
int network_auto_train_v(Network *net, const float *inputs, const float *targets,
                         int n_samples, int max_epochs, float lr, int verbose) {
    int   check_interval    = 100;
    int   patience          = 10;
    int   stall_count       = 0;
    float success_threshold = 0.01f;
    float min_improvement   = 0.001f;
    float prev_avg_error    = 1e9f;
    int   solved_epoch        = -1;
    int   conflicts_this_run  = 0;   /* count samples rejected this training run  */

    /*
     * Conflict threshold: if confidence > 0.8 AND prediction contradicts
     * the new label AND the network has committed neurons → reject that sample.
     * 0.8 means output must be > 0.9 or < 0.1 for a conflict to fire.
     */
    float conflict_threshold = 0.8f;

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        float total_error = 0.0f;

        for (int i = 0; i < n_samples; i++) {
            const float *sample_input   = inputs  + i * net->n_inputs;
            const float *sample_targets = targets + i * net->n_outputs;

            /*
             * CONFLICT GUARD: check whether this sample contradicts
             * what committed neurons have already learned. If the network
             * is highly confident AND the new label disagrees, skip the
             * weight update and warn the user (if verbose).
             *
             * This protects committed knowledge from being silently
             * overwritten by mislabelled or contradictory training data.
             */
            TrainVerdict verdict = network_check_data(net, sample_input,
                                                      sample_targets,
                                                      net->n_outputs,
                                                      conflict_threshold);
            if (verdict == VERDICT_CONFLICT || verdict == VERDICT_REVERIFY) {
                conflicts_this_run++;

                /*
                 * CRITICAL: still count this sample's error even though we
                 * skip the weight update. Without this, total_error only
                 * reflects the subset of non-rejected samples. A network
                 * that blocks all positive samples would drive their error
                 * to zero for 1 negative sample and falsely hit LEARNED.
                 * Every sample's squared error must feed the stall detector.
                 */
                float out_cur[MAX_OUTPUTS];
                network_forward(net, sample_input, out_cur);
                float se = 0.0f;
                for (int k = 0; k < net->n_outputs; k++) {
                    float ek = sample_targets[k] - out_cur[k];
                    se += ek * ek;
                }
                total_error += se / (float)net->n_outputs;

                if (verbose) {
                    if (verdict == VERDICT_CONFLICT) {
                        printf("  [CONFLICT] Sample %d: I'm confident the answer "
                               "is %.3f, but new label says %.1f — rejected.\n",
                               i, out_cur[0], sample_targets[0]);
                    } else {
                        printf("  [REVERIFY] Sample %d has conflicted %d+ times: "
                               "I believe %.3f, label insists %.1f.\n"
                               "  This is too consistent to be noise — possible "
                               "concept drift (the task changed).\n"
                               "  Action: call network_commit(), then retrain, "
                               "or call network_clear_conflicts() to reset.\n",
                               i, CONFLICT_REVERIFY_AT, out_cur[0], sample_targets[0]);
                    }
                }
                continue;
            }

            /*
             * Brain-native training: three-factor Hebbian with global
             * modulator (dopamine), WTA inhibition, and ACh attention gate.
             * Zero extra memory — no delta arrays, no gradient tape.
             */
            total_error += train_step_brain(net, sample_input,
                                            sample_targets, net->n_outputs, lr);
        }

        network_mature(net);

        float avg_error = total_error / (float)n_samples;

        if ((epoch + 1) % check_interval == 0) {
            float improvement = prev_avg_error - avg_error;

            /*
             * Verbose progress line — printed at every check_interval so a
             * long pre-train doesn't look hung. \r-overwritten so it stays
             * on one line; the LEARNED / stalled / converged branches below
             * print full lines that supersede it cleanly.
             *
             * Width-padded with spaces because subsequent overwrites of a
             * shorter string would otherwise leave trailing characters.
             */
            if (verbose) {
                int pct = (epoch + 1) * 100 / max_epochs;
                int bar_w = 30;
                int filled = pct * bar_w / 100;
                printf("\r  [");
                for (int b = 0; b < bar_w; b++) printf(b < filled ? "=" : " ");
                printf("] %3d%%  epoch %5d/%d  err=%.5f      ",
                       pct, epoch + 1, max_epochs, avg_error);
                fflush(stdout);
            }

            if (avg_error < success_threshold) {
                if (verbose) printf("\n");   /* close the \r progress line */
                if (verbose)
                    printf("  Epoch %d: error=%.6f — LEARNED!\n", epoch + 1, avg_error);

                /*
                 * AUTO-COMMIT: consolidate all mature neurons immediately.
                 *
                 * The brain doesn't require manual memory consolidation —
                 * LTP (Long-Term Potentiation) happens automatically when
                 * a circuit has fired reliably enough to reach synaptic
                 * stability. We mirror this: the moment error < threshold,
                 * every mature neuron's weights are locked in permanently.
                 *
                 * This means the system protects its own knowledge without
                 * any user action. The conflict guard in network_check_data
                 * will immediately start rejecting contradictory data after
                 * this point.
                 *
                 * NOTE: also clear the conflict log — the new committed state
                 * is a fresh baseline, old conflict counts from pre-convergence
                 * training are no longer meaningful.
                 */
                network_commit(net);
                network_clear_conflicts(net);
                if (verbose)
                    printf("  Consolidated: %d neurons committed — "
                           "knowledge is now permanent.\n",
                           network_committed_count(net));

                solved_epoch = epoch + 1;
                break;
            }

            if (improvement < min_improvement) {
                stall_count++;
                if (stall_count >= patience && avg_error > success_threshold * 10.0f) {
                    if (verbose) {
                        printf("\n  Epoch %d: error=%.6f (stalled %d checks) ",
                               epoch + 1, avg_error, stall_count);
                        network_neurogenesis(net);   /* prints growth summary */
                    } else {
                        network_neurogenesis_v(net, 0);   /* silent */
                    }
                    stall_count = 0;
                    prev_avg_error = avg_error;
                    continue;
                }
            } else {
                stall_count = 0;
            }

            prev_avg_error = avg_error;
        }
    }

    if (verbose && conflicts_this_run > 0) {
        printf("  [CONFLICT SUMMARY] %d sample(s) were rejected this run "
               "because they contradicted committed knowledge.\n"
               "  If this data is correct, commit was premature — "
               "reset and retrain from scratch.\n",
               conflicts_this_run);
    }

    return solved_epoch;
}

void network_auto_train(Network *net, const float *inputs, const float *targets,
                        int n_samples, int max_epochs, float lr) {
    network_auto_train_v(net, inputs, targets, n_samples, max_epochs, lr, 1);
}

// ============================================================
// NETWORK INFO — Counting and printing
// ============================================================

/*
 * Count total active neurons across all layers.
 * "Active" means IMMATURE, MATURE, or COMMITTED — anything that's
 * alive and part of the network, even if not yet fully mature.
 */
int network_neuron_count(Network *net) {
    int count = 0;
    for (int l = 0; l < net->n_layers; l++) {
        count += net->layers[l].count;
    }
    return count;
}

/*
 * Count committed neurons (those holding permanent knowledge).
 */
int network_committed_count(Network *net) {
    int count = 0;
    for (int l = 0; l < net->n_layers; l++) {
        Layer *layer = &net->layers[l];
        for (int i = 0; i < layer->count; i++) {
            if (layer->neurons[i].state == NEURON_COMMITTED) {
                count++;
            }
        }
    }
    return count;
}

/*
 * How confident is the network in its current prediction?
 *
 * Runs a forward pass and converts the output to a confidence score:
 *   confidence = |output - 0.5| * 2
 *
 * 0.0 = output = 0.5 (sitting exactly on the decision boundary,
 *       maximum uncertainty — like flipping a coin)
 * 1.0 = output = 0.0 or 1.0 (fully saturated — certain answer)
 *
 * Used by network_check_data to decide whether new training data
 * contradicts knowledge the network is already sure about.
 */
float network_confidence(Network *net, const float *inputs) {
    float outputs[MAX_OUTPUTS];
    network_forward(net, inputs, outputs);
    float min_conf = 1.0f;
    for (int k = 0; k < net->n_outputs; k++) {
        float c = fabsf(outputs[k] - 0.5f) * 2.0f;
        if (c < min_conf) min_conf = c;
    }
    return min_conf;
}

/*
 * FNV-1a hash over the raw bytes of the input float array.
 * Used to fingerprint training samples for the conflict log.
 */
static uint64_t hash_inputs(const float *inputs, int n) {
    uint64_t h = 14695981039346656037ULL;
    const uint8_t *b = (const uint8_t *)inputs;
    for (int i = 0; i < n * (int)sizeof(float); i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/*
 * Check whether a new (inputs, target) sample conflicts with what
 * the network currently knows, and track repeated conflicts for
 * reverification.
 *
 * Single conflict  → VERDICT_CONFLICT (likely bad label, reject).
 * Same conflict CONFLICT_REVERIFY_AT times → VERDICT_REVERIFY
 *   (too consistent to be noise — possible concept drift).
 *
 * NOTE: We intentionally do NOT gate on committed neurons.
 *
 * Early design gated on committed_count > 0, reasoning: "only protect
 * locked-in knowledge." This broke real use: a network trained to
 * convergence (output 0.001 for XOR (0,0)) should reject "train 0 0 1"
 * even if the user never typed `commit`. Confidence IS the gate.
 *
 * A network with confidence > 0.8 means output > 0.9 or < 0.1.
 * At that point the weights are already strongly organised around the
 * correct answer. Random weights start near 0.5 (confidence ≈ 0) and
 * only exceed the threshold after genuine convergence on that sample,
 * so the guard naturally activates at the right time without requiring
 * an explicit commit step.
 *
 * The committed-neurons concept still matters for weight freezing in
 * train_step_brain (committed weights never update), but should not
 * gate the conflict check here.
 */
TrainVerdict network_check_data(Network *net, const float *inputs,
                                const float *targets, int n_targets,
                                float conflict_threshold) {
    /*
     * Gate on committed neurons first.
     *
     * A fresh network with no committed neurons must never reject training
     * data. Random initial weights can produce confident outputs (near 0 or 1)
     * by chance — this would block valid samples before any learning has
     * happened, causing "LEARNED!" false positives from a single non-blocked
     * sample driving its error to zero while the rest are rejected.
     *
     * This check was removed once before ("confidence alone is the gate")
     * but that was wrong. The correct sequence is:
     *   1. No commits → accept everything (network is still forming beliefs)
     *   2. Auto-commit on convergence (network_auto_train_v handles this)
     *   3. After commit → confidence gate is meaningful (weights are trained)
     *
     * Now that network_auto_train_v auto-commits on LEARNED, committed_count
     * is always > 0 by the time real user interaction happens.
     */
    if (network_committed_count(net) == 0) return VERDICT_LEARN;

    float outputs[MAX_OUTPUTS];
    network_forward(net, inputs, outputs);

    /*
     * Check EACH output for conflict. A conflict fires when:
     *   confidence_k > threshold  AND  round(outputs[k]) != round(targets[k])
     * Any single conflicting output triggers the full conflict flow.
     */
    bool any_conflict = false;
    for (int k = 0; k < n_targets; k++) {
        float confidence = fabsf(outputs[k] - 0.5f) * 2.0f;
        if (confidence <= conflict_threshold) continue;
        int predicted = (outputs[k] > 0.5f) ? 1 : 0;
        int expected  = (targets[k]  > 0.5f) ? 1 : 0;
        if (predicted != expected) { any_conflict = true; break; }
    }
    if (!any_conflict) return VERDICT_LEARN;

    /*
     * Genuine conflict — fingerprint inputs AND targets together.
     * Hash inputs combined with targets hash so different target sets
     * for the same input are tracked as separate conflict records.
     */
    uint64_t h_inputs  = hash_inputs(inputs,  net->n_inputs);
    uint64_t h_targets = hash_inputs(targets, n_targets);
    uint64_t h = h_inputs ^ (h_targets * 2654435761ULL);

    for (int i = 0; i < net->n_conflict_records; i++) {
        ConflictRecord *r = &net->conflict_log[i];
        if (r->input_hash == h && r->target_hash == h_targets) {
            r->count++;
            return (r->count >= CONFLICT_REVERIFY_AT)
                   ? VERDICT_REVERIFY : VERDICT_CONFLICT;
        }
    }

    /* New conflict pattern */
    if (net->n_conflict_records < CONFLICT_LOG_SIZE) {
        ConflictRecord *r = &net->conflict_log[net->n_conflict_records++];
        r->input_hash  = h;
        r->target_hash = h_targets;
        r->count       = 1;
    } else {
        /* Log full: evict oldest */
        memmove(&net->conflict_log[0], &net->conflict_log[1],
                (CONFLICT_LOG_SIZE - 1) * sizeof(ConflictRecord));
        ConflictRecord *r = &net->conflict_log[CONFLICT_LOG_SIZE - 1];
        r->input_hash  = h;
        r->target_hash = h_targets;
        r->count       = 1;
    }

    return VERDICT_CONFLICT;
}

/* Reset conflict log — call after intentional reset+retrain */
void network_clear_conflicts(Network *net) {
    net->n_conflict_records = 0;
    memset(net->conflict_log, 0, sizeof(net->conflict_log));
}

/*
 * Print the full network state — topology, neuron states, pool stats.
 *
 * This gives a complete picture of the network's current structure
 * and neurogenesis history. Useful for understanding how the network
 * grew to solve (or fail to solve) a problem.
 */
void network_print(Network *net) {
    const char *state_names[] = {"DORMANT", "IMMATURE", "MATURE", "COMMITTED"};

    printf("\n--- Network State ---\n");
    printf("  Inputs: %d | Outputs: %d | Layers: %d | Growth gen: %d\n",
           net->n_inputs, net->n_outputs, net->n_layers, net->growth_gen);
    printf("  Total neurons: %d (%d committed)\n",
           network_neuron_count(net), network_committed_count(net));

    for (int l = 0; l < net->n_layers; l++) {
        Layer *layer = &net->layers[l];
        bool is_output = (l == net->n_layers - 1);

        printf("\n  Layer %d (%s): %d neurons, %d inputs each\n",
               l, is_output ? "output" : "hidden", layer->count, layer->input_size);

        for (int i = 0; i < layer->count; i++) {
            Neuron *n = &layer->neurons[i];
            printf("    Neuron #%d [%s] maturity=%.2f activity=%.4f age=%d",
                   n->id, state_names[n->state], n->maturity, n->activity, n->age);

            /* Print weights for small layers (don't flood output for large ones) */
            if (n->n_weights <= 8) {
                printf(" w=[");
                for (int w = 0; w < n->n_weights; w++) {
                    printf("%.3f%s", n->weights[w], w < n->n_weights - 1 ? ", " : "");
                }
                printf("] bias=%.3f", n->bias);
            }
            printf("\n");
        }
    }

    printf("\n");
    pool_print(&net->pool);
    printf("---------------------\n\n");
}
