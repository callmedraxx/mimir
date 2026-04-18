/*
 * MIMIR - The Perceptron
 *
 * THE HISTORICAL CONTEXT:
 * Frank Rosenblatt built the Mark I Perceptron in 1958 at Cornell.
 * It was a physical machine with 400 photocells, potentiometers for weights,
 * and electric motors to adjust them. The New York Times reported it as a
 * machine that "will be able to walk, talk, see, write, reproduce itself
 * and be conscious of its existence."
 *
 * Then in 1969, Minsky & Papert published "Perceptrons" proving that a
 * single-layer perceptron CANNOT learn XOR (or any non-linearly-separable
 * function). This essentially killed neural network research for 15 years
 * (the "AI Winter"). What they didn't emphasize enough: multi-layer
 * perceptrons CAN learn XOR. The field eventually recovered with
 * backpropagation (Rumelhart, Hinton & Williams, 1986).
 *
 * THE MATHEMATICAL MODEL:
 *   A perceptron computes: y = activation(w · x + b)
 *   Where:
 *     x = input vector
 *     w = weight vector (learned)
 *     b = bias scalar (learned)
 *     · = dot product
 *     activation = non-linear function
 *
 * Geometrically, a perceptron defines a hyperplane in input space.
 * Everything on one side maps to 1, the other to 0. That's why it can
 * learn AND and OR (linearly separable) but not XOR (not linearly separable).
 *
 * THIS IS THE ATOM. Everything else in deep learning—CNNs, transformers,
 * GPT-4, diffusion models—is built from arrangements of this basic unit.
 */

#include "mimir.h"

/*
 * Create a perceptron with n_inputs inputs and the given activation function.
 *
 * Memory layout: We allocate a contiguous array of n_inputs floats for weights.
 * For a single perceptron this is trivial, but the pattern matters later:
 * contiguous memory = cache-friendly = fast.
 *
 * ALTERNATIVES FOR WEIGHT STORAGE:
 * - We could use a struct with named fields (w1, w2, ...) but that doesn't
 *   generalize to variable input sizes.
 * - We could use a linked list of weights. This would be TERRIBLE for
 *   performance—pointer chasing kills cache locality. Never do this.
 * - For quantized weights (later), we'll pack multiple weights into a single
 *   integer: four 8-bit weights in one uint32_t, or eight 4-bit weights.
 *   This is how llama.cpp achieves its speed.
 *
 * INITIALIZATION STRATEGY:
 * We use Xavier/Glorot initialization: scale = 1/sqrt(fan_in).
 * This keeps the variance of activations roughly constant across layers.
 *
 * WHY NOT ZERO INIT?
 * If all weights start at 0, every neuron in a layer computes the same thing.
 * Gradients are identical, so they update identically. The network can never
 * break this symmetry—it's equivalent to having a single neuron. Random init
 * breaks symmetry so each neuron can specialize.
 *
 * HYPOTHETICAL: "Structured initialization" — Instead of random weights, what
 * if we initialized weights to approximate specific known functions? For example,
 * if we know the target is likely monotonic, initialize weights with the same
 * sign. Or use SVD of the training data to set initial weight directions along
 * principal components. WHY IT MIGHT WORK: You're starting closer to the
 * solution, so training converges faster. WHY IT MIGHT NOT: The loss landscape
 * near the optimum often has many equally good solutions—starting near one
 * arbitrary solution isn't necessarily better than random. Also, you need to
 * see the data first, which adds a preprocessing step.
 */
Perceptron perceptron_create(int n_inputs, Activation act) {
    Perceptron p;                 /* Stack-allocated struct, weights heap-allocated */
    p.n_inputs = n_inputs;        /* Store input dimensionality */
    p.act = act;                  /* Store which activation function to use */

    /*
     * malloc for weight array. We use malloc (not calloc) because we're
     * about to overwrite every value with random_init anyway.
     * sizeof(float) = 4 bytes on virtually all modern platforms (IEEE 754).
     *
     * COULD THIS BE BETTER? We could use aligned_alloc(32, ...) to get
     * 32-byte-aligned memory for AVX2 operations (_mm256_load_ps requires
     * alignment, _mm256_loadu_ps doesn't but is slightly slower on some
     * architectures). For a 2-input perceptron, alignment doesn't matter.
     * For large layers later, it will.
     */
    p.weights = (float *)malloc(n_inputs * sizeof(float));

    /*
     * Bias starts at zero. This is standard practice—the random weights
     * provide enough symmetry breaking. Some argue bias should also be
     * random, but empirically it makes no significant difference.
     *
     * EXCEPTION: For ReLU networks, initializing bias slightly positive
     * (e.g., 0.01) can help prevent dead neurons at the start of training.
     * We don't do this for the perceptron since we're using sigmoid here.
     */
    p.bias = 0.0f;

    /*
     * Xavier scale: 1/sqrt(n_inputs).
     *
     * The intuition: if inputs have variance 1 and weights have variance
     * 1/n, then the output z = sum(w_i * x_i) has variance ~1 (by the
     * variance of a sum of independent products). This keeps the signal
     * from growing or shrinking as it passes through layers.
     *
     * For a 2-input perceptron: scale = 1/sqrt(2) ≈ 0.707
     * So weights start in roughly [-1.4, 1.4] (within 2 standard deviations).
     *
     * ALTERNATIVES:
     * - He init: sqrt(2/n_inputs) — better for ReLU (accounts for half
     *   the outputs being zeroed)
     * - LeCun init: 1/sqrt(n_inputs) — same as Xavier for fan_in only
     * - Fixup init: Scales down residual branch weights by 1/sqrt(L)
     *   where L is network depth. Enables training very deep networks
     *   without normalization layers.
     */
    float scale = 1.0f / sqrtf((float)n_inputs);
    random_init(p.weights, n_inputs, scale);

    return p;  /* Return by value (struct copy). For large layers, we'll use pointers. */
}

/*
 * Forward pass: compute the perceptron's output for given inputs.
 *
 * This is the fundamental computation of ALL neural networks:
 *   1. Linear transformation: z = w·x + b (dot product + bias)
 *   2. Non-linear activation: y = f(z)
 *
 * Every layer in every neural network does exactly this. A 175-billion
 * parameter GPT-3 is just this operation repeated billions of times with
 * different weights.
 *
 * COMPUTATIONAL COST: O(n_inputs) for the dot product. For our 2-input
 * perceptron: 2 multiplies, 1 add, 1 activation. For a 12288-dimensional
 * GPT-3 layer: 12288 multiplies, 12287 adds, 1 activation. This is why
 * matrix multiplication speed is the bottleneck of LLMs.
 *
 * COULD THIS BE FASTER?
 * For 2 inputs? No. The compiler will unroll this loop entirely.
 * For large n_inputs? Yes:
 *   - SIMD (AVX2): Process 8 floats at once. ~4-8x speedup.
 *   - SIMD (AVX-512): Process 16 floats at once. ~8-16x speedup.
 *   - Quantized: Pack weights as int8, use VNNI instructions for
 *     8-bit dot products. ~4x over float SIMD.
 *   We'll add all of these as we scale up.
 *
 * HYPOTHETICAL: "Skip computation for near-zero weights" — If a weight is
 * very small (|w| < epsilon), skip that multiply-add entirely. This is
 * essentially dynamic sparsity. WHY IT MIGHT WORK: Many trained weights
 * ARE near-zero (that's why pruning works). Skipping them saves computation.
 * WHY IT MIGHT NOT: The branch (if statement) to check each weight costs
 * more than just doing the multiply on modern CPUs with deep pipelines.
 * Branch misprediction penalty is ~15-20 cycles, while a multiply is 3-5
 * cycles. You'd need very high sparsity (>80%) for the skip to pay off.
 * BETTER APPROACH: Static pruning—remove near-zero weights permanently
 * and use sparse matrix formats (CSR/CSC) that skip zeros for free.
 */
float perceptron_forward(Perceptron *p, const float *inputs) {
    /*
     * Start with bias. Adding bias first means we do n multiplies and n+1 adds
     * total. We could also add it after the loop—same result, same cost.
     * Adding it first is slightly more readable.
     */
    float z = p->bias;

    /*
     * The dot product: core of the forward pass.
     * This loop is where neural networks spend most of their time.
     * For n=2 the compiler will fully unroll this into:
     *   z += weights[0] * inputs[0];
     *   z += weights[1] * inputs[1];
     * which is optimal.
     */
    for (int i = 0; i < p->n_inputs; i++) {
        z += p->weights[i] * inputs[i];
    }

    /*
     * Apply activation function. This is what makes it a neuron instead
     * of just a linear function.
     *
     * After activation, the output is in:
     * - ACT_STEP:    {0, 1}         (binary)
     * - ACT_SIGMOID: (0, 1)         (continuous, interpretable as probability)
     * - ACT_RELU:    [0, +infinity) (unbounded positive)
     */
    return activate(z, p->act);
}

/*
 * Train the perceptron on a single input-target pair.
 *
 * This implements the delta rule (Widrow-Hoff, 1960), which is a
 * generalization of the original perceptron learning rule to
 * differentiable activation functions.
 *
 * THE LEARNING RULE:
 *   error = target - output
 *   delta = error * f'(z)            (f' is activation derivative)
 *   w_i  += learning_rate * delta * x_i
 *   b    += learning_rate * delta
 *
 * WHY THIS WORKS (intuitively):
 * - If output < target (error positive), we want to increase output.
 *   We increase weights for positive inputs and decrease for negative.
 * - The learning rate controls step size. Too large = oscillation.
 *   Too small = slow convergence.
 * - The derivative f'(z) scales the update by how sensitive the output
 *   is to changes in z. Near the sigmoid's flat tails, small updates
 *   (because the output barely changes). Near the center, large updates.
 *
 * CONVERGENCE GUARANTEE:
 * For the step activation + linearly separable data, the Perceptron
 * Convergence Theorem (Novikoff, 1962) PROVES convergence in finite steps.
 * For sigmoid + gradient descent, we converge to a local minimum (which
 * for a single perceptron is also the global minimum, since the loss
 * landscape is convex).
 *
 * ALTERNATIVES TO THIS UPDATE RULE:
 * - Perceptron rule: w += lr * error * x (no derivative). Simpler, works
 *   only with step activation. This is Rosenblatt's original.
 * - Gradient descent on MSE: Identical to what we're doing, but derived
 *   from minimizing (target - output)^2 via calculus. Our "error * derivative"
 *   IS the gradient of MSE.
 * - Gradient descent on cross-entropy: -[t*log(y) + (1-t)*log(1-y)].
 *   Better for classification because it penalizes confident wrong answers
 *   more heavily. The gradient simplifies to (y - t) for sigmoid output,
 *   which removes the derivative term and avoids the vanishing gradient
 *   problem of sigmoid. We'll use this for the full model.
 * - Adam optimizer: Adaptive per-parameter learning rates with momentum.
 *   Massive improvement over plain SGD for deep networks. Not needed for
 *   a perceptron (it converges fine with SGD).
 *
 * HYPOTHETICAL: "Anti-Hebbian unlearning" — After training, run the network
 * on its OWN outputs and apply negative learning (decrease weights that fire
 * together). Inspired by how the brain consolidates during sleep. Hopfield
 * networks use this to increase storage capacity. WHY IT MIGHT WORK: Could
 * reduce overfitting by making the network "forget" noise patterns. Similar
 * to how dropout works but applied post-training. WHY IT MIGHT NOT: For a
 * perceptron it's pointless (no overfitting on 4 examples). For larger
 * networks, the dynamics are chaotic and could destroy learned features.
 * Interestingly, recent diffusion model research uses a similar idea
 * (denoising = anti-learning on noise).
 *
 * LEARNING RATE CHOICE:
 * We pass lr=1.0 from main.c. This seems high but works because:
 * 1. Sigmoid's derivative maxes at 0.25, so effective lr is at most 0.25
 * 2. Our inputs are in {0, 1}, so updates are bounded
 * 3. We have only 4 training examples, so there's no noise from sampling
 * For larger networks, lr=0.001 to lr=0.01 is typical with Adam.
 */
void perceptron_train(Perceptron *p, const float *inputs, float target, float lr) {
    /* --- Forward pass (duplicated from perceptron_forward) --- */
    /*
     * We recompute the forward pass here instead of calling perceptron_forward
     * because we need the pre-activation value z (for the derivative).
     * perceptron_forward only returns the post-activation output.
     *
     * COULD THIS BE BETTER? Yes—we could cache z in the struct from the
     * forward pass. This is exactly what full neural network frameworks do
     * (PyTorch calls it "saving for backward"). For a perceptron the
     * recomputation cost is negligible. For a billion-parameter model,
     * the choice between caching and recomputation is a memory/compute
     * tradeoff called "gradient checkpointing."
     */
    float z = p->bias;
    for (int i = 0; i < p->n_inputs; i++) {
        z += p->weights[i] * inputs[i];
    }

    /* Get the output and the derivative at the pre-activation value */
    float output = activate(z, p->act);
    float deriv = activate_derivative(z, p->act);

    /*
     * Compute error (target - output).
     * This is the simplest form of loss gradient. For MSE loss:
     *   L = 0.5 * (target - output)^2
     *   dL/d_output = -(target - output) = output - target
     * We use (target - output) and absorb the sign into the update direction.
     *
     * Note: |error| indicates how wrong we are. For a well-trained perceptron
     * on AND/OR, errors should be < 0.1 on all examples.
     */
    float error = target - output;

    /*
     * delta = error * derivative
     * This is the chain rule in action:
     *   dL/dz = dL/dy * dy/dz = error * f'(z)
     * delta tells us "how should z change to reduce the error?"
     */
    float delta = error * deriv;

    /* --- Backward pass: update weights and bias --- */
    /*
     * Weight update: w_i += lr * delta * x_i
     *
     * The chain rule continues:
     *   dL/dw_i = dL/dz * dz/dw_i = delta * x_i
     * because z = sum(w_i * x_i) + b, so dz/dw_i = x_i.
     *
     * Note: if x_i = 0, the weight doesn't update (input had no contribution
     * to the error). If x_i = 1, the weight gets the full delta update.
     * This makes intuitive sense: only adjust weights for inputs that were
     * "active" and could have contributed to the error.
     */
    for (int i = 0; i < p->n_inputs; i++) {
        p->weights[i] += lr * delta * inputs[i];
    }

    /*
     * Bias update: b += lr * delta
     *
     * The bias gradient is just delta because dz/db = 1.
     * The bias is like an input that's always 1—it shifts the decision
     * boundary without depending on any input feature.
     *
     * INTERESTING FACT: Some architectures omit the bias entirely
     * (many modern transformers do). If you have layer normalization,
     * the bias is redundant because LayerNorm has its own shift parameter.
     */
    p->bias += lr * delta;
}

/*
 * Print the perceptron's state. Useful for debugging and understanding
 * what the perceptron has learned.
 *
 * FOR AND GATE, we expect:
 * - Both weights positive and roughly equal (both inputs contribute equally)
 * - Bias large and negative (need BOTH inputs to overcome the negative bias)
 *
 * FOR OR GATE, we expect:
 * - Both weights positive and roughly equal
 * - Bias slightly negative (only need ONE input to overcome the bias)
 *
 * FOR XOR GATE, we expect:
 * - Confused/small weights (it can't find a solution because none exists)
 */
void perceptron_print(Perceptron *p) {
    const char *act_names[] = {"step", "sigmoid", "relu"};
    printf("Perceptron(%d inputs, %s)\n", p->n_inputs, act_names[p->act]);
    printf("  weights: [");
    for (int i = 0; i < p->n_inputs; i++) {
        printf("%.4f%s", p->weights[i], i < p->n_inputs - 1 ? ", " : "");
    }
    printf("]\n");
    printf("  bias: %.4f\n", p->bias);
}

/*
 * Free the perceptron's allocated memory.
 * Set pointer to NULL after freeing to catch use-after-free bugs.
 * In C, using freed memory is undefined behavior—it might work, crash,
 * or silently corrupt data. NULL at least gives a clean segfault.
 *
 * ALTERNATIVE: Use a pool allocator that allocates all network memory
 * from a single large block and frees it all at once. Much faster than
 * individual malloc/free and avoids fragmentation. We'll consider this
 * when we have hundreds of layers.
 */
void perceptron_free(Perceptron *p) {
    free(p->weights);
    p->weights = NULL;
}
