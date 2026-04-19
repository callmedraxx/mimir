/*
 * MIMIR - Core math and random number generation
 *
 * This file contains the mathematical foundations that every neural network
 * needs: random number generation (for weight initialization) and activation
 * functions (for introducing non-linearity).
 *
 * RESEARCH NOTE: These two things—initialization and activation—are arguably
 * the most studied aspects of deep learning after architecture design itself.
 * Bad initialization can make a network untrainable. Bad activation functions
 * caused "dead neuron" problems for years before ReLU was popularized.
 */

/*
 * _USE_MATH_DEFINES: Required on some compilers (MSVC) to expose M_PI.
 * On GCC/Linux with -std=c11, M_PI isn't guaranteed by the standard,
 * so we define it manually below as a fallback.
 *
 * ALTERNATIVE: We could use our own constant everywhere, but #define M_PI
 * is the universal convention. No reason to fight it.
 */
#define _USE_MATH_DEFINES
#include "mimir.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// RANDOM NUMBER GENERATOR
// ============================================================

/*
 * We use xorshift64 instead of stdlib's rand() for several reasons:
 *
 * 1. rand() is not thread-safe on many platforms (uses global state)
 * 2. rand() has notoriously bad statistical properties on some implementations
 *    (e.g., low-order bits cycling with short periods on older glibc)
 * 3. xorshift64 gives us a 2^64-1 period, which is more than enough
 * 4. It's extremely fast: just 3 XOR + shift operations
 *
 * ALTERNATIVES CONSIDERED:
 * - Mersenne Twister (MT19937): Better statistical properties, but ~2.5KB of
 *   state and slower. Overkill for weight init. Used in NumPy/Python's random.
 * - PCG (Permuted Congruential Generator): Arguably better than xorshift in
 *   every way—smaller state, better statistical properties, similar speed.
 *   We could switch to this. The reason we didn't: xorshift is simpler to
 *   understand and implement, and for weight initialization the quality
 *   difference is negligible.
 * - /dev/urandom: Cryptographically secure but WAY too slow for generating
 *   millions of weights. We only need statistical randomness, not security.
 *
 * HYPOTHETICAL IDEA: "Quasi-random" initialization using low-discrepancy
 * sequences (Sobol, Halton) instead of pseudo-random. The theory: quasi-random
 * sequences fill the space more uniformly than pseudo-random ones. This MIGHT
 * give better initial weight coverage and faster convergence. Some papers
 * (e.g., "Quasi-Monte Carlo Weight Initialization") suggest marginal benefits.
 * WHY IT MIGHT WORK: Avoids clusters of similar weights that random init can
 * produce. WHY IT MIGHT NOT: The loss landscape is so complex that the uniform
 * coverage advantage gets washed out after a few gradient steps.
 */
static uint64_t rng_state = 0;  /* Global state. Fine for single-threaded init. */

static uint64_t xorshift64(void) {
    /*
     * Seed from time if not yet initialized.
     * CAVEAT: time(NULL) has second resolution, so two programs started
     * in the same second get the same sequence. For reproducible experiments
     * you'd want to set rng_state explicitly. We'll add a seed function later.
     */
    if (rng_state == 0) rng_state = (uint64_t)time(NULL);

    /*
     * The three magic shifts (13, 7, 17) are one of Marsaglia's recommended
     * triplets from his 2003 paper "Xorshift RNGs". Not all triplets work—
     * many produce sequences with poor statistical properties. These specific
     * values were chosen because they produce a full-period generator.
     *
     * CAN THIS BE DONE BETTER? Yes—xoshiro256** by Blackman & Vigna (2018)
     * is the modern successor, with better statistical properties. But for
     * weight init, it genuinely doesn't matter.
     */
    rng_state ^= rng_state << 13;  /* Left shift and XOR */
    rng_state ^= rng_state >> 7;   /* Right shift and XOR */
    rng_state ^= rng_state << 17;  /* Left shift and XOR */
    return rng_state;
}

/*
 * Convert the 64-bit integer to a float in [0, 1).
 * We mask to 32 bits first because float only has 23 bits of mantissa anyway—
 * using all 64 bits would be wasted precision.
 *
 * ALTERNATIVE: We could use the bit-manipulation trick where you set the
 * exponent bits directly (union { uint32_t i; float f; } with exponent = 127)
 * to get a float in [1, 2) then subtract 1. This avoids the division and is
 * slightly faster. We use division here because it's clearer.
 */
static float random_uniform(void) {
    return (float)(xorshift64() & 0xFFFFFFFF) / (float)0xFFFFFFFF;
}

/*
 * Public API: Set the RNG seed for reproducible experiments.
 *
 * For benchmarking, we need every training method to start with
 * IDENTICAL random weights. By seeding to the same value before
 * creating each network, we guarantee identical initialization.
 *
 * IMPORTANT: Seed must be non-zero (xorshift64 has an absorbing
 * state at 0 — it would output 0 forever). We silently fix this.
 */
void random_seed(uint64_t seed) {
    rng_state = seed ? seed : 1;
}

/*
 * Public wrapper for random_uniform().
 * Needed by training methods (NoProp, Perturbation) that require
 * random numbers during training, not just initialization.
 */
float random_uniform_f(void) {
    return random_uniform();
}

/*
 * Initialize weights from a normal distribution with mean=0, stddev=scale.
 *
 * Uses the Box-Muller transform to convert uniform random numbers to
 * normally distributed ones. This is the classic technique:
 *   z = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
 * where u1, u2 are uniform in (0, 1).
 *
 * WHY NORMAL DISTRIBUTION?
 * Research (Glorot & Bengio 2010, He et al. 2015) shows that the variance
 * of initial weights critically affects whether gradients vanish or explode
 * during training. Normal distribution with carefully chosen variance keeps
 * the signal magnitude stable across layers.
 *
 * WHY NOT UNIFORM DISTRIBUTION?
 * You actually can use uniform—Glorot uniform init samples from
 * U[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]. For a single
 * perceptron it barely matters. The difference becomes important in deep
 * networks where the Central Limit Theorem means layer outputs converge
 * to Gaussian anyway.
 *
 * ALTERNATIVES:
 * - Kaiming/He initialization: scale = sqrt(2/fan_in). Better for ReLU
 *   because ReLU kills half the distribution (negative values become 0),
 *   so you need 2x the variance to compensate.
 * - Orthogonal initialization: Initialize weight matrices as orthogonal
 *   matrices. Preserves gradient norms exactly. More complex but provably
 *   better for RNNs/deep networks.
 * - LSUV (Layer-Sequential Unit Variance): Initialize, then rescale each
 *   layer's weights so the output variance is exactly 1.0. Data-dependent
 *   but very effective.
 *
 * HYPOTHETICAL: "Lottery Ticket Initialization" — What if instead of random
 * init, we could predict which initial weights will matter? The Lottery Ticket
 * Hypothesis (Frankle & Carlin, 2018) showed sparse subnetworks exist at init
 * that can train to full accuracy. If we could FIND them cheaply at init time,
 * we could train a much smaller network. WHY IT MIGHT WORK: Pruning research
 * shows 90%+ of weights are unnecessary. WHY IT MIGHT NOT: Finding the winning
 * ticket currently requires training the full network first—a chicken-and-egg
 * problem. No one has a reliable way to identify important weights before training.
 */
void random_init(float *data, int n, float scale) {
    for (int i = 0; i < n; i++) {
        /*
         * 1e-10f added to u1 to avoid log(0) which is -infinity.
         * This is a standard numerical safety trick.
         */
        float u1 = random_uniform() + 1e-10f;
        float u2 = random_uniform();

        /*
         * Box-Muller transform. Generates pairs of independent standard
         * normal variables. We only use one (the cosine term). The sine
         * term would give us a second independent sample—we waste it here
         * for simplicity. In a production system, you'd cache the second
         * value (called the "polar" method optimization).
         *
         * CAN THIS BE DONE BETTER? The Ziggurat algorithm is ~3x faster
         * for generating normal deviates. It uses a precomputed table of
         * rectangles under the normal curve. We use Box-Muller because
         * it's self-contained and easy to understand. Weight init is a
         * one-time cost anyway.
         */
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);

        /* Scale the standard normal (mean=0, std=1) to our desired std */
        data[i] = z * scale;
    }
}

// ============================================================
// ACTIVATION FUNCTIONS
// ============================================================

/*
 * Activation functions introduce non-linearity into the network.
 * Without them, any stack of linear layers collapses to a single linear
 * transformation (because matrix multiplication is associative:
 * W3 * W2 * W1 * x = W_combined * x). Non-linearity is what gives
 * neural networks their expressive power.
 *
 * RESEARCH CONTEXT: The choice of activation function has been one of
 * the most impactful discoveries in deep learning history:
 * - 1958: Rosenblatt's Perceptron used step function
 * - 1986: Backpropagation paper used sigmoid (because it's differentiable)
 * - 2000s: tanh became popular (zero-centered, which helps optimization)
 * - 2010: ReLU (Nair & Hinton) revolutionized deep learning by solving
 *   the vanishing gradient problem
 * - 2017+: GELU, SiLU/Swish, Mish became popular in transformers
 *
 * HYPOTHETICAL: "Learned activation functions" — What if the activation
 * function itself was trainable? PReLU (parametric ReLU) does this minimally
 * with a learnable slope for negative inputs. Swish (x * sigmoid(beta * x))
 * has a learnable beta. Going further: use a small neural network AS the
 * activation function. WHY IT MIGHT WORK: Different parts of the network
 * might benefit from different non-linearities. WHY IT MIGHT NOT: Adds
 * parameters and computation for minimal gain—the specific shape of the
 * activation matters less than having SOME non-linearity. Also makes
 * the optimization landscape more complex.
 */

float activate(float x, Activation act) {
    switch (act) {
        case ACT_STEP:
            /*
             * The Heaviside step function: output 1 if input >= 0, else 0.
             * This is Rosenblatt's original (1958). It's simple and intuitive
             * but NOT differentiable at x=0, which makes gradient-based
             * training theoretically invalid. In practice we use it anyway
             * with the perceptron learning rule (which doesn't need gradients).
             *
             * WHY WE INCLUDE IT: Historical significance. The perceptron
             * convergence theorem guarantees this finds a solution IF one
             * exists (for linearly separable data).
             *
             * CAN'T BE DONE BETTER for the original perceptron algorithm—
             * it IS the original perceptron algorithm. But for gradient-based
             * training, literally any smooth function is better.
             */
            return x >= 0.0f ? 1.0f : 0.0f;

        case ACT_SIGMOID:
            /*
             * Sigmoid: sigma(x) = 1 / (1 + e^(-x))
             * Maps any real number to (0, 1). Smooth, differentiable,
             * with a beautiful derivative: sigma'(x) = sigma(x) * (1 - sigma(x))
             *
             * PROBLEMS:
             * 1. Vanishing gradients: For |x| > ~5, the gradient is near zero.
             *    In deep networks, gradients get multiplied layer by layer,
             *    so they shrink exponentially. This killed deep network training
             *    for decades.
             * 2. Not zero-centered: Output is always positive, which means
             *    gradients for weights are always the same sign, causing
             *    zig-zag optimization paths.
             * 3. expf() is expensive (~10-20 cycles vs 1 cycle for a multiply).
             *
             * WHY WE USE IT HERE: For a single perceptron learning logic gates,
             * none of these problems matter. Sigmoid is perfect for binary
             * classification (output interpretable as probability).
             *
             * ALTERNATIVES:
             * - tanh: Same shape but maps to (-1, 1). Zero-centered, so
             *   strictly better for hidden layers. tanh(x) = 2*sigmoid(2x) - 1.
             * - Hard sigmoid: max(0, min(1, 0.2x + 0.5)). Piecewise linear
             *   approximation. ~5x faster but slightly less accurate.
             *
             * HYPOTHETICAL: What about using a lookup table for sigmoid?
             * Precompute sigmoid for x in [-8, 8] at 0.001 resolution (16K entries).
             * Clamp inputs outside range to 0 or 1. WHY IT MIGHT WORK: Table
             * lookup + linear interpolation is ~3 cycles, much faster than expf.
             * WHY IT MIGHT NOT: Cache pressure in large networks could negate
             * the speed benefit. Modern CPUs also have fast exp approximations.
             * VERDICT: Actually used in production (TFLite uses lookup tables).
             */
            return 1.0f / (1.0f + expf(-x));

        case ACT_HDC_BIT:
            /* Same step shape as ACT_STEP, but used to mark bit-packed ±1
             * HDC neurons. The forward path in network.c computes z via
             * popcount-style ±1 weights, then falls through here. */
            return x > 0.0f ? 1.0f : 0.0f;

        case ACT_RELU:
            /*
             * ReLU: max(0, x)
             * The activation function that enabled modern deep learning.
             *
             * WHY IT'S SO GOOD:
             * 1. No vanishing gradient for positive inputs (gradient is exactly 1)
             * 2. Incredibly fast: a single comparison
             * 3. Induces sparsity: ~50% of neurons output 0, which is like
             *    free regularization and makes computation sparse
             * 4. Biologically plausible (neurons either fire or don't)
             *
             * PROBLEMS:
             * 1. "Dying ReLU": If a neuron's input is always negative, its
             *    gradient is always 0, so it never learns. Once dead, always dead.
             * 2. Not zero-centered (same issue as sigmoid, but less severe)
             * 3. Unbounded output can cause instability
             *
             * ALTERNATIVES:
             * - Leaky ReLU: max(0.01x, x). Fixes dying ReLU by allowing
             *   small gradients for negative inputs.
             * - ELU: x if x>0, alpha*(exp(x)-1) if x<=0. Smooth, zero-centered.
             * - GELU: x * Phi(x) where Phi is the normal CDF. Used in BERT/GPT.
             *   Basically a smooth approximation of ReLU that also considers
             *   the input's magnitude. We'll use this in our transformer later.
             *
             * CAN THIS BE DONE BETTER? For a perceptron, ReLU is overkill
             * (sigmoid is fine). ReLU's advantages show up in deep networks.
             * But we include it as a preview of what's to come.
             */
            return x > 0.0f ? x : 0.0f;

        default:
            return x;  /* Linear / identity activation */
    }
}

/*
 * Derivative of the activation function.
 * This is the key to backpropagation: the chain rule requires us to
 * multiply by the local gradient at each step. The derivative tells us
 * "how much does the output change when the input changes?"
 *
 * IMPORTANT INSIGHT: For gradient-based learning, we need the derivative
 * of EVERYTHING. This is why step function is bad (derivative is 0 or
 * undefined everywhere), sigmoid is OK (smooth derivative), and ReLU is
 * great (gradient is exactly 0 or 1—no vanishing, no computation).
 */
float activate_derivative(float x, Activation act) {
    switch (act) {
        case ACT_STEP:
            /*
             * The step function has derivative 0 everywhere except at x=0
             * where it's undefined (Dirac delta). We return 1.0 as a hack
             * to make the perceptron learning rule work—it's technically not
             * the derivative, but the learning rule doesn't actually use
             * derivatives (it predates backpropagation).
             *
             * ALTERNATIVE: Return 0.0 and use the classic perceptron update
             * rule (w += lr * error * x) without the derivative term. This
             * is more mathematically honest but our training function
             * multiplies by the derivative, so we'd need to restructure.
             */
            return 1.0f;

        case ACT_SIGMOID: {
            /*
             * sigma'(x) = sigma(x) * (1 - sigma(x))
             *
             * Beautiful closed-form derivative. Maximum value is 0.25 (at x=0).
             * This means EVERY layer multiplies the gradient by at most 0.25,
             * so after 10 layers: 0.25^10 = 0.00000095. Gradients vanish.
             * This is the fundamental problem that killed deep sigmoid networks.
             *
             * NOTE: We recompute sigma(x) here. If we had cached the forward
             * pass output, we could compute the derivative as out * (1 - out)
             * without calling expf again. We'll optimize this in the layer
             * implementation.
             */
            float s = 1.0f / (1.0f + expf(-x));
            return s * (1.0f - s);
        }

        case ACT_RELU:
            /*
             * ReLU derivative: 1 if x > 0, 0 otherwise.
             * At x = 0 it's technically undefined (subgradient), but we
             * define it as 0 by convention. This doesn't matter in practice
             * because hitting exactly 0.0f in floating point is rare.
             *
             * THIS IS WHY RELU SOLVED DEEP LEARNING: The gradient is either
             * 0 or 1. No multiplication by small numbers means no vanishing.
             * The gradient flows straight through active neurons unchanged.
             */
            return x > 0.0f ? 1.0f : 0.0f;

        default:
            return 1.0f;  /* Identity derivative */
    }
}
