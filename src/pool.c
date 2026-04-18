/*
 * MIMIR - Neuron Pool (The Stem Cell Niche)
 *
 * In the brain, neural stem cells live in specific regions (the
 * subventricular zone and hippocampal dentate gyrus). When the brain
 * needs new neurons — due to learning, injury, or enriched environment —
 * stem cells divide and produce new neurons on demand.
 *
 * Our NeuronPool serves the same role. It's the bookkeeper that tracks
 * every neuron birth and death across the entire network's lifetime.
 * Every neuron that ever exists passes through this pool.
 *
 * CURRENT IMPLEMENTATION: Statistics tracker only. The actual memory
 * allocation happens in the Layer (via realloc). This is fine for
 * thousands of neurons.
 *
 * FUTURE OPTIMIZATION (when we need billions):
 * Convert this to a chunk-based pool allocator:
 *   - Pre-allocate blocks of 1024+ neurons at a time
 *   - Hand out neurons from the current block
 *   - When a block is full, allocate a new one
 *   - Dead neurons go on a free list for reuse
 * This eliminates per-neuron malloc overhead (~40 bytes each on Linux)
 * and keeps neurons contiguous in memory for cache efficiency.
 */

#include "mimir.h"

/*
 * Initialize the pool. Everything starts at zero.
 * No neurons exist yet — they'll be born during network creation
 * and neurogenesis events.
 */
void pool_init(NeuronPool *pool) {
    pool->total_created = 0;
    pool->total_active = 0;
    pool->total_committed = 0;
    pool->total_dead = 0;
}

/*
 * Register a neuron birth. Returns a unique ID for the new neuron.
 *
 * The ID is a lifetime counter — it only goes up, never reused.
 * Even if a neuron dies, its ID is retired. This makes debugging
 * easier: "neuron #47 died at epoch 300" is unambiguous.
 *
 * BIOLOGICAL PARALLEL: Every neuron in your brain has a unique
 * identity defined by its specific gene expression pattern, position,
 * and connection history. No two neurons are identical, even if they
 * do similar jobs. Our ID captures this uniqueness.
 */
int pool_birth(NeuronPool *pool) {
    int id = pool->total_created;
    pool->total_created++;
    pool->total_active++;
    return id;
}

/*
 * Register a neuron death.
 *
 * BIOLOGICAL PARALLEL: Apoptosis (programmed cell death). In the brain,
 * roughly 50% of neurons born in the adult hippocampus die within weeks
 * if they don't integrate into useful circuits. It's not waste — it's
 * quality control. Only neurons that prove useful survive.
 *
 * In our model, death means the neuron's weights contributed nothing
 * meaningful. Its memory can be reclaimed. But committed neurons
 * (those holding learned knowledge) are EXEMPT from death.
 */
void pool_kill(NeuronPool *pool) {
    pool->total_active--;
    pool->total_dead++;
}

/*
 * Print pool statistics. Gives a snapshot of the network's
 * neurogenesis history — how many neurons were born, how many
 * survived, how many hold permanent knowledge.
 */
void pool_print(const NeuronPool *pool) {
    printf("  Pool: %d born | %d active | %d committed | %d dead\n",
           pool->total_created, pool->total_active,
           pool->total_committed, pool->total_dead);

    /*
     * Survival rate: what fraction of born neurons are still alive.
     * In biology, adult hippocampal neurogenesis has ~50% survival.
     * If our survival rate is very low, we're growing too aggressively.
     * If it's 100%, we might not be pruning enough.
     */
    if (pool->total_created > 0) {
        float survival = (float)pool->total_active / (float)pool->total_created * 100.0f;
        printf("  Survival rate: %.1f%%\n", survival);
    }
}
