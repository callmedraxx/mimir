/*
 * MIMIR - Network checkpoint: save and load trained state to/from disk.
 *
 * WHY: Training can take thousands of epochs. After code changes that only
 * affect non-training logic (CLI, output formatting, math ops), there is
 * no reason to retrain. Saving the committed network to a file lets the
 * binary pick up where it left off instead of starting cold every run.
 *
 * FORMAT (binary, little-endian on the host platform):
 *   [8 bytes]  magic "MIMIR002"
 *   [4 bytes]  n_layers
 *   [4 bytes]  n_inputs
 *   [4 bytes]  n_outputs
 *   [4 bytes]  default_act  (Activation enum, stored as int)
 *   [4 bytes]  growth_gen
 *   [4 bytes]  maturation_rate (float)
 *   [4 bytes]  rpe_baseline    (float)
 *   [16 bytes] NeuronPool      (4 × int)
 *   [4 bytes]  n_conflict_records
 *   [N bytes]  conflict_log[CONFLICT_LOG_SIZE]
 *   For each layer (0..n_layers-1):
 *     [4 bytes] count
 *     [4 bytes] input_size
 *     For each neuron (0..count-1):
 *       [4 bytes] bias
 *       [4 bytes] n_weights
 *       [4 bytes] maturity
 *       [4 bytes] activity
 *       [4 bytes] age
 *       [4 bytes] id
 *       [4 bytes] state      (NeuronState enum)
 *       [4 bytes] act        (Activation enum)
 *       [4 bytes] last_z
 *       [4 bytes] last_output
 *       [4 bytes] theta
 *       [4 bytes] mean_out
 *       [1 byte]  is_visual   (visual-modality neuron flag)
 *       [n_weights × 4 bytes] weights
 */

#include "mimir.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#define MAGIC     "MIMIR002"
#define MAGIC_LEN 8

static int write_bytes(FILE *f, const void *buf, size_t n) {
    return fwrite(buf, 1, n, f) == n ? 0 : -1;
}
static int read_bytes(FILE *f, void *buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}

/*
 * Serialize `net` to `path`.
 * Returns 0 on success, -1 on any I/O error.
 */
int network_save(const Network *net, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

#define W(ptr, sz) do { if (write_bytes(f, (ptr), (sz)) < 0) goto fail; } while(0)

    W(MAGIC, MAGIC_LEN);

    W(&net->n_layers,       sizeof(int));
    W(&net->n_inputs,       sizeof(int));
    W(&net->n_outputs,      sizeof(int));
    W(&net->default_act,    sizeof(Activation));
    W(&net->growth_gen,     sizeof(int));
    W(&net->maturation_rate,sizeof(float));
    W(&net->rpe_baseline,   sizeof(float));
    W(&net->pool,           sizeof(NeuronPool));
    W(&net->n_conflict_records, sizeof(int));
    W(net->conflict_log,    sizeof(ConflictRecord) * CONFLICT_LOG_SIZE);

    for (int l = 0; l < net->n_layers; l++) {
        const Layer *layer = &net->layers[l];
        W(&layer->count,      sizeof(int));
        W(&layer->input_size, sizeof(int));

        for (int j = 0; j < layer->count; j++) {
            const Neuron *n = &layer->neurons[j];
            W(&n->bias,        sizeof(float));
            W(&n->n_weights,   sizeof(int));
            W(&n->maturity,    sizeof(float));
            W(&n->activity,    sizeof(float));
            W(&n->age,         sizeof(int));
            W(&n->id,          sizeof(int));
            W(&n->state,       sizeof(NeuronState));
            W(&n->act,         sizeof(Activation));
            W(&n->last_z,      sizeof(float));
            W(&n->last_output, sizeof(float));
            W(&n->theta,       sizeof(float));
            W(&n->mean_out,    sizeof(float));
            W(&n->is_visual,   sizeof(uint8_t));
            W(n->weights,      sizeof(float) * (size_t)n->n_weights);
        }
    }

#undef W
    fclose(f);
    return 0;
fail:
    fclose(f);
    return -1;
}

/*
 * Deserialize a Network from `path`.
 *
 * Returns a heap-allocated Network * on success (caller owns it; call
 * network_free() then free() when done), or NULL on any error (file not
 * found, wrong magic, truncated file, allocation failure).
 *
 * Usage in main():
 *   Network *tmp = network_load(CHECKPOINT_PATH);
 *   if (tmp) { brain = *tmp; free(tmp); }
 */
Network *network_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    char magic[MAGIC_LEN];
    if (read_bytes(f, magic, MAGIC_LEN) < 0 ||
        memcmp(magic, MAGIC, MAGIC_LEN) != 0) {
        fclose(f);
        return NULL;
    }

    Network *net = calloc(1, sizeof(Network));
    if (!net) { fclose(f); return NULL; }

#define R(ptr, sz) do { if (read_bytes(f, (ptr), (sz)) < 0) goto fail; } while(0)

    R(&net->n_layers,        sizeof(int));
    R(&net->n_inputs,        sizeof(int));
    R(&net->n_outputs,       sizeof(int));
    R(&net->default_act,     sizeof(Activation));
    R(&net->growth_gen,      sizeof(int));
    R(&net->maturation_rate, sizeof(float));
    R(&net->rpe_baseline,    sizeof(float));
    R(&net->pool,            sizeof(NeuronPool));
    R(&net->n_conflict_records, sizeof(int));
    R(net->conflict_log,     sizeof(ConflictRecord) * CONFLICT_LOG_SIZE);

    net->n_layers_cap = net->n_layers;
    net->layers = calloc((size_t)net->n_layers, sizeof(Layer));
    if (!net->layers) goto fail;

    for (int l = 0; l < net->n_layers; l++) {
        Layer *layer = &net->layers[l];
        R(&layer->count,      sizeof(int));
        R(&layer->input_size, sizeof(int));
        layer->capacity = layer->count;

        layer->neurons = calloc((size_t)layer->count, sizeof(Neuron));
        layer->outputs = calloc((size_t)layer->count, sizeof(float));
        if (!layer->neurons || !layer->outputs) goto fail;

        for (int j = 0; j < layer->count; j++) {
            Neuron *n = &layer->neurons[j];
            R(&n->bias,        sizeof(float));
            R(&n->n_weights,   sizeof(int));
            R(&n->maturity,    sizeof(float));
            R(&n->activity,    sizeof(float));
            R(&n->age,         sizeof(int));
            R(&n->id,          sizeof(int));
            R(&n->state,       sizeof(NeuronState));
            R(&n->act,         sizeof(Activation));
            R(&n->last_z,      sizeof(float));
            R(&n->last_output, sizeof(float));
            R(&n->theta,       sizeof(float));
            R(&n->mean_out,    sizeof(float));
            R(&n->is_visual,   sizeof(uint8_t));
            n->weights = malloc(sizeof(float) * (size_t)n->n_weights);
            if (!n->weights) goto fail;
            R(n->weights, sizeof(float) * (size_t)n->n_weights);
            layer->outputs[j] = n->last_output;
        }
    }

#undef R
    fclose(f);
    return net;

fail:
    /* Free any layers/neurons that were successfully allocated */
    if (net->layers) {
        for (int l = 0; l < net->n_layers; l++) {
            Layer *layer = &net->layers[l];
            if (layer->neurons) {
                for (int j = 0; j < layer->count; j++)
                    free(layer->neurons[j].weights);
                free(layer->neurons);
            }
            free(layer->outputs);
        }
        free(net->layers);
    }
    free(net);
    fclose(f);
    return NULL;
}

/*
 * Ensure the directory that contains `path` exists.
 * Creates only the immediate parent (e.g. "data" for "data/mimir.brain").
 * Silently succeeds if it already exists.
 */
void checkpoint_mkdir(const char *path) {
    char dir[256];
    const char *slash = strrchr(path, '/');
    if (!slash) return;  /* no directory component */
    size_t len = (size_t)(slash - path);
    if (len == 0 || len >= sizeof(dir)) return;
    memcpy(dir, path, len);
    dir[len] = '\0';
    mkdir(dir, 0755);   /* ignore EEXIST */
}
