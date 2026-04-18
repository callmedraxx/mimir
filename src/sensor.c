/*
 * MIMIR — Sensor Registry
 *
 * Runtime-pluggable sensory input system.
 *
 * BIOLOGICAL PARALLEL — THE THALAMUS:
 * In the human brain, the thalamus is the sensory relay station that sits
 * between the specialised sensory cortices (V1, A1, S1, …) and the shared
 * association cortex where "thinking" happens. Almost every sensory signal
 * passes through the thalamus before reaching the cortex — it does not
 * process the signal, it routes it and gates it (attention modulates which
 * signals get amplified and which are suppressed).
 *
 * This file is Mimir's thalamus. It:
 *   1. Maintains the registry of all connected sensors.
 *   2. Validates that each sensor speaks the right protocol.
 *   3. Routes encode() calls to the correct sensor.
 *   4. Gates sensors via the active flag (attention / sensor failure).
 *
 * What this file does NOT do:
 *   - Process or interpret sensor output (that is the shared core's job).
 *   - Allocate raw input buffers (sensors own their own data).
 *   - Decide how to combine multiple sensor embeddings (TBD — attention
 *     pooling will live in the core, not here).
 *
 * Adding a new sensor:
 *   1. Write an encode() function that takes raw floats → 128-float embedding.
 *   2. Call sensor_register() with that function and any config in user_data.
 *   3. Done. The core picks it up automatically on the next forward pass.
 */

#include "mimir.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// ── Registry lifecycle ────────────────────────────────────────────────────────

void sensor_registry_init(SensorRegistry *reg) {
    /*
     * Zero the entire struct. This sets count=0, clears all name strings,
     * NULLs all pointers, and sets active=false for every slot.
     * Sensors become active only when explicitly registered.
     */
    memset(reg, 0, sizeof(*reg));
}

// ── Registration ──────────────────────────────────────────────────────────────

int sensor_register(SensorRegistry *reg,
                    const char     *name,
                    int             raw_size,
                    Network        *encoder,
                    SensorEncodeFn  encode_fn,
                    void           *user_data) {

    /* Hard limit — SENSOR_MAX is the one constant to change if this fires. */
    if (reg->count >= SENSOR_MAX) {
        fprintf(stderr, "[Sensor] Registry full (max %d). "
                        "Increase SENSOR_MAX in mimir.h to add more.\n",
                SENSOR_MAX);
        return -1;
    }

    /*
     * Reject duplicate names. Each sensor must be uniquely addressable
     * by name so sensor_find() is deterministic. Without this check, two
     * sensors named "audio" would make sensor_find() always return the
     * first one, silently ignoring the second.
     */
    if (sensor_find(reg, name) != NULL) {
        fprintf(stderr, "[Sensor] A sensor named '%s' is already registered. "
                        "Deregister it first or use a unique name.\n", name);
        return -1;
    }

    /* encode_fn is the only mandatory field — without it the sensor is useless. */
    if (!encode_fn) {
        fprintf(stderr, "[Sensor] Cannot register '%s' with a NULL encode function.\n",
                name);
        return -1;
    }

    /* raw_size must be positive — 0 or negative is a caller bug. */
    if (raw_size <= 0) {
        fprintf(stderr, "[Sensor] '%s' raw_size must be > 0, got %d.\n",
                name, raw_size);
        return -1;
    }

    /* Claim the next free slot. */
    int idx = reg->count++;
    Sensor *s = &reg->sensors[idx];

    /*
     * Copy name with explicit NUL termination. strncpy does NOT guarantee
     * a terminating NUL if the source is longer than the buffer, so we
     * force it. A silent truncation is better than a buffer overrun.
     */
    strncpy(s->name, name, sizeof(s->name) - 1);
    s->name[sizeof(s->name) - 1] = '\0';

    s->raw_size   = raw_size;
    s->encoder    = encoder;    /* NULL is valid — manual encoders don't need a net */
    s->encode     = encode_fn;
    s->user_data  = user_data;
    s->active     = true;       /* Sensors are live as soon as they register.
                                 * Set s->active = false to temporarily mute one. */

    printf("[Sensor] Registered '%s'  raw_size=%-4d  "
           "→  embedding=%d  encoder=%s\n",
           s->name,
           s->raw_size,
           MIMIR_EMBEDDING_SIZE,
           encoder ? "learned (Network)" : "manual");

    return idx;
}

// ── Deregistration ────────────────────────────────────────────────────────────

int sensor_deregister(SensorRegistry *reg, const char *name) {
    for (int i = 0; i < reg->count; i++) {
        if (strcmp(reg->sensors[i].name, name) != 0)
            continue;

        /*
         * Free the encoder network if the registry owns it.
         * The registry takes ownership on register; the caller must not
         * free an encoder network after passing it to sensor_register().
         */
        if (reg->sensors[i].encoder) {
            network_free(reg->sensors[i].encoder);
            free(reg->sensors[i].encoder);
            reg->sensors[i].encoder = NULL;
        }

        /*
         * Compact the array by shifting everything above index i down.
         *
         * WHY compact instead of leaving a gap?
         * The core iterates reg->sensors[0..count-1] on every single
         * forward pass. A gap means either a branch check per slot
         * ("is this slot valid?") or a separate active-sensor index
         * array. Compaction keeps the hot path branch-free and
         * cache-sequential. The cost (a memmove) only happens on
         * deregister, which is rare.
         */
        int tail = reg->count - 1;
        for (int j = i; j < tail; j++) {
            reg->sensors[j] = reg->sensors[j + 1];
        }
        memset(&reg->sensors[tail], 0, sizeof(Sensor));
        reg->count--;

        printf("[Sensor] Deregistered '%s'\n", name);
        return 0;
    }

    fprintf(stderr, "[Sensor] Cannot deregister '%s' — not found.\n", name);
    return -1;
}

// ── Lookup ────────────────────────────────────────────────────────────────────

Sensor *sensor_find(SensorRegistry *reg, const char *name) {
    /*
     * Linear scan. With at most SENSOR_MAX (16) entries this is faster
     * than a hash table: no hashing overhead, no collision handling, and
     * the whole array fits in a few cache lines.
     */
    for (int i = 0; i < reg->count; i++) {
        if (strcmp(reg->sensors[i].name, name) == 0)
            return &reg->sensors[i];
    }
    return NULL;
}

// ── Encoding ──────────────────────────────────────────────────────────────────

int sensor_encode(Sensor      *sensor,
                  const float *raw,   int raw_size,
                  float       *embedding) {

    if (!sensor) {
        fprintf(stderr, "[Sensor] sensor_encode called with NULL sensor.\n");
        return -1;
    }

    if (!sensor->active) {
        /*
         * An inactive sensor produces a zero embedding — a neutral signal.
         * This mirrors what the brain does when a sense is absent: the
         * corresponding cortical area goes quiet, not noisy. The core
         * sees silence, not garbage.
         */
        memset(embedding, 0, MIMIR_EMBEDDING_SIZE * sizeof(float));
        return 0;
    }

    /*
     * Validate raw_size BEFORE calling encode(). Sensors are written by
     * humans; a mismatch here is almost always a wiring bug (wrong sensor
     * connected to the wrong data source). Failing loudly here is far
     * better than silently producing a malformed embedding that trains
     * the network on garbage for thousands of epochs before anyone notices.
     */
    if (raw_size != sensor->raw_size) {
        fprintf(stderr, "[Sensor] '%s' expects raw_size=%d, got %d. "
                        "Check that the data source matches the sensor spec.\n",
                sensor->name, sensor->raw_size, raw_size);
        return -1;
    }

    if (!sensor->encode) {
        fprintf(stderr, "[Sensor] '%s' has no encode function — "
                        "was it registered correctly?\n", sensor->name);
        return -1;
    }

    /*
     * Delegate to the sensor's own encode() implementation.
     * The sensor knows its modality; we just forward the call.
     * The contract: write exactly MIMIR_EMBEDDING_SIZE floats
     * into embedding[], return 0 on success or -1 on error.
     */
    return sensor->encode(sensor, raw, raw_size, embedding, MIMIR_EMBEDDING_SIZE);
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

void sensor_registry_print(const SensorRegistry *reg) {
    printf("  Sensor registry: %d / %d slots used\n", reg->count, SENSOR_MAX);
    if (reg->count == 0) {
        printf("    (no sensors registered)\n");
        return;
    }
    printf("  %-16s  %-8s  %-20s  %-8s  %s\n",
           "Name", "Raw in", "Encoder", "Emb out", "Status");
    printf("  %-16s  %-8s  %-20s  %-8s  %s\n",
           "────────────────", "────────", "────────────────────", "────────", "──────");
    for (int i = 0; i < reg->count; i++) {
        const Sensor *s = &reg->sensors[i];
        printf("  %-16s  %-8d  %-20s  %-8d  %s\n",
               s->name,
               s->raw_size,
               s->encoder ? "learned (Network)" : "manual",
               MIMIR_EMBEDDING_SIZE,
               s->active ? "ACTIVE" : "inactive");
    }
}
