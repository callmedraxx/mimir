/*
 * MIMIR — Training Log
 *
 * A persistent, append-only measurement journal. Every real learning
 * event the brain goes through is written here with a timestamp and
 * structured key=value fields so you can track improvement over time.
 *
 * FILE FORMAT
 * ───────────
 * Tab-separated. One event per line. First two fields are always
 * TIMESTAMP and EVENT_TYPE; the rest are key=value pairs.
 *
 *   2026-04-10T14:23:01  STARTUP    gate_neurons=23  gate_committed=11  abc_neurons=64  abc_committed=0
 *   2026-04-10T14:23:15  LEARN      letter=A  word=apple  time_ms=847.3  neurons=34  committed=8  conf_before=0.12  conf_after=0.91
 *   2026-04-10T14:23:45  REPLAY     cycles=50  known=3  accuracy=0.800  corrections=6  committed=8
 *   2026-04-10T14:24:00  BENCHMARK  gate=AND  method=brain_native  epochs=3200  time_ms=11.3  memory_bytes=0  neurons=7  correct=4  total=4  solved=yes
 *   2026-04-10T14:30:00  SHUTDOWN   total_cycles=1400  accuracy=0.847  total_corrections=216
 *
 * SHELL USAGE
 * ───────────
 *   # Show all teach events, newest first:
 *   grep LEARN data/mimir_training.log | tac
 *
 *   # Plot accuracy over replay cycles (requires awk + gnuplot):
 *   grep REPLAY data/mimir_training.log | awk -F'accuracy=' '{print $2}' | cut -f1
 *
 *   # Compare benchmark times across sessions:
 *   grep BENCHMARK data/mimir_training.log
 *
 * THREAD SAFETY
 * ─────────────
 * The replay thread writes REPLAY events; the main thread writes
 * everything else. A mutex protects the FILE* so concurrent writes
 * don't interleave. Each tlog_* call is one atomic fprintf sequence.
 */

#include "mimir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Global log file handle and mutex.
 * Initialised by tlog_open(), destroyed by tlog_close(). */
static FILE           *g_log  = NULL;
static pthread_mutex_t g_lock;
static bool            g_open = false;

/* ── Timestamp helper ─────────────────────────────────────────────────── */

/*
 * Write the current local time as ISO 8601 into buf.
 * Format: "2026-04-10T14:23:01"  (always 19 chars + NUL)
 */
static void timestamp(char *buf, size_t size) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    if (tm_info)
        strftime(buf, size, "%Y-%m-%dT%H:%M:%S", tm_info);
    else
        snprintf(buf, size, "0000-00-00T00:00:00");
}

/* ── Lifecycle ────────────────────────────────────────────────────────── */

void tlog_open(void) {
    if (g_open) return;

    /* Create the data/ directory if it does not exist yet.
     * checkpoint_mkdir only makes one level, which is all we need. */
    checkpoint_mkdir(TLOG_PATH);

    g_log = fopen(TLOG_PATH, "a");   /* append — never truncates history */
    if (!g_log) {
        fprintf(stderr, "[TrainLog] Cannot open '%s' for writing — "
                        "measurements will not be saved.\n", TLOG_PATH);
        return;
    }

    pthread_mutex_init(&g_lock, NULL);
    g_open = true;

    /* Write a session separator so runs are visually distinct in the file */
    char ts[32];
    timestamp(ts, sizeof(ts));
    fprintf(g_log, "# ── session start %s ──\n", ts);
    fflush(g_log);
}

void tlog_close(void) {
    if (!g_open) return;

    char ts[32];
    timestamp(ts, sizeof(ts));

    pthread_mutex_lock(&g_lock);
    fprintf(g_log, "# ── session end   %s ──\n\n", ts);
    fflush(g_log);
    fclose(g_log);
    g_log = NULL;
    pthread_mutex_unlock(&g_lock);

    pthread_mutex_destroy(&g_lock);
    g_open = false;
}

/* ── Event writers ────────────────────────────────────────────────────── */

void tlog_startup(int gate_neurons, int gate_committed,
                  int abc_neurons,  int abc_committed) {
    if (!g_open) return;
    char ts[32]; timestamp(ts, sizeof(ts));

    pthread_mutex_lock(&g_lock);
    fprintf(g_log,
            "%s\tSTARTUP\t"
            "gate_neurons=%d\tgate_committed=%d\t"
            "abc_neurons=%d\tabc_committed=%d\n",
            ts,
            gate_neurons, gate_committed,
            abc_neurons,  abc_committed);
    fflush(g_log);
    pthread_mutex_unlock(&g_lock);

    /* Stdout summary */
    printf("  [Log] Startup recorded — gate:%d/%d committed  abc:%d/%d committed\n",
           gate_committed, gate_neurons, abc_committed, abc_neurons);
}

void tlog_pretrain(const char *task, int samples, int epochs_run,
                   double time_ms, bool converged) {
    if (!g_open) return;
    char ts[32]; timestamp(ts, sizeof(ts));

    pthread_mutex_lock(&g_lock);
    fprintf(g_log,
            "%s\tPRETRAIN\t"
            "task=%s\tsamples=%d\tepochs=%d\ttime_ms=%.1f\tconverged=%s\n",
            ts,
            task, samples, epochs_run, time_ms,
            converged ? "yes" : "no");
    fflush(g_log);
    pthread_mutex_unlock(&g_lock);

    printf("  [Log] Pre-train: %s  %d samples  %d epochs  %.0f ms  %s\n",
           task, samples, epochs_run, time_ms,
           converged ? "converged" : "hit limit");
}

void tlog_learn(char letter, const char *word,
                double time_ms, int neurons, int committed,
                float conf_before, float conf_after) {
    if (!g_open) return;
    char ts[32]; timestamp(ts, sizeof(ts));

    pthread_mutex_lock(&g_lock);
    fprintf(g_log,
            "%s\tLEARN\t"
            "letter=%c\tword=%s\ttime_ms=%.1f\t"
            "neurons=%d\tcommitted=%d\t"
            "conf_before=%.3f\tconf_after=%.3f\n",
            ts,
            letter, word, time_ms,
            neurons, committed,
            conf_before, conf_after);
    fflush(g_log);
    pthread_mutex_unlock(&g_lock);

    /*
     * Stdout summary — shows the confidence delta so the user can see
     * how much the brain improved from this single teaching event.
     */
    printf("  [Log] %c -> %s  %.0f ms  %d/%d neurons committed  "
           "confidence: %.0f%% -> %.0f%%\n",
           letter, word, time_ms, committed, neurons,
           conf_before * 100.0f, conf_after * 100.0f);
}

void tlog_replay(unsigned long cycles, int known,
                 double accuracy, unsigned long corrections,
                 int committed) {
    if (!g_open) return;
    char ts[32]; timestamp(ts, sizeof(ts));

    pthread_mutex_lock(&g_lock);
    fprintf(g_log,
            "%s\tREPLAY\t"
            "cycles=%lu\tknown=%d\taccuracy=%.3f\t"
            "corrections=%lu\tcommitted=%d\n",
            ts,
            cycles, known, accuracy,
            corrections, committed);
    fflush(g_log);
    pthread_mutex_unlock(&g_lock);

    /* Replay events are frequent — only print to stdout in verbose context.
     * The replay thread itself handles verbose stdout printing. */
}

void tlog_benchmark(const char *gate, const char *method,
                    int epochs, double time_ms, int memory_bytes,
                    int neurons, int correct, int total, bool solved) {
    if (!g_open) return;
    char ts[32]; timestamp(ts, sizeof(ts));

    pthread_mutex_lock(&g_lock);
    fprintf(g_log,
            "%s\tBENCHMARK\t"
            "gate=%s\tmethod=%s\t"
            "epochs=%d\ttime_ms=%.2f\tmemory_bytes=%d\t"
            "neurons=%d\tcorrect=%d\ttotal=%d\tsolved=%s\n",
            ts,
            gate, method,
            epochs, time_ms, memory_bytes,
            neurons, correct, total,
            solved ? "yes" : "no");
    fflush(g_log);
    pthread_mutex_unlock(&g_lock);
}

void tlog_shutdown(unsigned long total_cycles, double accuracy,
                   unsigned long total_corrections) {
    if (!g_open) return;
    char ts[32]; timestamp(ts, sizeof(ts));

    pthread_mutex_lock(&g_lock);
    fprintf(g_log,
            "%s\tSHUTDOWN\t"
            "total_cycles=%lu\taccuracy=%.3f\ttotal_corrections=%lu\n",
            ts, total_cycles, accuracy, total_corrections);
    fflush(g_log);
    pthread_mutex_unlock(&g_lock);

    printf("  [Log] Session saved to %s\n", TLOG_PATH);
}
