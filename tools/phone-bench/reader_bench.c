/* Mimir reader speed on the phone: reads the 25,000 words of aligned.tsv
 * over and over, pinned to chosen cores, and prints words/s and letters/s.
 * Last run is sustained (20 s, all cores) and reports the slowest second:
 * the worst-case, throttled number. */
#define _GNU_SOURCE
#include "mimir.h"
#include <pthread.h>
#include <sched.h>

static Reader R;
static char (*W)[READER_MAX_WORD];
static int NW;
static volatile int go, stop;

typedef struct { int cpu; long words, letters; } Job;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void *run(void *p) {
    Job *j = p;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(j->cpu, &s);
    sched_setaffinity(0, sizeof s, &s);
    int out[READER_MAX_WORD], i = j->cpu * 997 % NW;
    while (!go) ;
    while (!stop) {
        j->letters += reader_read_word(&R, W[i], out);
        j->words++;
        if (++i == NW) i = 0;
    }
    return NULL;
}

static void bench(const char *name, const int *cpus, int n, double secs, int per_second) {
    pthread_t t[8]; Job j[8] = {0};
    go = stop = 0;
    for (int k = 0; k < n; k++) { j[k].cpu = cpus[k]; pthread_create(&t[k], NULL, run, &j[k]); }
    double t0 = now(), last = t0, worst = 1e18; long lastw = 0;
    go = 1;
    while (now() - t0 < secs) {
        struct timespec d = {0, 100000000}; nanosleep(&d, NULL);
        if (per_second && now() - last >= 1.0) {
            long w = 0; for (int k = 0; k < n; k++) w += j[k].words;
            double r = (w - lastw) / (now() - last);
            if (r < worst) worst = r;
            lastw = w; last = now();
        }
    }
    stop = 1;
    double dt = now() - t0;
    long w = 0, l = 0;
    for (int k = 0; k < n; k++) { pthread_join(t[k], NULL); w += j[k].words; l += j[k].letters; }
    printf("%-28s %10.0f words/s %11.0f letters/s", name, w / dt, l / dt);
    if (per_second) printf("   slowest second: %.0f words/s", worst);
    printf("\n");
}

int main(void) {
    reader_init(&R);
    if (reader_load(&R, "reader.model")) { printf("no reader.model\n"); return 1; }
    FILE *f = fopen("aligned.tsv", "r");
    if (!f) { printf("no aligned.tsv\n"); return 1; }
    W = malloc(30000 * sizeof *W);
    char line[512];
    while (NW < 30000 && fgets(line, sizeof line, f)) {
        char *w = strtok(line, "\t");
        if (w && strlen(w) < READER_MAX_WORD) strcpy(W[NW++], w);
    }
    printf("%d words\n", NW);
    int x2[] = {7}, a710[] = {4}, a510[] = {0}, big4[] = {4, 5, 6, 7}, all[] = {0, 1, 2, 3, 4, 5, 6, 7};
    bench("1 thread, X2 (prime)", x2, 1, 3, 0);
    bench("1 thread, A710", a710, 1, 3, 0);
    bench("1 thread, A510 (little)", a510, 1, 3, 0);
    bench("4 threads, X2 + 3x A710", big4, 4, 3, 0);
    bench("8 threads, all cores", all, 8, 3, 0);
    bench("8 threads, sustained 20 s", all, 8, 20, 1);

    /* Reading a book while checking every word and fixing mistakes:
     * one thread on the prime core, the way `reader book` runs. */
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(7, &s);
    sched_setaffinity(0, sizeof s, &s);
    printf("dictionary: %d words\n", reader_lex_load(&R, "cmudict.dict"));
    for (int pass = 1; pass <= 2; pass++) {
        double t0 = now();
        long seen0 = R.seen;
        reader_book(&R, "alice.txt");
        double dt = now() - t0;
        printf("book pass %d, 1 thread X2, self-checking: %.0f words/s (%.2f s)\n", pass, (R.seen - seen0) / dt, dt);
    }
    reader_stats(&R);
    return 0;
}
