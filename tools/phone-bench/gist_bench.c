/* Mimir on the phone: take in a whole book, then get its gist (workspace of
 * 4 + cued recall over every event). Prints the time for each. */
#define _GNU_SOURCE
#include "mimir.h"
#include <sched.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc, char **argv) {
    int cpu = argc > 2 ? atoi(argv[2]) : 7;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); sched_setaffinity(0, sizeof s, &s);
    Reader r; reader_init(&r);
    if (reader_load(&r, "reader.model")) { printf("no reader.model\n"); return 1; }
    reader_lex_load(&r, "cmudict.dict"); r.self_learn = 1;
    Meaning m; meaning_init(&m);
    if (meaning_load(&m, "meaning.mng") <= 0) { printf("no meaning.mng\n"); return 1; }
    TextContext t; text_init(&t);
    double t0 = now();
    int n = text_take_file(&t, &r, argv[1]);
    double t1 = now();
    gist_show(&t, &m, GIST_BUDGET);
    double t2 = now();
    printf("core %d: took in %d tokens in %.2f s (%.0f tokens/s); gist in %.2f s (%.0f tokens/s); together %.0f tokens/s\n",
           cpu, n, t1 - t0, n / (t1 - t0), t2 - t1, n / (t2 - t1), n / (t2 - t0));
    return 0;
}
