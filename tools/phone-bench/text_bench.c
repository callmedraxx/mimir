/* Mimir taking in a whole book on the phone: tokens/s, one thread on the
 * prime core, reader checking and fixing every new word as it goes. Run 3
 * times: the 1st learns its new words, later runs show steady speed. */
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
    for (int run = 1; run <= 3; run++) {
        TextContext t; text_init(&t);
        double t0 = now();
        int n = text_take_file(&t, &r, argv[1]);
        double dt = now() - t0;
        printf("run %d, core %d: %d tokens in %.2f s = %.0f tokens/s (%u sentences, %u different, fixed %ld)\n",
               run, cpu, n, dt, n / dt, t.sentences, t.nvoc, t.fixed);
        text_free(&t);
    }
    return 0;
}
