/*
 * mimir-bench — measure what the phone's CPU can really do for a
 * Mimir-Reader style encoder.
 *
 *   1. Peak int8 throughput per core type (SDOT and SMMLA microkernels,
 *      register-resident, no memory traffic).
 *   2. Peak fp32 FMA throughput per core type, for reference.
 *   3. Real packed int8 GEMMs with the Mimir-Reader layer shapes
 *      (d=256, f=768), single core and multi-core.
 *   4. Projected Mimir-Reader tokens/sec from the measured GEMM speed.
 *
 * Build: aarch64-linux-android34-clang -O3 -march=armv8.6-a+i8mm+dotprod
 */
#define _GNU_SOURCE
#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

/* ── Peak microkernels ─────────────────────────────────────────────── */

/* 12 independent accumulators so the FMA pipes never wait on latency. */
static double peak_sdot(long iters) {
    int8x16_t a = vdupq_n_s8(1), b = vdupq_n_s8(2);
    int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0,
              c6 = c0, c7 = c0, c8 = c0, c9 = c0, c10 = c0, c11 = c0;
    double t = now();
    for (long i = 0; i < iters; i++) {
        c0 = vdotq_s32(c0, a, b);  c1 = vdotq_s32(c1, a, b);
        c2 = vdotq_s32(c2, a, b);  c3 = vdotq_s32(c3, a, b);
        c4 = vdotq_s32(c4, a, b);  c5 = vdotq_s32(c5, a, b);
        c6 = vdotq_s32(c6, a, b);  c7 = vdotq_s32(c7, a, b);
        c8 = vdotq_s32(c8, a, b);  c9 = vdotq_s32(c9, a, b);
        c10 = vdotq_s32(c10, a, b); c11 = vdotq_s32(c11, a, b);
        __asm__ volatile("" : "+w"(a), "+w"(b));
    }
    t = now() - t;
    int32x4_t s = vaddq_s32(vaddq_s32(vaddq_s32(c0, c1), vaddq_s32(c2, c3)),
                            vaddq_s32(vaddq_s32(c4, c5), vaddq_s32(c6, c7)));
    s = vaddq_s32(s, vaddq_s32(vaddq_s32(c8, c9), vaddq_s32(c10, c11)));
    if (vgetq_lane_s32(s, 0) == 42) puts("");
    /* each SDOT = 16 MACs = 32 ops */
    return iters * 12.0 * 32.0 / t / 1e9;
}

static double peak_smmla(long iters) {
    int8x16_t a = vdupq_n_s8(1), b = vdupq_n_s8(2);
    int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0,
              c6 = c0, c7 = c0, c8 = c0, c9 = c0, c10 = c0, c11 = c0;
    double t = now();
    for (long i = 0; i < iters; i++) {
        c0 = vmmlaq_s32(c0, a, b);  c1 = vmmlaq_s32(c1, a, b);
        c2 = vmmlaq_s32(c2, a, b);  c3 = vmmlaq_s32(c3, a, b);
        c4 = vmmlaq_s32(c4, a, b);  c5 = vmmlaq_s32(c5, a, b);
        c6 = vmmlaq_s32(c6, a, b);  c7 = vmmlaq_s32(c7, a, b);
        c8 = vmmlaq_s32(c8, a, b);  c9 = vmmlaq_s32(c9, a, b);
        c10 = vmmlaq_s32(c10, a, b); c11 = vmmlaq_s32(c11, a, b);
        __asm__ volatile("" : "+w"(a), "+w"(b));
    }
    t = now() - t;
    int32x4_t s = vaddq_s32(vaddq_s32(vaddq_s32(c0, c1), vaddq_s32(c2, c3)),
                            vaddq_s32(vaddq_s32(c4, c5), vaddq_s32(c6, c7)));
    s = vaddq_s32(s, vaddq_s32(vaddq_s32(c8, c9), vaddq_s32(c10, c11)));
    if (vgetq_lane_s32(s, 0) == 42) puts("");
    /* each SMMLA = (2x8)·(8x2) = 32 MACs = 64 ops */
    return iters * 12.0 * 64.0 / t / 1e9;
}

static double peak_fmla(long iters) {
    float32x4_t a = vdupq_n_f32(1.0001f), b = vdupq_n_f32(0.9999f);
    float32x4_t c0 = vdupq_n_f32(0), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0,
                c6 = c0, c7 = c0, c8 = c0, c9 = c0, c10 = c0, c11 = c0;
    double t = now();
    for (long i = 0; i < iters; i++) {
        c0 = vfmaq_f32(c0, a, b);  c1 = vfmaq_f32(c1, a, b);
        c2 = vfmaq_f32(c2, a, b);  c3 = vfmaq_f32(c3, a, b);
        c4 = vfmaq_f32(c4, a, b);  c5 = vfmaq_f32(c5, a, b);
        c6 = vfmaq_f32(c6, a, b);  c7 = vfmaq_f32(c7, a, b);
        c8 = vfmaq_f32(c8, a, b);  c9 = vfmaq_f32(c9, a, b);
        c10 = vfmaq_f32(c10, a, b); c11 = vfmaq_f32(c11, a, b);
        __asm__ volatile("" : "+w"(a), "+w"(b));
    }
    t = now() - t;
    float32x4_t s = vaddq_f32(vaddq_f32(vaddq_f32(c0, c1), vaddq_f32(c2, c3)),
                              vaddq_f32(vaddq_f32(c4, c5), vaddq_f32(c6, c7)));
    s = vaddq_f32(s, vaddq_f32(vaddq_f32(c8, c9), vaddq_f32(c10, c11)));
    if (vgetq_lane_f32(s, 0) == 42.0f) puts("");
    /* each FMLA = 4 MACs = 8 FLOPs */
    return iters * 12.0 * 8.0 / t / 1e9;
}

/* ── Packed int8 GEMM with SMMLA ───────────────────────────────────────
 * C[M][N] = A[M][K] · B[N][K]ᵀ   (B stored row-per-output, like weights)
 * Packed layout: rows in pairs, K in blocks of 8 →
 *   Ap[(i/2)][(k/8)] = 16 bytes = row i k..k+7, row i+1 k..k+7
 * Microkernel: 4 rows × 8 cols = 2 A regs × 4 B regs → 8 accumulators.
 */
typedef struct {
    const int8_t *Ap, *Bp;
    int32_t *C;
    int M, N, K, m0, m1, cpu;
} GemmJob;

static void pack(const int8_t *X, int8_t *Xp, int R, int K) {
    for (int r = 0; r < R; r += 2)
        for (int k = 0; k < K; k += 8) {
            int8_t *dst = Xp + ((size_t)(r / 2) * (K / 8) + k / 8) * 16;
            memcpy(dst, X + (size_t)r * K + k, 8);
            memcpy(dst + 8, X + (size_t)(r + 1) * K + k, 8);
        }
}

static void gemm_rows(const GemmJob *j) {
    const int KB = j->K / 8, N = j->N;
    for (int i = j->m0; i < j->m1; i += 4) {
        const int8_t *a0p = j->Ap + (size_t)(i / 2) * KB * 16;
        const int8_t *a1p = a0p + (size_t)KB * 16;
        for (int n = 0; n < N; n += 8) {
            const int8_t *b0p = j->Bp + (size_t)(n / 2) * KB * 16;
            const int8_t *b1p = b0p + (size_t)KB * 16;
            const int8_t *b2p = b1p + (size_t)KB * 16;
            const int8_t *b3p = b2p + (size_t)KB * 16;
            int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00,
                      c10 = c00, c11 = c00, c12 = c00, c13 = c00;
            for (int kb = 0; kb < KB; kb++) {
                int8x16_t a0 = vld1q_s8(a0p + kb * 16), a1 = vld1q_s8(a1p + kb * 16);
                int8x16_t b0 = vld1q_s8(b0p + kb * 16), b1 = vld1q_s8(b1p + kb * 16);
                int8x16_t b2 = vld1q_s8(b2p + kb * 16), b3 = vld1q_s8(b3p + kb * 16);
                c00 = vmmlaq_s32(c00, a0, b0); c01 = vmmlaq_s32(c01, a0, b1);
                c02 = vmmlaq_s32(c02, a0, b2); c03 = vmmlaq_s32(c03, a0, b3);
                c10 = vmmlaq_s32(c10, a1, b0); c11 = vmmlaq_s32(c11, a1, b1);
                c12 = vmmlaq_s32(c12, a1, b2); c13 = vmmlaq_s32(c13, a1, b3);
            }
            /* acc = [r0c0 r0c1 r1c0 r1c1] → scatter into C */
            int32x4_t acc[2][4] = {{c00, c01, c02, c03}, {c10, c11, c12, c13}};
            for (int ai = 0; ai < 2; ai++)
                for (int bj = 0; bj < 4; bj++) {
                    int r = i + ai * 2, c = n + bj * 2;
                    int32_t *o = j->C + (size_t)r * N + c;
                    o[0] = vgetq_lane_s32(acc[ai][bj], 0);
                    o[1] = vgetq_lane_s32(acc[ai][bj], 1);
                    o[N] = vgetq_lane_s32(acc[ai][bj], 2);
                    o[N + 1] = vgetq_lane_s32(acc[ai][bj], 3);
                }
        }
    }
}

static void *gemm_thread(void *p) {
    GemmJob *j = p;
    pin(j->cpu);
    gemm_rows(j);
    return NULL;
}

/* Run one GEMM over the given cpus, return GOPS (best of reps). */
static double gemm_bench(int M, int N, int K, const int *cpus, int ncpu,
                         const double *weights, int reps, int verify) {
    int8_t *A = malloc((size_t)M * K), *B = malloc((size_t)N * K);
    int8_t *Ap = aligned_alloc(64, (size_t)M * K), *Bp = aligned_alloc(64, (size_t)N * K);
    int32_t *C = aligned_alloc(64, (size_t)M * N * 4);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)((i * 7 + 3) % 15 - 7);
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (int8_t)((i * 5 + 1) % 13 - 6);
    pack(A, Ap, M, K);
    pack(B, Bp, N, K);

    /* split M across cpus in proportion to weights, multiples of 4 rows */
    GemmJob jobs[8];
    pthread_t th[8];
    double wsum = 0;
    for (int c = 0; c < ncpu; c++) wsum += weights[c];
    int m = 0;
    for (int c = 0; c < ncpu; c++) {
        int rows = (c == ncpu - 1) ? M - m : ((int)(M * weights[c] / wsum) / 4) * 4;
        jobs[c] = (GemmJob){Ap, Bp, C, M, N, K, m, m + rows, cpus[c]};
        m += rows;
    }

    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        double t = now();
        if (ncpu == 1) {
            gemm_rows(&jobs[0]);
        } else {
            for (int c = 0; c < ncpu; c++) pthread_create(&th[c], NULL, gemm_thread, &jobs[c]);
            for (int c = 0; c < ncpu; c++) pthread_join(th[c], NULL);
        }
        t = now() - t;
        if (t < best) best = t;
    }

    if (verify) {
        int bad = 0;
        for (int t = 0; t < 64; t++) {
            int i = (t * 37) % M, n = (t * 53) % N;
            int32_t ref = 0;
            for (int k = 0; k < K; k++) ref += A[(size_t)i * K + k] * B[(size_t)n * K + k];
            if (ref != C[(size_t)i * N + n]) bad++;
        }
        printf("    verify: %s\n", bad ? "MISMATCH" : "ok (64 spot checks)");
    }
    free(A); free(B); free(Ap); free(Bp); free(C);
    return 2.0 * M * N * K / best / 1e9;
}

int main(int argc, char **argv) {
    long iters = argc > 1 ? atol(argv[1]) : 20000000;
    const struct { const char *name; int cpu; } cores[] = {
        {"A510  (cpu0)", 0}, {"A710  (cpu4)", 4}, {"X2    (cpu7)", 7}};

    printf("== Peak per core (register-resident, GOPS) ==\n");
    printf("%-14s %10s %10s %12s\n", "core", "int8 SDOT", "int8 SMMLA", "fp32 FMLA");
    double x2_smmla = 0;
    for (int c = 0; c < 3; c++) {
        pin(cores[c].cpu);
        peak_smmla(iters / 10); /* warm up clocks */
        double sd = peak_sdot(iters), mm = peak_smmla(iters), fp = peak_fmla(iters);
        if (cores[c].cpu == 7) x2_smmla = mm;
        printf("%-14s %10.1f %10.1f %12.1f\n", cores[c].name, sd, mm, fp);
    }
    (void)x2_smmla;

    /* Mimir-Reader layer shapes: 512 tokens, d=256, f=768 */
    const int M = 512;
    struct { const char *what; int N, K; } shapes[] = {
        {"QKV   256->768", 768, 256},
        {"out   256->256", 256, 256},
        {"FFN up 256->1536 (gate+up)", 1536, 256},
        {"FFN dn 768->256", 256, 768},
    };
    int x2[] = {7}, big[] = {7, 4, 5, 6}, all[] = {7, 4, 5, 6, 0, 1, 2, 3};
    double wx2[] = {1}, wbig[] = {1.2, 1, 1, 1}, wall[] = {1.2, 1, 1, 1, .25, .25, .25, .25};

    printf("\n== Packed int8 GEMM, Mimir-Reader shapes, M=%d tokens (GOPS) ==\n", M);
    printf("%-30s %8s %12s %10s\n", "matrix", "X2", "X2+3xA710", "all 8");
    double t_x2 = 0, t_big = 0, t_all = 0, ops_layer = 0;
    for (int s = 0; s < 4; s++) {
        int N = shapes[s].N, K = shapes[s].K;
        double ops = 2.0 * M * N * K;
        double g1 = gemm_bench(M, N, K, x2, 1, wx2, 20, s == 0);
        double g4 = gemm_bench(M, N, K, big, 4, wbig, 20, 0);
        double g8 = gemm_bench(M, N, K, all, 8, wall, 20, 0);
        printf("%-30s %8.1f %12.1f %10.1f\n", shapes[s].what, g1, g4, g8);
        t_x2 += ops / (g1 * 1e9); t_big += ops / (g4 * 1e9); t_all += ops / (g8 * 1e9);
        ops_layer += ops;
    }
    /* attention with window 128: QK^T and AV ≈ 4·w·d ops/token, run at GEMM speed */
    double attn = 4.0 * 128 * 256 * M;
    int L = 6;
    printf("\n== Projected Mimir-Reader (d=256, L=6, f=768, w=128) ==\n");
    printf("ops per token: %.2f M\n", (ops_layer + attn) * L / M / 1e6);
    const char *nm[] = {"X2 only", "X2 + 3xA710", "all 8 cores"};
    double tt[] = {t_x2, t_big, t_all};
    for (int i = 0; i < 3; i++) {
        double g = ops_layer / tt[i];             /* effective GOPS on these shapes */
        double per_layer = tt[i] + attn / g;      /* seconds for M tokens, one layer */
        double tps = M / (per_layer * L);
        printf("%-12s  %7.1f GOPS effective  ->  %8.0f tokens/sec (matmul + attention only)\n",
               nm[i], g / 1e9 * 1e0, tps);
    }
    return 0;
}
