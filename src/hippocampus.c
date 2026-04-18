/*
 * MIMIR — Hippocampus: Episodic Mistake Memory
 *
 * The hippocampus is the brain's index of "what went wrong and when".
 * It does NOT store what words mean — that knowledge lives in the network
 * weights. It stores only ERROR METADATA: which letter was gotten wrong,
 * on which round, and how often. This distinction is critical:
 *
 *   Network weights  →  semantic memory  (what A stands for)
 *   Hippocampus      →  episodic memory  (I got A wrong 3 rounds ago)
 *
 * In the biological brain, the hippocampus does not store the content of
 * memories directly. It stores INDICES — pointers back to the cortical
 * circuits that hold the content. When you "remember" an event, the
 * hippocampus retrieves the index and the cortex reconstructs the experience.
 *
 * Our implementation follows the same principle:
 *   - The mistake ring stores letter_idx + round number (the "index")
 *   - The semantic content (what word was predicted vs correct) is NOT saved
 *   - The hippocampus guides WHICH letters get prioritised for practice,
 *     not WHAT to say about them — the network must re-derive that itself
 *
 * WHY THIS MATTERS FOR LEARNING:
 *
 * A model that memorises "I thought A=banana" is not learning a principle.
 * It is building a lookup table for its own past errors. Over time this
 * becomes a crutch — it avoids "A" not because it learned the A→apple
 * pattern but because it memorised the correction.
 *
 * By recording only letter_idx and round, the hippocampus forces the
 * network to RE-LEARN the correct pattern through the weight-update
 * mechanism (Hebbian learning). The episodic tag just ensures the right
 * letters get extra practice sessions.
 *
 * ════════════════════════════════════════════════════════════════════
 * PRIORITY SCORING
 * ════════════════════════════════════════════════════════════════════
 *
 * hippo_priority(h, letter) returns a score ≥ 0.0 used to weight the
 * sampling distribution in the quiz loop.
 *
 *   score = error_count * recency_boost
 *   recency_boost = 3.0 if last error < 3 rounds ago, else 1.0
 *
 * Letters never gotten wrong score 0 but are still included in the
 * quiz (every letter must be tested each full pass).
 *
 * ════════════════════════════════════════════════════════════════════
 * PERSISTENCE
 * ════════════════════════════════════════════════════════════════════
 *
 * Saved to HIPPO_PATH in a binary format. The file contains ONLY
 * numeric data — no word strings. Loading it on a fresh brain gives
 * the brain knowledge of its past failure patterns without giving it
 * the semantic shortcuts it must learn on its own.
 */

#include "mimir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

void hippo_init(Hippocampus *h) {
    memset(h, 0, sizeof(*h));
    for (int i = 0; i < 26; i++) {
        h->last_error_round[i] = -1;   /* -1 = never made this mistake */
        h->streak_correct[i]   =  0;
    }
    h->ring_head   = 0;
    h->ring_count  = 0;
    h->total_rounds = 0;
}

/* ── Recording events ────────────────────────────────────────────────────── */

void hippo_record_mistake(Hippocampus *h, int letter_idx) {
    if (letter_idx < 0 || letter_idx >= 26) return;

    h->error_count[letter_idx]++;
    h->last_error_round[letter_idx] = h->total_rounds;
    h->streak_correct[letter_idx]   = 0;   /* reset correct streak */

    /* Write into the ring buffer — overwrites oldest entry when full. */
    HippoMistake *slot = &h->ring[h->ring_head % HIPPO_RING_SIZE];
    slot->letter_idx = letter_idx;
    slot->round      = h->total_rounds;
    h->ring_head     = (h->ring_head + 1) % HIPPO_RING_SIZE;
    if (h->ring_count < HIPPO_RING_SIZE) h->ring_count++;
}

void hippo_record_correct(Hippocampus *h, int letter_idx) {
    if (letter_idx < 0 || letter_idx >= 26) return;
    h->correct_count[letter_idx]++;
    h->streak_correct[letter_idx]++;
}

void hippo_advance_round(Hippocampus *h) {
    h->total_rounds++;
}

/* ── Priority ────────────────────────────────────────────────────────────── */

/*
 * Priority score for a given letter — higher means "needs more practice".
 *
 * Recency boost: if the last mistake was within HIPPO_RECENCY_WINDOW rounds,
 * multiply the error count by HIPPO_RECENCY_BOOST. This means a letter
 * gotten wrong recently is prioritised over one that was hard long ago
 * but has since been mastered.
 *
 * A letter never gotten wrong returns 0.0 (will still appear in the quiz
 * because the quiz always covers all taught letters each full pass).
 */
float hippo_priority(const Hippocampus *h, int letter_idx) {
    if (letter_idx < 0 || letter_idx >= 26) return 0.0f;

    int errors = h->error_count[letter_idx];
    if (errors == 0) return 0.0f;

    int last = h->last_error_round[letter_idx];
    int recency = h->total_rounds - last;   /* rounds since last error */

    float boost = (recency <= HIPPO_RECENCY_WINDOW) ? HIPPO_RECENCY_BOOST : 1.0f;
    return (float)errors * boost;
}

/* ── Display ─────────────────────────────────────────────────────────────── */

void hippo_print(const Hippocampus *h) {
    printf("\n  ── Hippocampus (Mistake Memory) ──────────────────────\n");
    printf("  Total quiz rounds: %d\n", h->total_rounds);

    int any = 0;
    for (int i = 0; i < 26; i++) {
        if (h->error_count[i] == 0) continue;
        any = 1;
        int recency = (h->last_error_round[i] >= 0)
                      ? h->total_rounds - h->last_error_round[i]
                      : -1;
        printf("    %c  errors=%-3d  correct=%-3d  streak=%-3d  "
               "last_error=%d round(s) ago\n",
               (char)('A' + i),
               h->error_count[i], h->correct_count[i],
               h->streak_correct[i],
               recency);
    }
    if (!any) printf("  No mistakes recorded yet.\n");

    if (h->ring_count > 0) {
        int show = h->ring_count < 8 ? h->ring_count : 8;
        printf("\n  Last %d mistake(s) (most recent first):\n", show);
        for (int i = 0; i < show; i++) {
            /* Read backwards from ring_head */
            int idx = ((h->ring_head - 1 - i) + HIPPO_RING_SIZE) % HIPPO_RING_SIZE;
            printf("    round %-4d  →  letter %c\n",
                   h->ring[idx].round,
                   (char)('A' + h->ring[idx].letter_idx));
        }
    }
    printf("  ──────────────────────────────────────────────────────\n\n");
}

/* ── Persistence ─────────────────────────────────────────────────────────── */

/*
 * Binary save format:
 *   [4 bytes] magic  0x48495050  ("HIPP")
 *   [4 bytes] version 1
 *   [4 bytes] total_rounds
 *   [26*4 bytes] error_count[26]
 *   [26*4 bytes] correct_count[26]
 *   [26*4 bytes] last_error_round[26]
 *   [26*4 bytes] streak_correct[26]
 *   [4 bytes] ring_head
 *   [4 bytes] ring_count
 *   [HIPPO_RING_SIZE * sizeof(HippoMistake)] ring[]
 *
 * No strings. No word indices. Pure error metadata.
 */
#define HIPPO_MAGIC   0x48495050u
#define HIPPO_VERSION 1u

int hippo_save(const Hippocampus *h, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[Hippo] Cannot open '%s' for writing\n", path);
        return -1;
    }

    uint32_t magic   = HIPPO_MAGIC;
    uint32_t version = HIPPO_VERSION;
    fwrite(&magic,   sizeof(magic),   1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&h->total_rounds, sizeof(int), 1, f);
    fwrite(h->error_count,      sizeof(int), 26, f);
    fwrite(h->correct_count,    sizeof(int), 26, f);
    fwrite(h->last_error_round, sizeof(int), 26, f);
    fwrite(h->streak_correct,   sizeof(int), 26, f);
    fwrite(&h->ring_head,  sizeof(int), 1, f);
    fwrite(&h->ring_count, sizeof(int), 1, f);
    fwrite(h->ring, sizeof(HippoMistake), HIPPO_RING_SIZE, f);

    fclose(f);
    return 0;
}

int hippo_load(Hippocampus *h, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;   /* Not found — caller uses fresh hippocampus */

    uint32_t magic, version;
    if (fread(&magic,   sizeof(magic),   1, f) != 1 || magic   != HIPPO_MAGIC  ||
        fread(&version, sizeof(version), 1, f) != 1 || version != HIPPO_VERSION) {
        fclose(f);
        return -1;   /* Corrupt or wrong version */
    }

    hippo_init(h);

    /* Validate every read — a short read means the file is corrupt. */
    int ok = 1;
    ok &= (fread(&h->total_rounds,    sizeof(int), 1,  f) == 1);
    ok &= (fread(h->error_count,      sizeof(int), 26, f) == 26);
    ok &= (fread(h->correct_count,    sizeof(int), 26, f) == 26);
    ok &= (fread(h->last_error_round, sizeof(int), 26, f) == 26);
    ok &= (fread(h->streak_correct,   sizeof(int), 26, f) == 26);
    ok &= (fread(&h->ring_head,       sizeof(int), 1,  f) == 1);
    ok &= (fread(&h->ring_count,      sizeof(int), 1,  f) == 1);
    ok &= (fread(h->ring, sizeof(HippoMistake), HIPPO_RING_SIZE, f)
           == (size_t)HIPPO_RING_SIZE);

    fclose(f);
    if (!ok) { hippo_init(h); return -1; }   /* truncated file */
    return 0;
}
