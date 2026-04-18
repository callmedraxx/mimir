/* Dump vision_encode() output for all 26 images to stdout (CSV: name,f0,f1,...,f127) */
#include <stdio.h>
#include <stdlib.h>
#include "../src/mimir.h"

extern const char *alpha_word_bank_get(int letter, int idx);

int main(void) {
    float raw[VISION_RAW_SIZE];
    float feat[MIMIR_EMBEDDING_SIZE];
    for (int i = 0; i < 26; i++) {
        const char *word = alpha_word_bank_get(i, 0);
        if (!word) continue;
        char path[256];
        snprintf(path, sizeof(path), "data/images/%s.raw", word);
        if (vision_load_raw(path, raw) != 0) continue;
        vision_encode(raw, feat);
        printf("%s", word);
        for (int k = 0; k < MIMIR_EMBEDDING_SIZE; k++) printf(",%.6f", feat[k]);
        printf("\n");
    }
    return 0;
}
