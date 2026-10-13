#!/bin/sh
# Mimir (brain E, packed) on the no-passage tests (ARC-Easy, ARC-Challenge, OpenBookQA), in Colab or any Linux box.
#   !curl -sL https://raw.githubusercontent.com/callmedraxx/mimir/reader/tools/mimir_arc_colab.sh | sh
# Builds from source, downloads the brain from the brain-E release, checks it, answers on the CPU (one core).
set -e
[ -d mimir ] || git clone -q --depth 1 -b reader https://github.com/callmedraxx/mimir
cd mimir
R=https://github.com/callmedraxx/mimir/releases/download/brain-E
mkdir -p data/pack data/understand
for f in mimir_E.brain shelf_E.txt; do [ -f data/pack/$f ] || curl -sL $R/$f.gz | gunzip > data/pack/$f; done
[ -f data/understand/wordclass.tsv ] || curl -sL $R/wordclass.tsv.gz | gunzip > data/understand/wordclass.tsv
curl -sL $R/SHA256SUMS > /tmp/sums && (cd data/pack && grep -v wordclass /tmp/sums | sha256sum -c -) && (cd data/understand && grep wordclass /tmp/sums | sha256sum -c -)
gcc -O2 -std=gnu11 -D_GNU_SOURCE -w -o bg sandbox/neurons/bg.c src/meaning.c src/reader.c -Isrc -lm -fopenmp -lpthread
echo "mimir $(git rev-parse --short HEAD) | $(nproc) cores | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
[ -f tools/mimir_arc.env ] && . tools/mimir_arc.env && echo "settings: $(grep -v "^#" tools/mimir_arc.env)"
OMP_NUM_THREADS=1 STRENGTH=1 ./bg arc data/pack/mimir_E.brain data/arc_tests/easy_test.tsv data/arc_tests/challenge_test.tsv data/arc_tests/obqa_test.tsv
