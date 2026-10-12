#!/bin/sh
# Run Mimir on Typed Decisions in Colab (or any Linux box), the same test as tools/laya_colab.py.
#   !curl -sL https://raw.githubusercontent.com/callmedraxx/mimir/reader/tools/mimir_colab.sh | sh
set -e
[ -d mimir ] || git clone -q --depth 1 -b reader https://github.com/callmedraxx/mimir.git
cd mimir
echo "commit $(git rev-parse --short HEAD) | $(nproc) CPU cores: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
python3 tools/decide_fetch.py
gcc -O2 -std=gnu11 -D_POSIX_C_SOURCE=200809L -w -o decide sandbox/neurons/decide.c src/meaning.c -Isrc -lm
echo "--- Mimir learned from train (one CPU core) ---"
for r in 1 2 3; do ./decide test data/decide/train.jsonl data/decide/test.jsonl mimir_T.jsonl | grep -E "^RESULT|^T on|^learned"; done
echo "--- Mimir zero-shot ---"
./decide zero data/decide/test.jsonl mimir_Z.jsonl | grep "^Z on"
python3 tools/decide_score.py mimir_T.jsonl
