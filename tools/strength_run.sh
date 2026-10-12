#!/bin/sh
# docs/STRENGTH_COMPARE.md: before / A (read more, one hour) / B (same text + strength, fading, spaced practice).
#   tools/strength_run.sh BG_BINARY     (run from the repo root after tools/enwiki_fetch.py, practice_fetch.py)
set -e
BG=${1:-./bg}; R=sandbox/neurons/results; mkdir -p $R
TESTS="data/shelf/arc/easy_test.tsv data/shelf/arc/challenge_test.tsv data/shelf/arc/obqa_test.tsv"
SHELF=$(ls data/shelf/enwiki_*.txt | sort)
PRACTICE="data/shelf/arc/easy_train.tsv data/shelf/arc/challenge_train.tsv data/shelf/arc/obqa_train.tsv data/shelf/arc/sciq_train.tsv"
echo "commit $(git rev-parse --short HEAD)"
echo "=== before ===";  $BG arc data/shelf/mimir.brain $TESTS 2>&1 | tee $R/strength_before.txt
echo "=== A: read more for one hour ==="
FROM=data/shelf/mimir.brain $BG read data/shelf/mimir_A.brain 3600 $SHELF 2>&1 | tee $R/strength_read_A.txt
STOP=$(grep -o "read_bytes=[0-9]*" $R/strength_read_A.txt | cut -d= -f2)
echo "=== B: the same $STOP bytes, with strength and practice ==="
FROM=data/shelf/mimir.brain STRENGTH=1 STOPBYTES=$STOP PRACTICE="$PRACTICE" $BG read data/shelf/mimir_B.brain 100000 $SHELF 2>&1 | tee $R/strength_read_B.txt
echo "=== tests A ===";  $BG arc data/shelf/mimir_A.brain $TESTS 2>&1 | tee $R/strength_A.txt
echo "=== tests B ===";  STRENGTH=1 $BG arc data/shelf/mimir_B.brain $TESTS 2>&1 | tee $R/strength_B.txt
