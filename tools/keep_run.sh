#!/bin/sh
# docs/KEEP_COMPARE.md: E (as C) and F (as D), with everything the starting brain knew learned for good.
#   tools/topic_split OUTDIR ... first (data/shelf/topic/sci_NN.txt, gen_NN.txt), then
#   tools/keep_run.sh BG_BINARY     (run from the repo root)
set -e
BG=${1:-./bg}; R=sandbox/neurons/results; mkdir -p $R
TESTS="data/shelf/arc/easy_test.tsv data/shelf/arc/challenge_test.tsv data/shelf/arc/obqa_test.tsv"
PRACTICE="data/shelf/arc/easy_train.tsv data/shelf/arc/challenge_train.tsv data/shelf/arc/obqa_train.tsv data/shelf/arc/sciq_train.tsv"
SCI=$(ls data/shelf/topic/sci_*.txt | sort)
MIX=$(for f in $SCI; do echo $f; echo $f | sed 's/sci_/gen_/'; done)
echo "commit $(git rev-parse --short HEAD)"
echo "=== E: on topic, what it knew kept ==="
FROM=data/shelf/mimir.brain KEEP=1 STRENGTH=1 PRACTICE="$PRACTICE" $BG read data/shelf/mimir_E.brain 100000 $SCI 2>&1 | tee $R/keep_read_E.txt
echo "=== tests E ===";  STRENGTH=1 $BG arc data/shelf/mimir_E.brain $TESTS 2>&1 | tee $R/keep_E.txt
echo "=== F: own rooms, what it knew kept ==="
FROM=data/shelf/mimir.brain KEEP=1 STRENGTH=1 ROOMS="4500000 1500000" PRACTICE="$PRACTICE" $BG read data/shelf/mimir_F.brain 100000 $MIX 2>&1 | tee $R/keep_read_F.txt
echo "=== tests F ===";  STRENGTH=1 $BG arc data/shelf/mimir_F.brain $TESTS 2>&1 | tee $R/keep_F.txt
