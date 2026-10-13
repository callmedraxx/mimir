#!/bin/sh
# docs/TOPIC_COMPARE.md: C (read on topic only) and D (all of it, with a subject room and a general room).
#   tools/topic_split OUTDIR ... first (data/shelf/topic/sci_NN.txt, gen_NN.txt), then
#   tools/topic_run.sh BG_BINARY     (run from the repo root)
set -e
BG=${1:-./bg}; R=sandbox/neurons/results; mkdir -p $R
TESTS="data/shelf/arc/easy_test.tsv data/shelf/arc/challenge_test.tsv data/shelf/arc/obqa_test.tsv"
PRACTICE="data/shelf/arc/easy_train.tsv data/shelf/arc/challenge_train.tsv data/shelf/arc/obqa_train.tsv data/shelf/arc/sciq_train.tsv"
SCI=$(ls data/shelf/topic/sci_*.txt | sort)
MIX=$(for f in $SCI; do echo $f; echo $f | sed 's/sci_/gen_/'; done)
echo "commit $(git rev-parse --short HEAD)"
echo "=== C: on topic only ==="
FROM=data/shelf/mimir.brain STRENGTH=1 PRACTICE="$PRACTICE" $BG read data/shelf/mimir_C.brain 100000 $SCI 2>&1 | tee $R/topic_read_C.txt
echo "=== tests C ===";  STRENGTH=1 $BG arc data/shelf/mimir_C.brain $TESTS 2>&1 | tee $R/topic_C.txt
echo "=== D: on topic + general, each in its own room ==="
FROM=data/shelf/mimir.brain STRENGTH=1 ROOMS="4500000 1500000" PRACTICE="$PRACTICE" $BG read data/shelf/mimir_D.brain 100000 $MIX 2>&1 | tee $R/topic_read_D.txt
echo "=== tests D ===";  STRENGTH=1 $BG arc data/shelf/mimir_D.brain $TESTS 2>&1 | tee $R/topic_D.txt
