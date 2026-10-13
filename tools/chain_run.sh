#!/bin/sh
# docs/CHAIN_COMPARE.md: brain E (data/shelf/base.brain), tests once without (W=0) and with the chain (W=1, chosen on dev).
#   tools/chain_run.sh BG_BINARY     (run from the repo root)
set -e
BG=${1:-./bg}; R=sandbox/neurons/results; mkdir -p $R
TESTS="data/shelf/arc/easy_test.tsv data/shelf/arc/challenge_test.tsv data/shelf/arc/obqa_test.tsv"
echo "commit $(git rev-parse --short HEAD)"
echo "=== without the chain ==="; STRENGTH=1 BW="1 0.5 2 4 0" $BG arc data/shelf/base.brain $TESTS 2>&1 | tee $R/chain_without.txt
echo "=== with the chain ===";    STRENGTH=1 BW="1 0.5 2 4 1" $BG arc data/shelf/base.brain $TESTS 2>&1 | tee $R/chain_with.txt
