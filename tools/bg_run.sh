#!/bin/sh
# docs/BACKGROUND_COMPARE.md: the test before reading, an hour of reading, the same test after.
#   tools/bg_run.sh BG_BINARY      (run from the repo root; data from tools/shelf_fetch.py, understand_fetch.py, race_fetch.py)
set -e
BG=${1:-./bg}; R=sandbox/neurons/results; mkdir -p $R
TESTS_ARC="data/shelf/arc/easy_test.tsv data/shelf/arc/challenge_test.tsv"
TESTS_KNOW="data/understand/race/test_middle.txt data/understand/mctest/mc160.test.txt data/understand/mctest/mc500.test.txt"
echo "commit $(git rev-parse --short HEAD)  BW=$BW WBG=$WBG"
echo "=== before reading (empty brain) ==="
$BG arc - $TESTS_ARC 2>&1 | tee $R/bg_before.txt
$BG know - $TESTS_KNOW 2>&1 | tee -a $R/bg_before.txt
echo "=== reading: Simple English Wikipedia, then the ARC Corpus, for at most one hour ==="
$BG read data/shelf/mimir.brain 3600 data/shelf/simplewiki.txt data/shelf/arc_corpus.txt 2>&1 | tee $R/bg_read.txt
echo "=== after reading ==="
$BG arc data/shelf/mimir.brain $TESTS_ARC 2>&1 | tee $R/bg_after.txt
$BG know data/shelf/mimir.brain $TESTS_KNOW 2>&1 | tee -a $R/bg_after.txt
