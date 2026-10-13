#!/bin/sh
# docs/SCHOOL_BRAIN.md in Colab: build the school shelf and read it into the school brain, no room limit.
# Everything that must survive Colab closing goes to Google Drive (mounted at /content/drive):
#   $D/shelf/*.gz      the shelf, kept as it is made (a restart takes it from there instead of downloading again)
#   $D/school.brain    the brain, saved after every shelf file (written whole, then renamed); a restart continues from it
set -e
D=${D:-/content/drive/MyDrive/mimir_school}
R=https://github.com/callmedraxx/mimir/releases/download/brain-E
mkdir -p "$D"
cd /content; [ -d mimir ] || git clone -q --depth 1 -b reader https://github.com/callmedraxx/mimir
cd mimir && git pull -q && echo "mimir $(git rev-parse --short HEAD) | $(nproc) cores | $(free -g | awk '/Mem/{print $2}') GB RAM | drive $(df -h "$D" | awk 'NR==2{print $4}') free"
mkdir -p data/understand; [ -f data/understand/wordclass.tsv ] || curl -sL $R/wordclass.tsv.gz | gunzip > data/understand/wordclass.tsv
gcc -O2 -std=gnu11 -D_GNU_SOURCE -w -o bg sandbox/neurons/bg.c src/meaning.c src/reader.c -Isrc -lm -fopenmp -lpthread
python3 tools/school_shelf.py "$D/shelf"
if [ -f "$D/school.brain" ]; then echo "continuing from $D/school.brain"; export FROM="$D/school.brain"; fi
ROOM=0 STRENGTH=1 SAVEEACH=1 ./bg read "$D/school.brain" 1000000000 data/school/0*.txt
ls -la "$D"
