#!/usr/bin/env python3
"""docs/SURVEY.md in Colab (free CPU runtime), once:
  !git clone -q -b reader https://github.com/callmedraxx/mimir /content/mimir
  %cd /content/mimir
  %run tools/survey_colab.py
Builds bg, downloads brain-school-cx4 with its shelf (tools/brain_release.py), fetches the tests (tools/survey_fetch.py,
tools/race_fetch.py), runs each test file with SHOW=1 and writes /content/survey.txt (RESULT lines + every pick)."""
import os, subprocess, sys
sys.path.insert(0, "tools"); import brain_release
OUT = "/content/survey.txt"
def sh(cmd, log=None):
    print("$", cmd, flush=True); r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT); print(r.stdout[-3000:], flush=True)
    if log: open(log, "a").write("$ %s\n%s" % (cmd, r.stdout))
    return r
sh("nproc && free -g | head -2")
sh("gcc -O3 -march=native -std=gnu11 -D_GNU_SOURCE -w -o bg sandbox/brain/bg.c src/meaning.c src/reader.c -Isrc -lm -fopenmp -lpthread")
os.makedirs("data/understand", exist_ok=True)
if not os.path.exists("data/understand/wordclass.tsv"):
    sh("curl -sL https://github.com/callmedraxx/mimir/releases/download/brain-E/wordclass.tsv.gz | gunzip > data/understand/wordclass.tsv")
BRAIN = "/content/cx4.brain"
if not os.path.exists(BRAIN): brain_release.get(BRAIN)
if not os.path.exists("data/understand/race/test_high.txt"): sh("python3 tools/race_fetch.py")
if not os.path.exists("data/understand/babi/qa1_test.txt"): sh("python3 -c \"import sys; sys.path.insert(0, 'tools'); import understand_fetch as u; u.babi()\"")
for attempt in range(3):
    if sh("python3 tools/survey_fetch.py", OUT).returncode == 0: break
ORDER = ["wsc273", "winogrande", "storycloze", "sciq", "dream", "boolq", "cosmosqa", "race_high", "multirc", "record", "lambada", "cbt_ne", "cbt_cn", "cbt_v", "cbt_p", "babi", "quality"]
for t in ORDER:
    f = "data/understand/survey/%s.txt" % t
    if not os.path.exists(f): open(OUT, "a").write("=== %s: no file\n" % t); continue
    open(OUT, "a").write("=== %s\n" % t)
    r = sh("SHOW=1 ./bg passages %s %s" % (BRAIN, f), OUT)
    for l in r.stdout.split("\n"):
        if l.startswith("RESULT"): print("==>", l, flush=True)
print("done", flush=True)
