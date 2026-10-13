# docs/SCHOOL_BRAIN.md tests, in the Colab session that read the school brain (it and its shelf are on this machine):
# three brains (empty, E, school), the same answering settings (LAYA_LEVEL.md stages 1-2), each test once.
# Results are printed and sent to the browser's computer as school_tests.txt.   Run:  %run /content/mimir/tools/school_tests_colab.py
import os, subprocess
from google.colab import files
def sh(cmd):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end="", flush=True)
    if p.wait(): raise SystemExit("failed: " + cmd)
os.chdir("/content/mimir"); sh("git pull -q && git log --oneline -1")
sh("gcc -O2 -std=gnu11 -D_GNU_SOURCE -w -o bg sandbox/neurons/bg.c src/meaning.c src/reader.c -Isrc -lm -fopenmp -lpthread")
R = "https://github.com/callmedraxx/mimir/releases/download/brain-E"
os.makedirs("data/pack", exist_ok=True)
for f in ("mimir_E.brain", "shelf_E.txt"):
    if not os.path.exists("data/pack/" + f): sh("curl -sL %s/%s.gz | gunzip > data/pack/%s" % (R, f, f))
T = " ".join("data/arc_tests/%s.tsv" % t for t in ("mmlu_hs_test", "csqa_dev", "easy_test", "challenge_test", "obqa_test"))
out = "/content/school_tests.txt"; open(out, "w").write("")
for name, brain in (("empty", "-"), ("E", "data/pack/mimir_E.brain"), ("school", "/content/school.brain")):
    sh("echo '=== %s' >> %s; NOPEN=20 POLARITY=1 STRENGTH=1 OMP_NUM_THREADS=1 ./bg arc %s %s 2>&1 | tee -a %s" % (name, out, brain, T, out))
files.download(out)
print("school tests done", flush=True)
