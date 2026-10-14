# docs/RANKING_COMPARE.md in Colab: the released school brain (checked by SHA-256), its shelf rebuilt at the same paths,
# then the cells.   %run /content/mimir/tools/ranking_colab.py dev          the four cells on the five dev sets
#                   %run /content/mimir/tools/ranking_colab.py test        the frozen choice (tools/ranking.env) on the five tests, once
# Results are printed and sent to the browser's computer (ranking_dev.txt / ranking_test.txt).
import glob, hashlib, os, subprocess, sys
from google.colab import files
def sh(cmd):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end="", flush=True)
    if p.wait(): raise SystemExit("failed: " + cmd)
mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
os.chdir("/content")
if not os.path.isdir("mimir"): sh("git clone -q --depth 1 -b reader https://github.com/callmedraxx/mimir")
os.chdir("/content/mimir"); sh("git pull -q && git log --oneline -1 && nproc && free -g | head -2")
sh("gcc -O2 -std=gnu11 -D_GNU_SOURCE -w -o bg sandbox/neurons/bg.c src/meaning.c src/reader.c -Isrc -lm -fopenmp -lpthread")
os.makedirs("data/understand", exist_ok=True)
if not os.path.exists("data/understand/wordclass.tsv"):
    sh("curl -sL https://github.com/callmedraxx/mimir/releases/download/brain-E/wordclass.tsv.gz | gunzip > data/understand/wordclass.tsv")
BRAIN = "/content/school.brain"
if not os.path.exists(BRAIN):
    R = "https://github.com/callmedraxx/mimir/releases/download/brain-school"
    os.makedirs("/content/rel", exist_ok=True)
    sh("cd /content/rel && curl -sLO %s/SHA256SUMS && curl -sLO %s/school.brain.gz.part_00 && curl -sLO %s/school.brain.gz.part_01 && sha256sum -c SHA256SUMS" % (R, R, R))
    sh("cat /content/rel/school.brain.gz.part_* | gunzip > %s && rm -rf /content/rel && ls -l %s" % (BRAIN, BRAIN))
if not os.path.exists("data/school/07_vital.txt"): sh("python3 tools/school_shelf.py && rm -rf data/school/raw")
sh("ls -l data/school/")
ENV = "NOPEN=20 POLARITY=1 STRENGTH=1 OMP_NUM_THREADS=1"
if mode == "dev":
    T = " ".join("data/dev_sets/%s.tsv" % t for t in ("easy_dev", "challenge_dev", "obqa_dev", "mmlu_hs_dev", "csqa_traindev"))
    cells = (("G0", "NCAND=200"), ("G1", "NCAND=1000"), ("J0", "NCAND=200 CANDJOIN=1"), ("J1", "NCAND=1000 CANDJOIN=1"))
    out = "/content/ranking_dev.txt"
else:
    T = " ".join("data/arc_tests/%s.tsv" % t for t in ("mmlu_hs_test", "csqa_dev", "easy_test", "challenge_test", "obqa_test"))
    cells = (("chosen", open("tools/ranking.env").read().strip()),)
    out = "/content/ranking_test.txt"
open(out, "w").write("")
for name, env in cells:
    sh("echo '=== %s %s' >> %s; %s %s ./bg arc %s %s 2>&1 | tee -a %s" % (name, env, out, ENV, env, BRAIN, T, out))
files.download(out)
print("ranking", mode, "done", flush=True)
