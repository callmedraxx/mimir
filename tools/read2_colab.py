# docs/READ2.md in Colab.
#   %run /content/mimir/tools/read2_colab.py measure   step 0: brain-school-cx2 with its shelf; the general dev sets, then
#                                                      the five general tests once (COMPREHEND's measurement, rerun)
# The brain and its shelf: the GitHub release (tools/brain_release.py). Stops if open book is not working.
import os, re, subprocess, sys
from google.colab import files
def sh(cmd, out=None):
    print("$", cmd, flush=True); r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout, flush=True)
    if out: open(out, "a").write("$ %s\n%s" % (cmd, r.stdout))
    if r.returncode: raise SystemExit("failed: " + cmd)
    return r.stdout
mode = sys.argv[1] if len(sys.argv) > 1 else "measure"
os.chdir("/content")
if not os.path.isdir("mimir"): sh("git clone -q --depth 1 -b reader https://github.com/callmedraxx/mimir")
os.chdir("/content/mimir"); sh("git pull -q && git log --oneline -1 && nproc && free -g | head -2")
sh("gcc -O3 -march=native -std=gnu11 -D_GNU_SOURCE -w -o bg sandbox/brain/bg.c src/meaning.c src/reader.c -Isrc -lm -fopenmp -lpthread")
os.makedirs("data/understand", exist_ok=True)
if not os.path.exists("data/understand/wordclass.tsv"):
    sh("curl -sL https://github.com/callmedraxx/mimir/releases/download/brain-E/wordclass.tsv.gz | gunzip > data/understand/wordclass.tsv")
sys.path.insert(0, "tools"); import brain_release
BRAIN = "/content/cx2.brain"
if not brain_release.get(BRAIN): raise SystemExit("no brain release")
sh("ls -l data/school/")
if mode == "measure":
    out = "/content/read2_measure.txt"; open(out, "w").write("")
    G = " ".join("data/dev_sets/%s.tsv" % t for t in ("easy_dev", "challenge_dev", "obqa_dev", "mmlu_hs_dev", "csqa_traindev"))
    open(out, "a").write("=== general dev sets, brain-school-cx2 with its shelf (before COMPREHEND, with the shelf: 47.81 20.01 25.25 29.32 31.55)\n")
    o = sh("./bg arc %s %s" % (BRAIN, G), out)
    m = re.search(r"easy_dev.tsv.*open ([\d.]+)", o)
    if not m or float(m.group(1)) < 35: raise SystemExit("open book is not working (open alone %s on ARC-Easy dev)" % (m and m.group(1)))
    T = " ".join("data/arc_tests/%s.tsv" % t for t in ("mmlu_hs_test", "csqa_dev", "easy_test", "challenge_test", "obqa_test"))
    open(out, "a").write("=== general tests, once, brain-school-cx2 with its shelf (before COMPREHEND: 27.41 31.37 46.61 21.22 24.40)\n")
    sh("./bg arc %s %s" % (BRAIN, T), out)
    files.download(out)
print("read2", mode, "done", flush=True)
