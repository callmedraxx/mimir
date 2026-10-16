# docs/STATE.md in Colab (free CPU runtime; brain-school-cx4 with its shelf, tools/brain_release.py).
#   %run /content/mimir/tools/state_colab.py dev    S4 = 0 (the adopted reader), 25 (S1 + "is in" + no-longer-so), 31 (all, with the
#                                                   derived states) on bAbI train tasks 1,2,3,6,9,11,13, the dev passages and the
#                                                   70 dev-story questions; then the general dev sets at S4=31 (should equal FOLLOW's)
#   %run /content/mimir/tools/state_colab.py test   after the freeze: the tests, once (S4 as frozen below)
import os, re, subprocess, sys
def sh(cmd, out=None):
    print("$", cmd, flush=True); r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-4000:], flush=True)
    if out: open(out, "a").write("$ %s\n%s" % (cmd, r.stdout))
    if r.returncode: raise SystemExit("failed: " + cmd)
    return r.stdout
mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
FROZEN = os.environ.get("S4", "31")
os.chdir("/content")
if not os.path.isdir("mimir"): sh("git clone -q --depth 1 -b reader https://github.com/callmedraxx/mimir")
os.chdir("/content/mimir"); sh("git pull -q && git log --oneline -1 && nproc && free -g | head -2")
sh("gcc -O3 -march=native -std=gnu11 -D_GNU_SOURCE -w -o bg sandbox/brain/bg.c src/meaning.c src/reader.c -Isrc -lm -fopenmp -lpthread")
os.makedirs("data/understand", exist_ok=True)
if not os.path.exists("data/understand/wordclass.tsv"):
    sh("curl -sL https://github.com/callmedraxx/mimir/releases/download/brain-E/wordclass.tsv.gz | gunzip > data/understand/wordclass.tsv")
if not os.path.exists("data/understand/mctest/mc500.test.txt"): sh("python3 -c \"import sys; sys.path.insert(0, 'tools'); import understand_fetch as u; u.mctest()\"")
if not os.path.exists("data/understand/race/test_middle.txt"): sh("python3 tools/race_fetch.py")
if not os.path.exists("data/understand/babi/qa1_test.txt"): sh("python3 -c \"import sys; sys.path.insert(0, 'tools'); import understand_fetch as u; u.babi()\"")
sh("python3 tools/survey_fetch.py babi_train" if mode == "dev" else "python3 tools/survey_fetch.py babi")
sys.path.insert(0, "tools"); import brain_release, stories
BRAIN = "/content/cx4.brain"
if not os.path.exists(BRAIN) and not brain_release.get(BRAIN): raise SystemExit("no brain release")
ST = "/content/stories"; os.makedirs(ST, exist_ok=True)
for n in stories.SETS: stories.write(ST, n)
M = "data/understand/mctest/"; R = "data/understand/race/"; SV = "data/understand/survey/"
def run(jobs, out):
    """run (name, cmd) jobs one after another; each one's lines go to out under its name (RESULT lines and the picks)"""
    for n, c in jobs:
        o = sh(c)
        with open(out, "a") as f:
            f.write("=== %s\n" % n)
            for l in o.splitlines():
                if not l.startswith("  ") or "\tpick=" in l: f.write(l + "\n")
        for l in o.splitlines():
            if l.startswith("RESULT"): print("==>", n, l, flush=True)
STORIES = ("clock.txt", "dev_stories.txt", "bakery.txt", "smart_tests.txt")
if mode == "dev":
    out = "/content/state_dev.txt"; open(out, "w").write("")
    DEV = "%sbabi_train.txt %smc160.train.txt %smc160.dev.txt %smc500.train.txt %smc500.dev.txt %sdev_middle.txt %s" % (SV, M, M, M, M, R, " ".join(ST + "/" + s for s in STORIES))
    run([("S4 %s" % m, "S4=%s SHOW=1 ./bg passages %s %s" % (m, BRAIN, DEV)) for m in ("0", "25", "31")], out)
    G = " ".join("data/dev_sets/%s.tsv" % t for t in ("easy_dev", "challenge_dev", "obqa_dev", "mmlu_hs_dev", "csqa_traindev"))
    run([("general S4 31", "S4=31 ./bg arc %s %s" % (BRAIN, G))], out)
    sh("python3 tools/survey_notes.py %s 0 | head -12" % out)
else:
    out = "/content/state_test.txt"; open(out, "w").write("=== choice: S4=%s (frozen; docs/STATE.md)\n" % FROZEN)
    T = "%sbabi.txt %smc160.test.txt %smc500.test.txt %stest_middle.txt %s" % (SV, M, M, R, " ".join(ST + "/" + s for s in ("state_tests.txt", "follow_tests.txt", "read2_tests.txt", "comprehend_tests.txt", "episode_tests.txt", "story_tests.txt")))
    run([("S0 new stories + bAbI test", "S4=0 SHOW=1 ./bg passages %s %sbabi.txt %s/state_tests.txt" % (BRAIN, SV, ST)),
         ("S4 %s tests" % FROZEN, "S4=%s SHOW=1 ./bg passages %s %s" % (FROZEN, BRAIN, T))], out)
    G = " ".join("data/arc_tests/%s.tsv" % t for t in ("mmlu_hs_test", "csqa_dev", "easy_test", "challenge_test", "obqa_test"))
    run([("general S4 %s" % FROZEN, "S4=%s ./bg arc %s %s" % (FROZEN, BRAIN, G))], out)
print("done", flush=True)
