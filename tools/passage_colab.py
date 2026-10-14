# docs/PASSAGE.md in Colab: answering about a passage by understanding it.
#   %run /content/mimir/tools/passage_colab.py dev     U0-U3 on the dev passages (EA, SW chosen offline from two dumps),
#                                                       the general dev sets (must not move), the clock story
#   %run /content/mimir/tools/passage_colab.py test    after the choice is frozen in tools/passage_choice.env: the tests, once
# The brain: from the GitHub release brain-school-cx if there is one; else the school shelf is rebuilt and read with the
# cortex (about 1.5 hours) and, with the owner's GH_TOKEN in Colab Secrets, saved there for next time.
import glob, os, subprocess, sys
from google.colab import files
def sh(cmd, log=None):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1); out = []
    for line in p.stdout:
        print(line, end="", flush=True); out.append(line)
        if log and (not line.startswith("  ") or "\tpick=" in line): open(log, "a").write(line)
    if p.wait(): raise SystemExit("failed: " + cmd)
    return "".join(out)
mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
os.chdir("/content")
if not os.path.isdir("mimir"): sh("git clone -q --depth 1 -b reader https://github.com/callmedraxx/mimir")
os.chdir("/content/mimir"); sh("git pull -q && git log --oneline -1 && nproc && free -g | head -2")
sh("gcc -O3 -march=native -std=gnu11 -D_GNU_SOURCE -w -o bg sandbox/brain/bg.c src/meaning.c src/reader.c -Isrc -lm -fopenmp -lpthread")
os.makedirs("data/understand", exist_ok=True)
if not os.path.exists("data/understand/wordclass.tsv"):
    sh("curl -sL https://github.com/callmedraxx/mimir/releases/download/brain-E/wordclass.tsv.gz | gunzip > data/understand/wordclass.tsv")
if not os.path.exists("data/understand/mctest/mc500.test.txt"): sh("python3 -c \"import sys; sys.path.insert(0, 'tools'); import understand_fetch as u; u.mctest()\"")
if not os.path.exists("data/understand/race/test_middle.txt"): sh("python3 tools/race_fetch.py")
sys.path.insert(0, "tools"); import brain_release
BRAIN = "/content/cx.brain"
if not brain_release.get(BRAIN):                        # the school brain with its cortex, read file by file as the release was
    if not os.path.exists("data/school/07_vital.txt"): sh("python3 tools/school_shelf.py && rm -rf data/school/raw")
    for f in sorted(glob.glob("data/school/0*.txt")):
        sh("%s./bg read %s 1000000000 %s" % ("FROM=%s " % BRAIN if os.path.exists(BRAIN) else "", BRAIN, f))
    brain_release.put(BRAIN)

def stories(name, qfile, names):
    """stories in the passage layout: S / T (the story on one line) / Q single question A B C D letter"""
    out = "/content/%s.txt" % name
    with open(out, "w") as f:
        for s in names:
            f.write("S\tstory.%s\nT\t%s\n" % (s, " ".join(open("data/story/%s.txt" % s).read().split())))
            for l in open(qfile):
                q = l.rstrip("\n").split("\t")
                if q[0].rsplit("_", 1)[0] != s: continue
                f.write("Q\tsingle\t%s\t%s\t%s\n" % (q[1], "\t".join(q[3:7]), "ABCD"[int(q[7])]))
    return out
def offline(dump, EA, SW):
    """accuracy per set from a passages dump: inside an active episode meaning and fit scaled by EA, the slot weighted SW"""
    per = {}
    for l in open(dump):
        r = l.rstrip("\n").split("\t"); want, no, neg = int(r[1]), int(r[2]), int(r[3]); sc = []
        for o in range(no):
            v = r[4 + o].split(); f, j, op, me, fit, slot = map(float, v[:6]); act = int(v[6]); ea = EA if act else 1
            s = f + 0.5 * j + 2 * op + 4 * ea * me + ea * fit + SW * slot; sc.append(-s if neg else s)
        b = max(sc); nb = sum(1 for x in sc if x > b - 1e-9); ok = 1 / nb if sc[want] > b - 1e-9 else 0
        key = r[0].split(".")[0] if r[0].startswith("mc") else r[0].replace(".txt", "")
        per.setdefault(key, []).append(ok)
    return {k: 100 * sum(v) / len(v) for k, v in per.items()}
def mean(r): return sum(r.values()) / len(r)

M = "data/understand/mctest/"; R = "data/understand/race/"
OLD = ["salt", "train", "kite"], ["harbor", "garden", "snow"]
if mode == "dev":
    out = "/content/passage_dev.txt"; open(out, "w").write("")
    DEV = "%smc160.train.txt %smc160.dev.txt %smc500.train.txt %smc500.dev.txt %sdev_middle.txt" % (M, M, M, M, R)
    D = {}
    for ps in (2, 3):
        D[ps] = "/content/pa_dump_%d.txt" % ps
        if os.path.exists(D[ps]): os.remove(D[ps])
        sh("PSLOT=%d DUMP=%s ./bg passages %s %s" % (ps, D[ps], BRAIN, DEV), out)
    log = lambda name, EA, SW, r: open(out, "a").write("=== %s EA %.2f SW %.2f: %s mean %.2f\n" % (name, EA, SW, " ".join("%s=%.2f" % kv for kv in sorted(r.items())), mean(r)))
    u0 = offline(D[2], 1, 0); log("U0", 1, 0, u0)
    u1 = {ea: offline(D[2], ea, 0) for ea in (0, 0.25, 0.5)}
    for ea, r in u1.items(): log("U1", ea, 0, r)
    EA = max(u1, key=lambda e: (mean(u1[e]), e))
    u2 = {sw: offline(D[2], EA, sw) for sw in (1, 2, 4)}
    for sw, r in u2.items(): log("U2", EA, sw, r)
    SW2 = max(u2, key=lambda w: (mean(u2[w]), -w))
    u3 = {sw: offline(D[3], EA, sw) for sw in (1, 2, 4)}
    for sw, r in u3.items(): log("U3", EA, sw, r)
    SW3 = max(u3, key=lambda w: (mean(u3[w]), -w))
    best, bm = "PSLOT=0", mean(u0)                       # a step is kept only if more than 1 point above the step kept before it
    if mean(u1[EA]) > bm + 1: best, bm = "EA=%g" % EA, mean(u1[EA])
    ea_kept = EA if best != "PSLOT=0" else 1
    if mean(offline(D[2], ea_kept, SW2)) > bm + 1: best, bm = "EA=%g PSLOT=2 SW=%g" % (ea_kept, SW2), mean(offline(D[2], ea_kept, SW2))
    if mean(offline(D[3], ea_kept, SW3)) > bm + 1: best, bm = "EA=%g PSLOT=3 SW=%g" % (ea_kept, SW3), mean(offline(D[3], ea_kept, SW3))
    open(out, "a").write("=== chosen: %s (mean %.2f; U0 %.2f)\n" % (best, bm, mean(u0)))
    clock = stories("clock_dev", "data/story/clock_questions.tsv", ["clock"])
    for env in ("PSLOT=0", best): sh("SHOW=1 %s ./bg passages %s %s" % (env, BRAIN, clock), out)
    G = " ".join("data/dev_sets/%s.tsv" % t for t in ("easy_dev", "challenge_dev", "obqa_dev", "mmlu_hs_dev", "csqa_traindev"))
    open(out, "a").write("=== general dev sets, no episode (must be 47.81 20.01 25.25 29.32 31.55)\n"); sh("./bg arc %s %s" % (BRAIN, G), out)
    files.download(out)
else:
    out = "/content/passage_test.txt"; open(out, "w").write("")
    choice = open("tools/passage_choice.env").read().strip(); open(out, "a").write("=== choice: %s\n" % choice)
    st = stories("stories_new", "data/story/passage_tests.tsv", ["bicycle", "stoneowl", "swim"])
    so = stories("stories_old", "data/story/episode_tests.tsv", OLD[0]); sl = stories("stories_link", "data/story/story_tests.tsv", OLD[1])
    for env in ("PSLOT=0", choice):
        open(out, "a").write("=== %s\n" % env)
        sh("%s ./bg passages %s %smc160.test.txt %smc500.test.txt %stest_middle.txt" % (env, BRAIN, M, M, R), out)
        sh("SHOW=1 %s ./bg passages %s %s %s %s" % (env, BRAIN, st, so, sl), out)
    T = " ".join("data/arc_tests/%s.tsv" % t for t in ("mmlu_hs_test", "csqa_dev", "easy_test", "challenge_test", "obqa_test"))
    open(out, "a").write("=== general tests, no episode (must be 27.41 31.37 46.61 21.22 24.40)\n"); sh("./bg arc %s %s" % (BRAIN, T), out)
    files.download(out)
print("passage", mode, "done", flush=True)
