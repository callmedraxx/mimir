# docs/EPISODE.md in Colab: what it is reading now, apart from what it knows.
#   %run /content/mimir/tools/episode_colab.py dev     P0-P3 on the dev passages (EPW chosen offline from one dump each),
#                                                       the general dev sets (must not move), the clock story
#   %run /content/mimir/tools/episode_colab.py test    after the choice is frozen in tools/episode_choice.env: the tests, once
# The brain: /content/cx.brain (+ .cx) if this session has it; else the school shelf is rebuilt and read with the cortex.
import glob, os, subprocess, sys
from google.colab import files
def sh(cmd, log=None):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1); out = []
    for line in p.stdout:
        print(line, end="", flush=True); out.append(line)
        if log and not line.startswith("  ") or (log and line.startswith("  ") and "\tpick=" in line): open(log, "a").write(line)
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
BRAIN = "/content/cx.brain"
if not os.path.exists(BRAIN + ".cx"):                   # the school brain with its cortex, read file by file as the release was
    if not os.path.exists("data/school/07_vital.txt"): sh("python3 tools/school_shelf.py && rm -rf data/school/raw")
    for f in sorted(glob.glob("data/school/0*.txt")):
        sh("%s./bg read %s 1000000000 %s" % ("FROM=%s " % BRAIN if os.path.exists(BRAIN) else "", BRAIN, f))

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
def accs(text, tag="bgpass"): return {l.split("file=")[1].split()[0]: float(l.split("acc=")[1].split()[0]) for l in text.splitlines() if l.startswith("RESULT " + tag)}
def offline(dump, W):
    """accuracy per set from a passages dump, the episode weighted W over the shelf (as bg scores, ties share)"""
    per = {}
    for l in open(dump):
        r = l.rstrip("\n").split("\t"); want, no, neg = int(r[1]), int(r[2]), int(r[3]); sc = []
        for o in range(no):
            sf, sj, so, me, fit, ef, ej, eo = map(float, r[4 + o].split())
            s = max(sf, W * ef) + 0.5 * max(sj, W * ej) + 2 * max(so, W * eo) + 4 * me + fit; sc.append(-s if neg else s)
        b = max(sc); nb = sum(1 for x in sc if x > b - 1e-9); ok = 1 / nb if sc[want] > b - 1e-9 else 0
        key = r[0].split(".")[0] if r[0].startswith("mc") else r[0].replace(".txt", "")
        per.setdefault(key, []).append(ok)
    return {k: 100 * sum(v) / len(v) for k, v in per.items()}

M = "data/understand/mctest/"; R = "data/understand/race/"
if mode == "dev":
    out = "/content/episode_dev.txt"; open(out, "w").write("")
    DEV = "%smc160.train.txt %smc160.dev.txt %smc500.train.txt %smc500.dev.txt %sdev_middle.txt" % (M, M, M, M, R)
    clock = stories("clock_dev", "data/story/clock_questions.tsv", ["clock"])
    res = {}
    for m in (0, 1, 2, 3):
        d = "/content/ep_dump_%d.txt" % m
        if os.path.exists(d): os.remove(d)
        sh("EPMODE=%d DUMP=%s ./bg passages %s %s" % (m, d, BRAIN, DEV), out)
        for W in ((1,) if m == 0 else (1, 2, 4)):
            r = offline(d, W); mean = sum(r.values()) / len(r); res[(m, W)] = mean
            open(out, "a").write("=== P%d EPW %d: %s mean %.2f\n" % (m, W, " ".join("%s=%.2f" % kv for kv in sorted(r.items())), mean))
    best = (0, 1)                                       # a step is kept only if more than 1 point above the step kept before it
    for m in (1, 2, 3):
        W = max((1, 2, 4), key=lambda w: (res[(m, w)], -w))
        if res[(m, W)] > res[best] + 1: best = (m, W)
    open(out, "a").write("=== chosen P%d EPW %d (mean %.2f; P0 %.2f)\n" % (best[0], best[1], res[best], res[(0, 1)]))
    for m, W in ((0, 1), best): sh("SHOW=1 EPMODE=%d EPW=%d ./bg passages %s %s" % (m, W, BRAIN, clock), out)
    G = " ".join("data/dev_sets/%s.tsv" % t for t in ("easy_dev", "challenge_dev", "obqa_dev", "mmlu_hs_dev", "csqa_traindev"))
    open(out, "a").write("=== general dev sets, no episode (CORTEX_LINK's choice gave 47.81 20.01 25.25 29.32 31.55)\n"); sh("./bg arc %s %s" % (BRAIN, G), out)
    files.download(out)
else:
    out = "/content/episode_test.txt"; open(out, "w").write("")
    choice = open("tools/episode_choice.env").read().strip(); open(out, "a").write("=== choice: %s\n" % choice)
    st = stories("stories_test", "data/story/episode_tests.tsv", ["salt", "train", "kite"])
    sl = stories("stories_link", "data/story/story_tests.tsv", ["harbor", "garden", "snow"])
    for env in ("EPMODE=0", choice):
        open(out, "a").write("=== %s\n" % env)
        sh("%s ./bg passages %s %smc160.test.txt %smc500.test.txt %stest_middle.txt" % (env, BRAIN, M, M, R), out)
        sh("SHOW=1 %s ./bg passages %s %s %s" % (env, BRAIN, st, sl), out)
    T = " ".join("data/arc_tests/%s.tsv" % t for t in ("mmlu_hs_test", "csqa_dev", "easy_test", "challenge_test", "obqa_test"))
    open(out, "a").write("=== general tests, no episode (CORTEX_LINK's test: 27.41 31.37 46.61 21.22 24.40)\n"); sh("./bg arc %s %s" % (BRAIN, T), out)
    files.download(out)
print("episode", mode, "done", flush=True)
