# docs/COMPREHEND.md in Colab: reading what a sentence says ("not", names, what a thing is).
#   %run /content/mimir/tools/comprehend_colab.py dev    R0, R1 (NW 0.5 / 1 / 2), R2 (RD=1, NW 0.5 / 1 / 2) on the dev passages
#                                                        and dev stories; the general dev sets for each kept cell
#   %run /content/mimir/tools/comprehend_colab.py test   after the choice is frozen in tools/comprehend_choice.env: the tests, once
# The brain: the GitHub release brain-school-cx (tools/brain_release.py). Two runs at a time (the brain fits twice).
import os, re, subprocess, sys
from google.colab import files
def sh(cmd):
    print("$", cmd, flush=True); r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout, flush=True)
    if r.returncode: raise SystemExit("failed: " + cmd)
    return r.stdout
def pair(jobs, out):
    """run (name, cmd) jobs two at a time; each one's lines go to out under its name"""
    for i in range(0, len(jobs), 2):
        ps = [(n, c, subprocess.Popen(c, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)) for n, c in jobs[i:i + 2]]
        for n, c, p in ps:
            o = p.communicate()[0]; print("$", c, "\n" + o, flush=True)
            if p.returncode: raise SystemExit("failed: " + c)
            with open(out, "a") as f:
                f.write("=== %s\n" % n)
                for l in o.splitlines():
                    if not l.startswith("  ") or "\tpick=" in l: f.write(l + "\n")
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
sys.path.insert(0, "tools"); import brain_release, stories
BRAIN = "/content/cx.brain"
if not brain_release.get(BRAIN): raise SystemExit("no brain release")
ST = "/content/stories"; os.makedirs(ST, exist_ok=True)
for n in stories.SETS: stories.write(ST, n)
M = "data/understand/mctest/"; R = "data/understand/race/"

def acc(out):
    """{(cell, file): accuracy} from the RESULT lines under each === cell"""
    r, cell = {}, None
    for l in open(out):
        if l.startswith("=== "): cell = l[4:].strip()
        m = re.match(r"RESULT bg\w+ file=(\S+) n=(\d+) acc=([\d.]+)", l)
        if m and cell: r[(cell, m.group(1))] = (float(m.group(3)), int(m.group(2)))
    return r

if mode == "dev":
    out = "/content/comprehend_dev.txt"; open(out, "w").write("")
    DEV = "%smc160.train.txt %smc160.dev.txt %smc500.train.txt %smc500.dev.txt %sdev_middle.txt %s/clock.txt %s/dev_stories.txt" % (M, M, M, M, R, ST, ST)
    CELLS = [("R0", "NW=0 RD=0")] + [("R1 NW %g" % w, "NW=%g RD=0" % w) for w in (0.5, 1, 2)] + [("R2 NW %g" % w, "NW=%g RD=1" % w) for w in (0.5, 1, 2)]
    pair([(n, "SHOW=1 %s ./bg passages %s %s" % (e, BRAIN, DEV)) for n, e in CELLS], out)
    a = acc(out)
    def mean(c):   # MC160 (train + dev), MC500 (train + dev), RACE-M dev: the mean of the three
        def m(fs): n = sum(a[(c, f)][1] for f in fs); return sum(a[(c, f)][0] * a[(c, f)][1] for f in fs) / n
        return (m(["mc160.train.txt", "mc160.dev.txt"]) + m(["mc500.train.txt", "mc500.dev.txt"]) + a[(c, "dev_middle.txt")][0]) / 3
    def right(c): return sum(a[(c, f)][0] * a[(c, f)][1] / 100 for f in ("clock.txt", "dev_stories.txt"))
    with open(out, "a") as f:
        for n, e in CELLS: f.write("=== %s: passages mean %.2f, dev stories %.1f / 40\n" % (n, mean(n), right(n)))
    def best(p): return max([n for n, e in CELLS if n.startswith(p)], key=lambda n: (mean(n), right(n)))
    kept = "R0"
    for p in ("R1", "R2"):   # a step is kept if > 1 point on the passage mean, or >= 2 more dev-story questions and not > 0.5 lower
        b = best(p)
        if mean(b) > mean(kept) + 1 or (right(b) >= right(kept) + 2 and mean(b) >= mean(kept) - 0.5): kept = b
    env = dict(CELLS)[kept]
    open(out, "a").write("=== chosen: %s (%s)\n" % (kept, env))
    G = " ".join("data/dev_sets/%s.tsv" % t for t in ("easy_dev", "challenge_dev", "obqa_dev", "mmlu_hs_dev", "csqa_traindev"))
    open(out, "a").write("=== general dev sets, no episode (R0 was 47.81 20.01 25.25 29.32 31.55; RD changes how questions are read)\n")
    pair([("general %s" % n, "%s ./bg arc %s %s" % (e, BRAIN, G)) for n, e in {"R0": "NW=0 RD=0", kept: env}.items()], out)
    files.download(out)
else:
    out = "/content/comprehend_test.txt"; open(out, "w").write("")
    choice = open("tools/comprehend_choice.env").read().strip(); open(out, "a").write("=== choice: %s\n" % choice)
    NB = BRAIN    # with RD the shelf is read by the new reader too: the brain rebuilt by GitHub Actions (release brain-school-cx2)
    if "RD=1" in choice:
        NB = "/content/cx2.brain"; brain_release.URL = brain_release.URL.replace("/brain-school-cx/", "/brain-school-cx2/")
        if not brain_release.get(NB): raise SystemExit("no brain-school-cx2 release")
    T = "%smc160.test.txt %smc500.test.txt %stest_middle.txt %s/comprehend_tests.txt %s/episode_tests.txt %s/story_tests.txt" % (M, M, R, ST, ST, ST)
    pair([("R0", "SHOW=1 NW=0 RD=0 ./bg passages %s %s" % (BRAIN, T)), ("chosen", "SHOW=1 %s ./bg passages %s %s" % (choice, NB, T))], out)
    G = " ".join("data/arc_tests/%s.tsv" % t for t in ("mmlu_hs_test", "csqa_dev", "easy_test", "challenge_test", "obqa_test"))
    open(out, "a").write("=== general tests, no episode (R0: 27.41 31.37 46.61 21.22 24.40)\n")
    pair([("general chosen", "%s ./bg arc %s %s" % (choice, NB, G))], out)
    files.download(out)
print("comprehend", mode, "done", flush=True)
