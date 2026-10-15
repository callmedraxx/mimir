# docs/READ2.md in Colab.
#   %run /content/mimir/tools/read2_colab.py measure   step 0: brain-school-cx2 with its shelf; the general dev sets, then
#                                                      the five general tests once (COMPREHEND's measurement, rerun)
#   %run /content/mimir/tools/read2_colab.py dev       S2: S0 and S2 (S2=7: pronouns by sex, apposition, lists) on the dev
#                                                      passages, the 50 dev-story questions and the general dev sets
#   %run /content/mimir/tools/read2_colab.py test      after the freeze (S2=31, release brain-school-cx3 = the shelf read by
#                                                      the S2 reader): the tests, once; S0 on brain-school-cx2 for the new stories
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
if not os.path.exists("data/understand/mctest/mc500.test.txt"): sh("python3 -c \"import sys; sys.path.insert(0, 'tools'); import understand_fetch as u; u.mctest()\"")
if not os.path.exists("data/understand/race/test_middle.txt"): sh("python3 tools/race_fetch.py")
sys.path.insert(0, "tools"); import brain_release, stories
BRAIN = "/content/cx2.brain"
if not brain_release.get(BRAIN): raise SystemExit("no brain release")
sh("ls -l data/school/")
ST = "/content/stories"; os.makedirs(ST, exist_ok=True)
for n in stories.SETS: stories.write(ST, n)
M = "data/understand/mctest/"; R = "data/understand/race/"
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
def acc(out):
    """{(cell, file): (accuracy, n)} from the RESULT lines under each === cell"""
    r, cell = {}, None
    for l in open(out):
        if l.startswith("=== "): cell = l[4:].strip()
        m = re.match(r"RESULT bg\w+ file=(\S+) n=(\d+) acc=([\d.]+)", l)
        if m and cell: r[(cell, m.group(1))] = (float(m.group(3)), int(m.group(2)))
    return r
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
if mode == "dev":
    out = "/content/read2_dev.txt"; open(out, "w").write("")
    DEV = "%smc160.train.txt %smc160.dev.txt %smc500.train.txt %smc500.dev.txt %sdev_middle.txt %s/clock.txt %s/dev_stories.txt %s/bakery.txt" % (M, M, M, M, R, ST, ST, ST)
    # S2 reader (pronouns by sex, apposition, lists) = 7; + the main character 8; + yes/no questions judged whole 16
    CELLS = [("S0", "S2=0"), ("S2 7", "S2=7"), ("S2 15", "S2=15"), ("S2 31", "S2=31")]
    pair([(n, "SHOW=1 %s ./bg passages %s %s" % (e, BRAIN, DEV)) for n, e in CELLS], out)
    a = acc(out)
    def mean(c):   # MC160 (train + dev), MC500 (train + dev), RACE-M dev: the mean of the three
        def m(fs): n = sum(a[(c, f)][1] for f in fs); return sum(a[(c, f)][0] * a[(c, f)][1] for f in fs) / n
        return (m(["mc160.train.txt", "mc160.dev.txt"]) + m(["mc500.train.txt", "mc500.dev.txt"]) + a[(c, "dev_middle.txt")][0]) / 3
    def right(c): return round(sum(a[(c, f)][0] * a[(c, f)][1] / 100 for f in ("clock.txt", "dev_stories.txt", "bakery.txt")))   # whole questions (93.33% of 30 is 28, not 27.999)
    with open(out, "a") as f:
        for n, e in CELLS: f.write("=== %s: passages mean %.2f, dev stories %.1f / 50\n" % (n, mean(n), right(n)))
    # docs/READ2.md: S2 over S0 if the passage dev mean is > 1 better, or >= 2 more dev-story questions right with the mean not > 0.5 lower
    best = max([n for n, e in CELLS if n != "S0"], key=lambda n: (mean(n), right(n)))
    kept = best if mean(best) > mean("S0") + 1 or (right(best) >= right("S0") + 2 and mean(best) >= mean("S0") - 0.5) else "S0"
    open(out, "a").write("=== chosen by the passage rule: %s (%s); the general dev sets must not fall by more than 1\n" % (kept, dict(CELLS)[kept]))
    CELLS = [("S0", "S2=0"), (kept, dict(CELLS)[kept])] if kept != "S0" else [("S0", "S2=0"), (best, dict(CELLS)[best])]
    G = " ".join("data/dev_sets/%s.tsv" % t for t in ("easy_dev", "challenge_dev", "obqa_dev", "mmlu_hs_dev", "csqa_traindev"))
    open(out, "a").write("=== general dev sets (S0 with this shelf: 47.63 17.25 23.05 28.13 29.20; S2 changes how questions are read)\n")
    pair([("general %s" % n, "%s ./bg arc %s %s" % (e, BRAIN, G)) for n, e in CELLS], out)
    files.download(out)
if mode == "test":
    out = "/content/read2_test.txt"; open(out, "w").write("=== choice: S2=31 (frozen; docs/READ2.md)\n")
    NB = "/content/cx3.brain"
    if not brain_release.get(NB, "brain-school-cx3"): raise SystemExit("no brain-school-cx3 release")
    T = "%smc160.test.txt %smc500.test.txt %stest_middle.txt %s/read2_tests.txt %s/comprehend_tests.txt %s/episode_tests.txt %s/story_tests.txt" % (M, M, R, ST, ST, ST, ST)
    pair([("S0 new stories (brain-school-cx2)", "SHOW=1 S2=0 ./bg passages %s %s/read2_tests.txt" % (BRAIN, ST)),
          ("S2 passages (brain-school-cx3)", "SHOW=1 S2=31 ./bg passages %s %s" % (NB, T))], out)
    GT = " ".join("data/arc_tests/%s.tsv" % t for t in ("mmlu_hs_test", "csqa_dev", "easy_test", "challenge_test", "obqa_test"))
    open(out, "a").write("=== general tests, once (S0 on brain-school-cx2 with its shelf: 26.56 31.08 46.09 21.98 24.20)\n")
    pair([("general S2 (brain-school-cx3)", "S2=31 ./bg arc %s %s" % (NB, GT))], out)
    files.download(out)
print("read2", mode, "done", flush=True)
