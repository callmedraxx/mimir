# docs/FOLLOW.md in Colab.
#   %run /content/mimir/tools/follow_colab.py dev    S3 = 0 (the adopted reader), 15 (F1-F4), 16 (F5 alone), 31 (all) on the dev
#                                                    passages and the 70 dev-story questions; then the general dev sets for S0 and
#                                                    the chosen cell (brain-school-cx3: the shelf read by the READ2 reader)
#   %run /content/mimir/tools/follow_colab.py test   after the freeze: the tests, once (the frozen S3 on the brain named below)
# The brain and its shelf: the GitHub release (tools/brain_release.py). Stops if open book is not working.
import os, re, subprocess, sys
from google.colab import files
def sh(cmd, out=None):
    print("$", cmd, flush=True); r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout, flush=True)
    if out: open(out, "a").write("$ %s\n%s" % (cmd, r.stdout))
    if r.returncode: raise SystemExit("failed: " + cmd)
    return r.stdout
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
BRAIN = "/content/cx3.brain"
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
STORIES = ("clock.txt", "dev_stories.txt", "bakery.txt", "smart_tests.txt")
G = " ".join("data/dev_sets/%s.tsv" % t for t in ("easy_dev", "challenge_dev", "obqa_dev", "mmlu_hs_dev", "csqa_traindev"))
if mode == "dev":
    out = "/content/follow_dev.txt"; open(out, "w").write("")
    DEV = "%smc160.train.txt %smc160.dev.txt %smc500.train.txt %smc500.dev.txt %sdev_middle.txt %s" % (M, M, M, M, R, " ".join(ST + "/" + s for s in STORIES))
    CELLS = [("S0", "S3=0"), ("S3 15", "S3=15"), ("S3 16", "S3=16"), ("S3 31", "S3=31")]
    pair([(n, "SHOW=1 %s ./bg passages %s %s" % (e, BRAIN, DEV)) for n, e in CELLS], out)
    a = acc(out)
    def mean(c):   # MC160 (train + dev), MC500 (train + dev), RACE-M dev: the mean of the three
        def m(fs): n = sum(a[(c, f)][1] for f in fs); return sum(a[(c, f)][0] * a[(c, f)][1] for f in fs) / n
        return (m(["mc160.train.txt", "mc160.dev.txt"]) + m(["mc500.train.txt", "mc500.dev.txt"]) + a[(c, "dev_middle.txt")][0]) / 3
    def right(c): return round(sum(a[(c, f)][0] * a[(c, f)][1] / 100 for f in STORIES))   # whole questions (ties count a fraction; rounded)
    with open(out, "a") as f:
        for n, e in CELLS: f.write("=== %s: passages mean %.2f, dev stories %d / 70\n" % (n, mean(n), right(n)))
    # docs/FOLLOW.md: kept if >= 4 more dev-story questions right with the passage mean not > 0.5 lower (then the general dev sets, not > 1 worse)
    best = max([n for n, e in CELLS if n != "S0"], key=lambda n: (right(n), mean(n)))
    kept = best if right(best) >= right("S0") + 4 and mean(best) >= mean("S0") - 0.5 else "S0"
    open(out, "a").write("=== by the story rule: %s (%s); the general dev sets must not fall by more than 1\n" % (kept, dict(CELLS)[kept]))
    CELLS = [("S0", "S3=0"), (kept, dict(CELLS)[kept])] if kept != "S0" else [("S0", "S3=0"), (best, dict(CELLS)[best])]
    open(out, "a").write("=== general dev sets (S0 on brain-school-cx3; S3 changes how questions are read)\n")
    pair([("general %s" % n, "%s ./bg arc %s %s" % (e, BRAIN, G)) for n, e in CELLS], out)
    files.download(out)
if mode == "test":
    FROZEN = os.environ.get("S3", "31"); TAG = os.environ.get("TAG", "brain-school-cx4")
    out = "/content/follow_test.txt"; open(out, "w").write("=== choice: S3=%s (frozen; docs/FOLLOW.md), brain %s\n" % (FROZEN, TAG))
    NB = "/content/cx4.brain"
    if not brain_release.get(NB, TAG): raise SystemExit("no %s release" % TAG)
    T = "%smc160.test.txt %smc500.test.txt %stest_middle.txt %s/follow_tests.txt %s/read2_tests.txt %s/comprehend_tests.txt %s/episode_tests.txt %s/story_tests.txt" % (M, M, R, ST, ST, ST, ST, ST)
    pair([("S0 new stories (brain-school-cx3)", "SHOW=1 S3=0 ./bg passages %s %s/follow_tests.txt" % (BRAIN, ST)),
          ("S3 passages (%s)" % TAG, "SHOW=1 S3=%s ./bg passages %s %s" % (FROZEN, NB, T))], out)
    GT = " ".join("data/arc_tests/%s.tsv" % t for t in ("mmlu_hs_test", "csqa_dev", "easy_test", "challenge_test", "obqa_test"))
    open(out, "a").write("=== general tests, once (READ2 on brain-school-cx3: 26.87 31.12 46.34 22.00 24.20)\n")
    pair([("general S3 (%s)" % TAG, "S3=%s ./bg arc %s %s" % (FROZEN, NB, GT))], out)
    files.download(out)
print("follow", mode, "done", flush=True)
