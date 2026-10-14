# docs/CORTEX_LINK.md in Colab: one brain, the cortex learning from the facts as Mimir reads.
#   %run /content/mimir/tools/link_colab.py dev     learning cells L1/L2 (dev brains, 1/8 of the shelf), the whole shelf read
#                                                    with the chosen one, answering cells M0/C1a/C1b/C2 on the dev sets, clock story
#   %run /content/mimir/tools/link_colab.py test    after the choice is frozen in tools/link_choice.env: the tests, once
# Results are printed and sent to the browser's computer (link_dev.txt / link_test.txt).
import glob, os, subprocess, sys
from google.colab import files
def sh(cmd, log=None):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1); out = []
    for line in p.stdout:
        print(line, end="", flush=True); out.append(line)
        if log and not line.startswith("  "): open(log, "a").write(line)
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
if not os.path.exists("data/school/07_vital.txt"): sh("python3 tools/school_shelf.py && rm -rf data/school/raw")
SHELF = sorted(glob.glob("data/school/0*.txt")); sh("ls -l data/school/")
W = "data/cortex/wordsim/"; BRAIN = "/content/cx.brain"
DEV = " ".join("data/dev_sets/%s.tsv" % t for t in ("easy_dev", "challenge_dev", "obqa_dev", "mmlu_hs_dev", "csqa_traindev"))
TEST = " ".join("data/arc_tests/%s.tsv" % t for t in ("mmlu_hs_test", "csqa_dev", "easy_test", "challenge_test", "obqa_test"))
M0 = "CX=0"
def words(brain, which, log): return {l.split()[2].split("=")[1]: float(l.split("spearman=")[1]) for l in sh("./bg words %s %s" % (brain, " ".join(W + f for f in which)), log).splitlines() if l.startswith("RESULT words")}
def accs(text): return {l.split("file=")[1].split()[0]: float(l.split("acc=")[1].split()[0]) for l in text.splitlines() if l.startswith("RESULT bgarc")}

def pick_fit(dump):
    """C2 from a C1 dump: the fit weight from 0.25, 0.5, 1, 2 by the dev mean (scored as bg arc scores, ties share)"""
    rows = [l.rstrip("\n").split("\t") for l in open(dump)]
    def acc(w):
        per = {}
        for r in rows:
            name, want, no, neg = r[0], int(r[1]), int(r[2]), int(r[3])
            sc = []
            for o in range(no):
                f, j, op, me, fit = map(float, r[4 + o].split()); s = f + 0.5 * j + 2 * op + 4 * me + w * fit; sc.append(-s if neg else s)
            b = max(sc); nb = sum(1 for x in sc if x > b - 1e-9); ok = 1 / nb if sc[want] > b - 1e-9 else 0
            per.setdefault(name, []).append(ok)
        return {k: 100 * sum(v) / len(v) for k, v in per.items()}
    return {w: acc(w) for w in (0, 0.25, 0.5, 1, 2)}

def story(name, qfile, env, before, after, out):
    """ask a story's questions with brain `before` (not read it) and `after` (read it); write each answer"""
    Q = [l.rstrip("\n").split("\t") for l in open(qfile) if l.startswith(name)]
    inp = "".join("\t".join([q[1]] + q[3:3 + int(q[2])]) + "\n" for q in Q); res = {}
    import json
    for tag, b in (("before", before), ("after", after)):
        r = subprocess.run("%s ./bg serve %s" % (env, b), shell=True, input=inp, capture_output=True, text=True)
        ans = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")][1:]; right = 0
        for q, a in zip(Q, ans):
            opts = q[3:3 + int(q[2])]; ok = a["pick"] == int(q[-1]); right += ok
            open(out, "a").write("  %s %s\t%s\tpick=%s\tright=%s\t%s\n" % (tag, q[0], q[1], opts[a["pick"]], opts[int(q[-1])], "OK" if ok else "WRONG"))
        res[tag] = right
    open(out, "a").write("STORY %s env=%s before %d/%d after %d/%d\n" % (name, env, res["before"], len(Q), res["after"], len(Q)))
    return res

if mode == "dev":
    out = "/content/link_dev.txt"; open(out, "w").write("")
    # learning cells: dev brains from 1/8 of the shelf, one read each
    L = {}
    for cell in (1, 2):
        b = "/content/L%d.brain" % cell
        if not os.path.exists(b + ".cx"): sh("CX=1 CXSCHED=%d PART=8 ./bg read %s 1000000000 %s" % (cell, b, " ".join(SHELF)), out)
        r = words(b, ("EN-WS-353-ALL.txt", "EN-SimVerb-3500.txt"), out); L[cell] = (r["EN-WS-353-ALL.txt"] + r["EN-SimVerb-3500.txt"]) / 2
        open(out, "a").write("=== L%d mean %.4f\n" % (cell, L[cell])); sh("./bg near %s owl king france blue water dog" % b, out)
        os.remove(b); os.remove(b + ".cx")
    sched = max(L, key=lambda c: (L[c], -c)); open(out, "a").write("=== chosen L%d\n" % sched)
    # the whole shelf, file by file as the released school brain was read, the cortex learning all the way
    total = sum(os.path.getsize(f) for f in SHELF); done = 0
    for f in SHELF:
        if not (os.path.exists(BRAIN) and f in subprocess.run("strings -n 8 %s | head -40" % BRAIN, shell=True, capture_output=True, text=True).stdout):
            sh("CX=1 CXSCHED=%d CXPOS='%d %d' %s ./bg read %s 1000000000 %s" % (sched, done, total, "FROM=" + BRAIN if os.path.exists(BRAIN) else "", BRAIN, f), out)
        done += os.path.getsize(f)
    words(BRAIN, ("EN-WS-353-ALL.txt", "EN-SimVerb-3500.txt"), out); sh("./bg near %s owl king france blue water dog" % BRAIN, out)
    # answering cells on the dev sets
    cells = (("M0", M0), ("C1a", "CX=1 CXTH=0.25"), ("C1b", "CX=1 CXTH=0.40")); A = {}
    for name, env in cells:
        open(out, "a").write("=== %s %s\n" % (name, env))
        A[name] = accs(sh("%s DUMP=/content/dump_%s.txt ./bg arc %s %s" % (env, name, BRAIN, DEV), out))
        open(out, "a").write("=== %s mean %.2f\n" % (name, sum(A[name].values()) / len(A[name])))
    c1 = max(("C1a", "C1b"), key=lambda c: sum(A[c].values())); grid = pick_fit("/content/dump_%s.txt" % c1)
    for w, r in grid.items(): open(out, "a").write("=== C2 from %s fit %.2f: %s mean %.2f\n" % (c1, w, " ".join("%s=%.2f" % kv for kv in sorted(r.items())), sum(r.values()) / len(r)))
    # the clock story (dev, already seen: shown, not counted)
    if not os.path.exists("/content/clock.brain"): sh("CX=1 FROM=%s ./bg read /content/clock.brain 1000000000 data/story/clock.txt" % BRAIN, out)
    for env in (M0, "CX=1 CXTH=%s" % ("0.25" if c1 == "C1a" else "0.40")): story("clock", "data/story/clock_questions.tsv", env, BRAIN, "/content/clock.brain", out)
    files.download(out)
else:
    out = "/content/link_test.txt"; open(out, "w").write("")
    choice = open("tools/link_choice.env").read().strip(); open(out, "a").write("=== choice: %s\n" % choice)
    words(BRAIN, ("EN-SIMLEX-999.txt", "EN-MEN-TR-3k.txt"), out)
    if choice != M0:
        sh("%s ./bg arc %s %s" % (choice, BRAIN, TEST), out)
        for s in ("harbor", "garden", "snow"):
            b = "/content/%s.brain" % s
            if not os.path.exists(b): sh("CX=1 FROM=%s ./bg read %s 1000000000 data/story/%s.txt" % (BRAIN, b, s), out)
            for env in (M0, choice): story(s, "data/story/story_tests.tsv", env, BRAIN, b, out)
    files.download(out)
print("link", mode, "done", flush=True)
