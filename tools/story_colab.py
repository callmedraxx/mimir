# A small reading test (owner, 2026-10-14): the school brain answers 10 questions about a short story written for it
# (data/story/clock.txt, in nothing it has read), before and after reading it. Each answer with the sentence it came from.
#   %run /content/mimir/tools/story_colab.py        → story_test.txt is sent to the browser's computer
# The school shelf is not rebuilt here (35 min): what the brain read before is used as facts only, not re-read.
import json, os, subprocess
from google.colab import files
def sh(cmd):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end="", flush=True)
    if p.wait(): raise SystemExit("failed: " + cmd)
os.chdir("/content")
if not os.path.isdir("mimir"): sh("git clone -q --depth 1 -b reader https://github.com/callmedraxx/mimir")
os.chdir("/content/mimir"); sh("git pull -q && git log --oneline -1")
sh("gcc -O2 -std=gnu11 -D_GNU_SOURCE -w -o bg sandbox/neurons/bg.c src/meaning.c src/reader.c -Isrc -lm -fopenmp -lpthread")
os.makedirs("data/understand", exist_ok=True)
if not os.path.exists("data/understand/wordclass.tsv"):
    sh("curl -sL https://github.com/callmedraxx/mimir/releases/download/brain-E/wordclass.tsv.gz | gunzip > data/understand/wordclass.tsv")
BRAIN, AFTER = "/content/school.brain", "/content/story.brain"
if not os.path.exists(BRAIN):
    R = "https://github.com/callmedraxx/mimir/releases/download/brain-school"
    os.makedirs("/content/rel", exist_ok=True)
    sh("cd /content/rel && curl -sLO %s/SHA256SUMS && curl -sLO %s/school.brain.gz.part_00 && curl -sLO %s/school.brain.gz.part_01 && sha256sum -c SHA256SUMS" % (R, R, R))
    sh("cat /content/rel/school.brain.gz.part_* | gunzip > %s && rm -rf /content/rel" % BRAIN)
if not os.path.exists(AFTER): sh("ROOM=0 STRENGTH=1 FROM=%s ./bg read %s 1000000000 data/story/clock.txt" % (BRAIN, AFTER))
Q = [l.rstrip("\n").split("\t") for l in open("data/story/clock_questions.tsv")]
inp = "".join("\t".join([q[1]] + q[3:3 + int(q[2])]) + "\n" for q in Q)
out = open("/content/story_test.txt", "w")
for name, b in (("before reading", BRAIN), ("after reading", AFTER)):
    r = subprocess.run("NOPEN=20 POLARITY=1 STRENGTH=1 ./bg serve %s" % b, shell=True, input=inp, capture_output=True, text=True)
    lines = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")][1:]
    right = 0
    out.write("=== %s (%s)\n" % (name, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""))
    for q, a in zip(Q, lines):
        opts = q[3:3 + int(q[2])]; ok = a["pick"] == int(q[-1]); right += ok
        o = a["options"][a["pick"]]
        out.write("%s\t%s\tpick=%s\tright=%s\t%s\tscores=%s\tfrom=%s | %s\n" % (q[0], q[1], opts[a["pick"]], opts[int(q[-1])], "OK" if ok else "WRONG",
                  ",".join("%.3f" % x["score"] for x in a["options"]), o["book"], o["from"]))
    out.write("RESULT %s %d/10\n" % (name, right))
out.close(); print(open("/content/story_test.txt").read(), flush=True)
files.download("/content/story_test.txt")
print("story test done", flush=True)
