# docs/SCHOOL_BRAIN.md in Colab: build the school shelf, read it into the school brain one shelf file at a time (no room
# limit), and after every file send the brain to the computer the browser runs on (files.download), so nothing is lost
# if Colab closes. Run in a notebook cell:  %run school_colab.py  (it needs the notebook's browser connection).
#
# Continuing after a disconnect: upload the last checkpoint's parts (school.brain.gz.part_*) into /content/upload first;
# the brain lists the shelf files it has read, and those are skipped.
import glob, hashlib, os, subprocess, sys, time
from google.colab import files

R = "https://github.com/callmedraxx/mimir/releases/download/brain-E"
def sh(cmd):
    """run a command and show its output in the notebook as it comes"""
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end="", flush=True)
    if p.wait(): raise SystemExit("failed: " + cmd)

os.chdir("/content")
if not os.path.isdir("mimir"): sh("git clone -q --depth 1 -b reader https://github.com/callmedraxx/mimir")
os.chdir("/content/mimir"); sh("git pull -q && git rev-parse --short HEAD && nproc && free -g | head -2")
os.makedirs("data/understand", exist_ok=True)
if not os.path.exists("data/understand/wordclass.tsv"): sh("curl -sL %s/wordclass.tsv.gz | gunzip > data/understand/wordclass.tsv" % R)
sh("gcc -O2 -std=gnu11 -D_GNU_SOURCE -w -o bg sandbox/neurons/bg.c src/meaning.c src/reader.c -Isrc -lm -fopenmp -lpthread")
sh("python3 tools/school_shelf.py")

BRAIN, OUT = "/content/school.brain", "/content/out"
up = sorted(glob.glob("/content/upload/school.brain.gz.part_*"))
if up and not os.path.exists(BRAIN): sh("cat %s | gunzip > %s" % (" ".join(up), BRAIN)); print("continuing from the uploaded checkpoint", flush=True)

def send(tag):
    """the brain, gzipped and cut in parts, sent through the browser; a manifest with sizes and SHA-256 last"""
    sh("rm -rf %s && mkdir -p %s && gzip -1 -c %s | split -b 400M -d - %s/school.brain.gz.part_" % (OUT, OUT, BRAIN, OUT))
    parts = sorted(glob.glob(OUT + "/school.brain.gz.part_*")); man = OUT + "/school.brain.manifest.txt"
    with open(man, "w") as m:
        m.write("checkpoint after %s\n" % tag)
        for p in parts:
            h = hashlib.sha256(open(p, "rb").read()).hexdigest(); m.write("%s %d %s\n" % (os.path.basename(p), os.path.getsize(p), h))
    for p in parts + [man]:
        print("sending", os.path.basename(p), os.path.getsize(p) // 1000000, "MB", flush=True); files.download(p); time.sleep(3)

shelf = sorted(glob.glob("data/school/0*.txt"))
for f in shelf:
    env = "ROOM=0 STRENGTH=1"
    if os.path.exists(BRAIN):
        env += " FROM=" + BRAIN
        r = subprocess.run("strings -n 8 %s | head -40" % BRAIN, shell=True, capture_output=True, text=True).stdout
        if f in r: print(f, "read already", flush=True); continue
    t0 = time.time(); sh("%s ./bg read %s 1000000000 %s" % (env, BRAIN, f))
    print("%s read in %.0f s" % (f, time.time() - t0), flush=True)
    send(os.path.basename(f))
print("school brain done", flush=True)
