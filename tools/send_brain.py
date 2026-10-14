# Send a brain (and its cortex, BRAIN.cx) from Colab to the browser's computer: gzipped, cut in 100 MB pieces (bigger
# ones crash the shared browser on the server), one every 40 s, a manifest with sizes and SHA-256 first.
#   %run /content/mimir/tools/send_brain.py /content/cx.brain NAME
import glob, hashlib, os, subprocess, sys, time
from google.colab import files
brain, name = sys.argv[1], sys.argv[2]; out = "/content/send_" + name
os.makedirs(out, exist_ok=True)
for src, tag in ((brain, name + ".brain"), (brain + ".cx", name + ".brain.cx")):
    if not glob.glob("%s/%s.gz.p[0-9]*" % (out, tag)):
        if glob.glob("%s/%s.gz.part_*" % (out, tag)): cmd = "cat %s/%s.gz.part_* | split -b 100M -d -a 3 - %s/%s.gz.p" % (out, tag, out, tag)   # gzipped before
        else: cmd = "gzip -1 -c %s | split -b 100M -d -a 3 - %s/%s.gz.p" % (src, out, tag)
        subprocess.run(cmd, shell=True, check=True)
parts = sorted(glob.glob(out + "/*.gz.p[0-9]*")); man = out + "/%s.pieces.txt" % name
with open(man, "w") as m:
    for p in parts: m.write("%s %d %s\n" % (os.path.basename(p), os.path.getsize(p), hashlib.sha256(open(p, "rb").read()).hexdigest()))
files.download(man); time.sleep(10)
for p in parts:
    print("sending", os.path.basename(p), os.path.getsize(p) // 1000000, "MB", flush=True); files.download(p); time.sleep(40)
print("sent", len(parts), "pieces", flush=True)
