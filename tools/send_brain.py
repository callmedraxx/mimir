# Send a brain (and its cortex, BRAIN.cx) from Colab to the browser's computer: gzipped, cut in 400 MB parts,
# a manifest with sizes and SHA-256 last.   %run /content/mimir/tools/send_brain.py /content/cx.brain NAME
import glob, hashlib, os, subprocess, sys, time
from google.colab import files
brain, name = sys.argv[1], sys.argv[2]; out = "/content/send_" + name
subprocess.run("rm -rf %s && mkdir -p %s" % (out, out), shell=True, check=True)
for src, tag in ((brain, name + ".brain"), (brain + ".cx", name + ".brain.cx")):
    subprocess.run("gzip -1 -c %s | split -b 400M -d - %s/%s.gz.part_" % (src, out, tag), shell=True, check=True)
parts = sorted(glob.glob(out + "/*.part_*")); man = out + "/%s.manifest.txt" % name
with open(man, "w") as m:
    for p in parts: m.write("%s %d %s\n" % (os.path.basename(p), os.path.getsize(p), hashlib.sha256(open(p, "rb").read()).hexdigest()))
for p in parts + [man]:
    print("sending", os.path.basename(p), os.path.getsize(p) // 1000000, "MB", flush=True); files.download(p); time.sleep(5)
print("sent", len(parts), "parts", flush=True)
