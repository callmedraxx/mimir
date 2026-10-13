#!/usr/bin/env python3
"""Server side of tools/school_colab.py: checkpoints arrive in the shared browser's download folder; when a manifest
has arrived and every part matches its size and SHA-256, the checkpoint is moved to data/school_brain/ (the newest one
only; the one before is removed after the new one is complete). Runs until stopped, or until DONE_TAG arrives.

  python3 tools/school_receive.py [DOWNLOADS] [DONE_TAG]
"""
import glob, hashlib, os, re, shutil, sys, time
DL = sys.argv[1] if len(sys.argv) > 1 else "/root/Downloads"
DONE = sys.argv[2] if len(sys.argv) > 2 else "07_vital.txt"
DEST = "data/school_brain"; os.makedirs(DEST, exist_ok=True)

def base(n): return re.sub(r" \(\d+\)(?=\.|$)", "", n)          # the browser adds " (1)" to a name it has seen before
while True:
    for man in sorted(glob.glob(os.path.join(DL, "school.brain.manifest*.txt")), key=os.path.getmtime):
        lines = open(man).read().split("\n"); tag = lines[0]; want = [l.split() for l in lines[1:] if l.strip()]
        have = {base(os.path.basename(p)): p for p in glob.glob(os.path.join(DL, "school.brain.gz.part_*")) if not p.endswith(".crdownload")}
        if not all(n in have for n, _, _ in want): continue
        ok = all(os.path.getsize(have[n]) == int(sz) and hashlib.sha256(open(have[n], "rb").read()).hexdigest() == h for n, sz, h in want)
        if not ok: print(time.strftime("%H:%M:%S"), "parts do not match the manifest yet:", tag, flush=True); continue
        new = os.path.join(DEST, "new"); shutil.rmtree(new, ignore_errors=True); os.makedirs(new)
        for n, _, _ in want: shutil.move(have[n], os.path.join(new, n))
        shutil.move(man, os.path.join(new, "school.brain.manifest.txt"))
        cur = os.path.join(DEST, "latest"); shutil.rmtree(cur, ignore_errors=True); os.rename(new, cur)
        print(time.strftime("%H:%M:%S"), "checkpoint kept:", tag, sum(int(s) for _, s, _ in want) // 1000000, "MB", flush=True)
        if DONE in tag: print("school brain complete", flush=True); sys.exit(0)
    time.sleep(20)
