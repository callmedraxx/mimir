#!/usr/bin/env python3
"""Fetch RACE (Lai et al. 2017, CMU; non-commercial research use) as MCTest-format files:
data/understand/race/{train,dev,test}_{middle,high}.txt   (stdlib only; run from the repo root)
English reading exams for Chinese middle (ages 12-15) and high school students: a passage and
four-option questions, some written as a sentence with a blank ("_").
"""
import io, json, os, re, tarfile, urllib.request
URL = "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz"
import sys
src = open(sys.argv[1], "rb").read() if len(sys.argv) > 1 else urllib.request.urlopen(URL, timeout=600).read()   # optional: a local RACE.tar.gz
tf = tarfile.open(fileobj=io.BytesIO(src), mode="r|gz")   # read straight through
os.makedirs("data/understand/race", exist_ok=True)
clean = lambda s: re.sub(r"\s+", " ", s.replace("\t", " ")).strip()
files, counts = {}, {}
for m in tf:
    g = re.match(r"RACE/(train|dev|test)/(middle|high)/(\d+)\.txt$", m.name)
    if not g: continue
    key = "%s_%s" % (g[1], g[2]); d = json.loads(tf.extractfile(m).read())
    f = files.get(key) or files.setdefault(key, open("data/understand/race/%s.txt" % key, "w"))
    f.write("S\trace.%s.%s\nT\t%s\n" % (key, g[3], clean(d["article"])))
    for q, o, a in zip(d["questions"], d["options"], d["answers"]):
        if len(o) != 4: continue
        f.write("Q\tsingle\t%s\t%s\t%s\n" % (clean(q) or "_", "\t".join(clean(x) or "-" for x in o), a)); counts[key] = counts.get(key, 0) + 1
for f in files.values(): f.close()
print(counts)
