#!/usr/bin/env python3
"""OpenBookQA questions, and SciQ train as practice questions without their passage, in bg.c's question layout
(id, question, n options, options..., answer index). Run from the repo root after tools/shelf_fetch.py and sciq_fetch.py.

  data/shelf/arc/obqa_{train,dev,test}.tsv
  data/shelf/arc/sciq_train.tsv
"""
import json, os, subprocess, zipfile

Z = "data/shelf/raw/obqa.zip"
if not os.path.exists(Z): subprocess.run(["curl", "-sL", "-o", Z, "https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip"], check=True)
clean = lambda s: " ".join(s.replace("\t", " ").split())
with zipfile.ZipFile(Z) as z:
    for split in ("train", "dev", "test"):
        out = "data/shelf/arc/obqa_%s.tsv" % split; n = 0
        with z.open("OpenBookQA-V1-Sep2018/Data/Main/%s.jsonl" % split) as f, open(out, "w") as o:
            for line in f:
                r = json.loads(line); q = r["question"]; labels = [c["label"] for c in q["choices"]]
                o.write("\t".join([r["id"], clean(q["stem"]), str(len(labels))] + [clean(c["text"]) for c in q["choices"]] + [str(labels.index(r["answerKey"]))]) + "\n"); n += 1
        print(out, n)
n = 0
with open("data/understand/sciq/train.txt") as f, open("data/shelf/arc/sciq_train.tsv", "w") as o:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if p[0] != "Q" or len(p) < 8: continue
        o.write("\t".join(["sciq%d" % n, p[2], "4"] + p[3:7] + [str("ABCD".index(p[7]))]) + "\n"); n += 1
print("data/shelf/arc/sciq_train.tsv", n)
