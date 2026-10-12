#!/usr/bin/env python3
"""Fetch SciQ (AllenAI, CC BY-NC 3.0) as MCTest-format files: data/understand/sciq/{train,dev,test}.txt

    python3 tools/sciq_fetch.py        (from the repo root; stdlib only)

Only questions with a supporting passage are kept. Each becomes a story (the passage) with one
four-option question; the right answer's place among A-D is fixed by a seeded shuffle.
"""
import io, json, os, random, re, urllib.request, zipfile
ZIP = "https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip"     # AllenAI's own release
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(ZIP, timeout=300).read()))
def split_rows(name):
    for n in z.namelist():
        if n.endswith("/%s.json" % name): return json.loads(z.read(n))
    raise SystemExit("no %s.json in the zip" % name)
os.makedirs("data/understand/sciq", exist_ok=True)
clean = lambda s: re.sub(r"\s+", " ", s.replace("\t", " ")).strip()
for split, name in (("train", "train"), ("validation", "dev"), ("test", "test")):
    rows = split_rows("valid" if split == "validation" else split)
    rng = random.Random(1234 + len(name)); n = 0
    with open(f"data/understand/sciq/{name}.txt", "w") as f:
        for i, r in enumerate(rows):
            if not r["support"].strip(): continue
            opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]; order = list(range(4)); rng.shuffle(order)
            f.write("S\tsciq.%s.%d\nT\t%s\n" % (name, i, clean(r["support"])))
            f.write("Q\tsingle\t%s\t%s\t%s\n" % (clean(r["question"]), "\t".join(clean(opts[k]) for k in order), "ABCD"[order.index(0)]))
            n += 1
    print(name, n, "of", len(rows), "with a passage")
