#!/usr/bin/env python3
"""More shelf: English Wikipedia (wikimedia/wikipedia, 20231101.en, parquet) as plain text, same layout as
tools/shelf_fetch.py: one paragraph a line, a blank line between articles.

  python tools/enwiki_fetch.py FIRST LAST     files FIRST..LAST of 41 -> data/shelf/enwiki_NN.txt  (needs pyarrow)
Run from the repo root.
"""
import os, subprocess, sys
import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-%05d-of-00041.parquet"
first, last = int(sys.argv[1]), int(sys.argv[2])
os.makedirs("data/shelf/raw", exist_ok=True)
for i in range(first, last + 1):
    out = "data/shelf/enwiki_%02d.txt" % i
    if os.path.exists(out): continue
    raw = "data/shelf/raw/enwiki_%02d.parquet" % i
    subprocess.run(["curl", "-sL", "-o", raw, BASE % i], check=True)
    n = nb = 0
    with open(out + ".part", "w") as o:
        f = pq.ParquetFile(raw)
        for g in range(f.num_row_groups):
            for text in f.read_row_group(g, columns=["text"]).column("text").to_pylist():
                paras = [" ".join(p.split()) for p in text.split("\n")]
                paras = [p for p in paras if len(p.split()) >= 5 and p[-1] in ".!?\"')"]
                if paras: s = "\n".join(paras) + "\n\n"; o.write(s); n += 1; nb += len(s)
    os.rename(out + ".part", out); os.remove(raw)
    print(out, n, "articles", nb, "B", flush=True)
