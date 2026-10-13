#!/usr/bin/env python3
"""The general-knowledge tests of docs/SCHOOL_BRAIN.md, in bg.c's question layout (id, question, n, options..., answer index):

  data/arc_tests/mmlu_hs_test.tsv   MMLU (Hendrycks et al. 2021, MIT), test split of every high_school_* subject
  data/arc_tests/csqa_dev.tsv       CommonsenseQA (Talmor et al. 2019, MIT), validation split (its test answers are hidden)
Needs pyarrow. Run from the repo root.
"""
import io, json, urllib.request
import pyarrow.parquet as pq
def get(u): return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "mimir"}), timeout=120).read()
clean = lambda s: " ".join(str(s).replace("\t", " ").split())

files = [s["rfilename"] for s in json.loads(get("https://huggingface.co/api/datasets/cais/mmlu"))["siblings"]]
subj = sorted(f for f in files if f.startswith("high_school_") and "/test-" in f and f.endswith(".parquet"))
n = 0
with open("data/arc_tests/mmlu_hs_test.tsv", "w") as o:
    for f in subj:
        t = pq.read_table(io.BytesIO(get("https://huggingface.co/datasets/cais/mmlu/resolve/main/" + f))).to_pylist()
        for i, r in enumerate(t):
            o.write("\t".join(["%s_%d" % (f.split("/")[0], i), clean(r["question"]), str(len(r["choices"]))] + [clean(c) for c in r["choices"]] + [str(r["answer"])]) + "\n"); n += 1
print("mmlu_hs_test:", n, "questions,", len(subj), "subjects:", ", ".join(f.split("/")[0][12:] for f in subj))

t = pq.read_table(io.BytesIO(get("https://huggingface.co/datasets/tau/commonsense_qa/resolve/main/data/validation-00000-of-00001.parquet"))).to_pylist()
with open("data/arc_tests/csqa_dev.tsv", "w") as o:
    for r in t:
        L = list(r["choices"]["label"]); o.write("\t".join([r["id"], clean(r["question"]), str(len(L))] + [clean(x) for x in r["choices"]["text"]] + [str(L.index(r["answerKey"]))]) + "\n")
print("csqa_dev:", len(t), "questions")
