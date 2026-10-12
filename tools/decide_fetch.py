#!/usr/bin/env python3
"""Fetch the Typed Decisions benchmark (LocalLLaMA/typed-decisions, Apache-2.0) as JSON lines.

    python3 tools/decide_fetch.py        (from the repo root; stdlib only)

Writes data/decide/{train,test}.jsonl (not checked in): one case per line with
id, workflow, state, questions, gold (all JSON objects, as published).
"""
import json, os, time, urllib.request

ROWS = "https://datasets-server.huggingface.co/rows?dataset=LocalLLaMA/typed-decisions&config=all&split={}&offset={}&length=100"

def get(url):
    for t in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r: return json.loads(r.read())
        except Exception:
            if t == 5: raise
            time.sleep(2 + 3 * t)

os.makedirs("data/decide", exist_ok=True)
for split in ("train", "test"):
    rows, off = [], 0
    while True:
        d = get(ROWS.format(split, off)); rows += [r["row"] for r in d["rows"]]; off += 100
        if off >= d["num_rows_total"]: break
    with open(f"data/decide/{split}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "workflow": r["workflow"], "state": json.loads(r["state"]),
                                "questions": json.loads(r["questions"]), "gold": json.loads(r["gold"])}) + "\n")
    print(split, len(rows))
