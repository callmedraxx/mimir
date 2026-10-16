#!/usr/bin/env python3
"""docs/SURVEY.md: a compact reading of /content/survey.txt (the Colab runner's file: "=== test" headers, SHOW=1 picks,
RESULT lines) for the notes per test: accuracy, what it picks (yes/no, "none of the above"), bAbI by task, and the first
misses of each test.  python3 tools/survey_notes.py /content/survey.txt [misses per test, default 8]"""
import re, sys
from collections import Counter, defaultdict
src = sys.argv[1] if len(sys.argv) > 1 else "/content/survey.txt"; NM = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tests, cur = {}, None
for l in open(src, errors="replace"):
    l = l.rstrip("\n")
    if l.startswith("=== "): cur = l[4:].strip(); tests.setdefault(cur, {"picks": [], "result": None}); continue
    if cur is None: continue
    if l.startswith("RESULT"): tests[cur]["result"] = l; continue
    if l.startswith("  ") and "\tpick=" in l:
        f = l[2:].split("\t")
        if len(f) >= 5: tests[cur]["picks"].append((f[0], f[1], f[2][5:], f[3][6:], f[4] == "OK"))
CH = {"wsc273": 50, "winogrande": 50, "storycloze": 50, "boolq": 50, "dream": 33.3, "cosmosqa": 25, "race_high": 25, "quality": 25, "sciq": 25, "lambada": 10, "cbt_ne": 10, "cbt_cn": 10, "cbt_v": 10, "cbt_p": 10}
print("test            n     acc   chance  notes")
for t, d in tests.items():
    p = d["picks"]; n = len(p); acc = 100.0 * sum(ok for *_, ok in p) / n if n else 0
    note = []
    if t == "boolq": note.append("picks yes %.0f%%, right is yes %.0f%%" % (100.0 * sum(x[2] == "yes" for x in p) / n, 100.0 * sum(x[3] == "yes" for x in p) / n))
    if t == "cosmosqa": note.append("picks 'none of the above' %.0f%% (right %.0f%%)" % (100.0 * sum("none of the above" in x[2].lower() for x in p) / n, 100.0 * sum("none of the above" in x[3].lower() for x in p) / n))
    if t == "multirc": note.append("options per question vary")
    if t.startswith("babi"):
        bt = defaultdict(list)
        for x in p: bt[int(re.match(r"babi(\d+)_", x[0]).group(1))].append(x[4])
        note.append("by task: " + " ".join("%d:%.0f" % (k, 100.0 * sum(v) / len(v)) for k, v in sorted(bt.items())))
    if t.startswith("babi"): pass
    elif t in ("cbt_ne", "cbt_cn", "cbt_v", "cbt_p", "lambada", "record", "multirc"): note.append("right answer shorter than pick %.0f%%" % (100.0 * sum(len(x[3]) < len(x[2]) for x in p if not x[4]) / max(1, sum(not x[4] for x in p))))
    print("%-14s %5d  %6.2f  %5s  %s" % (t, n, acc, CH.get(t, "-"), "; ".join(x for x in note if x)))
print()
for t, d in tests.items():
    miss = [x for x in d["picks"] if not x[4]][:NM]
    print("--- %s misses" % t)
    for qid, q, pick, right, _ in miss: print("  %s | %s | pick=%s | right=%s" % (qid, q[:110], pick[:60], right[:60]))
