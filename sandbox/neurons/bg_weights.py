#!/usr/bin/env python3
"""Choose bg.c's answer weights on development dumps (docs/BACKGROUND_COMPARE.md); ties share credit.

  bg_weights.py arc DUMP     DUMP from `DUMP=f bg arc BRAIN dev files`: BW = (fact, joint, open, meaning)
  bg_weights.py know DUMP    DUMP from `DUMP=f bg know BRAIN dev files`: WBG (background weight on top of K open)
"""
import itertools, sys

def acc(rows, score):
    ok = 0.0
    for want, opts in rows:
        s = [score(o) for o in opts]; b = max(s); tied = [i for i, v in enumerate(s) if v > b - 1e-9]
        ok += (1.0 / len(tied)) if want in tied else 0.0
    return 100 * ok / len(rows)

mode, path = sys.argv[1], sys.argv[2]
rows = []
for line in open(path):
    f = line.rstrip("\n").split("\t")
    if mode == "arc": rows.append((int(f[1]), [tuple(map(float, x.split())) for x in f[3:3 + int(f[2])]]))
    else: rows.append((int(f[1]), [tuple(map(float, x.split())) for x in f[2:6]]))

if mode == "arc":
    G = [0, 0.5, 1, 2, 4]; res = []
    for w in itertools.product(G, repeat=4):
        if not any(w): continue
        res.append((acc(rows, lambda o: sum(a * b for a, b in zip(w, o))), w))
    res.sort(reverse=True)
    for a, w in res[:8]: print("%.2f%%  BW=%s" % (a, " ".join("%g" % x for x in w)))
    print("each alone:", " ".join("%.2f" % acc(rows, lambda o, i=i: o[i]) for i in range(4)), "| n", len(rows))
else:
    for w in [0, 0.1, 0.25, 0.5, 1, 2, 4]: print("WBG=%g  %.2f%%" % (w, acc(rows, lambda o: o[0] + w * o[1])))
