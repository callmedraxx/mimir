#!/usr/bin/env python3
"""LAYA_LEVEL.md stage 3: answer neurons that learn how much to trust each input, from training questions.

  laya_learn.py TRAIN_DUMP DEV_DUMP [OUT_WEIGHTS]

Dumps from `DUMP=f bg arc ...` (one line a question: file, gold, n, then per option fact joint open meaning chain
open2 open3 spec olen). Inputs per option (11): those nine, plus open and joint minus the mean of the other options.
A question that asks for the opposite (stage 2) multiplies the inputs by -1.
One weight per input, shared by all options; activity = weighted sum; options compete (softmax).
Learning: dw = eta * input * (reward - expected), reward 1 for the right option (the log-score rule), no backprop.
eta 0.05, 20 passes in a fixed shuffled order, inputs scaled to unit spread on the training questions.
Training sets compared on dev: (a) SciQ dev+test only, (b) all training questions. Ties share credit.
"""
import math, random, re, sys

W = re.compile(r"\b(not|least|except|false|unlikely|cannot|never|incorrect|untrue|isn't|doesn't|can't|won't|wouldn't|aren't)\b", re.I)
ARC = "/root/repos/mimir/data/shelf/arc/"
_q = {}
def questions(f):
    if f not in _q: _q[f] = [l.split("\t")[1] for l in open(ARC + f)]
    return _q[f]

def load(path):
    rows, idx = [], {}
    for line in open(path):
        x = line.rstrip("\n").split("\t"); f = x[0]; i = idx.get(f, 0); idx[f] = i + 1
        opts = [list(map(float, o.split())) for o in x[3:3 + int(x[2])]]
        neg = bool(W.search(questions(f)[i]))
        feats = []
        for k, o in enumerate(opts):
            others = [p for j, p in enumerate(opts) if j != k]
            v = o[:9] + [o[2] - sum(p[2] for p in others) / len(others), o[1] - sum(p[1] for p in others) / len(others)]
            feats.append([-a for a in v] if neg else v)
        rows.append((f, int(x[1]), feats))
    return rows

def acc(rows, w, sc):
    ok, per = 0.0, {}
    for f, want, feats in rows:
        s = [sum(wi * a / si for wi, a, si in zip(w, v, sc)) for v in feats]; b = max(s); t = [i for i, v in enumerate(s) if v > b - 1e-9]
        c = (1.0 / len(t)) if want in t else 0.0; ok += c; per.setdefault(f, [0, 0]); per[f][0] += c; per[f][1] += 1
    return 100 * ok / len(rows), {f: 100 * a / n for f, (a, n) in per.items()}

def learn(rows, eta=0.05, passes=20):
    allv = [v for _, _, feats in rows for v in feats]; n = len(allv[0])
    sc = []
    for i in range(n):
        m = sum(v[i] for v in allv) / len(allv); sd = math.sqrt(sum((v[i] - m) ** 2 for v in allv) / len(allv)); sc.append(sd if sd > 1e-9 else 1.0)
    w = [0.0] * n; order = list(range(len(rows))); random.Random(12345).shuffle(order)
    for _ in range(passes):
        for qi in order:
            _, want, feats = rows[qi]; x = [[a / s for a, s in zip(v, sc)] for v in feats]
            act = [sum(wi * a for wi, a in zip(w, v)) for v in x]; mx = max(act); e = [math.exp(a - mx) for a in act]; z = sum(e); p = [a / z for a in e]
            for k, v in enumerate(x):
                r = (1.0 if k == want else 0.0) - p[k]
                for i in range(n): w[i] += eta * v[i] * r
    return w, sc

if __name__ == "__main__":
    train, dev = load(sys.argv[1]), load(sys.argv[2])
    names = "fact joint open meaning chain open2 open3 spec olen open-vs-others joint-vs-others".split()
    base = [1, 0.5, 2, 4, 0, 0, 0, 0, 0, 0, 0]
    a0, per0 = acc(dev, base, [1] * 11)
    print("stages 1-2 (fixed weights) on dev: %.2f%%   %s" % (a0, "  ".join("%s %.2f" % (f.split("_")[0], v) for f, v in sorted(per0.items()))))
    best = None
    for name, sel in (("a: SciQ dev+test only", lambda f: f.startswith("sciq_")), ("b: all training questions", lambda f: True)):
        tr = [r for r in train if sel(r[0])]; w, sc = learn(tr); a, per = acc(dev, w, sc)
        print("%-28s %5d training q   dev %.2f%%   %s" % (name, len(tr), a, "  ".join("%s %.2f" % (f.split("_")[0], v) for f, v in sorted(per.items()))))
        print("    weights:", "  ".join("%s %+.3f" % (k, wi) for k, wi in zip(names, w)))
        if best is None or a > best[0]: best = (a, name, w, sc)
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as o:
            o.write("# answer neurons, LAYA_LEVEL.md stage 3, trained on %s; weight per input, already divided by its spread\n" % best[1])
            o.write(" ".join("%.6f" % (wi / s) for wi, s in zip(best[2], best[3])) + "\n")
        print("kept:", best[1], "dev %.2f%%" % best[0])
