#!/usr/bin/env python3
"""Score predictions on Typed Decisions the way its leaderboard does.

    python3 tools/decide_score.py PRED.jsonl [GOLD.jsonl]
    python3 tools/decide_score.py --reference        Prior and Uniform rows (checks this scorer)

PRED.jsonl: one line per case: {"id": ..., "answers": {question: {option: probability}}}.
Accuracy: the top option against the gold label (a tie shares the credit).
KL from gold: sum gold * log(gold / pred), pred floored at 1e-6.
Brier: sum over options of (pred - gold)^2, gold as its distribution.
Checked against the leaderboard's reference rows: Uniform KL 0.444 and Brier 0.238 exactly; Uniform and
Prior accuracy within 1.3 points (its tie-breaking and its Prior are not specified).
Means over all decisions (400 cases x 5 = 2,000).
"""
import json, math, sys, collections

def load(p): return [json.loads(l) for l in open(p)]
def gold_dist(g):
    return {k: float(v) for k, v in g["probabilities"].items()}

def score(preds, gold, show=True):
    P = {p["id"]: p["answers"] for p in preds}
    acc = kl = br = 0.0; n = 0; by = collections.defaultdict(lambda: [0.0, 0])
    for c in gold:
        for q, g in c["gold"].items():
            gd = gold_dist(g); opts = list(gd); pd = P[c["id"]][q]
            s = sum(max(pd.get(o, 0.0), 0.0) for o in opts) or 1.0
            pr = {o: max(pd.get(o, 0.0), 0.0) / s for o in opts}
            top = max(pr.values()); tied = [o for o in opts if pr[o] >= top - 1e-12]
            a = (1.0 / len(tied)) if g["label"] in tied else 0.0
            acc += a; by[(c["workflow"], q)][0] += a; by[(c["workflow"], q)][1] += 1
            kl += sum(gd[o] * math.log(gd[o] / max(pr[o], 1e-6)) for o in opts if gd[o] > 0)
            br += sum((pr[o] - gd[o]) ** 2 for o in opts)
            n += 1
    r = {"decisions": n, "accuracy": acc / n, "kl": kl / n, "brier": br / n}
    if show:
        print("decisions %d  accuracy %.3f  KL %.3f  Brier %.3f" % (n, r["accuracy"], r["kl"], r["brier"]))
    return r, by

def reference(train, test):
    freq = collections.defaultdict(collections.Counter)
    for c in train:
        for q, g in c["gold"].items(): freq[(c["workflow"], q)][g["label"]] += 1
    prior, unif = [], []
    for c in test:
        a, u = {}, {}
        for q, g in c["gold"].items():
            opts = list(g["probabilities"]); f = freq[(c["workflow"], q)]; t = sum(f.values())
            a[q] = {o: (f[o] + 0.0) / t for o in opts}; u[q] = {o: 1.0 / len(opts) for o in opts}
        prior.append({"id": c["id"], "answers": a}); unif.append({"id": c["id"], "answers": u})
    return prior, unif

if __name__ == "__main__":
    gold = load("data/decide/test.jsonl")
    if sys.argv[1] == "--reference":
        prior, unif = reference(load("data/decide/train.jsonl"), gold)
        print("Prior   (published 0.470 / 0.347 / 0.189):", end=" "); score(prior, gold)
        print("Uniform (published 0.308 / 0.444 / 0.238):", end=" "); score(unif, gold)
    else:
        score(load(sys.argv[1]), load(sys.argv[2]) if len(sys.argv) > 2 else gold)
