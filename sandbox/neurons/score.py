"""score.py — applies the rules in docs/NEURON_COMPARE.md to the phone results.
    python3 score.py res_prime.txt res_little.txt"""
import sys, collections
rows = []
for f in sys.argv[1:]:
    for line in open(f):
        if not line.startswith("core="): continue
        rows.append(dict(kv.split("=", 1) for kv in line.split()))
by = collections.defaultdict(list)
for r in rows: by[(r["design"], r["budget"], r["core"])].append(r)
def worst(d, b, core, k): return min(float(r[k]) for r in by[(d, b, core)])
def q(d, b, k): return float(by[(d, b, "prime")][0][k])
wins = {"v0.1": 0, "v0.2": 0, "tie": 0}
print(f"{'budget':7}{'cell':14}{'v0.1':>12}{'v0.2':>12}  winner")
for b in ["4MB", "16MB", "64MB"]:
    for k, name in [("T1", "T1 recall"), ("T2", "T2 old"), ("T3", "T3 recognise"), ("T4", "T4 next")]:
        a, c = q("v0.1", b, k) * 100, q("v0.2", b, k) * 100
        w = "tie" if abs(a - c) <= 1 else ("v0.1" if a > c else "v0.2"); wins[w] += 1
        print(f"{b:7}{name:14}{a:12.1f}{c:12.1f}  {w}")
    for k, name in [("store_tok_s", "store tok/s"), ("recall_s", "recall/s")]:
        a, c = worst("v0.1", b, "little", k), worst("v0.2", b, "little", k)
        w = "tie" if abs(a - c) <= 0.1 * max(a, c) else ("v0.1" if a > c else "v0.2"); wins[w] += 1
        print(f"{b:7}{name+' (L)':14}{a:12.0f}{c:12.0f}  {w}   prime worst: {worst('v0.1', b, 'prime', k):.0f} vs {worst('v0.2', b, 'prime', k):.0f}")
gate = {d: min(worst(d, b, "little", "store_tok_s") for b in ["4MB", "16MB", "64MB"]) for d in ["v0.1", "v0.2"]}
print("gate (little core store ≥ 10k tok/s):", {d: f"{v:.0f} {'pass' if v >= 10000 else 'FAIL'}" for d, v in gate.items()})
print("wins:", wins)
