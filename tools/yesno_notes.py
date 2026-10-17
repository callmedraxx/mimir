#!/usr/bin/env python3
"""docs/YESNO.md Y0: read a BoolQ dump written with EPDUMP=1 (file, want, n, neg, then per option "fact joint open mean fit active",
then per option "efact ejoint eopen") and say which part decides and how.  python3 tools/yesno_notes.py DUMP [SHOW]"""
import sys
from collections import Counter
BW = [1, 0.5, 2, 4, 1]; EPW = 4.0
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if "\t" in l]
def sc(p): return sum(BW[k] * p[k] for k in range(5))
c = Counter(); thin_acc = Counter(); src = Counter(); ev_hist = Counter()
for r in rows:
    want = int(r[1]); P = [list(map(float, x.split())) for x in r[4:6]]; E = [list(map(float, x.split())) for x in r[6:8]] if len(r) >= 8 else None
    s = [sc(P[0]), sc(P[1])]; pick = 0 if s[0] >= s[1] else 1; tie = abs(s[0] - s[1]) < 1e-9
    c["n"] += 1; c["want_yes"] += want == 0; c["pick_yes"] += pick == 0; c["right"] += pick == want; c["tie"] += tie
    for k, nm in enumerate(["fact", "joint", "open"]):
        d = P[0][k] - P[1][k]
        if abs(d) > 1e-9: c["%s favours %s" % (nm, "yes" if d > 0 else "no")] += 1
    if E:
        for o in range(2):
            for k, nm in enumerate(["fact", "joint", "open"]):
                merged, ep = P[o][k], EPW * E[o][k]
                src["%s from %s" % (nm, "episode" if abs(merged - ep) < 1e-6 else "shelf" if abs(merged) > 1e-9 else "nothing")] += 1
        thin = all(abs(E[o][0]) < 0.05 and abs(E[o][2]) < 0.05 for o in range(2))
        thin_acc["thin n" if thin else "stated n"] += 1; thin_acc["thin right" if thin else "stated right"] += pick == want; thin_acc["thin pick_yes" if thin else "stated pick_yes"] += pick == 0
        ev = (BW[0] * E[0][0] + BW[1] * E[0][1] + BW[2] * E[0][2]) - (BW[0] * E[1][0] + BW[1] * E[1][1] + BW[2] * E[1][2])
        for T in (0.02, 0.05, 0.1, 0.2, 0.4):
            st = "true" if ev > T else "false" if ev < -T else "unknown"; ans = 0 if st != "false" else 1
            ev_hist["T=%.2f %s" % (T, st)] += 1; ev_hist["T=%.2f right" % T] += ans == want
            ep_pick = 0 if ev >= 0 else 1; ev_hist["episode-only right"] += (ep_pick == want) / 5.0
n = c["n"]
print("n %d  acc %.2f  picks yes %.0f%%  right is yes %.0f%%  ties %d" % (n, 100.0 * c["right"] / n, 100.0 * c["pick_yes"] / n, 100.0 * c["want_yes"] / n, c["tie"]))
print("parts:", ", ".join("%s %d" % (k, v) for k, v in sorted(c.items()) if "favours" in k))
print("where the merged part came from:", ", ".join("%s %d" % (k, v) for k, v in sorted(src.items())))
if thin_acc:
    for kind in ("stated", "thin"):
        m = thin_acc[kind + " n"]
        if m: print("%s by the passage: n %d  acc %.2f  picks yes %.0f%%" % (kind, m, 100.0 * thin_acc[kind + " right"] / m, 100.0 * thin_acc[kind + " pick_yes"] / m))
    print("episode-only sign decides: acc %.2f" % (100.0 * ev_hist["episode-only right"] / n))
    for T in (0.02, 0.05, 0.1, 0.2, 0.4):
        print("Y2 with THIN2=%.2f: true %d false %d unknown %d -> acc %.2f (unknown answered yes)" % (T, ev_hist["T=%.2f true" % T], ev_hist["T=%.2f false" % T], ev_hist["T=%.2f unknown" % T], 100.0 * ev_hist["T=%.2f right" % T] / n))
if len(sys.argv) > 2:
    wrong = [l for l in open(sys.argv[2]) if l.startswith("  ") and l.rstrip("\n").endswith("WRONG")]
    print("\nfirst misses:"); [print(l.rstrip()[:200]) for l in wrong[:12]]
