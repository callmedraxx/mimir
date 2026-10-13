#!/usr/bin/env python3
"""Run Laya on the no-passage tests Mimir is scored on (ARC-Easy, ARC-Challenge, OpenBookQA test), no training.
Each question is a pick-one: the text is the question, the options are A, B, C, ... (as tools/laya_math.py).
Scored as bg.c: % right, ties share credit. Prints a table and one RESULT line per test.

In Colab (GPU runtime):  !curl -sL https://raw.githubusercontent.com/callmedraxx/mimir/reader/tools/laya_arc.py -o a.py && python a.py [TEST ...]
(TEST names from data/arc_tests/, default the three ARC/OpenBookQA tests; mmlu_hs_test and csqa_dev are the general ones)
"""
import subprocess, sys, time, urllib.request
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya"], check=False)
import laya, torch

B = "https://raw.githubusercontent.com/callmedraxx/mimir/reader/data/arc_tests/"
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("laya", laya.__version__, "| torch", torch.__version__, "| device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
ag = laya.load("convaiinnovations/laya", device=dev)
L = "ABCDEFGH"
TESTS = [a for a in sys.argv[1:]] or ["easy_test", "challenge_test", "obqa_test"]
for name in TESTS:
    rows = [l.rstrip("\n").split("\t") for l in urllib.request.urlopen(B + name + ".tsv").read().decode().splitlines() if l.strip()]
    ok = n = 0; t0 = time.time(); odd = 0
    for f in rows:
        no = int(f[2]); opts = f[3:3 + no]; want = int(f[3 + no])
        r = ag.predict(f[1], {"answer": {"type": "choice", "instructions": "Which answer to the question is correct?",
                                        "criteria": {L[i]: o for i, o in enumerate(opts)}}})
        a = r["answers"]["answer"]; p = a.get("probabilities") or {}
        if p:
            s = [float(p.get(L[i], 0)) for i in range(no)]; top = max(s); tied = [i for i in range(no) if s[i] >= top - 1e-12]
            ok += (1.0 / len(tied)) if want in tied else 0.0
        else:
            c = str(a.get("choice")); odd += c not in L[:no]; ok += c == L[want]
        n += 1
        if n % 500 == 0: print("  %s %d/%d  %.2f%%" % (name, n, len(rows), 100 * ok / n), flush=True)
    print("%-16s %5d q  %6.2f%%   %.1f ms/q%s" % (name, n, 100 * ok / n, 1000 * (time.time() - t0) / n, "  (%d answers not an option)" % odd if odd else ""), flush=True)
    print("RESULT laya_arc file=%s n=%d acc=%.2f" % (name, n, 100 * ok / n), flush=True)
