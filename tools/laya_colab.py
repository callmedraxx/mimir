#!/usr/bin/env python3
"""Run Laya on Typed Decisions (test, 400 cases, 2,000 decisions), scored as tools/decide_score.py.

In Colab (GPU runtime):  !curl -sL https://raw.githubusercontent.com/callmedraxx/mimir/reader/tools/laya_colab.py -o l.py && python l.py
Checkpoints: convaiinnovations/laya (as shipped: zero-shot here) and
convaiinnovations/laya-typed-decisions (fitted on this benchmark's train split).
Prints a table and one RESULT line per checkpoint.
"""
import json, math, subprocess, sys, time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya", "pandas", "pyarrow"], check=False)
import laya, pandas as pd, torch

URL = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all/test-00000-of-00001.parquet"
df = pd.read_parquet(URL)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("laya", laya.__version__, "| torch", torch.__version__, "| device", device,
      torch.cuda.get_device_name(0) if device == "cuda" else "", "| cases", len(df), flush=True)

def dist(ans, q):
    if q["type"] == "noul":
        p = float(ans.get("noul", ans.get("probabilities", {}).get("true", 0.5)))
        return {"false": 1 - p, "true": p}
    return {str(k): float(v) for k, v in ans["probabilities"].items()}

def score_one(pr, g):
    gd = {k: float(v) for k, v in g["probabilities"].items()}; opts = list(gd)
    s = sum(max(pr.get(o, 0.0), 0.0) for o in opts) or 1.0; p = {o: max(pr.get(o, 0.0), 0.0) / s for o in opts}
    top = max(p.values()); tied = [o for o in opts if p[o] >= top - 1e-12]
    acc = (1.0 / len(tied)) if str(g["label"]) in tied else 0.0
    kl = sum(gd[o] * math.log(gd[o] / max(p[o], 1e-6)) for o in opts if gd[o] > 0)
    br = sum((p[o] - gd[o]) ** 2 for o in opts)
    return acc, kl, br

rows = []
for name in ("convaiinnovations/laya", "convaiinnovations/laya-typed-decisions"):
    try:
        ag = laya.load(name, device=device)
    except Exception as e:
        print("could not load", name, repr(e), flush=True); continue
    st0 = json.loads(df.iloc[0]["state"]); ag.predict(st0, json.loads(df.iloc[0]["questions"]))   # warm-up, not timed
    A = K = B = 0.0; n = 0; secs = 0.0; per_wf = {}; errors = 0
    for _, r in df.iterrows():
        st, qs, gold = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"])
        t0 = time.perf_counter()
        try:
            res = ag.predict(st, qs)
        except Exception as e:
            errors += 1; res = {"answers": {}}
        if device == "cuda": torch.cuda.synchronize()
        secs += time.perf_counter() - t0
        for q, g in gold.items():
            ans = res.get("answers", {}).get(q)
            pr = dist(ans, qs[q]) if ans else {}
            a, k, b = score_one(pr, g); A += a; K += k; B += b; n += 1
            w = per_wf.setdefault(r["workflow"], [0.0, 0]); w[0] += a; w[1] += 1
    rows.append((name, A / n, K / n, B / n, 1000 * secs / len(df), errors))
    print("\n%s: accuracy %.3f  KL %.3f  Brier %.3f  | %.1f ms per case (5 questions) | errors %d" % rows[-1], flush=True)
    for wf, (a, m) in sorted(per_wf.items()): print("   %-28s %.3f" % (wf, a / m))
    print("RESULT %s acc=%.4f kl=%.4f brier=%.4f ms_per_case=%.1f errors=%d device=%s" % (rows[-1] + (device,)), flush=True)
    del ag
    if device == "cuda": torch.cuda.empty_cache()

print("\n%-42s %8s %7s %7s %10s" % ("checkpoint", "accuracy", "KL", "Brier", "ms/case"))
for r in rows: print("%-42s %8.3f %7.3f %7.3f %10.1f" % r[:5])
print("(published: laya 0.362, laya-typed-decisions 0.766; Mimir 0.726 / KL 0.154 / Brier 0.081)")
json.dump([dict(zip(("name", "acc", "kl", "brier", "ms", "errors"), r)) for r in rows], open("laya_results.json", "w"))
