#!/bin/sh
# Laya and Mimir on Typed Decisions in one Colab cell; ends with a summary narrow enough for a phone.
#   !curl -sL https://raw.githubusercontent.com/callmedraxx/mimir/reader/tools/both_colab.sh | sh
curl -sL https://raw.githubusercontent.com/callmedraxx/mimir/reader/tools/laya_colab.py -o l.py
python3 l.py > laya_log.txt 2>&1
curl -sL https://raw.githubusercontent.com/callmedraxx/mimir/reader/tools/mimir_colab.sh | sh > mimir_log.txt 2>&1
python3 - <<'PY'
import json, re
g = lambda k, s: re.search(k + r"=([0-9.]+)", s)
print("=== TYPED DECISIONS ===")
print("test: 400 cases")
print("2,000 decisions")
try:
    for r in json.load(open("laya_results.json")):
        n = "LAYA BASE" if r["name"].endswith("/laya") else "LAYA TRAINED"
        print("\n" + n); print("acc   %.3f" % r["acc"]); print("KL    %.3f" % r["kl"]); print("Brier %.3f" % r["brier"]); print("ms    %.1f /case" % r["ms"]); print("errs  %d" % r["errors"])
except Exception as e:
    print("\nLAYA failed:", str(e)[:60])
m = open("mimir_log.txt").read()
rs = re.findall(r"RESULT T acc=([0-9.]+) kl=([0-9.]+) brier=([0-9.]+) ms_per_case=([0-9.]+) mem_bytes=([0-9]+)", m)
z = re.search(r"Z on .*?accuracy ([0-9.]+)", m)
if rs:
    a, k, b, _, mb = rs[0]; ms = max(float(x[3]) for x in rs)
    print("\nMIMIR TRAINED"); print("acc   %.3f" % float(a)); print("KL    %.3f" % float(k)); print("Brier %.3f" % float(b)); print("ms    %.2f /case" % ms); print("      (1 CPU core,"); print("       worst of 3)"); print("mem   %.1f MB" % (int(mb) / 1e6))
if z: print("\nMIMIR ZERO-SHOT"); print("acc   %.3f" % float(z.group(1)))
gpu = re.search(r"device (\S+) ([^|]*)", open("laya_log.txt").read())
print("\nLaya on:", gpu.group(1) if gpu else "?"); print(gpu.group(2).strip()[:24] if gpu else "")
PY
