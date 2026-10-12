#!/usr/bin/env python3
"""Laya reads data/test_math/text.txt and answers its 10 questions (no training). Colab:
  !curl -sL https://raw.githubusercontent.com/callmedraxx/mimir/reader/tools/laya_math.py -o m.py && python m.py
"""
import json, subprocess, sys, urllib.request
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya"], check=False)
import laya, torch
B = "https://raw.githubusercontent.com/callmedraxx/mimir/reader/data/test_math/"
text = urllib.request.urlopen(B + "text.txt").read().decode()
qs = json.loads(urllib.request.urlopen(B + "questions.json").read().decode())
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("laya", laya.__version__, dev)
for name in ("convaiinnovations/laya", "convaiinnovations/laya-multilingual", "convaiinnovations/laya-typed-decisions"):
    try:
        ag = laya.load(name, device=dev)
    except Exception as e:
        print(name.split("/")[1], "failed", str(e)[:40]); continue
    got, ok = "", 0
    for q in qs:
        r = ag.predict(text, {"answer": {"type": "choice", "instructions": q["q"], "criteria": q["options"]}})
        c = str(r["answers"]["answer"]["choice"]); got += c; ok += c == q["answer"]
    print("\n" + name.split("/")[1].upper())
    print("score %d / 10" % ok)
    print("chose  " + got)
    print("key    " + "".join(q["answer"] for q in qs))
    del ag
