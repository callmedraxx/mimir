#!/usr/bin/env python3
"""GPT-2 on the no-passage tests Mimir is scored on (ARC-Easy, ARC-Challenge, OpenBookQA test), no training.
Scored as language-model papers do (lm-evaluation-harness): each option's log-probability after "Question: ...\\nAnswer:";
acc = highest total, acc_norm = highest per character. Ties share credit. One RESULT line per test and model.

In Colab:  !curl -sL https://raw.githubusercontent.com/callmedraxx/mimir/reader/tools/gpt2_arc.py -o g.py && python g.py [gpt2 gpt2-medium ...]
"""
import subprocess, sys, time, urllib.request
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=False)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

B = "https://raw.githubusercontent.com/callmedraxx/mimir/reader/data/arc_tests/"
dev = "cuda" if torch.cuda.is_available() else "cpu"
models = sys.argv[1:] or ["gpt2"]
tests = {n: [l.split("\t") for l in urllib.request.urlopen(B + n + ".tsv").read().decode().splitlines() if l.strip()]
         for n in ("easy_test", "challenge_test", "obqa_test")}

def credit(s, want):
    top = max(s); tied = [i for i, v in enumerate(s) if v >= top - 1e-9]
    return (1.0 / len(tied)) if want in tied else 0.0

for name in models:
    tok = AutoTokenizer.from_pretrained("openai-community/" + name if "/" not in name else name)
    lm = AutoModelForCausalLM.from_pretrained("openai-community/" + name if "/" not in name else name).to(dev).eval()
    print(name, "| torch", torch.__version__, "| device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
    for tn, rows in tests.items():
        ok = okn = 0.0; t0 = time.time()
        for f in rows:
            no = int(f[2]); opts = f[3:3 + no]; want = int(f[3 + no])
            ctx = tok("Question: " + f[1] + "\nAnswer:").input_ids
            s, sn = [], []
            for o in opts:
                cont = tok(" " + o).input_ids; ids = torch.tensor([ctx + cont], device=dev)
                with torch.no_grad(): lp = torch.log_softmax(lm(ids).logits[0, :-1].float(), -1)
                tgt = ids[0, 1:]; lps = lp[torch.arange(len(tgt)), tgt][len(ctx) - 1:].sum().item()
                s.append(lps); sn.append(lps / max(1, len(" " + o)))
            ok += credit(s, want); okn += credit(sn, want)
        n = len(rows)
        print("%-12s %-16s %5d q  acc %6.2f%%  acc_norm %6.2f%%  %.1f ms/q" % (name, tn, n, 100 * ok / n, 100 * okn / n, 1000 * (time.time() - t0) / n), flush=True)
        print("RESULT gpt2_arc model=%s file=%s n=%d acc=%.2f acc_norm=%.2f" % (name, tn, n, 100 * ok / n, 100 * okn / n), flush=True)
