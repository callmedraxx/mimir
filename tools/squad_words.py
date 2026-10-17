#!/usr/bin/env python3
"""docs/SPEAK.md: SQuAD 1.1 in the passage layout with A lines (A id question answer1 || answer2 ...): the answer must be said.
  python3 tools/squad_words.py <squad json> <out.txt> [N questions, 0 = all]
Dev (docs/SPEAK.md): the first 2,000 questions of train-v1.1.json; test: dev-v1.1.json whole (10,570)."""
import json, re, sys
src, out, N = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
def clean(s): return re.sub(r"\s+", " ", s.replace("\t", " ")).strip()
d = json.load(open(src)); n = 0; np = 0
with open(out, "w") as f:
    for art in d["data"]:
        for para in art["paragraphs"]:
            np += 1; f.write("S\tsquad.%d\nT\t%s\n" % (np, clean(para["context"])))
            for qa in para["qas"]:
                ans = []
                for a in qa["answers"]:
                    t = clean(a["text"])
                    if t and t not in ans: ans.append(t)
                if not ans: continue
                f.write("A\t%s\t%s\t%s\n" % (qa["id"], clean(qa["question"]), " || ".join(ans))); n += 1
                if N and n >= N: break
            if N and n >= N: break
        if N and n >= N: break
print("%s: %d passages, %d questions" % (out, np, n))
