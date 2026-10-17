#!/usr/bin/env python3
"""docs/SPEAK.md: bAbI answered in words. Writes data/understand/survey/babi_words_<split>.txt in the passage layout with
A lines (A id question answer) instead of Q lines: the answer must be said, not chosen.
  python3 tools/babi_words.py train 1 2 3 6 9 11 13 [N per task, default 200]   (STATE's dev tasks)
  python3 tools/babi_words.py test  (all 20 tasks, 200 each)"""
import os, sys
split = sys.argv[1] if len(sys.argv) > 1 else "train"
tasks = [int(a) for a in sys.argv[2:] if a.isdigit() and int(a) <= 20] or list(range(1, 21))
N = int(sys.argv[-1]) if len(sys.argv) > 2 and int(sys.argv[-1]) > 20 else 200
name = os.environ.get("BABI_WORDS_NAME", "babi_words_%s" % split)
out = open("data/understand/survey/%s.txt" % name, "w"); nq = 0
for t in tasks:
    story, n, sid = [], 0, 0
    for l in open("data/understand/babi/qa%d_%s.txt" % (t, split)):
        num, rest = l.rstrip("\n").split(" ", 1)
        if num == "1": story = []; sid += 1; fresh = True
        if "\t" in rest:
            q, a, _ = rest.split("\t")
            if fresh: out.write("S\tbabiw%d.%d\nT\t%s\n" % (t, sid, " ".join(story))); fresh = False
            else: out.write("S\tbabiw%d.%d\nT\t%s\n" % (t, sid, " ".join(story)))
            ans = a.strip().replace(",", " ")
            if t == 19: ans = " ".join({"n": "north", "s": "south", "e": "east", "w": "west"}.get(w, w) for w in ans.split())   # the path as words
            out.write("A\tbabiw%d_%d\t%s\t%s\n" % (t, sid, q.strip(), ans)); n += 1; nq += 1
            if n >= N: break
        else: story.append(rest.strip())
out.close(); print("%s: %d questions, tasks %s" % (name, nq, tasks))
