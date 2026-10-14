#!/usr/bin/env python3
"""docs/CORTEX.md cell B, the yardstick: word2vec CBOW (backprop, gensim), on the same text, the same held-out lines
(every 200th), the same vocabulary, one pass, scored on the same items as cortex.c.

  cortex_w2v.py VOCAB TEXT...      env: MAXTOK (words, 0 = all), WS, T2, STORIES, SPLIT, STORYW, DIM (300), WIN (2)
"""
import os, re, sys
import numpy as np
from gensim.models import Word2Vec

vocab_path, files = sys.argv[1], sys.argv[2:]
V = int(os.environ.get("V", 50000)); MAXTOK = int(os.environ.get("MAXTOK", 0))
DIM = int(os.environ.get("DIM", 300)); WIN = int(os.environ.get("WIN", 2)); STORYW = float(os.environ.get("STORYW", 1))
vocab = [l.split("\t")[0] for l in open(vocab_path)][:V]; known = set(vocab)
word = re.compile(r"[a-z]+")

class Text:
    """the training lines (held-out ones skipped), as known words only, stopping at MAXTOK words"""
    def __iter__(self):
        n = 0; line_no = 0
        for p in files:
            with open(p, errors="replace") as f:
                for line in f:
                    held = line_no % 200 == 0; line_no += 1
                    if held: continue
                    ws = [w for w in word.findall(line.lower()) if w in known]
                    if not ws: continue
                    yield ws; n += len(ws)
                    if MAXTOK and n >= MAXTOK: return

m = Word2Vec(vector_size=DIM, window=WIN, min_count=1, sg=0, negative=5, sample=1e-4, workers=2, epochs=1, seed=12345)
m.build_vocab(Text()); m.train(Text(), total_examples=m.corpus_count, epochs=1)
kv = m.wv; out = m.syn1neg
idx = {w: kv.key_to_index[w] for w in vocab if w in kv.key_to_index}
print("cell B  word2vec CBOW dim=%d WIN=%d, %d words known" % (DIM, WIN, len(idx)))

def spearman(x, y):
    rk = lambda a: np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    def rank(a):   # average ranks for ties
        a = np.asarray(a); s = np.argsort(a, kind="stable"); r = np.empty(len(a)); i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[s[j + 1]] == a[s[i]]: j += 1
            r[s[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
        return r
    rx, ry = rank(x), rank(y); return float(np.corrcoef(rx, ry)[0, 1])

for p in os.environ.get("WS", "").split():   # T1
    xs, ys, k = [], [], 0
    for l in open(p):
        a, b, h = l.split()[:3]; a, b = a.lower(), b.lower()
        if a in idx and b in idx: xs.append(float(kv.similarity(a, b))); k += 1
        else: xs.append(0.0)
        ys.append(float(h))
    print("T1 %-24s %4d pairs (%4d known)  spearman %.3f" % (os.path.basename(p), len(xs), k, spearman(xs, ys)))

order = np.array([idx[w] for w in vocab if w in idx]); names = [w for w in vocab if w in idx]; pos = {w: i for i, w in enumerate(names)}
def ids(text):
    left, right = text.split("___", 1) if "___" in text else (text, "")
    L = [w for w in word.findall(left.lower()) if w in idx]; R = [w for w in word.findall(right.lower()) if w in idx]
    return L, R
def rank(ctx, extra, ans):
    v = np.zeros(DIM, dtype=np.float32)
    for w in ctx: v += kv.vectors[idx[w]]
    for w in extra: v += kv.vectors[idx[w]] * (STORYW / len(extra))
    s = out[order] @ v
    for w in ctx: s[pos[w]] = -1e9
    a = s[pos[ans]]; return int((s >= a).sum()) - 1   # ties count against it
if os.environ.get("T2"):
    n = hit = 0
    for l in open(os.environ["T2"]):
        sent, ans = l.rstrip("\n").split("\t"); n += 1
        if ans not in idx or "___" not in sent: continue
        L, R = ids(sent); hit += rank(L[-WIN:] + R[:WIN], [], ans) < 10
    print("T2 fill the blank          %4d items  hit@10 %.2f%%" % (n, 100 * hit / max(n, 1)))
if os.environ.get("STORIES"):
    sp = os.environ.get("SPLIT", "dev"); d = os.path.dirname(os.environ["STORIES"]); n = hit = 0
    for l in open(os.environ["STORIES"]):
        if l.startswith("#"): continue
        s, name, sent, ans = l.rstrip("\n").split("\t")
        if s != sp: continue
        n += 1
        story = [w for w in word.findall(open(os.path.join(d, "stories", name + ".txt")).read().lower()) if w in idx]
        if ans not in idx: print("T3 %-45s %-12s not a known word" % (sent, ans)); continue
        L, R = ids(sent); r = rank(L[-WIN:] + R[:WIN], story, ans); hit += r < 10
        print("T3 %-45s %-12s rank %d%s" % (sent, ans, r + 1, "  (top 10)" if r < 10 else ""))
    print("T3 stories (%s)             %4d items  hit@10 %.2f%%" % (sp, n, 100 * hit / max(n, 1)))
for w in ("owl", "king", "water", "france", "blue"):
    if w in idx: print("%s: %s" % (w, " ".join("%s(%.2f)" % (a, b) for a, b in kv.most_similar(w, topn=10))))
