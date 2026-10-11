"""
gist_math.py — can Mimir get the gist of a story the way people do?
Math first, code later.

The model (Kintsch's construction-integration, the standard account of how
people remember what they read):
  * read in cycles, one sentence a cycle;
  * the workspace holds the new sentence plus the few (WM = 4) most active
    ideas carried over from before; nothing else is in mind;
  * ideas in the workspace activate each other: sharing a character or a
    thing (the same content word), or meaning something close (cosine of
    the learned word meanings, from sandbox/meaning.c);
  * activation spreads until it settles; whatever was active adds to that
    sentence's memory strength; the WM strongest are carried on.
  * The gist = the strongest sentences, in story order, up to BUDGET words.
"Who" = the content words with the most memory strength behind them.

Scored against the human plot summary (data/gist, from Wikipedia):
ROUGE-1 recall / F1 on content words, and ROUGE-2 recall, averaged over
the tales. Compared with: the first BUDGET words (lead), random sentences,
and a no-workspace version (each sentence scored against the whole story at
once — what a reader with unlimited memory would do).

    python3 sandbox/gist_math.py MEANINGS.mng [tale ...]
"""
import collections, glob, math, os, random, re, struct, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GIST = os.path.join(HERE, "..", "data", "gist")
BOOKS = os.path.join(HERE, "..", "data", "reader", "books")
WM, BUDGET = 4, 120
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def words_of(s):
    return WORD.findall(s.lower().replace("’", "'"))


def function_words(n=150):
    """The most frequent words in what Mimir has read: learned, not a list."""
    c = collections.Counter()
    for p in glob.glob(os.path.join(BOOKS, "shelf", "*.txt")) + glob.glob(os.path.join(BOOKS, "*.txt")):
        c.update(words_of(open(p, encoding="utf-8").read()))
    return {w for w, _ in c.most_common(n)}


def load_meanings(path):
    m = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"MMNG"
        D, n = struct.unpack("ii", f.read(8))
        for _ in range(n):
            l = f.read(1)[0]
            w = f.read(l).decode()
            sc = struct.unpack("f", f.read(4))[0]
            v = np.frombuffer(f.read(D), dtype=np.int8).astype(np.float32) * sc
            m[w] = v / (np.linalg.norm(v) + 1e-9)
    return m, D


def sentences(story):
    text = re.sub(r"\s+", " ", story.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"'))
    parts = re.split(r"(?<=[.!?])['\"]?\s+(?=['\"]?[A-Z])", text)
    out = [p.strip() for p in parts if len(words_of(p)) >= 3]
    if os.environ.get("CLAUSES"):
        # one event per unit: split at ; : and at ", and" / ", but" / ", so" / ", then"
        ev = []
        for p in out:
            for c in re.split(r"[;:]\s+|,\s+(?=(?:and|but|so|then|when|while)\b)", p):
                if len(words_of(c)) >= 3:
                    ev.append(c.strip(" ,"))
        return ev
    return out


class Reader:
    def __init__(self, meanings, D, stop):
        self.m, self.D, self.stop = meanings, D, stop

    def content(self, s):
        return [w for w in words_of(s) if w not in self.stop]

    def vec(self, ws):
        v = np.zeros(self.D, np.float32)
        for w in ws:
            if w in self.m:
                v += self.m[w]
        n = np.linalg.norm(v)
        return v / n if n else v

    def link(self, a, b):
        """How strongly two ideas activate each other."""
        share = len(set(a["c"]) & set(b["c"]))
        return share + 2.0 * max(0.0, float(a["v"] @ b["v"]))

    def read(self, story, use_workspace=True):
        S = [{"i": i, "s": s} for i, s in enumerate(sentences(story))]
        for x in S:
            x["c"] = self.content(x["s"])
            x["v"] = self.vec(x["c"])
        strength = np.zeros(len(S))
        if not use_workspace:                       # unlimited memory: everything at once
            for a in S:
                strength[a["i"]] = sum(self.link(a, b) for b in S if b is not a)
            return S, strength
        carried = []
        R = int(os.environ.get("RECALL", "0"))
        for x in S:                                  # one cycle per sentence
            # cued recall (hippocampus): the new idea brings back the R most
            # related earlier ideas that are no longer in the workspace
            recalled = []
            if R and x["i"]:
                held = {n["i"] for n in carried}
                past = [S[j] for j in range(x["i"]) if j not in held]
                past.sort(key=lambda y: -self.link(x, y))
                recalled = [y for y in past[:R] if self.link(x, y) > 0]
            nodes = carried + recalled + [x]
            n = len(nodes)
            W = np.array([[0.0 if i == j else self.link(nodes[i], nodes[j]) for j in range(n)] for i in range(n)])
            a = np.ones(n) / n
            for _ in range(30):                      # spread until it settles
                b = W @ a + a
                b /= b.sum() + 1e-12
                if np.abs(b - a).max() < 1e-4:
                    a = b
                    break
                a = b
            for node, act in zip(nodes, a):
                strength[node["i"]] += act
            order = np.argsort(-a)
            carried = [nodes[k] for k in order[:WM]]
        p = float(os.environ.get("FAIR", "0"))
        if p:
            # fairness: an idea read late has had fewer cycles to gather
            # strength; compare strength per chance it had (cycles since read)
            n = len(S)
            strength = strength / np.array([(n - x["i"]) ** p for x in S])
        return S, strength

    def gist(self, story, use_workspace=True, novelty=0.0, ends=0.0, spread=0.0):
        S, strength = self.read(story, use_workspace)
        N = len(S)
        strength = strength / (strength.max() + 1e-12)
        if spread:
            # characters carry the story: a word that keeps coming back across
            # the whole story (in many different parts) lifts its sentences
            parts = collections.defaultdict(set)
            for x in S:
                for w in set(x["c"]):
                    parts[w].add(x["i"] * 8 // max(1, N))
            for x in S:
                lift = max([len(parts[w]) for w in set(x["c"])] or [0]) / 8
                strength[x["i"]] += spread * lift
        if ends and N > 2:
            # the setting and the outcome are remembered best
            for x in S:
                pos = x["i"] / (N - 1)
                strength[x["i"]] += ends * (max(0.0, 1 - pos * 6) + max(0.0, 1 - (1 - pos) * 6))
        picked, used = [], 0
        cover = np.zeros(self.D, np.float32)
        cand = list(range(N))
        while cand and used < BUDGET:
            # repeats fold: a sentence close to what is already told counts less
            def value(k):
                v = strength[k]
                if novelty and picked:
                    v -= novelty * max(0.0, float(S[k]["v"] @ cover) / (np.linalg.norm(cover) + 1e-9))
                return v
            k = max(cand, key=value)
            cand.remove(k)
            n = len(words_of(S[k]["s"]))
            if used + n > BUDGET and used:
                continue
            picked.append(k)
            used += n
            cover += S[k]["v"]
        who = collections.Counter()
        for x in S:
            for w in set(x["c"]):
                who[w] += strength[x["i"]]
        return " ".join(S[k]["s"] for k in sorted(picked)), [w for w, _ in who.most_common(6)]


def rouge(pred, ref, stop):
    p = [w for w in words_of(pred) if w not in stop]
    r = [w for w in words_of(ref) if w not in stop]
    cp, cr = collections.Counter(p), collections.Counter(r)
    hit = sum(min(cp[w], cr[w]) for w in cr)
    rec, prec = hit / max(1, len(r)), hit / max(1, len(p))
    f1 = 2 * rec * prec / (rec + prec) if hit else 0.0
    bp = collections.Counter(zip(p, p[1:]))
    br = collections.Counter(zip(r, r[1:]))
    r2 = sum(min(bp[b], br[b]) for b in br) / max(1, sum(br.values()))
    return rec, f1, r2


def lead(story):
    out, used = [], 0
    for s in sentences(story):
        if used >= BUDGET:
            break
        out.append(s)
        used += len(words_of(s))
    return " ".join(out)


def rand(story, seed):
    S = sentences(story)
    rng = random.Random(seed)
    idx = list(range(len(S)))
    rng.shuffle(idx)
    out, used = [], 0
    for k in idx:
        if used >= BUDGET:
            break
        out.append(k)
        used += len(words_of(S[k]))
    return " ".join(S[k] for k in sorted(out))


def main():
    meanings, D = load_meanings(sys.argv[1])
    stop = function_words()
    rd = Reader(meanings, D, stop)
    tales = sorted(os.path.basename(p)[:-10] for p in glob.glob(os.path.join(GIST, "*.story.txt")))
    if len(sys.argv) > 2:
        tales = sys.argv[2:]
    scores = collections.defaultdict(list)
    show = len(sys.argv) > 2
    for t in tales:
        story = open(os.path.join(GIST, t + ".story.txt")).read()
        ref = open(os.path.join(GIST, t + ".summary.txt")).read()
        g, who = rd.gist(story, True)
        scores["Mimir: workspace of 4 (brain)"].append(rouge(g, ref, stop))
        g3, who3 = rd.gist(story, True, novelty=1.0)
        scores["  + repeats fold"].append(rouge(g3, ref, stop))
        g4, who4 = rd.gist(story, True, novelty=1.0, ends=0.5)
        scores["  + repeats fold + beginning/end"].append(rouge(g4, ref, stop))
        g5, who5 = rd.gist(story, True, novelty=1.0, ends=0.5, spread=1.0)
        scores["  + ... + characters carry it"].append(rouge(g5, ref, stop))
        if show:
            print(f"\n== {t} (all three)\nGist: {g5}\n")
        # oracle: greedily the sentences that best match the human summary
        S = sentences(story)
        best, used, cur = [], 0, ""
        while used < BUDGET:
            pick = max((k for k in range(len(S)) if k not in best),
                       key=lambda k: rouge(cur + " " + S[k], ref, stop)[0], default=None)
            if pick is None:
                break
            best.append(pick)
            used += len(words_of(S[pick]))
            cur = " ".join(S[k] for k in sorted(best))
        scores["Best possible sentences (oracle)"].append(rouge(cur, ref, stop))
        g2, _ = rd.gist(story, False)
        scores["Unlimited memory (whole story at once)"].append(rouge(g2, ref, stop))
        scores["First 120 words"].append(rouge(lead(story), ref, stop))
        rs = [rouge(rand(story, k), ref, stop) for k in range(10)]
        scores["Random sentences"].append(tuple(np.mean(rs, axis=0)))
        if show:
            print(f"\n== {t}\nWho: {', '.join(who)}\nGist: {g}\n")
    print(f"{len(tales)} tales, {BUDGET}-word retellings, scored against the Wikipedia plot summaries")
    print(f"{'':40s} {'R-1 recall':>10s} {'R-1 F1':>8s} {'R-2 recall':>10s}")
    for name, v in scores.items():
        a = np.mean(v, axis=0)
        print(f"{name:40s} {a[0]*100:10.1f} {a[1]*100:8.1f} {a[2]*100:10.1f}")


if __name__ == "__main__":
    main()
