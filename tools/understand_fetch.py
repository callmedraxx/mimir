#!/usr/bin/env python3
"""Fetch the understanding test sets and build the shared word-class table.

    python3 tools/understand_fetch.py        (from the repo root; stdlib only)

Writes data/understand/ (not checked in):
  babi/qaN_{train,test}.txt      bAbI tasks 1-20, English 1k (Weston et al. 2015), as published
  mctest/mcN.{train,dev,test}.txt MCTest (Richardson et al. 2013), one record per story:
                                  S <id>  /  T <story text, one line>  /  Q <single|multiple> <question> <A> <B> <C> <D> <answer letter>  (tabs)
  wordclass.tsv                   word, class, verb base, noun base: the commonest Brown-corpus tag of
                                  each word (classes below), base forms from WordNet's exception lists
                                  and -s/-ed/-ing rules checked against WordNet's lemmas
"""
import io, json, os, re, sys, tarfile, time, urllib.request, zipfile
from collections import Counter, defaultdict

OUT = "data/understand"
BABI = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"
ROWS = "https://datasets-server.huggingface.co/rows?dataset=sagnikrayc/mctest&config={}&split={}&offset={}&length=100"
NLTK = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{}.zip"

def get(url):
    for t in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r: return r.read()
        except Exception as e:
            if t == 5: raise
            time.sleep(2 + 3 * t)

def babi():
    os.makedirs(f"{OUT}/babi", exist_ok=True)
    tf = tarfile.open(fileobj=io.BytesIO(get(BABI)))
    for m in tf.getmembers():
        g = re.search(r"/en/qa(\d+)_[a-z-]+_(train|test)\.txt$", m.name)
        if g: open(f"{OUT}/babi/qa{g[1]}_{g[2]}.txt", "wb").write(tf.extractfile(m).read())

def mctest():
    os.makedirs(f"{OUT}/mctest", exist_ok=True)
    for cfg in ("mc160", "mc500"):
        for split, name in (("train", "train"), ("validation", "dev"), ("test", "test")):
            rows, off = [], 0
            while True:
                d = json.loads(get(ROWS.format(cfg, split, off)))
                rows += [r["row"] for r in d["rows"]]; off += 100
                if off >= d["num_rows_total"]: break
            stories = {}
            for r in rows: stories.setdefault(r["idx"]["story"], []).append(r)
            with open(f"{OUT}/mctest/{cfg}.{name}.txt", "w") as f:
                for sid, qs in stories.items():
                    text = qs[0]["story"].replace("\\newline", "\n").replace("\t", " ")
                    text = re.sub(r"\s*\n\s*", " ", text).strip()
                    f.write(f"S\t{sid}\nT\t{text}\n")
                    for r in sorted(qs, key=lambda r: r["idx"]["question"]):
                        o = r["answer_options"]; clean = lambda s: s.replace("\t", " ").replace("\n", " ").strip()
                        f.write("Q\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format("multiple" if r["question_is_multiple"] else "single",
                                clean(r["question"]), clean(o["A"]), clean(o["B"]), clean(o["C"]), clean(o["D"]), r["answer"]))

# Brown tag → class: N noun, P proper noun, V verb, X be/have/do, M modal, J adjective, R adverb,
# T particle, I preposition, D determiner, Q number, O pronoun, C and/or, S subordinator, W wh-word, E there, G not
def tclass(t):
    t = t.split("-")[0].split("+")[0].rstrip("$*") if not t.startswith("*") else "*"
    if t in ("*",): return "G"
    for p, c in (("np", "P"), ("nn", "N"), ("nr", "N"), ("vb", "V"), ("be", "X"), ("hv", "X"), ("do", "X"), ("md", "M"),
                 ("jj", "J"), ("rp", "T"), ("rb", "R"), ("ql", "R"), ("rn", "R"), ("in", "I"), ("to", "I"), ("at", "D"),
                 ("dt", "D"), ("ab", "D"), ("ap", "D"), ("cd", "Q"), ("od", "J"), ("pp", "O"), ("cc", "C"), ("cs", "S"),
                 ("w", "W"), ("ex", "E")):
        if t.startswith(p): return c
    return None

def wordclass():
    brown = zipfile.ZipFile(io.BytesIO(get(NLTK.format("brown"))))
    wn = zipfile.ZipFile(io.BytesIO(get(NLTK.format("wordnet"))))
    cnt = defaultdict(Counter)
    for n in brown.namelist():
        if not re.search(r"/c[a-r]\d\d$", n): continue
        for tok in brown.read(n).decode("latin-1").split():
            w, _, t = tok.rpartition("/")
            if not w or not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", w): continue
            c = tclass(t.lower())
            if c: cnt[w.lower()][c] += 1
    def lemmas(pos):
        s = set()
        for line in wn.read(f"wordnet/index.{pos}").decode().splitlines():
            if not line.startswith(" "): s.add(line.split()[0])
        return s
    def exc(pos):
        return dict(l.split()[:2] for l in wn.read(f"wordnet/{pos}.exc").decode().splitlines() if len(l.split()) >= 2)
    VL, NL, VX, NX = lemmas("verb"), lemmas("noun"), exc("verb"), exc("noun")
    def vbase(w):
        if w in VX: return VX[w]
        if w in VL: return w
        for suf, reps in (("ies", ["y"]), ("es", ["e", ""]), ("s", [""]), ("ied", ["y"]), ("ed", ["", "e"]), ("ing", ["", "e"])):
            if w.endswith(suf) and len(w) > len(suf) + 1:
                stem = w[: -len(suf)]
                for r in reps:
                    if stem + r in VL: return stem + r
                if len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in VL: return stem[:-1]
        return w
    def nbase(w):
        if w in NX: return NX[w]
        if w in NL: return w
        for suf, reps in (("ies", ["y"]), ("es", ["e", ""]), ("s", [""])):
            if w.endswith(suf) and len(w) > len(suf) + 1:
                for r in reps:
                    if w[: -len(suf)] + r in NL: return w[: -len(suf)] + r
        return w
    with open(f"{OUT}/wordclass.tsv", "w") as f:
        for w in sorted(cnt):
            c = cnt[w].most_common(1)[0][0]
            # a word seen as a verb at least a quarter as often as its commonest class keeps a V flag
            v = "v" if c != "V" and cnt[w]["V"] * 4 >= cnt[w][c] else ""
            f.write(f"{w}\t{c}{v}\t{vbase(w)}\t{nbase(w)}\n")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("bAbI", babi), ("MCTest", mctest), ("word classes", wordclass)):
        print(name, "...", flush=True); fn()
    print("done:", OUT)
