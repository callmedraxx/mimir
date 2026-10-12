#!/usr/bin/env python3
"""The shelf Mimir reads (docs/BACKGROUND_COMPARE.md), and the ARC questions.

  data/shelf/simplewiki.txt   Simple English Wikipedia articles as plain text: one paragraph a line, a blank line between articles
  data/shelf/arc_corpus.txt   the ARC Corpus (AllenAI): science sentences, one a line
  data/shelf/arc/{easy,challenge}_{train,dev,test}.tsv   id, question, n options, options..., answer index

Downloads into data/shelf/raw (simplewiki-latest-pages-articles.xml.bz2, ARC-V1-Feb2018.zip) if missing.
Run from the repo root.
"""
import bz2, json, os, re, subprocess, sys, xml.etree.ElementTree as ET, zipfile

RAW = "data/shelf/raw"
os.makedirs(RAW, exist_ok=True); os.makedirs("data/shelf/arc", exist_ok=True)
SW = os.path.join(RAW, "simplewiki.xml.bz2"); ARC = os.path.join(RAW, "arc.zip")
if not os.path.exists(SW): subprocess.run(["curl", "-sL", "-o", SW, "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"], check=True)
if not os.path.exists(ARC): subprocess.run(["curl", "-sL", "-o", ARC, "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip"], check=True)

# ---- ARC questions and corpus
with zipfile.ZipFile(ARC) as z:
    for name in z.namelist():
        if name.startswith("__MACOSX"): continue
        m = re.search(r"ARC-(Easy|Challenge)-(Train|Dev|Test)\.jsonl$", name)
        if m:
            out = "data/shelf/arc/%s_%s.tsv" % (m.group(1).lower(), m.group(2).lower()); n = 0
            with z.open(name) as f, open(out, "w") as o:
                for line in f:
                    r = json.loads(line); q = r["question"]; labels = [c["label"] for c in q["choices"]]
                    if r["answerKey"] not in labels: continue
                    clean = lambda s: " ".join(s.split())
                    o.write("\t".join([r["id"], clean(q["stem"]), str(len(labels))] + [clean(c["text"]) for c in q["choices"]] + [str(labels.index(r["answerKey"]))]) + "\n"); n += 1
            print(out, n, "questions")
        elif name.endswith("ARC_Corpus.txt") and not os.path.exists("data/shelf/arc_corpus.txt"):
            with z.open(name) as f, open("data/shelf/arc_corpus.txt", "wb") as o:
                for chunk in iter(lambda: f.read(1 << 20), b""): o.write(chunk)
            print("data/shelf/arc_corpus.txt", os.path.getsize("data/shelf/arc_corpus.txt"), "B")

# ---- Simple English Wikipedia as plain text
def strip_nested(s, a, b):
    out, depth, i = [], 0, 0
    while i < len(s):
        if s.startswith(a, i): depth += 1; i += len(a); continue
        if depth and s.startswith(b, i): depth -= 1; i += len(b); continue
        if not depth: out.append(s[i])
        i += 1
    return "".join(out)

def clean(t):
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t); t = re.sub(r"<ref.*?</ref>", "", t, flags=re.S)
    t = strip_nested(t, "{{", "}}"); t = strip_nested(t, "{|", "|}")
    t = re.sub(r"\[\[(?:File|Image|Category|Media):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", "", t, flags=re.I)
    t = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", t); t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", t); t = re.sub(r"\[https?://\S+\]", "", t)
    t = re.sub(r"<[^>]+>", "", t); t = t.replace("'''", "").replace("''", "")
    t = t.replace("&nbsp;", " ").replace("&amp;", "&").replace("&ndash;", "-").replace("&mdash;", "-")
    paras = []
    for line in t.split("\n"):
        line = " ".join(line.split())
        if not line or line[0] in "=*#:;|!{}" or line.startswith("__"): continue
        if len(line.split()) < 5 or line[-1] not in ".!?\"')": continue
        paras.append(line)
    return paras

if not os.path.exists("data/shelf/simplewiki.txt"):
    n = 0; nb = 0
    with bz2.open(SW, "rb") as f, open("data/shelf/simplewiki.txt.part", "w") as o:
        title = ns = None; redirect = False
        for ev, el in ET.iterparse(f, events=("end",)):
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "title": title = el.text
            elif tag == "ns": ns = el.text
            elif tag == "redirect": redirect = True
            elif tag == "text":
                if ns == "0" and not redirect and el.text:
                    paras = clean(el.text)
                    if paras: s = "\n".join(paras) + "\n\n"; o.write(s); n += 1; nb += len(s)
            elif tag == "page": title = ns = None; redirect = False; el.clear()
    os.rename("data/shelf/simplewiki.txt.part", "data/shelf/simplewiki.txt")
    print("data/shelf/simplewiki.txt", n, "articles", nb, "B")
