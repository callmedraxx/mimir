#!/usr/bin/env python3
"""The school shelf (docs/past/SCHOOL_BRAIN.md): what a child reads and hears from baby to high school, as plain text,
one paragraph a line, a blank line between documents, in reading order:

  data/school/01_babylm.txt       BabyLM 100M: speech to and around children (CHILDES), children's stories, BNC spoken, subtitles, TED
  data/school/02_children.txt     Project Gutenberg children's books (bookshelves "Children's ..." and "Children & Young Adult Reading",
                                  from Gutenberg's catalogue pg_catalog.csv; texts from the manu/project_gutenberg mirror)
  data/school/03_wikijunior.txt   Wikijunior (children's encyclopedia books on Wikibooks)
  data/school/04_simplewiki.txt   Simple English Wikipedia
  data/school/05_textbooks.txt    OpenStax textbooks (school level), from the philschatz/* markdown mirrors
  data/school/06_wikibooks.txt    Wikibooks (open textbooks)
  data/school/07_vital.txt        Wikipedia Vital Articles, level 4 (wikimedia/wikipedia 20231101.en)

  python tools/school_shelf.py [KEEP_DIR]     run from the repo root; with KEEP_DIR (e.g. on Google Drive) each finished
                                              file is also kept there gzipped, and taken from there next time
Needs pyarrow (Colab has it). Each file is made once; a file being made is written as .part first.
"""
import bz2, glob, gzip, json, os, re, shutil, subprocess, sys, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET, zipfile

OUT, RAW = "data/school", "data/school/raw"
os.makedirs(RAW, exist_ok=True)
KEEP = sys.argv[1] if len(sys.argv) > 1 else None
UA = {"User-Agent": "mimir-school-shelf/1.0 (https://github.com/callmedraxx/mimir)"}

def get(url, path=None):
    for i in range(5):
        try:
            if path: subprocess.run(["curl", "-sfL", "-A", UA["User-Agent"], "-o", path, url], check=True); return path
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r: return r.read()
        except Exception as e:
            print("  retry", url, e, flush=True); time.sleep(5 * (i + 1))
    raise SystemExit("could not get " + url)

def stage(name, make):
    out = os.path.join(OUT, name)
    if os.path.exists(out): print(out, "there already", flush=True); return
    if KEEP and os.path.exists(os.path.join(KEEP, name + ".gz")):
        with gzip.open(os.path.join(KEEP, name + ".gz"), "rb") as f, open(out + ".part", "wb") as o: shutil.copyfileobj(f, o, 1 << 22)
        os.rename(out + ".part", out); print(out, "restored from", KEEP, flush=True); return
    t0 = time.time(); n = 0
    with open(out + ".part", "w") as o:
        for doc in make():
            doc = [p for p in doc if p]
            if doc: o.write("\n".join(doc) + "\n\n"); n += 1
    os.rename(out + ".part", out)
    print("%s: %d documents, %.0f MB, %.0f s" % (out, n, os.path.getsize(out) / 1e6, time.time() - t0), flush=True)
    if KEEP:
        os.makedirs(KEEP, exist_ok=True)
        with open(out, "rb") as f, gzip.open(os.path.join(KEEP, name + ".gz.part"), "wb", compresslevel=3) as o: shutil.copyfileobj(f, o, 1 << 22)
        os.replace(os.path.join(KEEP, name + ".gz.part"), os.path.join(KEEP, name + ".gz"))

ENDS = ".!?\"')”’"
def keep_para(p, minw=3):
    p = " ".join(p.split())
    return p if len(p.split()) >= minw and p[-1] in ENDS and not p.isupper() else ""

# ---- wikitext (as the old shelf_fetch.py, in git history)
def strip_nested(s, a, b):
    out, depth, i = [], 0, 0
    while i < len(s):
        if s.startswith(a, i): depth += 1; i += len(a); continue
        if depth and s.startswith(b, i): depth -= 1; i += len(b); continue
        if not depth: out.append(s[i])
        i += 1
    return "".join(out)

def wiki_clean(t):
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

def wiki_pages(dump, keep):
    with bz2.open(dump, "rb") as f:
        title = ns = None; redirect = False
        for ev, el in ET.iterparse(f, events=("end",)):
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "title": title = el.text
            elif tag == "ns": ns = el.text
            elif tag == "redirect": redirect = True
            elif tag == "text":
                if not redirect and el.text and keep(ns, title or ""): yield wiki_clean(el.text)
            elif tag == "page": title = ns = None; redirect = False; el.clear()

# ---- 1. BabyLM
def babylm():
    import pyarrow.parquet as pq
    for i in range(2):
        p = get("https://huggingface.co/datasets/nilq/babylm-100M/resolve/main/data/train-%05d-of-00002.parquet" % i, os.path.join(RAW, "babylm%d.parquet" % i))
        f = pq.ParquetFile(p); block = []
        for g in range(f.num_row_groups):
            for line in f.read_row_group(g, columns=["text"]).column("text").to_pylist():
                line = " ".join((line or "").split())
                if line: block.append(line)
                if len(block) >= 200: yield block; block = []
        if block: yield block
        os.remove(p)

# ---- 2. Gutenberg children's books
def gutenberg_ids():
    """Project Gutenberg's own catalogue (one file): English texts on a "Children's ..." or "Children & Young Adult Reading" shelf"""
    import csv, io
    ids = {}
    for b in csv.DictReader(io.StringIO(get("https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv").decode("utf-8", "replace"))):
        shelves = [s.strip() for s in (b.get("Bookshelves") or "").split(";")]
        if b["Language"] == "en" and b["Type"] == "Text" and any(s.startswith("Children") or s == "Category: Children & Young Adult Reading" for s in shelves): ids[b["Text#"]] = b["Title"]
    print("  gutenberg children's books:", len(ids), flush=True); return ids

def gutenberg_text(t):
    a = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", t); b = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", t)
    t = t[a.end() if a else 0: b.start() if b else len(t)]
    return [keep_para(p) for p in re.split(r"\n\s*\n", t.replace("\r", ""))]

def children():
    import pyarrow.parquet as pq
    ids = gutenberg_ids(); seen = set()
    files = [s["rfilename"] for s in json.loads(get("https://huggingface.co/api/datasets/manu/project_gutenberg"))["siblings"] if s["rfilename"].startswith("data/en-")]
    for fn in sorted(files):
        p = get("https://huggingface.co/datasets/manu/project_gutenberg/resolve/main/" + fn, os.path.join(RAW, "pg.parquet"))
        f = pq.ParquetFile(p)
        for g in range(f.num_row_groups):
            rg = f.read_row_group(g, columns=["id", "text"])
            for i, t in zip(rg.column("id").to_pylist(), rg.column("text").to_pylist()):
                k = str(i).split("-")[0]
                if k in ids and k not in seen: seen.add(k); yield gutenberg_text(t)
        os.remove(p)
    print("  gutenberg children's books found in the mirror:", len(seen), flush=True)

# ---- 3, 6. Wikibooks (Wikijunior is namespace 110)
def wikibooks_dump():
    p = os.path.join(RAW, "wikibooks.xml.bz2")
    if not os.path.exists(p): get("https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2", p)
    return p
def wikijunior(): yield from wiki_pages(wikibooks_dump(), lambda ns, title: ns == "110")
def wikibooks(): yield from wiki_pages(wikibooks_dump(), lambda ns, title: ns == "0")

# ---- 4. Simple English Wikipedia
def simplewiki():
    p = os.path.join(RAW, "simplewiki.xml.bz2")
    if not os.path.exists(p): get("https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2", p)
    yield from wiki_pages(p, lambda ns, title: ns == "0")
    os.remove(p)

# ---- 5. OpenStax textbooks (school level), markdown mirrors
BOOKS = ["biology-concepts-book", "biology-book", "anatomy-book", "chemistry-book", "physics-book", "astronomy-book", "us-history-book",
         "economics-book", "psychology-book", "sociology-book", "statistics-book", "algebra-trigonometry-book", "precalculus-book"]
def md_paras(t):
    t = re.sub(r"\A---.*?---", "", t, flags=re.S)
    t = re.sub(r"<table.*?</table>", "", t, flags=re.S); t = re.sub(r"<figure.*?</figure>", "", t, flags=re.S)
    t = re.sub(r"\{:[^}]*\}", "", t); t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t); t = re.sub(r"\[\\?\[?link\\?\]?\]\([^)]*\)", "", t); t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = t.replace("**", "").replace("__", "").replace("\\", "")
    out = []
    for p in re.split(r"\n\s*\n", t):
        p = p.strip()
        if p.startswith(("#", "|", "* ", "- ")) or re.match(r"\d+\. ", p): continue   # headings, tables, lists
        out.append(keep_para(p, 5))
    return out
def textbooks():
    for book in BOOKS:
        z = get("https://github.com/philschatz/%s/archive/refs/heads/master.zip" % book, os.path.join(RAW, "book.zip"))
        with zipfile.ZipFile(z) as zf:
            names = {n.split("/contents/")[-1]: n for n in zf.namelist() if "/contents/" in n and n.endswith(".md")}
            summ = next((n for n in zf.namelist() if n.endswith("/SUMMARY.md")), None)
            order = re.findall(r"contents/(m\d+\.md)", zf.read(summ).decode()) if summ else sorted(names)
            n = 0
            for m in order:
                if m in names: yield md_paras(zf.read(names[m]).decode("utf-8", "replace")); n += 1
        print("  %s: %d sections" % (book, n), flush=True)

# ---- 7. Vital Articles, level 4
def vital_titles():
    api = "https://en.wikipedia.org/w/api.php?action=parse&format=json&prop=links&redirects=1&page="
    todo, done, titles = ["Wikipedia:Vital articles/Level 4"], set(), set()
    while todo:
        page = todo.pop()
        if page in done: continue
        done.add(page)
        try: links = json.loads(get(api + urllib.parse.quote(page)))["parse"]["links"]
        except KeyError: continue
        for l in links:
            t = l["*"]
            if l["ns"] == 0 and "missing" not in l: titles.add(t)     # (the API no longer marks links "exists"; titles not in the dump never match)
            elif l["ns"] == 4 and t.startswith(("Wikipedia:Vital articles/Level 4/", "Wikipedia:Vital articles/Level/4/")): todo.append(t)
    print("  vital articles, level 4:", len(titles), "from", len(done), "list pages", flush=True); return titles
def vital():
    import pyarrow.parquet as pq
    titles = vital_titles(); found = 0
    for i in range(41):
        p = get("https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-%05d-of-00041.parquet" % i, os.path.join(RAW, "enwiki.parquet"))
        f = pq.ParquetFile(p)
        for g in range(f.num_row_groups):
            rg = f.read_row_group(g, columns=["title", "text"])
            for ti, text in zip(rg.column("title").to_pylist(), rg.column("text").to_pylist()):
                if ti in titles:
                    found += 1; yield [p for p in (" ".join(x.split()) for x in text.split("\n")) if len(p.split()) >= 5 and p[-1] in ".!?\"')"]
        os.remove(p)
    print("  vital articles found:", found, flush=True)

for name, make in [("01_babylm.txt", babylm), ("02_children.txt", children), ("03_wikijunior.txt", wikijunior), ("04_simplewiki.txt", simplewiki),
                   ("05_textbooks.txt", textbooks), ("06_wikibooks.txt", wikibooks), ("07_vital.txt", vital)]:
    stage(name, make)
for p in glob.glob(os.path.join(RAW, "*")): os.remove(p)
empty = [p for p in sorted(glob.glob(os.path.join(OUT, "0*.txt"))) if os.path.getsize(p) == 0]
if empty: raise SystemExit("empty shelf files (a source failed): %s" % ", ".join(empty))
print("shelf ready:", sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, "0*.txt"))) / 1e6, "MB", flush=True)
