#!/usr/bin/env python3
"""docs/SURVEY.md: the standard reading tests as Mimir passage files, data/understand/survey/<test>.txt
(S id / T text / Q id question option… letters). Sources: HuggingFace's rows API (no library needed), the DREAM
github, and the repo's own bAbI, RACE and SciQ files. SURVEY_N=<n> caps every test at its first n questions
(smoke tests); otherwise the sizes in docs/SURVEY.md. Run from the repo root; stdlib only."""
import json, os, random, re, sys, time, urllib.request, urllib.parse
OUT = "data/understand/survey"; os.makedirs(OUT, exist_ok=True)
CAP = int(os.environ.get("SURVEY_N", "0"))
def clean(s): return re.sub(r"\s+", " ", str(s).replace("\t", " ")).strip()
try: import pyarrow.parquet as pq
except ImportError: pq = None
def fetch(u, tries=8):
    """GET with a long wait on 429 (HuggingFace's rate limit)"""
    for attempt in range(tries):
        try: return urllib.request.urlopen(u, timeout=300).read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1: raise
            wait = int(e.headers.get("Retry-After") or 0) or 20 * (attempt + 1); print("  429, waiting %ds" % wait, flush=True); time.sleep(wait)
        except Exception:
            if attempt == tries - 1: raise
            time.sleep(5 * (attempt + 1))
def rows(ds, config, split, n):
    """the first n rows of a HuggingFace dataset split: its parquet files when pyarrow is there (one download), else the
    rows API (100 a call, rate-limited)"""
    n = min(n, CAP) if CAP else n; out = []
    if pq and not CAP:
        try:
            urls = json.loads(fetch("https://huggingface.co/api/datasets/%s/parquet/%s/%s" % (ds, config, split)))
            for u in urls:
                fn = "/tmp/survey_%s.parquet" % re.sub(r"\W", "_", u[-80:]); open(fn, "wb").write(fetch(u))
                out += pq.read_table(fn).to_pylist(); os.remove(fn)
                if len(out) >= n: break
            return out[:n]
        except Exception as e: print("  parquet failed (%s), rows API" % e, flush=True); out = []
    off = 0
    while len(out) < n:
        u = "https://datasets-server.huggingface.co/rows?dataset=%s&config=%s&split=%s&offset=%d&length=%d" % (urllib.parse.quote(ds, safe=""), config, split, off, min(100, n - len(out)))
        d = json.loads(fetch(u)); got = [r["row"] for r in d.get("rows", [])]
        if not got: break
        out += got; off += len(got)
        if off >= d.get("num_rows_total", off): break
    return out[:n]
class W:
    def __init__(self, name): self.name = name; self.f = open("%s/%s.txt" % (OUT, name), "w"); self.n = 0; self.np = 0
    def passage(self, text): self.np += 1; self.f.write("S\t%s.%d\nT\t%s\n" % (self.name, self.np, clean(text) or "-"))
    def q(self, qid, question, options, right):
        """right: an index or a list of indices into options"""
        options = [clean(o) or "-" for o in options]
        if isinstance(right, int): right = [right]
        letters = "".join(chr(65 + r) for r in sorted(set(right)) if 0 <= r < len(options))
        if len(options) < 2 or len(options) > 24 or not letters: return
        self.f.write("Q\t%s\t%s\t%s\t%s\n" % (qid, clean(question) or "_", "\t".join(options), letters)); self.n += 1
    def done(self): self.f.close(); print("%-14s %5d passages %6d questions" % (self.name, self.np, self.n), flush=True); return self.n
def cbt():
    for kind in ("NE", "CN", "V", "P"):
        w = W("cbt_%s" % kind.lower())
        for i, r in enumerate(rows("cam-cst/cbt", kind, "test", 1000)):
            w.passage(" ".join(r["sentences"])); opts = list(r["options"])
            if r["answer"] not in opts: continue
            w.q("cbt%s_%d" % (kind, i), r["question"].replace("XXXXX", "_"), opts, opts.index(r["answer"]))
        w.done()
STOP = set("the a an and or but of to in on at for with from by as is was were be been are am it its this that these those he she they we you i his her their our your my me him them us not no so if then than there here what which who whom when where why how all any some one two very just also into over under up down out about after before again more most such only own same too can could will would shall should may might must do does did done have has had having said says say".split())
def lambada():
    w = W("lambada")
    for i, r in enumerate(rows("EleutherAI/lambada_openai", "default", "test", 2000)):
        t = r["text"].strip(); m = re.search(r"(\S+)\s*$", t)
        if not m: continue
        target = re.sub(r"[^\w'-]", "", m.group(1)); ctx = t[:m.start()].rstrip()
        if not target: continue
        parts = re.split(r"(?<=[.!?])\s+", ctx); last = parts[-1] if parts else ""
        if len(parts) > 1 and len(last.split()) < 3: last = parts[-2] + " " + last
        words = re.findall(r"[A-Za-z][\w'-]+", ctx); freq = {}
        for x in words:
            if len(x) >= 3 and x.lower() not in STOP and x.lower() != target.lower(): freq[x] = freq.get(x, 0) + 1
        dis = sorted(freq, key=lambda x: (-freq[x], x))[:9]; opts = dis + [target]
        random.Random(i).shuffle(opts)
        w.passage(ctx); w.q("lambada_%d" % i, last + " _", opts, opts.index(target))
    w.done()
def storycloze():
    w = W("storycloze")
    for i, r in enumerate(rows("MoE-UNC/story_cloze", "default", "test", 1871)):
        w.passage(" ".join(r["input_sentence_%d" % k] for k in (1, 2, 3, 4)))
        w.q("cloze_%d" % i, "What happened next?", [r["sentence_quiz1"], r["sentence_quiz2"]], int(r["answer_right_ending"]) - 1)
    w.done()
def wsc():
    w = W("wsc273")
    for i, r in enumerate(rows("lighteval/winograd_wsc", "wsc273", "test", 273)):
        t = r["text"]; loc = int(r["pronoun_loc"]); pr = r["pronoun"]
        if t[loc:loc + len(pr)] != pr: loc = t.find(pr)
        w.passage(t); w.q("wsc_%d" % i, t[:loc] + "_" + t[loc + len(pr):], list(r["options"]), int(r["label"]))
    w.done()
def winogrande():
    w = W("winogrande")
    for i, r in enumerate(rows("allenai/winogrande", "winogrande_xl", "validation", 1267)):
        w.passage(r["sentence"]); w.q("wg_%d" % i, r["sentence"], [r["option1"], r["option2"]], int(r["answer"]) - 1)
    w.done()
def cosmos():
    w = W("cosmosqa"); last = None
    for i, r in enumerate(rows("Samsoup/cosmos_qa", "default", "validation", 2985)):
        if r["context"] != last: w.passage(r["context"]); last = r["context"]
        w.q("cosmos_%d" % i, r["question"], [r["answer%d" % k] for k in range(4)], int(r["label"]))
    w.done()
def dream():
    w = W("dream"); d = json.loads(fetch("https://raw.githubusercontent.com/nlpdata/dream/master/data/test.json"))
    n = 0
    for dlg, qs, did in d:
        text = " ".join(re.sub(r"^M:", "Man:", re.sub(r"^W:", "Woman:", re.sub(r"^F:", "Woman:", t.strip()))) for t in dlg)
        w.passage(text)
        for q in qs:
            if CAP and n >= CAP: break
            if q["answer"] in q["choice"]: w.q("dream_%d" % n, q["question"], q["choice"], q["choice"].index(q["answer"])); n += 1
        if CAP and n >= CAP: break
    w.done()
def boolq():
    w = W("boolq")
    for i, r in enumerate(rows("google/boolq", "default", "validation", 3270)):
        q = r["question"].strip(); q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
        w.passage(r["passage"]); w.q("boolq_%d" % i, q, ["yes", "no"], 0 if str(r["answer"]) in ("True", "true", "1") else 1)
    w.done()
def multirc():
    w = W("multirc"); groups = {}; order = []
    for r in rows("aps/super_glue", "multirc", "validation", 4848):
        k = (r["idx"]["paragraph"], r["idx"]["question"])
        if k not in groups: groups[k] = (r["paragraph"], r["question"], []); order.append(k)
        groups[k][2].append((r["answer"], int(r["label"])))
    lastp = None
    for k in order:
        para, q, ans = groups[k]
        if para != lastp: w.passage(para); lastp = para
        right = [i for i, (a, l) in enumerate(ans) if l]
        if right and len(ans) >= 2: w.q("multirc_%d_%d" % k, q, [a for a, l in ans], right)
    w.done()
def record():
    w = W("record")
    for i, r in enumerate(rows("aps/super_glue", "record", "validation", 2000)):
        ents = []
        for e in r["entities"]:
            if e not in ents: ents.append(e)
        right = [k for k, e in enumerate(ents) if e in r["answers"]]
        if not right: continue
        if len(ents) > 24: ents = [e for e in ents if e in r["answers"]] + [e for e in ents if e not in r["answers"]][:24 - len(right)]; right = [k for k, e in enumerate(ents) if e in r["answers"]]
        w.passage(r["passage"].replace("@highlight", ".")); w.q("record_%d" % i, r["query"].replace("@placeholder", "_"), ents, right)
    w.done()
def quality():
    w = W("quality"); last = None
    for i, r in enumerate(rows("tasksource/QuALITY", "default", "validation", 1000)):
        if r["article"] != last: w.passage(r["article"]); last = r["article"]
        w.q("quality_%d" % i, r["question"], list(r["options"]), int(r["gold_label"]) - 1)
    w.done()
def babi(split="test", tasks=range(1, 21), name="babi"):
    w = W(name); per = CAP if CAP else 200
    for task in tasks:
        fn = "data/understand/babi/qa%d_%s.txt" % (task, split)
        if not os.path.exists(fn): print("missing", fn); continue
        lines = open(fn).read().split("\n"); answers = sorted({l.split("\t")[1].strip() for l in lines if "\t" in l})
        story, n = [], 0
        for l in lines:
            if not l.strip(): continue
            num, rest = l.split(" ", 1)
            if num == "1": story = []
            if "\t" in rest:
                if n >= per: break
                q, a = rest.split("\t")[:2]; a = a.strip(); others = [x for x in answers if x != a]
                dis = random.Random(task * 10007 + n).sample(others, min(9, len(others))); opts = dis + [a]; random.Random(n).shuffle(opts)
                w.passage(" ".join(story) if story else "-"); w.q("babi%d_%d" % (task, n), q.strip(), opts, opts.index(a)); n += 1
            else: story.append(rest.strip())
    w.done()
def sciq():
    """SciQ test (Welbl et al. 2017): the questions that come with a support paragraph, as the repo's local copy"""
    w = W("sciq"); last = None
    for i, r in enumerate(rows("allenai/sciq", "default", "test", 1000)):
        if not clean(r.get("support", "")): continue
        if r["support"] != last: w.passage(r["support"]); last = r["support"]
        opts = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]; random.Random(i).shuffle(opts)
        w.q("sciq_%d" % i, r["question"], opts, opts.index(r["correct_answer"]))
    w.done()
def copy_local(name, src):
    if not os.path.exists(src): print("missing", src); return
    n = 0; out = open("%s/%s.txt" % (OUT, name), "w")
    for l in open(src):
        if l.startswith("Q"):
            if CAP and n >= CAP: continue
            n += 1
        out.write(l)
    out.close(); print("%-14s %6d questions (copied)" % (name, n), flush=True)
def boolq_train():
    """docs/YESNO.md dev set: the first 1,000 of BoolQ train, from the SuperGLUE release (the HF rows API gave 502s); the
    validation split is the survey's test and is not developed on"""
    import zipfile, io
    z = zipfile.ZipFile(io.BytesIO(fetch("https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip")))
    w = W("boolq_train")
    for i, l in enumerate(z.open("BoolQ/train.jsonl").read().decode().split("\n")):
        if i >= 1000 or not l.strip(): break
        r = json.loads(l); q = clean(r["question"]); q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
        w.passage(r["passage"]); w.q("boolqt_%d" % i, q, ["yes", "no"], 0 if r["label"] else 1)
    w.done()
ALL = [("wsc273", wsc), ("winogrande", winogrande), ("storycloze", storycloze), ("boolq", boolq), ("dream", dream), ("cosmosqa", cosmos), ("multirc", multirc), ("record", record), ("lambada", lambada), ("cbt", cbt), ("quality", quality), ("babi", babi)]
if __name__ == "__main__":
    want = sys.argv[1:] or [n for n, _ in ALL] + ["race_high", "sciq"]   # sciq: HuggingFace too (the repo's data/understand is not in git)
    if "race_high" in want: copy_local("race_high", "data/understand/race/test_high.txt")
    if "sciq" in want: sciq()
    if "babi_train" in want: babi("train", (1, 2, 3, 6, 9, 11, 13), "babi_train")   # docs/STATE.md dev set
    if "boolq_train" in want: boolq_train()   # docs/YESNO.md dev set
    failed = []
    for name, fn in ALL:
        if name not in want: continue
        try: fn()
        except Exception as e: print("%-14s FAILED: %s" % (name, e), flush=True); failed.append(name)
    if failed: print("failed:", " ".join(failed)); sys.exit(1)
