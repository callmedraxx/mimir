# docs/CORTEX.md in Colab: the school shelf rebuilt, the vocabulary and fill-the-blank items made from it, then
#   %run /content/mimir/tools/cortex_colab.py dev      every cell of tools/cortex_grid.txt on the dev sets (MAXTOK words each)
#   %run /content/mimir/tools/cortex_colab.py test     the frozen choice (tools/cortex_final.txt) on the whole shelf, tests once
#   %run /content/mimir/tools/cortex_colab.py k3       v0.1: tools/cortex_grid3.txt on the dev sets, the best (mean of the three dev
#                                                       numbers) picked here, then trained on the whole shelf and tested once
# Results are printed and sent to the browser's computer (cortex_dev.txt / cortex_test.txt).
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from google.colab import files
def sh(cmd, quiet=False):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1); out = []
    for line in p.stdout:
        out.append(line)
        if not quiet: print(line, end="", flush=True)
    if p.wait(): print("".join(out[-20:])); raise SystemExit("failed: " + cmd)
    return "".join(out)
mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
os.chdir("/content")
if not os.path.isdir("mimir"): sh("git clone -q --depth 1 -b reader https://github.com/callmedraxx/mimir")
os.chdir("/content/mimir"); sh("git pull -q && git log --oneline -1 && nproc && free -g | head -2")
sh("gcc -O3 -march=native -fopenmp -w -o cortex sandbox/cortex/cortex.c -lm && pip -q install gensim 2>&1 | tail -1")
if not os.path.exists("data/school/07_vital.txt"): sh("python3 tools/school_shelf.py && rm -rf data/school/raw")
SHELF = " ".join(sorted("data/school/" + f for f in os.listdir("data/school") if f.endswith(".txt")))
W = "/content/cx"; os.makedirs(W, exist_ok=True)
if not os.path.exists(W + "/vocab.tsv"): sh("./cortex vocab %s/vocab.tsv %s" % (W, SHELF))
if not os.path.exists(W + "/t2_test.tsv"): sh("./cortex items %s/vocab.tsv %s/t2_dev.tsv %s/t2_test.tsv %s" % (W, W, W, SHELF))
D = "data/cortex/wordsim/"
import re
def mean_of(txt):
    ws = re.search(r"EN-WS-353-ALL.txt.*spearman (-?[0-9.]+)", txt); sv = re.search(r"EN-SimVerb-3500.txt.*spearman (-?[0-9.]+)", txt); t2 = re.search(r"T2 fill the blank.*hit@10 ([0-9.]+)%", txt)
    return (float(ws.group(1)) + float(sv.group(1)) + float(t2.group(1)) / 100) / 3
if mode == "k3":
    rows = [l.split(None, 2) for l in open("tools/cortex_grid3.txt") if l.strip() and not l.startswith("#")]
    evd = 'WS="%sEN-WS-353-ALL.txt %sEN-SimVerb-3500.txt" T2=%s/t2_dev.tsv STORIES=data/cortex/stories.tsv SPLIT=dev' % (D, D, W)
    evt = 'WS="%sEN-SIMLEX-999.txt %sEN-MEN-TR-3k.txt" T2=%s/t2_test.tsv STORIES=data/cortex/stories.tsv SPLIT=test' % (D, D, W)
    log = []
    for name, cell, env in ((r[0], r[1], r[2].strip()) for r in rows):   # one at a time: each uses both cores
        m = "%s/%s.cx" % (W, name)
        t = sh("%s MAXTOK=50000000 ./cortex train %s %s/vocab.tsv %s %s 2>&1 | grep RESULT" % (env, cell, W, m, SHELF), quiet=True)
        t += sh("%s %s ./cortex eval %s %s/vocab.tsv; ./cortex near %s %s/vocab.tsv owl king water france blue" % (env, evd, m, W, m, W), quiet=True)
        t = "=== %s %s %s (dev)\n%s" % (name, cell, env, t); print(t, flush=True); log.append((mean_of(t), name, cell, env, t))
    best = max(log)
    pick = "chosen by the mean of the three dev numbers: %s (%.4f); others: %s\n" % (best[1], best[0], ", ".join("%s %.4f" % (x[1], x[0]) for x in log))
    print(pick, flush=True)
    m = "%s/%s_full.cx" % (W, best[1])
    t = sh("%s MAXTOK=0 ./cortex train %s %s/vocab.tsv %s %s 2>&1 | grep RESULT" % (best[3], best[2], W, m, SHELF), quiet=True)
    t += sh("%s %s ./cortex eval %s %s/vocab.tsv; ./cortex near %s %s/vocab.tsv owl king water france blue" % (best[3], evt, m, W, m, W), quiet=True)
    t = "=== %s %s %s (whole shelf, TESTS)\n%s" % (best[1], best[2], best[3], t); print(t, flush=True)
    out = "/content/cortex_k3.txt"; open(out, "w").write("".join(x[4] for x in log) + pick + t)
    files.download(out); print("cortex k3 done", flush=True); raise SystemExit
if mode == "dev":
    rows = [l.split(None, 2) for l in open("tools/cortex_grid.txt") if l.strip() and not l.startswith("#")]
    ev = 'WS="%sEN-WS-353-ALL.txt %sEN-SimVerb-3500.txt" T2=%s/t2_dev.tsv STORIES=data/cortex/stories.tsv SPLIT=dev' % (D, D, W)
    maxtok = os.environ.get("MAXTOK", "50000000"); out = "/content/cortex_dev.txt"
else:
    rows = [l.split(None, 2) for l in open("tools/cortex_final.txt") if l.strip() and not l.startswith("#")]
    ev = 'WS="%sEN-SIMLEX-999.txt %sEN-MEN-TR-3k.txt" T2=%s/t2_test.tsv STORIES=data/cortex/stories.tsv SPLIT=test' % (D, D, W)
    maxtok = "0"; out = "/content/cortex_test.txt"
def run(r):
    name, cell, env = r[0], r[1], r[2].strip() if len(r) > 2 else ""
    if cell == "B":   # word2vec reads words, the cortex counts replayed (kept) words: B gets the same lines by reading 1.7x as many
        txt = sh("%s %s MAXTOK=%d python3 tools/cortex_w2v.py %s/vocab.tsv %s" % (env, ev, int(int(maxtok) * 1.7), W, SHELF), quiet=True)
    else:
        m = "%s/%s.cx" % (W, name)
        txt = sh("%s MAXTOK=%s ./cortex train %s %s/vocab.tsv %s %s 2>&1 | grep RESULT" % (env, maxtok, cell, W, m, SHELF), quiet=True)
        txt += sh("%s %s ./cortex eval %s %s/vocab.tsv; ./cortex near %s %s/vocab.tsv owl king water france blue" % (env, ev, m, W, m, W), quiet=True)
    txt = "=== %s %s %s\n%s" % (name, cell, env, txt); print(txt, flush=True); return txt
with ThreadPoolExecutor(2) as ex: res = list(ex.map(run, rows))
open(out, "w").write("".join(res))
files.download(out)
print("cortex", mode, "done", flush=True)
