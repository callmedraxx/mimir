#!/usr/bin/env python3
"""docs/YESNO.md in Colab (free CPU runtime): Y0 diagnose, and the dev runs.
  !git clone -q -b reader https://github.com/callmedraxx/mimir /content/mimir && cd /content/mimir && python3 tools/yesno_colab.py diag
diag: the adopted reader on BoolQ train with every pick (SHOW=1) and the parts of both statements, episode apart (EPDUMP=1),
      to /content/yesno_diag.show and .dump.
dev <S5 masks...>: each mask on BoolQ train, bAbI train, the dev stories, MC160/MC500 dev, RACE-M dev, the general dev sets."""
import os, subprocess, sys
sys.path.insert(0, "tools"); import brain_release
def sh(cmd, log=None, env=None):
    print("$", cmd, flush=True); e = dict(os.environ); e.update(env or {})
    r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e); print(r.stdout[-2500:], flush=True)
    if log: open(log, "a").write("$ %s\n%s" % (cmd, r.stdout))
    return r
sh("gcc -O3 -march=native -std=gnu11 -D_GNU_SOURCE -w -o bg sandbox/brain/bg.c src/meaning.c src/reader.c -Isrc -lm -fopenmp -lpthread")
os.makedirs("data/understand", exist_ok=True)
if not os.path.exists("data/understand/wordclass.tsv"): sh("curl -sL https://github.com/callmedraxx/mimir/releases/download/brain-E/wordclass.tsv.gz | gunzip > data/understand/wordclass.tsv")
BRAIN = "/content/cx4.brain"
if not os.path.exists(BRAIN): brain_release.get(BRAIN)
if not os.path.exists("data/understand/survey/boolq_train.txt"): sh("python3 tools/survey_fetch.py boolq_train")
mode = sys.argv[1] if len(sys.argv) > 1 else "diag"
if mode == "test":
    """after the freeze, once: python3 tools/yesno_colab.py test <S5 frozen>. BoolQ validation, the 18 test stories (S5=0 too, so the
    yes/no questions can be scored apart), MC160 / MC500 / RACE-M tests, bAbI test, the five general tests -> /content/yesno_test.txt"""
    FROZEN = sys.argv[2]; OUT = "/content/yesno_test.txt"; open(OUT, "w").write("=== choice: S5=%s (frozen; docs/YESNO.md)\n" % FROZEN)
    if not os.path.exists("data/understand/mctest/mc500.test.txt"): sh("python3 -c \"import sys; sys.path.insert(0, 'tools'); import understand_fetch as u; u.mctest()\"")
    if not os.path.exists("data/understand/race/test_middle.txt"): sh("python3 tools/race_fetch.py")
    if not os.path.exists("data/understand/survey/babi.txt"): sh("python3 tools/survey_fetch.py babi")
    if not os.path.exists("data/understand/survey/boolq.txt"): sh("python3 tools/survey_fetch.py boolq")
    sh("python3 tools/stories.py /content/st")
    ST = " ".join("/content/st/%s.txt" % t for t in ("state_tests", "follow_tests", "read2_tests", "comprehend_tests", "episode_tests", "story_tests"))
    for label, cmd in [("S5=0 stories", "S5=0 SHOW=1 ./bg passages %s %s" % (BRAIN, ST)),
                       ("S5=%s stories + BoolQ validation + bAbI test + MCTest + RACE-M tests" % FROZEN, "S5=%s SHOW=1 ./bg passages %s %s data/understand/survey/boolq.txt data/understand/survey/babi.txt data/understand/mctest/mc160.test.txt data/understand/mctest/mc500.test.txt data/understand/race/test_middle.txt" % (FROZEN, BRAIN, ST)),
                       ("S5=%s general tests" % FROZEN, "S5=%s ./bg arc %s %s" % (FROZEN, BRAIN, " ".join("data/arc_tests/%s.tsv" % t for t in ("mmlu_hs_test", "csqa_dev", "easy_test", "challenge_test", "obqa_test"))))]:
        open(OUT, "a").write("=== %s\n" % label); r = sh(cmd + " 2>/dev/null", OUT)
        for l in r.stdout.split("\n"):
            if l.startswith("RESULT"): print("==> test S5", FROZEN, l, flush=True)
    print("done", flush=True); sys.exit(0)
if mode == "diag":
    sh("SHOW=1 EPDUMP=1 DUMP=/content/yesno_diag.dump ./bg passages %s data/understand/survey/boolq_train.txt > /content/yesno_diag.show 2>/dev/null; grep RESULT /content/yesno_diag.show" % BRAIN)
else:
    if not os.path.exists("data/understand/mctest/mc160.dev.txt"): sh("python3 tools/understand_fetch.py")
    if not os.path.exists("data/understand/race/dev_middle.txt"): sh("python3 tools/race_fetch.py")
    if not os.path.exists("data/understand/survey/babi_train.txt"): sh("python3 tools/survey_fetch.py babi_train")
    if not os.path.exists("data/understand/survey/babi_words_train.txt"): sh("python3 tools/babi_words.py train 1 2 3 6 9 11 13")   # docs/SPEAK.md: answered in words
    sh("python3 tools/stories.py /content/st"); sh("python3 tools/general_tests.py >/dev/null 2>&1; ls data/dev_sets")
    OUT = "/content/yesno_dev.txt"; general_done = set()
    for m in sys.argv[2:]:
        env = {"S5": m, "EPDUMP": "1", "DUMP": "/content/yesno_dev_%s.dump" % m}   # the parts of every option, episode apart (tools/yesno_notes.py)
        for f in ["data/understand/survey/boolq_train.txt", "data/understand/survey/babi_train.txt", "data/understand/survey/babi_words_train.txt", "/content/st/smart_tests.txt /content/st/dev_stories.txt /content/st/clock.txt /content/st/bakery.txt",
                  "data/understand/mctest/mc160.dev.txt data/understand/mctest/mc500.dev.txt data/understand/race/dev_middle.txt"]:
            r = sh("SHOW=1 ./bg passages %s %s 2>/dev/null" % (BRAIN, f), OUT, env)
            for l in r.stdout.split("\n"):
                if l.startswith("RESULT"): print("==> S5", m, l, flush=True); open(OUT, "a").write("==> S5 %s %s\n" % (m, l))
        g = int(m) & (4 | 64 | 128 | 1024)   # only Y3, Y7, Y8, Y11 read for everything: the general sets move with those bits alone
        if not g or g in general_done: continue
        general_done.add(g)
        r = sh("./bg arc %s data/dev_sets/easy_dev.tsv data/dev_sets/challenge_dev.tsv data/dev_sets/obqa_dev.tsv data/dev_sets/mmlu_hs_dev.tsv data/dev_sets/csqa_traindev.tsv 2>/dev/null | grep RESULT" % BRAIN, OUT, env)
        for l in r.stdout.split("\n"):
            if l.startswith("RESULT"): print("==> general S5", m, l, flush=True); open(OUT, "a").write("==> general S5 %s %s\n" % (m, l))
print("done", flush=True)
