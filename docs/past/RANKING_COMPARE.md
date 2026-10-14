# Finding the right facts in the school brain (candidate ranking)

Owner (2026-10-14): "okay proceed" — the next step after the school brain became the base
(SCHOOL_BRAIN.md, VERSUS.md).

## Why

With 7× more facts (32.5M against brain E's 4.7M), the school brain re-reads better (open
book alone, ARC-Easy test: 47.3 against 45.9) but finds worse: the one-fact part (joint)
fell from 42.1 to 33.8. The right facts are not reaching the 200 candidates looked at per
option.

How candidates are found now (`bg_option`): for each of the option's words, its list of
facts (strongest first) is scanned, **only the first 200,000**, and the 200 facts holding the
most (rarest) question words are kept. In the school brain common words ("water", "bank",
"energy") hold far more than 200,000 facts, so a weaker but exact fact ("water boils at …")
can be cut before it is ever scored.

## What is changed (answering only; the brain is not touched)

| switch | what |
|---|---|
| `CANDJOIN=1` | when an option word's list is cut, the facts that hold it **and** a question word are found in full: from the question word's list when that is shorter, else from the rest of the option word's (at most 4M looked at per pair) |
| `NCAND=n` | candidates kept per option (now 200) |

Without them the answers are byte-identical to before (checked on brain E, 200 ARC-Easy
dev questions).

## Cells

Brain: the released school brain (`brain-school`, checked by SHA-256), shelf rebuilt in
Colab by `tools/school_shelf.py` at the same paths. Answering settings as in the school
tests: `NOPEN=20 POLARITY=1 STRENGTH=1`.

| cell | NCAND | CANDJOIN |
|---|---|---|
| **G0** (now) | 200 | 0 |
| G1 | 1000 | 0 |
| J0 | 200 | 1 |
| J1 | 1000 | 1 |

## Rules

* Developed on dev sets only: ARC-Easy dev (570), ARC-Challenge dev (299), OpenBookQA dev
  (500), **MMLU high school validation** (370, every `high_school_*` subject), **CommonsenseQA
  1,000 from its training split** (seed 12345). Built by `tools/general_tests.py`; kept in
  `data/dev_sets/`. The tests (`data/arc_tests/`) are not looked at.
* Choice: the cell with the best mean over the five dev sets. It is adopted only if that
  mean is more than 1 point above G0's; otherwise G0 stays and that is the result.
* Then frozen (commit), and the chosen cell run **once** on the five tests (MMLU high school,
  CommonsenseQA dev, ARC-Easy, ARC-Challenge, OpenBookQA). G0's test numbers are the
  school tests already run (`results/school_tests.txt`, same settings; answering is
  deterministic). Reported as they come out.
* Run in Colab (`tools/ranking_colab.py`), from the shared browser.

## Results (2026-10-14, Colab, dev sets only)

The shelf rebuilt in Colab came out at the original sizes, file for file (BabyLM 535 MB,
children's books 936 MB, …, 2.2 GB in all).

| cell | ARC-Easy dev | ARC-Challenge dev | OpenBookQA dev | MMLU-HS validation | CSQA train 1,000 | **mean** | ms/q |
|---|---|---|---|---|---|---|---|
| **G0** (now) | 47.54 | 20.18 | 18.35 | 29.68 | 26.05 | **28.36** | 175–245 |
| G1 (1000) | 43.86 | 19.68 | 17.05 | 28.40 | 26.10 | 27.02 | 450–760 |
| J0 (join) | 47.54 | 20.18 | 18.35 | 29.41 | 25.85 | 28.27 | 145–245 |
| J1 (both) | 43.86 | 19.68 | 16.85 | 28.40 | 26.10 | 26.98 | |

**No cell beats G0; G0 stays, and the tests were not run** (the rule: only a dev winner
is tested).

* **The 200,000 cut was not the problem**: finding every fact that holds an option word
  and a question word (J0) changes almost nothing. The right facts were already reachable.
* **Looking at more facts makes it worse** (G1, J1: −1.3 to −1.4 on the mean, −3.7 on
  ARC-Easy). The parts that take the best fact per option (joint, fact) get noisier: the
  more facts looked at, the likelier a wrong option also finds one that fits it well.
  With 32.5M facts that already happens at 200; this, not reach, is why joint fell
  from 42.1 (brain E) to 33.8.
* So the next step is not finding more but **telling a fact that fits by chance from one
  that answers the question**: e.g. judging an option's best fact against what a random
  option of the same commonness would find.

Results: `results/ranking_dev.txt`.
