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

## Results

(to be filled in)
