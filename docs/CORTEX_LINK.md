# One brain: the cortex learns from the facts as Mimir reads (cortex v1)

Owner (2026-10-14): "we are trying to make it one … the cortex doesn't need to read the books
itself it should take from the facts learnt by the hippocampus and this should be happening
live as it's reading. it should be a continous process. so as it's reading a story and
storing facts the cortex does its [thing] just using k3b … combination of both will be used to
pick an answer".

## What is built

**Reading.** Every sentence is read into facts as before (the fact memory, "hippocampus",
`bg read`). Each time a fact is read, new or met again, its words (verb, heads, modifiers) go
to the cortex. The cortex works **at the same time, in its own thread**, and learns from each
fact with the K3b rule (docs/past/CORTEX.md v0.1): reward for every two words that stand in the
same fact, penalty for a few random words. The fact is the window. Nothing else is read by
the cortex: no book text, only facts. A new word (a name in a story) gets cells the moment
it is first met.

Saved beside the brain (`BRAIN.cx`). Reading on from a brain (`FROM=`) carries on with its
cortex, so a story read later changes the cortex too.

**Answering.** Answering already compares words by meaning, using an old fixed table
(`data/meaning/meaning.mng`, 29.7k words, learned once in 2026-10 and never changed by
reading). With `CX=1` the brain's own cortex replaces that table everywhere it is used:
* finding: how well a fact covers the question and the option (joint, open book, chain);
* judging: how well the question's fact matches a known fact (fact);
* the option's meaning against the question's (mean);
* new, **fit**: how well the option fits the question as the cortex expects (the K3
  fill-the-blank score: the option's meaning cells against the question words' context
  cells), weight BW[5].

So the answer is chosen from both: the facts the hippocampus holds, and the cortex's
understanding of what the words mean and what fits.

## Settings fixed now (K3b's)

D = 300 cells a side, NEG = 5 penalties per reward, rate 0.025, frequent words thinned
(1e-4), the brain's own words (no 50k cut). Words met fewer than 5 times are matched only
exactly when answering (their cells have not learned enough).

## Cells

**Learning (word level, dev only).** A dev brain from 1/8 of the shelf (`PART=8`), read twice:

| cell | the rate as it reads |
|---|---|
| L1 | falls with the reading clock, like K3b, to 5% of the start by the shelf's end, then stays there |
| L2 | each word's own: falls as that word is met more often, 1/√(1 + n/10,000), not below 5% (a new word learns fast, a known one slowly) |

Choice: the higher mean of WordSim-353 and SimVerb-3500 (Spearman). Then the whole shelf is
read once with it, file by file as the released school brain was (each file reading on from
the last; the cortex's reading clock carried across, `CXPOS`), giving the school brain with
its cortex. The facts must come out the same as the released school brain (32,489,775); the
cortex does not change what is stored (checked on the droplet with a small shelf: the brain
file is byte-identical with the cortex on or off). So M0's test numbers are the school tests
already run (`results/school_tests.txt`).

**Answering (the full brain, the five dev sets of docs/past/RANKING_COMPARE.md).**

| cell | words compared by | fit part |
|---|---|---|
| **M0** (now) | the old fixed table | — |
| C1a | the cortex, a link counted from cosine 0.25 (as now) | — |
| C1b | the cortex, from cosine 0.40 | — |
| C2 | the better C1 | weight chosen from 0.25, 0.5, 1, 2 |

Everything else as in the school tests: `NOPEN=20 POLARITY=1 STRENGTH=1`, NCAND 200.

## Rules

* Developed on dev only: ARC-Easy dev, ARC-Challenge dev, OpenBookQA dev, MMLU high school
  validation, CommonsenseQA 1,000 from train (`data/dev_sets/`); WordSim-353, SimVerb-3500;
  the clock story (`data/story/clock*`, 10 questions, already seen, so it is only shown, not
  counted in the choice).
* Choice: the best mean over the five dev sets. Adopted only if more than 1 point above M0's;
  otherwise M0 stays and that is the result.
* Then frozen (commit) and run **once** on: the five tests (MMLU high school, CommonsenseQA
  dev, ARC-Easy, ARC-Challenge, OpenBookQA); **three new stories with 30 questions** written
  before any of this was built (`data/story/story_tests.tsv`, the harbor, garden and snow
  stories; asked before and after reading each story); SimLex-999 and MEN-3000 for the
  cortex learned from facts (against K3b learned from text: 0.464 / 0.739, and word2vec:
  0.398 / 0.717).
* All runs in Colab (`tools/link_colab.py`, `tools/link.ipynb`), from the shared browser, dev and
  test in one session (the brain with its cortex stays in the session). Reported as they come.
* Story test: each story read into a copy of the brain (the cortex reading on, a new name
  getting cells as it is first met), its questions asked before and after, with M0 and with
  the choice. If M0 stays, only SimLex-999 and MEN are run (they measure the cortex itself).

## Note during dev (2026-10-14, before any answering result)

The whole shelf, read with the cortex on (L2 chosen: WordSim 0.560 / SimVerb 0.208 against
L1's 0.448 / 0.202), gave **32,488,986 facts, not the release's 32,489,775** (789 fewer,
0.002%). The read is the same code file by file, so the shelf rebuilt today differs slightly
from the one the release read (Gutenberg mirror and Wikipedia lists change). So M0's test
numbers are **not** taken from the school tests: at test time M0 and the choice are both run
once on this same brain.

## Dev results (2026-10-14, Colab; `results/link_dev.txt`)

Learning: **L2** chosen (WordSim-353 0.560, SimVerb 0.208 on 1/8 of the shelf; L1 0.448 / 0.202).
The cortex from the whole shelf's facts: WordSim-353 **0.649**, SimVerb **0.266** (K3b from text,
on its 50M-word dev run: 0.504 / 0.251). Nearest words: owl → raven, squirrel, goose; dog →
hare, deer, wolf; water → rainwater, aquifer, groundwater.

Answering, dev sets (the new brain gives M0 exactly the ranking dev's G0 numbers):

| cell | ARC-Easy | ARC-Chall. | OBQA | MMLU-HS | CSQA | **mean** |
|---|---|---|---|---|---|---|
| **M0** (old table) | 47.54 | 20.18 | 18.35 | 29.68 | 26.05 | **28.36** |
| C1a (cortex, 0.25) | 44.65 | 20.01 | 22.05 | 28.99 | 27.65 | 28.67 |
| C1b (cortex, 0.40) | 44.30 | 21.35 | 22.85 | 31.62 | 27.55 | 29.53 |
| C2 fit 0.25 | 46.23 | 18.67 | 23.25 | 31.49 | 28.95 | 29.72 |
| C2 fit 0.5 | 47.63 | 18.67 | 23.45 | 30.68 | 29.95 | 30.08 |
| **C2 fit 1** | 47.81 | 20.01 | 25.25 | 29.32 | 31.55 | **30.79** |
| C2 fit 2 | 46.58 | 20.51 | 25.35 | 28.65 | 32.05 | 30.63 |

**Chosen: C2, fit 1 (`CX=1 CXTH=0.40 CXFIT=1`), +2.43 over M0** (more than 1 point: adopted).
Frozen in `tools/link_choice.env`.

**Warning from the clock story** (dev, shown, not counted): with the cortex (C1b, no fit)
3/10 after reading, against M0's 8/10. Matching by meaning lets near words (another
animal, another person) count as much as the story's exact words, so the story's own facts
lose to the shelf's. The test stories will show whether this holds for the choice.

## Test results (2026-10-14, Colab, once, frozen e440cbe; `results/link_test.txt`)

The cortex learned from the facts, on word tests (against K3b from text, docs/past/CORTEX.md,
and word2vec):

| | SimLex-999 | MEN-3000 |
|---|---|---|
| **cortex from facts** | 0.381 | **0.738** |
| K3b from text | 0.464 | 0.739 |
| word2vec | 0.398 | 0.717 |

Relatedness as good as from text; likeness ("same kind") lower: facts tie words by their
roles (owl, hunt, mouse), which says more about what goes together than what is alike.

The five tests, the same brain (32,488,986 facts), M0 and the choice each run once:

| test | **M0** (old table) | **chosen** (cortex 0.40, fit 1) | change |
|---|---|---|---|
| MMLU high school (3,420) | 27.85 | 27.41 | −0.4 tie |
| CommonsenseQA dev (1,221) | 25.50 | **31.37** | **+5.9 better** |
| ARC-Easy (2,376) | 45.52 | **46.61** | +1.1 better |
| ARC-Challenge (1,172) | 21.35 | 21.22 | −0.1 tie |
| OpenBookQA (500) | 20.67 | **24.40** | **+3.7 better** |
| mean | 28.18 | **30.20** | **+2.0** |

M0 on this brain gives the released brain's school-test numbers within 0.1.

Stories (CORTEX_LINK's three new ones), right after reading each:

| story | M0 before → after | chosen before → after |
|---|---|---|
| harbor | 3 → 5 | 2 → 3 |
| garden | 3 → 6 | 4 → 5 |
| snow | 2 → 8 | 3 → 6 |
| **all (30)** | 8 → **19** | 9 → **14** |

**Count: better 3, tie 2, worse 0 on the general tests; the stories worse (19 → 14).**
The cortex is adopted (the rule: dev). What it shows:

* Understanding words helps most where Mimir knew least: everyday knowledge
  (CommonsenseQA +5.9) and OpenBookQA (+3.7), both above their earlier best.
* What it just read loses to what it knows: the stories fall, as the clock story warned.
  Next: docs/EPISODE.md.

Losers deleted: the old meaning table as word meanings (kept only for its list of glue
words), L1 (the rate falling with the reading clock, `CXSCHED`, `CXPOS`), the 0.25 cosine.
The cortex is now always on.
