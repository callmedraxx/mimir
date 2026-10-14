# What it is reading now, apart from what it knows (episode)

Owner (2026-10-14): "how does a real human brain differentiate between general knowledge and
context associated with exactly what it is currently reading?" → "Yes" (write this up as the
next step).

## Why

With the cortex on, the clock story fell from 8/10 to 3/10 after reading it
(docs/CORTEX_LINK.md). The 43 facts of the story sit in one pool with 32.5M facts from the
shelf, with nothing saying they belong to what is being read now:

* a fact read once counts at half confidence (`conf = 1 − 0.5 / strength`), so the story's
  facts lose to shelf facts read thousands of times. In a person it is the other way round:
  what was just read is the easiest thing to recall;
* near-meaning matches add up: "cat" or "goose" for "owl", "Dumi" for "Imra";
* there is no "current story" to recall from first.

The same thing was seen before: adding background knowledge to passage questions hurt at
every weight tried (docs/past/KNOW_COMPARE.md, `bg know`).

## How a brain does it, and what is built

| in the brain | built |
|---|---|
| the hippocampus binds an episode to its context (this book, now) | **episode**: the facts of the text being read are kept apart from the shelf's, tagged with it, and with a pointer into it (open book works on it) |
| the situation being read stays active (prefrontal cortex); its people and things are already firing | an episode is **active** while its text is being read and asked about; a question whose words were met in it is about it |
| the story overrides world knowledge (a peanut in love: within a fraction of a second) | candidates come from the episode first; its best fit is weighted `EPW` over the shelf's; the shelf fills gaps ("Pell is an owl" + "owls are birds") |
| inhibition: context switches off the near meanings | inside an active episode, a near-meaning match counts only for a question word that no episode fact holds exactly |
| what was just read is the most accessible | an episode fact counts at full confidence (recency), not as "read once" |
| the cortex keeps learning while reading | the cortex learns from the episode's facts as before (CORTEX_LINK) |

Consolidation (the episode merged into the shelf's facts in "sleep") is not part of this step.

For measuring, each passage is its own episode and leaves nothing behind: its facts are
dropped and the cortex cells it changed are put back after its questions (`bg passages`),
so one passage cannot help another. With no episode active nothing changes: the five
general tests must give exactly the same answers (checked once on dev).

## Cells

The answering base is the one adopted by CORTEX_LINK (if it is not adopted, M0).

| cell | what |
|---|---|
| **P0** (now) | the passage read as facts into the one pool, answered as now |
| P1 | episode kept apart, active, candidates from it first; `EPW` chosen from 1, 2, 4 |
| P2 | P1 + inhibition of near meanings inside the episode |
| P3 | P2 + recency (episode facts at full confidence) |

Each cell adds to the one before; a step is kept only if it is more than 1 point better on
dev than the step before it (as LAYA_LEVEL did).

## Rules

* Dev only: MCTest MC160 train + dev (400 questions), MC500 train + dev (1,400), RACE-Middle
  dev (1,436); the clock story (10, shown, not counted). Choice: the best mean of the three.
* Then frozen (commit), and run **once** on: MCTest MC160 test (240), MC500 test (600),
  RACE-Middle test (1,436); **three new stories, 30 questions** (`data/story/episode_tests.tsv`:
  salt, train, kite), written before any code; the harbor, garden and snow stories
  (CORTEX_LINK's test, run once there, never developed on); and the five general tests,
  which must not move.
* Yardsticks already measured on these passage tests (docs/past/KNOW_COMPARE.md, before the
  school brain): the MCTest paper's sliding window: MC160 67.08, MC500 57.83, RACE-M 45.33;
  Mimir's facts with pointers, open book: 63.75, 52.83, 41.09.
* All runs in Colab, from the shared browser. Reported as they come; the journal updated;
  the losing cells' code deleted.

## How it was built (before any dev result)

* `bg passages BRAIN FILE...` reads each passage as the episode and answers its questions;
  dumps each option's shelf parts and episode parts, so `EPW` (1, 2, 4) is chosen offline
  from one run per cell (`tools/episode_colab.py`; the offline score reproduces `bg`'s).
* P0 "one pool": the passage's facts compete with the shelf's as candidates, ranked and
  scored the same way (they are not merged into the shelf's facts).
* Inhibition (P2), as competition: inside the episode, a word that the question or one of
  the options says exactly can only be matched exactly (so "cat" gets nothing from "Pell is
  an owl" when "owl" is itself an option).
* Each passage leaves nothing behind: the cortex's changed cells, its counts and its recent
  words are put back, words first met in the passage are let go, the reading clock is set
  back. Checked on the droplet: a passage set gives the same answers alone or after 70 others.
  With no episode the answers are unchanged (droplet check).
* A pipeline smoke run on the droplet's 3 MB test brain (Alice + War and Peace) went once
  through the new test stories and printed a score; nothing is chosen from it, and the
  tests run on the school brain.

## Dev results (2026-10-14, Colab; `results/episode_dev.txt`)

Means of MC160 (train + dev), MC500 (train + dev) and RACE-Middle dev; chance 25%:

| cell | EPW 1 | EPW 2 | EPW 4 |
|---|---|---|---|
| **P0** one pool (now) | 28.54 | | |
| P1 kept apart, active | 31.71 | 43.52 | **46.67** (MC160 53.90, MC500 48.14, RACE-M 37.97) |
| P2 + inhibition | 31.28 | 43.73 | 46.28 |
| P3 + recency | 34.54 | 44.91 | 46.31 |

**Chosen: P1, EPW 4 (+18.1 over P0).** P2 and P3 add nothing more than 1 point over P1,
so they are not kept. EPW 4 is the top of the grid and the score was still rising there.
Frozen in `tools/episode_choice.env`.

Clock story (shown, not counted): P0 5/10, **P1 EPW 4: 8/10** ("Pell is an owl" now right;
misses: "What does Pell eat?" → mice, the shelf's owls, not the story's beetles; the mayor).
General dev sets with no episode: exactly CORTEX_LINK's numbers (47.81, 20.01, 25.25,
29.32, 31.55), as required.

## Test results (2026-10-14, Colab, once, frozen 2deffed; `results/episode_test.txt`)

| test | **P0** one pool | **P1** episode, EPW 4 | change |
|---|---|---|---|
| MCTest MC160 test (240) | 25.31 | **50.00** | +24.7 better |
| MCTest MC500 test (600) | 28.18 | **49.19** | +21.0 better |
| RACE-Middle test (1,436) | 26.31 | **39.12** | +12.8 better |
| new stories: salt, train, kite (30) | 23.33 | **76.67** (23/30) | +53 better |
| harbor, garden, snow (30) | 46.67 | **90.00** (27/30) | +43 better |
| general tests, no episode | — | MMLU-HS 27.41, CSQA 31.37, ARC-E 46.61, ARC-C 21.22, OBQA 24.40 | **unchanged**, as required |

**Count: better 5, worse 0. Adopted.** Before the cortex (M0) the harbor/garden/snow stories
were 19/30; with the cortex and no episode 14/30; now 27/30.

Yardsticks (measured before the school brain, docs/past/KNOW_COMPARE.md): the MCTest paper's
sliding window MC160 67.08, MC500 57.83, RACE-M 45.33; Mimir's own passage-only reader,
open book, 63.75 / 52.83 / 41.09. **P1 is still below both**: the passage now wins over the
shelf, but answering a passage question is still done by the same association parts
(the "meaning" and "fit" parts weigh 4 and 1 and never look at the passage).

Misses in the new stories: "What is Firebird?" → a bird (the shelf), where the kite landed,
"Was Mrs. Vanek angry?" → "she cried" (negation: "She was not angry"), what she gave
(jam: "with a jar of plum jam" not read as given).

Losers deleted: P0 (one pool), P2 (inhibition), P3 (recency). `bg passages` keeps the
episode apart with EPW 4.
