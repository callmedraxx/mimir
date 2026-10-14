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
