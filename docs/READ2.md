# Reading the shelf without losing what it knew, and the reading misses left (read2)

Owner (2026-10-15): "ok proceed" (after COMPREHEND: find why ARC-Easy fell, then the misses left).

## Why

COMPREHEND (docs/COMPREHEND.md) made passages better (MC160 +5.1, new stories 16 → 21 of 30) but,
with the shelf re-read by its reader (`brain-school-cx2`), ARC-Easy fell 46.61 → 43.04 and
MMLU-HS 27.41 → 26.32 (ARC-Challenge and OpenBookQA rose). On dev, reading the questions the new way
with the old shelf left the general dev sets unchanged, so the loss most likely comes from what
the new reader made of the 2.2 GB shelf. Which rule did it is not known yet.

And the reading misses left in COMPREHEND's stories: "was no longer sick", a name said at the end of
a sentence ("He is a carpenter, Mr. Eze."), "made of red paper and bamboo" (two things), "he" /
"she" pointing at the right person (a man and a woman in one story).

## Steps

**0. Measure it properly first (found after this page was written).** COMPREHEND's general run had
no shelf on disk, so open book scored nothing (docs/COMPREHEND.md, Correction). The release now
carries its shelf. The five general **tests** are run once on `brain-school-cx2` with the shelf (the
measurement COMPREHEND should have had; no choice is made from it), and the general dev sets
(before COMPREHEND, with the shelf: 47.81 / 20.01 / 25.25 / 29.32 / 31.55). If ARC-Easy is then within 1 point of 46.61, there is no loss
to find: step 1 and S1 are dropped and S2 is built over S0.

**1. Find it (dev only).** The five general dev sets on `brain-school-cx` (old reader) and
`brain-school-cx2` (new), each option's parts dumped; for the dev questions whose answer changed,
the facts and shelf sentences each brain answered from (`bg serve`). What the new reader does to
those sentences says which rule hurts. Written up here before any change.

**2. The cells.**

| cell | what | brain |
|---|---|---|
| **S0** (now) | COMPREHEND as adopted | `brain-school-cx2` |
| S1 | the rule(s) step 1 finds harmful are changed or kept to passages (not the shelf); at most three versions, each built | rebuilt by GitHub Actions, one release per version (deleted after) |
| S2 | S1 + the misses: "no longer" as "not"; a name at the end of a sentence after a comma ("…, Mr. Eze.") names the phrase before it; "made of X and Y" keeps both; "he" / "she" to the latest man / woman (a person's sex from the pronouns used for them and from words like brother, mother, grandfather) | S1's brain re-read with S2's reader |

## Rules

* **Dev**: the five general dev sets (ARC-Easy and ARC-Challenge dev, OpenBookQA dev, MMLU-HS
  validation, 1,000 CommonsenseQA training questions); MCTest MC160 train + dev, MC500 train + dev,
  RACE-Middle dev; the dev stories: clock, bicycle, stone owl, swim and **bakery (new, 10
  questions, for developing the misses)**, 50 questions.
* **Choice.** S1 is kept if the general dev mean is more than 1 point better, or ARC-Easy dev at
  least 2 better with no general dev set more than 1 worse; and the passage dev mean not more than
  0.5 lower. S2 (over the kept one) if the passage dev mean is more than 1 better, or at least 2
  more dev-story questions right with the passage mean not more than 0.5 lower; and no general
  dev set more than 1 worse.
* Then frozen (commit), and run **once** on: the five general tests; MC160, MC500, RACE-Middle
  tests; **three new stories, 30 questions, written before any code** (`data/story/read2_tests.tsv`:
  orchard, paper lanterns, ferry: "no longer", a name at the end of a sentence, "made of X and Y",
  a man and a woman, "with"); the nine older test stories (lighthouse, paper dragon, night bus;
  salt, train, kite; harbor, garden, snow).
* Yardsticks: COMPREHEND's test numbers; before COMPREHEND (ARC-Easy 46.61, MMLU-HS 27.41); Laya;
  the sliding window.
* All runs in Colab; brains built by GitHub Actions. Reported as they come; journal updated;
  losers' code and releases deleted. Size is not a goal of this step (owner: smart first, prune later).
