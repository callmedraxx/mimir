# Reading what a sentence says: "not", names, what a thing is (comprehend)

Owner (2026-10-15): "why is the reader dropping facts when that's usually the job of the
cortex or hippocampus" → (the reader misreads, it does not choose to forget) → "okay let's do
that next".

## Why

After PASSAGE the misses in the stories are not in memory or answering; they happen when a
sentence is read into a fact (docs/PASSAGE.md). The hippocampus (the episode) can only keep
what reading gives it. Looked at in the code before writing this (no runs):

* **"not" is read, then almost ignored.** The reader marks "Nadia was not sad" as negative.
  When answering, a fact with the opposite "not" still counts 30% for the option; the open
  book part (re-reading the sentence, weight 2, × 4 in an episode) does not look at "not" at
  all, so "Nadia was not sad" fully supports "Nadia was sad". An option that only says "no" /
  "yes" to the question ("no, she was not sad") has no words of its own left and scores 0.
* **No reading of names.** "a bicycle named Comet", "the train is called the Grey Heron",
  "people call it Old Watch", "the baker, Mr. Oduya," do not become "Comet is a bicycle",
  "the Grey Heron is the train", "Mr. Oduya is the baker". So "What is Comet?" is answered from
  what "comet" usually means.
* **"made of", "without"**: "an owl made of stone" and "got on without a ticket" have no
  fact shape that says what the thing is made of, or what someone did not have.

## Cells

| cell | what | needs a new brain |
|---|---|---|
| **R0** (now) | PASSAGE as adopted | |
| R1 | **"not" counts against.** A fact (or a re-read sentence) that matches the option's statement but says the opposite ("not") is evidence against the option, weighted `NW` (chosen from 0.5, 1, 2), instead of 30% for it. An option that only answers yes or no is scored by its yes/no against the facts. | no (answering only) |
| R2 | R1 + **the reader reads names and what a thing is**: "X named / called Y", "is called", "call it Y", "X, Mr. Y," (apposition) → Y is X; "made of Z" → X is made of Z; "without Z" → X does not have Z. Questions and options are read by the same reader, so "What is Comet?" asks for that fact. | yes: the shelf is read by the same reader |

Each cell adds to the one before. With no episode active, R1 changes nothing that is not a
"not" fact; R2 changes the shelf's facts too.

## Rules

* **Dev**: MCTest MC160 train + dev, MC500 train + dev, RACE-Middle dev (as before); and the
  stories **clock, bicycle, stone owl, swim (40 questions)**. Bicycle, stone owl and swim were
  PASSAGE's test and their misses were read; from now on they are dev stories, not tests.
* **Choice**: a step is kept if it is more than 1 point better on the passage dev mean, **or**
  at least 2 more dev-story questions right with the passage dev mean not more than 0.5 lower.
  (Negation and names are rare in MCTest and RACE, so the old rule alone could not see this
  step; the new test stories below guard against fitting the dev stories.)
* Then frozen (commit), and run **once** on: MC160 test, MC500 test, RACE-Middle test;
  **three new stories, 30 questions, written before any code** (`data/story/comprehend_tests.tsv`:
  lighthouse, paper dragon, night bus: negation, names, "made of", "without", "said no", a
  story fact against general knowledge); the six older test stories (salt, train, kite,
  harbor, garden, snow); the five general tests.
* If R2 is chosen, the general tests run on a brain rebuilt with the frozen reader (GitHub
  Actions, a new release `brain-school-cx2`; `brain-school-cx` kept until the result). A
  general test more than 1 point lower counts as worse.
* Dev for R2 runs on the current brain (the shelf read by the old reader, the passages by the
  new one); the rebuilt brain is used only for the tests.
* Yardsticks: PASSAGE's test numbers (MC160 49.6, MC500 50.6, RACE-M 40.7; stories 23/30 and
  29/30); the sliding window (67.1 / 57.8 / 45.3); Mimir's passage-only reader (63.8 / 52.8 /
  41.1).
* Runs in Colab. Reported as they come; journal updated; losers' code deleted.
