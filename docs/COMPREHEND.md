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

## How it was built (before any dev result on the school brain)

Developed on the dev stories (clock, bicycle, stone owl, swim) and MCTest dev, on the droplet's
3 MB test brain (Alice + War and Peace), looking at what the reader makes of each sentence
(`bg facts`, new: the facts of each line on stdin). With `NW=0 RD=0` every answer is the same
as before, byte for byte (passages, dev stories, four general dev sets).

**R1 (`NW`)**, answering:
* A fact that matches the option's statement but has the opposite "not" (and the same doing:
  the verbs match) counts against it. For each statement fact the closest known fact decides,
  for or against; facts holding the option's own words decide first (otherwise "rides on
  Mondays" was as close to "ride on Friday" as "not on Fridays", by the cortex's day ~ day).
* Open book: the re-read sentence that covers the question and option best decides; if its
  part about the option's own words says the opposite, open book counts against.
* Inside an episode, the passage saying the opposite decides over the shelf's support
  ("Pell eats beetles, not mice" over "owls eat mice").
* Yes/no questions: "Was Nadia sad?" + "no, she was not sad" → "Nadia was not sad. she was
  not sad"; an option with no words of its own is scored by its yes or no.
* A question asking for what is NOT so ("What did Marta not take?", "Which animal did Luna
  not visit?") is read without its "not" and turned around, as before; with "not" now
  counting, the option the passage denies wins. A "why … not" or "how … not" question is not
  turned around (it asks for a reason).

**R2 (`RD`)**, the reader (also how questions become statements):
* "a bicycle named Comet", "a cat called Pepper" → the phrase is Comet (bicycle), and the fact
  "Comet is a bicycle"; "the train is called the Grey Heron", "the children call it the Whale",
  "Asha named the kite Ember" → "the Grey Heron is the train", …; apposition ("the baker,
  Mr. Oduya, fixed", "the mayor, a tall man named Orvo Kell, was") → Oduya is the baker.
* "made of", "shaped like" after a thing ("an owl made of stone") → a fact about that thing;
  passives read as actives ("was built by a clockmaker" → the clockmaker built it).
* "without a ticket" → did not have a ticket. "eats beetles, not mice" / "every Monday, but not
  on Fridays" → the "not" only for what follows it.
* "said no." → "no" (the tokenizer, shared with the app, reads "no." as the abbreviation of
  "number" and loses the sentence end; undone in the reader only); "yes" and "no" are words
  that count. "did not purr": a word after "did (not)" that the table lacks is the verb.
* A capital inside a sentence, or a capital word the table lacks, or two capitals in a row, is
  a name ("Old Watch"). "it" points to the latest thing done or named, not a place
  ("an owl made of stone" → it = the owl). A new rule for "he/she" (names that act) was tried
  and dropped: it cost MCTest dev.
* Questions: "what does Pell eat?" → "Pell eat <option>" (before: "Pell eat? <option>", read as
  two sentences, so the option was never in the statement's fact; an option that is a clause
  of its own stays a sentence of its own); "what is X" → "X is <option>"; "what was the
  cat's name" → "the cat was named <option>"; "on which day does Nadia ride" → "Nadia does
  ride … on <option>"; "Mr." does not end a re-read sentence.

On the test brain (dev only, 3 MB brain, not a result): MC160 dev 51.1 / 52.4 / 50.3, MC500
dev 52.4 / 56.1 / 55.9, clock 7.5 / 8 / 9 of 10, dev stories 19.5 / 25 / 26 of 30 (R0 / R1 /
R2, NW 1; NW 0.5 and 2 the same there).

## Dev results (2026-10-15, Colab, school brain; `results/comprehend_dev.txt`)

Means of MC160 (train + dev), MC500 (train + dev), RACE-Middle dev; dev stories = clock + bicycle, stone owl, swim (40):

| cell | MC160 | MC500 | RACE-M dev | mean | dev stories |
|---|---|---|---|---|---|
| **R0** (now) | 57.35 | 51.30 | 39.28 | 49.31 | 30 / 40 |
| R1, NW 0.5 / 1 / **2** | | | | 50.16 / 50.23 / **50.46** | 33 / 40 |
| R2, NW 0.5 / 1 / **2** | 59.90 | 53.39 | 41.17 | 51.38 / 51.36 / **51.49** | 35 / 40 |

**Chosen: R2, NW 2** (R1 +1.15 over R0, R2 +1.03 over R1; each more than 1 point). Clock 8.5 → 10 of 10.
General dev sets with no episode: 43.68 / 20.04 / 24.25 / 28.24 / 29.40 (R0 43.68 / 19.68 / 24.05 /
28.72 / 29.50); the old brain's shelf, the new reader only in the questions. R0 here: RACE-M dev 39.28
(PASSAGE's U1 39.12, same code; MC160 and MC500 identical) — a small run-to-run difference.
Frozen in `tools/comprehend_choice.env`. The tests use `brain-school-cx2` (the shelf read by the new
reader, built by GitHub Actions from 402003d: 32,484,331 facts, 753,686 words).

## Test results (2026-10-15, Colab, once, frozen 24e3a3c; `results/comprehend_test.txt`)

R0 on `brain-school-cx`; R2 (NW 2) on `brain-school-cx2`, the shelf read by the new reader.

| test | **R0** | **R2** | change |
|---|---|---|---|
| MCTest MC160 test (240) | 49.58 | **54.69** | +5.1 better |
| MCTest MC500 test (600) | 50.72 | **52.47** | +1.8 better |
| RACE-Middle test (1,436) | 40.68 | 40.38 | tie |
| new stories: lighthouse, paper dragon, night bus (30) | 16 / 30 | **21 / 30** | +5 better |
| salt, train, kite (30) | 23 / 30 | 23 / 30 | tie |
| harbor, garden, snow (30) | 29 / 30 | 28 / 30 | −1 question, worse |
| MMLU-HS | 27.41 | 26.32 | −1.1 worse |
| CommonsenseQA | 31.37 | 32.35 | tie (+1.0) |
| ARC-Easy | 46.61 | 43.04 | −3.6 worse |
| ARC-Challenge | 21.22 | **23.71** | +2.5 better |
| OpenBookQA | 24.40 | **25.57** | +1.2 better |

**Count: better 5, tie 3, worse 3** (general mean 30.20 both). Mixed; the owner chose: **adopted**
("okay let's adopt it"). On dev the new question reading with the old shelf left the general dev
sets unchanged, so the ARC-Easy and MMLU losses most likely come from the shelf being re-read by the
new reader; that is the next thing to look at, on dev.

Still missed in the new stories: "What did Marta not take?" (a candle), "What is the kite made
of?" (red paper and bamboo), where they flew the kite, the prize, who got the book, "Does Tomas
like loud music?", who was sick, "Who is Mr. Bayo?" (apposition at the end of a sentence), "was no
longer sick".

Losers deleted: the switches `NW` and `RD` (the old reading and answering are gone; NW is 2).
`brain-school-cx2` is now the brain (`tools/brain_release.py`).

## Correction (2026-10-15, after adoption)

The general tests above are **not a fair comparison**. R0's numbers (27.41 / 31.37 / 46.61 /
21.22 / 24.40) were measured with the shelf on disk; the R2 run downloaded the brain from its
release but **not the shelf**, so open book (which re-reads shelf sentences, weight 2) scored
nothing: "open 25.02" alone on ARC-Easy, against 45.97 with the shelf. The same shows on dev:
R0's code and brain gave 43.68 / 19.68 / 24.05 / 28.72 / 29.50 without the shelf, against 47.81 /
20.01 / 25.25 / 29.32 / 31.55 with it. So the ARC-Easy and MMLU "losses" are most likely the
missing shelf, not the reader. The passage results stand (R0 and R2 both ran without it). The
release now carries its shelf (build 4); the general tests are run again on `brain-school-cx2` with
the shelf, once, as the measurement this step should have had (docs/READ2.md step 1).
