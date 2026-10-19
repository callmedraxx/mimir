# An inference said in other words (infer)

Third step of the program (docs/SURVEY.md, the running scorecard). Started 2026-10-19, after YESNO (docs/YESNO.md).

## Why

CosmosQA 18.6 (chance 25, below it), Story Cloze 55.5 (chance 50), DREAM 43.0 (chance 33): the right option is what
the passage *implies* ("he likely fell", "she got sick", "the man means he is tired"), said in words the passage does
not use, while a wrong option reuses the passage's words. Mimir follows the words, so on CosmosQA it is worse than a
coin. The same gap is BoolQ's missing "no" side (docs/YESNO.md: "not in the Oyster zone" → cannot use the card) and
SQuAD's paraphrases (docs/SPEAK.md: "won" for "defeated … to earn the title"). Published: CosmosQA BERT 68.3, Story
Cloze GPT-2 86.4, DREAM BERT 63.2; GPT-3 few-shot Story Cloze 87.7. The owner's bar is Laya / GPT-2 / GPT-3 level.

## What the misses need

Read in the survey (docs/SURVEY.md) and in YESNO's "no" cases: (1) the option restates a *consequence* ("ate the whole
pie" → "got sick"), (2) the option restates a *state in other words* ("cancelled plans for a sequel" → "there will be
no second film"), (3) the option names a *reason* ("why did he do it": a state said just before the deed), (4) the
wrong option copies the passage's words and so wins today's coverage parts. Each of these is a general reading
skill, none is a dataset's.

## The method (brain-like: facts, kinds, the cortex's word links; nothing behind it)

**I1. Words first penalises copying (bit 1).** An option that repeats a passage sentence nearly whole is a quote,
not an answer: when the open-book part finds an option's words all in one sentence while the question asks "why",
"what will happen", "what does X mean", "how did X feel", the coverage parts count less for that option (the
passage-copy discount), and the fact and cortex parts decide. The discount is set on dev.

**I2. The cortex links the words that are not the same (bit 2).** The fact part already compares a statement to a
fact through `qsim` (the cortex's cosine). For an inference question the statement's words are *expected* to differ:
raise the cortex's weight for these question kinds (why / next / mean / feel) and lower the exact-word parts, so
"got sick" meets "ate the whole pie" through the links the shelf taught (sick ~ ill ~ ate too much), as a human's
associations do. The weights are set on dev.

**I3. What happens next: a state's usual sequel (bit 4).** Story Cloze asks for the ending. The episode's current
states (docs/STATE.md) and the last deed are the cue; the shelf holds what usually follows such a deed (the facts
read after it in the books: "ate too much" → "felt sick"). The candidate facts for "what happened next" are the
shelf's facts whose *subject* matches the story's and whose verb the cortex links to the last deed's; an ending
that those facts say wins over one that repeats the story. This is the sequence habit docs/SPEAK.md W5 and
docs/REPLAY.md R3 both wait for, in its first and cheapest form: not a next-word model but a next-fact lookup.

**I4. The reason is the state said just before (bit 8).** "Why did Sumit go to the bedroom?" → the state said of Sumit
before the deed ("Sumit is tired"), and for an option list, the option whose words the cortex links to that state.
For "why" questions in prose the same: the sentence before the deed, or a "because" / "so" clause.

**I5. A "no" from the other words (bit 16, for docs/YESNO.md).** With I2's links, a passage fact that the cortex
links to the statement's *opposite* (cancel ~ not happen, unable ~ cannot) counts against: the "no" side BoolQ lacks.

## Dev files (2026-10-19)

`cosmos_train.txt` (1,000 questions, 799 passages), `storycloze_val.txt` (1,871), `dream_train.txt` (1,000 questions,
639 dialogues), all in `data/understand/survey/`; format checked on the tiny brain on the first 40 of each. The full
brain's dev baseline for the frozen YESNO candidate is the first Colab run of this step.

## I1 measured (2026-10-19, tiny brain, `S7=1`, `CPW` 0.5)

A blanket discount on near-copy options for questions with "why", "what does", "feel", "mean", "likely": CosmosQA
train 22.6 → 23.6, Story Cloze val 52.9 → 52.9, DREAM train 41.2 → 38.9, MC160 dev 52.8 → 46.9, MC500 dev 58.9 → 51.9,
RACE-M dev 37.8 → 36.9. Out as it stands: those question words open ordinary questions too, whose quoted sentence is
the answer, and a word list cannot tell an inference question from a lookup. The discount can only follow a reason
(I3's "what happens next" has one: the option restates the story instead of continuing it), not the question's words.
The full brain's baseline (first Colab run): CosmosQA train 20.45, Story Cloze val 54.78 (the rest below).

## I3 built (2026-10-19, `S7=4`)

The story's last deed (the latest surface fact with a doer) keys the shelf's cue lists; the facts that share two of its
words or its verb are "what goes with such a deed in the books"; each ending's statement is scored by the closest of
them (kfact_sim, the cortex's links inside it), an ending whose words are all in the story is halved, and the score
enters the fact part weighted EPW × `NXW` (1.0). Tiny brain (64k shelf facts, no cortex links): Story Cloze val
52.9 → 55.5, MC160 dev unchanged. The full brain's shelf is five hundred times larger and its cortex links words,
so the first Colab dev run of the step measures I3 properly. **Measured (dev run 1): Story Cloze val 54.78 → 54.14.** The
tiny brain's gain did not carry: on the big shelf the facts that share the deed's words are many and near, and the
closest of them still favours the ending that repeats the story. Out as built. I3b (`S7=32` with 4): not the deed-like
facts but each one's *successor* in reading order, what the books say next after such a deed; this is the sequence
habit in its first form.

## Rules

* **Dev**: CosmosQA train (the first 1,000), Story Cloze 2016 validation (1,871), DREAM train (the first 1,000),
  all in `data/understand/survey/` (`tools/survey_fetch.py cosmos_train storycloze_val dream_train`); on
  `brain-school-cx4` in Colab; the tiny brain only for the format (its cortex has no links). Also in every dev run
  (nothing may fall): BoolQ train, bAbI train (choice and in words), the 70 dev stories, MC160 / MC500 / RACE-M dev,
  the five general dev sets.
* **Choice**: kept if the three dev sets gain at least 5 points on their mean over the frozen YESNO candidate; bAbI,
  BoolQ and the dev stories do not fall; the passage dev mean not more than 0.5 lower; no general dev set more than 1
  worse. The gain must not be a prior: the picks' distribution against the answers' is reported with every run.
* **Test, once**: CosmosQA validation, Story Cloze 2016 test, DREAM test (the survey's files), BoolQ validation, bAbI
  test (choice and words), the 18 test stories, MCTest / RACE-M tests, the general tests; then the survey rerun.
* Switch `S7`; `S7=0` identical to the frozen YESNO candidate (`S5=8257002` with Y19). Colab for every run, each
  runtime deleted after; adoption is the owner's call.
