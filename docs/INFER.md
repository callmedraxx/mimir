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
