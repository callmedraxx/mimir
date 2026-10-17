# Yes or no with thin support (yesno)

Second step of the program (docs/SURVEY.md, the running scorecard; owner 2026-10-16: "improve on all those tests one by
one without getting too bad on the other"). Started 2026-10-17.

## Why

BoolQ (3,270 yes/no questions on a Wikipedia paragraph; 62% of the answers are yes): Mimir 46.7, chance 50, and it
says **"no" 70% of the time**. The question is rarely stated in the paragraph; it is an inference from it ("Does
ethanol take more energy to make than it produces?" over a paragraph about energy balance). Two things are wrong at
once: the judgement has no third state (a yes/no question whose answer the text does not state should be *unknown*,
and only then decided by weaker evidence), and whatever evidence it does use leans to "no".

A first look (droplet, tiny brain, BoolQ **train**): the tiny brain says "yes" 95% of the time (56.3%; 461 of 1,000
are exact ties between the two statements, decided by order), and where the two differ the fact and open-book parts
favour "yes" 480 to 25 and 236 to 41. So the "no" lean is not in the passage reading: it comes from the **shelf**,
whose 32 million facts enter the judgement with the passage's. For the "no" statement ("X is not Y") a shelf fact
saying "X is Y" counts against it, and the other way round; which way the shelf tips depends on which of its facts
come up, and on this brain they come up "no".

For speaking (docs/SPEAK.md W3) this step must leave behind a judgement with three values, **true / false / unknown**,
with a strength, taken from the passage.

## Steps (switch `S5`: 1 Y1, 2 Y2, 4 Y3, 8 Y4; `S5=0` identical to the adopted reader)

**Y0. Diagnose (no change).** One Colab run of the adopted reader on BoolQ train with the parts dumped (fact, joint,
open book, meaning, fit, for both statements, episode and shelf apart) and every pick shown: which part decides, with
which sign, when the passage states the answer, and when it does not.

**Y1. A yes/no question about a passage is judged by the passage (bit 1).** When the episode is active, the shelf's
fact and open-book parts do not enter a yes/no judgement; they enter only when the passage says nothing (the
episode's parts both within `THIN` of zero). The shelf is for questions without a passage (the general tests, which
have no yes/no options and are untouched).

**Y2. Unknown, then the weaker evidence (bit 2).** The passage's evidence for the statement: fact and open book of
the "yes" statement minus those of the "no" statement. Above `+THIN`: true; below `-THIN`: false; between: unknown.
Unknown is decided by what the passage holds of the question: the share of the question's content words (and their
cortex neighbours at 0.40) found in the passage, against the same for the statement's "not" side, and a yes/no
question asked of a text is more often yes (BoolQ 62%, the dev set's prior enters as the sign of the tie only).
`THIN` is set on dev.

**Y3. "Not" against a light verb (bit 4).** `kfact_sim` turns the score around only when the verbs match at 0.5 or
more; "is" / "has" / "does" weigh 0.3 and so "X is not Y" against "X is Y" sometimes keeps a third of the score
instead of turning around. For a be/have/do statement the turn-around applies when the subject and the attribute
match. Taken only if Y0 shows the fact part misreading "not" in the passage.

**Y4. A bare "yes" has no words of its own (bit 8).** Found by Y0: "yes" is read as a content word of the option
while "no" is a glue word and is dropped, so the "yes" statement's option words were {yes} + the question's and the
"no" statement's just the question's. Every coverage score (joint, open book, meaning) then ran lower for "yes",
since no fact or sentence holds the word "yes". With Y4 both answers are about the question's words alone, and only
the facts' "not" logic tells them apart.

**Y5. A "not" about the same things counts against (bit 16).** For a yes/no judgement, a negated passage fact whose
arguments match the statement's counts against it at half even when the verb differs ("are not considered" against
"count"); today such a fact keeps 0.3 of its score *for* the statement.

**Y6. A yes/no question by its answers (bit 32).** "Calcium carbide is the raw material for acetylene?" does not
begin with an auxiliary; with bare yes/no options it is a yes/no question all the same (48 of the 1,000 were not judged).

**Y7. "As" before a noun phrase is a preposition (bit 64, reader).** "not considered as goals", "known as the queen",
"works as a teacher": the reader classed "as" as a subordinator and dropped the phrase; now it is kept as pp:as.
Not before a clause ("as he left").

**Y8. An opening participle describes the noun (bit 128, reader).** "Converted penalties are not considered": the
reader took "Converted" as the verb and the negated fact lost its subject; a participle opening the sentence before
a noun and its verb is a modifier. Tiny brain: the probes read right; dev stories 6 wrong of 70 as before; bAbI train
the same; MC160 dev 52.4 → 53.2, MC500 dev 57.4 → 57.9; BoolQ train's passage-side judgement barely moves (the
"no" evidence stays at 24 of 1,000: the Wikipedia prose is lost in more ways than these two).

## Dev run 1 (2026-10-17, Colab, `brain-school-cx4`; `results/yesno_dev.txt`)

| S5 | BoolQ train | says yes | bAbI train | dev stories | MC160 / MC500 / RACE-M dev |
|---|---|---|---|---|---|
| 0 (adopted) | 47.10 | 28% | 67.86 | 67/70 | 52.78 / 57.25 / 40.62 |
| 8 Y4 | 54.70 | 92% | 67.07 | 67/70 | 52.78 / 57.25 / 40.62 |
| 9 Y4 + Y1 | 53.35 | 92% | 67.07 | 67/70 | 52.78 / 57.25 / 40.62 |
| 11 Y4 + Y1 + Y2 | 59.00 | 94% | 67.07 | 67/70 | 52.78 / 57.25 / 40.62 |
| 15 + Y3 | 59.40 | | 67.00 | 67/70 | 52.78 / 58.25 / 40.55; general dev 47.19 / 18.26 / 24.85 / 30.34 / 30.45 (was 47.19 / 18.26 / 25.05 / 30.07 / 30.55) |

Ties (the two statements scored alike) count half in the accuracy and go to "yes" in the picks. With Y2 the judgement
was true 354 (right 64%), false 16 (right 44%), unknown 582 (answered yes, right 59%): the passage's evidence tells
"true" a little better than the prior (62%) and "false" worse. Y1 (the passage alone) costs 1.3. Dev run 2: masks
10 (Y4 + Y2), 42 (+ Y6), 234 (+ Y7 + Y8), 250 (+ Y5).

## Y0 result (2026-10-17, Colab, `brain-school-cx4`, BoolQ train 1,000; `tools/yesno_notes.py`)

acc 47.1, picks "yes" 28% (the answer is yes 62%). Which part leans which way, over the 1,000 questions: fact favours
yes 523 / no 48 / same 429; **joint favours no 999**; open book favours no 848 / yes 151; meaning same 915; fit same
916. The parts come mostly from the shelf (fact 1,452 of 2,000 option-parts, joint 1,586, open 1,536), not the
passage. Judged by the passage's parts alone the accuracy is 47.8, so the shelf is not the cause either: the cause
is the coverage asymmetry of Y4. The passage states something for both statements in 916 questions and nothing in
84 (acc 44.0 there, yes 52%). Y2 on the passage's evidence alone: THIN2 0.1 → true 207 / false 595 / unknown 198,
acc 49.2; THIN2 0.4 → true 180 / false 53 / unknown 767, acc 60.8 (that is mostly the prior). So the first dev run
is Y4 alone, then Y4 + Y1, Y4 + Y1 + Y2, all four.

## Rules

* **Dev**: BoolQ **train**, the first 1,000 (`data/understand/survey/boolq_train.txt`, from the SuperGLUE release;
  `tools/survey_fetch.py boolq_train`), on `brain-school-cx4` in Colab; the tiny brain for guidance only. Also in
  every dev run (nothing may fall): bAbI train (STATE's seven tasks, `babi_train.txt`), the 70 dev stories, MC160 /
  MC500 dev, RACE-M dev, the five general dev sets.
* **Choice**: kept if BoolQ train gains at least 5 points over the adopted reader on the same file and says "yes"
  between 50 and 75% of the time; bAbI train does not fall more than a point; the 70 dev-story questions do not
  fall; the passage dev mean (MC160, MC500, RACE-M) is not more than 0.5 lower; no general dev set more than 1 worse.
  At most three Colab dev runs after Y0.
* Then frozen (commit), **once** on: BoolQ validation (the survey's `boolq.txt`, 46.73 before); the hand-written
  stories' yes/no questions (the 18 stories, scored apart); MC160 / MC500 / RACE-M tests; bAbI test; the five
  general tests. Then the survey's 17 files once more (results/survey.txt grows a column).
* All runs in Colab, each runtime deleted after; droplet for code and the tiny brain. Reported as they come; journal
  updated; losers' code deleted; the switch removed on adoption. Adoption is the owner's call.
* The validation file is the test: not read, not run on, until the freeze.
