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

## Steps (switch `S5`; `S5=0` identical to the adopted reader)

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
