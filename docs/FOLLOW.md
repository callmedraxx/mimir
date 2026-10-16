# Following a story: who is who, when, and whether (follow)

Owner (2026-10-16): "how do we now fix this understanding gap so it understands past tense, present tense,
and a sense of now and the other things it missed in the market stall story. as a human I read the passage
and I followed each step of the way who is who, who did what to who and why" … "yes go let's see if we can
add all 1 to 5 improvements".

## Why

The market stall story (data/story/market.txt, 10 questions, dev) scored 5 of 10, the same with the reader
before READ2. The red boat story, one fact a sentence, scored 10 of 10. What the market stall needs and
the reader lacks, from its facts (`bg facts`) and the answer parts (`DUMP`):

| question | what went wrong | where |
|---|---|---|
| What is the stall made of now? → wood (tie wood / iron / bamboo) | "but the table on it is made of iron" kept the stall as subject (the list rule ran across "but"); "last year the stall was made of wood" has no time: three equal facts | reading (3), time (2) |
| Is Ada still afraid? → "yes, she is afraid of the man" | "For a long time Ada was afraid" lost Ada into the time phrase ("a long time Ada" one phrase); the right option's statement "Ada is not afraid" then opposes the subjectless "afraid of the goat" fact (fact −2.26) | reading (4), time (2) |
| What was the coin really made of? → silver | "He said it was not silver": "he" → Ngozi (sex unknown, so taken as a man, over Mr. Obi, a known man); the embedded clause "it was not silver" kept "he" as subject; then "It was made of tin": "it" → silver (a be-attribute became the latest thing) | pronouns (4) |
| Who did not laugh? → Mr. Obi | the fact NOT laugh who=Ngozi is read; the answering strips the "not", turns the scores around and takes the least supported option (every option's fact part is 0) | answering (1) |
| Who took one basket home? → Ada | the fact take who=woman(old) is read (fact 1.80 of 2.00, Ada 0.98) but open book prefers the sentence with "baskets of yams … to Ada" (the question's words) by more | answering (5) |

Also seen: "keeps the money and talks to the customers" read "talks" as a thing (then "it" pointed at it);
"Their stall" took Ada's sex from "their" and the title word "Stall" was a name, so "she" → stall;
"gave them to Ada": "them" → people, not the baskets; "instead of money" read as "of money".

## Steps

**F1. Polarity when answering (no reading change).** Inside an episode, a question negated at its verb
("Who did not laugh?", "What did Ada not want?") whose verb the episode has a NOT fact for is matched
directly: the option statement keeps its "not" and a NOT fact for it counts for it, a plain fact against.
Only when the episode has no such fact does the old turn-around run (the general tests have no episode,
so they are untouched). Yes/no options are read as the statement they make (done in COMPREHEND) and
judged by polarity the same way.

**F2. Tense and time.** Each frame read in an episode carries past / present from its verb group ("was
made" / "is made", "laughed" / "laughs") or from a time phrase ("last year", "years ago" → past; "now",
"today", "still" → present). An option statement's tense comes the same way. A fact whose tense differs
from the statement's counts TENSEW (dev) of itself: "last year the stall was made of wood" gives way to
"is made of bamboo" for "What is the stall made of now?", and "was afraid" gives way to "is no longer
afraid" for "Is Ada still afraid?". Shelf facts have no tense (nothing changes for them); the brain file
is unchanged (the bits go in the flags byte).

**F3. A new clause after "but".** No list runs across "but"; after "but" (and after "and", a comma), a
noun phrase, any prepositional phrases, then a verb start a new clause with that noun phrase as subject.

**F4. Who is who, things too.** (a) "he" / "she": among the latest people, one whose sex is known to fit
beats one whose sex is unknown. (b) A possessive gives its sex to the head only when the head is a person
word ("her brother", not "their stall"). (c) The latest thing: for "be", the subject noun is the thing, not
its attribute ("it was not silver" leaves the coin as the thing); for other verbs the object, as now.
(d) "them": the latest plural thing of this or the last sentence, else the latest group. (e) After a
reporting verb (say, think, know, …) a noun phrase and a verb are a clause of their own, with that noun
phrase as subject ("said the coin was older"). (f) A time phrase ("for a long time", "one day") does not
swallow a following name. (g) "and talks to": after "and", an -s word that can be a verb, followed by a
preposition or "to", is the verb.

**F5. An exact fact first.** In an episode, when one option's statement is matched by an episode fact
almost whole (its fact part at least FX of the most an episode fact can give) and no other option's is,
the open-book part is scaled by FO (dev) for that question: the fact decides, the sentence words only
break ties. (The journal's open item: "the answering misses the reader no longer causes".)

Each is a bit of the switch `S3` (1 F1, 2 F2, 4 F3, 8 F4, 16 F5), `S3=0` identical to the adopted reader;
the switch is removed on adoption and the brain rebuilt (`brain-school-cx4`) if the reading changed.

## Rules

* **Dev**: the tiny droplet brain for guidance; then in Colab on `brain-school-cx3`: the five general dev
  sets; MCTest MC160 / MC500 dev, RACE-Middle dev; the dev stories clock, bicycle, stone owl, swim, bakery
  (50) and red boat + market stall (20, `smart_tests`).
* **Choice**: the set of bits is kept if the dev-story questions right (70) go up by at least 4 with the
  passage dev mean (MC160, MC500, RACE-M) not more than 0.5 lower, and no general dev set more than 1
  worse. Bits that lose on the tiny brain are dropped before Colab; at most three Colab dev runs.
* Then frozen (commit), the brain rebuilt if the reading changed, and run **once** on: the five general
  tests; MC160, MC500, RACE-Middle tests; **three new stories, 30 questions, written before any code**
  (`data/story/follow_tests.tsv`: the well, the bridge, the parrot: then vs now, "Who did not …",
  "no longer", two people of one sex with one sex unknown, "said that … was", "it" / "them", a relative
  clause subject, a new subject after "but", questions whose words sit in the wrong sentence); the twelve
  older test stories.
* All runs in Colab, each runtime deleted after; droplet for code and the tiny brain only. Reported as they
  come; journal updated; losers' code deleted. Size is not a goal (owner: smart first, prune later).

## How it was built (2026-10-16, tiny droplet brain, before any school-brain result)

Switch `S3` in facts.c / bg.c; `S3=0` byte-identical to the adopted reader on the answer dumps of the dev
stories and MC160 dev. What the market stall taught on the way, beyond the five steps:

* The passage files carried the story title as text ("The Market Stall …"), so "Stall" and "Market" were names
  for 50 sentences and "she" pointed at the stall. `tools/stories.py` now leaves the title out (all sets).
* A yes/no option's statement ended "goat.. she is …": the question's "?" plus the added "." made one sentence
  of the two, and "she" became an argument of the first frame (COMPREHEND's yes/no had this all along). F1 fixes it.
* A yes/no option is judged by the facts holding its *own* extra words ("the man" in "yes, she is afraid of the
  man"), so the exact opposite fact was never the one that decided. F1: inside an episode a yes/no option is about
  the question's words, and a closest fact that says the opposite is not overridden by a weaker option-word match.
* "was afraid" (past) and "is no longer afraid" (present) are equally close to "Ada is not afraid"; the earlier
  one won every tie. F2: the tense of a frame counts (TENSEW 0.5), and of two episode facts equally close the
  later one decides; the re-read sentence check keeps tense too.
* A base-form verb ("the man pay with the coin Ada", built from "What did the man pay…") is not present tense:
  its tense is unknown. Only -s is present, -ed / irregular past.
* "talks" is a noun in the word table with the verb base "talk": after "and", an inflected noun with a verb base
  before a preposition or "to" is the verb.
* F4 grew: "it" in a repeated slot ("afraid of the goat … afraid of it"), partitives ("part of it" is the thing),
  the latest thing updated at clause boundaries (a passive's subject counts), "Ada is the younger sister" not a
  second person, the clause subject reset after "but" (S2's "an object pronoun is never the clause subject"
  applied to subject "she", whose word class is O).
* The "Who did not laugh?" statement was "not laugh Ada" (no frame at all): with no phrase between "did" and its
  verb the option is the subject, "Ada did not laugh".

Tiny brain, dev (guidance only; `S3=0` → `S3=31`): red boat + market stall 15 → **20** of 20, dev stories
24 → 27 of 30, clock 10 → 10, bakery 6 → 7: **55 → 64 of 70**. MC160 dev 49.44 → 52.36, MC500 dev 53.38 →
56.88. General dev subsets (80 each): ARC-E, OBQA, MMLU the same; CSQA −1 question (bit 8, how questions are
read). Knobs on the tiny brain: TENSEW 0.3 / 0.7, FX 0.75, FO 0 / 0.5 each moved one set up and another down;
kept at 0.5 / 0.85 / 0.25. Not fixed: bicycle_10, stoneowl_3, swim_8, bakery_7 / 9 / 10 (other causes).

Colab dev next: `tools/follow_colab.py dev` (S3 = 0, 15, 16, 31 on `brain-school-cx3`).

## Dev result (2026-10-16, Colab, school brain `brain-school-cx3`; `results/follow_dev.txt`)

| cell | MC160 train / dev | MC500 train / dev | RACE-M dev | passage mean | clock | dev stories | bakery | red boat + market | **of 70** |
|---|---|---|---|---|---|---|---|---|---|
| S0 (adopted) | 61.10 / 50.69 | 53.11 / 58.25 | 40.47 | 50.76 | 10 | 27 | 7 | 16 | **60** |
| S3 15 (F1-F4) | 60.92 / 54.44 | 53.11 / 59.75 | 40.26 | 51.10 | 10 | 28 | 8 | 18 | **64** |
| S3 16 (F5 alone) | 61.46 / 51.53 | 53.53 / 58.75 | 40.82 | 51.19 | 10 | 28 | 10 | 16 | **64** |
| **S3 31 (all)** | 61.28 / 54.44 | 53.78 / 58.75 | 40.24 | **51.32** | 10 | 29 | 9 | 19 | **67** |

General dev sets, S0 → S3 31: ARC-Easy 46.05 → 46.05, ARC-Challenge 17.42 → 17.42, OpenBookQA 21.85 → 22.05,
MMLU-HS 29.19 → 29.19, CommonsenseQA 30.25 → 30.45. The rule (4 more story questions, passage mean not 0.5
lower, no general set 1 worse) is met by S3 31: +7 questions, mean +0.56, no general loss. The market stall's one
miss on the school brain is the goat question, between two "no" answers. **Frozen at S3=31**; build 7
(`brain-school-cx4`, the shelf read with S3=31) for the test.

## Test result (2026-10-16, Colab, once, S3=31 on `brain-school-cx4`; `results/follow_test.txt`)

| | READ2 (`brain-school-cx3`) | FOLLOW (`brain-school-cx4`) | change |
|---|---|---|---|
| MC160 test | 54.48 | **58.23** | +3.75 |
| MC500 test | 52.06 | 52.11 | +0.05 |
| RACE-Middle test | 39.78 | **40.69** | +0.91 |
| new stories well / bridge / parrot (S0 on cx3 → S3 on cx4) | 20.5 / 30 | 21 / 30 | +0.5 |
| orchard / lanterns / ferry | 27 / 30 | 25 / 30 | −2 |
| lighthouse / dragon / night bus | 20 / 30 | 23 / 30 | +3 |
| salt / train / kite | 23 / 30 | 24 / 30 | +1 |
| harbor / garden / snow | 28 / 30 | 28 / 30 | 0 |
| all 150 story questions | 118.5 | 121 | +2.5 |
| MMLU-HS / CSQA / ARC-E / ARC-C / OBQA | 26.87 / 31.12 / 46.34 / 22.00 / 24.20 | 27.26 / 30.75 / 45.63 / 23.66 / 23.80 | +0.39 / −0.37 / −0.71 / +1.66 / −0.40 |
| general mean | 30.11 | 30.22 | +0.11 |

The shelf read with S3=31 makes 32,471,614 facts (cx3: 32,964,286; fewer list facts across "but" and
clauses split after "but"). The passage tests gain (MC160 +3.8, RACE-M +0.9), the story tests +2.5 of 150,
the general mean +0.1 (ARC-Challenge +1.7, ARC-Easy −0.7). The three new stories, written before the code,
gain only one question (S0 20.5 → 21): well_3 "Who did not cry?" and parrot_1 "made of now" are fixed,
parrot_9 "Does Tayo like the parrot now?" is lost (picked "yes, since that night" over "no, he still does not
like it"); still wrong: well_2 "What was the roof made of long ago?" (tin), well_5 (ten naira), bridge_9 "Who
did not get a fish?" (Zainab, S0 the children), parrot_4 "Who does not like the parrot?" (Mrs. Bello),
parrot_6 / 7 / 8, parrot_10 "Who is Mr. Audu?" → a parrot. The older stories: orchard_10, lantern_5, lantern_6,
lighthouse_10 newly wrong; ferry_7, lighthouse_1, dragon_3, nightbus_8, nightbus_10, kite_9 newly right.
Adoption is the owner's call (the dev gain of 7 on the dev stories became +2.5 on unseen stories: much of the
"who is who" and "then vs now" gap remains).
