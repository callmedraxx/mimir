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
