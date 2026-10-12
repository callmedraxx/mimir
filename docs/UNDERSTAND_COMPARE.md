# Understanding: do neurons that grow roles understand what they read?

Owner (2026-10-12): "let's test the neuron growing on understanding". This
page fixes the rules **before any code is written**, as for
CONTEXT_COMPARE.md. The results go below the line, whatever they are.

## What is missing today

Both context windows (today's, and the neuron one in CONTEXT_COMPARE.md)
find any part of what they read, but neither knows who did what to whom: the
neuron one keeps an event's content words as a set, so "the dog bit the man"
and "the man bit the dog" are the same event. Today's keeps the page, but
nothing in it reads roles out of the page.

## The three

| | A: today's | B: neurons as they are | B+: neurons that grow roles |
|---|---|---|---|
| holds | every token and the page (`src/text.c`) | event neurons: an event's content words; word neurons with cue lists (`sandbox/neurons/ctxwin.c`, grown version) | the same, plus **parts that grow while reading**: an event whose clause has a verb grows a **role part** (its frames: verb, who, what, to whom, and place/direction phrases, with negation, "either…or" and time words) and the times it was read. Glue words are used to read the roles, then forgotten; the page is still not kept |
| answers by | the best-known way to answer from kept text without understanding: the **sliding window + distance** baseline of the MCTest paper (Richardson et al. 2013, Figure 4), on its page | the same code, on what B holds: its events' content words in the order the events were first made; word counts from the cue lists | its frames: the question is read into a frame with the asked slot open, and the slot is filled from the story's frames (below); for multiple choice, the role score is added to B's sliding-window score |

Shared, not counted in anyone's bytes: the word meanings (data/meaning), and
a **word-class table** (noun, verb, adjective, preposition…: the commonest
tag of each word in the Brown corpus) with verb base forms (WordNet's
exception lists, then -s/-ed/-ing rules). A and B don't use them; B+ needs
them to read roles. Glue words for all three: the 150 commonest, as before.

### What B+ may do with its frames (fixed now, nothing added after the test run)

1. **Latest frame wins**: the asked slot is filled from the most recent frame
   whose other slots match the question (same word, same verb base form).
2. **Pronouns**: he/she/it → the latest single agent; they → the latest
   group ("Mary and Daniel").
3. **Carrying**: something taken (take, get, grab, pick up) moves with whoever
   took it until they let it go (drop, leave, discard, put down). The one
   piece of world knowledge, stated here.
4. **Yes / no / maybe**: from the latest frame about the thing asked, with
   negation ("no longer") and "either … or" (maybe).
5. **Counting and listing** what someone holds, from rule 3.
6. **Before / after**: frames are ordered by reading time and time words
   (yesterday < this morning < this afternoon < this evening).
7. **Kinds**: "X is a Y" — X has what Ys have (deduction); and what one Y has,
   another Y has (induction).
8. **Relations**: opposites (east/west, north/south, left/right,
   above/below, bigger/smaller, fits in/bigger) and chains of them (A left of
   B, B above C ⇒ where A is from C), including finding a path.
9. **Why / where will**: a person's latest state ("X is tired") is why; where
   they will go is the place in the story whose meaning is closest to that
   state.

Each story is read fresh (nothing carried between stories, for all three).
B+'s reading rules and the weight of its role score in multiple choice are
worked out on the **train and dev** stories only; the test stories are run
once, at the end.

## Tests

| cell | test | score |
|---|---|---|
| U1 | **bAbI** (Weston et al. 2015, 20 tasks, English 1k, test sets: 1,000 questions each) | mean exact-answer accuracy over the 20 tasks |
| U2 | **bAbI tasks passed** | tasks at ≥ 95% (the dataset's own pass mark) |
| U3 | **MCTest MC160 test** (240 questions, 4 choices, written by people for children's stories) | % correct |
| U4 | **MCTest MC500 test** (600 questions) | % correct |
| M  | **memory held** after War and Peace (as CONTEXT_COMPARE.md M1) | bytes |
| S  | **speed**, take in War and Peace (+ gist): phone little core if reachable, else droplet one core | tokens/s, worst of 3 |

Also shown, not scored: bAbI per task; MCTest split into questions that need
one sentence vs several; recall and find on War and Peace (CONTEXT_COMPARE.md
R1, F1) to check that growing roles costs nothing there; bytes of the role
parts.

A's check: on MCTest, A's sliding window + distance should come out near the
published 66.25% (MC160 test) and 56.67% (MC500 test); if it does not, A is
wrong and is fixed before anything is compared.

Fair warning written now: bAbI is made by a program from a small vocabulary,
and hand-written rules like 1–9 are known to solve much of it. A high bAbI
score shows the roles are there and used; it does not show understanding of
ordinary text. **MCTest is the real test**: stories and questions written by
people, which no list of rules covers.

## Who wins

As before: more than 1 point (scores; tasks passed counts as points) or
10% (bytes, speed) wins a cell, otherwise a tie. Counted between A and B+;
B is shown so the effect of growing roles can be seen on its own.

---

## Results (2026-10-12)

Harness: `sandbox/neurons/understand.c`, frozen in bb66433 before the test
sets were opened; data: `tools/understand_fetch.py`. B+ was developed on the
bAbI train sets and the MCTest train + dev stories only (reader bugs fixed
there; the role weight, 2, chosen there). The test sets were run once.

**A's check passes.** On MCTest test A gets 67.08% (MC160) and 57.83% (MC500)
against the published 66.25% and 56.67% for the same algorithm (train + dev:
70.3% and 59.1% against 67.75% and 58.21%). The small gap is tie-breaking
(theirs a coin flip) and word splitting.

| cell | test | A: today's | B: neurons | B+: neurons growing roles | winner (A vs B+) |
|---|---|---|---|---|---|
| U1 | bAbI, mean of 20 tasks (test) | 20.16% | 20.51% | **94.09%** | **B+** |
| U2 | bAbI tasks passed (≥ 95%) | 0 | 0 | **17** | **B+** |
| U3 | MCTest MC160 test | 67.08% | 65.42% | 67.50% | tie (0.4 pt) |
| U4 | MCTest MC500 test | **57.83%** | 55.67% | 54.17% | **A** (3.7 pt) |
| M  | memory held after War and Peace | 11.3 MB | 1.29 MB | **3.94 MB** | **B+** (2.9×) |
| S  | speed, phone little core, worst of 3 | 17.5k tok/s | — | 18.0k tok/s | tie (+3%) |

**Count: B+ 3 (U1, U2, M), A 1 (U4), 2 ties (U3, S).** Both pass the 10k gate.

### bAbI per task (test, 1,000 questions each)

| task | A | B | B+ | | task | A | B | B+ |
|---|---|---|---|---|---|---|---|---|
| 1 one fact | 3.4 | 11.5 | 100 | | 11 he/she | 4.9 | 0.6 | 100 |
| 2 two facts | 0.1 | 0.1 | 100 | | 12 "and" | 1.7 | 6.9 | 100 |
| 3 three facts | 1.5 | 0.5 | 100 | | 13 they | 1.8 | 1.2 | 100 |
| 4 two-place relations | 53.9 | 54.4 | 95.6 | | 14 time | 5.0 | 3.4 | 100 |
| 5 three-place (give) | 5.9 | 9.0 | **40.7** | | 15 deduction | 5.7 | 4.8 | 100 |
| 6 yes/no | 66.7 | 65.9 | 100 | | 16 induction | 22.4 | 17.8 | 100 |
| 7 counting | 0.0 | 0.0 | **80.1** | | 17 positions | 47.5 | 47.5 | 100 |
| 8 lists | 1.2 | 4.0 | 95.8 | | 18 sizes | 53.8 | 53.7 | 100 |
| 9 negation | 56.9 | 56.5 | 100 | | 19 path finding | 0.0 | 0.0 | 100 |
| 10 maybe | 67.6 | 66.5 | 100 | | 20 motives | 3.3 | 5.9 | **69.6** |

The three B+ fails are the rules as fixed, not reading errors: in task 5 the
story says "handed" or "passed" and the question "give" or "received" (rule 1
wants the same verb); in task 7 things change hands by giving, which rule 3
(take / let go) does not cover; in task 20 the word meanings choose the wrong
room for a feeling (bored → bedroom). A and B answer with the word that sits
nearest the question's words, which is usually the verb ("Mary **moved** to
the bathroom"): without roles nothing tells a place from a verb.

### What it shows

* **The basics of who-did-what are there.** Growing roles turns 20% into 94%:
  where people and things are, carrying, giving, time, kinds, negation,
  either-or, spatial and size chains, paths. Today's and the plain neurons
  cannot do any of it. As warned above, bAbI is made by a program and rules
  written for its kinds of questions solve it; this shows the roles exist and
  are used, not that ordinary text is understood.
* **On stories people wrote, roles add about nothing.** B+ is +2.1 points
  over B on MC160 and −1.5 on MC500; overall it is level with today's word
  matching. The reader breaks on ordinary prose: "It was Jessie Bear's
  birthday" gives a frame with "it" as who; "she asked them if they would
  like to have cake" loses "like"; the word-class table takes "bit" for a
  noun, so "The dog bit the man" gives no frame at all; "She gave the apple
  to John" makes John a place Mary went to. Understanding ordinary text is
  not there.
* **Roles cost memory.** 96,097 frames in War and Peace at about 20 B each
  (the pool), plus the part table and the word map: 2.6 MB, twice what B
  holds itself (1.29 MB). B+ is still 2.9× under A.

### Found after the test run (the count above is not changed)

1. **Role links in the shared cue lists cost recall.** The frozen B+ put its
   extra links (a pronoun read as "Pierre", "wolves" as "wolf") into the same
   word → event lists that recall uses. On War and Peace that makes recall
   96.44% instead of 97.42%, and recognising a repeated event counts them, so
   174 fewer neurons. Moved into lists of their own inside the role part
   (+0.19 MB): recall 97.42% and find 100% again, the same events as B.
   That version on the test sets (shown only): bAbI the same (94.09%, 17),
   MC160 68.33%, MC500 54.33%; memory 4.07 MB; phone little core worst of 3
   19.8k against A's 14.9k (A had one slow run; same-run ratios 1.10–1.33).
2. **The book harness compared word numbers across vocabularies.** B+'s
   reader adds words (base forms) to B's vocabulary, so the numbers no longer
   lined up with A's and recall/find came out 34.6% / 9.0%
   (`results/understand_book.txt`). That was the test, not the memory: the
   harness now matches words by spelling (the same numbers when the
   vocabularies agree, so ctxwin's results do not move).
3. **A workspace bug in the neuron context window** (CONTEXT_COMPARE.md): see
   the correction there.

Results: `sandbox/neurons/results/understand_*.txt` (`*_final` = after 1–2).
