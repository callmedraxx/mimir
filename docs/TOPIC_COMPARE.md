# Reading on its topic, and a room of its own for it

Owner (2026-10-13), after STRENGTH_COMPARE.md showed that reading all of English
Wikipedia pushed the science out: "do it, let's see". The two ideas from that result:
read **related topics** (what it is meant to be good at), and give that topic **its
own room** so a large general read cannot wipe it.

This page fixes the rules **before any code is written**; results go below the
line, whatever they are.

## What is "on topic" (fixed now, not tuned)

The topic is science, the subject of the tests below. An English Wikipedia article is
on topic if its words look more like the science shelf Mimir already read (the ARC
Corpus) than like general reading (Simple English Wikipedia):

* words: lowercase runs of letters; a word counts only if it was seen at least 20
  times in the two shelves together;
* each word's weight `log(p_science(w)) − log(p_general(w))` (counts + 1);
* an article's score: the mean weight of its counted words; articles with fewer than
  20 counted words are general;
* **the top 20% of articles by score are on topic.** The cut-off score is measured
  once on the first file (enwiki_00) and then used unchanged for all 28 files.

The tool (`tools/topic_split.c`) splits each `enwiki_NN.txt` into `sci_NN.txt` (on
topic) and `gen_NN.txt` (the rest), same order, nothing else changed. No test or
practice question is used to choose the topic.

## The brains

All start from today's brain (`data/shelf/mimir.brain`, 3.08M facts). All use the
strength rules of STRENGTH_COMPARE.md unchanged (fading, read again, practice,
learned for good after three sessions, the same 18,808 practice questions, spaced in
one slice per topic file with a review three files later).

| | what it reads | rooms |
|---|---|---|
| **before** | nothing more | (from the earlier run: 42.89 / 22.17 / 17.32) |
| **B** | all of English Wikipedia (from the earlier run, not rerun) | one room of 6M |
| **C: on topic** | only `sci_00` … `sci_27` | one room of 6M |
| **D: on topic + its own room** | `sci_00`, `gen_00`, `sci_01`, `gen_01`, … (all of it, same text as B) | **subject room 4.5M** (everything it knew before + on-topic reading) and **general room 1.5M** (the rest) |

Rooms in D:
* Each room has **its own clock**: the subject room's clock moves with on-topic
  sentences and practice questions, the general room's with general sentences.
  Reading about football does not make physics fade; only time spent on the subject
  does.
* When a room is full, only that room forgets (weakest by strength, down to 3/4 of
  that room). The general room can never push a subject fact out.
* A fact met again in either kind of reading is strengthened where it already lives.
* Answering uses both rooms the same way. The total room (6M) is the same as B and C.

Without `ROOMS` set, the build must give exactly the earlier results (checked
byte-for-byte on a small read before any test).

## Tests (unchanged)

H1 ARC-Easy test (2,376), H2 ARC-Challenge test (1,172), H3 OpenBookQA test (500), H4
brain size, facts, time. Same answer weights as before. The brain is read-only
during tests.

## Who wins

Pairs: C vs before, D vs before, C vs B, D vs B, D vs C. More than 1 point = better,
otherwise a tie. The count is given as it is.

---

## Results

(to come)
