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

## Results (2026-10-13)

Build frozen in 5a469fa before any test ran (checked first: without `ROOMS` it writes
a brain byte-identical to f6d7611's); whole run `tools/topic_run.sh`. The split
(`results/topic_split.txt`): cut-off score −0.0918 from enwiki_00; 667,745 of 4,163,358
articles on topic (16.0%, 2.2 GB); later files hold less science (13–14%) than the
first (20%, by construction).

| cell | test | before | B: all, one room | **C: on topic** | **D: all, own rooms** |
|---|---|---|---|---|---|
| H1 | ARC-Easy (2,376) | **42.89%** | 36.81% | 40.47% | 41.80% |
| H2 | ARC-Challenge (1,172) | 22.17% | 21.37% | 21.03% | 21.75% |
| H3 | OpenBookQA (500) | 17.32% | 18.77% | **19.87%** | 17.67% |
| H4 | brain | 81.3 MB, 3.08M facts | 206.1 MB, 4.95M | 220.4 MB, 5.94M (4,044 for good); 2.3 GB read in 2,539 s | 200.6 MB, 4.68M: subject 3.53M, general 1.15M (4,021 for good); 9.7 GB in 5,064 s |

Practice (training questions, right before being shown the answer): B 37.4%, C 41.4%,
D 40.7%.

| pair | H1 | H2 | H3 | count |
|---|---|---|---|---|
| C vs before | −2.4 hurt | −1.1 hurt | +2.6 better | better 1, hurt 2 |
| D vs before | −1.1 hurt | −0.4 tie | +0.4 tie | tie 2, hurt 1 |
| C vs B | +3.7 better | −0.3 tie | +1.1 better | **better 2, tie 1** |
| D vs B | +5.0 better | +0.4 tie | −1.1 hurt | better 1, tie 1, hurt 1 |
| D vs C | +1.3 better | +0.7 tie | −2.2 hurt | better 1, tie 1, hurt 1 |

### What it shows

* **Both ideas worked against reading everything in one room (B).** Reading on topic
  gave +3.7 on ARC-Easy and +1.1 on OpenBookQA; a room of its own gave +5.0 on
  ARC-Easy while reading exactly the same 9.7 GB as B. With its own room and clock,
  the subject barely moved while 65M general sentences went by.
* **Neither beat the brain it started from on ARC-Easy.** The best, D, is 1.1 below
  (41.8 vs 42.9). Even on-topic Wikipedia, read into the same room, pushes out some of
  the ARC Corpus sentences that ARC-Easy's answers are found in (open book is still
  the strongest part); D's subject room kept more of them.
* **OpenBookQA is best with on-topic reading (C, 19.9%)**, but it is still below
  chance (25%) for every brain, so its differences are small evidence.
* ARC-Challenge stays near 21–22% for all: more facts do not help there; it needs
  combining facts (the next step on the roadmap).
* Next, from this (not tested here): keep what it already knows as its own room (or
  make it permanent), and put new on-topic reading beside it instead of over it.

Results: `results/topic_split.txt`, `topic_read_C.txt`, `topic_C.txt`,
`topic_read_D.txt`, `topic_D.txt`.
