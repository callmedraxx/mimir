# Reading a lot first: background knowledge, before and after

Owner (2026-10-12): "we're going to make it read and read and read, but before it
starts reading let's take the test now and see the score, then after it reads for
like an hour, test it again."

This page fixes the rules **before any code is written**; results go below the
line, whatever they are.

## The experiment

One build, frozen before any test is run. Two brains:

| | brain |
|---|---|
| **before** | today's Mimir: word meanings only, nothing read yet (empty knowledge) |
| **after** | the same build after reading the shelf below |

The only difference between the two runs is what Mimir has read. The tests are run
with the before brain first, then reading starts, then the same tests are run again
with the after brain.

## What reading keeps (the owner's rule: understanding and where it is, not the words)

* Each sentence is read into facts (who / does what / to what / is what / where /
  when, negation), as in `know.c`.
* **Merge:** a fact already known is strengthened (counted), not stored again. It
  keeps at most two pointers to where it was read (file and byte offset of the
  sentence). The words of the text are let go.
* **Forget:** when the brain fills its room (set before reading, see below), facts
  read only once are forgotten, pointers included; what was read again and again
  stays.
* **Know where to find things:** cue lists word → facts.
* The brain is saved to one file; its size is reported. The shelf stays on disk (the
  books on the shelf, not counted), read again only through pointers.

## The shelf, read in this order, for up to one hour

1. **Simple English Wikipedia** (dump of 2026-10-01): general knowledge in plain English.
2. **ARC Corpus** (AllenAI, 14M science sentences, released with ARC as its
   background reading): the science shelf.

Reading stops after one hour or at the end of the shelf, whichever comes first.
Room: at most 6 million facts in the brain at any time (fits this 3 GB droplet and a phone).

## Tests

| cell | test | why | score |
|---|---|---|---|
| G1 | **ARC-Easy** test (2,376 grade-school science questions, no passage) | can only be answered from background knowledge | % correct (ties share credit) |
| G2 | **ARC-Challenge** test (1,172) | the harder ones | % correct |
| G3 | RACE-M test (1,436) | reading exams that need knowledge of the world | % correct |
| G4 | MC160 + MC500 test (840) | children's stories | % correct |
| G5 | brain: size on disk, facts kept, reading speed | the owner's rule | bytes, facts, MB/s |

Answering without a passage (G1, G2): the question with the option put where it
asks becomes a statement; it is scored by (a) how well it agrees in meaning with a
known fact, (b) one known fact that covers both the question and the option, (c)
open book: the pointers of those facts lead back to the shelf and those sentences
only are re-read, (d) how close in meaning the option's words are to the question's
(the only part that works with the before brain). With a passage (G3, G4): K open
(KNOW_COMPARE.md) plus the same background score with one weight.

Development: ARC-Easy and ARC-Challenge train + dev, RACE-M dev, MCTest train + dev,
with a development brain that reads 5% of the shelf. Weights chosen there and frozen
before the before-run. The test sets are not used for development.

## Who wins

After against before, per cell: more than 1 point better = reading helped; more
than 1 point worse = reading hurt; otherwise no change. The count is given as it is.

---

## Results (2026-10-12)

Harness `sandbox/neurons/bg.c`, frozen in 7450f8f before any test was run; whole run
`tools/bg_run.sh`. Weights chosen with a dev brain that read 5% of the shelf: ARC
answer weights (fact 1, joint 0.5, open 2, meaning 4) on ARC train + dev; the
background weight on passage questions on RACE-M dev and MCTest train/dev, where
background **hurt at every weight tried** (51.5% at 0 down to 37.0% at 4), so it was
set to 0. RACE-M and MCTest are therefore the same before and after by construction.

| cell | test | before (nothing read) | after reading | change |
|---|---|---|---|---|
| G1 | ARC-Easy (2,376) | 28.87% | **42.89%** | **+14.0: reading helped** |
| G2 | ARC-Challenge (1,172) | 22.68% | 22.17% | −0.5: no change |
| G3 | RACE-M (1,436) | 41.09% | 41.09% | no change (weight 0) |
| G4 | MC160 / MC500 (240 / 600) | 63.75% / 52.83% | 63.75% / 52.83% | no change (weight 0) |
| G5 | brain | empty | **81.3 MB** on disk: 3.08M facts, 961k words | 4.8% of the 1,694 MB read |

**Count: helped 1 (ARC-Easy), no change 3, hurt 0.** Chance on ARC is about 25%.

Reading: the whole shelf (1,694 MB, 18.2M sentences, 30.6M facts) in **669 s
(2.53 MB/s, one core)**, so it ran out of shelf after 11 minutes, not an hour. The
room (6M facts) filled three times; forgetting dropped 22.9M facts read too few
times and kept 3.08M, the ones read again and again.

### What it shows

* **Background knowledge works for questions without a passage.** Before reading,
  ARC-Easy is near chance (only word meanings); after, 42.9%. Most of it comes from
  open book: the facts tell it where to look on the shelf, it re-reads those
  sentences (open alone 42.0%).
* **ARC-Challenge does not move.** Those questions were chosen by ARC's makers as
  the ones retrieval and word co-occurrence get wrong; finding the right sentence is
  not enough there, it takes reasoning over several facts.
* **On stories (RACE-M, MCTest) background did not help in this form**: the answer
  is in the passage, and general facts pulled the choice away from it. Using
  knowledge for inference, not as extra evidence, is still to be built.
* For scale: ARC's published retrieval baseline (IR over a large web corpus) scores
  about 62% on ARC-Easy; Laya and Jev were not measured on ARC here.
* Understanding ended much smaller than the text (4.8%), because forgetting finally
  works: what was read once fades.

Results: `sandbox/neurons/results/bg_before.txt`, `bg_read.txt`, `bg_after.txt`.
