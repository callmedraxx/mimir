# What it already knows, learned for good

Owner (2026-10-13), after TOPIC_COMPARE.md showed that new reading, even on topic,
pushes out part of what the starting brain knew: "okay let's mark existing knowledge
as learned for good".

This page fixes the rules **before any code is written**; results go below the
line, whatever they are.

## The change (fixed now, not tuned)

When Mimir continues from a brain, **every fact in that brain is learned for good**,
exactly as if practice had consolidated it (STRENGTH_COMPARE.md): it no longer fades
and is never forgotten. Its strength is what it was (its read count). It can still
be weakened by practice when it leads to a wrong answer, but it stays. Everything
else (strength, fading, practice, rooms) is unchanged.

The starting brain is `data/shelf/mimir.brain` (3.08M facts), so in a 6M room about
2.9M places are left for new reading, which competes for them as before.

## The brains

| | reads | rooms | starting knowledge |
|---|---|---|---|
| **before** | nothing more | | (earlier: 42.89 / 22.17 / 17.32) |
| **C** | on topic only (`sci_*`) | one room, 6M | fades as usual (earlier result) |
| **D** | all (`sci_00`, `gen_00`, …) | subject 4.5M, general 1.5M | fades as usual (earlier result) |
| **E** | as C | as C | **learned for good** |
| **F** | as D | as D | **learned for good** |

Same text, practice questions and order as C and D; only the starting knowledge's
status differs. In F the subject room holds the 3.08M kept facts, so only about 1.4M
places are left there for new on-topic reading.

Without `KEEP` set, the build must give exactly the earlier results (checked
byte-for-byte on a small read before any test).

## Tests (unchanged)

H1 ARC-Easy test (2,376), H2 ARC-Challenge test (1,172), H3 OpenBookQA test (500), H4
brain size, facts, time. The brain is read-only during tests.

## Who wins

Pairs: E vs before, F vs before, E vs C, F vs D, F vs E. More than 1 point = better,
otherwise a tie. The count is given as it is.

---

## Results (2026-10-13)

Build frozen in bcf4723 before any test ran (checked first: without `KEEP` it writes
brains byte-identical to 5a469fa's, with and without rooms). While checking, a small
run showed that forgetting could drop facts learned for good when they filled more
than 3/4 of a room, and that a room full of them would never stop forgetting; both
were fixed before freezing (such facts are never dropped; a room full of them keeps no
new fact). At this run's sizes neither case arises. Whole run `tools/keep_run.sh`.

| cell | test | before | C | D | **E: as C, kept** | **F: as D, kept** |
|---|---|---|---|---|---|---|
| H1 | ARC-Easy (2,376) | 42.89% | 40.47% | 41.80% | **44.63%** | 44.51% |
| H2 | ARC-Challenge (1,172) | 22.17% | 21.03% | 21.75% | 22.32% | **22.95%** |
| H3 | OpenBookQA (500) | 17.32% | **19.87%** | 17.67% | 18.47% | 18.32% |
| H4 | brain | 81.3 MB, 3.08M | 220.4 MB, 5.94M | 200.6 MB, 4.68M | 167.2 MB, 4.72M; 2.3 GB in 2,166 s | 213.4 MB, 5.34M (subject 4.02M, general 1.32M); 9.7 GB in 4,947 s |

Practice (right before being shown the answer): C 41.4%, D 40.7%, **E 43.1%, F 42.6%**.
Facts newly learned for good through practice: E 240, F 252 (C 4,044, D 4,021): most
facts practice leans on were already kept.

| pair | H1 | H2 | H3 | count |
|---|---|---|---|---|
| E vs before | +1.7 better | +0.2 tie | +1.2 better | **better 2, tie 1** |
| F vs before | +1.6 better | +0.8 tie | +1.0 tie | **better 1, tie 2** |
| E vs C | +4.2 better | +1.3 better | −1.4 hurt | better 2, hurt 1 |
| F vs D | +2.7 better | +1.2 better | +0.6 tie | better 2, tie 1 |
| F vs E | −0.1 tie | +0.6 tie | −0.2 tie | tie 3 |

### What it shows

* **Keeping what it knew is what made reading more pay off.** For the first time a
  brain that read more beats the one it started from, and loses nothing: E +1.7 on
  ARC-Easy and +1.2 on OpenBookQA, F +1.6 on ARC-Easy. Against their twins that let it
  fade (C, D), ARC-Easy gains 2.7–4.2 points and ARC-Challenge 1.2–1.3.
* **With knowledge kept, rooms add nothing measurable** (F vs E: three ties). On-topic
  reading alone (E) does as well, reads 2.3 GB instead of 9.7, in less than half the
  time, with the smallest brain (167 MB).
* OpenBookQA is still below chance (25%) for every brain; C, which let old knowledge
  fade, stays best there, so those differences are weak evidence either way.
* ARC-Challenge is still near 22–23%: the next step is combining facts.

Results: `sandbox/neurons/results/keep_read_E.txt`, `keep_E.txt`, `keep_read_F.txt`,
`keep_F.txt`.
