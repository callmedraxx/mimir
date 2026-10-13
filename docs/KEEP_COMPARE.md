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

## Results

(to come)
