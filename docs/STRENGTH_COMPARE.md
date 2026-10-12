# What is read and used more sticks longer

Owner (2026-10-12): "ok let's read more. … repeated tasks or practiced tasks stick
with the brain. How can we use that in Mimir: what it reads more, or does more,
should stick longer, while what it doesn't use often fades faster."

This page fixes the rules **before any code is written**; results go below the
line, whatever they are.

## Strength (fixed now, not tuned)

Each fact gets a **strength** `s` and the **time it was last met** `L`. The clock `T`
moves one step for every sentence read and every practice question.

* **It fades:** what it is worth now, `e = s · 2^(−(T − L) / H)`.
* **Repetition makes it stick longer:** its half-life grows with its strength,
  `H = H0 · s`, with `H0` = 1,000,000 sentences (about 5% of an hour's reading). A
  fact met once fades in about a million sentences; one met 50 times barely fades.
* **Read again:** `s = e + 1`, `L = T`.
* **Used in practice:** after Mimir answers a practice question it is shown the
  right answer, like a student checking. The facts that support the right option
  get `s = e + 2`, `L = T`: doing counts more than reading. The facts behind a wrong
  choice get `s = e · 0.5`.
* **Learned for good (consolidation)** — added by the owner before any test ran
  ("important skills learnt and practiced shouldn't fade, like learning to do maths
  or coding"): a fact that helped answer a practice question right in **three
  different practice sessions** no longer fades and is never forgotten. Its
  strength can still be lowered if it leads to a wrong answer later. Mimir has no
  skills yet (maths procedures, coding), only facts; when skills come, the same
  rule applies to them.
* **Forgetting when full:** the weakest by `e` are dropped until a quarter of the
  room is free (room: 6 million facts, as before).
* **Trust:** when answering, a fact's confidence comes from `e` (before: from its
  read count).

Practice uses **training questions only**: ARC-Easy and ARC-Challenge train
(3,370), OpenBookQA train (4,957), SciQ train without its passage (10,481). It is
spaced: the practice set is cut into one slice per shelf file. After each file, it
practises that file's slice (new questions) and the slice from three files earlier
(a review), so each question comes back once, later. **During a test the brain is
read-only**: no test question changes anything.

## The three brains

All start from today's brain (`data/shelf/mimir.brain`: Simple English Wikipedia
plus the ARC Corpus, 3.08M facts, 81 MB).

| | how it reads more |
|---|---|
| **before** | today's brain, nothing more |
| **A: read more** | reads English Wikipedia (wikimedia/wikipedia 20231101.en, in file order) for one hour; strength = read count, forgetting as before |
| **B: read more + strength** | reads exactly the same text as A (stops at the same byte), with strength, fading and spaced practice as above |

A against before measures reading more. B against A measures strength by use.

## Tests (frozen build bg.c, answer weights unchanged from BACKGROUND_COMPARE.md)

| cell | test | score |
|---|---|---|
| H1 | ARC-Easy test (2,376) | % correct, ties share credit |
| H2 | ARC-Challenge test (1,172) | % |
| H3 | **OpenBookQA** test (500 science questions, 4 options; new here) | % |
| H4 | brain: size on disk, facts kept, time to read (and practise) | bytes, facts, s |

RACE-M and MCTest are left out: background knowledge has weight 0 there, so they
cannot change. Development only on small slices (ARC and OpenBookQA dev) to check
that the code works; no number above is tuned on them.

## Who wins

Each pair (A vs before, B vs A, B vs before): more than 1 point = better, otherwise
a tie. The count is given as it is.

---

## Results

(to come)
