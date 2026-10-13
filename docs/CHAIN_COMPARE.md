# Combining facts: one fact leads to another

Owner (2026-10-13): "combine facts first" (before capping what is kept for good).

ARC-Challenge has stayed at 21–23% whatever Mimir read: its questions were chosen as
the ones that finding one matching sentence gets wrong. Many need two facts joined
by something they share, e.g. "Which would help a plant live in a desert?" → *deserts
have little water* + *waxy leaves keep water in*: the link is *water*, a word in
neither the question nor the answer.

This page fixes the rules **before any code is written**; results go below the line,
whatever they are.

## The new part: a chain (fixed now)

For each option, besides the four parts it has (fact, joint, open, meaning):

* **question facts**: up to 200 known facts holding the question's words, ranked by
  how many of them they hold (rare words counting more) and their confidence;
* **option facts**: the 200 it already finds for the option (facts holding the
  option's own words);
* a **link**: a question fact and an option fact (two different facts) that share a
  word that is in neither the question nor the option, and is not a common word
  (held by fewer than 1 in 20 facts, `idf ≥ 3`);
* **chain** = the best, over linked pairs, of
  `conf(q-fact) · conf(o-fact) · (cover(question by q-fact) + cover(option by o-fact)) / 2`,
  with cover the same meaning-weighted coverage the joint part uses.

The score becomes `fact + 0.5·joint + 2·open + 4·meaning + W·chain`: the earlier weights
unchanged, only W new. With W = 0 the build must give exactly brain E's earlier
results (44.63 / 22.32 / 18.47). Practice and reading do not use the chain, so brains
read by this build are unchanged.

## Development (tests untouched)

Brain: `data/shelf/base.brain` (brain E). Its practice used the **train** sets, so
development uses only the **dev** sets, which it never saw: ARC-Easy dev (570),
ARC-Challenge dev (299), OpenBookQA dev (500). The chain's details may be developed
there. **W** is chosen from {0, 0.25, 0.5, 1, 2, 4, 8} by the most dev questions right
(all three together) and then frozen. If no W > 0 beats W = 0 on dev, the chain is
reported as not helping and the tests are still run once with the best W > 0, so
the result is on record.

## Tests

The same frozen build and brain E, tests run once: **without the chain** (W = 0) and
**with it** (the chosen W).

| cell | test |
|---|---|
| H1 | ARC-Easy test (2,376) |
| H2 | **ARC-Challenge test (1,172)**: the one this is for |
| H3 | OpenBookQA test (500) |
| H4 | time per question |

## Who wins

With against without, per cell: more than 1 point = better, otherwise a tie. The
count is given as it is.

---

## Results (2026-10-13)

Build frozen in 0148e47 before any test ran; whole run `tools/chain_run.sh`.

**Development** (dev sets only, 1,369 questions; dumps in `results/chain_dev*.txt`).
Four versions of the chain were tried: the fits added (as written above) or
multiplied (both facts must fit), link words rarer than 1 fact in 20 or in 150. Every
version stayed within ±9 questions of no chain, and the chain alone was below chance
on ARC-Challenge dev (16–19%). Chosen by the rule (most dev questions right):
multiplied, link rarer than 1 in 150, **W = 1** (421.3 right against 412.3, +0.7
points; ARC-Challenge dev itself went down, 19.0 → 17.5).

**Tests**, once:

| cell | test | without the chain | with it (W = 1) | change |
|---|---|---|---|---|
| H1 | ARC-Easy (2,376) | 44.63% | 44.45% | −0.2 tie |
| H2 | **ARC-Challenge (1,172)** | 22.32% | 22.55% | +0.2 tie |
| H3 | OpenBookQA (500) | 18.47% | 18.87% | +0.4 tie |
| H4 | time per question | 41–61 ms | 57–105 ms | slower |

Without the chain the build gives exactly brain E's earlier results, as required.

**Count: tie 3. Combining facts through a shared word did not help.**

### What it shows

* The chains found are real (two facts sharing a word, e.g. *(condition air system)*
  + *(progress condition fog)*) but the links are generic words, so a chain is
  association in two steps. Facts are kept as a verb and its words, without what
  relates them (causes, is a kind of, has, needs), so there is nothing to reason
  along.
* **Why ARC-Challenge and OpenBookQA sit below chance** (found on dev, after the
  freeze): both tests were built by removing every question that retrieval and
  word-association methods answer right. Mimir's answering is such a method, so its
  top choice is drawn to the wrong options by construction. On ARC-Challenge dev,
  the option it scores *lowest* is right 27.3% of the time against 19.6% for its
  top choice; on OpenBookQA dev 30.0% against 18.8%. This is not used: picking the
  least associated option is a property of how the tests were filtered, not
  understanding.
* One real gap it showed: questions that ask for the opposite ("not", "least",
  "except", "false"; 26 of 299 ARC-Challenge dev questions) score 12.5%, because the
  question's polarity is ignored.
* What could move these tests is reasoning that association cannot fake: facts
  that keep their relation (cause → effect, kind of, part of, property), chains
  that follow those relations, and reading the question's polarity.

Results: `sandbox/neurons/results/chain_without.txt`, `chain_with.txt`.
