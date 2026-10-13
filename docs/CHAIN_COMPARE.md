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

## Results

(to come)
