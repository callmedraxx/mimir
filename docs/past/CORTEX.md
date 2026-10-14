# Mimir's cortex, v0

Owner (2026-10-14): "This still feels like we are merely building a lookup table … how do
LLMs learn the same thing?" → "let's build the neocortex first" → "okay" (local, brain-like
learning rules; one backprop model only as a yardstick).

## Why

Mimir's memory (32.5M facts with pointers to where they were read) is a **hippocampus**:
quick, one experience at a time, nothing shared between facts. Answering is finding the
best-fitting fact, i.e. lookup. The story test (`results/story_test.txt`) showed what is
missing: "a blue owl named Pell" does not make Pell an owl, "he" is not Dumi, and nothing
learned about owls carries over to eagles.

A brain gets that from the **neocortex**: a fixed-size sheet where every word or concept
is a *pattern* over many units (about 1–2% active), similar things share units, and
learning is slow, local (each connection changes only from the two cells it joins plus a
"that was wrong" signal), competitive (inhibition: few cells win), and driven by
**prediction**. It learns from the hippocampus replaying what was experienced. Because it
has a fixed size and must predict, it is forced to compress, i.e. to generalise: that is
where "owl is like eagle" comes from. (LLMs get the same pressure from next-word
prediction with backprop.)

## What v0 is

A cortex that learns word patterns by predicting a hidden word from the words around it,
trained by the shelf being replayed through it once. It is not yet tied into answering
(v1: context across sentences, links into `bg arc`).

| part | brain | v0 |
|---|---|---|
| units | cortical columns | N units (2,048 to start) |
| a word | a sparse pattern | k active units (~2%), learned |
| input layer (L4) | input from senses | the patterns of the words around the hidden one, added up |
| association (L2/3) | lateral connections | weights W from the input pattern to a predicted pattern |
| inhibition | interneurons | only the k strongest units stay on (k-winners-take-all) |
| learning | Hebbian + error signal | W changes by (target − prediction) × input, each weight from its own two cells only |
| meaning | overlap of patterns | a word's pattern drifts toward what its contexts predict, so words used alike come to share units |
| homeostasis | cells kept in use | units used too often are damped, too rarely boosted (no collapse into one pattern) |
| replay | hippocampus in sleep | the shelf text is fed through once, in a shuffled order |

## Cells

| cell | what | role |
|---|---|---|
| **K0** | random sparse patterns, no learning | floor |
| **K1** | Hebbian only: a word's pattern = the k strongest of the summed fixed random patterns of the words near it (random indexing) | the simplest cortex |
| **K2** | predictive: the table above | the candidate |
| B | word2vec CBOW, 300 dimensions, backprop (gensim), same text | yardstick only, not for Mimir |

K1 and K2 settings (window, N, k, learning rates, homeostasis) are chosen on dev only.
Same vocabulary for all: the 50,000 most frequent words of the training text (lower case).

## Tests (fixed now, before any code)

| | dev (develop on these) | test (once, at the end) |
|---|---|---|
| T1 how related two words are | WordSim-353 (353 pairs), SimVerb-3500 | **SimLex-999**, **MEN-3000** |
| T2 fill the blank in sentences never trained on | 1,000 items | **2,000 items** |
| T3 fill the blank in a short story (illustrative, few items) | the clock story, 8 items | **3 new stories, 12 items** (`data/cortex/stories_test.tsv`) |

* T1: Spearman correlation between the cortex's similarity (pattern overlap; cosine for B)
  and human ratings. A pair with a word outside the vocabulary counts as similarity 0.
* T2: training skips every 200th paragraph of the shelf; from those, one content word per
  sentence is hidden (seed 12345); first 1,000 = dev, next 2,000 = test. Score: the hidden
  word among the 10 best of the 50,000 (hit@10).
* T3: the story is given as context, then a sentence with a blank; hit@10.
* Text: the school shelf (2.2 GB), rebuilt in Colab. **Everything runs in Colab** (owner,
  2026-10-14: "you're supposed to use Google colab"): the dev grid (`tools/cortex_grid.txt`,
  first 50M replayed words each) and, once frozen (`tools/cortex_final.txt`), the whole shelf
  and the tests (`tools/cortex_colab.py`, `tools/cortex.ipynb`). The T2 items come from the
  school shelf's held-out lines.

**v0 works if** the chosen cell (K1 or K2, by dev) beats K0 on SimLex-999 and MEN by more
than 0.10 and on T2 by more than 5 points. B's numbers are given beside it as they come out.

## Development notes

First tries were run on the droplet (on `enwiki_00`, 10M replayed words, dev sets only)
before the owner moved them to Colab; kept here because they shaped the grid:

* K0 (random): WordSim −0.03, fill-the-blank 0%, as a floor should be.
* K1 (Hebbian): WordSim 0.40 (cosine), fill-the-blank 8.3%, in 31 s. Neighbours it found by
  itself: france → netherlands, switzerland, poland, italy; owl → owls, eagle (aquila),
  stork; blue → yellow, bright, ochre.
* K2 (predictive) as first written **collapsed**: every word's pattern became the same,
  homeostasis did not stop it. Its fill-the-blank looked like 97% because ties were counted
  as hits; the scorer now counts ties against the cortex (K2 then: 0%). With each word
  anchored to its own identity (IDW 0.5 and 1) the identity won instead: patterns stayed
  random (owl → badly, uranium, lynch), fill-the-blank under 1%. One weaker anchor (0.25)
  is in the grid; K2 needs a better design (more capacity than one N×N layer), likely v1.

## Dev grid (Colab, school shelf, first 50M replayed words each; dev sets only)

| cell | settings | WS-353 | SimVerb | fill-the-blank | mean |
|---|---|---|---|---|---|
| K0 | random | −0.050 | 0.035 | 0.1% | |
| K1a | N 2048, K 40, WIN 2 | 0.319 (cos 0.408) | 0.068 (cos 0.076) | 5.5% | 0.180 |
| **K1b** | N 2048, K 40, WIN 5 | 0.417 (**cos 0.546**) | 0.051 (cos 0.050) | 4.8% | **0.215** |
| K1c | N 4096, K 80, WIN 2 | 0.294 (cos 0.409) | 0.072 (cos 0.074) | 5.2% | 0.179 |
| K1d | N 4096, K 80, WIN 5 | 0.456 (cos 0.532) | 0.065 (cos 0.059) | 4.1% | 0.211 |
| K2a | N 1024, K 20, IDW 0.25 | 0.017 | 0.005 | 0.6% | |
| B | word2vec, WIN 2 | 0.431 | 0.157 | 7.2% | 0.220 |
| B5 | word2vec, WIN 5 | 0.456 | 0.172 | 4.8% | **0.225** |

(cos: the same models scored with cosine similarity of the meanings instead of pattern
overlap, `SIM=1`; fill-the-blank does not depend on it.) Results:
`results/cortex_dev.txt`.

* **K2 failed again** (nothing learned; 46 min to train). It is out of v0.
* **K1 learns association, not likeness**: its nearest words are what appears *with* a
  word (france → its départements; owl → hooting; king → queen, throne, crowned), while
  word2vec's are words of the same kind (france → italy, spain; owl → crow, hawk). That is
  why K1 beats word2vec on WordSim-353 (which rates relatedness) and loses on SimVerb (which
  rates likeness). SimLex-999, a test set, also rates likeness: expect K1 to be weak there.

**How the choice is made** (the scorecard did not say how to combine the three dev numbers;
fixed here before any test): the mean of WS-353, SimVerb and fill-the-blank (as a fraction),
best similarity per cell. K1b (cosine) is chosen among the cortex cells, B5 among the
yardsticks. **Frozen** in `tools/cortex_final.txt`: K0, K1b (`SIM=1`), B5, each over the whole
school shelf, then the tests once.

## Results (tests, once; Colab, whole school shelf: 196M replayed words, 9 min for K1b)

| test | K0 (random) | **K1b (cortex, local rules)** | B5 (word2vec, backprop) |
|---|---|---|---|
| SimLex-999 (alike) | 0.028 | **0.185** | 0.398 |
| MEN-3000 (related) | 0.017 | **0.624** | 0.717 |
| fill the blank, 2,000 unseen sentences (hit@10) | 0.00% | **4.00%** | 6.60% |
| stories, 12 blanks (hit@10) | 0 | **0** | 1 (market, rank 6) |

**Against the bar set before the run:** K1b beats K0 on SimLex-999 by 0.157 and on MEN by
0.607 (bar 0.10, passed), but on fill-the-blank by 4.0 points (bar 5, **not passed**).
**v0 does not fully pass.**

* On the whole shelf word2vec is ahead on everything; on the dev runs (50M words) the
  cortex had been ahead on WordSim-353. word2vec gained more from 4× the text.
* What the cortex learned is association: owl → hoot, hooting, screech, horned; blue → red,
  yellow, purple, violet, pigment, tint; water → boiling, bucket, distilled, evaporate;
  king → crowned, courtiers, Uther, Lear. word2vec learned kinds: owl → crow, raven, hawk,
  sparrow; king → prince, emperor, tsar, monarch; france → belgium, italy, spain.
* Neither remembers a story (it is not what v0 does); both put story words a few hundred
  places up (lantern, caterpillars), from the sentence alone.
* What K1b lacks is exactly what word2vec has: **prediction with competition**. word2vec's
  words are pushed apart from the ones that did *not* fit a context; K1b only adds up what
  was seen together. The K2 cells tried to add that with local rules and did not learn.

Results: `results/cortex_test.txt`, `results/cortex_dev.txt`.

---

# v0.1: Hebbian + anti-Hebbian (K3)

Owner (2026-10-14): "let's mix Hebbian learning and anti-Hebbian learning instead of using
backprop: we will reward and penalize. We don't need to rerun word2vec … compare with the
scores we already have."

**K3.** Each word has meaning cells and context cells (D of each). When word t appears with a
neighbour u: **reward** — the links between t's meaning cells and u's context cells grow, by
how much the pair was *not* yet expected (1 − σ(agreement)); **penalty** — for NEG words drawn
at random (that did not appear there), the links shrink by how much they *were* expected
(σ(agreement)). Each change uses only the two cells it joins and the reward/penalty signal
(a three-factor rule); nothing is sent back through layers. With one layer this is close to
word2vec's negative sampling, written as local learning: the comparison asks whether the
local reward/penalty form reaches word2vec's level. Learning rate falls linearly over the one
pass. A word's meaning = its meaning cells (cosine for similarity); filling a blank = how
well a word's meaning cells fit the context cells of the words around the blank.

**Cells** (dev grid, first 50M replayed words, dev sets): K3a D 300 WIN 5 NEG 5; K3b D 300
WIN 2 NEG 5; K3c D 100 WIN 5 NEG 5 (phone-sized: 100 numbers per word).

**Choice, fixed now:** the best mean of WS-353, SimVerb and fill-the-blank (as a fraction),
picked by the script itself, then trained once on the whole school shelf and tested once
(SimLex-999, MEN, fill-the-blank test items, stories). Compared with the test scores already
run: K1b and B5 (word2vec, not rerun). Note: these tests have been looked at once (v0); the
v0.1 numbers are therefore a second look, said here so they are read that way.

**v0.1 works if** K3 beats K1b on SimLex-999 and fill-the-blank and reaches within 0.05 of
word2vec on SimLex-999 and MEN.

**Fix during the dev grid (before any test):** K3a's first dev run gave WS-353 0.562, SimVerb
0.268, but fill-the-blank 0.4%: the blank was scored by how well a word fits its neighbours
(log P(u|w)/P(u)) without how common the word is (log P(w)), so rare words won. Bayes adds
log P(w); the scorer now does, and the grid was started again (shelf kept).

## v0.1 results

Dev grid (first 50M replayed words, dev sets): K3a (D 300, WIN 5) 0.562 / 0.268 / 1.4%;
**K3b (D 300, WIN 2) 0.504 / 0.251 / 10.0%**; K3c (D 100, WIN 5) 0.594 / 0.243 / 1.3%
(WS-353 / SimVerb / fill-the-blank). Chosen by the fixed rule: K3b (mean 0.2850; K3c 0.2833,
K3a 0.2813). Trained on the whole school shelf (198M replayed words, 32 min, 2 threads), tests
once:

| test | K0 | K1b (v0, Hebbian) | **K3b (v0.1, Hebbian + anti-Hebbian)** | B5 word2vec (backprop, from v0) |
|---|---|---|---|---|
| SimLex-999 (alike) | 0.028 | 0.185 | **0.464** | 0.398 |
| MEN-3000 (related) | 0.017 | 0.624 | **0.739** | 0.717 |
| fill the blank, 2,000 sentences | 0.0% | 4.0% | **10.1%** | 6.6% |
| stories, 12 blanks | 0 | 0 | 0 (dog rank 15, lantern 67) | 1 |

**v0.1 works** by the bar set before it: it beats K1b on SimLex-999 (+0.28) and fill-the-blank
(+6.1 points), and is not within 0.05 of word2vec but above it on both SimLex-999 (+0.066) and
MEN (+0.022). (Second look at these tests; see above.)

* Reward + penalty is what was missing. Words are now grouped by kind: owl → owls, eagle,
  magpie, woodpecker, parrot, raven, **bird**; king → emperor, ruler, queen, monarch; france
  → belgium, italy, portugal, spain; blue → red, yellow, purple, magenta, cyan, sapphire.
* Every change is local (the two cells it joins and a reward/penalty signal); nothing is sent
  back through layers. With one layer this is close to word2vec's negative-sampling rule; the
  differences that helped here are the narrow window and reading pairs (skip-gram style).
* It still does not remember a story (not part of v0.x); story words come closer (dog 15th,
  lantern 67th, fisherman 126th) from the sentence alone.
* Size: 2 × 300 numbers per word, 50,000 words: 120 MB (float32), about 30 MB in 8 bits.

Results: `results/cortex_k3.txt`.

