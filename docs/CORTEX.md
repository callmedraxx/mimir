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
* Text: development on `data/shelf/enwiki_00.txt` (on the droplet); the final run on the
  school shelf (2.2 GB) in Colab, all four cells on the same text.

**v0 works if** the chosen cell (K1 or K2, by dev) beats K0 on SimLex-999 and MEN by more
than 0.10 and on T2 by more than 5 points. B's numbers are given beside it as they come out.

## Results

(to be filled in)
