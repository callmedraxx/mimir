# Mimir to Laya level on the no-passage tests

Owner (2026-10-13): "okay let's try to get Mimir to Laya level".

| | ARC-Easy | ARC-Challenge | OpenBookQA |
|---|---|---|---|
| **target: Laya** (as shipped, measured in Colab, VERSUS.md) | 50.08% | 30.46% | 40.00% |
| Mimir, brain E | 44.63% | 22.32% | 18.47% |

## Rules (for every stage)

* Each stage fixes its change here **before** its code is written, is developed only on
  the **dev** sets (ARC-Easy dev 570, ARC-Challenge dev 299, OpenBookQA dev 500; brain E
  never saw them) and, where it learns, on the **train** sets; then the build is frozen
  and the three tests are run **once**, in Colab (`tools/mimir_arc.ipynb`).
* A stage is kept only if it is better on dev (all three dev sets together); it is
  tested and reported either way.
* Laya was run closed-book (only the question and options). Anything Mimir is given
  that Laya was not (e.g. OpenBookQA's own book of 1,326 facts) is reported as a
  separate line, never mixed into the like-for-like comparison.
* Two kinds of gain are kept apart in the reports: **understanding** (Mimir knows or
  reads more, or reads the question better) and **learned test-taking** (weights learned
  from the train sets). ARC-Challenge and OpenBookQA were built by removing questions
  that word association answers; a learned weighting can pick up how those tests were
  filtered rather than science. That is what trained models such as Laya do too, but it
  is said plainly where it happens.

## Stages

1. **Read more of the shelf when answering (open book, wider).** Today only the
   sentences behind the 3 best facts are re-read. Re-read the sentences behind the best
   K facts, K chosen on dev from {3, 10, 20, 40}. Understanding (retrieval).
2. **Read the question's polarity.** "not", "least", "except", "false", "unlikely",
   "cannot": the question asks for the option that is *not* supported, so the support
   score is turned around for that question. Fixed rule, nothing learned. Understanding.
3. **Learned decision.** Each option's parts (fact, joint, open, meaning, chain, and
   new ones: how many sentences support it, how specific the option is, whether its
   words also support the other options) go to answer neurons that learn their weights
   from the train sets (ARC-Easy/Challenge train, OpenBookQA train, SciQ), with a
   three-factor rule as in DECIDE_COMPARE.md. Learned test-taking, reported as such.

   Fixed before code (2026-10-13):
   * **Inputs per option** (11): the five parts (fact, joint, open, meaning, chain); the
     2nd and 3rd best re-read sentence (open2, open3: more than one sentence agrees);
     option specificity (mean rarity of its own words); option length (its own words);
     and, for each option, open and joint **minus the mean of the other options**
     (does this option stand out, or do all of them match the question equally).
     The question's polarity multiplies all inputs by −1, as in stage 2.
   * **Answer neurons**: one weight per input, shared by all options; an option's
     activity is the weighted sum; the options compete (softmax). Learning is the
     local three-factor rule `Δw = η · input · (reward − expected)`, i.e. the
     log-score reward Laya's training also uses, without backpropagation (one layer).
     η = 0.05, 20 passes over the training questions in a fixed shuffled order, inputs
     scaled to unit spread on the training questions.
   * **Training questions**, chosen on dev between two sets: (a) SciQ dev + test
     (1,771, never practised by brain E: clean), (b) (a) plus ARC-Easy/Challenge train
     and OpenBookQA train (practised by brain E, so their facts are stronger there than
     on new questions).
   * Kept if dev (all three together) beats stages 1–2 (31.18%).

4. **Facts that keep their relation** (cause → effect, kind of, part of, has property,
   needs, used for) and chains that follow relations. Understanding (reasoning).
5. **OpenBookQA open-book** (separate line): the test's own book read into the brain.

---

## Results

### Stages 1 and 2 (dev, 2026-10-13)

Dev, all three dev sets together (1,369 questions); `sandbox/neurons/laya_stage_dev.py`
on the dumps `results/laya1_dev_open*.txt`:

| re-read behind | without polarity | with polarity |
|---|---|---|
| 3 facts (as before) | 30.12% | 30.74% |
| 10 | 30.08% | 30.56% |
| **20** | 30.63% | **31.18%** |
| 40 | 30.63% | 30.81% |

Chosen: **20 facts and polarity on** (+1.1 points on dev, ARC-Challenge dev 19.0 → 21.2).
Frozen with `tools/mimir_arc.env`; test below, once, in Colab.
