# Decisions: Mimir against Jev and Laya

Owner (2026-10-12): target Jev- and Laya-level reading and understanding — no
text generation — "while sticking with our architecture". This page fixes the
rules **before any code is written**; results go below the line, whatever they
are.

## The benchmark

**Typed Decisions** (huggingface.co/datasets/LocalLLaMA/typed-decisions,
Apache-2.0; fetched by `tools/decide_fetch.py`). Four workflows (agent traces,
customer service, invoices, security incidents). A case is a JSON state (numbers,
flags, names, lists, sometimes a conversation) and five typed questions:
`noul` (yes/no), `choice` (one of 4–5), `score` (0–3). Every answer is a
probability distribution. Gold is the mean of three samples from a ~4B teacher.
`train` 1,200 cases, `test` 400 cases = 2,000 decisions.

Scored by `tools/decide_score.py`: accuracy (top option against the gold label;
ties share the credit), KL from gold, Brier (sum of squared differences from the
gold distribution). Check: on the leaderboard's reference rows it gives Uniform
KL 0.444 and Brier 0.238 exactly; accuracy and the Prior row come within
1.3 points (the leaderboard does not say how it breaks ties or builds its
Prior). So against **published** numbers a win needs more than **2 points**;
against rows we compute ourselves (Prior, Uniform), more than 1.

## Who it is measured against (published, not run by us)

| | mode | accuracy | KL | Brier |
|---|---|---|---|---|
| Jev 1.13.0 (TypeSafe, closed) | zero-shot | 0.727 | 1.442 | 0.148 |
| Laya (Convai, 421M) | zero-shot | 0.362 | – | – |
| Laya typed-decisions | fitted on `train` | 0.766 | – | – |
| ModernBERT-base specialist (149M) | fitted on `train` | 0.646 | 0.223 | 0.119 |
| MiniLM-L6 specialist (22M) | fitted on `train` | 0.587 | 0.262 | 0.143 |
| teacher self-agreement (ceiling) | – | 0.735 | – | – |
| Prior / Uniform (computed by us) | – | 0.483 / 0.318 | 0.397 / 0.444 | 0.197 / 0.238 |

## Mimir, two modes, on the neuron architecture

**Reading a case.** The JSON is read into neurons: every leaf becomes an event
of feature neurons — its key words and its value (a word, a yes/no, or a number
felt as sign and size: 0, 1, 2–3, 4–7, … on a doubling scale); free text (a
customer's messages, a task) is read word by word as content words; a list
gives a count. Two generic comparisons, as a reader notices them: two numbers
under the same key name in different places (an invoice total and an order
total) give "same / a little different / very different"; a value that appears
twice in the state (an invoice id among earlier ones) gives "seen before".
Nothing specific to one workflow is written in.

**Z — zero-shot** (as Jev and Laya base): nothing from `train`. Each option is
scored by how close, in word meanings (data/meaning), its description is to the
case's words; the closeness is turned into probabilities.

**T — learns from `train`** (as Laya typed-decisions and the specialists):
Mimir reads the 1,200 training cases with their gold answers, once, as
experience. Two brain mechanisms, no backpropagation:

1. **Hebbian links**: every feature neuron grows a link to every answer neuron
   it fired with, strengthened by the answer's gold probability. A new case's
   active features vote through their links (log of how much more often the
   answer came with the feature than without).
2. **Episodes**: each training case is kept as an event neuron; a new case
   recalls the most similar ones through its features' cue lists (rare
   features count more), and their gold answers vote.

The two are mixed, and the mix, the number of episodes recalled, the smoothing
and one temperature per question type are chosen by **4-fold cross-validation
on `train` only**. The test set is run once.

## Tests

| cell | test | score |
|---|---|---|
| D1 | accuracy, zero-shot (Z) | against Jev 0.727, Laya 0.362, Prior |
| D2 | accuracy, learned from train (T) | against Laya typed-decisions 0.766, ModernBERT 0.646, MiniLM 0.587 |
| D3 | KL from gold (T) | against ModernBERT 0.223, MiniLM 0.262, Prior |
| D4 | Brier (T) | against ModernBERT 0.119, MiniLM 0.143, Prior |
| D5 | speed: ms per case (five decisions), S22 Ultra little core, worst of 3 | shown beside Laya's 32.8 ms (T4 GPU) and Jev's 236–710 ms (hosted); not a like-for-like race |
| D6 | size: program + base files + learned memory | against Laya 421M parameters |

Also shown, not scored: accuracy by workflow and question type; Z and T on the
`train` folds; what the learned memory holds.

## Who wins

A cell is won by more than 2 points against published numbers (1 against rows
we compute), or 10% for speed and size; otherwise a tie. Counted for Mimir
against each of Jev and Laya where they have a number, and against the Prior.
The count is given as it is, with the reasons.

---
