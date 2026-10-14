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

## Results (2026-10-12)

Engine: `sandbox/neurons/decide.c`, frozen in afbf448 before the test set was
opened. Developed on `train` only, by 4-fold cross-validation (first version
0.644; recall of similar cases weighted by each feature's learned relevance to
the question, 0.724; list items also read by their label and as the last one,
0.728). The test set was run once.

| cell | | Mimir | Jev | Laya | others | result |
|---|---|---|---|---|---|---|
| D1 | accuracy, zero-shot | 0.371 | **0.727** | 0.362 | Prior 0.483 | Jev wins; tie with Laya; **Prior beats Mimir** |
| D2 | accuracy, learned from train | 0.726 | (0.727 zero-shot) | **0.766** | ModernBERT 0.646, MiniLM 0.587 | **Laya wins** (4.0 pt); Mimir beats both specialists |
| D3 | KL from gold, learned | **0.154** | 1.442 | – | ModernBERT 0.223, MiniLM 0.262, Prior 0.397 | Mimir beats all with a number |
| D4 | Brier, learned | **0.081** | 0.148 | – | ModernBERT 0.119, MiniLM 0.143, Prior 0.197 | Mimir beats all with a number |
| D5 | speed | 2.0 ms per case, droplet CPU one core (phone not run yet) | 236–710 ms hosted | 32.8 ms T4 GPU | | not like-for-like; phone pending |
| D6 | size | program + base files ~8 MB, learned memory 2.9 MB | closed | 421M parameters | | Mimir far smaller |

**What it means.** Having read the 1,200 training cases once (0.03 s), Mimir
answers at 0.726 — level with Jev's 0.727 (which learned elsewhere, not from
these cases), under Laya trained on the same cases (0.766), above the two
trained specialists, and close to the 0.735 teacher ceiling. Its probabilities
are much closer to the gold than Jev's (KL 0.154 against 1.442). Without
training it has nothing: 0.371, under the Prior. So this is learning from
examples of the same kind of case, not general judgment: a new kind of case
(a workflow it has not read) would get the zero-shot result.

Results: `results/decide_test_*.txt`.

## Laya measured by us (2026-10-12, Google Colab, the same machine for both)

Both run in one Colab notebook (T4 GPU runtime), same test set, same scorer
(`tools/laya_colab.py`, `tools/mimir_colab.sh`, `tools/both_colab.sh`; Laya 0.4.6,
torch 2.11). Laya on the T4 GPU, one `predict` call per case (five questions);
Mimir on one CPU core of the same machine, worst of 3.

| | accuracy | KL | Brier | ms per case | where | size |
|---|---|---|---|---|---|---|
| Laya, as shipped | 0.362 | 0.569 | 0.315 | 124.5 | T4 GPU | 421M parameters (846 MB) |
| Laya typed-decisions (fitted on train) | **0.766** | **0.117** | **0.061** | 131.7 | T4 GPU | 421M parameters |
| Mimir, learned from train | 0.726 | 0.154 | 0.081 | **1.53** | 1 CPU core | 2.9 MB learned + ~8 MB program and base |
| Mimir, zero-shot | 0.371 | 0.643 | 0.337 | – | 1 CPU core | – |

Laya's published accuracies reproduce exactly (0.362, 0.766). Trained Laya
beats Mimir on accuracy (4.0 pt), KL and Brier; Mimir answers about 85× faster
on a CPU core than Laya on a GPU, with about 1/100 of the memory. Untrained,
both are at the level of guessing.

---

## Next tests (rules fixed before running, 2026-10-12)

**G1 — a kind of case never seen (leave one workflow out).** For each workflow,
Mimir learns from the `train` cases of the **other three** only, then answers
that workflow's `test` cases; the four results together cover all 2,000 test
decisions. A question is the same answer neurons wherever it appears if its
name, type and options are identical (only `urgency` is, in all four); a
question it has never learned is answered zero-shot (Z). Settings: the ones
already chosen for T (no new tuning). Compared with Mimir Z (0.371), Laya as
shipped (0.362, measured by us) and Jev (0.727, published, never trained on
these workflows). Win by more than 2 points against published, 1 against our
own runs.

**G2 — how fast it learns (learning curve).** Mimir learns from n training
cases per workflow, n = 10, 25, 50, 100, 200, 300 (all), drawn at random with 5
different draws for n < 300 (mean and worst shown), and answers the whole test
set; settings as T. Shown beside the Prior learned from the same cases, and
the fitted models at n = 300 (Laya typed-decisions 0.766, ModernBERT 0.646,
MiniLM 0.587). Reported: the smallest n at which Mimir's **worst** draw passes
the Prior, MiniLM and ModernBERT.

## Results: G1 and G2 (2026-10-12)

**G1 — unseen workflow: 0.361** (KL 0.654, Brier 0.341). Per workflow: agent
traces 0.282, customer service 0.330, invoices 0.426, security 0.408. Only
`urgency` (400 of 2,000 decisions) could use anything learned in the other
workflows; the rest were answered zero-shot. Against Mimir zero-shot 0.371:
nothing gained (−1.0 pt). Against Laya as shipped 0.362: tie. **Jev 0.727 wins
by 37 points.** Mimir does not carry what it learned to a new kind of case:
its knowledge is tied to the exact features and questions it saw.

**G2 — learning curve** (5 random draws below 300; worst draw in brackets):

| cases per workflow | Mimir accuracy | KL | Brier | Prior from the same cases |
|---|---|---|---|---|
| 10 | 0.512 (0.491) | 0.347 | 0.183 | 0.459 (0.441) |
| 25 | 0.585 (0.555) | 0.292 | 0.153 | 0.484 (0.454) |
| 50 | 0.637 (0.623) | 0.246 | 0.129 | 0.493 (0.474) |
| 100 | 0.680 (0.665) | 0.201 | 0.105 | 0.491 (0.470) |
| 200 | 0.712 (0.705) | 0.165 | 0.087 | 0.490 (0.481) |
| 300 | 0.726 | 0.154 | 0.081 | 0.483 |

Worst draw passes the Prior at **10** cases per workflow, MiniLM (0.587, fitted
on all 300) at **50**, ModernBERT (0.646, fitted on all 300) at **100**; Laya
typed-decisions (0.766) is not reached at 300. So Mimir learns fast from few
examples of a kind of case, and learns nothing that carries to a new kind.

Results: `results/decide_G1.txt`, `decide_G2.txt`.
