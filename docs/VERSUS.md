# Mimir against GPT-2 and Laya on the no-passage tests

Owner (2026-10-13): "test it against GPT-2 … did we run Laya against those tests? … use
[Colab in the shared browser] for all the tests, including Mimir and GPT-2, going forward".

All three were run in Google Colab, on the same questions (`data/arc_tests/`, copies of
the test sets Mimir is scored on), each test run once, nothing trained on them. Ties
share credit. Chance is about 25% on all three tests.

| system | size | ARC-Easy (2,376) | ARC-Challenge (1,172) | OpenBookQA (500) | speed | runs on |
|---|---|---|---|---|---|---|
| **Mimir, brain E** | 156 MB brain + 390 MB shelf | 44.63% | 22.32% | 18.47% | 31–53 ms/q | **1 CPU core** |
| GPT-2 small | 124M parameters | 43.81% | 19.03% | 12.80% | 44–47 ms/q | T4 GPU |
| GPT-2 medium | 355M | 49.07% | 21.50% | 16.20% | 84–86 ms/q | T4 GPU |
| GPT-2 large | 774M | 53.16% | 21.67% | 17.00% | 121–140 ms/q | T4 GPU |
| GPT-2 XL | 1.5B | **58.29%** | 25.00% | 18.20% | 188–245 ms/q | T4 GPU |
| **Laya** (as shipped) | 421M | 50.08% | **30.46%** | **40.00%** | 37–39 ms/q | T4 GPU |

GPT-2 is scored the way language-model papers do (lm-evaluation-harness): each option's
log-probability after `Question: …\nAnswer:`, highest total (acc). Scored per character
instead (acc_norm) it gets, small → XL: ARC-Easy 39.9 / 44.3 / 47.2 / 51.9, ARC-Challenge
22.4 / 25.2 / 25.0 / 28.6, OpenBookQA 26.8 / 29.6 / 29.6 / 29.4. GPT-2 small's numbers match
the published ones (ARC-Easy 43.8, ARC-Challenge 19.0 / 22.7 normalised).

Laya: one pick-one question per test question (the question as text, the options A, B,
C, …), checkpoint `convaiinnovations/laya`; Laya warns that some of its confidence
values are uncalibrated (see the result file).

## What it shows

* **Mimir is at GPT-2 level without a neural network**: above GPT-2 small on all three
  tests, above GPT-2 large on ARC-Challenge and OpenBookQA, on one CPU core.
* **ARC-Easy (knowing the fact) grows with size** for GPT-2: 44 → 49 → 53 → 58%. Mimir's
  equivalent is what it has read and kept (29% before reading, 45% now).
* **ARC-Challenge and OpenBookQA hardly move with GPT-2's size** (19 → 25%, 13 → 18%):
  they were built to defeat word association, which is how GPT-2 and Mimir both pick.
* **Laya is the only one clearly above chance on them** (30.5%, 40.0%): it was trained to
  decide between options for a question, not to continue text.

## How to run them (Colab, from the shared browser)

| notebook | what | runtime |
|---|---|---|
| `tools/mimir_arc.ipynb` | builds Mimir, downloads brain E from the `brain-E` release (checked by SHA-256), runs the tests | CPU |
| `tools/gpt2_arc.ipynb` | GPT-2, all four sizes | T4 GPU |
| `tools/laya_arc.ipynb` | Laya as shipped | T4 GPU |

The packed brain (`bg pack`) keeps only the shelf sentences the brain points to; it
gives exactly the scores of the full 3.9 GB shelf (checked on the droplet and in Colab).

Results: `sandbox/neurons/results/mimir_arc_colab.txt`, `gpt2_arc.txt`, `laya_arc.txt`.
