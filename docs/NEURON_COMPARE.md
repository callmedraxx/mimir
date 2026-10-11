# v0.1 vs v0.2 neurons: the scorecard (written before any result)

The owner asked for both neuron designs to be built and compared
objectively, at the **same memory**, to pick the better one, not to prove
either right. The winner is then cut down to see how small it can get. These
rules were fixed and committed **before** either design was run, so neither
can be tuned to the test afterwards.

## What both must do

The job both specs claim, and the one everything else (context window, gist,
cued recall, word order) is built on: **read a book event by event, store
every event, and later get events back**.

* **Input** (identical for both): War and Peace split into events (clauses
  cut at `, ; : . ! ?`), each event kept as its content words (the 150 most
  frequent words of the book dropped; events with fewer than 3 content words
  skipped; at most 12 words kept). Words arrive as vocabulary numbers, the
  same for both (the page stays outside the brain, for both).
* **Output** (identical for both): a set of word numbers.
* v0.1 as specified in `NEURON.md` commit 0f6a119: one neuron per word and
  per event; event content = majority vote of its words' D-bit codes;
  `part-of` synapses event → words, cue synapses word → events, `next`
  synapses event → event; recognise-or-birth by Hamming distance.
* v0.2 as specified in commit cf9e806: neurons hold only state; a word is a
  fixed assembly in the word area (receptor-like wiring); an event is a
  sparse assembly recruited from newborn event-area neurons, wired by
  synapses from its words, back to its words, among itself, and from the
  event before. Recognise by firing; recall by pattern completion.
  Word read-out from the word area is done by the test (like an electrode),
  since v0.2 neurons carry no labels.

## Same memory

Each brain gets a hard **byte budget** counted from its own records:
neurons (meta + activation + content where it has any), synapse blocks
(capacity × 8 B, power-of-two blocks), and any index it keeps. Budgets:
**4 MB, 16 MB, 64 MB**. When full, each uses the pruning its own spec
gives (drop the least used, then oldest, events, 10% at a time; for v0.2 a
night sweep removes synapses to dead neurons). Neither may exceed its budget.

## Tests (the same code calls both)

| # | test | how | score |
|---|---|---|---|
| T1 | **cued recall** | 2,000 events sampled evenly over the book; cue with half of the event's words (rounded up, at least 2); the brain returns a word set | mean F1 against the full event |
| T2 | **old memories** (forgetting) | T1 restricted to events from the first 10% of the book | mean F1 |
| T3 | **recognition** | 2,000 stored events vs 2,000 events from a book it never read (The Adventures of Sherlock Holmes); the brain gives a familiarity score | AUC (0.5 = guessing, 1 = perfect) |
| T4 | **what came next** | given a whole stored event, return the event after it | mean F1 against the true next event |
| T5 | **speed** | storing: tokens of the book per second; T1 recalls per second. Measured on the S22 Ultra prime core **and** a little core, worst of 3 runs | tokens/s, recalls/s |
| T6 | **memory** | bytes actually used, and events still stored at the end | bytes, events |

## Tuning, equal for both

Each design gets the **same number of settings (4)** tried on a **dev book**
(Pride and Prejudice, at 4 MB, so the budget bites), scored on T1 + T2 + T3 + T4.
Its best setting is then frozen and run on War and Peace. Nobody looks at
War and Peace numbers before the settings are frozen.

* v0.1 settings: content bits D ∈ {256, 1024} × cue synapses per word cap
  ∈ {64, 512}.
* v0.2 settings: word assembly k_w ∈ {16, 32} × event assembly k_e ∈ {8, 16}
  (fan-in 2 synapses per word assembly per event neuron, 2 back-projections
  per word, full recurrent wiring inside the event assembly, 2 order synapses
  per neuron).

## Who wins

* **Gate**: a design must store at ≥ 10,000 tokens/s on the phone's little
  core (the owner's floor). A design that fails is out.
* Otherwise, count wins over the 3 budgets × {T1, T2, T3, T4, store speed,
  recall speed} = 18 cells. A cell is a tie when the two are within
  1 point (F1, AUC × 100) or within 10% (speeds).
* The design with more wins is the winner. If wins are equal, the one with
  the better T1 at 4 MB wins (more knowledge in less memory).
* Every number is reported, including the ones the winner loses.

## After

The winner is cut down (smaller records, fewer bits, fewer synapses) and
re-run on the same tests, to find the smallest version that keeps its
scores.

---

# Results (2026-10-11)

Code: `sandbox/neurons/compare.c` (both brains and the tests in one file),
scored by `sandbox/neurons/score.py`. Raw lines in `sandbox/neurons/results/`.

## Mechanics check (not a scored test)

With unlimited memory both work: on Pride and Prejudice v0.2 recalls 96.7%
(v0.1 97.8%) and recognises 99.99% (v0.1 99.6%). So v0.2 does not lose
because it is broken. It loses on **cost**: it needed 20.5 MB for what v0.1
held in 1.7 MB.

## Tuning (Pride and Prejudice, 4 MB; sum of T1–T4)

| v0.1 setting | sum | v0.2 setting | sum |
|---|---|---|---|
| D 256, cap 64 | 3.939 | k_w 16, k_e 8 | **1.425** |
| D 1024, cap 64 | 3.953 | k_w 32, k_e 8 | 1.421 |
| D 256, cap 512 | 3.953 | k_w 16, k_e 16 | 1.166 |
| D 1024, cap 512 | **3.960** | k_w 32, k_e 16 | 1.189 |

Frozen: v0.1 D 1024 cap 512; v0.2 k_w 16, k_e 8.

## War and Peace, S22 Ultra (speeds = worst of 3, little core)

| budget | test | v0.1 | v0.2 | winner |
|---|---|---|---|---|
| 4 MB | T1 recall | 38.1 | 18.7 | v0.1 |
| 4 MB | T2 old | 32.9 | 20.1 | v0.1 |
| 4 MB | T3 recognise | 56.9 | 55.5 | v0.1 |
| 4 MB | T4 next | 13.0 | 2.7 | v0.1 |
| 4 MB | store tok/s | 207,151 | 48,800 | v0.1 |
| 4 MB | recall/s | 16,240 | 5,753 | v0.1 |
| 16 MB | T1 / T2 / T3 / T4 | 96.4 / 95.7 / 99.9 / 99.3 | 32.9 / 25.1 / 64.1 / 14.4 | v0.1 ×4 |
| 16 MB | store tok/s / recall/s | 198,542 / 6,161 | 31,251 / 1,719 | v0.1 ×2 |
| 64 MB | T1 / T2 / T3 / T4 | 96.4 / 95.7 / 99.9 / 99.3 | 63.8 / 28.2 / 82.0 / 55.4 | v0.1 ×4 |
| 64 MB | store tok/s / recall/s | 193,291 / 6,051 | 11,533 / 418 | v0.1 ×2 |

Gate: both pass (v0.1 193 k, v0.2 11.5 k tokens/s on the little core).
**Wins: v0.1 18, v0.2 0, ties 0. Winner: v0.1.**

Events kept: v0.1 holds all of War and Peace (36,613 events) in 13.3 MB;
v0.2 holds 1,040 at 4 MB, 5,339 at 16 MB, 20,823 at 64 MB.

## Why, as far as the numbers show

* **Bytes per memory.** v0.1 ≈ 360 B per event (including its 17 k word
  neurons). v0.2 ≈ 3 KB: every event is 8 neurons, and each one repeats the
  binding to every word (2 in, 2 back per word) plus 7 links inside the
  assembly. Redundancy is what lets an assembly survive losing a neuron; it
  is paid for in bytes. At equal memory v0.2 simply keeps fewer memories.
* **Recognition by firing** walks every synapse of every active word neuron;
  frequent words have long axons. v0.1 walks one capped list per word.
* **What this test does not measure.** Both store exact episodes. v0.2's
  claimed strengths (graceful damage, similarity between memories, learning
  without labels) are not scored here. With memory to spare it recognised
  slightly better (99.99 vs 99.6).
* **One advantage given to v0.2.** Its word identities came free from fixed
  hashed wiring, and its words were read out by the test, not by stored
  synapses.

## The winner cut down (v0.1-lean)

Same behaviour, fewer bytes:
* no content bits (recognise by synapse overlap, Jaccard ≥ 0.85, instead of
  Hamming on 1,024 bits);
* word neuron = its vocabulary number (no meta, no label table);
* event record 8 B (start, count, uses, next as a step);
* part-of synapse 3 B, cue synapse 4 B (24-bit event + 8-bit strength);
* pruning compacts at night.

| War and Peace | v0.1 as specified | v0.1-lean |
|---|---|---|
| memory for the whole book | 13.3 MB | **2.0 MB** (6.6× smaller) |
| bytes per event | ≈ 360 | **≈ 55** |
| T1 / T2 / T3 / T4 | 96.4 / 95.7 / 99.9 / 99.3 | 96.7 / 95.4 / 99.98 / 99.4 |
| at 4 MB T1 | 38.1 | 96.7 |
| at 1 MB T1 / T2 | — | 61.1 / 36.7 (15,912 events kept)¹ |
| store, little core worst | 193 k tok/s | **863 k tok/s** |
| store, prime core worst | 1.05 M tok/s | **7.46 M tok/s** |
| recall/s, little core worst | 6,051 | **35,313** |

Everything moved by at most 0.3 points; T2 is 0.3 lower (95.7 → 95.4), the
rest equal or higher.

¹ First reported as 46.0 / 34.9: lean's night pruning freed events but kept
the room their cue lists had used, so at small budgets it pruned far more
than needed (and at 0.5 MB never got under budget). Fixed; the v0.1 / v0.2
runs were not affected (their pruning already gave room back).

## Shrinking further (owner: "keep shrinking")

Each step keeps the behaviour and is checked on the same tests. Where the
bytes went guided each cut (lean: cues 1.03 MB, event words 573 KB, events
295 KB, list headers 128 KB).

| version | what changed | whole book | bytes / event |
|---|---|---|---|
| v0.1 as specified | — | 13.3 MB | ≈ 360 |
| lean | no content bits, word = its number, 8 B event, 3–4 B synapses | 2.03 MB | ≈ 55 |
| lean2 | event = 1 byte (words 4 bits, log use counter 2 bits, next 2 bits) + a start every 64 events; word numbers as varints; cue lists as gaps (varints), no per-synapse strength (the event's use count decides what a full list drops); lists grow in quarter steps | 959 KB | ≈ 26 |
| lean3 | cue list headers 12 → 7 B; lists of ≤ 32 bits inside the header; gaps bit-packed as exp-Golomb codes, order chosen per word | 819 KB | ≈ 22 |
| lean4 | event words bit-packed too (exp-Golomb order 11, chosen on the dev book) | **758 KB** | **≈ 21** |

On the whole book every version scores the same: T1 96.7, T2 95.4, T3 99.98,
T4 99.35. Under pressure the smaller ones keep more (0.5 MB budget):

| 0.5 MB | events kept | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|
| lean | 7,339 | 44.2 | 34.5 | 66.7 | 19.8 |
| lean2 | 16,786 | 62.7 | 36.8 | 78.4 | 45.5 |
| lean3 | 20,457 | 69.5 | 37.8 | 82.6 | 55.3 |
| lean4 | 24,141 | 76.1 | 38.5 | 86.8 | 65.4 |

Speed, S22 Ultra, whole book, worst of 3 (`results/phone_shrink.txt`):

| | store, prime | store, little | recall/s, prime | recall/s, little |
|---|---|---|---|---|
| lean | 3.86 M tok/s | 934 k | 177 k | 36.4 k |
| lean2 | 2.63 M | 763 k | 115 k | 33.4 k |
| lean3 | 1.49 M | 678 k | 72 k | 31.2 k |
| lean4 | 1.46 M | 675 k | 62 k | 29.4 k |

The price of each cut is decoding: lean4 stores 1.4× slower than lean on the
little core, still 67× the 10 k floor.

### What is left, and its limit

lean4 stores each word-in-event twice, ≈ 14 bits each way, close to the
information it carries (one of ~17 k words; one of ~37 k events). Further
cuts must remove information, not packing:

* **Read the words back from the page.** Both specs keep the original text
  outside the brain. If an event keeps only where it is on the page (~1 B),
  its word list (≈ 320 KB) goes: an estimated ~0.45 MB for the book. The brain
  then needs the page to say an event's words (recognition and cued recall
  still work without it).
* Keep one direction only (cue lists): ~0.44 MB, but recall must search every
  list (estimated tens of times slower).
