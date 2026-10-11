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
