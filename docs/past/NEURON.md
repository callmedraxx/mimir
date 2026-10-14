# The Mimir neuron — spec, v0.2 (for review)

One kind of neuron that everything in Mimir is built from. **As in the
brain, a neuron holds no information. Information lives in the synapses:
which neurons are connected, and how strongly.** A word, a sound pattern, an
event or a memory is an **assembly**: a group of neurons that fire together
because of the synapses between them. Status: design only; nothing below is
built.

> v0.1 put a label and 1,024 content bits inside each neuron. The owner
> asked for the brain's way instead (2026-10-11): neurons only have state;
> knowledge is in the synapses. The 1,024 content bits of v0.1 become 1,024
> neurons, and "which bits are on" becomes "which neurons are in the
> assembly".

## 1. What a neuron is today (for comparison)

`Neuron` in `src/mimir.h`: a perceptron unit with a lifecycle,
`output = activation(bias + Σ weightᵢ × inputᵢ) × maturity`, connected
all-to-all to the layer before, fixed when the layer is made. Gate brain 23,
ABC brain 282; reader, context, meanings and gist use tables, not neurons.

## 2. The rule

* A **neuron** has only **state**: how active it is now, how excitable it is,
  how old, how mature, its place in its lifecycle. No label, no content, no
  payload. Two neurons of the same area are interchangeable until their
  synapses make them different.
* A **synapse** joins one neuron's axon to another neuron's dendrite. It has a
  **strength** and a **type** (excitatory, inhibitory, modulatory). The
  sending neuron owns it (it is part of its axon); the receiving neuron sees it
  as an input on its dendrites.
* **Information = the set of synapses and their strengths.** Learning changes
  synapses (and grows or removes them, and recruits new neurons). Nothing
  else is learned.

## 3. Areas: structure, not information

Neurons are grouped into **areas**, like brain regions. An area says where a
neuron sits and what it can connect to, as anatomy does; it holds no
knowledge.

| area | like | what its neurons do |
|---|---|---|
| **letter receptors** | retina / early visual cortex | one neuron per (letter, position in a 9-letter window): fires when that letter is there. Meaning fixed by wiring to the outside, as a retina cell's is by its place |
| **text receptors** | — | punctuation, digits, word edges, the same way |
| **letter-group area** | visual word form area | neurons that come to fire for letter combinations ("sch" at the start), through learned synapses from receptors |
| **word area** | temporal cortex (meaning) | words as assemblies; meaning = which other assemblies they connect to |
| **sound area (motor)** | speech motor cortex | one neuron per sound (162). Firing = saying that sound |
| **event area** | hippocampus | each event read is a new sparse assembly bound to the word assemblies in it, and to the event before it |
| **workspace** | prefrontal cortex | holds whatever few assemblies are most active (k-winners, k ≈ 4 events) |

Receptor and motor neurons are the only ones whose meaning is fixed in
advance, and only by where they are wired, never by anything stored in them.

## 4. Records (fixed sizes, contiguous, fast)

### 4.1 Neuron: 24 bytes, state only

| field | type | bytes | meaning |
|---|---|---|---|
| `act` | i16 | 2 | activation now (fixed point) |
| `theta` | i16 | 2 | excitability threshold; rises when it fires a lot, falls when idle (homeostasis, as BCM today) |
| `last` | u32 | 4 | clock tick it last fired (for fading) |
| `born` | u32 | 4 | clock tick it was born |
| `uses` | u32 | 4 | times it has fired |
| `syn` | u32 | 4 | first synapse of its axon in the synapse arena |
| `nsyn` | u16 | 2 | synapses in use (block size is the next power of two) |
| `area` | u8 | 1 | which area it sits in (anatomy) |
| `life` | u8 | 1 | free, newborn, mature, committed; maturity in the top bits |

### 4.2 Synapse: 8 bytes, where the information is

| field | type | bytes | meaning |
|---|---|---|---|
| `to` | u32 | 4 | the neuron it reaches |
| `w` | i16 | 2 | strength (fixed point) |
| `type` | u8 | 1 | excitatory, inhibitory, modulatory; fast or slow (for order) |
| `trace` | u8 | 1 | how recently it was used (eligibility for learning, age for pruning) |

A neuron's axon is one block of synapses in the arena (blocks of 4, 8, 16…
with free lists). Each neuron's outgoing synapses are capped per area (a
device setting), keeping the strongest.

### 4.3 What things cost

| thing | as neurons and synapses | memory |
|---|---|---|
| a word | an assembly of ~32 neurons in the word area, ~100 synapses each (into letter groups, to other words, to sounds) | ~25 KB for its synapses, shared with other words |
| an event | a new sparse assembly of ~16 neurons in the event area, bound to the words in it and to the event before | ~16 neurons × ~40 synapses ≈ 5 KB |
| a million tokens of context (≈60,000 events) | event assemblies and their synapses | estimate 100–300 MB: **more than v0.1's ~20 MB**, the price of storing memories the brain's way. Event assemblies are reused when events repeat, and pruned when the device is full (§7). To be measured. |

The original text stays **outside the brain**, as the page on the desk
(today's context bytes), so a sentence can always be quoted exactly. Brains
do the same: they look at the page again.

## 5. Firing

* **Input**: receptor neurons are set by what is read (letter "s" at window
  position −1 fires).
* **Spreading**: a firing neuron adds `w × act` to every neuron its axon
  reaches. Inhibitory synapses subtract.
* **Fading**: `act × 2^(−(now − last) / half_life)`, computed lazily when a
  neuron is touched; idle neurons cost nothing. Half-life per area (events
  last longer than letter groups).
* **Threshold and winners**: within an area, only the k most active neurons
  above their `theta` fire (lateral inhibition, k-winners-take-all). This is
  what keeps assemblies sparse and the workspace at ~4.
* **Cost** is proportional to synapses walked, never to how many neurons
  exist.

## 6. Recognise, complete, or recruit

When input arrives, activation spreads up the areas.

* **Recognised**: an assembly fires strongly (enough of its neurons above
  threshold). That is Mimir "knowing" the thing; there is nothing to look up.
* **Completed**: a partial cue (one word of an event, a misspelt word) lights
  part of an assembly, and the synapses inside the assembly light the rest
  (pattern completion, as hippocampal area CA3 is thought to do). This is
  cued recall.
* **New**: nothing fires strongly. **Recruit**: take k newborn neurons in that
  area (neurogenesis), and grow synapses from the currently active inputs to
  them and among them (one-shot Hebbian binding). They are now the assembly
  for this new thing. Fresh neurons for new memories keep them from
  overwriting old ones (pattern separation, the job adult neurogenesis is
  thought to do in the hippocampus).

## 7. Learning and lifecycle

**Learning (all local, no backprop):**

1. **Hebbian, three-factor**: `Δw = η × pre × post × M`. M is a modulator:
   +1 normally, larger on surprise, negative after a mistake (the reader's
   fixes today). Synapses that do not exist are grown when their two neurons
   fire together.
2. **Order**: a synapse from an assembly that fired just *before* another
   strengthens (spike-timing). Word order and event order are these
   synapses.
3. **Homeostasis**: each neuron's outgoing strengths are rescaled to a
   bounded total, and `theta` follows its firing rate, so frequent words
   do not drown everything else.

**Lifecycle (neurogenesis everywhere):**

```
 free ──recruited──► NEWBORN ──fires enough──► MATURE ──stable 2 nights──► COMMITTED
                       │                         │
                       └──unused, room short─────┴──► free (pruned, synapses removed)
```

* Newborn neurons fire weakly (maturity scales their output) and learn fast.
* Committed neurons' synapses do not weaken below a floor: committed
  knowledge is not forgotten. They are never pruned.
* Synapses are pruned at night if weak and unused (`trace`), newest first.
* **Capacity is a device setting** per area: neurons, synapses per neuron.
  Full area → prune least-used newborns. Never refuse to learn.

## 8. Night (dream)

1. Replay the day's events (re-activate their assemblies) so the word area
   learns from them slowly (the hippocampus teaching the cortex).
2. Normalise association synapses (what PPMI does for meanings today).
3. Commit stable assemblies; prune where room is short.
4. Check before keeping: a fixed sample of committed knowledge must not get
   worse, or the night is undone (as the reader's dream does today).

## 9. Everything built from it

| today | brain-exact |
|---|---|
| **Reader** table (hashed rows, shared, so fixes caused forgetting) | letter receptors → letter-group neurons (recruited for groups actually seen) → sound motor neurons. Reading a letter = its window fires receptors; the group neurons that fire drive the sounds; the strongest sound wins |
| **Sight words** | a word assembly with strong order synapses to its sounds |
| **Word meanings** (int8 vectors) | word assemblies; similar words share neurons. Learned by co-activation while reading and replay at night |
| **Word order** | order synapses between word assemblies |
| **Context window** | event assemblies, each bound to its words and to the event before; the page stays outside |
| **Workspace of 4** | k-winners among event neurons |
| **Cued recall** | pattern completion from a word cue |
| **Gist** | event assemblies that stay active the longest, read out as which words their synapses reach |
| **Gate and ABC brains** | their units become neurons and their weights synapses, last |

## 10. Reading a neuron (for people only)

A neuron has no name, so the bench works one out the way neuroscientists
do:

* **receptive field**: what makes it fire (follow synapses backward to the
  receptors: "fires for 'sch' at the start of a word");
* **projective field**: what it makes fire (follow its axon forward: "drives
  S, K");
* **assembly**: the other neurons it fires with.

These names are computed for display. Mimir itself never uses them.

## 11. Speed budget on the phone (estimates, to be measured)

Today (S22 Ultra, War and Peace, 667,685 tokens): take in + gist 160 k
tokens/s on the prime core, 20 k on a little core.

Brain-exact, per word: ~32 assembly neurons × ~100 synapses ≈ 3,200
synapse updates, plus letter-group firing for each letter, plus the event
area. At ~1 ns a synapse read on the prime core: an estimated ~200–300 k
words/s, ~30–50 k on a little core. Target unchanged: ≥ 10,000 tokens/s on
the slowest core, worst case measured. Recognition by firing instead of a
hash lookup is the biggest cost; it is the first thing to measure.

## 12. How we build it (each step tested before the next)

1. **The neuron and synapse arena, alone, on the bench.** Firing, fading,
   k-winners, recruiting, the three learning rules, lifecycle, night,
   save/load. A view into any neuron: its receptive and projective fields.
   Speed on the phone.
2. **A small proof: assemblies.** Store 1,000 random patterns as
   assemblies, recall each from half of it (pattern completion), keep
   recalling the old ones after storing 10,000 more (no forgetting thanks to
   new neurons). Capacity and speed measured.
3. **Reader on neurons.** Unseen words ≥ 63.2%, no forgetting when fixing.
4. **Meanings on word assemblies.** WordSim / SimLex not below today's.
5. **Context window and gist on event assemblies.** Gist test ≥ 13.3%,
   War and Peace speed measured.
6. **Word order, then speaking.**
7. **Gate and ABC brains**, last.

## 13. Risks, stated plainly

* **Speed of recognition.** Firing through synapses costs more than a hash
  lookup. Sparse assemblies and capped synapses keep it bounded; step 1
  measures it.
* **Interference.** Shared synapses let new learning disturb old memories:
  the forgetting we already met in the reader. Answers: sparse assemblies,
  recruiting new neurons for new things, consolidation at night, committed
  floors. Step 2 tests exactly this.
* **Memory for the context.** Storing events as assemblies costs more than a
  token array (§4.3). The page stays outside; pruning keeps it within the
  device.
* **Harder to inspect.** Receptive and projective fields on the bench (§10).

## 14. Decisions for the owner

1. **Assembly size.** Proposed ~32 neurons a word, ~16 an event (sparse:
   fast and low interference). Bigger is more robust, slower.
2. **The page.** Proposed: keep the original text outside the brain (as a
   page you look at again), everything understood inside. Brains do this.
3. **When the device is full.** Proposed: prune the least used newborns,
   never committed ones, never refuse to learn.
4. **Gate and ABC brains.** Proposed: last.
