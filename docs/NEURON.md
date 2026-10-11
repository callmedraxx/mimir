# The Mimir neuron — spec, v0.1 (for review)

One brick that everything in Mimir is built from: the reader, word meanings,
word order, the context window, the gist, and later the gate and ABC brains.
Every neuron can **store something**, **connect to any other neuron**, **stay
active for a while**, **learn locally**, and is **born, matures, commits or
dies** (neurogenesis everywhere). Status: design only; nothing below is built.

## 1. What a neuron is today (for comparison)

`Neuron` in `src/mimir.h`: a perceptron unit with a lifecycle.

```
output = activation(bias + Σ weightᵢ × inputᵢ) × maturity
```

* ~64 bytes of state + 4 bytes (or 1 bit, HDC) per input weight.
* Connects only to every neuron of the layer before it, fixed when the layer
  is made. No links inside its layer, none backwards, none to anything named.
* Stores nothing you can point at: knowledge is smeared across its weights.
* No sense of time beyond an activity average.
* Lifecycle dormant → immature → mature → committed, apoptosis by low
  activity; growth only when layer training stalls.
* In use: gate brain 23 (12 dormant, never woken), ABC brain 282 (256 frozen).
  Reader, context, meanings, gist: 0 neurons (tables and vectors).

Good pattern detector, not a memory unit. The new neuron is a memory unit
that can also detect patterns.

## 2. The new neuron in one picture

```
            synapses in (any neuron)             synapses out (any neuron)
   word "school" ──part-of──►  ┌───────────────┐ ──next──►  event #4,812
   word "boy"    ──part-of──►  │ EVENT #4,811  │ ──source─► text bytes 9,120..9,177
   event #4,810  ──next────►   │ content: 1024 │
                               │ bits (HDC)    │   activation 0.62, fading
                               │ committed     │   born at tick 51,033
                               └───────────────┘   fired 7 times
```

A neuron **stands for one thing** (a letter group, a sound, a word, an event,
an image...). What it stands for is in three places:

1. its **label** (text, if it has one: "school"),
2. its **content**: a binary pattern of D bits (hyperdimensional), so two
   neurons can be compared by how many bits differ,
3. its **synapses**: what it is connected to, how strongly, and how.

This is closer to a **cell assembly** (a small group of real neurons that
fire together for one thing; the "concept cells" found in the human
medial temporal lobe fire for single people or places) than to one biological cell.
That is deliberate: one Mimir neuron = one meaningful unit, cheap enough to
have millions.

## 3. Memory layout (fixed sizes, contiguous, fast)

All neurons live in one **pool** of parallel arrays (struct of arrays), so
hot loops touch only what they need and the phone's caches stay full.

### 3.1 Per neuron: `NeuronMeta`, 32 bytes

| field | type | bytes | meaning |
|---|---|---|---|
| `label` | u32 | 4 | offset into the label text arena (0 = no label) |
| `syn` | u32 | 4 | first synapse in the synapse arena |
| `nsyn` | u16 | 2 | synapses in use |
| `capsyn` | u16 | 2 | synapse block size (a power of two) |
| `kind` | u8 | 1 | what sort of thing: letter group, sound, word, event, image… |
| `state` | u8 | 1 | free, newborn, mature, committed (§6) |
| `maturity` | u8 | 1 | 0–255; output is scaled by it while newborn |
| `flags` | u8 | 1 | kind-specific bits |
| `born` | u32 | 4 | clock tick when born |
| `last` | u32 | 4 | clock tick when it last fired (for fading, §5) |
| `uses` | u32 | 4 | times it has fired (for maturing, committing, pruning) |
| `value` | u32 | 4 | small payload: a sound class, a text offset, a count |

Plus, in their own arrays:

* `act[N]`: activation, 16-bit fixed point (2 bytes), the hottest field.
* `content[N][D/64]`: D bits. D is set **per kind** and per device profile:
  0 for kinds that need no content (sounds), 256–1024 for words and events.
  1024 bits = 128 bytes.

### 3.2 Per synapse: 8 bytes

| field | type | bytes | meaning |
|---|---|---|---|
| `to` | u32 | 4 | the neuron it connects to |
| `w` | i16 | 2 | strength, fixed point (negative = inhibits) |
| `kind` | u8 | 1 | how: part-of, next, says (letter group → sound), associates, cue, source |
| `age` | u8 | 1 | nights since last strengthened (for pruning) |

A neuron's synapses sit together in one block of the **synapse arena**
(blocks of 4, 8, 16… synapses, with free lists per size). Growing past the
block moves the list to the next size up. Relations that are read both ways
(word ↔ event) are stored as a synapse in each direction.

### 3.3 Cost per neuron

| kind | meta | content | typical synapses (estimate) | total |
|---|---|---|---|---|
| letter group (reader) | 32 B | 0 | ~3 to sounds: 24 B | ~58 B |
| word | 32 B | 128 B | ~40 associates: 320 B | ~480 B |
| event (context) | 32 B | 128 B | its ~8 words + next + source: ~80 B | ~240 B |

## 4. Finding a neuron: recognise or give birth

Every time something comes in, Mimir asks: *do I already have a neuron for
this?*

* **Labelled things** (letter groups, words): exact lookup, label → neuron, in
  a hash table. Miss → birth.
* **Unlabelled things** (events, image patterns): compare content with the
  candidates (Hamming distance: XOR, then count bits; 1024 bits are 16
  machine words) and take the closest. If it is closer than the **novelty
  threshold τ**, that neuron is *recognised* (it fires, it is strengthened).
  Otherwise a new neuron is *born* for it. This keeps similar but different
  memories apart (pattern separation, the job the brain's hippocampus is
  thought to do with its newborn neurons).

## 5. Firing: activation that fades, and only a few winners

* A neuron is **activated** by input (it was just read) or through its
  synapses: `act[target] += w × act[source]`, through synapses of the kinds
  the current task uses.
* Activation **fades** with time: `act × 2^(−(now − last) / half_life)`,
  computed only when the neuron is next touched (lazy), so idle neurons cost
  nothing. Half-life is per kind (events fade over a few events, words
  faster).
* **Only a few win** (lateral inhibition, k-winners-take-all): within an area
  (a kind) only the k most active stay active. **The workspace of 4 is
  exactly this**: k = 4 among event neurons. Nothing else is needed to "hold
  things in mind".
* **Cost** is proportional to the neurons and synapses touched, never to the
  number of neurons that exist. A million idle neurons cost nothing per token.

## 6. Learning: local only, no backprop

Three rules, all local to one synapse or one neuron:

1. **Hebbian, three-factor**, on a synapse: when its two neurons are active
   together, `Δw = η × act_pre × act_post × M`, where M is a modulator: +1
   normally, larger on surprise, negative on a mistake (as the reader's fixes
   are today). A synapse that does not exist is born the first time its two
   neurons fire together (synaptogenesis).
2. **Order (spike-timing)**: if A fired just *before* B, the `next` synapse
   A → B is strengthened. This is word order and event order.
3. **Content drift** (newborn neurons only): a newborn neuron's content moves
   toward the bundle (majority vote) of the contents of the neurons it fires
   with. That is how a new word's meaning forms from its context, the way
   meanings form today. Committed neurons' content is frozen.

**Homeostasis** keeps it stable: each neuron's outgoing weights are
rescaled so their total stays bounded (synaptic scaling), so a word seen a
million times does not drown everything else.

## 7. Lifecycle: neurogenesis everywhere

```
 free ──born──► NEWBORN ──used enough──► MATURE ──stable over 2 nights──► COMMITTED
                  │                        │
                  └──unused, out of room───┴──► free (pruned)
```

| step | when | what happens |
|---|---|---|
| **birth** | nothing recognises the input (§4) | take a free neuron; label and/or content set; maturity 0 |
| **maturing** | each use | maturity rises; output is scaled by it, so a newborn cannot dominate |
| **mature** | `uses` ≥ a per-kind count | full strength; synapses learn normally |
| **commit** | mature and unchanged across two nights | content frozen; its synapses can still grow but not fall below a floor, so committed knowledge is not forgotten |
| **pruning** | at night, when the device's budget for that kind is short | free the least used non-committed neurons and synapses (`age` high, `uses` low). Committed neurons are never pruned. |

**Capacity is a device setting**, per kind: how many neurons, how many bits of
content, how many synapses per neuron. A phone gets more than the browser
bench; a PC more than a phone. When a kind is full, Mimir forgets its least
useful newborns to make room. It never refuses to learn and never forgets
committed things.

## 8. Night (dream): consolidation

The replay thread already dreams. With neurons, a night:

1. **Normalises association synapses** (what PPMI does for meanings today:
   strength = how much more together than chance).
2. **Updates newborn content** from those synapses (meanings of new words).
3. **Commits** neurons that were stable since the last night.
4. **Prunes** where the device is short of room.
5. **Checks before keeping** (as the reader's dream does): a fixed sample of
   committed knowledge must not get worse, or the night is undone.

## 9. Everything built from it

| today | as neurons |
|---|---|
| **Reader** table: 32,768 hashed rows shared by many letter groups (the cause of the forgetting we fought) | one **letter-group neuron** per letter group actually seen (592,123 in the training words; 159,966 seen 3+ times, the rest pruned when room is short), each with `says` synapses to the 162 **sound neurons**. Reading a letter = fire its 28 group neurons, sum onto sounds, the strongest sound wins. No sharing, so fixing one word cannot disturb another. |
| **Sight words** | **word neurons** with ordered `says` synapses to sounds |
| **Word meanings** (vectors in a file) | **word neurons**: content = meaning, `associates` synapses = the day's counts; born on first sight, so new words (Rumpelstiltskin) get a meaning while reading |
| **Word order** (next build) | `next` synapses between word neurons (rule 2) |
| **Context window** (token array) | one **event neuron** per event, born as it is read: `part-of` synapses to its word neurons in order (so the exact wording can be rebuilt), `next` to the following event, a `source` value pointing to the original text bytes (for case and punctuation) |
| **Workspace of 4** | k-winners-take-all among event neurons |
| **Cued recall** (word → last 8 events) | a word neuron's `part-of` synapses to events, strongest and most recent kept |
| **Gist** (strength per event) | each event neuron's accumulated activation (`uses` and a strength value) |
| **Gate and ABC brains** | later: their perceptron units become neurons whose synapses carry the weights (dense kinds); same lifecycle |

## 10. Speed budget on the phone (estimates, to be measured)

Measured today (S22 Ultra, War and Peace, 667,685 tokens): take in 1.15 M
tokens/s on the prime core; gist 185 k tokens/s; together 160 k; little core
together 20 k.

With neurons:

* **Comparing meanings**: 1024-bit Hamming (16 × XOR + popcount, NEON CNT)
  instead of a 256-float cosine. Estimated ~10 ns instead of ~70 ns.
* **Cued recall**: walking a word neuron's synapses is a sequential read of
  8-byte records.
* **Estimate**: gist several times faster than today (most of its time is
  the cosine), so take in + gist around 500 k tokens/s on the prime core and
  60–100 k on a little core. The target stays: ≥ 10,000 tokens/s on the
  slowest core, measured worst case.
* **Memory**: about 20 MB per million tokens of context (event neurons,
  their synapses, the original text), about what the context memory takes
  today (21 MB per million).

## 11. How we build it (each step tested before the next)

1. **One neuron, alone, on the bench.** The pool, synapse arena, birth,
   recognise-or-birth, firing with fading, k-winners, the three learning
   rules, lifecycle, night, save and load. Tests for each, and a speed test
   on the phone (neurons touched per second, synapses walked per second).
   A bench view to look inside any neuron: content, synapses, lifecycle.
2. **Context window + gist on neurons.** Must keep the gist test at least
   at 13.3% and War and Peace at least as fast.
3. **Meanings on word neurons**, new words born while reading. WordSim and
   SimLex must not drop (the binary content must be checked first: §12).
4. **Reader on letter-group neurons.** Unseen words ≥ 63.2%, and no
   forgetting when fixing (the test we used for sight words).
5. **Word order** (`next` synapses), then speaking (`say`).
6. **Gate and ABC brains** onto neurons, last.

## 12. Risks, stated plainly

* **Binary content may lose meaning.** Today's meanings are 256 int8 numbers.
  Turning them into 1024 bits (keeping only each number's sign, in a random
  projection) usually keeps most of the similarity, but this must be measured
  on WordSim and SimLex before anything moves.
* **One neuron per letter group needs room.** 160 k–600 k reader neurons is
  ~10–35 MB. Fine on the phone; the bench uses a smaller budget.
* **Capped synapse lists lose rare links.** A word like "said" connects to
  thousands of events; keeping only the strongest and most recent drops some.
  The gist test will show whether it matters.
* **More moving parts than tables.** That is why step 1 is one neuron, alone,
  fully tested, before anything is rebuilt on it.

## 13. Decisions for the owner

1. **Content size per device.** Proposed: 1024 bits on phone and PC, 256 on
   the browser bench.
2. **Context window wording.** Proposed: rebuild exact words from event
   neurons' synapses, and keep the original bytes only for case and
   punctuation (so the page can always be quoted exactly).
3. **When the device is full.** Proposed: forget the least used newborns,
   never committed ones, never refuse to learn.
4. **Gate and ABC brains.** Proposed: move them onto neurons last (step 6),
   so the reading work does not wait on them.
