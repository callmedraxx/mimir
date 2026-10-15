# Answering about a passage by understanding it (passage)

Owner (2026-10-14): "yes proceed" (after EPISODE: write the scorecard for answering passage
questions by understanding, not association).

## Why

With the episode (docs/EPISODE.md) the passage now wins over the shelf: MC160 50.0, MC500
49.2, RACE-Middle 39.1, new stories 23/30. That is still below the MCTest paper's sliding
window (67.1 / 57.8 / 45.3) and Mimir's own passage-only reader of 10-12 (63.8 / 52.8 /
41.1). What is left wrong:

* **Association still votes.** Two parts never look at the passage: "meaning" (how close the
  option's words are to the question's, weight 4) and "fit" (how well the option fits the
  question as the cortex expects, weight 1). On a passage question they say what is usual,
  not what the passage says ("What is Firebird?" → a bird).
* **The asked slot is not used.** The reader turns a question into a frame with the asked
  slot left open ("Who fixed the chain?" → fix(who = ?, what = chain)). Answering does not
  look for the passage fact that fills that slot; it scores how much each option overlaps.
* **Negation is not read in yes/no questions.** "Was Mrs. Vanek angry?" → "she cried",
  though the story says "She was not angry."

## What is built (all inside an active episode; general questions unchanged)

| cell | what |
|---|---|
| **U0** (now) | EPISODE as adopted |
| U1 | the passage decides: inside an active episode, the meaning and fit parts are scaled by `EA` (chosen from 0, 0.25, 0.5; now 1) |
| U2 | U1 + **the asked slot**: the question is read as a frame with its open slot; the episode facts that match the rest of the question best (verb and the other slots, by role) give the filler of the open slot; each option is scored by how well its words match that filler (exact first, near by the cortex), a new part `slot` weighted `SW` (chosen from 1, 2, 4) |
| U3 | U2 + **polarity**: for a yes/no question the episode fact that matches the question decides yes or no by its negation; options that say "no" / "not" / "never" are scored by it (and the rest by the opposite) |

Each cell adds to the one before; a step is kept only if it is more than 1 point better on
dev than the step kept before it. With no episode active, nothing changes (the general dev
sets must give the same numbers).

## Rules

* Dev only: MCTest MC160 train + dev, MC500 train + dev, RACE-Middle dev (as EPISODE); the
  clock story (shown, not counted). Choice: the best mean of the three.
* Then frozen and run **once** on: MC160 test, MC500 test, RACE-Middle test; **three new
  stories, 30 questions** written before any code (`data/story/passage_tests.tsv`: bicycle,
  stone owl, swim; with negation, a story fact against general knowledge, giving); the six
  stories of CORTEX_LINK and EPISODE (salt, train, kite, harbor, garden, snow; tested before,
  never developed on); the five general tests (must not move).
* Yardsticks: EPISODE's test numbers above; the sliding window; Mimir's passage-only reader.
* Runs in Colab. The brain: from a GitHub release if one was saved (owner's token in Colab
  Secrets, `GH_TOKEN`); else rebuilt in the session (about 1.5 hours).
* Reported as they come; journal updated; losers' code deleted.

## How it was built (before any dev result)

* `bg passages` with `PSLOT=2|3`, `EA`, `SW`; the dump has each option's parts and whether the
  episode was active, so `EA` and `SW` are chosen offline from one run per slot version
  (`tools/passage_colab.py`; the offline score reproduces `bg`'s). U0 and U1 come from the
  `PSLOT=2` dump with `SW = 0`.
* The asked slot: the question is read with `read_question` (the reader's question mode,
  which leaves the asked argument open). An episode fact's match: verb and the question's
  other arguments, by role (as the fact part); its fillers: every argument in the asked role
  that the question did not name. A passive fact ("was built by a clockmaker") is matched as
  active. Developed on the clock story (dev): "What does Pell eat?" became right; "Who built
  the clock?" still wrong ("a clockmaker named Imra Batsho": the name is not linked by the
  reader; "Pell eats beetles, not mice": the "not" is lost in reading). Those are reader
  limits, not changed here.
* With the defaults (`EA=1`, `SW=0`, `PSLOT=0`) the answers are EPISODE's, and with no
  episode nothing changes (droplet checks).
* The brain is kept as a GitHub release `brain-school-cx` (`tools/brain_release.py`) when the
  owner's `GH_TOKEN` is in Colab Secrets; else rebuilt in the session.

## Dev results (2026-10-15, Colab; `results/passage_dev.txt`)

The brain rebuilt in the session: 32,488,986 facts, the same as EPISODE's (U0 gives EPISODE's
dev numbers exactly). Means of MC160 (train + dev), MC500 (train + dev), RACE-Middle dev:

| cell | settings | mean |
|---|---|---|
| **U0** (now) | | 46.67 |
| **U1** | EA 0 (association parts off inside the episode) | **49.26** (MC160 57.35, MC500 51.30, RACE-M 39.12) |
| | EA 0.25 / 0.5 | 48.32 / 47.41 |
| U2 | EA 0 + slot, SW 1 / **2** / 4 | 49.31 / **49.62** / 49.20 |
| U3 | EA 0 + slot and polarity, SW 1 / **2** / 4 | 49.33 / **49.69** / 49.27 |

**Chosen: U1, EA 0 (+2.6).** The slot (+0.4) and polarity (+0.1 more) help a little but
less than the 1 point a step needs, so they are not kept. Frozen in `tools/passage_choice.env`.
Clock story: 8/10 → 8.5/10 (a tie). General dev sets with no episode: unchanged.

## Test results (2026-10-15, Colab, once, frozen 9834049; `results/passage_test.txt`)

| test | **U0** (EPISODE) | **U1** (EA 0) | change |
|---|---|---|---|
| MCTest MC160 test (240) | 50.00 | 49.58 | −0.4 tie |
| MCTest MC500 test (600) | 49.19 | **50.56** | +1.4 better |
| RACE-Middle test (1,436) | 39.12 | **40.68** | +1.6 better |
| new stories: bicycle, stone owl, swim (30) | 63.33 (19/30) | **71.67** (21.5/30) | +8.3 better |
| salt, train, kite (30) | 76.67 | 76.67 | tie |
| harbor, garden, snow (30) | 90.00 | **96.67** | +6.7 better |
| general tests, no episode | | 27.41, 31.37, 46.61, 21.22, 24.40 | unchanged |

**Count: better 4, tie 2, worse 0. Adopted (EA 0).** A small step: MCTest and RACE are still
below the sliding window (67.1 / 57.8 / 45.3) and Mimir's passage-only reader (63.8 / 52.8 / 41.1).

What the misses show (the new stories, after U1): most are **negation** and **what a thing
is**, both lost in reading, not in answering:
"Was Nadia sad?" → angry ("Nadia was not sad"); "Which is true about Lenno?" → he keeps
bees ("does not like bees"); "What did Kofi not have?" → a coat ("without a ticket");
"What is Old Watch?" → a real bird ("It is not a real bird", "an owl made of stone");
"What is the Grey Heron / the Moth?" → a bird / an insect ("the train is called the Grey
Heron", "His boat is called the Moth"); "What did the mayor say?" → maybe ("said no").

Losers deleted: the asked slot (U2) and yes/no polarity (U3).
