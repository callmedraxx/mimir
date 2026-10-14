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
