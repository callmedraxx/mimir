# Mimir journal

One place for where we are: the goal, what is in use, what we tried and how it went, and a
dated log. Read this first; update it at the end of every step. The full write-up of each
finished experiment is in `docs/past/`. Code that lost or is no longer used is deleted from
the branch; all of it is in git history (`git checkout before-cleanup -- <path>` brings a
file back).

## Goal

A brain that runs on a phone: it reads anything its owner gives it, keeps it, understands
it, and answers from it, on the phone's own CPU, built like a brain (no LLM, no
backpropagation). Each owner decides what its memory holds; a bigger device holds more; it
stays fast everywhere (≥ 10k tokens/s on the S22 Ultra's little core). Producing sentences
comes later.

Now: **Laya-level understanding.** Laya (421M parameters) scores MMLU high school 33.2,
CommonsenseQA 44.7, ARC-Easy 50.1, ARC-Challenge 30.5, OpenBookQA 40.0.

The design the owner set (2026-10-14): **one brain.** The fact memory ("hippocampus")
stores facts as it reads; at the same time the cortex learns from those facts (Hebbian +
anti-Hebbian, reward and penalty); answers use both.

## How we work

* Scorecard first (`docs/<NAME>.md`): what is compared, the cells, the dev sets, the choice
  rule, the tests. Committed before any code or run.
* Develop on dev sets only. Freeze (commit). Run the tests **once**. Report as they come.
* A version is adopted only if it wins by the rule; **the losers and their code are then
  deleted** and listed below under "Tried".
* All training, dev and test runs in **Google Colab** through the shared browser; the
  droplet is for writing code and tiny smoke tests only. Delete each Colab runtime when done.
* The owner's private tests (maths, idioms) are never developed on.

## In use now

| part | where | what |
|---|---|---|
| the brain | `sandbox/brain/bg.c` | `bg read` (shelf → facts with pointers), `bg arc` (answer a test file), `bg serve` (one question at a time), `bg words` / `bg near` (the cortex) |
| **the cortex** | `sandbox/brain/bg.c` | learns from each fact in its own thread while reading (K3b's reward/penalty rule; each word's rate falls as it is met more often); saved as `BRAIN.cx`; compares words when answering (a link from cosine 0.40) and scores how well an option fits the question (weight 1) |
| **the episode** | `sandbox/brain/bg.c`, `bg passages` | the text being read kept apart from the shelf's facts, active while asked about; its facts answer first, weighted 4 (EPISODE); inside it the association parts are off and the passage decides (PASSAGE); after each passage everything is put back |
| passage tests | `data/understand/mctest` (`tools/understand_fetch.py`), `data/understand/race` (`tools/race_fetch.py`) | MCTest MC160/MC500, RACE-Middle |
| reading sentences into facts | `sandbox/brain/facts.c` | word classes, chunks, frames (verb, who, what, is-what, place/time, negation) |
| tokens, word meanings, word reader | `src/text.c`, `src/meaning.c`, `src/reader.c` | shared with the app |
| **the school brain with its cortex** | GitHub release `brain-school-cx` (built by `.github/workflows/brain.yml`; Colab gets it with `tools/brain_release.py`) | 32,488,986 facts, 725k words, cortex 1.7 GB; rebuilt when `tools/brain_build.txt` changes |
| the school brain (before the cortex) | GitHub release `brain-school` | 32.5M facts from 2.2 GB, baby to high school (docs/past/SCHOOL_BRAIN.md) |
| its shelf | `tools/school_shelf.py` | rebuilds the 2.2 GB of books in Colab (35 min); the brain's pointers lead into it |
| word classes | `data/understand/wordclass.tsv` (`tools/understand_fetch.py`; release `brain-E`) | |
| glue words | `data/meaning/meaning.mng` | only its list of the most common words (the, of, …), which the reader skips; its word meanings are no longer used |
| dev sets | `data/dev_sets/` (`tools/general_tests.py`) | ARC-Easy, ARC-Challenge, OpenBookQA dev; MMLU-HS validation; CommonsenseQA 1,000 from train |
| tests | `data/arc_tests/` | MMLU-HS test, CommonsenseQA dev, ARC-Easy, ARC-Challenge, OpenBookQA test |
| stories | `data/story/` | clock (dev, 10 questions); harbor, garden, snow (30, tested in CORTEX_LINK and EPISODE); salt, train, kite (30, tested in EPISODE) |
| word tests | `data/cortex/wordsim/` | WordSim-353, SimVerb-3500 (dev); SimLex-999, MEN-3000 (test) |
| results | `results/` | every run's output |

Answering settings (now the defaults): fact 1, joint 0.5, open book 2, meaning 4, fit 1; 20 facts
re-read; 200 candidate facts per option; "not / least / except" questions turn the score
around; strength fades with time; no room limit.

**Also in the repo, not part of the brain:** the app (`src/`, `Makefile`, `scripts/`),
which the Workbench Lab bench builds (`tools/wasm_instrument.py`, `tools/wiring.py`), and
its alphabet images (`tools/fetch_*.py`, `tools/vision_diag.c`). Kept because the live bench
uses them; say if they should go too.

## Scores now (school brain + cortex, 2026-10-14)

| | MMLU-HS | CommonsenseQA | ARC-Easy | ARC-Challenge | OpenBookQA |
|---|---|---|---|---|---|
| **Mimir, school brain + cortex** | 27.4 | **31.4** | **46.6** | 21.2 | **24.4** |
| Mimir, school brain alone | 27.9 | 25.5 | 45.5 | 21.3 | 20.7 |
| GPT-2 XL (1.5B) | 29.9 | 37.1 | 58.3 | 25.0 | 18.2 |
| Laya (421M) | 33.2 | 44.7 | 50.1 | 30.5 | 40.0 |
| chance | 25 | 20 | 25 | 25 | 25 |

Reading a passage and answering about it (`bg passages`, with the episode):

| | MC160 | MC500 | RACE-Middle | new stories (30) | harbor/garden/snow (30) |
|---|---|---|---|---|---|
| **Mimir, school brain + cortex + episode, passage decides** | 49.6 | 50.6 | 40.7 | 23/30 (salt/train/kite), 21.5/30 (bicycle/stone owl/swim) | 29/30 |
| sliding window (MCTest paper) | 67.1 | 57.8 | 45.3 | | |
| Mimir's passage-only reader (10-12) | 63.8 | 52.8 | 41.1 | | |
Word tests (cortex learned from facts): SimLex-999 0.381, MEN 0.738 (word2vec 0.398 / 0.717).

## Tried

| when | what | result | now |
|---|---|---|---|
| 10-11 | Neurons: one thing per neuron (v0.1) vs knowledge only in synapses (v0.2) | v0.1 won 18–0; cut down to lean4 (War and Peace in 758 KB) | removed (code in history) |
| 10-11 | Context window on neurons vs today's | neurons 3, today's 2, ties 3 | removed |
| 10-12 | Roles (who did what to whom) | bAbI 94% vs 20%; ordinary prose not understood | the reader lives on in `facts.c` |
| 10-12 | Decisions like Jev/Laya (Typed Decisions) | learned 0.726 (Jev 0.727, Laya 0.766); unseen 0.361 | removed |
| 10-12 | Facts with pointers, closed/open book | open book wins SciQ (93.1 vs 91.2), loses RACE/MCTest; 4.7× smaller | **in use**: the brain's design |
| 10-12 | Reading a lot first (background) | ARC-Easy 28.9 → 42.9; ARC-Challenge flat | **in use** |
| 10-12 | Read more + strength (used facts stick, unused fade) | reading more alone hurt (ARC-Easy −6); strength took back OpenBookQA | strength **in use**; practice questions removed |
| 10-13 | Topic reading, separate rooms | on-topic better than all; rooms add nothing | removed |
| 10-13 | Keep what it knew "for good" (brain E) | first brain to beat its start without losing anything | removed (school brain has no room limit, so nothing is pushed out) |
| 10-13 | Chains of two facts through a shared word | ties on all three tests | removed |
| 10-13 | Laya level stage 1–2: re-read 20 facts, read "not" questions | ARC-Easy +1.8 | **in use** |
| 10-13 | Stage 3: answer weights learned from 10k training questions | ARC-Challenge 29.8, OpenBookQA 30.7 but ARC-Easy −5: learned test-taking (longer option), not understanding | removed |
| 10-13 | Versus GPT-2 and Laya | see scores above | scores kept; scripts removed |
| 10-13 | School brain, no room limit | 32.5M facts in 11 min; MMLU-HS 27.9 | **in use: the base** |
| 10-14 | Candidate ranking (more candidates, full join) | no cell beat the current one; more candidates hurt | removed |
| 10-14 | Cortex v0: random / Hebbian / predictive | Hebbian learns association, not kinds (SimLex 0.19); predictive collapsed | removed |
| 10-14 | Cortex v0.1: Hebbian + anti-Hebbian (reward/penalty), from text | **beats word2vec**: SimLex 0.464 (w2v 0.398), MEN 0.739 (0.717), fill-the-blank 10.1% (6.6%); owl → eagle, raven, bird | its rule (K3b) goes into the brain (CORTEX_LINK); the text-trained version removed |
| 10-14 | **CORTEX_LINK: one brain**, the cortex learning live from the facts as Mimir reads, used when answering | tests: CommonsenseQA +5.9, OpenBookQA +3.7, ARC-Easy +1.1, MMLU/ARC-C ties (mean 28.2 → 30.2); **stories worse** (19 → 14 of 30) | **in use**; rate falling with the reading clock, cosine 0.25 and the old meaning table removed |
| 10-14 | **EPISODE**: the text being read kept apart from what it knows, active while asked about | passages: MC160 25 → 50, MC500 28 → 49, RACE-M 26 → 39; stories 7/30 → 23/30 and 14 → 27/30; general tests unchanged. Inhibition and recency added nothing | **in use** (EPW 4); one pool, inhibition, recency removed |
| 10-15 | **PASSAGE**: answering about a passage by understanding: association off inside it, the asked slot, yes/no polarity | association off: MC500 +1.4, RACE-M +1.6, stories 19 → 21.5 and 27 → 29 of 30; the slot +0.4 and polarity +0.1 on dev (under the bar) | association off **in use**; slot and polarity removed |

Lessons that keep coming back:
* Reading more helps ARC-Easy only; ARC-Challenge and OpenBookQA were built against
  word association, so association-based answering is drawn to the wrong options there.
* What looks like a gain can be test-taking (stage 3). Count only what the brain knows.
* More candidate facts add noise, not answers: the problem is telling a fact that fits by
  chance from one that answers the question.
* General knowledge drowns out what was just read (passage questions with background, 10-12;
  stories with the cortex, 10-14). Fixed by giving the text being read its own place (EPISODE).
* Sending big files (100 MB and up) from Colab through the shared browser fails. The brain is
  built by GitHub Actions instead and published as a release (no key needed: GitHub's own job
  token); a Colab session downloads it in minutes.

## Open

* **Next: reading negation and "what a thing is"** (docs/PASSAGE.md misses): "was not sad", "does not like", "without a ticket", "is not a real bird", "the train is called X", "said no" are lost when sentences are read into facts. Before: answering about a passage by understanding it, not by association: the meaning and
  fit parts never look at the passage (passage tests still below Mimir's own passage-only reader
  and the sliding window); negation ("She was not angry"); "X gave Y with Z".
* Then: sentences, not just words: pronouns ("He" = Dumi), "an owl named Pell", negation,
  relations (cause, kind of, part of).
* Size (owner 2026-10-15: "let's first make it smart then we begin to prune later"): today ~2.8 GB on disk + 2.2 GB shelf, ~5 GB in memory (facts 80 B each = 2.6 GB, cortex 1.7 GB float32, cue lists) against Laya 846 MB, GPT-2 XL 3.1 GB. Pruning comes after the smartness steps: cortex to the ~50k words met often, 8-bit (~30-60 MB); facts packed (~25-30 B); weak facts let go; shelf compressed.
* Later: the cortex small enough for the phone (8-bit, ~30 MB for 50k words); a cap on what
  is kept for good; the GPT-2 vs Mimir chat page (half built).

## Log

**2026-10-15**
* PASSAGE tested: association off inside a passage adopted (MC500 +1.4, RACE-M +1.6, stories better); the asked slot and polarity not kept. Remaining misses are negation and naming, lost in reading.

**2026-10-14**
* The school brain with its cortex published as release `brain-school-cx` by GitHub Actions (owner: "you have gh access create the key you need"; GitHub does not let tools create keys, and the existing all-repos key was not put in Colab; the workflow uses GitHub's own job token). Same 32,488,986 facts as the Colab builds.
* PASSAGE scorecard and code; dev running.
* EPISODE tested: adopted (MC160 25 → 50, MC500 28 → 49, RACE-M 26 → 39; new stories 23/30, harbor/garden/snow 27/30; general tests unchanged). Losers deleted.
* CORTEX_LINK tested: adopted (general tests mean 28.2 → 30.2; CommonsenseQA +5.9, OpenBookQA +3.7), stories worse (19 → 14 of 30). Losing settings deleted; the cortex is always on. EPISODE scorecard written (3 new stories, 30 questions).
* Deleted from the droplet (owner: "yes delete it"): `data/shelf/` (22 GB, brain E's reading), `data/school_brain/` (1.5 GB copy of the release), `data/pack/` (0.75 GB); brain E and the school brain stay on their GitHub releases. Disk 90% → 70% full.
* Cleanup (owner: "any code we are not using please delete them … let's have a journal").
  Tag `before-cleanup` holds everything before. The brain is now two files,
  `sandbox/brain/bg.c` + `facts.c` (964 lines, was 3,803 across six); gone: practice,
  keep-for-good, rooms, chains, learned weights, the passage mode, pack, the ranking
  switches, and the code of every finished experiment. Checked on a small brain read on the
  droplet (Alice + War and Peace): the brain file is byte-identical and every answer on
  320 dev questions and the clock story is identical. Finished write-ups moved to
  `docs/past/`; results to `results/`.
* CORTEX_LINK scorecard written (one brain: cortex learns from facts, live). 30 story test
  questions written for harbor, garden, snow before any code.
* Cortex v0.1 (Hebbian + anti-Hebbian) beat word2vec on all three word tests.
* Cortex v0 did not pass its bar.
* Story test: 1/10 → 8/10 after reading.
* Candidate ranking: no winner.

**2026-10-13**: school brain became the base; versus GPT-2 and Laya; chains tied; Laya-level
stages 1–3; keep-for-good (brain E).

**2026-10-11 – 12**: reader on the phone; neurons; context window; roles; decisions;
facts with pointers; background reading; strength.
