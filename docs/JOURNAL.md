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
comes after understanding: the owner (2026-10-17): "the goal is to be brain like no backdrop … a method no matter
how unconventional that will help mimir to be able to answer questions with words so that it can reply to
conversations; after we nail the understanding tests benchmarks let's nail the answering test benchmarks". The
method is settled in docs/SPEAK.md (meaning first, words after: the question is a frame with a hole, the answer is
the filler said as the text said it, "I don't know" when nothing fills it, a conversation is an episode, a habit of
word order last); its tests are SQuAD, SQuAD 2.0, CoQA, QuAC, DROP, NarrativeQA and bAbI in words.

The owner (2026-10-17): "it should reflect real judgement; don't stop until Mimir passes those tests at Laya level or
GPT-2 or 3 level and can start answering also." A gain that is a prior or an artefact does not count.

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
| reading sentences into facts | `sandbox/brain/facts.c` | word classes, chunks, frames (verb, who, what, is-what, place/time, negation); names ("named", "called", apposition), "made of", "without", "X, not Y", passives as actives (COMPREHEND); he / she / him / her to the latest person of the right sex (sex from titles, words like wife or brother, and the pronouns used; the main character over a passer-by; plurals and things called "it" excluded), a name or "a Y" after a comma at the end of a sentence or before the verb describes the phrase before it, lists ("flour, water and salt") and "made of" in passives (READ2); each fact marked past or now (aux, verb form, time words: ago, now, still, no longer, "last year"), "but" starts a new clause (no list across it, a new subject), "he" to a known man over a man-maybe, "it" to the subject of "be" (not what it is) or the same slot of the earlier saying, "them" to the latest plural thing, "said that X was" takes X as the subject, a be-attribute is not a person (FOLLOW); `bg facts` shows what it makes of a sentence ([past] / [now]), `PRDBG=1` traces the pronouns |
| **the state now** | `sandbox/brain/bg.c` (`ep_states`), `facts.c` | inside an episode the facts filling one state (someone's place, a thing's holder, what a thing is made of) are ordered as said; the latest is current, the earlier ones superseded (×0.25 for a question about the present; for an "is" statement the superseded fact says the opposite now); a motion with "to" is also a place, take / give / drop / put also having or no longer having (derived facts answer state statements only); a thing is where its holder is; "Where is X?" + a bare option reads as "X is in <option>"; a past fact never supersedes a present one (STATE) |
| **"not" counts against** | `sandbox/brain/bg.c` | a fact or re-read sentence saying the opposite of an option counts against it (weight 2); the passage's "not" decides over the shelf; yes/no options read by their yes or no (COMPREHEND); for a yes/no question every option's statement is judged whole (READ2); a fact of another time counts half (TENSEW 0.5), of two equally close episode facts the later one decides, "Who did not …" matched as said when the episode has that NOT fact, a yes/no question is about its own words, and one option said almost whole by the passage (fact ≥ 0.85 of the episode weight) silences open book and joint (×0.25) for that question (FOLLOW) |
| tokens, word meanings, word reader | `src/text.c`, `src/meaning.c`, `src/reader.c` | shared with the app |
| **the school brain with its cortex** | GitHub release `brain-school-cx4` (built by `.github/workflows/brain.yml`; Colab gets it with `tools/brain_release.py`) | the shelf read by the COMPREHEND + READ2 + FOLLOW reader (STATE acts on episodes only, so no rebuild): 32,471,614 facts, cortex 1.8 GB; rebuilt when `tools/brain_build.txt` changes (its last `env:` line is the read's environment) |
| the school brain (before the cortex) | GitHub release `brain-school` | 32.5M facts from 2.2 GB, baby to high school (docs/past/SCHOOL_BRAIN.md) |
| its shelf | `tools/school_shelf.py` | rebuilds the 2.2 GB of books in Colab (35 min); the brain's pointers lead into it |
| word classes | `data/understand/wordclass.tsv` (`tools/understand_fetch.py`; release `brain-E`) | |
| glue words | `data/meaning/meaning.mng` | only its list of the most common words (the, of, …), which the reader skips; its word meanings are no longer used |
| dev sets | `data/dev_sets/` (`tools/general_tests.py`) | ARC-Easy, ARC-Challenge, OpenBookQA dev; MMLU-HS validation; CommonsenseQA 1,000 from train |
| tests | `data/arc_tests/` | MMLU-HS test, CommonsenseQA dev, ARC-Easy, ARC-Challenge, OpenBookQA test |
| stories | `data/story/` (`tools/stories.py` writes them as passages) | dev: clock, bicycle, stone owl, swim (40), red boat + market stall (20, `smart_tests`); tested: harbor, garden, snow; salt, train, kite; lighthouse, paper dragon, night bus (COMPREHEND); orchard, lanterns, ferry (READ2); well, bridge, parrot (FOLLOW); umbrella, goat gate, library key (`state_tests`, STATE) |
| word tests | `data/cortex/wordsim/` | WordSim-353, SimVerb-3500 (dev); SimLex-999, MEN-3000 (test) |
| results | `results/` | every run's output |

Answering settings (now the defaults): fact 1, joint 0.5, open book 2, meaning 4, fit 1; 20 facts
re-read; 200 candidate facts per option; "not / least / except" questions turn the score
around; strength fades with time; no room limit.

**Also in the repo, not part of the brain:** the app (`src/`, `Makefile`, `scripts/`),
which the Workbench Lab bench builds (`tools/wasm_instrument.py`, `tools/wiring.py`), and
its alphabet images (`tools/fetch_*.py`, `tools/vision_diag.c`). Kept because the live bench
uses them; say if they should go too.

## Scores now (school brain + cortex, 2026-10-16)

| | MMLU-HS | CommonsenseQA | ARC-Easy | ARC-Challenge | OpenBookQA |
|---|---|---|---|---|---|
| **Mimir, COMPREHEND + READ2 + FOLLOW + STATE reader** (`brain-school-cx4` with its shelf; STATE leaves these untouched) | 27.3 | 30.8 | 45.6 | **23.7** | 23.8 |
| Mimir, COMPREHEND + READ2 reader (`brain-school-cx3` with its shelf) | 26.9 | 31.1 | **46.3** | 22.0 | 24.2 |
| Mimir, COMPREHEND reader (`brain-school-cx2` with its shelf; READ2 step 0) | 26.6 | 31.1 | 46.1 | 22.0 | 24.2 |
| Mimir, school brain + cortex (before COMPREHEND, with shelf) | 27.4 | 31.4 | **46.6** | 21.2 | 24.4 |
| Mimir, school brain alone | 27.9 | 25.5 | 45.5 | 21.3 | 20.7 |
| GPT-2 XL (1.5B) | 29.9 | 37.1 | 58.3 | 25.0 | 18.2 |
| Laya (421M) | 33.2 | 44.7 | 50.1 | 30.5 | 40.0 |
| chance | 25 | 20 | 25 | 25 | 25 |

The standard reading tests, once (docs/SURVEY.md, 2026-10-16, `brain-school-cx4`; chance in brackets): SciQ 89.5 (25),
MultiRC 59.5 (~40), Story Cloze 55.5 (50), Winograd 49.6 (50), WinoGrande 50.4 (50), BoolQ 46.7 (50), DREAM 43.0 (33),
bAbI choice version 42.8 → **57.0 with STATE** (task 1 "where is" 66 → 97, he/they 8-10 → 70-72, deduction 0; 94 in words on 10-12; the
other sixteen tests within 0.1 after STATE, rerun 2026-10-17), CBT names 31.7 /
nouns 29.5 / verbs 16.5 / prepositions 4.3 (10), RACE-High 30.7 (25), QuALITY 23.4 (25), ReCoRD 23.1 (~8), CosmosQA
18.6 (25), LAMBADA 10-choice 16.0 (10). Not run: SQuAD, CoQA, QuAC, DROP, NarrativeQA (an answer must be written out).

Reading a passage and answering about it (`bg passages`, with the episode):

| | MC160 | MC500 | RACE-Middle | new stories (30) | harbor/garden/snow (30) |
|---|---|---|---|---|---|
| Mimir + COMPREHEND (reads "not", names; "not" counts against) | 54.7 | 52.5 | 40.4 | 23/30 (salt/train/kite), 21/30 (lighthouse/dragon/night bus, was 16) | 28/30 |
| **Mimir + STATE (the latest state wins; went to = is in; a thing is where its holder is)** | **58.7** | 52.3 | 40.8 | 16/30 (umbrella/goat/key, S0 15), the fifteen older 121/150 both | |
| Mimir + FOLLOW (past / now, a clause after "but", who is who, an exact fact first) | 58.2 | 52.1 | 40.7 | 21/30 (well/bridge/parrot, S0 20.5), 25/30 (orchard/lanterns/ferry), 23/30 (lighthouse/dragon/night bus), 24/30 (salt/train/kite) | 28/30 |
| Mimir + READ2 (he/she, apposition, lists; yes/no judged whole) | 54.5 | 52.1 | 39.8 | 27/30 (orchard/lanterns/ferry, S0 25), 20/30 (lighthouse/dragon/night bus), 23/30 (salt/train/kite) | 28/30 |
| Mimir, episode, passage decides (PASSAGE) | 49.6 | 50.6 | 40.7 | 23/30 (salt/train/kite) | 29/30 |
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

| 10-15 | **COMPREHEND**: reading what a sentence says ("not", names, made of, without) and "not" counting against | MC160 +5.1, MC500 +1.8, new stories 16 → 21 of 30, ARC-C +2.5, OBQA +1.2; ARC-E −3.6, MMLU −1.1, one story question lost (5 better, 3 worse, 3 ties; general mean the same). **Correction:** the general run had no shelf (open book off), so those general numbers do not compare; rerun with the shelf in READ2 step 1. A new "he/she" rule cost MCTest dev and was dropped | **in use** (owner adopted); the old reading and the switches removed |
| 10-15 | **READ2**: step 0 remeasured COMPREHEND with its shelf (ARC-E −0.5, not −3.6: no rule to hunt). **S2**: he/she by sex and recency, apposition at the end of a sentence, lists and "made of" in passives, the main character, yes/no judged whole | dev: 45 vs 43 of 50 story questions, passage mean +0.25, general dev identical. Test, once: new stories 25 → 27 of 30; MC160 −0.2, MC500 −0.4, RACE-M −0.6; general mean +0.1 (ARC-E 46.3, MMLU 26.9); one lighthouse question lost | **in use** (owner adopted); switch removed, `brain-school-cx3` |
| 10-16 | **FOLLOW**: a sense of time (each fact past or now; a fact of another time counts half; the later of two equal facts), "but" starting a new clause, who is who (a known man over a maybe, "it" to the be-subject or the same slot, "them", "said that X was", be-attributes not people), "Who did not …" matched as said, an exact fact silencing open book | dev (Colab, cx3): 67 vs 60 of 70 dev-story questions, passage mean +0.56, general dev the same or +0.2. Test, once (cx4): MC160 54.5 → 58.2, RACE-M 39.8 → 40.7, MC500 the same; new stories 20.5 → 21 of 30; the twelve older test stories 98 → 100 of 120; general mean 30.11 → 30.22 (ARC-C +1.7, ARC-E −0.7) | **in use** (owner adopted); switch removed, `brain-school-cx4` |

Lessons that keep coming back:
* Reading more helps ARC-Easy only; ARC-Challenge and OpenBookQA were built against
  word association, so association-based answering is drawn to the wrong options there.
* What looks like a gain can be test-taking (stage 3). Count only what the brain knows.
* More candidate facts add noise, not answers: the problem is telling a fact that fits by
  chance from one that answers the question.
* General knowledge drowns out what was just read (passage questions with background, 10-12;
  stories with the cortex, 10-14). Fixed by giving the text being read its own place (EPISODE).
* A brain from a release needs its shelf too: without it open book is silently 0 (COMPREHEND's general
  run). The release now carries the shelf; check "open" alone is above chance before trusting a run.
* Sending big files (100 MB and up) from Colab through the shared browser fails. The brain is
  built by GitHub Actions instead and published as a release (no key needed: GitHub's own job
  token); a Colab session downloads it in minutes.
| 10-16 | STATE (docs/STATE.md): the latest state supersedes, derived states from motion / take / give / drop / put, a thing is where its holder is | bAbI test 42.8 → 57.0, new state stories 15 → 16/30, nothing fell | **in use** (switch removed) |

## Open

* **Then (owner 2026-10-17): answering in words** (docs/SPEAK.md): after the understanding program, the frame-with-a-hole
  method on SQuAD / CoQA / QuAC / DROP / NarrativeQA / bAbI in words; each understanding step notes what it adds for it.
* **Now (next in the program): yes/no with thin support** (BoolQ 46.7, picks "no" 70% of the time while 62% of the
  answers are "yes"). Then, carried from STATE: the natural phrasings of a state are not read as states yet ("the key
  is with Mrs. Okafor now", "has a zinc roof now", "not angry anymore", "gave it back to her", "went home with the
  umbrella"); a "before X" question (bAbI task 3) takes the current state; deduction (task 15) 0, path finding (task
  19) 8, counting (7) and lists (8) untouched.
* **The program (owner 2026-10-16: "ensure we improve on all those tests one by one without getting too bad on the
  other")**: one scorecard per survey test or skill, in this order: STATE (bAbI; adopted 10-16) → yes/no with thin support
  (BoolQ) → an inference said in other words (CosmosQA, Story Cloze, DREAM) → pronouns needing knowledge of the world
  (Winograd, WinoGrande) → cloze over names and nouns (CBT-NE / CN, ReCoRD, LAMBADA) → verbs and prepositions as
  answers (CBT-V / P) → long texts (QuALITY, RACE-High). Each step's choice rule lists every earlier test's dev half
  and requires none to fall more than a point; after each adoption the survey's test files are rerun once
  (results/survey.txt grows a column) so the whole picture is always current.
* From the survey, after that: an inference said in other words (CosmosQA 18.6 below chance, Story Cloze 55.5, DREAM
  43.0, QuALITY 23.4: the wrong options reuse the passage's words and win); knowledge of the world joined to one
  sentence (Winograd / WinoGrande at chance; bAbI deduction 0); yes/no with thin support says "no" (BoolQ 46.7, "no"
  70% of picks, yes right 62%); a verb or a function word as the answer (CBT-V 16.5, CBT-P 4.3); answers that must be
  written out (SQuAD, CoQA, QuAC, DROP, NarrativeQA: not runnable yet).
* Following a story on stories it has not seen. FOLLOW's dev gain (60 → 67 of 70) became +0.5 on the three
  unseen stories (well / bridge / parrot, 21 of 30): "What was the roof made of long ago?" → tin (the "now" roof),
  "Who did not get a fish?" → Zainab, "Who does not like the parrot?" → Mrs. Bello, "Who is Mr. Audu?" → a parrot,
  "Does Tayo like the parrot now?" → yes (lost), what Mr. Audu said / who brought the parrot down. The rules fit
  the stories they were built on; the next step needs a way of reading that carries over (test on stories written
  before the code, as FOLLOW did).
* The answering misses the reader no longer causes. On the droplet the facts read right but the
  answer is still wrong: "What is the kite made of?" → plastic (fact: paper, bamboo), "Who is Mr. Bayo?"
  → a doctor (fact: grandfather), "Was Mr. Bayo still sick?" / "Is Bisi shy?" (the "no longer" fact is
  there), "Who won the prize?" ("gave the prize to her"), "How did Mr. Sola pay?" (open book favours
  sentences with the option's words). Likely the open-book part (lexical) outvoting an exact fact.
  Also: the subject lost after a relative clause ("Her grandfather, who …, could not help her").
* Then: sentences, not just words: pronouns ("He" = Dumi), "an owl named Pell", negation,
  relations (cause, kind of, part of).
* Size (owner 2026-10-15: "let's first make it smart then we begin to prune later"): today ~2.8 GB on disk + 2.2 GB shelf, ~5 GB in memory (facts 80 B each = 2.6 GB, cortex 1.7 GB float32, cue lists) against Laya 846 MB, GPT-2 XL 3.1 GB. Pruning comes after the smartness steps: cortex to the ~50k words met often, 8-bit (~30-60 MB); facts packed (~25-30 B); weak facts let go; shelf compressed.
* Later: the cortex small enough for the phone (8-bit, ~30 MB for 50k words); a cap on what
  is kept for good; the GPT-2 vs Mimir chat page (half built).

## Log

* 2026-10-17 12:00: paused by the owner ("stop for now"). State: YESNO dev candidate S5=4062698 (docs/YESNO.md Y2-Y18 less
  Y1/Y3/Y5/Y5b, plus docs/SPEAK.md's answering path); dev run 7 cut short (rerun it, then freeze and test once). Full-brain
  dev so far: bAbI train choice 67.9 → 83.6, bAbI in words 95.6 (seven tasks) / 87.8 (twenty), BoolQ 61.3 (the prior; no real
  "no" signal in the passage, the inference step's work), MC500 dev +1.0, nothing down. Answering in words (docs/SPEAK.md W1,
  W2) exists: bAbI all twenty 90.3 on the tiny brain, SQuAD harness ready (1 of 40 right on the tiny brain; backlog listed).
* 2026-10-17: YESNO (docs/YESNO.md) dev: the "no" lean was an artefact ("yes" counted as an option word, "no" not); with it
  fixed and two reader fixes ("as" + noun phrase, an opening participle) BoolQ train 47.1 → 61.0 and nothing falls, but it
  says yes 97%: the gain is the prior. Owner: "it should reflect real judgement; don't stop until Mimir passes those tests at
  Laya level or GPT-2 or 3 level and can start answering also." No test run; dev continues on the reader and on evidence
  against a statement.
* 2026-10-17: the owner set the direction after understanding: brain-like with nothing behind it, answer in words, reply to
  conversations; docs/SPEAK.md written (the method and the answering benchmarks), the Goal updated. Next: BoolQ.
* 2026-10-17: the survey rerun after STATE (Colab, once, 17 files, 21:50-00:08 UTC): bAbI 42.8 → 57.0, every other test
  within 0.1 of before (results/survey.txt STATE column; docs/SURVEY.md). Runtime deleted. Next in the program: BoolQ.
* 2026-10-16: STATE adopted (owner: "we will adopt it"): the S4 switch and the EPDBG trace removed (byte-identical to
  S4=31 on the tiny brain), tools/state_colab.py deleted, no brain rebuild (episode facts only).
* 2026-10-16: STATE tested once in Colab (S4=31, brain-school-cx4): bAbI test 42.8 → 57.0 (task 1 66 → 97, task 2
  22 → 74, he/she 8 → 70, they 10 → 72); new state stories 15 → 16 of 30; the fifteen older stories 121 of 150 both;
  MC160 58.2 → 58.7, MC500 52.1 → 52.3, RACE-M 40.7 → 40.8; general tests the same. results/state_test.txt. The natural phrasings of a state ("the key is with X now", "has a zinc roof", "not
  angry anymore", "gave it back to her") are not read as states yet: the misses of the new stories.
* 2026-10-16: STATE (docs/STATE.md) dev in Colab on brain-school-cx4: S4=31 (the latest state supersedes and is the
  opposite now; "went to" is "is in", take / give / drop / put are having; a thing is where its holder is) bAbI train
  33.2 → 67.9, dev stories 67 of 70 both, passage mean +0.08, general dev untouched. Frozen at S4=31; test started.
* 2026-10-16: SURVEY (docs/SURVEY.md): the adopted reader on 17 standard reading tests, once, in Colab on
  brain-school-cx4 (results/survey.txt). Strong only where the passage states the answer (SciQ 89.5, MultiRC 59.5);
  chance or below on changed state (bAbI 42.8), inference in other words (CosmosQA 18.6), pronouns needing world
  knowledge (Winograd 49.6), verbs / prepositions (CBT 16.5 / 4.3), yes/no (BoolQ 46.7). Harness: 2-24 options and
  several right letters (byte-identical with four). Owner: "ok let's improve mimir" → next step: state over time.
* 2026-10-16: FOLLOW adopted by the owner ("ok adopt FOLLOW"): switch S3 and FDBG traces removed (byte-identical
  to S3=31 on the tiny brain), brain-school-cx4 is the brain (brain_release.py, brain.yml), follow_colab.py deleted,
  brain_build.txt notes the adoption (rebuilds cx4 with the switch-free reader, as READ2 did); brain-school-cx3
  (and cx2, cx) deleted by the owner the same day: brain-school-cx4 is the only cortex brain left (releases brain-school and brain-E stay).
* 2026-10-16: FOLLOW tested once in Colab (S3=31 reader on brain-school-cx4, 32.47M facts): MC160 54.5 → 58.2,
  MC500 52.1 → 52.1, RACE-M 39.8 → 40.7; new stories well / bridge / parrot 20.5 → 21 of 30; the twelve older
  test stories 98 → 100 of 120; general mean 30.11 → 30.22 (ARC-C +1.7, ARC-E −0.7). results/follow_test.txt.
  Awaiting the owner's call on adoption. Colab runtime deleted.
* 2026-10-16: FOLLOW (docs/FOLLOW.md) dev in Colab on brain-school-cx3: S3=31 (polarity, tense, a clause after
  "but", who is who, an exact fact first) 67 vs 60 of 70 dev-story questions, passage mean 51.32 vs 50.76, general
  dev sets the same or +0.2. Frozen at S3=31; build 7 → brain-school-cx4 for the test.

* 2026-10-16: owner asked how smart the reader is. Two new stories (data/story/redboat.txt, market.txt,
  `smart_tests.tsv`, dev). Red boat (plain, one fact a sentence) 10/10. Market stall (two sisters, "last
  year" vs now, "not silver … made of tin", "did not laugh", "no longer", a relative-clause subject) 5/10,
  the same with the pre-S2 reader. Reading misses: the subject after "but" kept the old subject ("the table
  on it is made of iron" → stall); "it" resolved to a predicate ("silver") instead of the coin; no time on
  facts ("last year the stall was made of wood" ties with "is made of bamboo"). Answering misses: "Who did
  not laugh?" has the fact NOT laugh who=Ngozi but the answering ignores the negation; "no longer" yes/no
  still goes to the option sharing most words. Tiny brain, droplet.

**2026-10-15 (night, adopted)**
* READ2 S2 tested once and adopted by the owner ("adopt it"): new stories 25 → 27 of 30, general mean
  +0.1 (ARC-E 46.3), MCTest/RACE within noise (−0.2 / −0.4 / −0.6), one lighthouse question lost. Switch
  removed (byte-identical to S2=31 on the droplet); `brain-school-cx3` is the brain; READ2 runner deleted;
  `brain-school-cx2` left for the owner to delete (permission guard).

**2026-10-15 (night, later)**
* READ2 S2 built (switch `S2`, S2=0 identical) and dev-run on the school brain: S2 31 wins by the rule
  (45 vs 43 dev-story questions, passage mean +0.25, general dev identical). Frozen; build 6 published
  `brain-school-cx3` (S2 reader on the shelf, 32.96M facts, 761k words). Test started in Colab.

**2026-10-15 (night)**
* READ2 step 0: the general tests rerun on `brain-school-cx2` with its shelf: 26.56 / 31.08 / 46.09 /
  21.98 / 24.20, mean 29.98 vs 30.20 before COMPREHEND. ARC-Easy −0.5 (within the 1-point rule), so
  step 1 is dropped and S2 goes over S0. Build 4 had lost the Wikipedia articles (the API no longer
  marks links "exists"); fixed, build 5 stops if a shelf file is empty.
* Session handover: the Mimir Claude session that ran CORTEX_LINK → READ2 was closed by the owner;
  a new session continues from this journal.

**2026-10-15 (evening)**
* Found: COMPREHEND's general tests ran without the shelf (release had only the brain), so open book
  was off. Release now carries the shelf (build 4); the general tests to be rerun.

**2026-10-15 (later)**
* COMPREHEND tested once and adopted by the owner (mixed: passages and the new stories better, ARC-E and
  MMLU lower). `brain-school-cx2` is the brain now. Owner on size: first smart, then prune.

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
