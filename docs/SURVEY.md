# The standard reading tests, once (survey)

Owner (2026-10-16): "how about we run the current mimir on all of those tests all of them and see how it does
and notes why it performed well or fail each".

## Why

Mimir has been measured on bAbI (10-12, 94%), MCTest, RACE-Middle, the five general sets and fifteen
hand-written stories. The reading tests the builders of language models used are a wider set: real children's
books (the Children's Book Test), novels (LAMBADA), everyday five-sentence stories (Story Cloze), pronouns
(Winograd, WinoGrande), narrative with commonsense questions (CosmosQA), dialogue (DREAM), yes/no on Wikipedia
(BoolQ), several sentences at once (MultiRC), news cloze over named entities (ReCoRD), whole stories from
Project Gutenberg (QuALITY), high-school exams (RACE-High). This step **measures** the adopted reader
(docs/FOLLOW.md, `brain-school-cx4`) on all of them once and writes down, per test, why it did well or badly.
No reading or answering code changes here; whatever it shows goes into the journal's Open list.

## The tests and how each becomes a passage file

Mimir answers by choosing among given options about a passage (`bg passages`: `S` passage id, `T` text,
`Q id question option… letter`). The harness is widened to 2–24 options and to questions with more than one
right option (letters "AC"); with four options and one letter it is byte-identical to before. Each test is
turned into that form by `tools/survey_fetch.py` (HuggingFace's rows API or the original files; nothing is
rewritten by hand). Where a set is large, the first N questions are taken, in the published order.

| test | source, split, size used | passage | question | options | note |
|---|---|---|---|---|---|
| **Children's Book Test** (Hill et al. 2016) | `cam-cst/cbt` test: NE, CN, V, P, 1,000 of 2,500 each | the 20 sentences of the book | the 21st sentence with "_" for the hidden word | the 10 given candidates | as published |
| **LAMBADA** (Paperno et al. 2016) | `EleutherAI/lambada_openai` test, 2,000 of 5,153 | the passage without its last word | its last sentence with "_" | the last word + the 9 most frequent other content words of the passage | **10-choice version**, not the word-prediction test the LLM numbers are |
| **Story Cloze** (Mostafazadeh et al. 2016) | `MoE-UNC/story_cloze` test, 1,871 | the four sentences | "What happened next?" | the two endings | as published |
| **Winograd Schema** (Levesque 2012) | `lighteval/winograd_wsc` wsc273, 273 | the sentence | the sentence with "_" for the pronoun | the two candidates | as published |
| **WinoGrande** (Sakaguchi et al. 2020) | `allenai/winogrande` xl dev, 1,267 | the sentence (its "_" kept) | the same | the two | as published |
| **CosmosQA** (Huang et al. 2019) | `Samsoup/cosmos_qa` dev, 2,985 | the context | the question | the four | as published |
| **DREAM** (Sun et al. 2019) | `nlpdata/dream` test, 2,041 | the dialogue, turns joined ("Man: …", "Woman: …") | the question | the three | as published |
| **BoolQ** (Clark et al. 2019) | `google/boolq` dev, 3,270 | the Wikipedia passage | the question | yes / no | as published |
| **MultiRC** (Khashabi et al. 2018) | `super_glue` multirc dev, 4,848 answer rows → questions | the paragraph | the question | all its answer options, several may be right | **one pick per question**, right if it is one of the right ones (the published scores are per-option F1 and exact match) |
| **ReCoRD** (Zhang et al. 2018) | `super_glue` record dev, 2,000 of 10,000 | the news passage | the query with "_" for the placeholder | the entity candidates (up to 24) | as published (several may be right) |
| **QuALITY** (Pang et al. 2022) | `tasksource/QuALITY` dev, 1,000 of 2,086 | the whole story (2,000–6,000 words) | the question | the four | as published |
| **RACE-High** (Lai et al. 2017) | `tools/race_fetch.py` test_high, 3,498 | as RACE-Middle | | four | as published |
| **bAbI** (Weston et al. 2015) | `data/understand/babi` tasks 1–20 test, 200 of 1,000 each | the story so far | the question | the answer + up to 9 other answers seen in that task (seeded) | **choice version**; 10-12 answered in words (94%) |
| SciQ (Welbl et al. 2017) | `data/understand/sciq` test, 884 | the support paragraph | the question | four | already in the repo |

Not run, and why: **SQuAD, CoQA, QuAC, DROP, NarrativeQA** ask for an answer written out (a span of the text,
a number, a free sentence). Mimir chooses among options; it does not yet write an answer. That is a gap in its
own right and goes to the journal's Open list.

## What is measured and written down

* Accuracy per test (`RESULT` lines) against chance and, where it exists, the published LLM or baseline number
  (GPT-2 1.5B on CBT-CN 93.3 / CBT-NE 89.1 / LAMBADA 63.2 / Story Cloze 86.4 (GPT-3 zero-shot 83.2) /
  Winograd 70.7; GPT-3 175B zero-shot WinoGrande 70.2 / BoolQ 60.5 / MultiRC 72.9 F1a / ReCoRD 90.2 /
  RACE-H 45.5; CosmosQA BERT-large 68.3, DREAM BERT 63.2, QuALITY RoBERTa 39.5 (human 93.5); bAbI 1k MemN2N
  mean ~86%).
* For each test the picks (`SHOW=1`) and ten misses read by hand; the note says what the test asks of a reader
  and which part of Mimir answered or failed (the episode facts, open book, word similarity, the "not" rule).

## Rules

* Measurement only: no change to facts.c or the answering, no knob moved; the harness widening is checked
  byte-identical on the FOLLOW dumps (tiny brain) before the run.
* Format checked on the droplet with the tiny brain on the first 40 questions of each test; then **one** Colab
  run (`tools/survey_colab.py`, `brain-school-cx4` with its shelf, free CPU runtime, runtime deleted after),
  results in `results/survey.txt`, reported as they come.
* Nothing here is a dev set for later steps unless the owner says so; the published dev splits used here are
  named so a later step can choose its dev and test halves.

## Results (2026-10-16, Colab, once, `brain-school-cx4`; `results/survey.txt`)

| test | n | Mimir | chance | published | what the misses show |
|---|---|---|---|---|---|
| SciQ (support paragraph) | 884 | **89.5** | 25 | | the paragraph states the answer in the question's words: the exact fact and the word parts work |
| MultiRC (one pick a question) | 950 | 59.5 | ~40 | GPT-3 72.9 F1 | options restate a sentence; misses are inferences ("Total memory loss") |
| Story Cloze | 1,871 | 55.5 | 50 | GPT-2 86.4 | picks the ending with the story's words; the right one is a consequence ("ate the whole pie" → "got sick") |
| Winograd Schema | 273 | 49.6 | 50 | GPT-2 70.7 | picks the subject nearly every time: "_ was too small" needs what fits into what, not a fact in the sentence |
| WinoGrande | 1,267 | 50.4 | 50 | GPT-3 70.2 | the same: "the _ was canceled" (class, not story) is knowledge of the world joined to the sentence |
| BoolQ | 3,270 | 46.7 | 50 | GPT-3 60.5 | says "no" 70% of the time while 62% of the answers are yes: when the fact support is thin the yes/no judging falls to "no" |
| DREAM (dialogue) | 2,041 | 43.0 | 33 | BERT 63.2 | "What does the man mean?", "Where does the conversation take place?": inference from turns; picks the option with the dialogue's words |
| bAbI, 20 tasks, choice version | 4,000 | 42.8 | varies | 94 in words (10-12) | task 1 "Where is Sandra?" 66%: every move is a fact ("go … to kitchen"), the latest does not beat the earlier ones and "is" does not match "go"; he/they 8 / 10%, deduction 0, path 8 |
| CBT named entities | 1,000 | 31.7 | 10 | GPT-2 89.1 | three times chance: a fact's shape carries ("the King", "the Princess"); misses pick another character of the same kind |
| RACE-High | 3,498 | 30.7 | 25 | GPT-3 45.5 | longer passages, "the passage mainly suggests": inference over the whole text |
| CBT common nouns | 1,000 | 29.5 | 10 | GPT-2 93.3 | as named entities; misses are nouns in a new role ("at the top of his shrill little _" → voice) |
| QuALITY (whole stories) | 1,000 | 23.4 | 25 | RoBERTa 39.5 | motives and reasons over 5,000 words; at chance |
| ReCoRD (entity cloze) | 2,000 | 23.1 | ~8 | GPT-3 90.2 | the query paraphrases a sentence; picks a frequent entity of the right kind |
| CosmosQA | 2,985 | 18.6 | 25 | BERT 68.3 | **below chance**: the right answer is an inference in other words ("he likely fell"), the wrong ones reuse the passage's words, and Mimir follows the words |
| CBT verbs | 1,000 | 16.5 | 10 | ~70 | the blank is the verb: facts are keyed by their verb, so none matches; picks a verb seen in the passage |
| LAMBADA (10-choice) | 2,000 | 16.0 | 10 | GPT-2 63.2 (prediction) | "said _": who is speaking needs the turn-taking of the dialogue; picks a frequent word |
| CBT prepositions | 1,000 | 4.3 | 10 | ~70 | **below chance**: a preposition is no fact; the candidates that look like content words win |

Not run (the answer must be written out): SQuAD, CoQA, QuAC, DROP, NarrativeQA.

**What this says about Mimir.** It answers when the passage states the answer in words close to the question's
(SciQ 89.5, MultiRC 59.5, MCTest 58 / 52, the hand-written stories). It is at or below chance wherever the answer
is (1) a state that changed over the text (bAbI 1-3: where someone is now; the FOLLOW misses "made of now", "does
he like it now"), (2) an inference said in other words (CosmosQA, Story Cloze, DREAM, QuALITY), (3) knowledge of the
world joined to one sentence (Winograd, WinoGrande), (4) a function word or a verb (CBT-V, CBT-P), or (5) a yes/no
judgement with thin support (BoolQ leans "no"). The first is the cleanest to fix and has a ready dev set (bAbI
train, 1,000 a task) and test (bAbI test, the stories' "now" questions): the next step.
