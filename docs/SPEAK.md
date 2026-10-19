# Answering in words (speak)

Owner (2026-10-17): "the goal is to be brain like no backdrop. let's improve the understanding and start working on a
method no matter how unconventional that will help mimir to be able to answer questions with words so that it can
reply to conversations so after we nail the understanding tests benchmarks let's nail the answering test benchmarks."

This scorecard is written now so that every understanding step keeps what answering will need, and so the method is
settled before the code. The runs come after the understanding program (docs/SURVEY.md, the running scorecard),
unless the owner moves it up.

## Why

Mimir picks among options that someone else wrote. It has no part that produces a word, so it cannot reply, and five
of the standard tests could not be run (SQuAD, CoQA, QuAC, DROP, NarrativeQA: the answer must be written out). An
LLM speaks because it is a next-word predictor and nothing else; its understanding is a by-product of speaking. Mimir
is built the other way round and stays so: **meaning first, words after, no model of language behind it.** The brain
does the same: speech starts from a thought (slips of the tongue swap whole words of one kind; the
tip-of-the-tongue state has the meaning before the word), memory is apart from language (a person without a
hippocampus speaks fluently and remembers nothing new), and prediction of the next word is a habit that serves the
thought, not the other way round.

## The method

Mimir already turns a sentence into a frame (`facts.c`: verb, who, what, is-what, place, time, past / now, not). The
same reader, run on the question, gives a **frame with a hole**; answering is filling the hole from memory and saying
what filled it. Five parts, in the order they are built:

**W1. The question is a frame with a hole.** The question word marks the slot: *who* → the person in the who or what
slot; *where* → the place; *when* → the time; *what / which X* → the what slot, the is-what slot or the thing of kind X;
*how many / how much* → a count; *why* → the cause (a relation the reader does not read yet; the first version
re-reads the sentence that follows "because" / "so"); *how* → the manner or the means ("by …", "with …"). A yes/no
question is a full frame, judged true / false / unknown as the yes/no judging does now. The same episode and shelf
candidates that score options today fill the hole: the fact closest to the frame, with the hole's slot filled, in
the episode first (weight EPW), the shelf after. STATE's current/superseded facts and FOLLOW's past / now apply as
they do for options, so "Where is Sandra?" takes the current place and "Where was Sandra before?" the superseded one.

**W2. Say the filler as the text said it.** The short answer is the filler's own phrase, taken from the sentence the
fact points to (the pointer is already there: the open-book part re-reads it). The full answer is the question's
frame with the hole filled, put into words by the sentence that filled it: Mimir echoes the question with the answer
in it ("Where is the football?" → "The football is in the garden"). Children answer this way before they can
compose. No new words are invented: every word said is either the question's or the text's. This is what SQuAD,
CoQA, QuAC and DROP measure (exact match and word F1 against the reference span).

**W3. Nothing fills the hole: say so.** When no fact fits the frame above a threshold (set on dev), the answer is
"I don't know" (SQuAD 2.0's unanswerable half, QuAC's "cannot answer"). This is the brain-like default and it is
honest; an LLM has no such state. The threshold is the only knob.

**W4. A conversation is an episode.** Each turn, question and answer alike, is read into the episode as the passage
was, so the pronouns of the next question ("Where did *she* go then?") resolve by the rules FOLLOW and READ2 already
have, and an answer given becomes a fact Mimir remembers having said. CoQA and QuAC are conversations over a passage;
a chat with the owner is the same thing without the passage: the shelf is the memory it answers from, and what the
owner says is read in and kept (the owner's memory is the brain's, docs/JOURNAL.md Goal).

**W5. A habit of word order, later.** Echoing the text is stilted. The brain's fluency is a sequence habit (the
next word is pre-activated before it is heard: the N400, skipped predictable words). Mimir's version: a small
sequence memory over the shelf's words, learned the Hebbian way the cortex is (each word strengthens the link to the
one that follows it, weakened when it does not), used only to **order and connect** the words a frame already chose,
never to add content. Pruned to the words met often it fits the phone. This is the last part, and it is not an LLM:
it chooses no fact, it holds no knowledge, it is the mouth and not the mind.

Unconventional and allowed by the owner: no backpropagation anywhere, no pretrained model behind it, no text that
does not point back to a page in the shelf or a turn in the conversation.

## Benchmarks (the answering tests)

| test | what | n | measure | published |
|---|---|---|---|---|
| bAbI 1-20 in words | the state / list / count / yes-no answer as a word | 20 × 1,000 test | exact match | the roles reader (10-12) 94 mean on the same tasks |
| SQuAD 1.1 dev | a span of the paragraph | 10,570 | EM / F1 | sliding window 20 F1, logistic regression 51, BiDAF 77, human 91 |
| SQuAD 2.0 dev | half unanswerable (W3) | 11,873 | EM / F1 | BiDAF++ 65 F1, human 89 |
| CoQA dev | conversation over a passage (W4) | 7,983 | F1 | DrQA + PGNet 65, human 89 |
| QuAC dev | conversation, "cannot answer" (W3 + W4) | 7,354 | F1 | BiDAF++ 60, human 81 |
| DROP dev | numbers, dates, counting | 9,536 | EM / F1 | NAQANet 47, human 96 |
| NarrativeQA (summaries) | free answers | 10,557 | Rouge-L | BiDAF 36, human ~ |
| the hand-written stories (18 so far) | the right option's text as the reference | 180 | F1 | |

**Dev**: SQuAD 1.1 train (a fixed 2,000), bAbI train in words (tasks 1, 2, 6, 9, 11, 13 as STATE used), CoQA train
(a fixed 500 conversations), the dev stories answered in words. **Test, once**: the dev splits above (the standard
test splits are hidden). The owner's private tests are never developed on.

## Rules

* Measurement first: `bg ask <brain> <file>` (W1 + W2 with no threshold) run once on the dev sets gives the baseline
  before any tuning; the choice test files already in `data/understand/survey` are reused where the right option's
  text is the reference.
* Every earlier test (the survey's 17 files, MCTest / RACE, the stories, the five general tests) stays in the
  choice rule: answering in words must cost nothing in choosing among options (none falls more than a point).
* All runs in Colab, once per test, runtime deleted after; droplet for code and the tiny brain. Reported as they come;
  losers' code deleted; the journal updated. Adoption is the owner's call.
* Order: W1 + W2 (SQuAD 1.1, bAbI in words, the stories) → W3 (SQuAD 2.0, QuAC) → W4 (CoQA, QuAC) → DROP's
  counting and dates (with bAbI 7 counting, which is 42 today) → W5 last, on the stories and a conversation with the
  owner, judged by the owner.

## First measurement (2026-10-17, droplet, tiny brain; `bg passages` on A lines, `tools/babi_words.py`)

W1 + W2 in their first form: the question read as a frame with a hole (the reader's own wh slot), the episode fact
closest to it that fills the slot, the filler's lemma said. bAbI train in words, STATE's seven tasks, 200 each:
**73.3** (task 1 where-is 99, 2 a carried thing 78, 3 "before" 14, 6 yes/no 83, 9 negation 82, 11 he/she 73, 13 they
84). The yes/no tasks were 52 / 59 until the judgement used STATE's one-value-per-slot rule (docs/YESNO.md Y12: the
current state of the asked slot says yes with the same value, no with another). The roles reader of 10-12 had 94 on
these in words; the frames here are the general reader's, not bAbI's. Not run on the full brain yet (Colab, with the
next YESNO dev run).

Then three answers from STATE's slots (W1b, W1c): "Where was the football before the bathroom?" (the subject's place
facts in the order said: the last value before the named one), "What is Mary carrying?" (every thing whose current
holder is Mary, "nothing" when none) and "How many objects is Mary carrying?" (their count as a word). All twenty tasks
in words, 200 each, tiny brain: **62.6** (1:99 2:78 3:55 4:36 5:74 6:83 7:80 8:81 9:82 10:82 11:73 12:90 13:84 14:72 15:24
16:0 17:46 18:52 19:0 20:58). With Y13 (a STATE order bug found through this path) and Y14 (pronouns), tiny brain: task 3 97, 11 99.5, 12 99, 13 99;
all twenty 67.4. Then W1d, what is said of the subject's kind (deduction: "Gertrude is a mouse. Mice are afraid of wolves")
or of the other members of its kind (induction: the commonest value; "what colour" takes an adjective, not the kind), and
W1e, a relation read the other way ("the office is north of the kitchen" answers "what is the kitchen south of"): all
twenty **80.3** (1:100 2:84 3:97 4:100 5:76 6:86 7:86 8:83 9:84 10:84 11:100 12:99 13:99 14:72 15:100 16:96 17:46 18:52
19:0 20:60). Every rule here is a general reading rule (a kind's facts hold for its members; a relation has its inverse),
none is bAbI's: they are what the shelf's facts will need too. With Y15 (states follow the time words said: "yesterday … this
morning …", docs/YESNO.md) task 14 72 → 97 and all twenty **81.5**. With Y16 (relations, inverses and chains) positions 46 → 100 and sizes 52 → 99: all twenty **86.5**. With W1f, a path
over the places' relations ("How do you go from the kitchen to the garden?" → "south east", bAbI 19 0 → 100): **91.5**. A derived state (has, is in) now answers only a question about a state, not "who gave" (task 5 76 → 78), putting down
and leaving are dropping and "either … or" is maybe (Y17: tasks 7, 8, 10 to 100 / 99 / 96), and a question with only a
weak fact and no kind to lean on gets no answer ("-") instead of a guess: task 20 falls from 60 to 6, because its 60 was
luck (someone else's destination) and its answers ("bored people go to the garden") are the world's knowledge, not the
story's. All twenty **90.9**, with that honest loss. Still open: three-argument facts (5 at 78), negation and yes/no in words
(6, 9 at 84-86), and motives (20: "Sumit is tired. Where will Sumit go?" needs the world's knowledge that tired people go to bed;
only the shelf can give that). Left: relations (4 "east of"), deduction (15), induction (16 "what colour is Greg": by his
kind), positions (17), size (18), paths (19), motives (20, "will go"), and the pronoun tasks at 73-84.

## bAbI in words, test (2026-10-19, Colab, once, full brain, the frozen YESNO candidate; `results/yesno_test.txt`)

All twenty tasks, test split, 200 each: **91.4** exact match. By task: 1:100 2:90 3:92 4:100 5:76 6:100 7:100 8:100 9:85
10:100 11:100 12:100 13:100 14:96 15:100 16:98 17:88 18:99 19:100 20:4. The roles reader of 10-12 had 94 on tasks 1-20
in words with rules written for bAbI; these are the general reader's rules, and task 20's 4 is the honest number for
"where will a tired person go" without the world's knowledge.

## Say it, then choose it (Y19, bit 4194304; 2026-10-19, tiny brain)

For a question whose options are single words (a name, a place, yes / no), Mimir first answers in words and, when a rule
decided the answer (the state slot, a relation chain, a kind, a count, a path, "before"; or a fact close to the
question), picks the option that says the same. bAbI train choice 83.2 → 92.8 ("before" questions 36 → 97), BoolQ,
RACE-M dev, the dev stories and bAbI in words unchanged, MC500 dev one question down (59.4 → 58.9). Reading a question
now leaves the reader as the story left it (names, sexes, the sentence count, the pronouns' people are snapshotted
and restored), which the earlier question reads did not. Full brain (dev run 8, `results/yesno_dev.txt`): bAbI train
choice 83.6 → **95.6**, everything else identical to the frozen candidate (BoolQ 61.3, words 95.6 / 91.5, dev stories
67/70, MC160 51.9, MC500 57.75, RACE-M 41.0, general dev the same). Y19 came after the YESNO freeze, so it is not in the
candidate under test; it goes into the next step's test.

## SQuAD, first look (2026-10-17, tiny brain, the first 40 dev questions as a format check only)

The harness runs (`tools/squad_words.py`, A lines with several right answers, exact match and word F1). Said right: "Cam
Newton", "Carolina Panthers" (after Y18 kept the object before a score), "24-10" (a number said in figures from the
sentence). What the misses need, in order of reach: (1) "What was the theme / final score of X?": the reader makes
"the N of X" a relation (as for "west of the kitchen"); the N is the thing asked about, with "of X" its phrase; (2) a
predicate said in other words ("won" for "defeated … to earn the title", "take place" for "was played at"), which is
the inference step's skill, with the cortex's word links on the full brain; (3) "the game" for "Super Bowl 50"
(coreference by kind); (4) why-questions (the cause: a state said just before the deed, or a "because" clause); (5)
the world's knowledge for bAbI 20 ("tired people go to bed"), which only the shelf can give. The dev measure for SQuAD
is the first 2,000 of train, as the scorecard says; nothing is tuned on these forty.

## When

After the understanding program's steps (BoolQ next), or when the owner says so. Each understanding step from here
notes under "For speaking" what it adds that W1-W4 will use (STATE: the current place and holder, the superseded
"before"; BoolQ: a true / false / unknown judgement for W3).
