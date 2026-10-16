# The state now: the latest fact wins (state)

Owner (2026-10-16): "ok let's improve mimir", after the survey (docs/SURVEY.md).

## Why

A story changes things as it goes: Sandra goes to the bedroom, then to the kitchen; the stall was wood, then
iron; Baba Ike was angry, then not. Mimir reads each change as a fact and keeps them all with equal weight.
Asked "Where is Sandra?" it scores every place she has been nearly alike (fact 0.64 vs 0.67 on the tiny brain
after two moves) and the later fact only wins an exact tie; after three moves it is a coin toss. That is bAbI
task 1 at 66% (choice version; the 10-12 reader answered it in words at 100%), task 2 and 3 at 22 / 29% ("Where
is the football?": Mary took it, Mary went to the kitchen), and the FOLLOW misses "What is the stall made of
now?", "Does Tayo like the parrot now?", "What was the roof made of long ago?". A human keeps one current
state per thing and remembers the earlier ones as "before".

## Steps (switch `S4`: 1 S1, 2 S2, 4 S3; `S4=0` identical to the adopted reader)

**S1. One state per slot: the latest supersedes.** Inside an episode, facts about the same subject that
fill the same slot with a different value are one *state*: motion verbs with "to" (go, move, travel, journey,
walk, run, return, come, head, get, drive, fly, ride, ... → the subject's place), "be" / "become" / "stay"
with a place or an attribute (angry, afraid, made of, colour, called ...), "have" / "hold" / "carry" / "get" /
"take" / "pick up" / "grab" / "give … to" (who has a thing). The latest fact of a state is *current*; the
earlier ones are *superseded* (a flag bit in the episode fact, set as the episode is read; nothing in the
brain file changes). When a question asks about the present (present tense, "now", "still", no time word;
"Where is X?", "Who has Y?", "What is Z made of?"), a superseded fact counts SUPW (dev, 0.25) of itself and a
current fact counts whole; when the question asks about the past ("Where was X before …?", "What was Z made
of long ago?", a past-tense statement), superseded facts count whole and nothing is discounted (F2's tense
rule still applies). A drop / loss ends a possession state ("Ruth lost the key": nobody has it until someone
finds / takes it).

**S2. "Is at" means "went to".** The option statement "Sandra is in the kitchen" matched against the fact
"go who=sandra pp:to=kitchen": a motion-with-"to" fact is also read as a place state "be who=sandra pp:in=
kitchen" (added as a second episode fact with the same clock, flagged as derived), so "Where is X?" meets a
"be" fact and the verb mismatch (be vs go, which costs part of the fact score) is gone. The same for "get /
take / pick up Y" → "have who=X what=Y" and "give Y to Z" → "have who=Z what=Y" (and X's "have Y" ends).

**S3. A thing is where its holder is.** "Where is the football?" when the football's latest state is "have
who=mary": the answer is Mary's current place, as a derived fact "be who=football pp:in=<Mary's place>" kept
in step with Mary's moves while she holds it, and fixed where she dropped it when she does (bAbI tasks 2, 3).

Each step is a bit of `S4`; bits that lose on the tiny brain are dropped before Colab.

## Rules

* **Dev**: the tiny droplet brain for guidance, then Colab on `brain-school-cx4`: bAbI **train** tasks 1, 2,
  3, 6, 9, 11, 13 (choice version, 200 each, `data/understand/survey/babi_train.txt`); the dev stories (70:
  clock, bicycle, stone owl, swim, bakery, red boat, market stall); MCTest MC160 / MC500 dev, RACE-M dev; the
  five general dev sets. The owner's private tests are not used.
* **Choice**: kept if bAbI train tasks 1-3 gain at least 20 points on their mean, the 70 dev-story questions do
  not fall, the passage dev mean (MC160, MC500, RACE-M) is not more than 0.5 lower and no general dev set is
  more than 1 worse. At most three Colab dev runs.
* Then frozen (commit), **once** on: bAbI **test** tasks 1-20 (the survey's 200 a task), the three new stories
  written before this code (`data/story/state_tests.tsv`: the blue umbrella, the goat and the gate, the
  library key; who has it now / where is she now / made of now and before / still angry; tiny brain before:
  57.5%), the fifteen older test stories, MC160 / MC500 / RACE-M tests, the five general tests. The brain is
  not rebuilt (the shelf's facts have no episode; S1-S3 act on episode facts only).
* All runs in Colab, each runtime deleted after; droplet for code and the tiny brain only. Reported as they
  come; journal updated; losers' code deleted; the switch removed on adoption.
