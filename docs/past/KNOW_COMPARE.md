# Understanding what it reads: facts with pointers, not words

Owner (2026-10-12): "a human can read a lot to understand things but most of the
information is usually not needed. If needed we go back to the information, but
we usually have an idea of where the data is … when we do exams we read, retain
some knowledge or understanding, and answer in our words or see the choice and
pick; we don't keep all information verbatim." Then: "start".

This page fixes the rules **before any code is written**; results go below the
line, whatever they are.

## The four

| | keeps | answers by |
|---|---|---|
| A: today's | every token and the page | sliding window + distance over the page (MCTest paper) |
| B+: neurons growing roles (UNDERSTAND_COMPARE.md, frozen) | event neurons + role frames | B's sliding window over event words + role score |
| **K closed book** | **understanding only**: each sentence read into facts (who / does what / to what / is what / where / when / how many, negation), each fact with a **pointer** to where it was read (which text, which sentence). The words of the text are not kept. | a question and each option become a fact with a gap filled in; the option whose fact best agrees **in meaning** (word meanings, data/meaning) with the facts understood wins |
| **K open book** | the same | as closed book; then it follows the pointers of the facts that matter to the question back to the text, re-reads only those sentences, and adds what they say |

The text K reads stays where it is (a file), like a book on the shelf: it is not
counted in K's bytes, and in closed book K never looks at it again. Nothing is
learned from the test sets; K's reading rules and weights are developed on the
train and dev sets below only.

## Tests (test splits run once, at the end)

| cell | test | why | score |
|---|---|---|---|
| K1 | **RACE-M** test (English reading exams for Chinese middle-school students, ages 12–15; 1,436 questions, 4 options) | ordinary passages, wrong options usually also in the passage — word matching is not enough | % correct |
| K2 | **SciQ** test (884 science questions with their passage, 4 options) | factual text like a textbook | % correct |
| K3 | MCTest MC160 test (240) | children's stories | % correct |
| K4 | MCTest MC500 test (600) | children's stories | % correct |
| K5 | **size of understanding**: K's facts + pointers against the text it read (RACE-M test passages; War and Peace) | the owner's rule: keep understanding, not words | bytes |

Development: RACE train/dev (middle), SciQ train/dev, MCTest train/dev. Dev
starting points (today's readers): RACE-M dev A 44.7%, SciQ dev A 91.9%, MC160+500
train/dev ≈ 61%.

Also shown, not scored: K closed against K open (how much going back to the text
adds); Laya on RACE-M test, run by us in Colab if it can be. The owner's own tests
(the maths test, the idioms test to come) are **not** used for development; they
are run later on a frozen build.

## Who wins

K open and K closed are each counted against A: more than 1 point wins a cell,
10% for bytes; otherwise a tie. The count is given as it is.

---

## Results (2026-10-12)

Harness `sandbox/neurons/know.c`, frozen in 52f2902 before the test sets were
opened. Developed on RACE-M dev, SciQ dev and MCTest train + dev: the reader got
"reader 2" fixes (pronouns to the latest person or thing named, verb particles,
"would like"; behind a switch, so B+ is unchanged); an option is put where the
question asks ("…one or more of *nucleotides* that…"); the decisive measure
became **one place** — a sentence's facts in closed book, the re-read sentences in
open book — that covers both the question and the option in meaning; weights
chosen there. Test sets run once.

| cell | test | A: today's | B+ | K closed | K open | winner (K open / K closed vs A) |
|---|---|---|---|---|---|---|
| K1 | RACE-M (1,436) | **45.33%** | 42.34% | 36.28% | 41.09% | A / A |
| K2 | SciQ (884) | 91.18% | 91.06% | 87.67% | **93.10%** | **K open** (+1.9) / A |
| K3 | MC160 (240) | **67.08%** | 68.33% | 56.25% | 63.75% | A / A |
| K4 | MC500 (600) | **57.83%** | 54.33% | 50.83% | 52.83% | A / A |
| K5 | size | page + tokens: War and Peace 11.3 MB | 3.9 MB | **2.42 MB** (facts + pointers 2.26 MB + word map 0.16 MB) | same | **K** (4.7× under A) |

**Count — K open: 2 (SciQ, size), A: 3 (RACE-M, MC160, MC500). K closed: 1
(size), A: 4.**

Size against the text itself (not A): War and Peace understood in 2.42 MB, 74% of
its 3.27 MB text; RACE-M test 0.81×, SciQ 0.65×, MCTest 0.90–1.00×. The facts still
keep nearly every content word — nothing is forgotten yet — so understanding is not
yet much smaller than the words.

### What it shows

* **Closed book, from facts alone, loses 4–11 points to keeping the page.** The
  facts drop function words, word order inside phrases, most adjectives beyond one,
  and whatever the reader misreads; the RACE and MCTest questions often turn on
  exactly those.
* **Open book recovers most of it** (+1.6 to +7.5 points over closed book) by
  going back to only the few sentences its facts point to, and beats keeping the
  page on SciQ — factual text where the right sentence is easy to find.
* **On stories and exams it is still word matching underneath.** The parts that
  check meaning help (fact agreement alone is 30–43%, barely over chance), but the
  answer mostly comes from "which place mentions both": no reasoning, no general
  knowledge. RACE-M's real difficulty (inference, the writer's purpose, "which is
  true") is untouched: all of today's readers are at 41–45%; middle-school students
  score about 85%.

Results: `results/know_*.txt`.
