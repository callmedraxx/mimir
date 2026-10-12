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
