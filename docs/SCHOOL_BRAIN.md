# The school brain: from baby to high school graduate

Owner (2026-10-13): "remove that limit, let it read as many as it wants. … using Google
Colab, read every book necessary for a student, from baby till high school, so that
normal communication becomes easy. It doesn't need to know how to do tough things,
just the general knowledge any high school graduate should have. That will be its
base brain. Then I can expand it later on one specific field, like a human child."

This page fixes the rules **before any code is written**; results go below the line,
whatever they are.

## What changes in Mimir

* **No room limit.** Mimir keeps every fact it reads. The only limit is the machine:
  if free memory falls under 1.5 GB, the weakest quarter is forgotten (as before) and
  this is reported. Strength and fading stay (STRENGTH_COMPARE.md): what is met again
  stays strong, what is not fades in confidence, but is no longer thrown away.
* Nothing else changes in reading. No practice questions in this first school brain:
  it only reads, so what it knows comes from the books.
* Answering uses the settings frozen in LAYA_LEVEL.md stages 1–2 (re-read behind 20
  facts, question polarity).

## The shelf, read in this order (like a child growing up)

| stage | age | what | source (open licence) |
|---|---|---|---|
| 1 | 0–6 | speech to and around children, picture-book stories | BabyLM 100M: CHILDES transcripts, children's stories, BNC spoken, subtitles, TED (Hugging Face `nilq/babylm-100M`) |
| 2 | 6–12 | children's books | Project Gutenberg, the Children's Literature and children's instruction shelves (Gutendex `topic=children`, English) |
| 3 | 6–14 | children's encyclopedia and general knowledge | Wikijunior (Wikibooks) and Simple English Wikipedia |
| 4 | 12–18 | high school textbooks | OpenStax books from their markdown mirrors (`philschatz/*-book`): concepts of biology, biology, anatomy, chemistry, physics, astronomy, US history, economics, psychology, sociology, statistics, algebra and trigonometry, precalculus; then Wikibooks (open textbooks). World history, American government and prealgebra have no mirror there and are left out |
| 5 | 14–18 | what an educated person knows | Wikipedia's Vital Articles, level 4 (10,027 articles Wikipedia's editors list as most essential; text from wikimedia/wikipedia 20231101.en) |

Files in reading order: `01_babylm`, `02_children`, `03_wikijunior`, `04_simplewiki`, `05_textbooks`, `06_wikibooks`, `07_vital`.
All downloaded and read in Colab (`tools/school_brain.ipynb`); after every shelf file the brain is sent through the browser to the server (`tools/school_receive.py` checks it against its SHA-256 list and keeps the newest), and a restart continues from it (owner chose this over granting Colab's full Drive permissions); the books stay on the shelf so
open-book answering can re-read them; the brain is packed (`bg pack`) for release.

## Tests (once, in Colab, after the brain is frozen)

Three brains, the same answering settings:

| | brain |
|---|---|
| **empty** | nothing read (word meanings only) |
| **E** | today's base (science reading, VERSUS.md) |
| **school** | the school brain above |

| cell | test | what it measures |
|---|---|---|
| S1 | MMLU, high-school subjects (test split, all `high_school_*` subjects) | what a high school graduate should know, across subjects |
| S2 | CommonsenseQA (validation, 1,221, 5 options) | everyday common sense |
| S3 | ARC-Easy test | grade-school science |
| S4 | ARC-Challenge test | harder science |
| S5 | OpenBookQA test | applying science facts |
| S6 | brain: facts, size, reading time | |

School against empty and against E, per cell: more than 1 point = better. The count
is given as it is.

Note: Mimir does not speak; "normal communication becomes easy" is measured here as
understanding (questions about everyday things and school subjects), not as talking.

---

## Results

(to come)
