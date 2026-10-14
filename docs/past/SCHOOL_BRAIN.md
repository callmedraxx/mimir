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
| 2 | 6–12 | children's books | Project Gutenberg, English books on its "Children's …" and "Children & Young Adult Reading" shelves (6,929 in Gutenberg's own catalogue `pg_catalog.csv`; Gutendex was too slow, 65 s a page) |
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

## Results (2026-10-13)

**Reading** (Colab, 2 cores, 12 GB): the whole shelf, 2.2 GB of text (BabyLM 535 MB,
children's books 936 MB from 6,929 Gutenberg titles, Wikijunior 5 MB, Simple English
Wikipedia 210 MB, textbooks 24 MB, Wikibooks 283 MB, Vital Articles 207 MB), in about
11 minutes. **30.6M sentences, 32.5M facts kept (nothing forgotten: no room limit, memory
never ran low), 725k words; 1.08 GB brain** (494 MB gzipped). The per-stage log was lost
when the shared browser was restarted under memory pressure on the server; the brain was
already complete and arrived intact (checked against its SHA-256 list,
`results/school_brain_manifest.txt`).

**Tests** (once, Colab, the same answering settings for all three brains):

| cell | test | empty | E (science) | **school** | chance |
|---|---|---|---|---|---|
| S1 | MMLU high school (3,420) | 24.98% | 26.66% | **27.88%** | 25% |
| S2 | CommonsenseQA dev (1,221) | 20.86% | 26.92% | 25.50% | 20% |
| S3 | ARC-Easy (2,376) | 28.69% | 46.46% | 45.48% | 25% |
| S4 | ARC-Challenge (1,172) | 22.97% | 22.96% | 21.26% | 25% |
| S5 | OpenBookQA (500) | 24.80% | 17.67% | 20.67% | 25% |
| S6 | brain | 0 facts | 4.72M facts, 167 MB | 32.5M facts, 1.08 GB | |

Answering time: school 142–247 ms a question against E's 36–73 (a brain 7× larger).

| pair | count |
|---|---|
| school vs empty | **better 3** (MMLU +2.9, CommonsenseQA +4.6, ARC-Easy +16.8), worse 2 (ARC-Challenge −1.7, OpenBookQA −4.1) |
| school vs E | better 2 (MMLU +1.2, OpenBookQA +3.0), tie 1 (ARC-Easy −1.0), worse 2 (CommonsenseQA −1.4, ARC-Challenge −1.7) |

### What it shows

* **General reading gave general knowledge, a little.** The school brain, which read
  no science corpus, does as well as E on grade-school science (ARC-Easy 45.5 vs 46.5)
  and is the best of the three on high-school subjects (MMLU 27.9), but MMLU is still
  barely above chance: most of its questions need reasoning or calculation, not one
  remembered sentence.
* **Everyday common sense** (CommonsenseQA) is clearly above chance (25.5 vs 20) but not
  better than E: the books say a lot about things, little about how people use them.
* **Finding is now the bottleneck, not knowing.** On ARC-Easy the school brain's open
  book alone is better than E's (47.3 vs 45.9) but the joint part falls (33.8 vs 42.1):
  with 7× more facts, the 200 candidate facts taken per option are crowded by
  common ones. Ranking candidates better is the next lever for this brain.
* The tests that reward reasoning (ARC-Challenge, OpenBookQA) stay near chance, as for
  every brain so far.

Results: `results/school_tests.txt`.
