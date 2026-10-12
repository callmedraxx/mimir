# Mimir roadmap

**End goal.** A brain that runs on a phone: it reads anything its owner gives it,
keeps it, understands it, and decides and answers from it — on the phone's own
CPU, no cloud, built like a brain (neurons that grow parts when needed; no LLM,
no backpropagation). Each owner decides what its memory holds; a bigger device
holds more; it stays fast everywhere (≥ 10k tokens/s on the S22 Ultra's little
core). Text generation comes later.

**How we work.** Every step has its rules written and committed before its code
(docs/*_COMPARE.md), is measured against something real, and reports worst
cases. A step is done when its result is committed, whatever it is.

**What ships.** Like an app: one program, a small set of base files (word
meanings, word classes), and the owner's memory as one file that keeps growing.

| # | step | status | where |
|---|---|---|---|
| 1 | Read fast on the phone (≥ 10k tok/s little core) | done | reader, NEURON*.md |
| 2 | Neurons: one thing each, grow parts when needed (lean4 + parts) | done | NEURON_COMPARE.md, sandbox/neurons/grow.c |
| 3 | Context window on neurons: hold a whole book, recall and find any part | done — War and Peace in 1.29 MB (today's 11.3 MB), recall 97.4%, find 100% | CONTEXT_COMPARE.md |
| 4 | Roles (who did what to whom, where): first understanding | done — bAbI 94% (17/20 tasks); ordinary prose not understood yet | UNDERSTAND_COMPARE.md |
| 5 | **Decisions like Jev and Laya**: read a case, answer yes/no, pick-one and 0–3 questions as probabilities (Typed Decisions benchmark) | **in progress** | DECIDE_COMPARE.md |
| 6 | Runnable package: `mimir read / ask / decide / save / load`, memory as a `.brain` file; Linux and Android builds people can download | next | — |
| 7 | Android app (APK) around the same engine | after 6 | — |
| 8 | Reading ordinary prose well (a better role reader; MCTest, longer documents) | after 5 | — |
| 9 | Long-document questions against small open LLMs on the same phone | after 8 | — |
| 10 | Producing sentences | later | — |

Updated at the end of every step.
