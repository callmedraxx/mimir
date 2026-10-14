# Context window: today's vs one built from lean4 neurons

Owner (2026-10-12): "let's build a context window with these neurons and test
against the current context window". This page fixes the rules **before any
code is written**, as for the neuron comparison (NEURON_COMPARE.md). The
results go below the line at the end, whatever they are.

## The two

| | A: today's context window | B: neuron context window |
|---|---|---|
| holds | every token in order (12 B each) and the page (every byte taken in), `src/text.c` | lean4 event neurons (`sandbox/neurons`): an event's content words and the event after it; word neurons with cue lists. Tokens and the page are **not** kept: text streams through |
| gist | when asked, cuts the whole context into events and runs construction-integration over all of them (`src/gist.c`) | runs **while reading**, on the neurons: firing parts = the workspace of 4; cued recall through the word neurons' cue lists; a meaning part grows on an event while it is in mind or a candidate and shrinks after; a strength part grows on events that gather strength; the strongest 64 keep a gist part when reading stops |
| constants | workspace 4, recall 4, last 8 events per cue word, horizon 400 | the same numbers; gist part 64 events. Nothing tuned on the test sets |

Shared and identical: text intake (`text_take`: folding, numbers said as
words, tokens; both run without the reader's first reading of new words,
which costs the same for both), the event cuts (gist.c's rules: `. ! ? ; :`
and ", and/but/so/then/when/while", at least 3 words), what counts as a
content word (not one of the 150 commonest), and the word meanings
(data/meaning, 256 numbers a word). B keeps at most 12 content words of an
event (lean4's limit); A keeps all.

## Tests (the same code scores both)

| cell | test | score |
|---|---|---|
| G1 | **Gist.** 32 Grimm tales (data/gist), each retold in **60 content words**, scored against its Wikipedia plot summary | content-word F1, mean over tales. A retells by quoting the page, B by saying its events' words; the budget counts content words for both so neither gets more words. A's old score (120 words of page, 13.3) is shown for reference only |
| R1 | **Recall.** War and Peace, 2,000 events: given half of an event's content words, give the event back | F1 of content words against the event (all its content words) |
| F1 | **Find.** War and Peace, 500 content words (250 in ≤20 events, 250 in more): every event the word is in | share of the word's events returned, mean |
| Q | **Quote.** Give back an event exactly as written | share of 2,000 events given back byte for byte |
| M1 | **Memory held** after taking in War and Peace | bytes, not counting the shared word strings and meanings |
| M2 | **Memory peak** while taking it in and giving its gist | bytes, same exclusions |
| S1 | **Speed, little core**: take in War and Peace and give its gist, S22 Ultra | tokens/s, worst of 3 |
| S2 | **Speed, prime core**, same | tokens/s, worst of 3 |

Gate: ≥ 10,000 tokens/s on the little core, or that design fails S1.
If the phone cannot be reached, speeds are measured on the droplet and
marked so.

Also shown, not scored: who each says War and Peace is about; the parts'
bytes at rest; B at a 0.5 MB budget.

## Who wins

A cell is won by more than 1 point (scores) or 10% (bytes, speed);
otherwise it is a tie. The result is the count of cells, given as it is,
with the reasons.

---

## Results (2026-10-12)

Harness: `sandbox/neurons/ctxwin.c` (A is today's text.c + gist.c, included
unchanged; B streams the same text through the same intake). Checks: both
cut War and Peace into the same 41,045 events; A's old gist score comes out
at 13.3% again (120 words of page), as before.

One change made after the first run, for **both**: a word's meaning norm is
worked out once instead of on every use (identical numbers, faster). B also
got a word → meaning map (4 B a word, counted in B's bytes). Scores were
the same before and after.

**The phone was not reachable** (adb "offline"; wireless debugging needs
re-pairing), so speed is the droplet, one core, worst of 3. The phone binary
is built and the run is ready (`ctxwin-android`).

| cell | test | A: today's | B: neurons | winner |
|---|---|---|---|---|
| G1 | gist, 32 tales, F1 at 60 content words | 13.5% | 14.0% | tie (0.4 pt) |
| R1 | recall from half an event's words, War and Peace | 97.4% | 92.4% | **A** |
| F1 | find every event a word is in | 100% | 90.4% | **A** |
| Q | quote an event exactly | 100% | 0% | **A** |
| M1 | memory held after War and Peace | 11.3 MB (19.1 MB as allocated) | 1.13 MB | **B** (10×) |
| M2 | memory peak, taking it in + gist | 58.6 MB | 1.93 MB | **B** (30×) |
| S | speed, take in + gist (droplet, one core, worst of 3) | 64.9k tok/s | 53.0k tok/s | **A** (B 18% slower) |
| S1/S2 | phone little / prime core | — | — | not run (phone offline) |

**Count: A 4, B 2, 1 tie, phone speed pending.** Both are far over the
10k gate on the droplet; the phone has to confirm it on the little core.

### Why, as far as the numbers show

* **Recall and find (A):** 10.3% of War and Peace's events have more than
  12 content words, and B keeps only the first 12 (lean4's limit). 7.3% of
  all content words are past that limit, so B cannot find them or give them
  back. Find on rare words (≤20 events) is 89.2% and common words 91.5%:
  the limit, not the 512-event cue cap, is the cost here.
* **Quote (A):** B keeps no page and no glue words; it gives back the event's
  content words, never the sentence as written.
* **Memory (B):** A keeps every token (12 B) and every byte of the page, and
  its gist builds a float meaning for every event of the book at once
  (41,045 × 1 KB) plus a word index. B holds 38,924 event neurons (≈21 B
  each) and its gist works on what is in mind: at the last page its parts
  were 725 KB (mostly meaning parts on recalled candidates), at rest 2 KB
  (the 64 gist events).
* **Speed (A):** B does not keep a meaning per event, so every older event
  that comes back as a candidate has its meaning rebuilt from its words:
  894,386 times against A's 41,045. That is B's cost for holding 10× less.
* **Gist (tie):** B merges repeated events into one neuron (a refrain such
  as "fisherman went home to his wife" is one event that gathers strength
  every time), which helps on tales built on repetition (Fisherman and his
  Wife 25.6% vs 12.8%) and hurts on some others. Over 32 tales the
  difference is under a point.

Not scored: a question costs A a scan of its whole context (recall 4–10 ms,
find 1–3 ms in War and Peace) and B one cue list (recall ~0.04 ms, find
~0.001 ms). Who War and Peace is about: A "prince andrew princess man pierre
mary", B "prince andrew oh chapter eighteen five" (B counts only its 64 gist
events, and chapter headings got in). Neither gist of War and Peace is good;
there is no human summary to score it against.

B at 0.5 MB (not scored): 19,949 events kept, recall 67.1%, find 49.4%;
peak 1.53 MB (the budget covers the neurons; the meaning parts, intake
buffers and word map are on top).

Results: `results/ctx_*.txt`.

## B grown: no fixed limits (owner: "the whole idea is grow when needed so shouldn't limit to 12")

Same tests, same harness, nothing else changed. Two limits lean4 had are
gone:

* **Words past 12.** An event with more than 12 content words grows an
  **extra-words part** (6 B: where its words are in an exp-Golomb pool, how
  many) and every one of its words gets its cue link. Shorter events grow
  nothing. In War and Peace 4,219 events (10.3%) grew one: 119 KB.
* **512 links per word.** With words past 12 fixed, recall was still 3.6
  points behind A; all 124 events A got right and B got wrong had been
  dropped from a common word's 512 links ("looked", "me"). Cue links now
  grow as needed; only the memory budget prunes (at night, least used and
  oldest first, as before). Cost: 12 KB.

With `LIMIT12=1` the harness gives the old B exactly (recall 92.36%, find
90.37%, same bytes), so the change is only the growing.

| cell | A: today's | B grown | B before | winner now |
|---|---|---|---|---|
| G1 gist (32 tales) | 13.5% | 14.0% (14.05) | 14.0% (13.95) | tie |
| R1 recall | 97.42% | **97.42%** | 92.4% | tie |
| F1 find | 100% | **100%** | 90.4% | tie |
| Q quote | 100% | 0% | 0% | A |
| M1 memory held | 11.3 MB | 1.29 MB | 1.13 MB | **B** (8.7×) |
| M2 memory peak | 58.6 MB | 2.25 MB | 1.93 MB | **B** (26×) |
| S speed, droplet one core, worst of 3 | 56.4k tok/s | 58.2k tok/s | 48.0k tok/s | tie |

**Count: B 2, A 1, 4 ties** (phone speed still pending). B now gives back
and finds everything A does; the one thing it cannot do is quote the
sentence as written, because it keeps no page.

Speed honestly: the droplet was noisy this time (A ranged 56k–124k). Worst
of 3 is a tie, but run by run B was 1.03×, 0.84× and 0.73× of A in the same
run, so B is probably still somewhat slower; the phone will settle it.
Growing made B *faster* than the capped B (cut lists no longer get
rewritten every time they pass the cap).

Under a tight budget growing costs: at 0.5 MB the grown B keeps fewer events
(the extra words take room), recall 63.1% and find 47.1% (capped: 67.1%,
49.4%). The owner decides what memory holds; a long event costs what it is.

## Phone (S22 Ultra, War and Peace, worst of 3)

The phone came back; same binary for A and both Bs, each run takes in War
and Peace and gives its gist.

| core | A: today's | B grown | B capped | winner (A vs B grown) |
|---|---|---|---|---|
| **little** (S1) | 15.9k tok/s | **21.3k** | 21.6k | **B** (+34%) |
| **prime** (S2) | **137.2k** | 95.3k | 105.8k | **A** (B −31%) |

Run by run (same run, B grown ÷ A): little 1.18, 1.10, 1.13; prime 0.63,
0.69, 0.69. Both pass the 10k gate on the little core.

Why it flips: on the prime core A's work is plain float maths and the core
is fast at it, so B's extra work (rebuilding meanings of recalled events)
shows. On the little core memory is the cost: A walks 58 MB (every token,
the page, a 1 KB float meaning for every one of 41,045 events); B works in
about 2 MB, which stays in the caches.

**Final count with the phone (grown B): B 3 (memory held, memory peak,
little-core speed), A 2 (quote, prime-core speed), 3 ties (gist, recall,
find).** Results: `results/ctx_phone.txt`.

## Correction (2026-10-12): workspace bug

Found while testing understanding (UNDERSTAND_COMPARE.md): when an event
already carried in B's workspace was read again and was not carried on, it
was never let go, so the workspace grew past 4 and wrote past the end of an
array (AddressSanitizer). Fixed in `cw_event`. Every score and byte count
above is unchanged (gist 14.05%, recall 97.42%, find 100%, 1.29 MB held,
2.25 MB peak). Speed is not: the extra carried events kept their meaning
parts, which saved rebuilds. Phone again, worst of 3
(`results/understand_phone_final.txt`, B = neurons):

| core | A: today's | B fixed | before the fix | winner |
|---|---|---|---|---|
| little | 19.6k | **23.3k** (+19%) | 21.3k vs 15.9k (+34%) | **B** |
| prime | **123.1k** | 94.7k (−23%) | 95.3k vs 137.2k (−31%) | **A** |

Same winners, so the count stays **B 3, A 2, 3 ties**; B's lead on the little
core is 19%, not 34%.
