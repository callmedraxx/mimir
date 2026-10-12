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

## Results

(not yet run)
