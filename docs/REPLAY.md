# Replay: the brain corrects itself offline (replay)

Owner (2026-10-19): "whenever I play COD sometimes when I close my eyes I see myself still playing and thinking on how to
improve involuntarily. How can that help Mimir to become a self correcting model?"

## What the owner is describing, in the brain

That is **replay**. After a demanding episode the hippocampus re-runs it during rest and sleep, compressed and
sometimes backwards, and the cortex learns from the re-run rather than from the moment itself: rats that run a maze
replay the maze while they sleep, and blocking the replay blocks the learning. Two things ride on it. The re-run is
not a recording: the brain re-reads the episode *with everything it knows now*, so the ambush that made no sense at
the time makes sense in replay ("he was already behind the crate"). And the re-run runs forward past what happened
into what could have happened ("if I had gone left"), and the difference between the two is the error signal that
changes the next game. No teacher marks anything. The signal is prediction error: the episode's own continuation
against what the model would have predicted. That is why it is involuntary and why it improves play: the brain is
grading itself against the only ground truth it has, what actually happened next.

## What Mimir has today

Mimir reads a passage once, forward, and keeps the facts as read. A pronoun resolved wrongly in sentence 2 stays
wrong when sentence 6 settles who was meant; a word classed wrongly ("shoot" as a verb in "shoot-out") stays wrong
after the sentence has made the shape clear; two current states for one slot sit side by side. It never re-reads
with what it learned, and nothing inside it says "that cannot be right". The corrections of the last days (docs/
YESNO.md Y13, Y14, Y17) were all cases a replay would have caught, and I found each by hand.

## The method: three parts of replay, in the order they pay

**R1. Re-read with what is known now (a second pass).** After the first pass over a passage, read it again with the
first pass's knowledge held: who is of which sex, which names are people, what each thing's kind is, which words the
text uses as nouns ("the shoot-out"), which pronoun pointed where. Facts that come out differently replace the first
pass's; facts that come out the same are confirmed (their strength rises, as a re-met fact's does on the shelf). Cost:
reading twice; the episode is small. The brain's compressed replay is exactly this: faster, informed, and it changes
what is stored.

**R2. Check what cannot be true, and go back to the sentence.** A set of *consistency rules*, each a reason to doubt:
one slot with two current values (docs/STATE.md); a thing held by two people; a pronoun with no person before it; a
frame with a verb and no subject; a negated fact and its affirmation both current with the same order; a kind fact
"X is a Y" where Y is an adjective. Each failure names the sentence(s) involved, and those sentences alone are
re-read with the conflict as context (the open-book re-read exists; it gets a reason). What the re-read cannot settle
stays marked *doubtful*, and a doubtful fact answers with less weight and says "I am not sure" in docs/SPEAK.md's
terms (W3). This is the "that cannot be right" the owner feels when the replay shows the mistake.

**R3. Predict the next sentence, and learn from the miss (self-grading).** While reading a story a second time, before
each sentence Mimir states what it expects for the state slots it tracks (where each person is, who holds what,
what each thing is): for "Mary went to the kitchen. Mary picked up the apple. ..." it expects the apple with Mary.
The next sentence either confirms or contradicts. A contradiction with no explicit change in between means a
reading rule fired wrongly somewhere, and the fact and the rule that produced it are penalised; a confirmation
rewards them. The cortex already learns by reward and penalty from each fact (K3b, docs/past/CORTEX.md); R3 gives
it a reward it earns from the text itself, with no answer key, on every passage it ever reads, including the owner's
own writing. This is the part that makes it *self-correcting* rather than corrected by me: the error signal is the
world's next sentence. It is also the mechanism the owner's replay uses: the replayed game runs forward, and where
the run and the memory part ways, something is learned.

Not in this: no backpropagation, no model behind it, nothing that is not the text and the rules. Replay is a brain's
trick for getting more out of one experience, and the whole of it is "read it again, knowing more, and notice where
you were wrong".

## First measurement (2026-10-19, tiny brain, switch `S6=1`: R1 as a plain second pass)

The second pass alone changes nothing on the dev stories (0 facts differ from the first pass on every story), and the
scores stay: dev stories 6 wrong of 70 both, bAbI train choice 83.17 both, bAbI in words 90.33 both, MC500 dev 59.38
both, MC160 dev 52.78 → 53.19 (one question). The one thing it exposed: a second read that lets the cortex learn again
costs 4 points on MC500 dev and 4 story questions (the passage's words doubled), so the replay must be silent in the
cortex (`cx_quiet`), as the brain's replay changes the cortex by its own rule and not as a second day's reading. So
R1 by itself is not where the gain is: what the first pass learns, it mostly already learned forward on these short
texts. The gain has to come from R2 (a reason to doubt, then the sentence re-read) and R3 (the text's own next sentence
as the grade), which is where the owner's replay differs from a mere re-run: it looks for the mistake.

**R2's first four rules (tiny brain, `S6=2`, counted only):** one slot with two current values, a thing held by two
people, a fact with no argument, an argument still a pronoun. They fire on none of the 70 dev-story questions'
passages and on none of bAbI's 4,000 in words: STATE's superseding, the pronoun rules and the frame reader already
leave no such trace, and the 387 wrong answers in words all come from passages the rules call clean. So a doubt must
be about the *answer*, not the store: how far the chosen fact beats the next, whether the slot's value was derived or
said, whether the question's words were all found. The abstentions already there ("-" when no fact fits) are the
first such doubt; their precision is measured next.

## Scorecard

* **Measure first, no change.** On the dev stories and bAbI train (choice and in words): how many facts a second pass
  changes, and whether the changed ones were right (R1); how many consistency failures R2's rules raise per passage,
  and how many of them sit on a question that was answered wrongly (the signal's precision); how many next-sentence
  predictions R3 gets wrong and what reading rule was behind each (the first ten, by hand).
* **Dev**: the 70 dev-story questions, bAbI train (STATE's seven tasks, choice and words), MC160 / MC500 dev, RACE-M
  dev, BoolQ train, the five general dev sets (R1 changes reading, so everything). **Choice**: dev stories not down,
  bAbI train and the passage dev mean not down more than 0.5, no general dev set down more than 1, and at least one of:
  dev stories up, bAbI words up 2, MC dev mean up 0.5. R2's doubt marks must be right more often than wrong on the
  dev stories' misses.
* **Test, once**: the 18 test stories, bAbI test (choice and words), MCTest and RACE-M tests, BoolQ validation, the
  general tests; then the survey rerun.
* Switch `S6` (1 R1, 2 R2, 4 R3), `S6=0` identical to the adopted reader. Colab for every run; the tiny brain for the
  reading parts (R1, R2 are brain-independent). Adoption is the owner's call. Scheduled after the YESNO freeze
  (docs/YESNO.md), as the next reading step; it folds into every later step, since every step reads.
