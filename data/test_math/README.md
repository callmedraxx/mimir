# A small maths test (2026-10-12)

Written for the owner's last test of the day: a short maths text (`text.txt`) and 10
four-option questions about it (`questions.json`, answer key included), committed
before Mimir or Laya saw them. 8 questions are answered in the text, 2 ask to apply
an idea from it. Neither system is trained on it: each reads the text once and answers.

- Mimir: `understand mctest data/test_math/mctest_format.txt` (the story reader of
  UNDERSTAND_COMPARE.md, frozen: neurons growing roles, role weight 2).
- Laya: `tools/laya_math.py` in Colab, all three checkpoints as shipped.
Chance is 2.5 of 10.

## Result (2026-10-12)

| | score | answers (key B C A A D B A B C C) |
|---|---|---|
| Laya (as shipped) | **6 / 10** | B D A D D C A B C D |
| Laya typed-decisions | 6 / 10 | B D A D D C A B C D |
| Laya multilingual | 5 / 10 | C C A A D D A D A A |
| Mimir (neurons growing roles) | 3 / 10 | C B C D D B A D A A |
| chance | 2.5 / 10 | |

Both got: Euclid (Q5), the name of the theorem (Q7). Only Laya: the only even
prime (Q1), the least common multiple of 4 and 6 (Q3), how many factors a prime has
(Q8), which number is divisible by 3 (Q9, an applied question). Only Mimir: the
divisible-by-5 rule (Q6). Neither: the greatest common divisor (Q2), 1 is neither
prime nor composite (Q4), which number is prime (Q10).
