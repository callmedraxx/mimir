Dev sets: develop on these, never on `data/arc_tests/`.

| file | from |
|---|---|
| easy_dev.tsv, challenge_dev.tsv | ARC (Clark et al. 2018, CC BY-SA 4.0), dev split |
| obqa_dev.tsv | OpenBookQA (Mihaylov et al. 2018, Apache 2.0), dev split |
| mmlu_hs_dev.tsv | MMLU (Hendrycks et al. 2021, MIT), validation split of every high_school_* subject |
| csqa_traindev.tsv | CommonsenseQA (Talmor et al. 2019, MIT), 1,000 of its training questions (shuffled, seed 12345) |

Built by `tools/general_tests.py` (the last two); layout as in `data/arc_tests/`.
