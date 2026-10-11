# Meaning data

- `meaning.mng`: Mimir's word meanings, small profile (256 numbers a word,
  29,702 words), learned by `sandbox/meaning.c` from the books in
  `data/reader/books/` plus 30M words of Simple English Wikipedia
  (`tools/get_simplewiki.py`). Words are stored most common first.
  Rebuild or learn a larger profile with `meaning learn <numbers> <min count> <files...>`.
- `wordsim353.csv`: WordSim-353 (Finkelstein et al., 2002), human ratings of
  how related word pairs are. From gabrilovich.com/resources/data/wordsim353.
- `simlex999.txt`: SimLex-999 (Hill, Reichart and Korhonen, 2015), human
  ratings of how alike word pairs are. From fh295.github.io/simlex.html.

Both test sets are used only to score Mimir, never to learn from.
