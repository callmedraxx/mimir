"""
reader_data.py — build the 12-year-old reading set for mimir's reader.

Math first, code later: this script prepares and checks the data before
any C is written.

  1. Vocabulary: the N most frequent everyday English words (subtitle
     frequencies, data/reader/en_50k.txt) that have a pronunciation in
     CMUdict (data/reader/cmudict.dict). N=25,000 ≈ a 12-year-old's
     reading vocabulary.
  2. Split: deterministic 90/10 train/test by word hash. The test words
     are never shown during training — scoring on them measures READING
     (sounding out a new word), not memorising.
  3. Alignment: each letter is paired with the sound it makes, via
     Viterbi EM. A letter maps to one phoneme, nothing (silent "e"),
     or two phonemes ("x" -> K S).
     Output: data/reader/aligned.tsv  word \t split \t phonemes \t per-letter labels
"""
import collections, math, os, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data", "reader")
N_WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 25000


def load_cmu():
    pron = {}
    for line in open(os.path.join(D, "cmudict.dict"), encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        w, *ph = line.split()
        if "(" in w:          # alternate pronunciations: keep the first
            continue
        pron[w] = ph
    return pron


def load_vocab(pron):
    words = []
    for line in open(os.path.join(D, "en_50k.txt"), encoding="utf-8"):
        w = line.split()[0]
        if w.isascii() and w.isalpha() and w in pron and (len(w) > 1 or w in ("a", "i")):
            words.append(w)
            if len(words) == N_WORDS:
                break
    return words


def split_of(w):
    return "test" if zlib.crc32(w.encode()) % 10 == 0 else "train"


# ── Alignment ────────────────────────────────────────────────────────
# chunk = tuple of 0, 1 or 2 phonemes emitted by one letter.
def viterbi(word, ph, logp):
    n, m = len(word), len(ph)
    INF = -1e18
    best = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n):
        for j in range(m + 1):
            if best[i][j] == INF:
                continue
            for k in (0, 1, 2):
                if j + k > m:
                    continue
                chunk = tuple(ph[j:j + k])
                s = best[i][j] + logp(word[i], chunk)
                if s > best[i + 1][j + k]:
                    best[i + 1][j + k] = s
                    back[i + 1][j + k] = (j, chunk)
    if best[n][m] == INF:
        return None
    out, j = [], m
    for i in range(n, 0, -1):
        pj, chunk = back[i][j]
        out.append(chunk)
        j = pj
    return out[::-1]


def align(pairs, rounds=6):
    counts = None
    for r in range(rounds):
        if counts is None:
            # round 0: prefer 1 phoneme per letter, penalise 0 and 2
            def logp(c, ch):
                return {0: -2.0, 1: 0.0, 2: -3.0}[len(ch)]
        else:
            tot = {c: sum(v.values()) for c, v in counts.items()}
            def logp(c, ch, counts=counts, tot=tot):
                return math.log((counts[c].get(ch, 0) + 0.01) / (tot.get(c, 0) + 1.0))
        new = collections.defaultdict(collections.Counter)
        res, fails = {}, 0
        for w, ph in pairs:
            a = viterbi(w, ph, logp)
            if a is None:
                fails += 1
                continue
            res[w] = a
            for c, ch in zip(w, a):
                new[c][ch] += 1
        counts = new
        print(f"  EM round {r}: aligned {len(res)}, failed {fails}")
    return res, counts


def main():
    pron = load_cmu()
    words = load_vocab(pron)
    print(f"vocab: {len(words)} words (last: {words[-1]!r})")
    pairs = [(w, pron[w]) for w in words]
    res, counts = align(pairs)

    labels = collections.Counter()
    with open(os.path.join(D, "aligned.tsv"), "w") as f:
        for w in words:
            if w not in res:
                continue
            lab = ["_".join(ch) if ch else "-" for ch in res[w]]
            labels.update(lab)
            f.write(f"{w}\t{split_of(w)}\t{' '.join(pron[w])}\t{' '.join(lab)}\n")
    sp = collections.Counter(split_of(w) for w in res)
    print(f"split: {dict(sp)}")
    print(f"output classes: {len(labels)}")
    for c in "aceghtxy":
        print(f"  '{c}' ->", ", ".join(f"{'_'.join(k) or '-'}:{v}" for k, v in counts[c].most_common(6)))
    for w in ["phone", "knight", "box", "laughed", "through", "cat", "school", "would"]:
        if w in res:
            print(f"  {w:9s}", " ".join(f"{c}={'_'.join(ch) or '-'}" for c, ch in zip(w, res[w])))


if __name__ == "__main__":
    main()
