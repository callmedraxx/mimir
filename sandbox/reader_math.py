"""
reader_math.py — which hidden scheme lets mimir sound out words it has
never seen?  Math first, code later.

Task (NETtalk 1987): for each letter of a word, look at a window of
letters around it and predict the sound(s) that letter makes
(data/reader/aligned.tsv from reader_data.py).  Scored on held-out
TEST words that are never trained on.

Candidates:
  A  HDC frozen random ±1 hidden + trained softmax output (mimir today)
  B  linear softmax on the one-hot window (no hidden layer)
  C  HDC binding: hashed letter n-grams around the centre letter
     (bind = position ⊗ letters) + trained softmax output.  Output layer
     only, no backprop — fits mimir's mission and is a table lookup in C.
  D  NETtalk MLP trained with backprop (reference)

Usage: python3 sandbox/reader_math.py [A B C D]
"""
import os, sys, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data", "reader")
W = 4                       # window: 4 letters each side (9 total)
ALPHA = "#abcdefghijklmnopqrstuvwxyz"
A2I = {c: i for i, c in enumerate(ALPHA)}
rng = np.random.default_rng(0)


def load():
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(D, "aligned.tsv"))]
    classes = sorted({lab for r in rows for lab in r[3].split()})
    c2i = {c: i for i, c in enumerate(classes)}
    data = {"train": [], "test": []}
    for w, sp, ph, lab in rows:
        data[sp].append((w, [c2i[x] for x in lab.split()]))
    return data, classes


def windows(words):
    X, Y, wid = [], [], []
    for k, (w, lab) in enumerate(words):
        p = "#" * W + w + "#" * W
        for i, y in enumerate(lab):
            X.append([A2I[c] for c in p[i:i + 2 * W + 1]])
            Y.append(y)
            wid.append(k)
    return np.array(X, np.int32), np.array(Y, np.int32), np.array(wid, np.int32)


# ── feature maps: each returns (B, F) int indices into a feature table ──
NB = 1 << 18   # hash buckets for C


def onehot_idx(X):                       # B: position ⊗ letter
    return X + 27 * np.arange(X.shape[1])[None, :]


def ngram_idx(words_win):
    """C: every substring of the window that touches/contains the centre,
    bound with its offset.  ~ (2W+1) + ... features per letter."""
    out = []
    c = W
    for row in words_win:
        s = "".join(ALPHA[i] for i in row)
        feats = []
        for a in range(0, c + 1):
            for b in range(c + 1, 2 * W + 2):
                if b - a <= 6 or (a == c and b - a <= 4):
                    feats.append(zlib.crc32(f"{a - c}:{s[a:b]}".encode()) % NB)
        for a in range(2 * W + 1):            # single letters with position
            feats.append(zlib.crc32(f"u{a - c}:{s[a]}".encode()) % NB)
        out.append(feats)
    L = max(map(len, out))
    arr = np.full((len(out), L), NB, np.int64)   # NB = padding row (stays 0)
    for i, f in enumerate(out):
        arr[i, :len(f)] = f
    return arr


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train_sparse(Ftr, Ytr, nfeat, ncls, epochs=8, lr=0.5, bs=256):
    """Softmax readout over sparse binary features (rows of a table).
    Update is local: only the active feature rows change (delta rule)."""
    Wt = np.zeros((nfeat + 1, ncls), np.float32)
    n = len(Ytr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            F = Ftr[idx]
            P = softmax(Wt[F].sum(1))
            P[np.arange(len(idx)), Ytr[idx]] -= 1.0
            np.add.at(Wt, F, (-lr / len(idx) * P)[:, None, :] * 1.0)
            Wt[nfeat] = 0
        lr *= 0.7
    return Wt


def predict_sparse(Wt, F):
    out = []
    for s in range(0, len(F), 4096):
        out.append(Wt[F[s:s + 4096]].sum(1).argmax(1))
    return np.concatenate(out)


def hdc_hidden(Xidx, R, theta):
    # R: (27*(2W+1), H) ±1 ; input one-hot → z = sum of rows ; h = z >= theta
    # built in chunks and stored as uint8 so it fits a 3 GB machine
    out = np.empty((len(Xidx), R.shape[1]), np.uint8)
    for s in range(0, len(Xidx), 2048):
        out[s:s + 2048] = R[Xidx[s:s + 2048]].sum(1, dtype=np.int16) >= theta
    return out


def train_dense(Htr, Ytr, ncls, epochs=8, lr=0.1, bs=256):
    Wt = np.zeros((Htr.shape[1] + 1, ncls), np.float32)
    n = len(Ytr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            H = np.hstack([Htr[idx].astype(np.float32), np.ones((len(idx), 1), np.float32)])
            P = softmax(H @ Wt)
            P[np.arange(len(idx)), Ytr[idx]] -= 1.0
            Wt -= lr / len(idx) * (H.T @ P)
        lr *= 0.7
    return Wt


def train_mlp(Xtr, Ytr, ncls, H=512, epochs=12, lr=0.05, bs=128):
    nin = 27 * (2 * W + 1)
    W1 = rng.normal(0, 0.1, (nin, H)).astype(np.float32); b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, 1 / np.sqrt(H), (H, ncls)).astype(np.float32); b2 = np.zeros(ncls, np.float32)
    n = len(Ytr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            Xi = onehot_idx(Xtr[idx])
            z1 = W1[Xi].sum(1) + b1
            h = np.maximum(z1, 0)
            P = softmax(h @ W2 + b2)
            P[np.arange(len(idx)), Ytr[idx]] -= 1.0
            P /= len(idx)
            dh = (P @ W2.T) * (z1 > 0)
            W2 -= lr * h.T @ P; b2 -= lr * P.sum(0)
            np.add.at(W1, Xi, -lr * dh[:, None, :]); b1 -= lr * dh.sum(0)
        lr *= 0.85
    def pred(X):
        out = []
        for s in range(0, len(X), 4096):
            Xi = onehot_idx(X[s:s + 4096])
            out.append((np.maximum(W1[Xi].sum(1) + b1, 0) @ W2 + b2).argmax(1))
        return np.concatenate(out)
    return pred


def score(name, pred, Y, wid, classes, t):
    strip = np.array([c.translate(str.maketrans("", "", "012")) for c in classes])
    let = (pred == Y).mean()
    let_ns = (strip[pred] == strip[Y]).mean()
    nw = wid.max() + 1
    wrong = np.bincount(wid, weights=(pred != Y), minlength=nw)
    wrong_ns = np.bincount(wid, weights=(strip[pred] != strip[Y]), minlength=nw)
    print(f"{name:44s} letters {let*100:5.1f}%  words {np.mean(wrong == 0)*100:5.1f}%  "
          f"| no-stress: letters {let_ns*100:5.1f}%  words {np.mean(wrong_ns == 0)*100:5.1f}%  ({t:.0f}s)")


def main():
    which = sys.argv[1:] or ["A", "B", "C", "D"]
    data, classes = load()
    K = len(classes)
    Xtr, Ytr, _ = windows(data["train"])
    Xte, Yte, wte = windows(data["test"])
    print(f"train letters {len(Ytr)}, test letters {len(Yte)} ({len(data['test'])} unseen words), classes {K}, window ±{W}\n")
    print("Scored on TEST words only (never trained on).")

    if "B" in which:
        t = time.time()
        Wt = train_sparse(onehot_idx(Xtr), Ytr, 27 * (2 * W + 1), K)
        score("B  linear one-hot", predict_sparse(Wt, onehot_idx(Xte)), Yte, wte, classes, time.time() - t)

    if "A" in which:
        for H, theta in [(2048, 1), (2048, 3)]:
            t = time.time()
            R = rng.choice([-1, 1], size=(27 * (2 * W + 1), H)).astype(np.int8)
            Htr = hdc_hidden(onehot_idx(Xtr), R, theta)
            Hte = hdc_hidden(onehot_idx(Xte), R, theta)
            Wt = train_dense(Htr, Ytr, K)
            pred = (Hte.astype(np.float32) @ Wt[:-1] + Wt[-1]).argmax(1)
            score(f"A  HDC random ±1, H={H}, fire if z>={theta}", pred, Yte, wte, classes, time.time() - t)
            del Htr, Hte

    if "C" in which:
        t = time.time()
        Ftr, Fte = ngram_idx(Xtr), ngram_idx(Xte)
        print(f"   C uses {Ftr.shape[1]} bound features per letter, {NB} buckets")
        Wt = train_sparse(Ftr, Ytr, NB, K, epochs=10, lr=1.0)
        score("C  HDC binding (hashed n-grams) + readout", predict_sparse(Wt, Fte), Yte, wte, classes, time.time() - t)
        np.save(os.path.join(D, "proto_C_weights.npy"), Wt)

    if "D" in which:
        t = time.time()
        pred = train_mlp(Xtr, Ytr, K)
        score("D  NETtalk MLP, backprop, H=512", pred(Xte), Yte, wte, classes, time.time() - t)


if __name__ == "__main__":
    main()
