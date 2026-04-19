"""
vision_math.py — NumPy reproduction of mimir's vision_encode pipeline and
a benchmark of candidate hidden-layer schemes.

Purpose: per user's "math first, code later" rule, this script tests ideas
numerically BEFORE we touch vision.c.  It replicates vision_encode exactly
(LCN -> 8 Gabor orientations -> 4x4 pool -> z-score), then evaluates a
wide menu of hidden-layer proposals including invented ones.

Run: python3 sandbox/vision_math.py
"""
import os, glob, math
import numpy as np

# ------------------------------------------------------------------
# Constants from mimir.h / vision.c
# ------------------------------------------------------------------
IMG_SIZE          = 128
RAW_SIZE          = IMG_SIZE * IMG_SIZE      # 16384
N_ORIENT          = 8
POOL_GRID         = 4                        # 4x4 -> 16 cells
POOL_CELL         = IMG_SIZE // POOL_GRID    # 32
FEATURE_SIZE      = N_ORIENT * POOL_GRID * POOL_GRID   # 128
EMBED_SIZE        = 128
ALPHA_RAW_LETTER  = 26
ALPHA_RAW_QUERY   = 5
ALPHA_RAW_SIZE    = ALPHA_RAW_LETTER + ALPHA_RAW_QUERY   # 31
GABOR_KSIZE       = 11
GABOR_R           = GABOR_KSIZE // 2
LCN_R             = 3
SIGMA, LAMBDA, GAMMA = 2.5, 5.5, 0.5
IMG_DIR           = "/root/mimir/data/images"

rng = np.random.default_rng(42)

# ------------------------------------------------------------------
# Step 1: build Gabor kernels (matches vision_init_gabor)
# ------------------------------------------------------------------
def build_gabor_kernels():
    kernels = np.zeros((N_ORIENT, GABOR_KSIZE, GABOR_KSIZE), np.float32)
    ys, xs = np.mgrid[-GABOR_R:GABOR_R+1, -GABOR_R:GABOR_R+1]
    for o in range(N_ORIENT):
        theta = o * math.pi / N_ORIENT
        ct, st = math.cos(theta), math.sin(theta)
        xp =  xs*ct + ys*st
        yp = -xs*st + ys*ct
        env = np.exp(-(xp*xp + GAMMA*GAMMA*yp*yp) / (2*SIGMA*SIGMA))
        carrier = np.cos(2*math.pi*xp / LAMBDA)
        k = env * carrier
        k /= math.sqrt((k*k).sum() + 1e-8)
        kernels[o] = k
    return kernels

# ------------------------------------------------------------------
# Step 2: LCN + convolve + pool + zscore (matches vision_encode)
# ------------------------------------------------------------------
def lcn(img):
    # sliding-window mean/std, zero-pad borders
    H, W = img.shape
    out = np.zeros_like(img)
    pad = LCN_R
    P = np.zeros((H+2*pad, W+2*pad), np.float32)
    P[pad:pad+H, pad:pad+W] = img
    # build valid-mask for correct count at edges
    M = np.zeros_like(P); M[pad:pad+H, pad:pad+W] = 1.0
    # integral images for fast local sum
    def integral(a):
        ii = np.pad(a, ((1,0),(1,0))).cumsum(0).cumsum(1)
        return ii
    IIv  = integral(P)
    IIv2 = integral(P*P)
    IIm  = integral(M)
    W2 = 2*pad+1
    for y in range(H):
        for x in range(W):
            y0, x0 = y, x
            y1, x1 = y + W2, x + W2
            s   = IIv [y1,x1]-IIv [y0,x1]-IIv [y1,x0]+IIv [y0,x0]
            ss  = IIv2[y1,x1]-IIv2[y0,x1]-IIv2[y1,x0]+IIv2[y0,x0]
            cnt = IIm [y1,x1]-IIm [y0,x1]-IIm [y1,x0]+IIm [y0,x0]
            m = s/cnt; v = ss/cnt - m*m
            std = math.sqrt(v + 1e-3)
            out[y,x] = (img[y,x] - m) / std
    return out

def conv2d_valid_padded(img, k):
    # 2D convolution with zero-padding (matches vision_conv)
    H, W = img.shape
    R = k.shape[0]//2
    P = np.zeros((H+2*R, W+2*R), np.float32)
    P[R:R+H, R:R+W] = img
    # simple strided windows via stride_tricks
    from numpy.lib.stride_tricks import sliding_window_view
    W_ = sliding_window_view(P, k.shape)  # (H, W, K, K)
    return (W_ * k).sum(axis=(-1,-2)).astype(np.float32)

def vision_encode(raw, kernels):
    img = raw.reshape(IMG_SIZE, IMG_SIZE).astype(np.float32)
    n   = lcn(img)
    emb = np.zeros(EMBED_SIZE, np.float32)
    for o in range(N_ORIENT):
        resp = np.abs(conv2d_valid_padded(n, kernels[o]))
        # 4x4 average pool
        pooled = resp.reshape(POOL_GRID, POOL_CELL, POOL_GRID, POOL_CELL).mean(axis=(1,3))
        emb[o*POOL_GRID*POOL_GRID:(o+1)*POOL_GRID*POOL_GRID] = pooled.flatten()
    # z-score
    m = emb.mean(); s = emb.std() + 1e-9
    return (emb - m) / s

# ------------------------------------------------------------------
# Step 3: load 26 images
# ------------------------------------------------------------------
def load_images():
    # index.txt maps letter -> word, but mimir names files <word>.raw
    # The alphabet ordering we want: a..z with the default vocab.
    # For sim purposes, use whatever 26 .raw files exist, sorted.
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.raw")))
    imgs, names = [], []
    for p in paths:
        data = np.fromfile(p, dtype=np.float32, count=RAW_SIZE)
        if data.size != RAW_SIZE: continue
        imgs.append(data)
        names.append(os.path.basename(p).replace(".raw",""))
    return imgs, names

# ------------------------------------------------------------------
# Main: build embeddings
# ------------------------------------------------------------------
print("Building Gabor bank...")
kernels = build_gabor_kernels()
print("Loading images...")
raws, names = load_images()
N = len(raws)
print(f"  loaded {N} images: {names}")

print("Encoding (Gabor pipeline)...")
E_full = np.stack([vision_encode(r, kernels) for r in raws])
# Zero out text dims [0..ALPHA_RAW_SIZE) as vision_train does
E = E_full.copy()
E[:, :ALPHA_RAW_SIZE] = 0.0
print(f"  E shape {E.shape}, effective dims = {EMBED_SIZE - ALPHA_RAW_SIZE}")

# Save for reuse
np.savez("/root/mimir/sandbox/embeddings.npz", E=E, E_full=E_full, names=np.array(names))
print("Saved embeddings to sandbox/embeddings.npz")

# ------------------------------------------------------------------
# Diagnostic 1: pairwise cosine similarity
# ------------------------------------------------------------------
def cosmat(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return Xn @ Xn.T

C = cosmat(E)
off = C[~np.eye(N, dtype=bool)]
print(f"\n[Diagnostic 1] Pairwise cosine similarity (visual-only dims):")
print(f"  mean={off.mean():.3f}  median={np.median(off):.3f}  "
      f"max(off-diag)={off.max():.3f}  min={off.min():.3f}")
# rank
rk = np.linalg.matrix_rank(E)
print(f"  matrix rank of {N}x{EMBED_SIZE} embeddings = {rk}")

# ------------------------------------------------------------------
# Diagnostic 2: theoretical ceiling — linear classifier on raw embedding
# (i.e., what's the best possible with output-layer weights if hidden
# layer passed the full 97-dim signal through linearly)
# ------------------------------------------------------------------
def one_hot(n):
    return np.eye(n, dtype=np.float32)

def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z); return e / e.sum(axis=1, keepdims=True)

def linear_probe_accuracy(H, Y):
    """Least-squares fit W s.t. H @ W ~= Y; return argmax accuracy."""
    W, *_ = np.linalg.lstsq(H, Y, rcond=None)
    P = H @ W
    return (P.argmax(axis=1) == Y.argmax(axis=1)).mean()

Y = one_hot(N)  # targets: each image -> its own class
acc_linear_full = linear_probe_accuracy(E, Y)
print(f"\n[Diagnostic 2] Linear probe directly on {E.shape[1]}-dim embedding: "
      f"{acc_linear_full*100:.1f}%  (upper bound if hidden = identity)")

# ------------------------------------------------------------------
# Diagnostic 3: why 8 random hidden fails
# ------------------------------------------------------------------
def hidden_sigmoid(E, W, b):
    z = E @ W + b
    return 1.0 / (1.0 + np.exp(-z))

def run_sigmoid_hidden(E, W_hid, b_hid, n_epochs=3000, lr=0.1):
    """Delta-rule output layer training with sigmoid-activated hidden."""
    H = hidden_sigmoid(E, W_hid, b_hid)   # (N, K)
    K = H.shape[1]
    # Output layer: 26 neurons, sigmoid, delta rule on target one-hot
    W_out = rng.normal(0, 1/math.sqrt(K), (K, N)).astype(np.float32)
    b_out = np.zeros(N, np.float32)
    Y = one_hot(N)
    for ep in range(n_epochs):
        Z = H @ W_out + b_out
        O = 1.0 / (1.0 + np.exp(-Z))
        err = Y - O
        # delta rule (sigmoid * derivative skipped — matches mimir code)
        W_out += lr * H.T @ err / N
        b_out += lr * err.mean(axis=0)
    # eval
    pred = O.argmax(axis=1); tgt = Y.argmax(axis=1)
    return (pred == tgt).mean(), O

# Baseline: 8 Xavier random hidden neurons, bias = -5 (matches current code)
def xavier_hidden(n_hidden, seed=0):
    g = np.random.default_rng(seed)
    # Weights only on visual-region dims [ALPHA_RAW_SIZE..128]
    vis_dims = EMBED_SIZE - ALPHA_RAW_SIZE  # 97
    W = np.zeros((EMBED_SIZE, n_hidden), np.float32)
    scale = 1.0 / math.sqrt(vis_dims)
    W[ALPHA_RAW_SIZE:, :] = g.uniform(-scale, scale, (vis_dims, n_hidden))
    b = np.full(n_hidden, -5.0, np.float32)
    return W, b

def imprint_hidden(E_imprint, bias=-5.0, normalize=True):
    """Each hidden neuron = prototype weights of one training sample."""
    W = np.zeros((EMBED_SIZE, E_imprint.shape[0]), np.float32)
    for i, e in enumerate(E_imprint):
        v = e.copy()
        if normalize:
            v = v / (np.linalg.norm(v) + 1e-9)
        W[:, i] = v
    b = np.full(E_imprint.shape[0], bias, np.float32)
    return W, b

# --------- Scheme A: current (8 Xavier random, bias=-5) -----------
print("\n========= CANDIDATE HIDDEN-LAYER SCHEMES =========\n")
results = []

for K in [8, 16, 32, 64, 128]:
    W, b = xavier_hidden(K, seed=7)
    acc, _ = run_sigmoid_hidden(E, W, b)
    results.append((f"Xavier random K={K}, bias=-5", acc))
    print(f"  Xavier random K={K:3d} bias=-5      -> {acc*100:5.1f}%")

# --------- Scheme B: Hebbian imprint (26, one per class) ----------
W, b = imprint_hidden(E, bias=-5.0)
acc, _ = run_sigmoid_hidden(E, W, b)
results.append(("Imprint K=26, bias=-5", acc))
print(f"  Imprint K=26 bias=-5             -> {acc*100:5.1f}%")

# bias sweep for imprint (since bias matters a lot for sigmoid)
for bb in [-3, -1, 0, 1, 3]:
    W, b = imprint_hidden(E, bias=bb)
    acc, _ = run_sigmoid_hidden(E, W, b)
    results.append((f"Imprint K=26, bias={bb:+d}", acc))
    print(f"  Imprint K=26 bias={bb:+d}              -> {acc*100:5.1f}%")

# --------- Scheme C: Mean-subtracted imprint (prototype sharpening) ----
# Each neuron's weight = E[i] - mean_j!=i E[j].  Amplifies class-specific
# features, suppresses shared ones.
def mean_subtract_imprint(E, bias=-1.0):
    mean_all = E.mean(axis=0)
    W = np.zeros((EMBED_SIZE, E.shape[0]), np.float32)
    for i, e in enumerate(E):
        # mean of others
        others = np.delete(E, i, axis=0).mean(axis=0)
        v = e - others
        v = v / (np.linalg.norm(v) + 1e-9)
        W[:, i] = v
    return W, np.full(E.shape[0], bias, np.float32)

W, b = mean_subtract_imprint(E, bias=-1.0)
acc, _ = run_sigmoid_hidden(E, W, b)
results.append(("Mean-sub imprint K=26, bias=-1", acc))
print(f"  Mean-sub imprint K=26 bias=-1    -> {acc*100:5.1f}%")

# --------- Scheme D: Gram-Schmidt orthogonalized imprint ----------
def gs_imprint(E, bias=-1.0):
    W = np.zeros((EMBED_SIZE, E.shape[0]), np.float32)
    basis = []
    for i, e in enumerate(E):
        v = e.copy()
        for u in basis:
            v = v - (v @ u) * u
        nrm = np.linalg.norm(v)
        if nrm > 1e-6:
            u = v / nrm
            basis.append(u)
            W[:, i] = u
        else:
            W[:, i] = e / (np.linalg.norm(e)+1e-9)
    return W, np.full(E.shape[0], bias, np.float32)

W, b = gs_imprint(E, bias=-1.0)
acc, _ = run_sigmoid_hidden(E, W, b)
results.append(("Gram-Schmidt imprint K=26, bias=-1", acc))
print(f"  Gram-Schmidt imprint K=26 bias=-1 -> {acc*100:5.1f}%")

# --------- Scheme E: Oja's rule on top of random init (unsupervised) ----
def oja_refine(E, K, n_iter=2000, lr=0.01, seed=1):
    g = np.random.default_rng(seed)
    vis_dims = EMBED_SIZE - ALPHA_RAW_SIZE
    W = np.zeros((EMBED_SIZE, K), np.float32)
    scale = 1/math.sqrt(vis_dims)
    W[ALPHA_RAW_SIZE:, :] = g.uniform(-scale, scale, (vis_dims, K))
    for it in range(n_iter):
        i = g.integers(0, E.shape[0])
        x = E[i]
        y = x @ W   # linear output
        # Oja: dW = lr * y * (x - y*W)
        W += lr * np.outer(x, y) - lr * W * (y*y)
    b = np.full(K, 0.0, np.float32)
    return W, b

for K in [26, 64]:
    W, b = oja_refine(E, K)
    acc, _ = run_sigmoid_hidden(E, W, b)
    results.append((f"Oja K={K}, bias=0", acc))
    print(f"  Oja K={K:3d} bias=0              -> {acc*100:5.1f}%")

# --------- Scheme F: BCM on top of random init ---------------------
def bcm_refine(E, K, n_iter=3000, lr=0.01, theta_tau=0.05, seed=2):
    g = np.random.default_rng(seed)
    vis_dims = EMBED_SIZE - ALPHA_RAW_SIZE
    W = np.zeros((EMBED_SIZE, K), np.float32)
    scale = 1/math.sqrt(vis_dims)
    W[ALPHA_RAW_SIZE:, :] = g.uniform(-scale, scale, (vis_dims, K))
    theta = np.ones(K, np.float32) * 0.1
    for it in range(n_iter):
        i = g.integers(0, E.shape[0])
        x = E[i]
        y = np.tanh(x @ W)
        # BCM: dW = lr * y * (y - theta) * x
        W += lr * np.outer(x, y * (y - theta))
        theta = (1 - theta_tau) * theta + theta_tau * (y*y)
    b = np.full(K, 0.0, np.float32)
    return W, b

for K in [26, 64]:
    W, b = bcm_refine(E, K)
    acc, _ = run_sigmoid_hidden(E, W, b)
    results.append((f"BCM K={K}", acc))
    print(f"  BCM K={K:3d}                      -> {acc*100:5.1f}%")

# --------- Scheme G: k-WTA hidden (sparse coding) -----------------
def kwta_hidden(E, W, b, k=4):
    # Normal forward + hard top-k sparsification
    z = E @ W + b
    h = 1.0/(1.0+np.exp(-z))
    # keep top-k per sample, zero rest
    for i in range(h.shape[0]):
        idx = np.argpartition(h[i], -k)[-k:]
        mask = np.zeros_like(h[i]); mask[idx] = 1.0
        h[i] *= mask
    return h

def run_kwta_trained(E, W_hid, b_hid, k=4, n_epochs=3000, lr=0.1):
    H = kwta_hidden(E, W_hid, b_hid, k=k)
    K = H.shape[1]
    W_out = rng.normal(0, 1/math.sqrt(K), (K, N)).astype(np.float32)
    b_out = np.zeros(N, np.float32)
    Y = one_hot(N)
    for ep in range(n_epochs):
        Z = H @ W_out + b_out
        O = 1.0/(1.0+np.exp(-Z))
        err = Y - O
        W_out += lr * H.T @ err / N
        b_out += lr * err.mean(axis=0)
    pred = O.argmax(axis=1)
    return (pred == Y.argmax(axis=1)).mean()

# Imprint + k-WTA (novel: prototypes then force sparsity)
W, b = imprint_hidden(E, bias=-1.0)
for k in [1, 3, 5, 8]:
    acc = run_kwta_trained(E, W, b, k=k)
    results.append((f"Imprint + k-WTA k={k}", acc))
    print(f"  Imprint K=26 + k-WTA k={k}        -> {acc*100:5.1f}%")

# --------- Scheme H (INVENTED): Anti-pair imprint -----------------
# For each class, create TWO hidden neurons: prototype and anti-prototype
# (prototype with sign flipped).  Output layer sees contrast, not absolute.
def anti_pair_imprint(E, bias=0.0):
    W = np.zeros((EMBED_SIZE, 2*E.shape[0]), np.float32)
    for i, e in enumerate(E):
        v = e / (np.linalg.norm(e)+1e-9)
        W[:, 2*i]   = v
        W[:, 2*i+1] = -v
    return W, np.full(2*E.shape[0], bias, np.float32)

W, b = anti_pair_imprint(E, bias=0.0)
acc, _ = run_sigmoid_hidden(E, W, b)
results.append(("Anti-pair imprint K=52 bias=0", acc))
print(f"  (INV) Anti-pair imprint K=52     -> {acc*100:5.1f}%")

# --------- Scheme I (INVENTED): Class-difference hidden ----------
# Hidden neuron = difference between class pair (Fisher-discriminant-like).
# Only use 26 pairs: (i, (i+1)%26) — cheap cyclic pairing.
def class_diff_hidden(E, bias=0.0):
    N = E.shape[0]
    W = np.zeros((EMBED_SIZE, N), np.float32)
    for i in range(N):
        d = E[i] - E[(i+1) % N]
        W[:, i] = d / (np.linalg.norm(d)+1e-9)
    return W, np.full(N, bias, np.float32)

W, b = class_diff_hidden(E, bias=0.0)
acc, _ = run_sigmoid_hidden(E, W, b)
results.append(("Class-diff hidden K=26 bias=0", acc))
print(f"  (INV) Class-diff hidden K=26     -> {acc*100:5.1f}%")

# --------- Scheme J (INVENTED): Random sparse masks (dropout-by-design) ----
# Each hidden neuron = random 20%-dense gabor-region mask.  The mask
# specialization gives combinatorial diversity.
def sparse_mask_hidden(E, K=128, density=0.2, seed=3):
    g = np.random.default_rng(seed)
    vis_dims = EMBED_SIZE - ALPHA_RAW_SIZE
    W = np.zeros((EMBED_SIZE, K), np.float32)
    for k in range(K):
        mask = g.random(vis_dims) < density
        signs = g.choice([-1, 1], size=vis_dims)
        W[ALPHA_RAW_SIZE:, k] = mask * signs / math.sqrt(density*vis_dims)
    return W, np.full(K, 0.0, np.float32)

for K in [26, 64, 128]:
    W, b = sparse_mask_hidden(E, K=K)
    acc, _ = run_sigmoid_hidden(E, W, b)
    results.append((f"Sparse random masks K={K}", acc))
    print(f"  (INV) Sparse masks K={K:3d}         -> {acc*100:5.1f}%")

# --------- Scheme K (INVENTED): "Prototype + margin tune" --------
# Imprint, then analytically set each neuron's bias so it fires ~ONLY
# on its own class.  For sigmoid, we want z(own) > 0 and z(others) < 0
# for all others.  Pick bias = -(max z on others) - eps.
def margin_tuned_imprint(E, eps=0.05):
    W = np.zeros((EMBED_SIZE, E.shape[0]), np.float32)
    for i, e in enumerate(E):
        W[:, i] = e / (np.linalg.norm(e)+1e-9)
    # each neuron's pre-activation on all samples
    Z = E @ W    # (N, N)
    # For neuron i, the "own" response is Z[i,i]; "others" are Z[j!=i, i]
    biases = np.zeros(E.shape[0], np.float32)
    for i in range(E.shape[0]):
        others = np.delete(Z[:, i], i)
        biases[i] = -(others.max() + eps)
    return W, biases

W, b = margin_tuned_imprint(E)
acc, _ = run_sigmoid_hidden(E, W, b)
results.append(("(INV) Margin-tuned imprint K=26", acc))
print(f"  (INV) Margin-tuned imprint K=26  -> {acc*100:5.1f}%")
# also report activation stats
Z = E @ W + b
H = 1.0/(1.0+np.exp(-Z))
own_diag = np.diag(H)
print(f"       own-class sigmoid: mean={own_diag.mean():.3f} min={own_diag.min():.3f}")
offdiag = H[~np.eye(N, dtype=bool)]
print(f"       off-class sigmoid: mean={offdiag.mean():.3f} max={offdiag.max():.3f}")

# --------- Scheme L (INVENTED): Iterative decorrelating imprint ----
# After initial imprint, repeatedly subtract component along most-similar
# other prototype until all pairwise correlations < tau.
def iter_decorrelate_imprint(E, tau=0.15, max_iter=200, bias=-0.5):
    W = np.zeros((EMBED_SIZE, E.shape[0]), np.float32)
    for i, e in enumerate(E):
        W[:, i] = e / (np.linalg.norm(e)+1e-9)
    for it in range(max_iter):
        # cosine similarity of prototypes
        S = W.T @ W
        np.fill_diagonal(S, 0)
        if np.abs(S).max() < tau: break
        # find most correlated pair, push them apart
        i, j = np.unravel_index(np.argmax(np.abs(S)), S.shape)
        # remove j's component from i (and vice versa)
        W[:, i] = W[:, i] - (W[:, i] @ W[:, j]) * W[:, j] * 0.5
        W[:, j] = W[:, j] - (W[:, j] @ W[:, i]) * W[:, i] * 0.5
        # renormalize
        W[:, i] /= (np.linalg.norm(W[:, i])+1e-9)
        W[:, j] /= (np.linalg.norm(W[:, j])+1e-9)
    return W, np.full(E.shape[0], bias, np.float32)

W, b = iter_decorrelate_imprint(E)
acc, _ = run_sigmoid_hidden(E, W, b)
results.append(("(INV) Iter-decorrelate imprint K=26", acc))
print(f"  (INV) Iter-decorrelate imprint   -> {acc*100:5.1f}%")

# --------- Scheme M (INVENTED): "Committee by random imprint+noise" ----
# Train on imprint, then add a random noise perturbation to each neuron
# and re-imprint.  Similar to dropout-trained ensembles.
def noisy_imprint(E, K=64, sigma=0.3, seed=4):
    g = np.random.default_rng(seed)
    W = np.zeros((EMBED_SIZE, K), np.float32)
    for k in range(K):
        i = g.integers(0, E.shape[0])
        v = E[i] + g.normal(0, sigma, EMBED_SIZE) * np.where(np.arange(EMBED_SIZE) < ALPHA_RAW_SIZE, 0, 1)
        W[:, k] = v / (np.linalg.norm(v)+1e-9)
    return W, np.full(K, -0.5, np.float32)

for K in [26, 64, 128]:
    W, b = noisy_imprint(E, K=K)
    acc, _ = run_sigmoid_hidden(E, W, b)
    results.append((f"(INV) Noisy imprint K={K}", acc))
    print(f"  (INV) Noisy imprint K={K:3d}        -> {acc*100:5.1f}%")

# ------------------------------------------------------------------
# Final ranking
# ------------------------------------------------------------------
print("\n========= RANKED RESULTS =========")
results.sort(key=lambda x: -x[1])
for name, acc in results:
    mark = "✓" if acc >= 0.95 else ("~" if acc >= 0.75 else "✗")
    print(f"  {mark} {acc*100:5.1f}%  {name}")

# ------------------------------------------------------------------
# Robustness check: add gaussian noise, re-evaluate top-3
# ------------------------------------------------------------------
print("\n========= ROBUSTNESS (embedding + 0.3 sigma noise) =========")
top = results[:5]
for name, _ in top:
    # Re-run with noisy embeddings
    noise = rng.normal(0, 0.3, E.shape).astype(np.float32)
    noise[:, :ALPHA_RAW_SIZE] = 0
    En = E + noise
    # Very rough: use margin-tuned imprint as canonical robust scheme
    if "Margin-tuned" in name:
        W, b = margin_tuned_imprint(E)
        acc, _ = run_sigmoid_hidden(En, W, b)
        print(f"    {name} on noisy: {acc*100:5.1f}%")
    elif "Iter-decorrelate" in name:
        W, b = iter_decorrelate_imprint(E)
        acc, _ = run_sigmoid_hidden(En, W, b)
        print(f"    {name} on noisy: {acc*100:5.1f}%")
    elif "Anti-pair" in name:
        W, b = anti_pair_imprint(E, bias=0.0)
        acc, _ = run_sigmoid_hidden(En, W, b)
        print(f"    {name} on noisy: {acc*100:5.1f}%")
    elif name.startswith("Imprint K=26, bias=-1"):
        W, b = imprint_hidden(E, bias=-1.0)
        acc, _ = run_sigmoid_hidden(En, W, b)
        print(f"    {name} on noisy: {acc*100:5.1f}%")
    elif name.startswith("Imprint K=26, bias=+0") or name.startswith("Imprint K=26, bias=+1"):
        W, b = imprint_hidden(E, bias=0.0)
        acc, _ = run_sigmoid_hidden(En, W, b)
        print(f"    {name} on noisy: {acc*100:5.1f}%")
