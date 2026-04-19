"""
vision_math_2.py — Round 2 what-ifs.

New questions to answer before touching C:
 (A) Is there a TRIVIAL fix? Just change bias on existing K=8 hidden?
 (B) Does output-layer softmax alone give enough margin?
 (C) Can we beat imprint with smarter hidden inits (PCA, cos-sim)?
 (D) Encoding variants: does |.| in Gabor throw away useful sign info?
     Does the z-score discard useful magnitude?
 (E) Invented: hyperdimensional binary, multi-view ensemble,
     hypersphere-gating, sparse-distributed-memory (Kanerva).
 (F) Robustness to noise / shift / brightness.
 (G) Is mimir's 'maturity' multiplier the actual confidence killer?
 (H) Two-layer compound init: random expand -> imprint readout.
"""
import os, math, glob, numpy as np
from numpy.lib.stride_tricks import sliding_window_view

rng = np.random.default_rng(0)

# Load cached embeddings from round 1 sandbox
data = np.load("/root/mimir/sandbox/embeddings.npz", allow_pickle=True)
E_text_zeroed = data["E"]      # (26, 128), text dims zeroed
E_full        = data["E_full"] # (26, 128), full Gabor + z-score
names         = data["names"]
N, D = E_text_zeroed.shape
ALPHA_RAW_SIZE = 31
E = E_text_zeroed

def one_hot(n): return np.eye(n, dtype=np.float32)
Y = one_hot(N)

def sigmoid(z): return 1.0/(1.0+np.exp(-z))
def softmax(Z):
    Z = Z - Z.max(axis=-1, keepdims=True)
    e = np.exp(Z); return e/e.sum(axis=-1, keepdims=True)

def run_sigmoid_output(H, n_epochs=3000, lr=0.1, seed=7):
    g = np.random.default_rng(seed)
    K = H.shape[1]
    W = g.normal(0, 1/math.sqrt(K), (K, N)).astype(np.float32)
    b = np.zeros(N, np.float32)
    for ep in range(n_epochs):
        O = sigmoid(H @ W + b)
        err = Y - O
        W += lr * H.T @ err / N
        b += lr * err.mean(axis=0)
    pred = O.argmax(axis=1)
    correct = pred == Y.argmax(axis=1)
    # also return "conf >= 0.55" strict
    argmax_acc = correct.mean()
    own_conf = O[np.arange(N), Y.argmax(axis=1)]
    strict_acc = (correct & (own_conf >= 0.55)).mean()
    return argmax_acc, strict_acc, own_conf.mean(), own_conf.min()

def run_softmax_output(H, n_epochs=3000, lr=0.1, seed=7):
    """Same delta-style, but on softmax output — natural margin."""
    g = np.random.default_rng(seed)
    K = H.shape[1]
    W = g.normal(0, 1/math.sqrt(K), (K, N)).astype(np.float32)
    b = np.zeros(N, np.float32)
    for ep in range(n_epochs):
        Z = H @ W + b
        O = softmax(Z)
        err = Y - O
        W += lr * H.T @ err / N
        b += lr * err.mean(axis=0)
    pred = O.argmax(axis=1)
    correct = pred == Y.argmax(axis=1)
    argmax_acc = correct.mean()
    own_conf = O[np.arange(N), Y.argmax(axis=1)]
    strict_acc = (correct & (own_conf >= 0.55)).mean()
    return argmax_acc, strict_acc, own_conf.mean(), own_conf.min()

def report(name, acc_arg, acc_strict, conf_mean, conf_min, extra=""):
    marker = "✓" if acc_strict >= 0.95 else ("~" if acc_strict >= 0.75 else "✗")
    print(f"  {marker} argmax {acc_arg*100:5.1f}%  strict(≥0.55) {acc_strict*100:5.1f}%"
          f"  conf μ={conf_mean:.2f} min={conf_min:.2f}   {name}")
    if extra: print(f"         {extra}")

print("\n========= (A) CHEAP FIX: bias sweep on K=8 Xavier =========")
def xavier_hidden(K, bias, seed=3):
    g = np.random.default_rng(seed)
    vis = D - ALPHA_RAW_SIZE
    W = np.zeros((D, K), np.float32)
    W[ALPHA_RAW_SIZE:, :] = g.uniform(-1/math.sqrt(vis), 1/math.sqrt(vis), (vis, K))
    b = np.full(K, bias, np.float32)
    return W, b

for bb in [-5, -3, -1, 0, 1, 3]:
    W, b = xavier_hidden(8, bb)
    H = sigmoid(E @ W + b)
    a, s, cm, cn = run_sigmoid_output(H)
    report(f"K=8 Xavier bias={bb:+d}", a, s, cm, cn)

print("\n========= (B) OUTPUT SOFTMAX vs SIGMOID (same hidden) =========")
# Use plain K=26 Imprint, compare sigmoid-output vs softmax-output
def imprint_W(E_, bias=-1.0):
    W = np.zeros((D, E_.shape[0]), np.float32)
    for i, e in enumerate(E_):
        W[:, i] = e / (np.linalg.norm(e)+1e-9)
    return W, np.full(E_.shape[0], bias, np.float32)

for bias in [-5, -1, 0]:
    W, b = imprint_W(E, bias=bias)
    H = sigmoid(E @ W + b)
    a1, s1, cm1, cn1 = run_sigmoid_output(H)
    report(f"Imprint K=26 bias={bias} + SIGMOID output", a1, s1, cm1, cn1)
    a2, s2, cm2, cn2 = run_softmax_output(H)
    report(f"Imprint K=26 bias={bias} + SOFTMAX output", a2, s2, cm2, cn2)

# Softmax output with the CURRENT K=8 Xavier setup
for bb in [-5, -1, 0]:
    W, b = xavier_hidden(8, bb)
    H = sigmoid(E @ W + b)
    a, s, cm, cn = run_softmax_output(H)
    report(f"K=8 Xavier bias={bb} + SOFTMAX output", a, s, cm, cn)

print("\n========= (C) PCA & COSINE-SIM HIDDEN =========")
# (C1) Top-N principal components of the 26-class matrix
U, S, Vt = np.linalg.svd(E, full_matrices=False)   # V rows are PCs
for K in [8, 16, 26]:
    W_pca = Vt[:K, :].T.astype(np.float32)   # (D, K)
    # bias -1.0 is a middling start
    b_pca = np.full(K, -1.0, np.float32)
    H = sigmoid(E @ W_pca + b_pca)
    a, s, cm, cn = run_sigmoid_output(H)
    report(f"PCA K={K} bias=-1 (sigmoid out)", a, s, cm, cn)
    a, s, cm, cn = run_softmax_output(H)
    report(f"PCA K={K} bias=-1 (softmax out)", a, s, cm, cn)

# (C2) Cosine-similarity hidden: h_i = cos(x, w_i), no sigmoid, no bias
def cos_hidden(E_inputs, E_protos):
    Xn = E_inputs / (np.linalg.norm(E_inputs, axis=1, keepdims=True)+1e-9)
    Pn = E_protos / (np.linalg.norm(E_protos, axis=1, keepdims=True)+1e-9)
    return Xn @ Pn.T

H = cos_hidden(E, E).astype(np.float32)  # (N, N)
a, s, cm, cn = run_sigmoid_output(H)
report("Cosine-sim hidden K=26 (sigmoid out)", a, s, cm, cn)
a, s, cm, cn = run_softmax_output(H)
report("Cosine-sim hidden K=26 (softmax out)", a, s, cm, cn)

print("\n========= (D) ENCODING VARIANTS (reload raw, re-encode) =========")
# We don't have the raw pipeline here — use the cached E_full and manipulate:
# (D1) no z-score: simulate by adding back a scaled mean offset
#   (approximation — real test would redo Gabor pipeline)
#   Skip: can't truly invert z-score from cached data.  Flag as needs-rerun.
print("  (D1/D2 skipped — would need to rerun Gabor pipeline; will revisit if promising)")

print("\n========= (E) INVENTED METHODS =========")

# (E1) Hyperdimensional binary projection (HDC-style).
#   Project E to H_dim via sign(random_proj).  Class = nearest prototype by Hamming.
def hdc_hidden(E_, H_dim=1024, seed=11):
    g = np.random.default_rng(seed)
    P = g.normal(0, 1, (D, H_dim)).astype(np.float32)
    H = np.sign(E_ @ P).astype(np.float32)   # (N, H_dim) in {-1,+1}
    return H

for Hd in [256, 1024, 4096]:
    H = hdc_hidden(E, H_dim=Hd)
    # Normalize binary hidden to roughly [0,1] by mapping -1 → 0
    H01 = (H + 1) * 0.5
    a, s, cm, cn = run_sigmoid_output(H01)
    report(f"(INV) HDC binary H={Hd} (sigmoid out)", a, s, cm, cn)
    a, s, cm, cn = run_softmax_output(H01)
    report(f"(INV) HDC binary H={Hd} (softmax out)", a, s, cm, cn)

# (E2) Multi-view ensemble: imprint but per-orientation.
#   26 classes × 2 orientation groups (orient 0-3 vs 4-7) = 52 neurons.
def multi_view_imprint(E_):
    # Orient groups: feature_idx = o*16 + gy*4 + gx; group1 = o in {0-3}, group2 = o in {4-7}
    W = np.zeros((D, 2*N), np.float32)
    mask1 = np.zeros(D); mask2 = np.zeros(D)
    for o in range(8):
        for g in range(16):
            idx = o*16 + g
            if o < 4: mask1[idx] = 1.0
            else:     mask2[idx] = 1.0
    for i, e in enumerate(E_):
        v1 = e * mask1; v1 /= (np.linalg.norm(v1)+1e-9)
        v2 = e * mask2; v2 /= (np.linalg.norm(v2)+1e-9)
        W[:, 2*i]   = v1
        W[:, 2*i+1] = v2
    return W, np.full(2*N, -0.5, np.float32)

W, b = multi_view_imprint(E)
H = sigmoid(E @ W + b)
a, s, cm, cn = run_sigmoid_output(H)
report("(INV) Multi-view imprint K=52 (sigmoid out)", a, s, cm, cn)
a, s, cm, cn = run_softmax_output(H)
report("(INV) Multi-view imprint K=52 (softmax out)", a, s, cm, cn)

# (E3) Hypersphere gating — pure cosine with learned threshold per neuron.
#   h_i = max(0, cos(x, w_i) - theta_i).  theta = max off-class cos.
def hypersphere_gate(E_, slack=0.0):
    # Prototypes = own embeddings, normalized
    Pn = E_ / (np.linalg.norm(E_, axis=1, keepdims=True)+1e-9)
    # cos similarity matrix
    S = Pn @ Pn.T
    theta = np.zeros(N, np.float32)
    for i in range(N):
        others = np.delete(S[:, i], i)
        theta[i] = others.max() + slack
    return Pn, theta

def run_hypersphere(E_, slack=0.0):
    Pn, theta = hypersphere_gate(E_, slack)
    Xn = E_ / (np.linalg.norm(E_, axis=1, keepdims=True)+1e-9)
    H = np.maximum(0.0, Xn @ Pn.T - theta[None, :])
    # Feed to softmax output (linear, no bias)
    a, s, cm, cn = run_softmax_output(H)
    return a, s, cm, cn

for sl in [0.0, 0.05, 0.1]:
    a, s, cm, cn = run_hypersphere(E, slack=sl)
    report(f"(INV) Hypersphere gate slack={sl} (softmax out)", a, s, cm, cn)

# (E4) Sparse Distributed Memory (Kanerva-style).
#   Random binary addresses (1024 cells).  Each input activates the
#   K_sparse nearest addresses by Hamming.  Class vote averaged.
def sdm_predict(E_train, Y_train, E_test, M=1024, K_sparse=50, seed=17):
    g = np.random.default_rng(seed)
    addrs = np.sign(g.normal(0, 1, (M, D))).astype(np.float32)
    # Encode each input as {-1,+1} via sign; compute Hamming-equivalent dot
    X_train = np.sign(E_train).astype(np.float32)
    X_test  = np.sign(E_test).astype(np.float32)
    # For each training sample, store its class in the top-K nearest addresses
    store = np.zeros((M, N), np.float32)
    for i in range(E_train.shape[0]):
        dots = addrs @ X_train[i]
        top = np.argpartition(-dots, K_sparse)[:K_sparse]
        store[top] += Y_train[i]
    # Predict: for each test sample, sum stored votes of top-K nearest addresses
    preds = []
    for i in range(E_test.shape[0]):
        dots = addrs @ X_test[i]
        top = np.argpartition(-dots, K_sparse)[:K_sparse]
        vote = store[top].sum(axis=0)
        preds.append(vote)
    return np.stack(preds)

P = sdm_predict(E, Y, E)
pred = P.argmax(axis=1); correct = pred == Y.argmax(axis=1)
conf = softmax(P)[np.arange(N), Y.argmax(axis=1)]
report("(INV) SDM (Kanerva) K=50 M=1024", correct.mean(),
       (correct & (conf >= 0.55)).mean(), conf.mean(), conf.min())

print("\n========= (F) TWO-LAYER COMPOUND =========")
# Random K=64 expansion, then imprint K=26 on expanded space
def two_layer(E_, K1=64, seed=5):
    g = np.random.default_rng(seed)
    vis = D - ALPHA_RAW_SIZE
    W1 = np.zeros((D, K1), np.float32)
    W1[ALPHA_RAW_SIZE:, :] = g.normal(0, 1/math.sqrt(vis), (vis, K1))
    b1 = np.full(K1, 0.0, np.float32)
    H1 = np.tanh(E_ @ W1 + b1)
    # Imprint K=26 on H1
    W2 = np.zeros((K1, N), np.float32)
    for i in range(N):
        v = H1[i]
        W2[:, i] = v / (np.linalg.norm(v)+1e-9)
    # margin-tune bias
    Z = H1 @ W2
    b2 = np.zeros(N, np.float32)
    for i in range(N):
        others = np.delete(Z[:, i], i); b2[i] = -(others.max()+0.05)
    # Apply
    H2 = sigmoid(H1 @ W2 + b2)
    return H2

H = two_layer(E, K1=64)
a, s, cm, cn = run_sigmoid_output(H)
report("Two-layer: random-64 tanh -> imprint-26 sigmoid", a, s, cm, cn)
a, s, cm, cn = run_softmax_output(H)
report("Two-layer: random-64 tanh -> imprint-26 softmax", a, s, cm, cn)

print("\n========= (G) MATURITY MULTIPLIER CHECK =========")
# mimir multiplies output by neuron.maturity in ∈ [0, 1].  If maturity<1,
# confidence is scaled down.  Simulate: imprint K=26 with margin-tune bias,
# then apply maturity ∈ {1.0, 0.8, 0.5, 0.3}.
W, b = imprint_W(E, bias=0)
# margin-tune hidden
Z = E @ W
for i in range(N):
    others = np.delete(Z[:, i], i); b[i] = -(others.max()+0.05)
H = sigmoid(E @ W + b)
for mat in [1.0, 0.8, 0.5, 0.3]:
    # Train output normally
    a1, s1, cm1, cn1 = run_sigmoid_output(H)
    # Then apply maturity multiplier to output
    K = H.shape[1]
    g = np.random.default_rng(7)
    Wo = g.normal(0, 1/math.sqrt(K), (K, N)).astype(np.float32)
    bo = np.zeros(N, np.float32)
    for ep in range(3000):
        O = sigmoid(H @ Wo + bo)
        err = Y - O
        Wo += 0.1 * H.T @ err / N
        bo += 0.1 * err.mean(axis=0)
    O_mat = sigmoid(H @ Wo + bo) * mat  # apply maturity
    pred = O_mat.argmax(axis=1); correct = pred == Y.argmax(axis=1)
    conf = O_mat[np.arange(N), Y.argmax(axis=1)]
    report(f"Margin-imprint + maturity={mat}", correct.mean(),
           (correct & (conf >= 0.55)).mean(), conf.mean(), conf.min())

print("\n========= (H) ROBUSTNESS: noise / brightness / shift-by-noise =========")
# Using margin-tuned imprint K=26 as baseline
W, b = imprint_W(E, bias=0)
Z = E @ W
for i in range(N):
    others = np.delete(Z[:, i], i); b[i] = -(others.max()+0.05)
for sigma in [0.05, 0.1, 0.2, 0.4, 0.8]:
    noise = rng.normal(0, sigma, E.shape).astype(np.float32)
    noise[:, :ALPHA_RAW_SIZE] = 0
    En = E + noise
    H = sigmoid(En @ W + b)
    a, s, cm, cn = run_softmax_output(H)
    report(f"Margin-imprint + noise σ={sigma} (softmax)", a, s, cm, cn)

# Brightness: scale whole embedding
for scale in [0.5, 0.8, 1.2, 2.0]:
    H = sigmoid((E*scale) @ W + b)
    a, s, cm, cn = run_softmax_output(H)
    report(f"Margin-imprint + scale={scale} (softmax)", a, s, cm, cn)

print("\n========= DONE =========")
