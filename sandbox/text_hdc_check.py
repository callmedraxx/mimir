"""Quick check: does HDC / imprint work for TEXT input (31-dim one-hot letter+query)?"""
import numpy as np, math
rng = np.random.default_rng(0)

ALPHA_RAW_LETTER = 26
ALPHA_RAW_QUERY  = 5
D = 128   # embedding size
N = 26    # classes

# Build text-style inputs: one-hot letter + 5-dim query = 31 dims, rest zero
def text_input(letter_idx, query_idx=0):
    v = np.zeros(D, np.float32)
    v[letter_idx] = 1.0
    v[ALPHA_RAW_LETTER + query_idx] = 1.0
    return v

E_text = np.stack([text_input(i, 0) for i in range(N)])
Y = np.eye(N, dtype=np.float32)

def softmax(Z):
    Z = Z - Z.max(axis=-1, keepdims=True); e = np.exp(Z); return e/e.sum(axis=-1, keepdims=True)

def run_softmax_output(H, n_epochs=300, lr=0.2, seed=7):
    g = np.random.default_rng(seed)
    K = H.shape[1]
    W = g.normal(0, 1/math.sqrt(K), (K, N)).astype(np.float32)
    b = np.zeros(N, np.float32)
    for ep in range(n_epochs):
        O = softmax(H @ W + b)
        err = Y - O
        W += lr * H.T @ err / N
        b += lr * err.mean(axis=0)
    pred = O.argmax(axis=1); correct = pred == Y.argmax(axis=1)
    conf = O[np.arange(N), Y.argmax(axis=1)]
    return correct.mean(), (correct & (conf>=0.55)).mean(), conf.mean(), conf.min()

print("TEXT INPUT TESTS (31-dim one-hot, 26 classes)\n")

# HDC on text
def hdc(E, Hd, seed):
    g = np.random.default_rng(seed)
    P = g.normal(0, 1, (D, Hd)).astype(np.float32)
    H = np.sign(E @ P).astype(np.float32)
    return (H + 1) * 0.5

for Hd in [128, 256, 1024]:
    H = hdc(E_text, Hd, seed=11)
    a, s, cm, cn = run_softmax_output(H)
    print(f"  HDC H={Hd}: argmax {a*100:5.1f}% strict {s*100:5.1f}% conf μ={cm:.2f} min={cn:.2f}")

# Imprint on text
def imprint(E):
    W = (E / (np.linalg.norm(E, axis=1, keepdims=True)+1e-9)).T.astype(np.float32)
    # margin-tuned bias
    Z = E @ W
    b = np.zeros(E.shape[0], np.float32)
    for i in range(E.shape[0]):
        others = np.delete(Z[:, i], i); b[i] = -(others.max() + 0.05)
    return W, b

W, b = imprint(E_text)
H = 1.0/(1.0+np.exp(-(E_text @ W + b)))
a, s, cm, cn = run_softmax_output(H)
print(f"  Margin-imprint K=26: argmax {a*100:5.1f}% strict {s*100:5.1f}% conf μ={cm:.2f} min={cn:.2f}")

# SDM on text
def sdm(E_train, E_test, M=1024, K_sparse=50, seed=17):
    g = np.random.default_rng(seed)
    addrs = np.sign(g.normal(0, 1, (M, D))).astype(np.float32)
    Xtr = np.sign(E_train).astype(np.float32); Xtr[Xtr==0] = 1   # sign(0)=0 fix
    Xte = np.sign(E_test).astype(np.float32);  Xte[Xte==0] = 1
    store = np.zeros((M, N), np.float32)
    for i in range(E_train.shape[0]):
        dots = addrs @ Xtr[i]
        top = np.argpartition(-dots, K_sparse)[:K_sparse]
        store[top] += Y[i]
    preds = []
    for i in range(E_test.shape[0]):
        dots = addrs @ Xte[i]
        top = np.argpartition(-dots, K_sparse)[:K_sparse]
        preds.append(store[top].sum(axis=0))
    return np.stack(preds)

P = sdm(E_text, E_text)
pred = P.argmax(axis=1); correct = pred == Y.argmax(axis=1)
conf = softmax(P)[np.arange(N), Y.argmax(axis=1)]
print(f"  SDM M=1024 K=50: argmax {correct.mean()*100:5.1f}% strict "
      f"{(correct & (conf>=0.55)).mean()*100:5.1f}% conf μ={conf.mean():.2f} min={conf.min():.2f}")

# Cross-modal: can ONE hidden scheme handle BOTH text and vision?
# Load vision embeddings
data = np.load("/root/mimir/sandbox/embeddings.npz", allow_pickle=True)
E_vis = data["E"]

print("\nCROSS-MODAL (same weights, different inputs)")
# HDC: same random projection for both modalities
g = np.random.default_rng(11)
P = g.normal(0, 1, (D, 1024)).astype(np.float32)
H_text = (np.sign(E_text @ P) + 1) * 0.5
H_vis  = (np.sign(E_vis  @ P) + 1) * 0.5
# Concatenate training: same classes, both modality signatures
# Train ONE output layer on BOTH
def run_combined(H_list, n_epochs=300, lr=0.2, seed=7):
    g = np.random.default_rng(seed)
    K = H_list[0].shape[1]
    W = g.normal(0, 1/math.sqrt(K), (K, N)).astype(np.float32)
    b = np.zeros(N, np.float32)
    for ep in range(n_epochs):
        for H in H_list:
            O = softmax(H @ W + b)
            err = Y - O
            W += lr * H.T @ err / N
            b += lr * err.mean(axis=0)
    # Test each modality
    results = []
    for H in H_list:
        O = softmax(H @ W + b)
        pred = O.argmax(axis=1); correct = pred == Y.argmax(axis=1)
        conf = O[np.arange(N), Y.argmax(axis=1)]
        results.append(((correct & (conf>=0.55)).mean(), conf.mean(), conf.min()))
    return results

r = run_combined([H_text, H_vis])
print(f"  HDC H=1024 (shared): TEXT strict={r[0][0]*100:.1f}% conf μ={r[0][1]:.2f} "
      f"|| VISION strict={r[1][0]*100:.1f}% conf μ={r[1][1]:.2f}")
