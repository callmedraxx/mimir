"""
Math test for bit-packed HDC (Option 1).

Reference: Gaussian-weight HDC (what we shipped) — 100% strict on text+vision.
Goal: find out if ±1-weight HDC matches it, since ±1 enables 32x memory
reduction (1 bit/weight) and XOR+popcount forward pass.

Variants tested:
  A. Gaussian weights W ~ N(0,1), step activation       [reference — current C code]
  B. ±1 weights W = sign(N(0,1)), step activation       [proposed bit-packed]
  C. ±1 weights, bipolar input x' = sign(x) (x != 0)     [full binary HDC]
  D. B but with H sweep {64, 128, 256, 512}
  E. B on zero-centered input (so "no-text" dims aren't biased)
  F. Noise robustness: add Gaussian noise to vision embedding and check decay

Metrics:
  argmax accuracy, strict@0.55, mean/min confidence, train epochs to converge,
  margin (winner - runner-up).
"""
import numpy as np, math, time

rng = np.random.default_rng(0)
D = 128
N = 26
ALPHA_RAW_LETTER = 26
ALPHA_RAW_QUERY  = 5

# ── Build inputs ───────────────────────────────────────────────────────────
def text_input(letter_idx, query_idx=0):
    v = np.zeros(D, np.float32)
    v[letter_idx] = 1.0
    v[ALPHA_RAW_LETTER + query_idx] = 1.0
    return v

E_text = np.stack([text_input(i, 0) for i in range(N)])
Y = np.eye(N, dtype=np.float32)

try:
    data = np.load("/root/mimir/sandbox/embeddings.npz", allow_pickle=True)
    E_vis = data["E"].astype(np.float32)
    print(f"Loaded vision embeddings: {E_vis.shape}")
except Exception as e:
    print(f"Could not load vision embeddings: {e}")
    E_vis = None

def softmax(Z):
    Z = Z - Z.max(axis=-1, keepdims=True); e = np.exp(Z); return e / e.sum(axis=-1, keepdims=True)

def train_output(H, n_epochs=300, lr=0.2, seed=7):
    g = np.random.default_rng(seed)
    K = H.shape[1]
    W = g.normal(0, 1/math.sqrt(K), (K, N)).astype(np.float32)
    b = np.zeros(N, np.float32)
    conv_epoch = -1
    for ep in range(n_epochs):
        O = softmax(H @ W + b)
        err = Y - O
        W += lr * H.T @ err / N
        b += lr * err.mean(axis=0)
        if conv_epoch < 0:
            pred = O.argmax(axis=1)
            if (pred == np.arange(N)).all() and O[np.arange(N), np.arange(N)].min() >= 0.90:
                conv_epoch = ep
    pred = O.argmax(axis=1)
    correct = pred == Y.argmax(axis=1)
    conf = O[np.arange(N), Y.argmax(axis=1)]
    # Margin: winner - runner-up
    sorted_O = np.sort(O, axis=1)
    margin = sorted_O[:, -1] - sorted_O[:, -2]
    return {
        "argmax":  correct.mean(),
        "strict":  (correct & (conf >= 0.55)).mean(),
        "conf_m":  conf.mean(),
        "conf_min": conf.min(),
        "margin_m": margin.mean(),
        "margin_min": margin.min(),
        "conv_ep": conv_epoch,
    }

# ── HDC variants ───────────────────────────────────────────────────────────
def hdc_gaussian(E, H, seed):
    g = np.random.default_rng(seed)
    P = g.normal(0, 1, (D, H)).astype(np.float32)
    return (np.sign(E @ P).astype(np.float32) + 1) * 0.5

def hdc_binary(E, H, seed):
    """±1 weights, float input, step activation."""
    g = np.random.default_rng(seed)
    P = np.sign(g.normal(0, 1, (D, H))).astype(np.float32)
    P[P == 0] = 1
    return (np.sign(E @ P).astype(np.float32) + 1) * 0.5

def hdc_binary_bipolar_input(E, H, seed):
    """±1 weights, bipolar input (sign(x) with zeros → 0), step activation."""
    g = np.random.default_rng(seed)
    P = np.sign(g.normal(0, 1, (D, H))).astype(np.float32)
    P[P == 0] = 1
    X = np.sign(E).astype(np.float32)  # leaves true zeros at 0, else ±1
    return (np.sign(X @ P).astype(np.float32) + 1) * 0.5

def hdc_binary_fully_bipolar(E, H, seed):
    """±1 weights, fully bipolar input (sign(x); zeros → ±1 coin flip)."""
    g = np.random.default_rng(seed)
    P = np.sign(g.normal(0, 1, (D, H))).astype(np.float32)
    P[P == 0] = 1
    X = np.sign(E).astype(np.float32)
    # Replace exact zeros with random ±1 (makes it a true binary vector)
    z_mask = (X == 0)
    flip = (g.integers(0, 2, size=z_mask.sum()) * 2 - 1).astype(np.float32)
    X[z_mask] = flip
    return (np.sign(X @ P).astype(np.float32) + 1) * 0.5

# ── Run ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("A. GAUSSIAN HDC (reference — current C code)")
print("=" * 78)
for H in [128, 256, 512]:
    rt = train_output(hdc_gaussian(E_text, H, seed=11))
    label = f"  H={H:4d} TEXT"
    print(f"{label}  argmax {rt['argmax']*100:5.1f}%  strict {rt['strict']*100:5.1f}%  "
          f"conf μ={rt['conf_m']:.2f} min={rt['conf_min']:.2f}  "
          f"margin μ={rt['margin_m']:.2f} min={rt['margin_min']:.2f}  conv@{rt['conv_ep']}")
    if E_vis is not None:
        rv = train_output(hdc_gaussian(E_vis, H, seed=11))
        print(f"  H={H:4d} VIS   argmax {rv['argmax']*100:5.1f}%  strict {rv['strict']*100:5.1f}%  "
              f"conf μ={rv['conf_m']:.2f} min={rv['conf_min']:.2f}  "
              f"margin μ={rv['margin_m']:.2f} min={rv['margin_min']:.2f}  conv@{rv['conv_ep']}")

print("\n" + "=" * 78)
print("B. ±1 WEIGHTS + FLOAT INPUT (proposed bit-packed; input stays float)")
print("=" * 78)
for H in [64, 128, 256, 512]:
    rt = train_output(hdc_binary(E_text, H, seed=11))
    label = f"  H={H:4d} TEXT"
    print(f"{label}  argmax {rt['argmax']*100:5.1f}%  strict {rt['strict']*100:5.1f}%  "
          f"conf μ={rt['conf_m']:.2f} min={rt['conf_min']:.2f}  "
          f"margin μ={rt['margin_m']:.2f} min={rt['margin_min']:.2f}  conv@{rt['conv_ep']}")
    if E_vis is not None:
        rv = train_output(hdc_binary(E_vis, H, seed=11))
        print(f"  H={H:4d} VIS   argmax {rv['argmax']*100:5.1f}%  strict {rv['strict']*100:5.1f}%  "
              f"conf μ={rv['conf_m']:.2f} min={rv['conf_min']:.2f}  "
              f"margin μ={rv['margin_m']:.2f} min={rv['margin_min']:.2f}  conv@{rv['conv_ep']}")

print("\n" + "=" * 78)
print("C. ±1 WEIGHTS + BIPOLAR INPUT (sign(x), zeros kept at 0) — half-binary HDC")
print("=" * 78)
for H in [128, 256, 512]:
    rt = train_output(hdc_binary_bipolar_input(E_text, H, seed=11))
    label = f"  H={H:4d} TEXT"
    print(f"{label}  argmax {rt['argmax']*100:5.1f}%  strict {rt['strict']*100:5.1f}%  "
          f"conf μ={rt['conf_m']:.2f} min={rt['conf_min']:.2f}  conv@{rt['conv_ep']}")
    if E_vis is not None:
        rv = train_output(hdc_binary_bipolar_input(E_vis, H, seed=11))
        print(f"  H={H:4d} VIS   argmax {rv['argmax']*100:5.1f}%  strict {rv['strict']*100:5.1f}%  "
              f"conf μ={rv['conf_m']:.2f} min={rv['conf_min']:.2f}  conv@{rv['conv_ep']}")

print("\n" + "=" * 78)
print("D. ±1 WEIGHTS + FULLY BIPOLAR INPUT (x==0 → random ±1)")
print("   ← this is the TRUE XOR+popcount form.  Input and weights both bit-packable.")
print("=" * 78)
for H in [128, 256, 512]:
    rt = train_output(hdc_binary_fully_bipolar(E_text, H, seed=11))
    label = f"  H={H:4d} TEXT"
    print(f"{label}  argmax {rt['argmax']*100:5.1f}%  strict {rt['strict']*100:5.1f}%  "
          f"conf μ={rt['conf_m']:.2f} min={rt['conf_min']:.2f}  conv@{rt['conv_ep']}")
    if E_vis is not None:
        rv = train_output(hdc_binary_fully_bipolar(E_vis, H, seed=11))
        print(f"  H={H:4d} VIS   argmax {rv['argmax']*100:5.1f}%  strict {rv['strict']*100:5.1f}%  "
              f"conf μ={rv['conf_m']:.2f} min={rv['conf_min']:.2f}  conv@{rv['conv_ep']}")

# ── Stability across seeds (is ±1 robust?) ─────────────────────────────────
print("\n" + "=" * 78)
print("E. SEED STABILITY  (H=256, ±1 weights, vision) — how sensitive to projection?")
print("=" * 78)
if E_vis is not None:
    strict_list = []
    for seed in range(10):
        r = train_output(hdc_binary(E_vis, 256, seed=seed))
        strict_list.append(r["strict"])
        print(f"  seed={seed:2d}  strict {r['strict']*100:5.1f}%  conf_min {r['conf_min']:.2f}  margin_min {r['margin_min']:.2f}")
    s = np.array(strict_list)
    print(f"  → strict across seeds: μ={s.mean()*100:.1f}%  σ={s.std()*100:.2f}%  min={s.min()*100:.1f}%  max={s.max()*100:.1f}%")

# ── Noise robustness ───────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("F. NOISE ROBUSTNESS  (train clean, test noisy — vision only, H=256)")
print("=" * 78)
if E_vis is not None:
    g = np.random.default_rng(11)
    P_gauss = g.normal(0, 1, (D, 256)).astype(np.float32)
    P_bin   = np.sign(P_gauss).astype(np.float32); P_bin[P_bin == 0] = 1

    # Train both on clean embeddings, evaluate on noisy
    H_gauss_clean = (np.sign(E_vis @ P_gauss) + 1) * 0.5
    H_bin_clean   = (np.sign(E_vis @ P_bin)   + 1) * 0.5

    # Reuse delta weights from training
    def train_weights(H, n_epochs=300, lr=0.2, seed=7):
        gg = np.random.default_rng(seed)
        K = H.shape[1]
        W = gg.normal(0, 1/math.sqrt(K), (K, N)).astype(np.float32)
        b = np.zeros(N, np.float32)
        for ep in range(n_epochs):
            O = softmax(H @ W + b)
            W += lr * H.T @ (Y - O) / N
            b += lr * (Y - O).mean(axis=0)
        return W, b

    W_g, b_g = train_weights(H_gauss_clean)
    W_b, b_b = train_weights(H_bin_clean)

    for sigma in [0.05, 0.10, 0.20, 0.30, 0.50]:
        noisy = E_vis + g.normal(0, sigma, E_vis.shape).astype(np.float32)
        Hg = (np.sign(noisy @ P_gauss) + 1) * 0.5
        Hb = (np.sign(noisy @ P_bin)   + 1) * 0.5
        Og = softmax(Hg @ W_g + b_g)
        Ob = softmax(Hb @ W_b + b_b)
        pg = Og.argmax(axis=1); pb = Ob.argmax(axis=1)
        cg = Og[np.arange(N), np.arange(N)]; cb = Ob[np.arange(N), np.arange(N)]
        print(f"  σ={sigma:.2f}  gauss: argmax {(pg == np.arange(N)).mean()*100:5.1f}% conf μ={cg.mean():.2f}  "
              f"||  ±1: argmax {(pb == np.arange(N)).mean()*100:5.1f}% conf μ={cb.mean():.2f}")

# ── Actual memory savings estimate ─────────────────────────────────────────
print("\n" + "=" * 78)
print("MEMORY ESTIMATE  (ABC brain hidden layer only)")
print("=" * 78)
for H in [128, 256, 512]:
    gauss_bytes = H * D * 4             # float32
    bit_bytes   = H * ((D + 7) // 8)    # 1 bit per weight, byte-aligned
    print(f"  H={H:4d}:  Gaussian {gauss_bytes/1024:6.1f} KB   ±1 packed {bit_bytes/1024:6.2f} KB   "
          f"({gauss_bytes/bit_bytes:.1f}× smaller)")
