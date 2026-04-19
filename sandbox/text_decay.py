"""
text_decay.py — analytic check: is text decay during 'learn all' caused by
(a) vision_train touching text weights, or
(b) alpha_teach of new letters overwriting output weights for prior letters
    (classic catastrophic forgetting)?

Evidence from live run:
  After teaching X:   vision_train diagnostic shows text 7/24
  After teaching Y:   vision_train diagnostic shows text 5/25  (two dropped)
  After teaching Z:   vision_train diagnostic shows text 5/26

Within any single vision_train call, text N/M stays CONSTANT across epochs.
That rules out (a) — vision_train is not damaging text weights during its
own loop (which is consistent with the code: it only writes visual_bias
and weights[h] for is_visual hidden).

So the culprit is (b): alpha_teach between vision_train calls.  Simulate
this with a simple sigmoid classifier over sequentially-taught classes to
quantify expected decay vs observed.
"""
import numpy as np, math

rng = np.random.default_rng(0)
N_CLASSES = 26
HIDDEN    = 26
EPOCHS_PER_TEACH = 3000
LR = 0.1
THR = 0.55   # alpha recall threshold

# Each class gets a random 'hidden activation pattern' — stand-in for the
# hidden-layer representation that alpha_forward produces for letter i.
# Make them moderately separable (like the actual alphabet brain).
H = rng.normal(0, 1, (N_CLASSES, HIDDEN)).astype(np.float32)
H = H / (np.linalg.norm(H, axis=1, keepdims=True)+1e-9)

def teach_new(W, b, target_idx, sample_h, n_epochs, all_seen):
    """Simulate alpha_teach: train ONLY on the new letter's sample, for many epochs,
    AND include a few replay samples of prior letters.  Vary the replay weight."""
    Y_target = np.zeros(N_CLASSES, np.float32); Y_target[target_idx] = 1.0
    for ep in range(n_epochs):
        # Forward
        z = sample_h @ W + b
        o = 1/(1+np.exp(-z))
        err = Y_target - o
        W += LR * np.outer(sample_h, err)
        b += LR * err
    return W, b

def accuracy(W, b, H, taught):
    correct = 0
    for i in taught:
        z = H[i] @ W + b
        o = 1/(1+np.exp(-z))
        if o.argmax() == i and o[i] >= THR: correct += 1
    return correct

# Scenario 1: naive sequential teach, no replay
W = np.zeros((HIDDEN, N_CLASSES), np.float32); b = np.zeros(N_CLASSES, np.float32)
taught = []
print("Scenario 1: sequential alpha_teach, NO replay")
for step in range(N_CLASSES):
    W, b = teach_new(W, b, step, H[step], EPOCHS_PER_TEACH, taught[:])
    taught.append(step)
    if step % 4 == 0 or step == N_CLASSES-1:
        acc = accuracy(W, b, H, taught)
        print(f"  after teaching letter {step+1:2d}: recall {acc}/{len(taught)}")

# Scenario 2: with replay (re-train old letters at low weight)
W = np.zeros((HIDDEN, N_CLASSES), np.float32); b = np.zeros(N_CLASSES, np.float32)
taught = []
print("\nScenario 2: sequential alpha_teach + replay 1 sample per prior letter per epoch")
for step in range(N_CLASSES):
    Y = np.zeros(N_CLASSES, np.float32); Y[step] = 1.0
    for ep in range(EPOCHS_PER_TEACH):
        # Main: the new letter
        z = H[step] @ W + b; o = 1/(1+np.exp(-z))
        err = Y - o
        W += LR * np.outer(H[step], err); b += LR * err
        # Replay: every other epoch, also train one prior sample
        if taught and ep % 2 == 0:
            i = taught[ep % len(taught)]
            Yp = np.zeros(N_CLASSES, np.float32); Yp[i] = 1.0
            zp = H[i] @ W + b; op = 1/(1+np.exp(-zp))
            errp = Yp - op
            W += LR * np.outer(H[i], errp); b += LR * errp
    taught.append(step)
    if step % 4 == 0 or step == N_CLASSES-1:
        acc = accuracy(W, b, H, taught)
        print(f"  after teaching letter {step+1:2d}: recall {acc}/{len(taught)}")

# Scenario 3: imprinted output (one-shot, no gradient)
# Each class i has output weight proxying its hidden pattern; bias tuned for margin.
print("\nScenario 3: imprint output weights (no sequential teach) — upper bound")
W = H.T.copy()  # weights = prototypes
# margin-tune bias per class
b = np.zeros(N_CLASSES, np.float32)
Z = H @ W  # (N, N)
for i in range(N_CLASSES):
    others = np.delete(Z[:, i], i)
    b[i] = -(others.max() + 0.05)
acc = accuracy(W, b, H, list(range(N_CLASSES)))
print(f"  recall {acc}/{N_CLASSES}")

print("\n--- Interpretation ---")
print("If scenario 1 degrades (catastrophic forgetting) and scenario 2 recovers,")
print("the fix in mimir is to beef up replay during alpha_teach.")
print("If scenario 3 works, imprinting-style output weight init also helps text.")
