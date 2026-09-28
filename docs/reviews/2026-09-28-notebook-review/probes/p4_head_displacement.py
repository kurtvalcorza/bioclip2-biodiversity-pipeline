
"""Probe P4: how far can the capstone's head move under CONFIG (lr=1e-4, wd=0.01, 20 epochs, batch 8)?
Stdlib re-implementation of capstone.fit_head: Linear(768->4) initialised to logit_scale*text,
bias 0, mean cross-entropy, torch-default AdamW (betas 0.9/0.999, eps 1e-8, decoupled decay),
shuffled batches of 8, epoch 0 included in minimum-validation-loss selection.
SYNTHETIC unit-norm embeddings, not BioCLIP embeddings: zero-shot is deliberately wrong on a subset
with a cosine gap of 0.02 (2 logits at scale 100), i.e. a plausible close confusion."""
import json, math, random, sys
D, C, SCALE = 768, 4, 100.0
rng = random.Random(0)
def unit(v):
    n = math.sqrt(sum(x * x for x in v)); return [x / n for x in v]
def rand_unit(): return unit([rng.gauss(0, 1) for _ in range(D)])
def dot(a, b): return sum(x * y for x, y in zip(a, b))
text = [rand_unit() for _ in range(C)]
def make(label, wrong=None, gap=0.02):
    # feature with cos(f, t_label) = 0.30 (+/- noise); if wrong, cos(f, t_wrong) = 0.30 + gap
    noise = rand_unit()
    base = [0.30 * t for t in text[label]]
    if wrong is not None:
        base = [b + (0.30 + gap) * t for b, t in zip(base, text[wrong])]
    f = unit([b + 0.9 * n for b, n in zip(base, noise)])
    return f
train, val = [], []
for c in range(C):
    for k in range(8): train.append((make(c, (c + 1) % C if k < 3 else None), c))
    for k in range(3): val.append((make(c, (c + 1) % C if k < 1 else None), c))
def logits(W, b, f): return [SCALE * 0 + dot(W[c], f) + b[c] for c in range(C)]
def softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e); return [v / s for v in e]
def loss(W, b, data):
    return sum(-math.log(softmax(logits(W, b, f))[y]) for f, y in data) / len(data)
def fit(lr, wd=0.01, epochs=20, bs=8, seed=42):
    W = [[SCALE * x for x in t] for t in text]; b = [0.0] * C
    W0 = [row[:] for row in W]
    mW = [[0.0] * D for _ in range(C)]; vW = [[0.0] * D for _ in range(C)]; mb = [0.0] * C; vb = [0.0] * C
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    r = random.Random(seed); hist = []; best = (float("inf"), None, None, 0)
    for epoch in range(epochs + 1):
        if epoch:
            order = list(range(len(train))); r.shuffle(order)
            for s in range(0, len(order), bs):
                batch = [train[i] for i in order[s:s + bs]]
                gW = [[0.0] * D for _ in range(C)]; gb = [0.0] * C
                for f, y in batch:
                    p = softmax(logits(W, b, f))
                    for c in range(C):
                        g = (p[c] - (c == y)) / len(batch); gb[c] += g
                        row = gW[c]
                        for d in range(D): row[d] += g * f[d]
                step += 1
                bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
                for c in range(C):
                    for d in range(D):
                        W[c][d] *= 1 - lr * wd
                        mW[c][d] = b1 * mW[c][d] + (1 - b1) * gW[c][d]
                        vW[c][d] = b2 * vW[c][d] + (1 - b2) * gW[c][d] ** 2
                        W[c][d] -= lr * (mW[c][d] / bc1) / (math.sqrt(vW[c][d] / bc2) + eps)
                    b[c] *= 1 - lr * wd
                    mb[c] = b1 * mb[c] + (1 - b1) * gb[c]; vb[c] = b2 * vb[c] + (1 - b2) * gb[c] ** 2
                    b[c] -= lr * (mb[c] / bc1) / (math.sqrt(vb[c] / bc2) + eps)
        vl = loss(W, b, val); hist.append(round(vl, 6))
        if vl < best[0]: best = (vl, [row[:] for row in W], b[:], epoch)
    _, Wb, bb, sel = best
    def summary(Wx, bx):
        disp = max(abs(Wx[c][d] - W0[c][d]) for c in range(C) for d in range(D))
        dl = max(abs(a - z) for f, _ in train + val for a, z in zip(logits(Wx, bx, f), logits(W0, [0.0] * C, f)))
        flips = sum(max(range(C), key=lambda c: logits(Wx, bx, f)[c]) != max(range(C), key=lambda c: logits(W0, [0.0] * C, f)[c]) for f, _ in train + val)
        acc_tr = sum(max(range(C), key=lambda c: logits(Wx, bx, f)[c]) == y for f, y in train) / len(train)
        acc_va = sum(max(range(C), key=lambda c: logits(Wx, bx, f)[c]) == y for f, y in val) / len(val)
        return {"max_abs_weight_change": disp, "max_abs_logit_change": dl, "prediction_flips_vs_zero_shot": flips,
                "train_accuracy": acc_tr, "validation_accuracy": acc_va}
    return {"learning_rate": lr, "optimizer_steps": step, "selected_epoch": sel,
            "validation_loss_by_epoch": hist, "selected": summary(Wb, bb), "final_epoch": summary(W, b)}
zs_train = sum(max(range(C), key=lambda c: dot(text[c], f)) == y for f, y in train) / len(train)
zs_val = sum(max(range(C), key=lambda c: dot(text[c], f)) == y for f, y in val) / len(val)
res = {"zero_shot_train_accuracy": zs_train, "zero_shot_validation_accuracy": zs_val,
       "analytic_bound": {"per_element_displacement_typical_adam": 80 * 1e-4,
                          "per_class_logit_change_bound_unit_norm_features": math.sqrt(768) * 80 * 1e-4 + 80 * 1e-4,
                          "note": "|delta w| <~ lr per Adam step; 80 steps; Cauchy-Schwarz with ||f||=1"},
       "runs": [fit(1e-4)]}
if len(sys.argv) > 1 and sys.argv[1] == "--contrast":
    res["runs"].append(fit(1e-2))
print(json.dumps(res, indent=2))
