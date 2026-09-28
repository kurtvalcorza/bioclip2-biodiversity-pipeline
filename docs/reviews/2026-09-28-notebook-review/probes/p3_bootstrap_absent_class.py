
"""Probe P3: macro-F1 bootstrap interval artefact from absent classes.
Stdlib port of biodiversity_core.metrics (macro-F1 path) and biodiversity_core.bootstrap.
Uses the real test-split layout (labels, observers) from the carried data_manifest.json.
The carried code uses numpy.random.default_rng(42); this port uses random.Random(42),
so individual draws differ, but the class-absence probability is structural."""
import json, random, itertools, sys
from fractions import Fraction
from collections import Counter
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
classes = [c["label"] for c in manifest["classes"]]
test = [r for r in manifest["records"] if r["split"] == "test"]
y = [classes.index(r["label"]) for r in test]
groups = [r["observer"] for r in test]

def macro_f1(y_true, y_pred, C):
    # Mirrors core.metrics: f1 = 2tp/(support+predicted), 0.0 when both are zero; mean over ALL C classes.
    f1s = []
    for i in range(C):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == i and b == i)
        support = sum(1 for a in y_true if a == i)
        predicted = sum(1 for b in y_pred if b == i)
        f1s.append(2 * tp / (support + predicted) if support + predicted else 0.0)
    return sum(f1s) / C

def quantile(values, q):  # numpy default 'linear' interpolation
    v = sorted(values); pos = (len(v) - 1) * q; lo = int(pos); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)

C = len(classes)
perfect = list(y)  # a classifier that is correct on every test photograph
out = {"test_records": len(y), "test_observers": dict(Counter(groups)),
       "full_sample_macro_f1_perfect": macro_f1(y, perfect, C)}

# (a) record bootstrap, 2000 draws, as core.bootstrap(groups=None)
rng = random.Random(42); vals = []; missing = 0
for _ in range(2000):
    ix = [rng.randrange(len(y)) for _ in range(len(y))]
    yt = [y[i] for i in ix]; yp = [perfect[i] for i in ix]
    missing += len(set(yt)) < C
    vals.append(macro_f1(yt, yp, C))
out["record_bootstrap"] = {"draws": 2000, "fraction_resamples_missing_a_class": missing / 2000,
    "macro_f1_low": quantile(vals, 0.025), "macro_f1_high": quantile(vals, 0.975),
    "accuracy_interval": [1.0, 1.0]}
# exact probability a record resample of 12 (3 per class) lacks >=1 class (inclusion-exclusion)
n = len(y); per = Counter(y)
p = Fraction(0)
for k in range(1, C + 1):
    for combo in itertools.combinations(range(C), k):
        removed = sum(per[c] for c in combo)
        p += (-1) ** (k + 1) * Fraction(n - removed, n) ** n
out["record_bootstrap"]["exact_probability_missing_a_class"] = float(p)

# (b) observer-cluster bootstrap, exact enumeration over all len(G)^len(G) equally likely draws
G = sorted(set(groups)); dist = []
for draw in itertools.product(G, repeat=len(G)):
    ix = [i for g in draw for i in range(len(y)) if groups[i] == g]
    dist.append(macro_f1([y[i] for i in ix], [perfect[i] for i in ix], C))
cnt = Counter(round(v, 4) for v in dist)
out["observer_cluster_bootstrap_exact"] = {"clusters": len(G), "draws_enumerated": len(dist),
    "macro_f1_distribution": {str(k): v / len(dist) for k, v in sorted(cnt.items())},
    "macro_f1_2.5pct": quantile(dist, 0.025), "macro_f1_97.5pct": quantile(dist, 0.975)}
print(json.dumps(out, indent=2))
