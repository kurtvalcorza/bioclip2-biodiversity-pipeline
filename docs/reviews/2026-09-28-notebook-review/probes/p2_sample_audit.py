
"""Probe P2: frozen-sample audit against spec sections 5-6 (stdlib only)."""
import json, sys
from collections import Counter, defaultdict
dm = json.load(open(sys.argv[1], encoding="utf-8")); recs = dm["records"]; probes = dm.get("probes", [])
labels = [c["label"] for c in dm["classes"]]
out = {"classes": dm["classes"], "records": len(recs), "probes": [(p["label"], p.get("probe_type"), p.get("scientific_name")) for p in probes]}
cnt = Counter((r["label"], r["split"]) for r in recs); obs = defaultdict(set)
for r in recs: obs[(r["label"], r["split"])].add(r["observer"])
out["per_species_split_images_observers"] = {l: {s: [cnt[(l, s)], len(obs[(l, s)])] for s in ("train", "validation", "test")} for l in labels}
role = defaultdict(set)
for r in recs: role[r["observer"]].add(r["split"])
out["observers_total"] = len(role); out["observers_in_multiple_roles"] = sorted(o for o, s in role.items() if len(s) > 1)
out["observers_per_role"] = dict(Counter(next(iter(s)) for s in role.values()))
for s in ("validation", "test"):
    c = Counter(r["observer"] for r in recs if r["split"] == s)
    out[f"{s}_photos_by_observer"] = dict(c.most_common())
    out[f"{s}_largest_observer_share"] = max(c.values()) / sum(c.values())
out["max_photos_per_observer_per_species"] = max(Counter((r["observer"], r["label"]) for r in recs).values())
for f in ("id", "photo_id", "observation_id", "sha256", "perceptual_hash"):
    v = [r.get(f) for r in recs + probes if r.get(f) is not None]; out[f"duplicates_{f}"] = len(v) - len(set(v))
ph = [int(r["perceptual_hash"], 16) for r in recs + probes if len(r.get("perceptual_hash", "")) == 16]
out["min_pairwise_dhash_hamming"] = min(bin(a ^ b).count("1") for i, a in enumerate(ph) for b in ph[i + 1:])
out["licences"] = dict(Counter(r["license_code"] for r in recs + probes))
out["quality_grade"] = dict(Counter(r["quality_grade"] for r in recs)); out["wild"] = dict(Counter(r["wild"] for r in recs))
out["taxon_rank_not_exact_species"] = [(r["id"], r["label"], r["observed_scientific_name"], r["taxonomic_resolution"], r["split"]) for r in recs if r["observed_taxon_id"] != r["taxon_id"]]
out["download_hosts"] = dict(Counter(r["url"].split("/")[2] for r in recs + probes if r.get("url")))
out["total_image_bytes"] = sum(r["bytes"] for r in recs + probes); out["max_image_bytes"] = max(r["bytes"] for r in recs + probes)
out["observed_on_range"] = [min(r["observed_on"] for r in recs), max(r["observed_on"] for r in recs)]
out["observed_on_by_year"] = dict(sorted(Counter(r["observed_on"][:4] for r in recs).items()))
forbidden = {"latitude", "longitude", "coordinates", "location", "geojson", "place_guess", "positional_accuracy", "locality", "private_location", "private_geojson", "gps"}
def keys(v):
    if isinstance(v, dict):
        for k, x in v.items(): yield k; yield from keys(x)
    elif isinstance(v, list):
        for x in v: yield from keys(x)
out["forbidden_location_keys_present"] = sorted({k for k in keys(dm) if k.lower() in forbidden or k.lower().startswith("private_")})
out["visual_review"] = dm.get("visual_review")
print(json.dumps(out, indent=2))
