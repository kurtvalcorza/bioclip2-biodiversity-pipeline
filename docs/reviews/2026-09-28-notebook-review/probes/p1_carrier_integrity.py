
"""Probe P1: notebook identity, carrier integrity, manifest digests (stdlib only)."""
import ast, hashlib, json, sys
from collections import Counter
raw = open(sys.argv[1], "rb").read(); nb = json.loads(raw)
cells = nb["cells"]
carrier = next(c for c in cells if c.get("metadata", {}).get("dimer", {}).get("embedded_sources"))
tree = ast.parse("".join(carrier["source"]))
files = ast.literal_eval(tree.body[0].value); hashes = ast.literal_eval(tree.body[1].value)
meta = nb["metadata"]["dimer"]
out = {"notebook_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "cells": len(cells),
       "cell_types": dict(Counter(c["cell_type"] for c in cells)),
       "executed_code_cells": sum(1 for c in cells if c["cell_type"] == "code" and c.get("execution_count")),
       "outputs_present": sum(len(c.get("outputs", [])) for c in cells if c["cell_type"] == "code"),
       "metadata": {k: meta[k] for k in ("notebook_spec", "profile", "pedagogical_mode", "release_status", "standalone")},
       "generated_from": {k: meta["generated_from"][k] for k in ("repository", "base_revision", "generator", "generator_sha256")},
       "carried_files": {}}
for name, text in files.items():
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out["carried_files"][name] = {"bytes": len(text.encode("utf-8")), "sha256_matches_CARRIED_HASHES": h == hashes[name],
                                  "sha256_matches_notebook_metadata": meta["generated_from"]["files"].get(name) == h if name in meta["generated_from"]["files"] else "not listed"}
dm = json.loads(files["data_manifest.json"])
payload = {k: v for k, v in dm.items() if k != "manifest_sha256"}
enc = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
out["data_manifest_digest_ok"] = hashlib.sha256(enc).hexdigest() == dm["manifest_sha256"]
out["exclusions_digest_ok"] = hashlib.sha256(files["exclusions.json"].encode()).hexdigest() == dm["exclusions_sha256"]
req = files["requirements.txt"]
out["lock"] = {"pinned_packages": sum(1 for l in req.splitlines() if "==" in l and not l.startswith((" ", "#"))),
               "all_hashed": all("--hash=sha256:" in blk for blk in req.split("\n\n") if "==" in blk) ,
               "key_pins": [l.split(" ")[0] for l in req.splitlines() if l.startswith(("torch==", "torchvision==", "open-clip-torch==", "transformers==", "nvidia-cuda-runtime==", "numpy==", "scikit-learn=="))]}
print(json.dumps(out, indent=2))
