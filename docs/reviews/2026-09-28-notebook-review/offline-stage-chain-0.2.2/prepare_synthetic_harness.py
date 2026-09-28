
"""Run the carried capstone.py prepare stage end to end on a SYNTHETIC photo cache.
Every canonical/probe photo is replaced by a random image with the manifest's dimensions; record
bytes, SHA-256 and dHash are rewritten to match, and the manifest digest is removed. This checks the
stage code path (validation, colour baseline, audit tables, contact sheet, refusal probes, receipts),
not the real photographs."""
import hashlib, io, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

carrier, root = Path(sys.argv[1]), Path(sys.argv[2])
shutil.rmtree(root, ignore_errors=True)
shutil.copytree(carrier, root)
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
manifest.pop("manifest_sha256")
rng = np.random.default_rng(0)
(root / "cache").mkdir()
for record in manifest["records"] + manifest["probes"]:
    if record.get("synthetic"):
        continue
    pixels = rng.integers(0, 256, size=(record["height"] // 8, record["width"] // 8, 3), dtype=np.uint8)
    image = Image.fromarray(pixels).resize((record["width"], record["height"]), Image.Resampling.NEAREST)
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=90)
    data = stream.getvalue()
    decoded = Image.open(io.BytesIO(data)).convert("RGB")
    grey = np.asarray(decoded.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = grey[:, 1:] > grey[:, :-1]
    record.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                  perceptual_hash=f"{int(''.join('1' if b else '0' for b in bits.ravel()), 2):016x}")
    (root / "cache" / f"{record['id']}.img").write_bytes(data)
(root / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
result = subprocess.run([sys.executable, "-u", str(root / "capstone.py"), "--root", str(root), "--stage", "prepare"],
                        capture_output=True, text=True)
print(result.stdout[-2000:]); print(result.stderr[-4000:])
print("exit", result.returncode)
