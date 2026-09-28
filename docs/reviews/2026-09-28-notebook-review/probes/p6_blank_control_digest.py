
"""Probe P6: can the synthetic blank control's manifest digest be reproduced without Pillow?
fetch_asset() encodes Image.new('RGB',(224,224),(128,128,128)) as PNG and requires the exact
manifest bytes/SHA-256. This builds the same PNG with stdlib zlib (level 6, per-row adaptive
filter choice for a constant image) and reports which encoder settings match."""
import hashlib, json, struct, sys, zlib
dm = json.load(open(sys.argv[1], encoding="utf-8"))
blank = next(p for p in dm["probes"] if p.get("synthetic") == "blank")
W = H = 224; px = bytes([128, 128, 128]) * W
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
def rows(strategy):
    if strategy == "none": return b"".join(b"\x00" + px for _ in range(H))
    return b"\x01" + px[:3] + bytes(len(px) - 3) + b"".join(b"\x02" + bytes(len(px)) for _ in range(H - 1))
results = []
for strategy in ("none", "adaptive(sub,up...)"):
    for level in (1, 6, 9):
        c = zlib.compressobj(level, zlib.DEFLATED, 15, 8, 0)
        data = c.compress(rows(strategy)) + c.flush()
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + chunk(b"IDAT", data) + chunk(b"IEND", b"")
        results.append({"filter": strategy, "level": level, "bytes": len(png), "matches_manifest": hashlib.sha256(png).hexdigest() == blank["sha256"]})
print(json.dumps({"manifest_bytes": blank["bytes"], "manifest_sha256": blank["sha256"],
                  "python_zlib_runtime": zlib.ZLIB_RUNTIME_VERSION, "results": results}, indent=2))
