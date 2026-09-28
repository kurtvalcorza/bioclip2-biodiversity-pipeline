"""CPU-only contracts and statistics for the Philippine biodiversity capstone."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROLES = ("train", "validation", "test")
BYOD_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FORBIDDEN = {
    "latitude",
    "longitude",
    "coordinates",
    "location",
    "geojson",
    "place_guess",
    "positional_accuracy",
    "locality",
    "private_location",
    "private_geojson",
    "gps",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_path(root: Path, name: str) -> Path:
    if "\\" in name or ":" in name or name.startswith("/") or ".." in Path(name).parts:
        raise ValueError(f"Unsafe relative path: {name}")
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Path escapes root")
    return candidate


def check_private(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in FORBIDDEN or key.lower().startswith("private_"):
                raise ValueError(f"Precise location/private metadata forbidden: {key}")
            check_private(item)

    elif isinstance(value, list):
        for item in value:
            check_private(item)


def assign_byod_roles(records: list[dict]) -> None:
    """Assign whole observers using integer constraints; refuse infeasible grouped evaluation."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    observers = sorted({r["observer"] for r in records})
    classes = sorted({r["label"] for r in records})
    variables = len(observers) * 3
    matrix, low, high = [], [], []
    for i, _ in enumerate(observers):
        row = np.zeros(variables)
        row[3 * i : 3 * i + 3] = 1
        matrix.append(row)
        low.append(1)
        high.append(1)
    for label in classes:
        counts = Counter(r["observer"] for r in records if r["label"] == label)
        if sum(counts.values()) < 12 or len(counts) < 6:
            raise ValueError("Grouped BYOD needs >=12 photos and >=6 observers per class (two per role)")
        for role, minimum in enumerate((6, 3, 3)):
            row, groups = np.zeros(variables), np.zeros(variables)
            for i, observer in enumerate(observers):
                row[3 * i + role] = counts[observer]
                groups[3 * i + role] = int(counts[observer] > 0)
            matrix.extend([row, groups])
            low.extend([minimum, 2])
            high.extend([np.inf, np.inf])
    objective = np.array([0.0 if i % 3 == 0 else 1.0 for i in range(variables)])
    objective += np.arange(variables) / (variables * 1000)
    solved = milp(
        objective,
        integrality=np.ones(variables),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.array(matrix), low, high),
        options={"time_limit": 30},
    )
    if not solved.success:
        raise ValueError("No feasible observer-disjoint BYOD split within 30 seconds")
    assignments = {
        observer: ROLES[int(np.argmax(solved.x[3 * i : 3 * i + 3]))] for i, observer in enumerate(observers)
    }
    for record in records:
        record["split"] = assignments[record["observer"]]


def prepare_byod(
    root: Path, zip_path: str, class_prompts: dict | None = None, authorization: bool = False
) -> dict:
    """Sanitise a bounded user ZIP; missing observer groups permits inference only."""
    if not authorization:
        raise ValueError("Confirm permission to process the supplied images in this hosted runtime")
    class_prompts = class_prompts or {}
    source = Path(zip_path)
    if source.stat().st_size > 250 * 1024**2:
        raise ValueError("BYOD archive too large")
    rows, payloads = [], {}
    with zipfile.ZipFile(source) as bundle:
        members = bundle.infolist()
        if len(members) > 5001 or sum(m.file_size for m in members) > 250 * 1024**2:
            raise ValueError("BYOD expanded archive ceiling")
        for member in members:
            safe_path(root / "unpack-check", member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlink refused")
            if member.file_size > 20 * 1024**2:
                raise ValueError("ZIP member too large")
        names = [m.filename for m in members if not m.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member")
        csv_name = next((n for n in names if n == "labels.csv" or n.endswith("/labels.csv")), None)
        if csv_name is not None:
            reader = csv.DictReader(io.StringIO(bundle.read(csv_name).decode("utf-8-sig")))
            missing = {"id", "image", "label"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"labels.csv lacks required column(s) {sorted(missing)}; expected "
                    "id,image,label,scientific_name,observer_group"
                )
            prefix = csv_name[: -len("labels.csv")]
            rows = [{**row, "image": prefix + (row.get("image") or "")} for row in reader]
        else:
            image_names = [n for n in sorted(names) if Path(n).suffix.lower() in BYOD_SUFFIXES]
            tops = {Path(n).parts[0] for n in image_names}
            # Zipping a folder usually adds one top-level directory: accept <top>/<label>/<image>.
            strip = len(tops) == 1 and image_names and all(len(Path(n).parts) == 3 for n in image_names)
            for name in image_names:
                path = Path(name)
                parts = path.parts[1:] if strip else path.parts
                if len(parts) == 2:
                    label = parts[0]
                    rows.append(
                        {
                            "id": f"byod_{len(rows):04d}",
                            "image": name,
                            "label": label,
                            "scientific_name": class_prompts.get(label, ""),
                        }
                    )
        if not rows:
            raise ValueError(
                "BYOD has no images: use <label>/<image> folders (optionally inside one top-level "
                "folder) or a labels.csv with id,image,label,scientific_name,observer_group"
            )
        if len(rows) > 5000:
            raise ValueError("BYOD exceeds 5000 images")
        for row in rows:
            check_private(row)
            if row["image"] not in names:
                raise ValueError("BYOD CSV references missing archive image")
            payloads[row["image"]] = bundle.read(row["image"])
    records, classes, seen_pixels = [], {}, set()
    for index, row in enumerate(rows):
        label, scientific = row["label"], row.get("scientific_name") or class_prompts.get(row["label"])
        if not scientific or len(scientific) > 200 or len(label) > 100:
            raise ValueError("Supply a bounded scientific name for every class")
        if label in classes and classes[label]["scientific_name"] != scientific:
            raise ValueError("Inconsistent scientific names")
        classes[label] = {"label": label, "scientific_name": scientific, "common_name": label}
        with Image.open(io.BytesIO(payloads[row["image"]])) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"} or min(image.size) < 32 or max(image.size) > 8192:
                raise ValueError("BYOD image format/dimensions refused")
            clean = ImageOps.exif_transpose(image).convert("RGB")
            pixel_digest = hashlib.sha256(str(clean.size).encode() + clean.tobytes()).hexdigest()
            if pixel_digest in seen_pixels:
                raise ValueError("Duplicate BYOD pixels")
            seen_pixels.add(pixel_digest)
            output = io.BytesIO()
            clean.save(output, format="PNG")  # EXIF/GPS stripped; payload is never exported.
            data = output.getvalue()
            if len(data) > 20_000_000:
                raise ValueError("Sanitised BYOD image exceeds 20 MB")
        identity = f"byod_{index:04d}"
        target = safe_path(root / "cache", identity + ".img")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append(
            {
                "id": identity,
                "label": label,
                "scientific_name": scientific,
                "common_name": label,
                "observer": row.get("observer_group", ""),
                "split": "probe",
                "photo_id": None,
                "observation_id": None,
                "taxon_id": None,
                "observation_url": "",
                "url": "",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "width": clean.width,
                "height": clean.height,
                "license_code": "user-authorized",
                "license_url": "user-authorized",
                "attribution": "User-supplied; private runtime only",
                "observed_on": "",
                "perceptual_hash": pixel_digest,
            }
        )
    if not 2 <= len(classes) <= 20:
        raise ValueError("BYOD requires 2..20 classes")
    if sum(r["bytes"] for r in records) > 250_000_000:
        raise ValueError("Sanitised BYOD dataset exceeds 250 MB")
    inference_only = not all(r["observer"] for r in records)
    if not inference_only:
        assign_byod_roles(records)
    manifest = {
        "byod": True,
        "inference_only": inference_only,
        "classes": list(classes.values()),
        "records": records,
        "probes": [],
        "authorization_confirmed": True,
    }
    if not inference_only:
        validate_manifest(manifest, strict=False)
    write_json(root / "data_manifest.json", manifest)
    return manifest


def validate_manifest(manifest: dict, strict: bool = True) -> dict:
    """Refuse attribution, privacy, duplicate and grouped-split violations before downloads."""
    check_private(manifest)
    if "manifest_sha256" in manifest:
        payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        if hashlib.sha256(encoded).hexdigest() != manifest["manifest_sha256"]:
            raise ValueError("Dataset manifest digest mismatch")
    records = manifest["records"]
    classes = [c["label"] for c in manifest["classes"]]
    if len(classes) != len(set(classes)) or not 2 <= len(classes) <= 20:
        raise ValueError("Invalid class vocabulary")
    if strict and (len(classes) != 4 or len(records) != 56):
        raise ValueError("Canonical dataset requires four classes and 56 records")
    all_records = records + manifest.get("probes", [])
    for field in ("id", "photo_id", "observation_id", "sha256"):
        values = [r[field] for r in all_records if r.get(field) is not None]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {field}")
    observers: dict[str, str] = {}
    group_roles: dict[str, str] = {}
    counts = Counter()
    groups: dict[tuple, set] = {}
    for record in all_records:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", record["id"]):
            raise ValueError("Unsafe record id")
        if record.get("license_code") not in {"cc0", "cc-by", "user-authorized"}:
            raise ValueError("Disallowed photo licence")
        if record.get("license_code") == "user-authorized" and not manifest.get("byod"):
            raise ValueError("Canonical images need an open photo licence")
        if not record.get("attribution") or not record.get("license_url"):
            raise ValueError("Missing attribution/licence URL")
        if record.get("media_type", "image/png") not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("Unsupported image format")
        if not re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", "")):
            raise ValueError("Invalid SHA-256")
        if not 0 < record["bytes"] <= 20 * 1024**2:
            raise ValueError("Image byte ceiling")
        if min(record["width"], record["height"]) < 32 or max(record["width"], record["height"]) > 8192:
            raise ValueError("Image dimension ceiling")
        if record["split"] == "probe":
            continue
        label, role, observer = record["label"], record["split"], record.get("observer")
        if label not in classes or role not in ROLES or not observer:
            raise ValueError("Invalid label/role/observer")
        if observer in observers and observers[observer] != role:
            raise ValueError("Observer leakage across roles")
        observers[observer] = role
        duplicate_group = record.get("duplicate_group", record.get("perceptual_hash", record["sha256"]))
        if duplicate_group in group_roles and group_roles[duplicate_group] != role:
            raise ValueError("Perceptual duplicate crosses roles")
        group_roles[duplicate_group] = role
        counts[label, role] += 1
        groups.setdefault((label, role), set()).add(observer)
    for label in classes:
        for role, target in zip(ROLES, (8, 3, 3), strict=True):
            count = counts[label, role]
            if (strict and count != target) or count < 1 or len(groups.get((label, role), set())) < 2:
                raise ValueError(f"Insufficient grouped support: {label}/{role}")
    if strict and max(Counter((r["observer"], r["label"]) for r in records).values()) > 3:
        raise ValueError("Observer per-class photo cap exceeded")
    if sum(r["bytes"] for r in all_records) > 250 * 1024**2:
        raise ValueError("Total image byte ceiling")
    return {
        "records": len(records),
        "probes": len(manifest.get("probes", [])),
        "classes": classes,
        "observers": len(observers),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "observer_disjoint": True,
        "precise_locations_absent": True,
    }


def fetch_asset(root: Path, record: dict) -> Path:
    """Download a bounded manifest-listed photo; corrupt caches are refused, not silently repaired."""
    target = safe_path(root / "cache", record["id"] + ".img")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if record.get("synthetic") == "blank":
            stream = io.BytesIO()
            Image.new("RGB", (224, 224), (128, 128, 128)).save(stream, format="PNG")
            payload = stream.getvalue()
        else:
            url = urllib.parse.urlsplit(record["url"])
            if url.scheme != "https" or url.hostname not in {
                "inaturalist-open-data.s3.amazonaws.com",
                "static.inaturalist.org",
            }:
                raise ValueError("Photo download host not allowed")
            for attempt in range(3):
                try:
                    request = urllib.request.Request(
                        record["url"], headers={"User-Agent": "DIMER-Educational-Capstone/1"}
                    )
                    with urllib.request.urlopen(request, timeout=60) as response:
                        payload = response.read(record["bytes"] + 1)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Photo size/hash mismatch: {record['id']}")
        target.write_bytes(payload)
    if target.stat().st_size != record["bytes"] or sha256(target) != record["sha256"]:
        raise ValueError(f"Modified image cache: {record['id']}")
    return target


def validate_images(root: Path, manifest: dict) -> dict:
    result = validate_manifest(manifest, strict=not manifest.get("byod", False))
    pixel_hashes = set()
    perceptual_hashes = []
    for record in manifest["records"] + manifest.get("probes", []):
        path = fetch_asset(root, record)
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Unsupported image format")
            if image.size != (record["width"], record["height"]):
                raise ValueError("Image dimensions do not match manifest")
            if image.getexif().get(34853):
                raise ValueError("GPS EXIF metadata forbidden")
            pixels = ImageOps.exif_transpose(image).convert("RGB")
            digest = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
            if digest in pixel_hashes:
                raise ValueError("Duplicate decoded image pixels")
            pixel_hashes.add(digest)
            if not manifest.get("byod"):
                grey = np.asarray(pixels.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
                bits = grey[:, 1:] > grey[:, :-1]
                actual = int("".join("1" if b else "0" for b in bits.ravel()), 2)
                if f"{actual:016x}" != record["perceptual_hash"]:
                    raise ValueError("Perceptual hash mismatch")
                if any((actual ^ previous).bit_count() <= 4 for previous in perceptual_hashes):
                    raise ValueError("Near-duplicate images refused")
                perceptual_hashes.append(actual)
    return {**result, "image_hashes_verified": True, "decoded_pixel_duplicates": 0}


def checked_scores(scores: np.ndarray, n: int | None = None) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or (n is not None and values.shape[0] != n):
        raise ValueError("Scores must be N by C with C >= 2")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite scores")
    return values


def softmax(logits: np.ndarray) -> np.ndarray:
    values = checked_scores(logits)
    exponent = np.exp(values - values.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    y = np.asarray(y_true)
    values = checked_scores(scores, len(y))
    if y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Nonempty integer labels required")
    classes = values.shape[1]
    if np.any(y < 0) or np.any(y >= classes):
        raise ValueError("Labels outside vocabulary")
    order = np.argsort(-values, axis=1, kind="stable")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (y, order[:, 0]), 1)
    per_class = []
    for i in range(classes):
        tp, support, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class.append(
            dict(
                class_index=i, precision=precision, recall=recall, f1=f1, support=support, predicted=predicted
            )
        )
    return {
        "accuracy": float(np.mean(order[:, 0] == y)),
        "macro_f1": float(np.mean([p["f1"] for p in per_class])),
        "top2_accuracy": float(np.mean(np.any(order[:, :2] == y[:, None], axis=1))),
        "confusion": matrix.tolist(),
        "per_class": per_class,
    }


def bootstrap(
    y: np.ndarray, scores: np.ndarray, groups=None, n_boot: int = 2000, stratify: bool = False
) -> dict:
    """Percentile intervals over test records.

    `stratify=True` resamples records within each reference class, so every resample keeps the
    per-class support of the test set. Macro-F1 averages over the fixed class vocabulary and scores
    an absent class as 0, so unstratified record or cluster resampling of a tiny balanced test set
    mixes "a species went missing from the resample" into the interval. The canonical capstone
    therefore uses the stratified form. `groups` (whole-cluster resampling) remains available for
    diagnostics but is not interpretable with only a handful of clusters.
    """
    y = np.asarray(y)
    scores = checked_scores(scores, len(y))
    if len(y) == 0 or n_boot < 2:
        raise ValueError("Bootstrap needs observations and >=2 draws")
    if stratify and groups is not None:
        raise ValueError("Choose stratified record resampling or cluster resampling, not both")
    rng = np.random.default_rng(42)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError("Bootstrap groups misaligned")
    unique = np.unique(groups)
    strata = [np.flatnonzero(y == c) for c in np.unique(y)]
    values = []
    for _ in range(n_boot):
        if stratify:
            index = np.concatenate([rng.choice(members, len(members)) for members in strata])
        else:
            index = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique))])
        result = metrics(y[index], scores[index])
        values.append([result["accuracy"], result["macro_f1"]])
    low, high = np.quantile(values, [0.025, 0.975], axis=0)
    return {
        "accuracy_low": float(low[0]),
        "accuracy_high": float(high[0]),
        "macro_f1_low": float(low[1]),
        "macro_f1_high": float(high[1]),
    }


def knn_scores(trainfeatures: np.ndarray, labels: np.ndarray, query: np.ndarray, k: int = 5) -> np.ndarray:
    train, query, labels = np.asarray(trainfeatures), np.asarray(query), np.asarray(labels)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1] or len(labels) != len(train):
        raise ValueError("kNN shape mismatch")
    if not np.isfinite(train).all() or not np.isfinite(query).all() or not 1 <= k <= len(train):
        raise ValueError("kNN nonfinite features/invalid k")
    if set(labels.tolist()) != set(range(int(labels.max()) + 1)):
        raise ValueError("kNN class support missing")
    train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    similarity = query @ train.T
    nearest = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    scores = np.zeros((len(query), int(labels.max()) + 1), dtype=np.float64)
    for row, indices in enumerate(nearest):
        np.add.at(scores[row], labels[indices], np.maximum(similarity[row, indices], 0))
    return scores


def colour_features(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (224, 224), method=Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float64) / 255
    return np.concatenate([pixels.mean((0, 1)), pixels.std((0, 1))])


def risk_rows(y: np.ndarray, scores: np.ndarray, thresholds=None) -> list[dict]:
    y = np.asarray(y)
    scores = checked_scores(scores, len(y))
    if not len(y):
        raise ValueError("Empty risk set")
    if np.any(scores < 0) or not np.allclose(scores.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Referral requires normalised scores")
    order = np.argsort(-scores, axis=1, kind="stable")
    margin = (
        np.take_along_axis(scores, order[:, :1], axis=1)[:, 0]
        - np.take_along_axis(scores, order[:, 1:2], axis=1)[:, 0]
    )
    if thresholds is None:
        thresholds = sorted(set([0.0, *margin.tolist(), float(np.nextafter(margin.max(), np.inf))]))
    rows = []
    for threshold in thresholds:
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid threshold")
        accepted = margin >= threshold
        count = int(accepted.sum())
        correct = int(np.sum((order[:, 0] == y) & accepted))
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": count / len(y),
                "count": count,
                "selective_accuracy": correct / count if count else None,
                "incorrect_accepted": count - correct,
                "referred": len(y) - count,
            }
        )
    return rows


def select_threshold(y: np.ndarray, scores: np.ndarray) -> dict:
    choices = [r for r in risk_rows(y, scores) if r["coverage"] >= 0.5]
    qualified = [r for r in choices if r["selective_accuracy"] >= 0.8]
    if qualified:
        return max(qualified, key=lambda r: (r["coverage"], r["threshold"]))
    return max(choices, key=lambda r: (r["selective_accuracy"], r["coverage"], r["threshold"]))


def parity(
    original: np.ndarray, reloaded: np.ndarray, threshold: float, atol: float = 1e-6, rtol: float = 1e-5
) -> dict:
    original, reloaded = checked_scores(original), checked_scores(reloaded)
    if original.shape != reloaded.shape or not np.allclose(original, reloaded, atol=atol, rtol=rtol):
        raise ValueError("Reload score parity failed")
    a, b = np.argsort(-original, axis=1, kind="stable"), np.argsort(-reloaded, axis=1, kind="stable")
    if not np.array_equal(a[:, 0], b[:, 0]):
        raise ValueError("Reload labels changed")
    aa, bb = np.take_along_axis(original, a, axis=1), np.take_along_axis(reloaded, b, axis=1)
    # Require entire stable ranking: stricter than the minimum top-2 requirement.
    if not np.array_equal(a[:, :2], b[:, :2]):
        raise ValueError("Reload top-2 order changed")
    if not np.array_equal(aa[:, 0] - aa[:, 1] >= threshold, bb[:, 0] - bb[:, 1] >= threshold):
        raise ValueError("Reload referral decisions changed")
    return {
        "score_allclose": True,
        "identical_labels": True,
        "identical_referrals": True,
        "max_absolute_error": float(np.max(np.abs(original - reloaded))),
        "atol": atol,
        "rtol": rtol,
    }
