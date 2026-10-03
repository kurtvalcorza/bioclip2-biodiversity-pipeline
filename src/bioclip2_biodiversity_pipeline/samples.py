"""The embedded sample dataset and the labelled-image-dataset contract for species classification.

The tutorial dataset is 48 real photographs — 12 each of four North American sparrow species
(Passerellidae) — taken from research-grade iNaturalist observations whose photos carry the CC0 1.0
licence, centre-cropped and resized to 224x224, and embedded in `sample_data.py` so the standalone
notebook needs no dataset download. The four species were chosen **a priori** to make colour a
weak cue by construction: all four are streaked or grey-brown "little brown birds", so a colour
baseline has little to work with and any separation the model achieves comes from finer structure
(head pattern, breast streaking, bill colour). This is sanity evidence for the adaptation contract
on real field data, not a benchmark: 48 images, one seeded split, no dispersion estimate.

Two honest caveats, stated here and in the tutorial. The images are unfiltered beyond licence and
quality grade — some subjects are small in the frame, one is held in a hand, and backgrounds vary —
which is what field data looks like. And a random split of observations from one region can share
photographers, locations and seasons across splits; the observers are recorded so a reader can see
how many distinct sources there are, but the pipeline cannot detect that kind of leakage for you.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import ACCEPTED_FORMATS, MAX_IMAGE_SIDE, MIN_IMAGE_SIDE, decode_image, image_digest
from .sample_data import (
    SAMPLE_IMAGE_LICENSE,
    SAMPLE_IMAGE_SIDE,
    SAMPLE_IMAGES_B64,
    SAMPLE_RECORDS,
    SAMPLE_SOURCE,
)

DATASET_REPRESENTATION = "io.github.kurtvalcorza.dataset.vision.image-labels.v1"
SAMPLE_CLASSES: tuple[str, ...] = (
    "chipping_sparrow",
    "dark_eyed_junco",
    "song_sparrow",
    "white_throated_sparrow",
)
# Scientific names are what BioCLIP was trained to match; the dataset keys are what the metrics report.
SAMPLE_CLASS_PROMPTS: dict[str, str] = {
    "chipping_sparrow": "Spizella passerina",
    "dark_eyed_junco": "Junco hyemalis",
    "song_sparrow": "Melospiza melodia",
    "white_throated_sparrow": "Zonotrichia albicollis",
}
SAMPLE_COMMON_NAMES: dict[str, str] = {
    "chipping_sparrow": "Chipping Sparrow",
    "dark_eyed_junco": "Dark-eyed Junco",
    "song_sparrow": "Song Sparrow",
    "white_throated_sparrow": "White-throated Sparrow",
}
SAMPLE_SIZE = len(SAMPLE_RECORDS)  # 48: 12 per species
MIN_RECORDS = 8
MAX_RECORDS = 5_000
MAX_CLASSES = 50
MIN_RECORDS_PER_CLASS = 3
MAX_ID_CHARS = 64
MAX_LABEL_CHARS = 64
REQUIRED_COLUMNS = ("id", "image", "label")  # `image` is a file path in CSV/JSON; bytes in memory
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
# Folders an archiver adds next to the data (macOS resource forks); never a label.
IGNORED_DIRS = ("__MACOSX",)


def generate_sample_dataset() -> list[dict[str, Any]]:
    """The 48 embedded sample images as `{id, image_bytes, label, ...provenance}` records.

    Deterministic by construction (the bytes are literals); `dataset_digest` over the result is the
    tutorial's dataset identity.
    """
    records: list[dict[str, Any]] = []
    for rec in SAMPLE_RECORDS:
        file = str(rec["file"])
        records.append(
            {
                "id": file.rsplit(".", 1)[0],
                "image_bytes": base64.b64decode(SAMPLE_IMAGES_B64[file]),
                "label": str(rec["label"]),
                "file": file,
                "scientific_name": rec["scientific_name"],
                "common_name": rec["common_name"],
                "source": rec["inat_observation_url"],
                "license": rec["license_code"],
                "observer": rec["observer"],
            }
        )
    return records


def sample_provenance() -> dict[str, Any]:
    """What the sample is and where it came from, for the dataset manifest and the card."""
    observers = sorted({str(r["observer"]) for r in SAMPLE_RECORDS})
    return {
        "source": SAMPLE_SOURCE,
        "image_license": SAMPLE_IMAGE_LICENSE,
        "image_side": SAMPLE_IMAGE_SIDE,
        "n_images": len(SAMPLE_RECORDS),
        "n_observers": len(observers),
        "classes": {
            c: {"scientific_name": SAMPLE_CLASS_PROMPTS[c], "common_name": SAMPLE_COMMON_NAMES[c]}
            for c in SAMPLE_CLASSES
        },
        "location_data": "not collected",
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 over the canonical (id, image sha256, label) rows; recorded in provenance (OUT9)."""
    canon = json.dumps(
        [[r["id"], image_digest(r["image_bytes"]), r["label"]] for r in records], separators=(",", ":")
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    classes: Sequence[str] | None = None,
    min_records: int = MIN_RECORDS,
    min_per_class: int = MIN_RECORDS_PER_CLASS,
) -> dict[str, Any]:
    """Check a labelled image dataset against the contract; return its manifest.

    Every error names the record and the violated rule (VAL4/DAT19). Images are decoded — that is
    the check — and their sizes and digests recorded; nothing verifies that an image shows the
    labelled organism, and the manifest says so.
    """
    if isinstance(records, str | bytes | Mapping) or not isinstance(records, Sequence):
        raise TypeError("records must be a list of {'id', 'image_bytes', 'label'} mappings")
    if len(records) < min_records:
        raise ValueError(f"dataset has {len(records)} records; at least {min_records} are required")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"dataset has {len(records)} records; ceiling is {MAX_RECORDS}")
    seen_ids: set[str] = set()
    seen_digests: dict[str, str] = {}
    counts_by_class: dict[str, int] = {}
    widths: list[int] = []
    heights: list[int] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise TypeError(f"record[{i}] must be a mapping, got {type(rec).__name__}")
        missing = [c for c in ("id", "image_bytes", "label") if c not in rec]
        if missing:
            raise ValueError(
                f"record[{i}] is missing required field(s) {missing}; "
                "required: ['id', 'image_bytes', 'label']"
            )
        rid = str(rec["id"]).strip()
        if not rid or len(rid) > MAX_ID_CHARS:
            raise ValueError(f"record[{i}] id must be 1..{MAX_ID_CHARS} characters")
        if rid in seen_ids:
            raise ValueError(f"record[{i}] duplicates id {rid!r}")
        seen_ids.add(rid)
        data = rec["image_bytes"]
        if not isinstance(data, bytes | bytearray) or not data:
            raise ValueError(f"record[{i}] ({rid}) image_bytes must be non-empty encoded image bytes")
        try:
            im = decode_image(bytes(data))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"record[{i}] ({rid}) {exc}") from exc
        digest = image_digest(bytes(data))
        if digest in seen_digests:
            raise ValueError(f"record[{i}] ({rid}) duplicates the image of {seen_digests[digest]!r}")
        seen_digests[digest] = rid
        widths.append(im.size[0])
        heights.append(im.size[1])
        label = rec["label"]
        if not isinstance(label, str) or not label.strip() or len(label) > MAX_LABEL_CHARS:
            raise ValueError(
                f"record[{i}] ({rid}) label must be a non-empty string of at most {MAX_LABEL_CHARS} chars"
            )
        counts_by_class[label] = counts_by_class.get(label, 0) + 1
    if classes is None:
        class_list = sorted(counts_by_class)
    else:
        class_list = [str(c) for c in classes]
        unknown = sorted(set(counts_by_class) - set(class_list))
        if unknown:
            raise ValueError(f"labels {unknown} are not in the class list {class_list}")
    if len(class_list) < 2:
        raise ValueError(f"classification needs at least 2 classes, found {class_list}")
    if len(class_list) > MAX_CLASSES:
        raise ValueError(f"{len(class_list)} classes exceeds the ceiling of {MAX_CLASSES}")
    thin = [c for c in class_list if counts_by_class.get(c, 0) < min_per_class]
    if thin:
        raise ValueError(f"classes {thin} have fewer than {min_per_class} records each (class coverage rule)")
    return {
        "verdict": "accepted",
        "representation": DATASET_REPRESENTATION,
        "validation": (
            "decode, format and pixel-size checks only; "
            "nothing verifies that an image shows the labelled organism"
        ),
        "n_records": len(records),
        "classes": class_list,
        "class_counts": {c: counts_by_class.get(c, 0) for c in class_list},
        "image_width": {"min": min(widths), "max": max(widths)},
        "image_height": {"min": min(heights), "max": max(heights)},
        "ceilings": {
            "min_image_side": MIN_IMAGE_SIDE,
            "max_image_side": MAX_IMAGE_SIDE,
            "accepted_formats": list(ACCEPTED_FORMATS),
            "max_records": MAX_RECORDS,
            "max_classes": MAX_CLASSES,
            "min_records": min_records,
            "min_records_per_class": min_per_class,
        },
        "digest": dataset_digest(records),
        "findings": [],
    }


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Stratified random train/validation/test split (assumes independent images, SPL3).

    Field-image datasets are rarely independent: several photos of one individual, one
    photographer's style, one site's background. A split by observation, site or photographer is
    the right tool there; the tutorial sample keeps one photo per observation and per observer
    within a species, which is why a random split is defensible here, not because it is generally.
    """
    if not (0.0 < val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("val_fraction and test_fraction must be in (0, 1) and sum to less than 1")
    manifest = validate_dataset(records)
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = {c: [] for c in manifest["classes"]}
    for rec in records:
        by_class[rec["label"]].append(dict(rec))
    out: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    need = minimum_records_per_class(len(manifest["classes"]), val_fraction, test_fraction)
    for cls in manifest["classes"]:
        rows = by_class[cls]
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * val_fraction))
        n_test = max(1, round(len(rows) * test_fraction))
        n_train = len(rows) - n_val - n_test
        # Adaptation re-validates the training split against MIN_RECORDS_PER_CLASS, so a class
        # that leaves fewer training records than that is refused here, before any model runs.
        if n_train < MIN_RECORDS_PER_CLASS:
            raise ValueError(
                f"class {cls!r} has {len(rows)} records: the split sends {n_val} to validation and "
                f"{n_test} to test, which leaves {max(n_train, 0)} in the training split, and adaptation "
                f"needs at least {MIN_RECORDS_PER_CLASS} per class there. With "
                f"val_fraction={val_fraction} and test_fraction={test_fraction}, supply at least {need} "
                "records per class."
            )
        out["validation"].extend(rows[:n_val])
        out["test"].extend(rows[n_val : n_val + n_test])
        out["train"].extend(rows[n_val + n_test :])
    if len(out["train"]) < MIN_RECORDS:
        raise ValueError(
            f"the training split would hold {len(out['train'])} records and adaptation needs at least "
            f"{MIN_RECORDS}. With {len(manifest['classes'])} classes, val_fraction={val_fraction} and "
            f"test_fraction={test_fraction}, supply at least {need} records per class."
        )
    for part in out.values():
        rng.shuffle(part)
    return out


def minimum_records_per_class(
    n_classes: int, val_fraction: float = 0.2, test_fraction: float = 0.25
) -> int:
    """The smallest per-class count for which `split_dataset` leaves a training split that
    adaptation accepts: at least MIN_RECORDS_PER_CLASS per class and MIN_RECORDS in total.

    With the default fractions this is 5 per class for 3 or more classes and 7 per class for 2.
    """
    if n_classes < 2:
        raise ValueError("classification needs at least 2 classes")
    for n in range(MIN_RECORDS_PER_CLASS, MAX_RECORDS + 1):
        n_train = n - max(1, round(n * val_fraction)) - max(1, round(n * test_fraction))
        if n_train >= MIN_RECORDS_PER_CLASS and n_train * n_classes >= MIN_RECORDS:
            return n
    raise ValueError("no per-class count up to MAX_RECORDS satisfies the split; lower the fractions")


def _label_dirs(path: Path) -> list[Path]:
    return sorted(
        d for d in path.iterdir() if d.is_dir() and d.name not in IGNORED_DIRS and not d.name.startswith(".")
    )


def _label_root(path: Path) -> Path:
    """The folder whose sub-folders are the labels. A zip often wraps everything in one enclosing
    folder (`my_photos/<label>/<image>`); that single folder is stepped into."""
    children = _label_dirs(path)
    if len(children) == 1:
        inner = children[0]
        has_images = any(f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES for f in inner.iterdir())
        if not has_images and _label_dirs(inner):
            return inner
    return path


def load_byod_dataset(source: str | Path) -> list[dict[str, Any]]:
    """Read a user-supplied dataset: a CSV with `id,image,label` (image paths relative to the CSV),
    a JSON array / JSONL of `{id, image, label}` objects, or a directory of `<label>/<image files>`.

    Image files are read as bytes and nothing about them is rewritten (VAL7). The records are then
    validated with `validate_dataset`, whose errors name the offending row.
    """
    path = Path(source)
    records: list[dict[str, Any]] = []
    if path.is_dir():
        root = _label_root(path)
        for label_dir in _label_dirs(root):
            for img in sorted(p for p in label_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
                # Stored relative to the label root and joined to it once below, so a relative
                # upload directory (the notebook's `work/byod`) is not joined twice.
                records.append(
                    {
                        "id": f"{label_dir.name}/{img.name}",
                        "image": img.relative_to(root).as_posix(),
                        "label": label_dir.name,
                    }
                )
        if not records:
            raise ValueError(
                f"no <label>/<image> files found under {path}: the expected layout is one folder per label "
                f"holding that label's images ({', '.join(IMAGE_SUFFIXES)}), for example "
                "song_sparrow/IMG_0001.jpg, or a .csv/.json/.jsonl table of image paths"
            )
        base = root
    elif path.is_file():
        text = path.read_text(encoding="utf-8-sig")
        if not text.strip():
            raise ValueError(f"BYOD dataset file is empty: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            reader = csv.DictReader(text.splitlines())
            header = [h.strip() for h in (reader.fieldnames or [])]
            missing = [c for c in REQUIRED_COLUMNS if c not in header]
            if missing:
                raise ValueError(f"CSV header {header} is missing required column(s) {missing}")
            for row in reader:
                records.append({c: (row.get(c) or "").strip() for c in REQUIRED_COLUMNS})
        elif suffix == ".jsonl":
            for line_no, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_no} is not valid JSON: {exc}") from exc
        elif suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"file is not valid JSON: {exc}") from exc
            if not isinstance(data, list):
                raise TypeError("JSON dataset must be a top-level array of objects")
            records = data
        else:
            raise ValueError(f"unsupported BYOD file type {suffix!r}; use .csv, .json, .jsonl or a directory")
        base = path.parent
    else:
        raise FileNotFoundError(f"BYOD dataset not found: {path}")
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise TypeError(f"record[{i}] must be a mapping, got {type(rec).__name__}")
        missing = [c for c in REQUIRED_COLUMNS if c not in rec]
        if missing:
            raise ValueError(
                f"record[{i}] is missing required column(s) {missing}; required: {list(REQUIRED_COLUMNS)}"
            )
        img_path = Path(str(rec["image"]))
        if not img_path.is_absolute():
            img_path = base / img_path
        if not img_path.is_file():
            raise FileNotFoundError(f"record[{i}] ({rec['id']}) image file not found: {img_path}")
        data = img_path.read_bytes()
        _refuse_multi_picture(data, f"record[{i}] ({rec['id']}) {img_path.name}")
        out.append(
            {
                "id": str(rec["id"]),
                "image_bytes": data,
                "label": str(rec["label"]),
                "file": img_path.name,
            }
        )
    validate_dataset(out)
    return out


def _refuse_multi_picture(data: bytes, where: str) -> None:
    """Some cameras save `.jpg` files as MPO (a JPEG followed by further pictures). The decoder
    accepts JPEG, PNG and WEBP only, so name the conversion instead of a bare format error."""
    import io

    from PIL import Image, UnidentifiedImageError

    try:
        fmt = Image.open(io.BytesIO(data)).format
    except UnidentifiedImageError:
        return  # validate_dataset reports undecodable bytes with the record and the rule
    if fmt == "MPO":
        raise ValueError(
            f"{where} is an MPO file (a multi-picture JPEG that some cameras write). Re-save it as a plain "
            "JPEG, for example PIL.Image.open(path).save(new_path, 'JPEG', quality=95), and upload again."
        )


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the images to `<path's directory>/images/` and a CSV (`id,image,label`) beside them,
    so users have a BYOD template that `load_byod_dataset` reads back."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img_dir = out.parent / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(REQUIRED_COLUMNS)
        for i, r in enumerate(records):
            name = Path(str(r.get("file") or f"{r['id']}.jpg")).name
            if name.lower() in used:
                # Cameras reuse names (IMG_0001.jpg) across label folders: keep every image.
                name = f"{i:04d}_{name}"
            used.add(name.lower())
            (img_dir / name).write_bytes(r["image_bytes"])
            writer.writerow([r["id"], f"images/{name}", r["label"]])
    return out
