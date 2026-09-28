"""Standalone Philippine bird capstone stages; model imports are deliberately lazy."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

import biodiversity_core as core

# Stage processes render to files only. A notebook kernel exports its own inline backend
# (Colab: MPLBACKEND=module://matplotlib_inline.backend_inline), which does not exist in the
# isolated environment and makes `import matplotlib` itself fail, so override it first.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")

STAGES = (
    "prepare",
    "siglip-zero-shot",
    "bioclip-zero-shot",
    "representations",
    "adapt-head",
    "evaluate",
    "triage",
    "reload",
    "report",
)
BIO_ID = "imageomics/bioclip-2"
BIO_REV = "2957b322090f9cb17ae72c71981c7218a28d81e0"
CONFIG = {
    "seed": 42,
    "epochs": 20,
    # Predeclared grid: the rate is selected together with the epoch by minimum validation
    # cross-entropy, before any test result is computed. The head starts at logit_scale x text
    # (about 3.6 per weight), so 80 AdamW steps at 1e-4 alone cannot move a logit by more than
    # about 0.2; a single tiny rate would make "the head did not beat zero-shot" a foregone result.
    "learning_rates": [0.0001, 0.001, 0.01],
    "weight_decay": 0.01,
    "batch_size": 8,
    "precision": "float32",
    "k": 5,
    "bootstrap": 2000,
    "atol": 1e-6,
    "rtol": 1e-5,
}
NOTEBOOK_REVISION = "0.2.1-candidate"
# Transformers SigLIP 2 guidance: lowercased label text, the pipeline template and fixed
# padding to 64 tokens, which is how the model was trained.
SIGLIP_TEMPLATE = "This is a photo of {}."
SIGLIP_TEXT_KWARGS = {"padding": "max_length", "max_length": 64}
# Okabe-Ito qualitative palette (colour-blind safe).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
# Systems whose scores are one-hot by construction; top-2 accuracy is undefined for them.
UNSCORED_SYSTEMS = {"majority"}


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError("Cannot export an empty table")
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows
        )


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def context(root):
    manifest = core.read_json(root / "data_manifest.json")
    classes = [c["label"] for c in manifest["classes"]]
    records = manifest["records"] + manifest.get("probes", [])
    return manifest, classes, records


def indexes(records, role):
    return np.array([i for i, r in enumerate(records) if r["split"] == role], dtype=int)


def truths(records, classes, indices):
    return np.array([classes.index(records[i]["label"]) for i in indices], dtype=int)


def images(root, records):
    from PIL import Image, ImageOps

    result = []
    for r in records:
        path = core.fetch_asset(root, r)
        with Image.open(path) as image:
            result.append(ImageOps.exif_transpose(image).convert("RGB").copy())
    return result


def figure(root, name, fig):
    import matplotlib.pyplot as plt

    destination = root / "outputs" / "figures" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=130, bbox_inches="tight")
    plt.close(fig)


def centre_crop_box(width, height):
    """BioCLIP's eval transform: shortest-side resize, then a centred square crop."""
    side = min(width, height)
    return (width - side) / 2, (height - side) / 2, side


def contact(root, records, name, annotations=None, crop_boxes=False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    selected = records[:12]
    fig, axes = plt.subplots(
        (len(selected) + 2) // 3, 3, figsize=(15, 4.5 * ((len(selected) + 2) // 3)), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for j, (ax, r, im) in enumerate(zip(axes.flat, selected, images(root, selected), strict=False)):
        ax.imshow(im)
        if crop_boxes:
            x, y, side = centre_crop_box(*im.size)
            box = Rectangle((x, y), side, side, fill=False, edgecolor=PALETTE[1], linewidth=2, linestyle="--")
            ax.add_patch(box)
        title = f"{r['id']} · {r.get('common_name', r['label'])}"
        if annotations:
            title += "\n" + annotations[j]
        title += f"\n{r.get('attribution', 'Synthetic control')} · {r.get('license_code', 'CC0')}"
        ax.set_title(title, fontsize=8, wrap=True)
    if crop_boxes:
        fig.suptitle(
            "Dashed box: region BioCLIP sees after its centre crop. "
            "SigLIP resizes the whole frame to 224 x 224.",
            fontsize=10,
        )
    figure(root, name, fig)


def confusion_figure(root, y, scores, classes, name, title):
    import matplotlib.pyplot as plt

    matrix = np.array(core.metrics(y, scores)["confusion"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted species",
        ylabel="Reference species",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            pct = matrix[i, j] / max(1, matrix[i].sum())
            ax.text(j, i, f"{matrix[i, j]}\n{pct:.0%}", ha="center", va="center", color="black")
    figure(root, name, fig)


def simple_scores(root, records, classes):
    train = indexes(records, "train")
    labels = truths(records, classes, train)
    features = np.stack([core.colour_features(core.fetch_asset(root, r)) for r in records])
    centroids = np.stack([features[train][labels == i].mean(0) for i in range(len(classes))])
    distances = ((features[:, None] - centroids[None]) ** 2).sum(2)
    colour = core.softmax(-distances)
    counts = np.bincount(labels, minlength=len(classes))
    winner = sorted(range(len(classes)), key=lambda i: (-counts[i], classes[i]))[0]
    majority = np.zeros((len(records), len(classes)))
    majority[:, winner] = 1
    return {"majority": majority, "colour": colour}


def prepare(root):
    manifest, classes, records = context(root)
    checks = core.validate_manifest(manifest, strict=not manifest.get("byod", False))
    if not manifest.get("byod") and core.sha256(root / "exclusions.json") != manifest["exclusions_sha256"]:
        raise ValueError("Dataset exclusions digest mismatch")
    cache = root / "cache"
    cached_before = {p.name for p in cache.glob("*.img")} if cache.exists() else set()
    started = time.monotonic()
    image_checks = core.validate_images(root, manifest)
    fetch_seconds = time.monotonic() - started
    out = root / "outputs"
    fetched = [r for r in records if f"{r['id']}.img" not in cached_before]
    core.write_json(
        out / "download_dataset.json",
        {
            "downloaded_records": len(fetched),
            "downloaded_bytes": sum(r["bytes"] for r in fetched),
            "seconds": fetch_seconds,
            "note": "fetch plus hash, decode, dHash and EXIF validation of every manifest image",
        },
    )
    for filename in ("data_manifest.json", "model_manifest.json"):
        shutil.copyfile(root / filename, out / filename)
    if not manifest.get("byod"):
        shutil.copyfile(root / "exclusions.json", out / "exclusions.json")
    rows = []
    for c in classes:
        for split in ("train", "validation", "test"):
            selected = [r for r in records if r["label"] == c and r["split"] == split]
            rows.append(
                {
                    "label": c,
                    "split": split,
                    "images": len(selected),
                    "observers": len({r["observer"] for r in selected}),
                }
            )
    write_csv(out / "dataset_summary.csv", rows)
    write_csv(out / "observer_roles.csv", observer_role_rows(records))
    write_csv(out / "dataset_audit.csv", audit_rows(manifest, records, checks))
    write_csv(
        out / "split_manifest.csv",
        [{k: r.get(k) for k in ("id", "label", "observer", "split")} for r in records],
    )
    write_csv(
        out / "attribution.csv",
        [
            {k: r.get(k) for k in ("id", "attribution", "observation_url", "license_code", "license_url")}
            for r in records
        ],
    )
    scores = simple_scores(root, records, classes)
    np.savez(out / "baseline_scores.npz", **scores)
    val = indexes(records, "validation")
    y = truths(records, classes, val)
    write_csv(
        out / "validation_baselines.csv",
        [{"system": name, **scalar_metrics(y, s[val], name)} for name, s in scores.items()],
    )
    confusion_figure(
        root,
        y,
        scores["colour"][val],
        classes,
        "baseline_validation.png",
        "Colour baseline · validation only",
    )
    selected = [
        r for c in classes for r in [r for r in records if r["split"] == "train" and r["label"] == c][:3]
    ]
    contact(root, selected, "contact_train.png", crop_boxes=True)
    if (root / "licenses").exists():
        shutil.copytree(root / "licenses", out / "licenses", dirs_exist_ok=True)
    core.write_json(
        out / "licenses" / "model_licenses.json",
        {
            "bioclip": {"license": "MIT", "source": f"https://huggingface.co/{BIO_ID}/tree/{BIO_REV}"},
            "siglip": {
                "license": "Apache-2.0",
                "source": "https://huggingface.co/google/siglip2-base-patch16-224",
            },
            "photos": "Per-photo attribution and canonical licence links are in attribution.csv.",
        },
    )
    refusals = refusal_probes(root, manifest)
    write_csv(out / "refusal_probe_results.csv", refusals)
    core.write_json(
        out / "prepare.json",
        {
            "dataset": checks,
            "images": image_checks,
            "source_image_bytes": sum(r["bytes"] for r in records),
            "status": "Candidate",
            "refusal_probes_passed": len(refusals),
        },
    )


def observer_role_rows(records):
    """Whole-sample observer concentration per role: grouped splits can still be dominated by one person."""
    from collections import Counter

    rows = []
    for role in core.ROLES:
        counts = Counter(r["observer"] for r in records if r["split"] == role)
        if not counts:
            continue
        top = max(counts.values())
        rows.append(
            {
                "role": role,
                "images": sum(counts.values()),
                "observers": len(counts),
                "largest_observer_images": top,
                "largest_observer_share": top / sum(counts.values()),
                "species_covered_by_largest_observer": len(
                    {r["label"] for r in records if r["split"] == role and counts[r["observer"]] == top}
                ),
            }
        )
    return rows


def audit_rows(manifest, records, checks):
    """Flat, learner-readable audit of the frozen sample (every value also lives in data_manifest.json)."""
    from collections import Counter

    canonical = [r for r in records if r["split"] != "probe"]
    dated = sorted(r["observed_on"] for r in canonical if r.get("observed_on"))
    licences = Counter(r.get("license_code") for r in records)
    rows = [
        ("canonical photographs", len(canonical)),
        ("probe images (never scored)", len(records) - len(canonical)),
        ("species", len(manifest["classes"])),
        ("observers (canonical)", checks["observers"]),
        ("split counts", "/".join(str(checks["split_counts"].get(r, 0)) for r in core.ROLES)),
        ("observation dates", f"{dated[0]} to {dated[-1]}" if dated else "not recorded"),
        ("width range (px)", f"{min(r['width'] for r in records)} to {max(r['width'] for r in records)}"),
        ("height range (px)", f"{min(r['height'] for r in records)} to {max(r['height'] for r in records)}"),
        ("photo licences", "; ".join(f"{k}: {v}" for k, v in sorted(licences.items()))),
        ("precise location fields in the manifest", "none (refused if present)"),
        ("GPS EXIF in downloaded images", "none (refused if present)"),
        ("duplicate photo/observation IDs or bytes", "none (refused if present)"),
        ("near-duplicate images (dHash distance <= 4)", "none (refused if present)"),
    ]
    finer = [r for r in canonical if r.get("taxonomic_resolution", "exact_species") != "exact_species"]
    for r in finer:
        rows.append(
            (
                f"identified below species: {r['id']}",
                f"{r.get('observed_scientific_name')} kept as {r['label']} ({r.get('taxonomic_resolution')})",
            )
        )
    if dated:
        # BioCLIP 2's preprint appeared in May 2025; photographs observed afterwards cannot be in
        # its training data. This is descriptive only and does not resolve overlap for the rest.
        after = sum(d >= "2025-06" for d in dated)
        rows.append(("canonical photographs observed after 2025-05 (BioCLIP 2 release)", after))
    review = manifest.get("visual_review") or {}
    for item in review.get("crop_limitations", []):
        rows.append((f"crop limitation, observation {item.get('observation_id')}", item.get("note", "")))
    return [{"check": key, "value": value} for key, value in rows]


def scalar_metrics(y, scores, system=None):
    result = {k: v for k, v in core.metrics(y, scores).items() if k not in ("confusion", "per_class")}
    if system in UNSCORED_SYSTEMS:
        result["top2_accuracy"] = None  # one-hot scores: the "second choice" is only a tie-break
    return result


def refusal_probes(root, manifest):
    """Exercise real validators without editing canonical records or image bytes."""
    rows = []
    mutations = {
        "disallowed licence": lambda m: m["records"][0].update(license_code="all-rights-reserved"),
        "missing attribution": lambda m: m["records"][0].update(attribution=""),
        "duplicate photo": lambda m: m["records"][1].update(photo_id=m["records"][0]["photo_id"]),
        "duplicate observation": lambda m: m["records"][1].update(
            observation_id=m["records"][0]["observation_id"]
        ),
        "duplicate bytes": lambda m: m["records"][1].update(sha256=m["records"][0]["sha256"]),
        "coordinates": lambda m: m["records"][0].update(latitude=0),
        "too small": lambda m: m["records"][0].update(width=1),
        "too large": lambda m: m["records"][0].update(height=9000),
        "unsupported format": lambda m: m["records"][0].update(media_type="image/gif"),
        "missing test class": lambda m: m["records"].__setitem__(
            slice(None),
            [
                r
                for r in m["records"]
                if not (r["split"] == "test" and r["label"] == m["classes"][0]["label"])
            ],
        ),
    }
    if manifest.get("byod"):
        mutations.pop("duplicate photo")
        mutations.pop("duplicate observation")
    for name, mutation in mutations.items():
        altered = copy.deepcopy(manifest)
        altered.pop("manifest_sha256", None)  # Exercise the specific contract beneath the digest guard.
        mutation(altered)
        try:
            core.validate_manifest(altered, strict=not manifest.get("byod", False))
        except (ValueError, KeyError) as exc:
            rows.append({"probe": name, "refused": True, "reason": str(exc)})
        else:
            raise ValueError(f"Refusal probe unexpectedly accepted: {name}")
    altered = copy.deepcopy(manifest)
    altered.pop("manifest_sha256", None)
    train = next(r for r in altered["records"] if r["split"] == "train")
    next(r for r in altered["records"] if r["split"] == "test")["observer"] = train["observer"]
    tests = [
        ("observer leakage", lambda: core.validate_manifest(altered, strict=not manifest.get("byod", False))),
        ("path traversal", lambda: core.safe_path(root, "../outside")),
    ]
    for field, value in (("bytes", manifest["records"][0]["bytes"] + 1), ("sha256", "0" * 64)):
        row = {**manifest["records"][0], field: value}
        tests.append((f"changed {field}", lambda row=row: core.fetch_asset(root, row)))
    for name, check in tests:
        try:
            check()
        except ValueError as exc:
            rows.append({"probe": name, "refused": True, "reason": str(exc)})
        else:
            raise ValueError(f"Refusal probe unexpectedly accepted: {name}")
    return rows


def stage_model(root, key, record_download=True):
    """Download only immutable declared snapshot files, verify before model construction."""
    from huggingface_hub import hf_hub_download

    m = core.read_json(root / "model_manifest.json")[key]
    expected = {
        "bioclip": (BIO_ID, BIO_REV),
        "siglip": ("google/siglip2-base-patch16-224", "5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2"),
    }
    if (m["modelId"], m["revision"]) != expected[key]:
        raise ValueError("Model identity does not match the predeclared comparison")
    folder = root / "weights" / key
    folder.mkdir(parents=True, exist_ok=True)
    started, downloaded = time.monotonic(), 0
    for entry in m["files"]:
        path = core.safe_path(folder, entry["path"])
        if not path.exists():
            hf_hub_download(m["modelId"], entry["path"], revision=m["revision"], local_dir=str(folder))
            downloaded += entry["bytes"]
        if path.stat().st_size != entry["bytes"] or core.sha256(path) != entry["sha256"]:
            raise ValueError(f"Model snapshot mismatch: {entry['path']}")
    core.write_json(folder / "dimer-base-manifest.json", m)
    if record_download:
        core.write_json(
            root / "outputs" / f"download_{key}.json",
            {
                "downloaded_bytes": downloaded,
                "seconds": time.monotonic() - started,
                "model_id": m["modelId"],
                "revision": m["revision"],
            },
        )
    return folder


def torch_runtime():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This model stage requires the Colab T4 GPU runtime.")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    return torch


def siglip_zero_shot(root):
    torch = torch_runtime()
    from transformers import AutoModel, AutoProcessor

    manifest, classes, records = context(root)
    folder = stage_model(root, "siglip")
    model = (
        AutoModel.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        .float()
        .cuda()
        .eval()
    )
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    data, used = {}, {}
    for condition, field in (("common", "common_name"), ("scientific", "scientific_name")):
        prompts = siglip_prompts(manifest["classes"], field)
        used[condition] = prompts
        scores = []
        for start in range(0, len(records), 8):
            inputs = processor(
                text=prompts,
                images=images(root, records[start : start + 8]),
                return_tensors="pt",
                **SIGLIP_TEXT_KWARGS,
            ).to("cuda")
            if tuple(inputs["input_ids"].shape) != (len(prompts), SIGLIP_TEXT_KWARGS["max_length"]):
                raise ValueError("SigLIP text was not padded to its documented 64-token length")
            with torch.no_grad():
                output = model(**inputs)
                scores.extend(torch.sigmoid(output.logits_per_image).float().cpu().numpy())
        data[f"siglip_{condition}"] = np.asarray(scores)
    np.savez(root / "outputs" / "siglip_scores.npz", **data)
    core.write_json(
        root / "outputs" / "siglip_preprocessing.json",
        {
            "template": SIGLIP_TEMPLATE,
            "label_text_lowercased": True,
            **SIGLIP_TEXT_KWARGS,
            "score": "sigmoid(logits_per_image); not normalised across candidates",
            "prompts": used,
            "source": "Transformers SigLIP 2 usage notes (lowercased text, padding to 64 tokens)",
        },
    )
    val = indexes(records, "validation")
    y = truths(records, classes, val)
    write_csv(
        root / "outputs" / "validation_siglip.csv",
        [{"system": k, **scalar_metrics(y, s[val], k)} for k, s in data.items()],
    )
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def siglip_prompts(classes, field):
    """Documented SigLIP 2 convention: the pipeline template with lowercased label text."""
    return [SIGLIP_TEMPLATE.format(c[field].lower()) for c in classes]


def bioclip_pipe(root, record_download=True):
    torch_runtime()
    from bioclip2_biodiversity_pipeline.pipeline import BioClip2Pipeline

    return BioClip2Pipeline.from_pretrained(
        device="cuda", weights_dir=stage_model(root, "bioclip", record_download), allow_download=False
    )


def bioclip_zero_shot(root):
    manifest, classes, records = context(root)
    pipe = bioclip_pipe(root)
    vectors = []
    for start in range(0, len(records), 8):
        vectors.extend(pipe.embed_images(images(root, records[start : start + 8]))["embeddings"])
    features = np.asarray(vectors, dtype=np.float32)
    # Determinism is checked with the same batch composition as the original pass; a different
    # batch size may legitimately select different GPU kernels, so that difference is recorded
    # descriptively instead of being asserted at float32 noise level.
    first = records[:8]
    repeated = np.asarray(pipe.embed_images(images(root, first))["embeddings"], dtype=np.float32)
    repeat_error = float(np.max(np.abs(features[: len(first)] - repeated)))
    if not np.allclose(features[: len(first)], repeated, atol=CONFIG["atol"], rtol=CONFIG["rtol"]):
        raise ValueError("BioCLIP repeated embedding parity failed")
    single = np.asarray(pipe.embed_images(images(root, records[:1]))["embeddings"], dtype=np.float32)
    cross_batch_error = float(np.max(np.abs(features[:1] - single)))
    # The fresh-process reload probe is embedded here as one batch so the reload stage can
    # repeat exactly the same computation.
    probe_ix = indexes(records, "test")[:2]
    probe_features = np.asarray(
        pipe.embed_images(images(root, [records[i] for i in probe_ix]))["embeddings"], dtype=np.float32
    )
    np.save(root / "outputs" / "reload_probe_features.npy", probe_features)
    if not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5):
        raise ValueError("BioCLIP embedding norms drifted")
    data = {}
    for condition, field in (("common", "common_name"), ("scientific", "scientific_name")):
        prompts = [f"a photo of {c[field]}." for c in manifest["classes"]]
        text = np.asarray(pipe.embed_texts(prompts)["embeddings"], dtype=np.float32)
        data[f"bioclip_{condition}"] = core.softmax(pipe.logit_scale * features @ text.T)
        if condition == "scientific":
            np.save(root / "outputs" / "text_embeddings.npy", text)
    np.save(root / "outputs" / "image_embeddings.npy", features)
    np.savez(root / "outputs" / "bioclip_scores.npz", **data)
    core.write_json(
        root / "outputs" / "embedding_identity.json",
        {
            "ids": [r["id"] for r in records],
            "logit_scale": pipe.logit_scale,
            "repeat_verified": True,
            "repeat_same_batch_max_abs_error": repeat_error,
            "cross_batch_size_max_abs_error": cross_batch_error,
            "reload_probe_ids": [records[i]["id"] for i in probe_ix],
            "dimension": features.shape[1],
        },
    )
    val = indexes(records, "validation")
    write_csv(
        root / "outputs" / "validation_bioclip.csv",
        [{"system": k, **scalar_metrics(truths(records, classes, val), s[val], k)} for k, s in data.items()],
    )
    all_zero = dict(data)
    with np.load(root / "outputs" / "siglip_scores.npz", allow_pickle=False) as saved:
        all_zero.update({k: saved[k] for k in saved.files})
    y = truths(records, classes, val)
    write_csv(
        root / "outputs" / "validation_per_class.csv",
        [
            {"system": system, "label": classes[i], **row}
            for system, values in all_zero.items()
            for i, row in enumerate(core.metrics(y, values[val])["per_class"])
        ],
    )
    confusion_figure(
        root,
        y,
        data["bioclip_scientific"][val],
        classes,
        "zero_shot_validation.png",
        "BioCLIP scientific names · validation only",
    )


def byod_infer(root):
    manifest, classes, records = context(root)
    if not manifest.get("byod") or not manifest.get("inference_only"):
        raise ValueError("BYOD inference stage requires the explicit inference-only manifest")
    pipe = bioclip_pipe(root)
    rows = []
    for condition, field in (("common", "common_name"), ("scientific", "scientific_name")):
        prompts = {c["label"]: c.get(field, c["scientific_name"]) for c in manifest["classes"]}
        for start in range(0, len(records), 8):
            batch = records[start : start + 8]
            result = pipe.zero_shot(images(root, batch), prompts, names=[r["id"] for r in batch])
            for p in result["predictions"]:
                rows.append(
                    {
                        "id": p["id"],
                        "condition": condition,
                        "prediction": p["label"],
                        "scores": p["scores"],
                        "heldout_evaluation": False,
                    }
                )
    write_csv(root / "outputs" / "byod_predictions.csv", rows)


def representations(root):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    _, classes, records = context(root)
    features = np.load(root / "outputs" / "image_embeddings.npy", allow_pickle=False)
    train, val = indexes(records, "train"), indexes(records, "validation")
    scores = core.knn_scores(features[train], truths(records, classes, train), features, k=5)
    np.save(root / "outputs" / "knn_scores.npy", scores)
    write_csv(
        root / "outputs" / "validation_knn.csv",
        [{"system": "bioclip_5nn", **scalar_metrics(truths(records, classes, val), scores[val])}],
    )
    projection = PCA(n_components=2, svd_solver="full").fit(features[train]).transform(features)
    fig, ax = plt.subplots(figsize=(9, 6))
    for split, marker in (("train", "o"), ("validation", "^"), ("test", "s")):
        ix = indexes(records, split)
        for c, label in enumerate(classes):
            chosen = [i for i in ix if records[i]["label"] == label]
            ax.scatter(
                projection[chosen, 0],
                projection[chosen, 1],
                marker=marker,
                color=PALETTE[c % len(PALETTE)],
                label=f"{label} · {split}",
                alpha=0.8,
            )
    ax.set(
        xlabel="Principal component 1",
        ylabel="Principal component 2",
        title="Train-fitted PCA · descriptive visualisation only",
    )
    ax.legend(fontsize=7, bbox_to_anchor=(1, 1))
    figure(root, "pca.png", fig)


def fit_head(train_x, train_y, val_x, val_y, text, scale, epochs=20, learning_rates=None):
    """CPU-testable head fitting with validation-only selection of (learning rate, epoch).

    Every rate in the predeclared grid starts from the same zero-shot initialisation and seed.
    Epoch zero (the zero-shot classifier itself) competes for every rate. The selected state has
    the lowest validation cross-entropy; ties keep the earlier epoch, then the smaller rate.
    """
    import torch

    rates = list(learning_rates or CONFIG["learning_rates"])
    train_x, val_x = torch.tensor(train_x, dtype=torch.float32), torch.tensor(val_x, dtype=torch.float32)
    train_y, val_y = torch.tensor(train_y, dtype=torch.long), torch.tensor(val_y, dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss()
    history, best_key, best = [], None, None
    for rate_index, rate in enumerate(rates):
        torch.manual_seed(CONFIG["seed"])
        head = torch.nn.Linear(train_x.shape[1], len(text))
        with torch.no_grad():
            head.weight.copy_(torch.tensor(text, dtype=torch.float32) * scale)
            head.bias.zero_()
        optimizer = torch.optim.AdamW(head.parameters(), lr=rate, weight_decay=CONFIG["weight_decay"])
        for epoch in range(epochs + 1):
            if epoch:
                for ix in torch.randperm(len(train_y)).split(CONFIG["batch_size"]):
                    optimizer.zero_grad()
                    loss_fn(head(train_x[ix]), train_y[ix]).backward()
                    optimizer.step()
            with torch.no_grad():
                train_loss = float(loss_fn(head(train_x), train_y))
                val_loss = float(loss_fn(head(val_x), val_y))
            history.append(
                {"learning_rate": rate, "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss}
            )
            key = (val_loss, epoch, rate_index)
            if best_key is None or key < best_key:
                best_key = key
                best = {"head." + k: v.detach().clone() for k, v in head.state_dict().items()}
    selection = {"learning_rate": rates[best_key[2]], "epoch": best_key[1], "validation_loss": best_key[0]}
    return best, history, selection


def head_change(features, tensors, text, scale):
    """How far the selected head moved from its zero-shot initialisation, on the given features."""
    weight, bias = (
        np.asarray(tensors[k].detach().cpu().numpy() if hasattr(tensors[k], "detach") else tensors[k])
        for k in ("head.weight", "head.bias")
    )
    initial = np.asarray(text, dtype=np.float32) * scale
    head_logits = np.asarray(features, dtype=np.float32) @ weight.T + bias
    zero_shot_logits = np.asarray(features, dtype=np.float32) @ initial.T
    return {
        "max_abs_weight_change": float(np.max(np.abs(weight - initial))),
        "max_abs_bias": float(np.max(np.abs(bias))),
        "max_abs_logit_change": float(np.max(np.abs(head_logits - zero_shot_logits))),
        "predictions_changed_vs_zero_shot": int(np.sum(head_logits.argmax(1) != zero_shot_logits.argmax(1))),
        "records_compared": len(head_logits),
    }


def head_scores(features, tensors):
    weight = (
        tensors["head.weight"].detach().cpu().numpy()
        if hasattr(tensors["head.weight"], "detach")
        else tensors["head.weight"]
    )
    bias = (
        tensors["head.bias"].detach().cpu().numpy()
        if hasattr(tensors["head.bias"], "detach")
        else tensors["head.bias"]
    )
    return core.softmax(np.asarray(features, dtype=np.float32) @ weight.T + bias)


def load_head(folder, expected_classes=None):
    from safetensors.numpy import load_file

    m = core.read_json(folder / "manifest.json")
    if m["base_model"] != {"model_id": BIO_ID, "revision": BIO_REV}:
        raise ValueError("Adapter base identity mismatch")
    classes = m["classes"]
    if (
        len(classes) < 2
        or len(classes) != len(set(classes))
        or (expected_classes is not None and classes != expected_classes)
    ):
        raise ValueError("Adapter class order mismatch")
    if set(m["prompts"]) != set(classes):
        raise ValueError("Adapter prompts mismatch")
    file = folder / "adapter.safetensors"
    if file.stat().st_size != m["bytes"] or core.sha256(file) != m["sha256"]:
        raise ValueError("Adapter bytes or digest mismatch")
    tensors = load_file(file)
    expected = {"head.weight": [len(classes), 768], "head.bias": [len(classes)]}
    if m["tensors"] != expected or set(tensors) != set(expected):
        raise ValueError("Adapter tensor scope mismatch")
    for name, tensor in tensors.items():
        if (
            list(tensor.shape) != expected[name]
            or tensor.dtype != np.float32
            or not np.isfinite(tensor).all()
        ):
            raise ValueError("Adapter tensor shape, dtype or finite check failed")
    return tensors, m


def adapt_head(root):
    import matplotlib.pyplot as plt
    from safetensors.torch import save_file

    manifest, classes, records = context(root)
    out = root / "outputs"
    features = np.load(out / "image_embeddings.npy", allow_pickle=False)
    text = np.load(out / "text_embeddings.npy", allow_pickle=False)
    train, val = indexes(records, "train"), indexes(records, "validation")
    scale = core.read_json(out / "embedding_identity.json")["logit_scale"]
    tensors, history, selection = fit_head(
        features[train],
        truths(records, classes, train),
        features[val],
        truths(records, classes, val),
        text,
        scale,
        epochs=CONFIG["epochs"],
    )
    selected = selection["epoch"]
    scores = head_scores(features, tensors)
    fitted = np.concatenate([train, val])
    change = head_change(features[fitted], tensors, text, scale)
    equals_zero_shot = selected == 0
    policy = core.select_threshold(truths(records, classes, val), scores[val])
    policy.update(
        {
            "selected_learning_rate": selection["learning_rate"],
            "selected_epoch": selected,
            "head_equals_zero_shot": equals_zero_shot,
            "selection": (
                "minimum validation cross-entropy over the predeclared learning-rate grid and epochs 0-"
                f"{CONFIG['epochs']}; ties keep the earlier epoch, then the smaller rate"
            ),
            "selection_split": "validation",
            "locked_before_test": True,
        }
    )
    core.write_json(out / "selected_policy.json", policy)
    np.save(out / "head_scores.npy", scores)
    write_csv(out / "training_history.csv", history)
    folder = out / "adapter"
    folder.mkdir(exist_ok=True)
    save_file(tensors, str(folder / "adapter.safetensors"))
    training = {
        **CONFIG,
        "optimizer": "AdamW",
        "trainable_names": sorted(tensors),
        "trainable_parameters": sum(v.numel() for v in tensors.values()),
        "selected_learning_rate": selection["learning_rate"],
        "selected_epoch": selected,
        "selection": policy["selection"],
        "change_from_zero_shot_on_train_and_validation": change,
    }
    adapter = {
        "format": "dimer-birds-linear-head-v1",
        "base_model": {"model_id": BIO_ID, "revision": BIO_REV},
        "classes": classes,
        "prompts": {c["label"]: f"a photo of {c['scientific_name']}." for c in manifest["classes"]},
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
        "bytes": (folder / "adapter.safetensors").stat().st_size,
        "sha256": core.sha256(folder / "adapter.safetensors"),
        "training": training,
    }
    core.write_json(folder / "manifest.json", adapter)
    # Flat learner-facing record of the head configuration and how far it moved.
    core.write_json(
        out / "head_training.json",
        {
            "trainable_tensors": ", ".join(sorted(tensors)),
            "trainable_parameters": training["trainable_parameters"],
            "frozen": "both BioCLIP towers",
            "optimizer": "AdamW",
            "learning_rate_grid": ", ".join(str(v) for v in CONFIG["learning_rates"]),
            "selected_learning_rate": selection["learning_rate"],
            "weight_decay": CONFIG["weight_decay"],
            "batch_size": CONFIG["batch_size"],
            "max_epochs": CONFIG["epochs"],
            "selected_epoch": selected,
            "selected_validation_cross_entropy": selection["validation_loss"],
            "head_equals_zero_shot": equals_zero_shot,
            **{k + "_train_validation": v for k, v in change.items()},
        },
    )
    loaded, _ = load_head(folder, classes)
    core.parity(scores[val], head_scores(features[val], loaded), policy["threshold"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for index, rate in enumerate(CONFIG["learning_rates"]):
        rows = [r for r in history if r["learning_rate"] == rate]
        colour = PALETTE[index % len(PALETTE)]
        ax.plot([r["epoch"] for r in rows], [r["validation_loss"] for r in rows], color=colour,
                label=f"validation · lr {rate:g}")
        ax.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], color=colour, linestyle=":",
                label=f"train · lr {rate:g}")
    ax.axvline(selected, color="black", linestyle="--",
               label=f"Selected: lr {selection['learning_rate']:g}, epoch {selected}")
    ax.set(
        xlabel="Epoch (0 = zero-shot initialisation)",
        ylabel="Cross-entropy loss",
        title="Frozen BioCLIP features · head-only adaptation · validation selects rate and epoch",
    )
    ax.legend(fontsize=7, ncol=2)
    figure(root, "loss.png", fig)


def all_scores(root):
    out = root / "outputs"
    scores = {}
    for filename in ("baseline_scores.npz", "siglip_scores.npz", "bioclip_scores.npz"):
        with np.load(out / filename, allow_pickle=False) as values:
            scores.update({k: values[k] for k in values.files})
    scores["bioclip_5nn"] = np.load(out / "knn_scores.npy", allow_pickle=False)
    scores["bioclip_head"] = np.load(out / "head_scores.npy", allow_pickle=False)
    return scores


def prediction_rows(records, ix, classes, scores, system, threshold=None):
    rows = []
    for i in ix:
        rank = np.argsort(-scores[i], kind="stable")
        margin = float(scores[i, rank[0]] - scores[i, rank[1]])
        rows.append(
            {
                "id": records[i]["id"],
                "system": system,
                "split": records[i]["split"],
                "truth": records[i]["label"],
                "prediction": classes[rank[0]],
                "scores": scores[i].tolist(),
                "top1_score": float(scores[i, rank[0]]),
                "top2_score": float(scores[i, rank[1]]),
                "top2_label": classes[rank[1]],
                "margin": margin,
                "correct": records[i]["label"] == classes[rank[0]],
                "accepted": margin >= threshold if threshold is not None else None,
            }
        )
    return rows


ERROR_CATEGORIES = (
    "highest-margin correct",
    "highest-margin incorrect",
    "lowest margin",
    "head and SigLIP (scientific) disagree",
)


def error_panel_selection(correct, margins, disagree):
    """One record per diagnostic category, falling back to the next candidate when a record already
    represents an earlier category. Empty categories are reported, never filled."""
    candidates = (
        np.where(correct)[0][np.argsort(-margins[correct], kind="stable")],
        np.where(~correct)[0][np.argsort(-margins[~correct], kind="stable")],
        np.argsort(margins, kind="stable"),
        np.where(disagree)[0][np.argsort(margins[disagree], kind="stable")],
    )
    chosen, rows = [], []
    for category, options in zip(ERROR_CATEGORIES, candidates, strict=True):
        pick = next((int(j) for j in options if int(j) not in chosen), None)
        if pick is not None:
            chosen.append(pick)
        rows.append((category, pick, len(options)))
    return rows


def evaluate(root):
    _, classes, records = context(root)
    out = root / "outputs"
    policy = core.read_json(out / "selected_policy.json")
    if not policy["locked_before_test"] or policy["selection_split"] != "validation":
        raise ValueError("Test reveal requires locked validation policy")
    ix = indexes(records, "test")
    y = truths(records, classes, ix)
    scores = all_scores(root)
    metric_rows, per_class, predictions = [], [], []
    canonical_correct = scores["bioclip_scientific"][ix].argmax(1) == y
    test_observers = len({records[i]["observer"] for i in ix})
    for name, values in scores.items():
        metric_rows.append(
            {
                "system": name,
                **scalar_metrics(y, values[ix], name),
                # Stratified by reference species: every resample keeps the test support per
                # species, so a species cannot vanish from a resample and score F1 = 0 (BC-M3).
                **core.bootstrap(y, values[ix], n_boot=CONFIG["bootstrap"], stratify=True),
                "bootstrap_method": "records resampled within each species; support fixed",
                "test_records": len(ix),
                "test_observers": test_observers,
                "paired_correctness_difference_vs_bioclip_scientific": float(
                    np.mean((values[ix].argmax(1) == y).astype(int) - canonical_correct.astype(int))
                ),
            }
        )
        details = core.metrics(y, values[ix])
        per_class.extend(
            {"system": name, "label": classes[i], **row} for i, row in enumerate(details["per_class"])
        )
        predictions.extend(
            prediction_rows(
                records, ix, classes, values, name, policy["threshold"] if name == "bioclip_head" else None
            )
        )
        confusion_figure(
            root, y, values[ix], classes, f"confusion_test_{name}.png", f"{name} · held-out test"
        )
    write_csv(out / "metrics.csv", metric_rows)
    write_csv(out / "per_class_metrics.csv", per_class)
    write_csv(out / "predictions.csv", predictions)
    head = scores["bioclip_head"][ix]
    order = np.argsort(-head, axis=1, kind="stable")
    margins = head[np.arange(len(ix)), order[:, 0]] - head[np.arange(len(ix)), order[:, 1]]
    correct = order[:, 0] == y
    disagree = scores["siglip_scientific"][ix].argmax(1) != order[:, 0]
    panel = error_panel_selection(correct, margins, disagree)
    panel_rows, chosen, annotations = [], [], []
    for category, j, available in panel:
        if j is None:
            panel_rows.append({"category": category, "id": "", "available_records": available,
                               "shown": False, "note": "no test record in this category"})
            continue
        decision = "accept" if margins[j] >= policy["threshold"] else "refer"
        chosen.append(j)
        annotations.append(
            f"[{category}]\n"
            f"Truth: {classes[y[j]]}\n"
            f"Top 1: {classes[order[j, 0]]} {head[j, order[j, 0]]:.3f}\n"
            f"Top 2: {classes[order[j, 1]]} {head[j, order[j, 1]]:.3f}\n"
            f"Margin {margins[j]:.3f} · {decision}"
        )
        panel_rows.append(
            {
                "category": category,
                "id": records[ix[j]]["id"],
                "available_records": available,
                "shown": True,
                "truth": classes[y[j]],
                "prediction": classes[order[j, 0]],
                "margin": float(margins[j]),
                "decision": decision,
            }
        )
    write_csv(out / "error_panel.csv", panel_rows)
    if chosen:
        contact(root, [records[ix[j]] for j in chosen], "errors.png", annotations)
    core.write_json(
        out / "error_categories.json",
        {
            "categories": {row["category"]: row["id"] or None for row in panel_rows},
            "note": "Absent categories are reported, never fabricated.",
        },
    )
    # The reload probe was embedded as one batch in the bioclip-zero-shot stage; the fresh
    # process repeats exactly that computation, so parity compares like with like (BC-m3).
    probe_ids = core.read_json(out / "embedding_identity.json")["reload_probe_ids"]
    if probe_ids != [records[i]["id"] for i in ix[:2]]:
        raise ValueError("Reload probe identities changed")
    tensors, _ = load_head(out / "adapter", classes)
    np.save(out / "reload_expected.npy", head_scores(np.load(out / "reload_probe_features.npy"), tensors))
    core.write_json(
        out / "reload_probe.json",
        {
            "ids": probe_ids,
            "pid": os.getpid(),
            "adapter_sha256": core.sha256(out / "adapter" / "adapter.safetensors"),
            "batch": "the same two test records, embedded together, as in the bioclip-zero-shot stage",
        },
    )


def neighbour_thresholds(candidates, locked):
    """The display activity moves to the adjacent distinct validation candidate on each side."""
    values = sorted({float(v) for v in candidates})
    lower = max((v for v in values if v < locked), default=locked)
    higher = min((v for v in values if v > locked), default=locked)
    return {"lower": lower, "canonical": locked, "higher": higher}


def triage(root, display_threshold=None):
    import matplotlib.pyplot as plt

    _, classes, records = context(root)
    out = root / "outputs"
    scores = np.load(out / "head_scores.npy", allow_pickle=False)
    policy = core.read_json(out / "selected_policy.json")
    rows = []
    for split in ("validation", "test"):
        ix = indexes(records, split)
        for row in core.risk_rows(truths(records, classes, ix), scores[ix]):
            rows.append({"split": split, **row, "canonical": float(row["threshold"]) == policy["threshold"]})
        if not any(r["split"] == split and r["canonical"] for r in rows):
            rows.extend(
                {"split": split, **r, "canonical": True}
                for r in core.risk_rows(
                    truths(records, classes, ix), scores[ix], thresholds=[policy["threshold"]]
                )
            )
    rows.sort(key=lambda r: (r["split"] != "validation", r["threshold"]))
    if display_threshold is not None:
        # The largest validation candidate is nextafter(max margin), which can exceed 1.0 by one ulp.
        if not np.isfinite(display_threshold) or not 0 <= display_threshold <= float(np.nextafter(1.0, 2.0)):
            raise ValueError("Display threshold must lie in [0,1]")
        ix = indexes(records, "test")
        core.write_json(
            out / "activity.json",
            {
                "display_only": True,
                "canonical_threshold": policy["threshold"],
                "display_threshold": display_threshold,
                "comparison": core.risk_rows(
                    truths(records, classes, ix),
                    scores[ix],
                    thresholds=sorted(set([policy["threshold"], display_threshold])),
                ),
            },
        )
        rows = prediction_rows(records, ix, classes, scores, "bioclip_head", display_threshold)
        chosen = []
        for accepted in (True, False):
            chosen.extend([r for r in rows if r["accepted"] == accepted][:3])
        contact(
            root,
            [next(r for r in records if r["id"] == p["id"]) for p in chosen],
            "activity.png",
            [
                f"Display-only threshold {display_threshold:.6g}\n"
                f"margin {p['margin']:.6g} · {p['prediction']} · {'accept' if p['accepted'] else 'refer'}"
                for p in chosen
            ],
        )
        return
    write_csv(out / "risk_coverage.csv", rows)
    validation_candidates = [r["threshold"] for r in rows if r["split"] == "validation"]
    core.write_json(
        out / "activity_thresholds.json",
        {
            **neighbour_thresholds(validation_candidates, policy["threshold"]),
            "rule": "adjacent distinct validation candidate thresholds around the locked value",
        },
    )
    probes = indexes(records, "probe")
    head_probe_rows = prediction_rows(records, probes, classes, scores, "bioclip_head", policy["threshold"])
    probe_rows = list(head_probe_rows)
    # Every predeclared system still names one of the four candidates for each probe (BC-m9);
    # only the head carries a referral decision. Probes are described, never scored.
    for name, values in all_scores(root).items():
        if name != "bioclip_head":
            probe_rows.extend(prediction_rows(records, probes, classes, values, name))
    for row in probe_rows:
        row.pop("correct")
    write_csv(
        out / "probe_predictions.csv",
        probe_rows,
        fields=list(probe_rows[0]) if probe_rows else ["id", "system", "prediction", "accepted"],
    )
    if len(probes):
        contact(
            root,
            [records[i] for i in probes],
            "probes.png",
            [
                f"Head candidate: {r['prediction']}\n"
                f"Margin {r['margin']:.3f} · {'accept' if r['accepted'] else 'refer'}"
                for r in head_probe_rows
            ],
        )
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, (split, marker) in enumerate((("validation", "o"), ("test", "s"))):
        chosen = [
            r
            for r in rows
            if r["split"] == split and r.get("selective_accuracy") is not None and not r["canonical"]
        ]
        chosen.sort(key=lambda r: r["coverage"])
        ax.plot(
            [r["coverage"] for r in chosen],
            [1 - r["selective_accuracy"] for r in chosen],
            marker=marker,
            color=PALETTE[index],
            label=f"{split} (all thresholds, diagnostic)" if split == "test" else split,
        )
        for r in rows:
            if r["split"] == split and r["canonical"] and r.get("selective_accuracy") is not None:
                ax.scatter(
                    r["coverage"], 1 - r["selective_accuracy"], s=160, facecolors="none", edgecolors="black",
                    linewidths=1.5, zorder=3,
                )
    ax.set(
        xlabel="Accepted fraction (coverage)",
        ylabel="Error fraction among accepted",
        title="Risk versus coverage · rings mark the locked validation threshold",
    )
    ax.legend()
    figure(root, "risk_coverage.png", fig)
    ix = indexes(records, "test")
    primary = prediction_rows(records, ix, classes, scores, "bioclip_head", policy["threshold"])
    write_csv(
        out / "referral_by_species.csv",
        [
            {
                "label": label,
                "count": sum(r["truth"] == label for r in primary),
                "referred": sum(r["truth"] == label and not r["accepted"] for r in primary),
                "incorrect_accepted": sum(
                    r["truth"] == label and r["accepted"] and not r["correct"] for r in primary
                ),
            }
            for label in classes
        ],
    )


def reload(root):
    _, classes, records = context(root)
    out = root / "outputs"
    probe = core.read_json(out / "reload_probe.json")
    if probe["pid"] == os.getpid():
        raise ValueError("Reload must run in a fresh process")
    if probe["adapter_sha256"] != core.sha256(out / "adapter" / "adapter.safetensors"):
        raise ValueError("Adapter changed after held-out evaluation")
    tensors, _ = load_head(out / "adapter", classes)
    pipe = bioclip_pipe(root, record_download=False)
    selected = [next(r for r in records if r["id"] == iid) for iid in probe["ids"]]
    features = np.array(pipe.embed_images(images(root, selected))["embeddings"], dtype=np.float32)
    actual = head_scores(features, tensors)
    result = core.parity(
        np.load(out / "reload_expected.npy", allow_pickle=False),
        actual,
        core.read_json(out / "selected_policy.json")["threshold"],
    )
    core.write_json(
        out / "verification_reload.json",
        {
            **result,
            "fresh_process": True,
            "original_pid": probe["pid"],
            "reload_pid": os.getpid(),
            "adapter_sha256": probe["adapter_sha256"],
            "batch_composition": probe.get("batch", "not recorded"),
        },
    )


def report(root):
    out = root / "outputs"
    _, classes, records = context(root)
    rows = read_csv(out / "predictions.csv")
    metrics_rows = read_csv(out / "metrics.csv")
    expected_ids = {r["id"] for r in records if r["split"] == "test"}
    for metric in metrics_rows:
        selected = [r for r in rows if r["system"] == metric["system"]]
        if len(selected) != len(indexes(records, "test")):
            raise ValueError("Prediction export count mismatch")
        if {r["id"] for r in selected} != expected_ids:
            raise ValueError("Prediction export record identities mismatch")
        for row in selected:
            scores = np.asarray(json.loads(row["scores"]))
            if scores.shape != (len(classes),) or not np.isfinite(scores).all():
                raise ValueError("Prediction CSV score type/shape mismatch")
            winner = classes[int(scores.argmax())]
            if row["prediction"] != winner or row["correct"] != str(winner == row["truth"]):
                raise ValueError("Prediction CSV labels/correctness mismatch")
        rebuilt = scalar_metrics(
            np.array([classes.index(r["truth"]) for r in selected]),
            np.array([json.loads(r["scores"]) for r in selected]),
            metric["system"],
        )
        for key in ("accuracy", "macro_f1", "top2_accuracy"):
            if rebuilt[key] is None:
                if metric[key] != "":
                    raise ValueError("Undefined metric exported with a value")
            elif not np.isclose(rebuilt[key], float(metric[key]), atol=1e-12):
                raise ValueError("Prediction/metric export parity failed")
    report_started = time.monotonic()
    write_csv(out / "runtime_summary.csv", runtime_rows(root, STAGES[:-1]))
    csv_counts = {}
    numeric_fields = {
        "accuracy",
        "macro_f1",
        "top2_accuracy",
        "precision",
        "recall",
        "f1",
        "support",
        "predicted",
        "margin",
        "top1_score",
        "top2_score",
        "threshold",
        "coverage",
        "count",
        "referred",
        "incorrect_accepted",
        "train_loss",
        "validation_loss",
        "epoch",
        "images",
        "observers",
    }
    for path in out.glob("*.csv"):
        table = read_csv(path)
        if any(None in row or any(value is None for value in row.values()) for row in table):
            raise ValueError("Malformed CSV row width")
        for row in table:
            for key in numeric_fields & row.keys():
                if row[key] and not np.isfinite(float(row[key])):
                    raise ValueError("Non-finite CSV numeric field")
        csv_counts[path.name] = len(table)
    receipts = [core.read_json(root / "receipts" / f"{s}.json") for s in STAGES[:-1]]
    versions = {}
    for package in (
        "torch",
        "torchvision",
        "open_clip_torch",
        "transformers",
        "safetensors",
        "Pillow",
        "numpy",
        "matplotlib",
        "scikit-learn",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    verification = {
        "status": "Candidate",
        "csv_counts": csv_counts,
        "prediction_metric_parity": True,
        "reload": core.read_json(out / "verification_reload.json"),
        "dataset": core.read_json(out / "prepare.json"),
        "hosted_release_review": "pending",
        "attribution_review": "pending",
        "archive": (
            "results.zip is verified after it is written (CRC, member digests, safe paths, excluded "
            "assets, size and headline-metric recomputation); see archive_verification.json, which "
            "cannot be inside the archive it verifies"
        ),
    }
    core.write_json(out / "verification.json", verification)
    bootstrap_record = core.read_json(out / "bootstrap.json") if (out / "bootstrap.json").exists() else {}
    core.write_json(
        out / "run_summary.json",
        {
            "status": "Candidate",
            "notebook_revision": NOTEBOOK_REVISION,
            "wall_clock_seconds_since_session_start": (
                time.time() - bootstrap_record["session_started_at"]
                if "session_started_at" in bootstrap_record
                else None
            ),
            "timing": {
                "bootstrap": bootstrap_record,
                "downloads": {
                    key: core.read_json(out / f"download_{key}.json")
                    for key in ("dataset", "siglip", "bioclip")
                    if (out / f"download_{key}.json").exists()
                },
                "report_seconds_before_archive": time.monotonic() - report_started,
            },
            "environment": {
                "python": platform.python_version(),
                "os": platform.platform(),
                "versions": versions,
            },
            "models": core.read_json(root / "model_manifest.json"),
            "policy": core.read_json(out / "selected_policy.json"),
            "metrics": metrics_rows,
            "stages": receipts,
            "stage_seconds_excluding_report": sum(r["seconds"] for r in receipts),
            "limitations": [
                "Research Grade labels can be wrong.",
                "Pretraining overlap unresolved for both models.",
                "Referral is not novelty detection or calibrated uncertainty.",
                "Small frozen sample is not national performance evidence.",
            ],
        },
    )
    (out / "conclusion.md").write_text(
        "On this observer-disjoint sample of [n] Philippine bird photographs from [number] held-out "
        "photographers, [system] achieved [macro-F1 with its species-stratified interval] compared "
        "with [baselines]. The selected head [did / did not] equal the zero-shot classifier "
        "(learning rate and epoch chosen on validation). The review policy accepted [coverage] with "
        "[selective accuracy], while [failure pattern] remained. This supports [bounded use], "
        "but does not establish pretraining independence, open-set recognition, nationwide "
        "accuracy or expert-level identification. Next collect [evidence].\n\n"
        "Completion notes are optional, not required submissions.\n",
        encoding="utf-8",
    )
    # Photos and photo-derived figures stay out of the default archive.
    files = [
        p
        for p in out.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".csv", ".md", ".txt", ".safetensors")
        and p.name not in ("checksums.json", "activity.json", "archive_verification.json")
    ]
    hashes = {p.relative_to(out).as_posix(): core.sha256(p) for p in sorted(files)}
    core.write_json(out / "checksums.json", hashes)
    archive = out / "results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files + [out / "checksums.json"]:
            z.write(p, p.relative_to(out).as_posix())
    if archive.stat().st_size > 100_000_000:
        raise ValueError("Results archive exceeds 100 MB")
    core.write_json(out / "archive_verification.json", verify_archive(archive, hashes, classes))


EXCLUDED_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".img", ".npy", ".npz", ".bin", ".pt", ".pth", ".whl")
EXCLUDED_PREFIXES = ("cache/", "weights/", "env/", "figures/")


def verify_archive(archive, hashes, classes):
    """Reopen results.zip and check CRC, digests, paths, excluded assets and headline metrics."""
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if z.testzip() is not None or set(names) != set(hashes) | {"checksums.json"}:
            raise ValueError("Archive CRC/member verification failed")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
                raise ValueError(f"Unsafe archive member path: {name}")
        for name, digest in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("Archive digest mismatch")
        excluded = [
            n for n in names
            if n.lower().endswith(EXCLUDED_SUFFIXES) or n.startswith(EXCLUDED_PREFIXES)
        ]
        safetensors = [n for n in names if n.endswith(".safetensors")]
        if excluded or safetensors != ["adapter/adapter.safetensors"]:
            raise ValueError(f"Archive contains excluded assets: {excluded or safetensors}")
        rows = list(csv.DictReader(io.StringIO(z.read("predictions.csv").decode("utf-8"))))
        headline = {}
        for metric in csv.DictReader(io.StringIO(z.read("metrics.csv").decode("utf-8"))):
            chosen = [r for r in rows if r["system"] == metric["system"]]
            rebuilt = scalar_metrics(
                np.array([classes.index(r["truth"]) for r in chosen]),
                np.array([json.loads(r["scores"]) for r in chosen]),
                metric["system"],
            )
            for key in ("accuracy", "macro_f1"):
                if not np.isclose(rebuilt[key], float(metric[key]), atol=1e-12):
                    raise ValueError("Archived predictions do not reproduce archived metrics")
            headline[metric["system"]] = {"accuracy": rebuilt["accuracy"], "macro_f1": rebuilt["macro_f1"]}
    size = archive.stat().st_size
    if size > 100_000_000:
        raise ValueError("Results archive exceeds 100 MB")
    return {
        "crc": True,
        "member_hashes": True,
        "safe_member_paths": True,
        "members": len(names),
        "bytes": size,
        "excluded_assets_absent": True,
        "checked_exclusions": ", ".join(EXCLUDED_SUFFIXES + EXCLUDED_PREFIXES),
        "headline_metrics_recomputed_from_archived_predictions": True,
        "systems_recomputed": len(headline),
    }


def runtime_rows(root, stages):
    """One flat row per measured step: bootstrap, downloads, stages (report is added by its receipt)."""
    out = root / "outputs"
    rows = []
    if (out / "bootstrap.json").exists():
        boot = core.read_json(out / "bootstrap.json")
        for key in ("uv_download_seconds", "environment_seconds", "install_seconds"):
            if key in boot:
                rows.append({"step": "bootstrap: " + key.replace("_seconds", "").replace("_", " "),
                             "seconds": boot[key]})
    for key in ("dataset", "siglip", "bioclip"):
        path = out / f"download_{key}.json"
        if path.exists():
            record = core.read_json(path)
            rows.append({"step": f"download: {key} (included in its stage)", "seconds": record["seconds"],
                         "downloaded_bytes": record["downloaded_bytes"]})
    for stage in stages:
        receipt = core.read_json(root / "receipts" / f"{stage}.json")
        resources = receipt.get("resources", {})
        rows.append(
            {
                "step": f"stage: {stage}",
                "seconds": receipt.get("seconds"),
                "peak_vram_bytes": resources.get("peak_vram_bytes"),
                "peak_host_ram_bytes": resources.get("peak_host_ram_bytes"),
            }
        )
    total = sum(float(r["seconds"] or 0) for r in rows if not r["step"].startswith("download:"))
    rows.append({"step": "total: bootstrap + stages before report", "seconds": total})
    return rows


def identity(root):
    paths = [
        root / "data_manifest.json",
        root / "model_manifest.json",
        Path(__file__),
        root / "biodiversity_core.py",
    ]
    paths += sorted((root / "bioclip2_biodiversity_pipeline").glob("*.py"))
    paths += [p for p in (root / "requirements.txt", root / "source.json") if p.exists()]
    return {
        "files": {
            str(p.relative_to(root)) if p.is_relative_to(root) else p.name: core.sha256(p) for p in paths
        },
        "config": CONFIG,
    }


def begin_stage(root, stage, invalidate=True):
    current = identity(root)
    receipts = root / "receipts"
    receipts.mkdir(exist_ok=True)
    for preceding in STAGES[: STAGES.index(stage)]:
        receipt = core.read_json(receipts / f"{preceding}.json")
        if receipt["identity"] != current:
            raise ValueError("Source/config/manifest changed; rerun from prepare")
        for name, digest in receipt["products"].items():
            if core.sha256(core.safe_path(root, name)) != digest:
                raise ValueError(f"Prerequisite changed: {name}")
    if stage != "prepare":
        _, _, records = context(root)
        for r in records:
            path = root / "cache" / f"{r['id']}.img"
            if path.stat().st_size != r["bytes"] or core.sha256(path) != r["sha256"]:
                raise ValueError("Cached dataset changed; rerun prepare")
    if invalidate:
        for following in STAGES[STAGES.index(stage) :]:
            (receipts / f"{following}.json").unlink(missing_ok=True)
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=(*STAGES, "byod-infer"), required=True)
    parser.add_argument("--display-threshold", type=float)
    args = parser.parse_args()
    root = args.root.resolve()
    (root / "outputs").mkdir(exist_ok=True)
    if args.stage == "byod-infer":
        byod_infer(root)
        return
    if args.display_threshold is not None:
        if args.stage != "triage":
            raise ValueError("Display threshold belongs only to triage")
        begin_stage(root, "triage", invalidate=False)
        triage(root, args.display_threshold)
        return
    source_identity = begin_stage(root, args.stage)
    inputs = {
        stage: core.read_json(root / "receipts" / f"{stage}.json")["products"]
        for stage in STAGES[: STAGES.index(args.stage)]
    }
    before = {
        p.relative_to(root).as_posix(): (core.sha256(p), p.stat().st_mtime_ns)
        for p in (root / "outputs").rglob("*")
        if p.is_file()
    }
    started = time.monotonic()
    globals()[args.stage.replace("-", "_")](root)
    if identity(root) != source_identity:
        raise ValueError("Source changed during stage")
    products = {
        p.relative_to(root).as_posix(): core.sha256(p)
        for p in (root / "outputs").rglob("*")
        if p.is_file()
        and before.get(p.relative_to(root).as_posix()) != (core.sha256(p), p.stat().st_mtime_ns)
    }
    resources = {}
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if torch.cuda.is_available():
            resources.update(
                {
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                    "gpu": torch.cuda.get_device_name(),
                    "cuda": torch.version.cuda,
                }
            )
    try:
        import resource

        resources["peak_host_ram_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        resources["peak_host_ram_bytes"] = None
    core.write_json(
        root / "receipts" / f"{args.stage}.json",
        {
            "identity": source_identity,
            "stage": args.stage,
            "products": products,
            "input_products": inputs,
            "seconds": time.monotonic() - started,
            "resources": resources,
            "pid": os.getpid(),
        },
    )
    print(json.dumps({"stage": args.stage, "products": len(products), "status": "complete"}))


if __name__ == "__main__":
    main()
