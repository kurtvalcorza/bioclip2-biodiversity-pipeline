"""Standalone Philippine bird capstone stages; model imports are deliberately lazy."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

import biodiversity_core as core
import matplotlib
import numpy as np

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
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "batch_size": 8,
    "precision": "float32",
    "k": 5,
    "bootstrap": 2000,
    "atol": 1e-6,
    "rtol": 1e-5,
}


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


def contact(root, records, name, annotations=None):
    import matplotlib.pyplot as plt

    selected = records[:12]
    fig, axes = plt.subplots(
        (len(selected) + 2) // 3, 3, figsize=(15, 4.5 * ((len(selected) + 2) // 3)), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for j, (ax, r, im) in enumerate(zip(axes.flat, selected, images(root, selected), strict=False)):
        ax.imshow(im)
        title = f"{r['id']} · {r.get('common_name', r['label'])}"
        if annotations:
            title += "\n" + annotations[j]
        title += f"\n{r.get('attribution', 'Synthetic control')} · {r.get('license_code', 'CC0')}"
        ax.set_title(title, fontsize=8, wrap=True)
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
    image_checks = core.validate_images(root, manifest)
    out = root / "outputs"
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
        [{"system": name, **scalar_metrics(y, s[val])} for name, s in scores.items()],
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
    contact(root, selected, "contact_train.png")
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


def scalar_metrics(y, scores):
    return {k: v for k, v in core.metrics(y, scores).items() if k not in ("confusion", "per_class")}


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
    data = {}
    for condition, field in (("common", "common_name"), ("scientific", "scientific_name")):
        prompts = [f"This is a photo of {c[field]}." for c in manifest["classes"]]
        scores = []
        for start in range(0, len(records), 8):
            inputs = processor(
                text=prompts,
                images=images(root, records[start : start + 8]),
                padding="max_length",
                return_tensors="pt",
            ).to("cuda")
            with torch.no_grad():
                output = model(**inputs)
                scores.extend(torch.sigmoid(output.logits_per_image).float().cpu().numpy())
        data[f"siglip_{condition}"] = np.asarray(scores)
    np.savez(root / "outputs" / "siglip_scores.npz", **data)
    val = indexes(records, "validation")
    y = truths(records, classes, val)
    write_csv(
        root / "outputs" / "validation_siglip.csv",
        [{"system": k, **scalar_metrics(y, s[val])} for k, s in data.items()],
    )
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


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
    repeated = np.asarray(pipe.embed_images(images(root, records[:1]))["embeddings"], dtype=np.float32)
    if not np.allclose(features[:1], repeated, atol=1e-6, rtol=1e-5):
        raise ValueError("BioCLIP repeated embedding parity failed")
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
            "dimension": features.shape[1],
        },
    )
    val = indexes(records, "validation")
    write_csv(
        root / "outputs" / "validation_bioclip.csv",
        [{"system": k, **scalar_metrics(truths(records, classes, val), s[val])} for k, s in data.items()],
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
                color=plt.get_cmap("tab10")(c),
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


def fit_head(train_x, train_y, val_x, val_y, text, scale, epochs=20):
    """CPU-testable head fitting; epoch zero participates in validation-only selection."""
    import torch

    torch.manual_seed(42)
    train_x, val_x = torch.tensor(train_x, dtype=torch.float32), torch.tensor(val_x, dtype=torch.float32)
    train_y, val_y = torch.tensor(train_y, dtype=torch.long), torch.tensor(val_y, dtype=torch.long)
    head = torch.nn.Linear(train_x.shape[1], len(text))
    with torch.no_grad():
        head.weight.copy_(torch.tensor(text, dtype=torch.float32) * scale)
        head.bias.zero_()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    history, best_loss, best, selected = [], float("inf"), None, 0
    for epoch in range(epochs + 1):
        if epoch:
            for ix in torch.randperm(len(train_y)).split(8):
                optimizer.zero_grad()
                loss_fn(head(train_x[ix]), train_y[ix]).backward()
                optimizer.step()
        with torch.no_grad():
            train_loss = float(loss_fn(head(train_x), train_y))
            val_loss = float(loss_fn(head(val_x), val_y))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, selected = val_loss, epoch
            best = {"head." + k: v.detach().clone() for k, v in head.state_dict().items()}
    return best, history, selected


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
    tensors, history, selected = fit_head(
        features[train],
        truths(records, classes, train),
        features[val],
        truths(records, classes, val),
        text,
        scale,
    )
    scores = head_scores(features, tensors)
    policy = core.select_threshold(truths(records, classes, val), scores[val])
    policy.update(
        {
            "selected_epoch": selected,
            "selection": "minimum validation cross-entropy; earlier epoch on ties, including epoch zero",
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
    adapter = {
        "format": "dimer-birds-linear-head-v1",
        "base_model": {"model_id": BIO_ID, "revision": BIO_REV},
        "classes": classes,
        "prompts": {c["label"]: f"a photo of {c['scientific_name']}." for c in manifest["classes"]},
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
        "bytes": (folder / "adapter.safetensors").stat().st_size,
        "sha256": core.sha256(folder / "adapter.safetensors"),
        "training": {
            **CONFIG,
            "optimizer": "AdamW",
            "trainable_names": sorted(tensors),
            "trainable_parameters": sum(v.numel() for v in tensors.values()),
            "selected_epoch": selected,
            "selection": policy["selection"],
        },
    }
    core.write_json(folder / "manifest.json", adapter)
    loaded, _ = load_head(folder, classes)
    core.parity(scores[val], head_scores(features[val], loaded), policy["threshold"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for key in ("train_loss", "validation_loss"):
        ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
    ax.axvline(selected, color="black", linestyle="--", label=f"Selected epoch {selected}")
    ax.set(
        xlabel="Epoch", ylabel="Cross-entropy loss", title="Frozen BioCLIP features · head-only adaptation"
    )
    ax.legend()
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
    for name, values in scores.items():
        metric_rows.append(
            {
                "system": name,
                **scalar_metrics(y, values[ix]),
                **core.bootstrap(y, values[ix], n_boot=2000),
                **{
                    "observer_cluster_" + key: value
                    for key, value in core.bootstrap(
                        y, values[ix], groups=[records[i]["observer"] for i in ix], n_boot=2000
                    ).items()
                },
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
    chosen = []
    for candidate in (
        np.where(correct)[0][np.argsort(-margins[correct])],
        np.where(~correct)[0][np.argsort(-margins[~correct])],
        np.argsort(margins),
        np.where(scores["siglip_scientific"][ix].argmax(1) != order[:, 0])[0],
    ):
        for j in candidate[:1]:
            if int(j) not in chosen:
                chosen.append(int(j))
    annotations = [
        f"Truth: {classes[y[j]]}\n"
        f"Top 1: {classes[order[j, 0]]} {head[j, order[j, 0]]:.3f}\n"
        f"Top 2: {classes[order[j, 1]]} {head[j, order[j, 1]]:.3f}\n"
        f"Margin {margins[j]:.3f} · {'accept' if margins[j] >= policy['threshold'] else 'refer'}"
        for j in chosen
    ]
    contact(root, [records[ix[j]] for j in chosen], "errors.png", annotations)
    core.write_json(
        out / "error_categories.json",
        {
            "correct_available": bool(correct.any()),
            "incorrect_available": bool((~correct).any()),
            "disagreement_available": bool((scores["siglip_scientific"][ix].argmax(1) != order[:, 0]).any()),
            "note": "Absent categories are reported, never fabricated.",
        },
    )
    np.save(out / "reload_expected.npy", head[:2])
    core.write_json(
        out / "reload_probe.json",
        {
            "ids": [records[i]["id"] for i in ix[:2]],
            "pid": os.getpid(),
            "adapter_sha256": core.sha256(out / "adapter" / "adapter.safetensors"),
        },
    )


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
    if display_threshold is not None:
        if not np.isfinite(display_threshold) or not 0 <= display_threshold <= 1:
            raise ValueError("Display threshold must lie in [0,1]")
        ix = indexes(records, "test")
        core.write_json(
            out / "activity.json",
            {
                "display_only": True,
                "canonical_threshold": policy["threshold"],
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
                f"Display-only threshold {display_threshold:.3f}\n"
                f"{p['prediction']} · {'accept' if p['accepted'] else 'refer'}"
                for p in chosen
            ],
        )
        return
    write_csv(out / "risk_coverage.csv", rows)
    probes = indexes(records, "probe")
    probe_rows = prediction_rows(records, probes, classes, scores, "bioclip_head", policy["threshold"])
    for row in probe_rows:
        row.pop("correct")
    write_csv(
        out / "probe_predictions.csv",
        probe_rows,
        fields=list(probe_rows[0]) if probe_rows else ["id", "prediction", "accepted"],
    )
    if len(probes):
        contact(
            root,
            [records[i] for i in probes],
            "probes.png",
            [
                f"Candidate: {r['prediction']}\n"
                f"Margin {r['margin']:.3f} · {'accept' if r['accepted'] else 'refer'}"
                for r in probe_rows
            ],
        )
    fig, ax = plt.subplots(figsize=(8, 5))
    for split, marker in (("validation", "o"), ("test", "s")):
        chosen = [r for r in rows if r["split"] == split and r.get("selective_accuracy") is not None]
        ax.plot(
            [r["coverage"] for r in chosen],
            [1 - r["selective_accuracy"] for r in chosen],
            marker=marker,
            label=split,
        )
        for r in chosen:
            if r["canonical"]:
                ax.scatter(
                    r["coverage"], 1 - r["selective_accuracy"], s=140, facecolors="none", edgecolors="black"
                )
    ax.set(
        xlabel="Accepted fraction (coverage)",
        ylabel="Error fraction among accepted",
        title="Risk versus coverage · rings mark locked validation threshold",
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
        )
        for key in ("accuracy", "macro_f1", "top2_accuracy"):
            if not np.isclose(rebuilt[key], float(metric[key]), atol=1e-12):
                raise ValueError("Prediction/metric export parity failed")
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
    }
    core.write_json(out / "verification.json", verification)
    core.write_json(
        out / "run_summary.json",
        {
            "status": "Candidate",
            "environment": {
                "python": platform.python_version(),
                "os": platform.platform(),
                "versions": versions,
            },
            "models": core.read_json(root / "model_manifest.json"),
            "policy": core.read_json(out / "selected_policy.json"),
            "metrics": metrics_rows,
            "stages": receipts,
            "elapsed_seconds": sum(r["seconds"] for r in receipts),
            "limitations": [
                "Research Grade labels can be wrong.",
                "Pretraining overlap unresolved for both models.",
                "Referral is not novelty detection or calibrated uncertainty.",
                "Small frozen sample is not national performance evidence.",
            ],
        },
    )
    (out / "conclusion.md").write_text(
        "On this observer-disjoint sample of [n] Philippine bird photographs, [system] achieved "
        "[macro-F1] compared with [baselines]. The review policy accepted [coverage] with "
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
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or set(z.namelist()) != set(hashes) | {"checksums.json"}:
            raise ValueError("Archive CRC/member verification failed")
        for name, digest in hashes.items():
            core.safe_path(out, name)
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("Archive digest mismatch")
    core.write_json(
        out / "archive_verification.json",
        {
            "crc": True,
            "member_hashes": True,
            "members": len(hashes) + 1,
            "bytes": archive.stat().st_size,
            "source_photos_and_contact_sheets_excluded": True,
        },
    )


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
