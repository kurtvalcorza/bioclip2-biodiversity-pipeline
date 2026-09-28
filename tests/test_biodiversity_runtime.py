"""Offline software integration; synthetic features are never release evidence."""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import biodiversity_capstone as run  # noqa: E402
import biodiversity_core as core  # noqa: E402


def fixture_root(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    records = [
        {"id": f"{role}_{c}_{j}", "label": f"c{c}", "split": role, "observer": f"{role}_{j}"}
        for role, count in (("train", 8), ("validation", 3), ("test", 3))
        for c in range(2)
        for j in range(count)
    ]
    core.write_json(
        tmp_path / "data_manifest.json",
        {
            "classes": [{"label": f"c{c}", "scientific_name": f"Species {c}"} for c in range(2)],
            "records": records,
            "probes": [],
        },
    )
    core.write_json(tmp_path / "model_manifest.json", {})
    labels = np.array([int(r["label"][1:]) for r in records])
    features = np.eye(768, dtype=np.float32)[labels]
    np.save(out / "image_embeddings.npy", features)
    np.save(out / "text_embeddings.npy", np.eye(768, dtype=np.float32)[:2])
    core.write_json(out / "embedding_identity.json", {"logit_scale": 2.0})
    return out, records, features


def test_head_epoch_zero_is_valid_candidate():
    pytest.importorskip("torch")
    features = np.eye(768, dtype=np.float32)[:2]
    labels = np.array([0, 1])
    tensors, history, selected = run.fit_head(features, labels, features, labels, features, 2, epochs=0)
    assert selected == 0 and len(history) == 1
    np.testing.assert_allclose(tensors["head.weight"].numpy(), features * 2)
    assert run.head_scores(features, tensors).argmax(1).tolist() == [0, 1]


def test_adapt_export_refuses_tampering_and_wrong_class_order(tmp_path):
    pytest.importorskip("torch")
    out, _, features = fixture_root(tmp_path)
    run.adapt_head(tmp_path)
    policy = core.read_json(out / "selected_policy.json")
    assert policy["locked_before_test"] and policy["selection_split"] == "validation"
    history = run.read_csv(out / "training_history.csv")
    assert len(history) == 21
    selected = min(history, key=lambda r: float(r["validation_loss"]))
    assert policy["selected_epoch"] == int(selected["epoch"])
    tensors, manifest = run.load_head(out / "adapter", ["c0", "c1"])
    assert manifest["training"]["trainable_parameters"] == 2 * 769
    np.testing.assert_allclose(run.head_scores(features, tensors), np.load(out / "head_scores.npy"))
    with pytest.raises(ValueError, match="class order"):
        run.load_head(out / "adapter", ["c1", "c0"])
    artifact = out / "adapter" / "adapter.safetensors"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="digest"):
        run.load_head(out / "adapter")


def test_synthetic_evaluation_triage_and_archive_parity(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    out, records, _ = fixture_root(tmp_path)
    run.adapt_head(tmp_path)
    perfect = core.softmax(np.eye(2)[[int(r["label"][1:]) for r in records]] * 2)
    np.savez(out / "baseline_scores.npz", majority=np.tile([1, 0], (len(records), 1)), colour=perfect)
    np.savez(out / "siglip_scores.npz", siglip_common=perfect, siglip_scientific=perfect)
    np.savez(out / "bioclip_scores.npz", bioclip_common=perfect, bioclip_scientific=perfect)
    np.save(out / "knn_scores.npy", perfect)
    monkeypatch.setattr(run, "contact", lambda *a, **kw: None)
    original_bootstrap = core.bootstrap
    monkeypatch.setattr(
        core,
        "bootstrap",
        lambda y, s, groups=None, n_boot=2000: original_bootstrap(y, s, groups=groups, n_boot=30),
    )
    run.evaluate(tmp_path)
    assert len(run.read_csv(out / "predictions.csv")) == 8 * 6
    assert len(run.read_csv(out / "metrics.csv")) == 8
    run.triage(tmp_path)
    before = core.sha256(out / "selected_policy.json")
    run.triage(tmp_path, display_threshold=1.0)
    assert before == core.sha256(out / "selected_policy.json")
    assert run.read_csv(out / "probe_predictions.csv") == []
    # Explicit software fixtures, NOT a substitute for actual fresh-process model reload.
    core.write_json(out / "verification_reload.json", {"test_fixture": True})
    core.write_json(out / "prepare.json", {"test_fixture": True})
    for stage in run.STAGES[:-1]:
        core.write_json(tmp_path / "receipts" / f"{stage}.json", {"seconds": 0, "test_fixture": True})
    run.report(tmp_path)
    assert core.read_json(out / "verification.json")["prediction_metric_parity"]
    with zipfile.ZipFile(out / "results.zip") as bundle:
        assert "adapter/adapter.safetensors" in bundle.namelist()
        assert not any(name.endswith((".png", ".jpg")) for name in bundle.namelist())
        checksums = json.loads(bundle.read("checksums.json"))
        import hashlib

        assert all(
            hashlib.sha256(bundle.read(name)).hexdigest() == digest for name, digest in checksums.items()
        )
    rows = run.read_csv(out / "predictions.csv")
    rows[0]["scores"] = "[0, 1]"
    run.write_csv(out / "predictions.csv", rows)
    with pytest.raises(ValueError):
        run.report(tmp_path)


def test_stage_receipt_refuses_mutated_prerequisite(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "identity", lambda root: {"test_fixture": True})
    (tmp_path / "outputs").mkdir()
    product = tmp_path / "outputs" / "prepare.json"
    core.write_json(product, {"good": True})
    core.write_json(
        tmp_path / "receipts" / "prepare.json",
        {
            "identity": {"test_fixture": True},
            "products": {"outputs/prepare.json": core.sha256(product)},
        },
    )
    core.write_json(product, {"changed": True})
    with pytest.raises(ValueError, match="Prerequisite changed"):
        run.begin_stage(tmp_path, "siglip-zero-shot")
