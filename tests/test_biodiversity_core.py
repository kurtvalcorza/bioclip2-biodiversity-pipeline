"""Hand-calculated scientific contracts and refusal cases, without model weights."""

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import biodiversity_core as core  # noqa: E402


def manifest():
    records = []
    for ci in range(2):
        for role in core.ROLES:
            for j in range(2):
                identity = f"c{ci}_{role}_{j}"
                records.append(
                    {
                        "id": identity,
                        "photo_id": identity,
                        "observation_id": identity,
                        "label": f"c{ci}",
                        "observer": identity,
                        "split": role,
                        "sha256": hashlib.sha256(identity.encode()).hexdigest(),
                        "license_code": "cc-by",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/",
                        "attribution": "Test photographer",
                        "bytes": 100,
                        "width": 224,
                        "height": 224,
                    }
                )
    return {"classes": [{"label": "c0"}, {"label": "c1"}], "records": records, "probes": []}


def test_metrics_hand_calculated_and_stable_score_ties():
    scores = np.array([[0.9, 0.1], [0.6, 0.4], [0.7, 0.3], [0.1, 0.9]])
    result = core.metrics(np.array([0, 0, 1, 1]), scores)
    assert result["confusion"] == [[2, 0], [1, 1]]
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert result["top2_accuracy"] == 1
    assert core.metrics(np.array([0]), np.array([[0.5, 0.5]]))["accuracy"] == 1


@pytest.mark.parametrize("scores", [np.array([[np.nan, 1]]), np.array([[1]]), np.array([1, 2])])
def test_invalid_scores_refused(scores):
    with pytest.raises(ValueError):
        core.metrics(np.array([0]), scores)


def test_bootstrap_reproducible_and_cluster_option():
    y = np.array([0, 0, 1, 1])
    scores = np.eye(2)[y]
    first = core.bootstrap(y, scores, n_boot=30)
    assert first == core.bootstrap(y, scores, n_boot=30)
    assert first["accuracy_low"] == first["accuracy_high"] == 1
    assert core.bootstrap(y, scores, groups=["a", "a", "b", "b"], n_boot=30)["accuracy_low"] == 1


def test_referral_selects_validation_policy_and_handles_all_referred():
    y = np.array([0, 0, 0, 1])
    scores = np.array([[0.95, 0.05], [0.8, 0.2], [0.1, 0.9], [0.49, 0.51]])
    selected = core.select_threshold(y, scores)
    # No eligible >=50%-coverage threshold reaches80%; fullcoverage accuracy.75 wins.
    assert selected["coverage"] == 1
    assert selected["selective_accuracy"] == 0.75
    empty = core.risk_rows(y, scores, [1.1])[0]
    assert empty["selective_accuracy"] is None and empty["referred"] == 4
    assert core.select_threshold(np.array([0, 1]), np.array([[0.8, 0.2], [0.1, 0.9]]))["coverage"] == 1


def test_knn_uses_positive_similarity_and_lexical_vote_tie():
    training = np.array([[1.0, 0], [1.0, 0], [-1.0, 0], [-1.0, 0], [0.0, 1]])
    scores = core.knn_scores(training, np.array([0, 1, 0, 1, 1]), np.array([[1.0, 0]]))
    assert scores.tolist() == [[1, 1]]
    assert scores.argmax(1).tolist() == [0]


def test_manifest_group_and_licence_audit():
    result = core.validate_manifest(manifest(), strict=False)
    assert result["split_counts"] == {"train": 4, "validation": 4, "test": 4}


def test_frozen_manifest_integrity_and_grouped_support():
    data = core.read_json(ROOT / "tools/biodiversity_data.json")
    result = core.validate_manifest(data)
    assert result["records"] == 56 and result["probes"] == 5
    assert result["split_counts"] == {"train": 32, "validation": 12, "test": 12}
    assert core.sha256(ROOT / "tools/biodiversity_exclusions.json") == data["exclusions_sha256"]
    data["records"][0]["attribution"] += " changed"
    with pytest.raises(ValueError, match="manifest digest"):
        core.validate_manifest(data)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda m: m["records"][2].update(observer=m["records"][0]["observer"]), "Observer leakage"),
        (lambda m: m["records"][0].update(license_code="cc-by-nc"), "Disallowed"),
        (lambda m: m["records"][0].update(attribution=""), "attribution"),
        (lambda m: m["records"][0].update(latitude=1), "metadata"),
        (lambda m: m["records"][0].update(width=10), "dimension"),
        (lambda m: m["records"][1].update(sha256=m["records"][0]["sha256"]), "Duplicate"),
        (lambda m: m["records"][0].update(id="../escape"), "Unsafe"),
    ],
)
def test_manifest_refusals(change, match):
    data = manifest()
    change(data)
    with pytest.raises(ValueError, match=match):
        core.validate_manifest(data, strict=False)


@pytest.mark.parametrize("name", ["../x", "/x", "C:/x", "a\\x", "a/../../x"])
def test_path_refusal(tmp_path, name):
    with pytest.raises(ValueError):
        core.safe_path(tmp_path, name)


def test_corrupt_image_cache_refused(tmp_path):
    row = manifest()["records"][0]
    path = tmp_path / "cache" / (row["id"] + ".img")
    path.parent.mkdir()
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Modified"):
        core.fetch_asset(tmp_path, row)


def test_parity_refuses_threshold_crossing_even_with_close_scores():
    a = np.array([[0.7, 0.3]])
    b = np.array([[0.7000001, 0.2999999]])
    with pytest.raises(ValueError, match="referral"):
        core.parity(a, b, threshold=0.4000001)
    assert core.parity(a, a, threshold=0.4)["identical_labels"]


def test_byod_zip_traversal_and_ungrouped_inference_only(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("../outside.png", b"x")
    with pytest.raises(ValueError, match="Unsafe"):
        core.prepare_byod(tmp_path / "run", str(bad), authorization=True)
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        for index in range(2):
            image = Image.new("RGB", (32, 32), (index * 100, 20, 30))
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            archive.writestr(f"c{index}/image.png", stream.getvalue())
    result = core.prepare_byod(tmp_path / "run", str(good), {"c0": "Species A", "c1": "Species B"}, True)
    assert result["inference_only"] and all(r["split"] == "probe" for r in result["records"])
    assert core.read_json(tmp_path / "run" / "data_manifest.json") == result


def test_grouped_byod_assignment():
    records = [
        {"label": label, "observer": f"observer{observer}"}
        for label in ("a", "b")
        for observer in range(7)
        for _ in range(2)
    ]
    core.assign_byod_roles(records)
    by_observer = {}
    for row in records:
        by_observer.setdefault(row["observer"], set()).add(row["split"])
    assert all(len(roles) == 1 for roles in by_observer.values())
    for label in ("a", "b"):
        for role in core.ROLES:
            assert len({r["observer"] for r in records if r["label"] == label and r["split"] == role}) >= 2


def test_no_predicted_class_has_zero_f1():
    result = core.metrics(np.array([0, 1]), np.array([[1.0, 0], [1.0, 0]]))
    assert result["per_class"][1] == {
        "class_index": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 1,
        "predicted": 0,
    }
