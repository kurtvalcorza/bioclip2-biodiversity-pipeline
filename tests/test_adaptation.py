"""Offline tests for the embedded sample, the image dataset contract, splits, metrics, baselines,
BYOD loaders and artifact-manifest checks."""

from __future__ import annotations

import json

import pytest

from bioclip2_biodiversity_pipeline import (
    ARTIFACT_FORMAT,
    ARTIFACT_MANIFEST_NAME,
    MIN_RECORDS,
    MIN_RECORDS_PER_CLASS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_CLASS_PROMPTS,
    SAMPLE_CLASSES,
    SAMPLE_COMMON_NAMES,
    SAMPLE_SIZE,
    BioClip2Pipeline,
    auroc,
    classification_metrics,
    color_baseline,
    color_features,
    dataset_digest,
    decode_image,
    generate_sample_dataset,
    load_byod_dataset,
    majority_baseline,
    sample_provenance,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from bioclip2_biodiversity_pipeline.sample_data import SAMPLE_RECORDS
from conftest import make_image_bytes

# The dataset identity the tutorial records; any edit to sample_data.py changes it.
SAMPLE_DIGEST = "f8fc68cf251e03655e343fd4ab3f49556248350d0d77ba4261698602e4a33cc6"

# --- the embedded sample --------------------------------------------------------------------------


def test_sample_dataset_is_the_recorded_cc0_set():
    records = generate_sample_dataset()
    assert len(records) == SAMPLE_SIZE == 48
    manifest = validate_dataset(records)
    assert manifest["verdict"] == "accepted"
    assert manifest["classes"] == list(SAMPLE_CLASSES)
    assert manifest["class_counts"] == {c: 12 for c in SAMPLE_CLASSES}
    assert manifest["digest"] == dataset_digest(records) == SAMPLE_DIGEST
    assert manifest["image_width"] == {"min": 224, "max": 224}
    assert manifest["image_height"] == {"min": 224, "max": 224}
    assert "nothing verifies that an image shows" in manifest["validation"]


def test_sample_provenance_is_complete_and_location_free():
    for rec in SAMPLE_RECORDS:
        assert rec["license_code"] == "cc0"
        assert str(rec["inat_observation_url"]).startswith("https://www.inaturalist.org/observations/")
        assert rec["scientific_name"] == SAMPLE_CLASS_PROMPTS[str(rec["label"])]
        assert rec["common_name"] == SAMPLE_COMMON_NAMES[str(rec["label"])]
        assert "place_guess" not in rec and "latitude" not in rec and "location" not in rec
    prov = sample_provenance()
    assert prov["image_license"] == "CC0-1.0" and prov["n_images"] == 48
    assert prov["location_data"] == "not collected"
    # one photo per observer within a species: the split's independence assumption rests on this
    for cls in SAMPLE_CLASSES:
        observers = [r["observer"] for r in SAMPLE_RECORDS if r["label"] == cls]
        assert len(observers) == len(set(observers)) == 12


def test_sample_records_carry_provenance_fields():
    rec = generate_sample_dataset()[0]
    assert {"id", "image_bytes", "label", "file", "scientific_name", "source", "license", "observer"} <= set(
        rec
    )
    assert decode_image(rec["image_bytes"]).size == (224, 224)


# --- dataset validation ---------------------------------------------------------------------------


def _records(n_per_class=MIN_RECORDS, classes=("x", "y")):
    out = []
    seed = 0
    for c in classes:
        for i in range(n_per_class):
            seed += 1
            out.append({"id": f"{c}-{i}", "image_bytes": make_image_bytes(seed=seed), "label": c})
    return out


def test_validate_dataset_rejections_are_actionable():
    good = _records()
    assert validate_dataset(good)["classes"] == ["x", "y"]
    with pytest.raises(TypeError, match="list of"):
        validate_dataset({"id": 1})
    with pytest.raises(ValueError, match=f"at least {MIN_RECORDS}"):
        validate_dataset(good[:2])
    bad = [dict(r) for r in good]
    del bad[0]["label"]
    with pytest.raises(ValueError, match=r"missing required field\(s\) \['label'\]"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[1]["id"] = bad[0]["id"]
    with pytest.raises(ValueError, match="duplicates id"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[2]["image_bytes"] = b"garbage"
    with pytest.raises(ValueError, match="could not be decoded"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[3]["image_bytes"] = bad[4]["image_bytes"]
    with pytest.raises(ValueError, match="duplicates the image"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[4]["image_bytes"] = "not bytes"
    with pytest.raises(ValueError, match="encoded image bytes"):
        validate_dataset(bad)
    one_class = [dict(r, label="x") for r in good]
    with pytest.raises(ValueError, match="at least 2 classes"):
        validate_dataset(one_class)
    thin = good + [{"id": "z-0", "image_bytes": make_image_bytes(seed=99), "label": "z"}]
    with pytest.raises(ValueError, match=f"fewer than {MIN_RECORDS_PER_CLASS}"):
        validate_dataset(thin)
    with pytest.raises(ValueError, match="not in the class list"):
        validate_dataset(good, classes=["x"])
    # an evaluation split may be small when the caller says so
    assert validate_dataset(good[:2] + good[-2:], min_records=2, min_per_class=1)["n_records"] == 4


def test_split_is_stratified_disjoint_and_seeded():
    records = generate_sample_dataset()
    s1 = split_dataset(records, seed=42)
    assert s1 == split_dataset(records, seed=42)
    assert split_dataset(records, seed=1) != s1
    ids = [r["id"] for part in s1.values() for r in part]
    assert len(ids) == len(set(ids)) == len(records)
    for part in s1.values():
        labels = [r["label"] for r in part]
        assert len({labels.count(c) for c in SAMPLE_CLASSES}) == 1  # equal per class
    assert {k: len(v) for k, v in s1.items()} == {"train": 28, "validation": 8, "test": 12}
    with pytest.raises(ValueError, match="sum to less than 1"):
        split_dataset(records, val_fraction=0.6, test_fraction=0.5)


# --- metrics and baselines --------------------------------------------------------------------------


def test_auroc_handles_ties_and_degenerate_labels():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1, 0, 1], [0.5] * 4) == 0.5
    assert auroc([1, 1], [0.1, 0.2]) is None


def test_classification_metrics_multiclass():
    classes = ["a", "b", "c"]
    y = ["a", "b", "c", "a"]
    p = ["a", "b", "a", "a"]
    scores = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.5, 0.2, 0.3], [0.6, 0.2, 0.2]]
    m = classification_metrics(y, p, scores, classes)
    assert m["n"] == 4 and m["accuracy"] == 0.75
    assert m["per_class"]["c"]["recall"] == 0.0 and m["per_class"]["a"]["predicted"] == 3
    assert "one-vs-rest" in m["auroc_definition"] and 0.0 <= m["auroc"] <= 1.0
    with pytest.raises(ValueError, match="outside the class list"):
        classification_metrics(y, ["zzz"] * 4, None, classes)


def test_color_features_and_baseline():
    red = make_image_bytes(color=(250, 10, 10))
    blue = make_image_bytes(color=(10, 10, 250))
    f_red, f_blue = color_features(red), color_features(blue)
    assert len(f_red) == 6 and f_red[0] > 0.9 and f_blue[2] > 0.9
    train = [
        {"id": f"r{i}", "image_bytes": make_image_bytes(color=(240 - i, 10, 10), seed=i), "label": "red"}
        for i in range(4)
    ]
    train += [
        {
            "id": f"b{i}",
            "image_bytes": make_image_bytes(color=(10, 10, 240 - i), seed=10 + i),
            "label": "blue",
        }
        for i in range(4)
    ]
    evaluation = [
        {"id": "e1", "image_bytes": red, "label": "red"},
        {"id": "e2", "image_bytes": blue, "label": "blue"},
    ]
    cb = color_baseline(train, evaluation, ["blue", "red"])
    assert cb["accuracy"] == 1.0 and "colour" in cb["baseline"]
    maj = majority_baseline(train, evaluation, ["blue", "red"])
    assert maj["accuracy"] == 0.5 and maj["predicted_label"] == "blue"


def test_sample_baselines_sit_at_chance_by_design():
    records = generate_sample_dataset()
    splits = split_dataset(records, seed=42)
    classes = list(SAMPLE_CLASSES)
    assert majority_baseline(splits["train"], splits["test"], classes)["accuracy"] == 0.25
    cb = color_baseline(splits["train"], splits["test"], classes)
    # four similarly coloured species: colour must not do much better than the 0.25 majority rule
    assert cb["accuracy"] <= 0.5


# --- BYOD loaders -----------------------------------------------------------------------------------


def test_byod_csv_roundtrip_and_rejections(tmp_path):
    records = generate_sample_dataset()[:16]
    path = write_dataset_csv(records, tmp_path / "birds.csv")
    loaded = load_byod_dataset(path)
    assert [(r["id"], r["label"], r["image_bytes"]) for r in loaded] == [
        (r["id"], r["label"], r["image_bytes"]) for r in records
    ]
    (tmp_path / "bad.csv").write_text("id,picture,label\nm1,x.jpg,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing required column\(s\) \['image'\]"):
        load_byod_dataset(tmp_path / "bad.csv")
    (tmp_path / "gone.csv").write_text("id,image,label\nm1,missing.jpg,x\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="image file not found"):
        load_byod_dataset(tmp_path / "gone.csv")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_byod_dataset(tmp_path / "empty.csv")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "nope.csv")
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported BYOD file type"):
        load_byod_dataset(tmp_path / "x.txt")


def test_byod_directory_layout(tmp_path):
    records = generate_sample_dataset()
    for r in records[:8] + records[12:20]:
        d = tmp_path / "tree" / r["label"]
        d.mkdir(parents=True, exist_ok=True)
        (d / r["file"]).write_bytes(r["image_bytes"])
    loaded = load_byod_dataset(tmp_path / "tree")
    assert len(loaded) == 16 and {r["label"] for r in loaded} == {records[0]["label"], records[12]["label"]}
    assert loaded[0]["id"].startswith(loaded[0]["label"] + "/")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no <label>/<image> files"):
        load_byod_dataset(tmp_path / "empty")


def test_byod_json_and_jsonl(tmp_path):
    records = generate_sample_dataset()[:16]
    write_dataset_csv(records, tmp_path / "birds.csv")  # writes images/ beside it
    rows = [{"id": r["id"], "image": f"images/{r['file']}", "label": r["label"]} for r in records]
    (tmp_path / "d.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert [r["id"] for r in load_byod_dataset(tmp_path / "d.json")] == [r["id"] for r in records]
    assert [r["id"] for r in load_byod_dataset(tmp_path / "d.jsonl")] == [r["id"] for r in records]
    (tmp_path / "obj.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(TypeError, match="top-level array"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "broken.jsonl").write_text('{"id": 1}\n{bad\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        load_byod_dataset(tmp_path / "broken.jsonl")


# --- artifact manifest checks -------------------------------------------------------------------------


def _fake_loaded(tmp_path):
    pipe = BioClip2Pipeline(lambda ims: [[0.0] * 768 for _ in ims], lambda ts: [], "cpu")
    pipe.model = object()
    pipe.weights_dir = tmp_path
    return pipe


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path):
    pipe = _fake_loaded(tmp_path)
    art = tmp_path / "adapter"
    art.mkdir()
    base = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps({"format": ARTIFACT_FORMAT, "base_model": {**base, "model_revision": "0" * 40}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="this package pins"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps({"format": ARTIFACT_FORMAT, "base_model": base, "classes": ["only"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="at least two unique classes"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "format": ARTIFACT_FORMAT,
                "base_model": base,
                "classes": list(SAMPLE_CLASSES),
                "files": [{"path": "adapter.safetensors", "bytes": 1, "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="artifact file missing"):
        pipe.load_artifact(art)
    (art / "adapter.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        pipe.load_artifact(art)
