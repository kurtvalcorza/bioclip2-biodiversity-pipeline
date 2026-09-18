"""Offline tests for the public validation-stage helpers (input manifest and dataset manifest)."""

from __future__ import annotations

from bioclip2_biodiversity_pipeline import (
    ACCEPTED_FORMATS,
    INPUT_SCHEMA,
    MAX_IMAGES_PER_CALL,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    generate_sample_dataset,
    validate_dataset,
    validate_inputs,
)
from conftest import make_image_bytes


def test_validate_inputs_returns_manifest_with_schema_and_identity():
    manifest = validate_inputs([make_image_bytes(seed=1), make_image_bytes(seed=2)], names=["a", "b"])
    assert manifest["verdict"] == "accepted" and manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["images"] == [1, MAX_IMAGES_PER_CALL]
    assert manifest["schema"]["image_side_pixels"][0] == MIN_IMAGE_SIDE
    assert "a photo of a rock is embedded" in manifest["schema"]["validation"]
    assert manifest["n_images"] == 2
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert manifest["requires_remote_code"] is False


def test_validate_inputs_default_ids():
    manifest = validate_inputs([make_image_bytes()])
    assert [row["id"] for row in manifest["inputs"]] == ["img-0"]


def test_dataset_manifest_reports_ceilings():
    manifest = validate_dataset(generate_sample_dataset())
    assert manifest["ceilings"]["min_image_side"] == MIN_IMAGE_SIDE
    assert manifest["ceilings"]["accepted_formats"] == list(ACCEPTED_FORMATS)
    assert len(manifest["digest"]) == 64
