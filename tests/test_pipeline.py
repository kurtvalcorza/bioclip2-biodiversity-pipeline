"""Offline pipeline tests: identity, snapshot verification, image validation, and the embed /
zero-shot / classify / artifact contracts with injected backends (no weights, no open_clip)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from bioclip2_biodiversity_pipeline import (
    CONFIG_NAME,
    CONTEXT_LENGTH,
    DEFAULT_WEIGHTS_DIR,
    EMBED_DIM,
    IMAGE_SIZE,
    MANIFEST_NAME,
    MAX_IMAGES_PER_CALL,
    MAX_LABELS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    VISION_LAYERS,
    VOCAB_SIZE,
    WEIGHTS_NAME,
    BioClip2Pipeline,
    decode_image,
    image_digest,
    stage_missing_files,
    validate_inputs,
    verify_snapshot,
)
from conftest import make_image_bytes

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "weights" / MODEL_KEY / MANIFEST_NAME
IMG_A = make_image_bytes(seed=1)
IMG_B = make_image_bytes(seed=2)


def _unit(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def _fake(classes=None, logits=None):
    """Image embedder: a unit vector whose first coordinates encode the image size; text embedder:
    a unit vector keyed by prompt length. Enough to exercise the contracts deterministically."""

    def image_embedder(images):
        return [_unit([im.size[0], im.size[1]] + [1.0] * (EMBED_DIM - 2)) for im in images]

    def text_embedder(texts):
        return [_unit([len(t), 1.0] + [0.5] * (EMBED_DIM - 2)) for t in texts]

    pipe = BioClip2Pipeline(image_embedder, text_embedder, "cpu", 100.0)
    if classes is not None:
        pipe.classes = list(classes)
        rows = logits or [[0.0, 1.0]] * MAX_IMAGES_PER_CALL
        pipe._classifier = lambda images: [rows[i] for i in range(len(images))]
    return pipe


def test_identity_constants_and_manifest():
    assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION)
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["modelId"] == MODEL_ID
        assert manifest["revision"] == MODEL_REVISION
        paths = {f["path"] for f in manifest["files"]}
        assert {CONFIG_NAME, WEIGHTS_NAME} <= paths
    config = REPO / "weights" / MODEL_KEY / CONFIG_NAME
    if config.is_file():
        cfg = json.loads(config.read_text(encoding="utf-8"))["model_cfg"]
        assert cfg["embed_dim"] == EMBED_DIM
        assert cfg["vision_cfg"]["image_size"] == IMAGE_SIZE
        assert cfg["vision_cfg"]["layers"] == VISION_LAYERS
        assert cfg["text_cfg"]["context_length"] == CONTEXT_LENGTH
        assert cfg["text_cfg"]["vocab_size"] == VOCAB_SIZE
        assert "quick_gelu" not in cfg  # LAION-2B lineage: plain GELU, and the loader relies on it


def _write_snapshot(
    tmp_path: Path, content: bytes, sha256: str, revision: str = MODEL_REVISION, include_weights: bool = True
) -> Path:
    files = [{"path": CONFIG_NAME, "bytes": len(content), "sha256": sha256}]
    (tmp_path / CONFIG_NAME).write_bytes(content)
    if include_weights:
        (tmp_path / WEIGHTS_NAME).write_bytes(content)
        files.append(
            {"path": WEIGHTS_NAME, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        )
    manifest = {"modelId": MODEL_ID, "revision": revision, "files": files}
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_verify_snapshot_accepts_and_rejects(tmp_path):
    content = b'{"model_cfg": {}}'
    good = hashlib.sha256(content).hexdigest()
    assert verify_snapshot(_write_snapshot(tmp_path, content, good))["revision"] == MODEL_REVISION
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(_write_snapshot(tmp_path, content, "0" * 64))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(_write_snapshot(tmp_path, content, good, revision="0" * 40))
    root = _write_snapshot(tmp_path, content, good)
    (root / CONFIG_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)


def test_verify_snapshot_refuses_a_manifest_without_the_weights(tmp_path):
    content = b"{}"
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest(), include_weights=False)
    with pytest.raises(ValueError, match=WEIGHTS_NAME):
        verify_snapshot(root)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path):
    content = b"{}"
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["files"].append({"path": "README.md", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()})
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    fetched = []

    def downloader(rel, dst):
        fetched.append(rel)
        (dst / rel).write_bytes(b"abc")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == ["README.md"]
    assert fetched == ["README.md"]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []


def test_stage_refuses_manifest_for_another_model(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"modelId": "other/model", "revision": MODEL_REVISION, "files": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True)


# --- image decoding and input validation ---------------------------------------------------------


def test_decode_image_accepts_bytes_paths_and_pil(tmp_path):
    from PIL import Image

    im = decode_image(IMG_A)
    assert im.mode == "RGB" and im.size == (64, 48)
    path = tmp_path / "a.jpg"
    path.write_bytes(IMG_A)
    assert decode_image(path).size == (64, 48)
    assert decode_image(Image.new("L", (40, 40))).mode == "RGB"  # greyscale converted
    assert decode_image(make_image_bytes(fmt="PNG")).size == (64, 48)
    assert decode_image(make_image_bytes(fmt="WEBP")).size == (64, 48)


def test_decode_image_rejections_name_the_rule(tmp_path):
    with pytest.raises(ValueError, match="could not be decoded"):
        decode_image(b"definitely not an image")
    with pytest.raises(ValueError, match="format BMP is not accepted"):
        decode_image(make_image_bytes(fmt="BMP"))
    with pytest.raises(ValueError, match="shorter side must be at least"):
        decode_image(make_image_bytes(size=(16, 100)))
    with pytest.raises(TypeError, match="bytes, a path or a PIL image"):
        decode_image(12345)
    with pytest.raises(FileNotFoundError):
        decode_image(tmp_path / "missing.jpg")


def test_image_digest_is_stable_and_content_addressed(tmp_path):
    assert image_digest(IMG_A) == hashlib.sha256(IMG_A).hexdigest()
    assert image_digest(IMG_A) != image_digest(IMG_B)
    path = tmp_path / "a.jpg"
    path.write_bytes(IMG_A)
    assert image_digest(path) == image_digest(IMG_A)
    assert len(image_digest(decode_image(IMG_A))) == 64


def test_validate_inputs_manifest_and_rejections():
    manifest = validate_inputs([IMG_A, IMG_B], names=["a", "b"])
    assert manifest["verdict"] == "accepted"
    assert manifest["requires_remote_code"] is False
    assert [row["id"] for row in manifest["inputs"]] == ["a", "b"]
    assert manifest["inputs"][0]["width"] == 64 and manifest["inputs"][0]["sha256"] == image_digest(IMG_A)
    with pytest.raises(TypeError, match="list of image"):
        validate_inputs(IMG_A)
    with pytest.raises(ValueError, match="could not be decoded"):
        validate_inputs([b"nope"])
    with pytest.raises(ValueError, match=f"1..{MAX_IMAGES_PER_CALL}"):
        validate_inputs([IMG_A] * (MAX_IMAGES_PER_CALL + 1))
    with pytest.raises(ValueError, match="exactly one id per image"):
        validate_inputs([IMG_A, IMG_B], names=["a"])
    with pytest.raises(ValueError, match="names must be unique"):
        validate_inputs([IMG_A, IMG_B], names=["a", "a"])


# --- embed / zero-shot / classify / artifact contracts -----------------------------------------------


def test_embed_images_contract_with_injected_backend():
    out = _fake().embed_images([IMG_A, IMG_B], names=["a", "b"])
    assert out["ids"] == ["a", "b"] and out["dimension"] == EMBED_DIM
    assert len(out["embeddings"][0]) == EMBED_DIM
    assert out["images"][0]["sha256"] == image_digest(IMG_A)
    assert out["model_revision"] == MODEL_REVISION


def test_embed_images_rejects_backend_shape_drift():
    pipe = BioClip2Pipeline(lambda ims: [[0.0] * 3 for _ in ims], lambda ts: [], "cpu")
    with pytest.raises(RuntimeError, match="wrong shape"):
        pipe.embed_images([IMG_A])


def test_embed_texts_contract_and_rejections():
    out = _fake().embed_texts(["a photo of Melospiza melodia."])
    assert out["n_texts"] == 1 and len(out["embeddings"][0]) == EMBED_DIM
    with pytest.raises(TypeError, match="list of strings"):
        _fake().embed_texts("a photo")
    with pytest.raises(ValueError, match="non-empty"):
        _fake().embed_texts([""])
    with pytest.raises(ValueError, match=f"1..{MAX_LABELS}"):
        _fake().embed_texts(["x"] * (MAX_LABELS + 1))


def test_zero_shot_contract_and_label_handling():
    pipe = _fake()
    out = pipe.zero_shot([IMG_A], {"song": "Melospiza melodia", "junco": "Junco hyemalis"}, names=["a"])
    p = out["predictions"][0]
    assert set(p["scores"]) == {"song", "junco"} and abs(sum(p["scores"].values()) - 1.0) < 1e-9
    assert p["label"] in ("song", "junco") and p["score"] == max(p["scores"].values())
    assert out["prompts"] == {"song": "a photo of Melospiza melodia.", "junco": "a photo of Junco hyemalis."}
    assert out["classes"] == ["song", "junco"]  # order preserved
    assert "not calibrated" in out["decision_rule"]
    # a list of names reports the names themselves as labels
    assert pipe.zero_shot([IMG_A], ["Aves", "Mammalia"])["classes"] == ["Aves", "Mammalia"]
    with pytest.raises(ValueError, match="placeholder"):
        pipe.zero_shot([IMG_A], ["Aves", "Mammalia"], template="a photo")
    with pytest.raises(ValueError, match="2.."):
        pipe.zero_shot([IMG_A], ["Aves"])
    with pytest.raises(ValueError, match="unique"):
        pipe.zero_shot([IMG_A], ["Aves", "Aves"])
    with pytest.raises(TypeError, match="list of taxon names"):
        pipe.zero_shot([IMG_A], "Aves")


def test_classify_requires_adaptation():
    with pytest.raises(RuntimeError, match="adapt"):
        _fake().classify([IMG_A])


def test_classify_contract_preserves_class_order():
    pipe = _fake(classes=["junco", "song"], logits=[[2.0, 0.0], [0.0, 2.0]])
    out = pipe.classify([IMG_A, IMG_B], names=["a", "b"])
    assert [p["label"] for p in out["predictions"]] == ["junco", "song"]
    assert list(out["predictions"][0]["scores"]) == ["junco", "song"]
    assert out["predictions"][0]["scores"]["junco"] == pytest.approx(0.8808, abs=1e-3)
    assert out["predictions"][0]["sha256"] == image_digest(IMG_A)
    assert "not calibrated" in out["decision_rule"]


def test_classify_rejects_logits_that_do_not_match_classes():
    pipe = _fake(classes=["a", "b", "c"], logits=[[0.0, 1.0]] * 4)
    with pytest.raises(RuntimeError, match="does not match the class list"):
        pipe.classify([IMG_A])


def test_save_and_load_artifact_require_a_loaded_model(tmp_path):
    pipe = _fake()
    with pytest.raises(RuntimeError, match="adapt"):
        pipe.save_artifact(tmp_path / "adapter")
    with pytest.raises(RuntimeError, match="from_pretrained"):
        pipe.load_artifact(tmp_path / "adapter")
    with pytest.raises(RuntimeError, match="from_pretrained"):
        pipe.adapt([], [])
