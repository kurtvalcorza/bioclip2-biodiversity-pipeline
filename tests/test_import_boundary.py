"""Import-boundary contract (fleet RTM-001).

Rejected requests never import model libraries; the snapshot is verified before anything is
imported. Pillow is allowed: decoding an image is the validation.
"""

import hashlib
import json

import pytest

from bioclip2_biodiversity_pipeline.pipeline import (
    CONFIG_NAME,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    WEIGHTS_NAME,
    BioClip2Pipeline,
    validate_inputs,
)
from conftest import make_image_bytes

_CONFIG = json.dumps({"model_cfg": {"embed_dim": 768}}).encode()


def _snapshot(root, tamper=False):
    files = []
    for name in (CONFIG_NAME, WEIGHTS_NAME):
        (root / name).write_bytes(_CONFIG)
        digest = "0" * 64 if (tamper and name == CONFIG_NAME) else hashlib.sha256(_CONFIG).hexdigest()
        files.append({"path": name, "bytes": len(_CONFIG), "sha256": digest})
    manifest = {"modelId": MODEL_ID, "revision": MODEL_REVISION, "files": files}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_from_pretrained_refuses_without_manifest_before_model_imports(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="no snapshot manifest"):
        BioClip2Pipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_tampered_snapshot_before_model_imports(tmp_path, forbid_model_imports):
    _snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        BioClip2Pipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_config_drift_is_refused_before_model_imports(tmp_path, forbid_model_imports):
    _snapshot(tmp_path)
    with pytest.raises((ValueError, KeyError)):
        BioClip2Pipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_undecodable_images_are_rejected_before_model_imports(forbid_model_imports):
    with pytest.raises(ValueError, match="could not be decoded"):
        validate_inputs([b"not an image"])
    with pytest.raises(ValueError, match="shorter side"):
        validate_inputs([make_image_bytes(size=(8, 8))])
