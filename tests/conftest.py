import builtins
import io

import pytest


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries.

    Pillow is not a model library: decoding an image *is* the validation, so it is allowed."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in {"torch", "open_clip", "safetensors", "timm", "huggingface_hub"}:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def make_image_bytes(color=(120, 160, 90), size=(64, 48), fmt="JPEG", seed=None) -> bytes:
    """A small encoded test image; `seed` adds deterministic pixel noise so digests differ."""
    from PIL import Image

    im = Image.new("RGB", size, color)
    if seed is not None:
        import random

        rnd = random.Random(seed)
        px = im.load()
        for _ in range(64):
            x, y = rnd.randrange(size[0]), rnd.randrange(size[1])
            px[x, y] = (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
    buf = io.BytesIO()
    im.save(buf, fmt)
    return buf.getvalue()
