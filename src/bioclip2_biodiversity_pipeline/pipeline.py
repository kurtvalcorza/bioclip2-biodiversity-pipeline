"""BioCLIP 2 (`imageomics/bioclip-2`) DIMER pipeline: verified snapshot, organism image embeddings,
zero-shot species classification from taxonomic names, and bounded species-classification fine-tuning
with a portable adapter.

BioCLIP 2 is a CLIP model (ViT-L/14 image tower, 12-layer masked-attention text tower, 768-d joint
space) trained on TreeOfLife-200M with hierarchical contrastive learning. It is published in the
`open_clip` checkpoint format, so this package builds the `open_clip.model.CLIP` module from the
**pinned** `open_clip_config.json` (not from open_clip's built-in model registry) and loads the
pinned `open_clip_model.safetensors` with `strict=True`. No remote code is executed: the checkpoint
is a plain state dict and the architecture is open_clip's own.

Two design choices are stated rather than hidden:

* **The text tokenizer is open_clip's bundled CLIP BPE**, not the `tokenizer.json` the upstream
  repository also ships. Both encode the same 49,408-token CLIP vocabulary; identical token ids
  were verified for the tutorial prompts (see docs/WEIGHTS.md). The HF tokenizer files are
  therefore not staged.
* **Adaptation initialises the classification head from the zero-shot text classifier** when class
  prompts are supplied, so fine-tuning starts exactly where zero-shot classification is and every
  epoch's validation number is comparable to the zero-shot number.

Everything model-related is imported lazily so that snapshot verification and input validation run
(and can refuse) before `torch` or `open_clip` are imported (fleet RTM-001).
"""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "imageomics/bioclip-2"
MODEL_REVISION = "2957b322090f9cb17ae72c71981c7218a28d81e0"
MODEL_LICENSE = "mit"
MODEL_KEY = "bioclip-2"
ARTIFACT_FORMAT = "org.valcorza.bioclip2-biodiversity.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
CONFIG_NAME = "open_clip_config.json"
WEIGHTS_NAME = "open_clip_model.safetensors"

# Architecture facts from the pinned open_clip_config.json; asserted against the file at load time.
EMBED_DIM = 768
IMAGE_SIZE = 224
VISION_LAYERS = 24
CONTEXT_LENGTH = 77
VOCAB_SIZE = 49408
# Ceilings. Images are resized to IMAGE_SIZE on the short side and centre-cropped, so anything from
# MIN_IMAGE_SIDE up is accepted; MAX_IMAGE_SIDE bounds decode memory. Batches are bounded so that a
# 1.7 GB tower on a CPU runtime finishes a call in seconds, not minutes.
MIN_IMAGE_SIDE = 32
MAX_IMAGE_SIDE = 8192
MAX_IMAGES_PER_CALL = 32
MAX_LABELS = 64
MAX_PROMPT_CHARS = 200
ACCEPTED_FORMATS = ("JPEG", "PNG", "WEBP")
# BioCLIP's prompt convention: "a photo of <name>." where the name is a taxonomic string. The
# upstream authors trained with full 7-rank taxonomic strings and scientific names; a scientific
# binomial is the recommended minimum.
DEFAULT_PROMPT_TEMPLATE = "a photo of {}."


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    listed = {entry["path"] for entry in manifest["files"]}
    for required in (CONFIG_NAME, WEIGHTS_NAME):
        if required not in listed:
            raise ValueError(f"manifest does not list {required}; refusing to proceed")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest.hexdigest()} != manifest {entry['sha256']}")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the snapshot against its DIMER manifest (size + SHA-256 of every listed file)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _verify_manifest(root, MODEL_ID, MODEL_REVISION)


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest but
    git-ignores the 1.7 GB safetensors file). `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


INPUT_SCHEMA: dict[str, Any] = {
    "input": "1..MAX_IMAGES_PER_CALL images, each as encoded bytes (JPEG, PNG or WEBP) or a PIL image",
    "images": [1, MAX_IMAGES_PER_CALL],
    "image_side_pixels": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "labels_per_zero_shot_call": [2, MAX_LABELS],
    "validation": (
        "decodability, format, mode convertible to RGB and pixel-size ceilings only. Nothing checks "
        "that an image shows an organism, that the organism fills the frame, or that a label names a "
        "real taxon -- a photo of a rock is embedded and classified without complaint"
    ),
    "preprocessing": (
        "resize so the shorter side is 224 px (bicubic), centre-crop 224x224, convert to RGB, "
        "normalise with the CLIP mean/std from the pinned open_clip_config.json; embeddings are "
        "L2-normalised 768-d vectors from the image tower's projection (CLIP convention)"
    ),
}


def decode_image(image: Any) -> Any:
    """Return a PIL RGB image from bytes, a file path, or a PIL image; raise on anything else.

    Pillow is imported here (not a model library). Decoding is the only way to know an image is
    valid, so validation and execution share this function and cannot diverge.
    """
    from PIL import Image, UnidentifiedImageError

    if isinstance(image, Image.Image):
        im = image
    elif isinstance(image, bytes | bytearray):
        try:
            im = Image.open(io.BytesIO(bytes(image)))
        except UnidentifiedImageError as exc:
            raise ValueError("image bytes could not be decoded as JPEG, PNG or WEBP") from exc
    elif isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"image file not found: {path}")
        try:
            im = Image.open(path)
        except UnidentifiedImageError as exc:
            raise ValueError(f"{path.name} could not be decoded as JPEG, PNG or WEBP") from exc
    else:
        raise TypeError(f"image must be bytes, a path or a PIL image, got {type(image).__name__}")
    fmt = getattr(im, "format", None)
    if fmt is not None and fmt not in ACCEPTED_FORMATS:
        raise ValueError(f"image format {fmt} is not accepted; use one of {list(ACCEPTED_FORMATS)}")
    width, height = im.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image is {width}x{height}; the shorter side must be at least {MIN_IMAGE_SIDE} px")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image is {width}x{height}; the longer side must be at most {MAX_IMAGE_SIDE} px")
    im.load()
    return im.convert("RGB") if im.mode != "RGB" else im


def image_digest(image: Any) -> str:
    """SHA-256 of the encoded bytes (or, for a PIL image, of its raw RGB pixels and size)."""
    if isinstance(image, bytes | bytearray):
        return hashlib.sha256(bytes(image)).hexdigest()
    if isinstance(image, str | Path):
        return hashlib.sha256(Path(image).read_bytes()).hexdigest()
    im = decode_image(image)
    h = hashlib.sha256(f"{im.size[0]}x{im.size[1]}".encode())
    h.update(im.tobytes())
    return h.hexdigest()


def _check_images(images: Any, names: Any = None) -> tuple[list[Any], list[str], list[dict[str, Any]]]:
    """Raise TypeError/ValueError naming the first violated rule; return (pil_images, ids, facts).

    ``embed_images``, ``zero_shot``, ``classify`` and ``validate_inputs`` all route through this
    function so their acceptance criteria cannot diverge.
    """
    if isinstance(images, str | bytes | bytearray | Path) or not isinstance(images, Sequence):
        raise TypeError("images must be a list of image bytes, paths or PIL images")
    if not 1 <= len(images) <= MAX_IMAGES_PER_CALL:
        raise ValueError(f"images must hold 1..{MAX_IMAGES_PER_CALL} items, got {len(images)}")
    decoded: list[Any] = []
    facts: list[dict[str, Any]] = []
    for i, image in enumerate(images):
        try:
            im = decode_image(image)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            raise type(exc)(f"images[{i}]: {exc}") from exc
        decoded.append(im)
        facts.append({"width": im.size[0], "height": im.size[1], "sha256": image_digest(image)})
    if names is None:
        ids = [f"img-{i}" for i in range(len(decoded))]
    else:
        if isinstance(names, str | bytes) or not isinstance(names, Sequence) or len(names) != len(decoded):
            raise ValueError("names must be a list with exactly one id per image")
        ids = [str(n) for n in names]
        if len(set(ids)) != len(ids):
            raise ValueError("names must be unique")
    return decoded, ids, facts


def _check_labels(labels: Any) -> list[tuple[str, str]]:
    """Return (label, prompt_text) pairs from a list of names or a {label: name} mapping."""
    if isinstance(labels, Mapping):
        pairs = [(str(k), str(v)) for k, v in labels.items()]
    elif isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("labels must be a list of taxon names or a {label: taxon name} mapping")
    else:
        pairs = [(str(x), str(x)) for x in labels]
    if not 2 <= len(pairs) <= MAX_LABELS:
        raise ValueError(f"zero-shot classification needs 2..{MAX_LABELS} labels, got {len(pairs)}")
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError("labels must be unique")
    for label, text in pairs:
        if not label.strip() or not text.strip():
            raise ValueError("labels and their prompt names must be non-empty")
        if len(text) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt name for {label!r} exceeds {MAX_PROMPT_CHARS} characters")
    return pairs


def _softmax(logits: Sequence[float]) -> list[float]:
    import math

    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


@dataclass
class BioClip2Pipeline:
    """BioCLIP 2 pipeline: `embed_images`, `embed_texts` and `zero_shot` always; `classify` after
    `adapt` or `from_artifact`."""

    _image_embedder: Callable[[list[Any]], list[list[float]]]
    _text_embedder: Callable[[list[str]], list[list[float]]]
    device: str
    logit_scale: float = 100.0
    load_warnings: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    _classifier: Callable[[list[Any]], list[list[float]]] | None = None
    model: Any = None
    preprocess: Any = None
    tokenizer: Any = None
    classifier_model: Any = None
    weights_dir: Path | None = None
    adaptation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BioClip2Pipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(f"no snapshot manifest at {root} and allow_download={allow_download}")
        # Stage and verify before importing model libraries (RTM-001).
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        cfg = json.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
        model_cfg, pre_cfg = cfg["model_cfg"], cfg["preprocess_cfg"]
        expected = {
            "embed_dim": (model_cfg["embed_dim"], EMBED_DIM),
            "image_size": (model_cfg["vision_cfg"]["image_size"], IMAGE_SIZE),
            "vision_layers": (model_cfg["vision_cfg"]["layers"], VISION_LAYERS),
            "context_length": (model_cfg["text_cfg"]["context_length"], CONTEXT_LENGTH),
            "vocab_size": (model_cfg["text_cfg"]["vocab_size"], VOCAB_SIZE),
        }
        drift = {k: v for k, v in expected.items() if v[0] != v[1]}
        if drift:
            raise ValueError(f"pinned open_clip_config.json disagrees with the package constants: {drift}")

        import torch
        from open_clip import image_transform
        from open_clip.model import CLIP
        from open_clip.tokenizer import SimpleTokenizer
        from safetensors.torch import load_file

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Built from the pinned config, not from open_clip's registry, so the architecture the
            # weights load into is the one the snapshot describes. `quick_gelu` is absent from the
            # pinned config and defaults to False (LAION-2B lineage, not OpenAI's).
            model = CLIP(
                embed_dim=model_cfg["embed_dim"],
                vision_cfg=model_cfg["vision_cfg"],
                text_cfg=model_cfg["text_cfg"],
                quick_gelu=bool(model_cfg.get("quick_gelu", False)),
            )
            state = load_file(str(root / WEIGHTS_NAME))
            model.load_state_dict(state, strict=True)
            del state
            preprocess = image_transform(
                model_cfg["vision_cfg"]["image_size"],
                is_train=False,
                mean=tuple(pre_cfg["mean"]),
                std=tuple(pre_cfg["std"]),
                resize_mode=pre_cfg.get("resize_mode", "shortest"),
                interpolation=pre_cfg.get("interpolation", "bicubic"),
            )
            tokenizer = SimpleTokenizer(context_length=model_cfg["text_cfg"]["context_length"])
        model = model.to(resolved_device).eval()
        messages = [f"{w.category.__name__}: {w.message}" for w in caught]
        pipe = cls(
            cls._make_image_embedder(model, preprocess, resolved_device),
            cls._make_text_embedder(model, tokenizer, resolved_device),
            resolved_device,
            float(model.logit_scale.exp().item()),
            messages,
        )
        pipe.model, pipe.preprocess, pipe.tokenizer, pipe.weights_dir = model, preprocess, tokenizer, root
        return pipe

    # -- backends ---------------------------------------------------------------------------------

    @classmethod
    def _make_image_embedder(
        cls, model: Any, preprocess: Any, device: str
    ) -> Callable[[list[Any]], list[list[float]]]:
        import torch

        def embedder(images: list[Any]) -> list[list[float]]:
            batch = torch.stack([preprocess(im) for im in images]).to(device)
            # no_grad, not inference_mode: tensors produced here must stay usable by a later
            # training epoch that shares this module.
            with torch.no_grad():
                feats = model.encode_image(batch, normalize=True)
            return feats.float().cpu().tolist()

        return embedder

    @classmethod
    def _make_text_embedder(
        cls, model: Any, tokenizer: Any, device: str
    ) -> Callable[[list[str]], list[list[float]]]:
        import torch

        def embedder(texts: list[str]) -> list[list[float]]:
            tokens = tokenizer(texts).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens, normalize=True)
            return feats.float().cpu().tolist()

        return embedder

    @classmethod
    def _make_classifier(
        cls, clf: Any, preprocess: Any, device: str
    ) -> Callable[[list[Any]], list[list[float]]]:
        import torch

        def classifier(images: list[Any]) -> list[list[float]]:
            batch = torch.stack([preprocess(im) for im in images]).to(device)
            with torch.no_grad():
                logits = clf(batch)
            return logits.float().cpu().tolist()

        return classifier

    # -- public stages ----------------------------------------------------------------------------

    def embed_images(self, images: Sequence[Any], *, names: Sequence[str] | None = None) -> dict[str, Any]:
        """L2-normalised image-tower embedding per image (EMBED_DIM floats each)."""
        decoded, ids, facts = _check_images(images, names)
        vectors = self._image_embedder(decoded)
        if len(vectors) != len(decoded) or any(len(v) != EMBED_DIM for v in vectors):
            raise RuntimeError("backend returned embeddings of the wrong shape")
        return {
            "ids": ids,
            "embeddings": [[float(x) for x in v] for v in vectors],
            "dimension": EMBED_DIM,
            "pooling": "image-tower projection, L2-normalised (CLIP joint space)",
            "unit": "one unit vector per image; representations, not species predictions",
            "images": [{"id": i, **f} for i, f in zip(ids, facts, strict=True)],
            "n_images": len(decoded),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def embed_texts(self, texts: Sequence[str]) -> dict[str, Any]:
        """L2-normalised text-tower embedding per string (same joint space as the images)."""
        if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a list of strings")
        if not 1 <= len(texts) <= MAX_LABELS:
            raise ValueError(f"texts must hold 1..{MAX_LABELS} items, got {len(texts)}")
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"texts[{i}] must be a non-empty string")
            if len(t) > MAX_PROMPT_CHARS:
                raise ValueError(f"texts[{i}] exceeds {MAX_PROMPT_CHARS} characters")
        vectors = self._text_embedder([str(t) for t in texts])
        if len(vectors) != len(texts) or any(len(v) != EMBED_DIM for v in vectors):
            raise RuntimeError("backend returned embeddings of the wrong shape")
        return {
            "texts": list(texts),
            "embeddings": [[float(x) for x in v] for v in vectors],
            "dimension": EMBED_DIM,
            "n_texts": len(texts),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def zero_shot(
        self,
        images: Sequence[Any],
        labels: Sequence[str] | Mapping[str, str],
        *,
        names: Sequence[str] | None = None,
        template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> dict[str, Any]:
        """Classify images against a label set with no training: softmax over scaled cosine
        similarities between each image embedding and the text embedding of each label's prompt.

        `labels` is a list of taxon names, or a `{label: taxon name}` mapping when the reported
        label should differ from the text put into the prompt (e.g. a dataset key -> a scientific
        name). `template` must contain one `{}` placeholder.
        """
        if "{}" not in template:
            raise ValueError("template must contain a '{}' placeholder for the taxon name")
        pairs = _check_labels(labels)
        decoded, ids, facts = _check_images(images, names)
        prompts = [template.format(text) for _, text in pairs]
        text_vecs = self._text_embedder(prompts)
        image_vecs = self._image_embedder(decoded)
        classes = [label for label, _ in pairs]
        predictions = []
        for iid, fact, iv in zip(ids, facts, image_vecs, strict=True):
            logits = [self.logit_scale * sum(a * b for a, b in zip(iv, tv, strict=True)) for tv in text_vecs]
            scores = _softmax(logits)
            best = max(range(len(scores)), key=scores.__getitem__)
            predictions.append(
                {
                    "id": iid,
                    "sha256": fact["sha256"],
                    "label": classes[best],
                    "score": scores[best],
                    "scores": dict(zip(classes, scores, strict=True)),
                }
            )
        return {
            "predictions": predictions,
            "classes": classes,
            "prompts": dict(zip(classes, prompts, strict=True)),
            "template": template,
            "logit_scale": self.logit_scale,
            "decision_rule": (
                "argmax over softmax(logit_scale * cosine(image, text)); scores are relative to the "
                "supplied label set only and are not calibrated probabilities"
            ),
            "n_images": len(decoded),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def zero_shot_evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        class_prompts: Mapping[str, str],
        *,
        template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> dict[str, Any]:
        """Zero-shot metrics on a labelled dataset: the model's own prior, before any training."""
        from .metrics import classification_metrics
        from .samples import validate_dataset

        # An evaluation split may legitimately be small; the coverage rule applies to training data.
        manifest = validate_dataset(records, classes=list(class_prompts), min_records=2, min_per_class=1)
        classes = list(manifest["classes"])
        predicted: list[str] = []
        scores: list[list[float]] = []
        for start in range(0, len(records), MAX_IMAGES_PER_CALL):
            chunk = records[start : start + MAX_IMAGES_PER_CALL]
            result = self.zero_shot(
                [r["image_bytes"] for r in chunk],
                {c: class_prompts[c] for c in classes},
                names=[r["id"] for r in chunk],
                template=template,
            )
            for p in result["predictions"]:
                predicted.append(p["label"])
                scores.append([p["scores"][c] for c in classes])
        metrics = classification_metrics([r["label"] for r in records], predicted, scores, classes)
        return {"baseline": "zero-shot (text classifier, no training)", "template": template, **metrics}

    def classify(self, images: Sequence[Any], *, names: Sequence[str] | None = None) -> dict[str, Any]:
        """Class scores and argmax label per image from the adapted head; requires a prior `adapt`
        or `from_artifact`."""
        if self._classifier is None or not self.classes:
            raise RuntimeError(
                "classify requires an adapted head: call adapt(...) or load from_artifact(...) first"
            )
        decoded, ids, facts = _check_images(images, names)
        logits = self._classifier(decoded)
        predictions = []
        for iid, fact, row in zip(ids, facts, logits, strict=True):
            if len(row) != len(self.classes):
                raise RuntimeError("backend returned a logits row that does not match the class list")
            scores = _softmax(row)
            best = max(range(len(scores)), key=scores.__getitem__)
            predictions.append(
                {
                    "id": iid,
                    "sha256": fact["sha256"],
                    "label": self.classes[best],
                    "score": scores[best],
                    "scores": dict(zip(self.classes, scores, strict=True)),
                }
            )
        return {
            "predictions": predictions,
            "classes": list(self.classes),
            "decision_rule": (
                "argmax over softmax(logits); scores are softmax outputs, not calibrated probabilities"
            ),
            "n_images": len(decoded),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adaptation": dict(self.adaptation),
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Held-out species-classification metrics (see metrics.classification_metrics)."""
        from .metrics import classification_metrics
        from .samples import validate_dataset

        validate_dataset(records, classes=self.classes, min_records=2, min_per_class=1)
        predicted: list[str] = []
        scores: list[list[float]] = []
        for start in range(0, len(records), MAX_IMAGES_PER_CALL):
            chunk = records[start : start + MAX_IMAGES_PER_CALL]
            result = self.classify([r["image_bytes"] for r in chunk], names=[r["id"] for r in chunk])
            for p in result["predictions"]:
                predicted.append(p["label"])
                scores.append([p["scores"][c] for c in self.classes])
        return classification_metrics([r["label"] for r in records], predicted, scores, self.classes)

    def _build_classifier(self, n_classes: int) -> Any:
        """A fresh copy of the image tower with a linear head on its (unnormalised) projection."""
        import copy

        import torch

        class ImageClassifier(torch.nn.Module):
            def __init__(self, visual: Any, embed_dim: int, n: int) -> None:
                super().__init__()
                self.visual = visual
                self.head = torch.nn.Linear(embed_dim, n, bias=True)

            def forward(self, pixels: Any) -> Any:
                feats = self.visual(pixels)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                return self.head(feats)

        return ImageClassifier(copy.deepcopy(self.model.visual), EMBED_DIM, n_classes)

    def adapt(
        self,
        train_records: Sequence[Mapping[str, Any]],
        val_records: Sequence[Mapping[str, Any]] | None = None,
        *,
        classes: Sequence[str] | None = None,
        class_prompts: Mapping[str, str] | None = None,
        template: str = DEFAULT_PROMPT_TEMPLATE,
        epochs: int = 4,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        trainable_blocks: int = 0,
        weight_decay: float = 0.01,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Bounded gradient fine-tuning of a species-classification head on the verified base.

        Copies the image tower, adds a linear head over its L2-normalised projection, freezes
        everything except the head and the last `trainable_blocks` transformer blocks of the tower,
        and runs AdamW for `epochs` passes. When `class_prompts` maps every class to a taxon name,
        the head's weights are initialised from the zero-shot text classifier (scaled by the
        checkpoint's logit scale) so epoch 0 *is* zero-shot classification; otherwise the head is
        randomly initialised. Validation records are monitored per epoch only; the final epoch's
        weights are kept (no selection).

        `trainable_blocks=0` (the default) trains the head alone on image features computed once by
        the frozen tower, which is what a few-dozen-image dataset supports: on the tutorial sample,
        unfreezing even one block of the 24 degraded a saturated zero-shot classifier (recorded in
        MODEL_CARD.md). With `trainable_blocks>0` the tower runs every epoch and the unfrozen blocks
        receive gradients.
        """
        if self.model is None or self.preprocess is None or self.weights_dir is None:
            raise RuntimeError("adapt requires a pipeline built by from_pretrained (no loaded base model)")
        from .samples import validate_dataset

        if not 1 <= int(epochs) <= 50:
            raise ValueError("epochs must be in 1..50 (tutorial-scale adaptation)")
        if not 1 <= int(batch_size) <= MAX_IMAGES_PER_CALL:
            raise ValueError(f"batch_size must be in 1..{MAX_IMAGES_PER_CALL}")
        if not 0 <= int(trainable_blocks) <= VISION_LAYERS:
            raise ValueError(
                f"trainable_blocks must be in 0..{VISION_LAYERS} (the image tower has that many)"
            )
        train_manifest = validate_dataset(train_records, classes=classes)
        class_list = list(train_manifest["classes"])
        if val_records is not None:
            validate_dataset(val_records, classes=class_list, min_records=2, min_per_class=1)
        head_init = "random"
        if class_prompts is not None:
            missing = [c for c in class_list if c not in class_prompts]
            if missing:
                raise ValueError(f"class_prompts lacks an entry for classes {missing}")
            _check_labels({c: class_prompts[c] for c in class_list})
            head_init = "zero-shot text classifier"

        import random

        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        clf = self._build_classifier(len(class_list))
        if head_init != "random":
            prompts = [template.format(class_prompts[c]) for c in class_list]  # type: ignore[index]
            text_vecs = torch.tensor(self._text_embedder(prompts), dtype=torch.float32)
            with torch.no_grad():
                clf.head.weight.copy_(text_vecs * self.logit_scale)
                clf.head.bias.zero_()
        clf = clf.to(self.device)
        for p in clf.parameters():
            p.requires_grad = False
        blocks = clf.visual.transformer.resblocks
        for block in blocks[len(blocks) - int(trainable_blocks) :] if trainable_blocks else []:
            for p in block.parameters():
                p.requires_grad = True
        for p in clf.head.parameters():
            p.requires_grad = True
        trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in clf.parameters())
        optimizer = torch.optim.AdamW(
            [p for p in clf.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay
        )
        label_index = {c: i for i, c in enumerate(class_list)}
        examples = [(decode_image(r["image_bytes"]), label_index[r["label"]]) for r in train_records]
        self.classes = class_list
        self.classifier_model = clf
        self._classifier = self._make_classifier(clf, self.preprocess, self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        cached: list[Any] | None = None
        if not trainable_blocks:
            # Frozen tower: its output for a fixed image never changes, so compute it once.
            clf.eval()
            cached = []
            with torch.no_grad():
                for start in range(0, len(examples), MAX_IMAGES_PER_CALL):
                    pixels = torch.stack(
                        [self.preprocess(im) for im, _ in examples[start : start + MAX_IMAGES_PER_CALL]]
                    ).to(self.device)
                    feats = clf.visual(pixels)
                    cached.extend(feats / feats.norm(dim=-1, keepdim=True))

        history: list[dict[str, Any]] = []
        if val_records:
            clf.eval()
            val0 = self.evaluate(val_records)
            history.append(
                {
                    "epoch": 0,
                    "train_loss": None,
                    "n_batches": 0,
                    "val_accuracy": val0["accuracy"],
                    "val_macro_f1": val0["macro_f1"],
                    "note": f"before training; head = {head_init}",
                }
            )
        for epoch in range(1, int(epochs) + 1):
            clf.train()
            order = list(range(len(examples)))
            random.shuffle(order)
            total_loss, n_batches = 0.0, 0
            for start in range(0, len(order), int(batch_size)):
                idx = order[start : start + int(batch_size)]
                labels = torch.tensor([examples[i][1] for i in idx], device=self.device)
                optimizer.zero_grad()
                if cached is not None:
                    logits = clf.head(torch.stack([cached[i] for i in idx]))
                else:
                    pixels = torch.stack([self.preprocess(examples[i][0]) for i in idx]).to(self.device)
                    logits = clf(pixels)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
            clf.eval()
            entry: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(total_loss / max(1, n_batches), 6),
                "n_batches": n_batches,
            }
            if val_records:
                val = self.evaluate(val_records)
                entry["val_accuracy"] = val["accuracy"]
                entry["val_macro_f1"] = val["macro_f1"]
            history.append(entry)
        clf.eval()
        self.adaptation = {
            "method": "gradient fine-tuning (AdamW) of a linear head over the L2-normalised image projection"
            + (
                f" and the last {int(trainable_blocks)} image-tower block(s)"
                if trainable_blocks
                else " (frozen tower; features computed once)"
            ),
            "head_initialisation": head_init,
            "template": template if head_init != "random" else None,
            "classes": class_list,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
            "trainable_blocks": int(trainable_blocks),
            "seed": int(seed),
            "precision": "float32",
            "trainable_parameters": int(n_trainable),
            "total_parameters": int(n_total),
            "trainable_parameter_names": trainable,
            "train_records": len(train_records),
            "val_records": len(val_records) if val_records else 0,
            "selection": "final epoch kept; validation metrics are monitoring only",
            "history": history,
        }
        return dict(self.adaptation)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Export the trainable tensors as safetensors plus a JSON manifest binding them to the base."""
        if self.classifier_model is None or not self.classes:
            raise RuntimeError("save_artifact requires an adapted head (call adapt first)")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adaptation.get("trainable_parameter_names", []))
        state = self.classifier_model.state_dict()
        tensors = {k: v.detach().cpu().contiguous() for k, v in state.items() if k in names}
        if not tensors:
            raise RuntimeError("no trainable tensors recorded; nothing to export")
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path))
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "license": MODEL_LICENSE},
            "requires_remote_code": False,
            "classes": list(self.classes),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": digest}
            ],
            "tensors": sorted(tensors),
            "serving_state_tensors": [],
            "adaptation": {k: v for k, v in self.adaptation.items() if k != "trainable_parameter_names"},
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Rebuild the classification head from an exported artifact (manifest verified before loading)."""
        if self.model is None or self.weights_dir is None:
            raise RuntimeError("load_artifact requires a pipeline built by from_pretrained")
        art = Path(artifact_dir)
        manifest_path = art / ARTIFACT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("model_id"), base.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError(f"artifact was trained on {base}, this package pins {MODEL_ID}@{MODEL_REVISION}")
        classes = [str(c) for c in manifest.get("classes", [])]
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("artifact manifest must list at least two unique classes")
        for entry in manifest["files"]:
            fp = art / entry["path"]
            if not fp.is_file():
                raise FileNotFoundError(f"artifact file missing: {fp}")
            if fp.stat().st_size != entry["bytes"]:
                raise ValueError(f"{entry['path']}: size {fp.stat().st_size} != manifest {entry['bytes']}")
            if hashlib.sha256(fp.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"{entry['path']}: sha256 mismatch against the artifact manifest")
        from safetensors.torch import load_file

        clf = self._build_classifier(len(classes))
        tensors = load_file(str(art / ARTIFACT_WEIGHTS_NAME))
        if set(tensors) != set(manifest.get("tensors", [])):
            raise ValueError("artifact tensors do not match the names listed in its manifest")
        _missing, unexpected = clf.load_state_dict(tensors, strict=False)
        if unexpected:
            raise ValueError(
                f"artifact carries tensors the base architecture does not have: {sorted(unexpected)[:5]}"
            )
        clf = clf.to(self.device).eval()
        self.classes = classes
        self.classifier_model = clf
        self._classifier = self._make_classifier(clf, self.preprocess, self.device)
        self.adaptation = {**manifest.get("adaptation", {}), "loaded_from_artifact": str(art)}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BioClip2Pipeline:
        """Verified base snapshot + exported adapter, ready for `classify`."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe


def validate_inputs(images: Sequence[Any], *, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, verdict).

    Decodes every image (that is the only real check), records its size and digest, and raises
    exactly what `embed_images` / `classify` would raise. Nothing here knows what the image shows.
    """
    _decoded, ids, facts = _check_images(images, names)
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": i, **f} for i, f in zip(ids, facts, strict=True)],
        "n_images": len(ids),
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "requires_remote_code": False,
    }
