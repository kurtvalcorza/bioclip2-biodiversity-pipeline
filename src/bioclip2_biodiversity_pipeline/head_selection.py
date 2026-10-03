"""Head-only adaptation with a predeclared learning-rate grid and validation selection.

`BioClip2Pipeline.adapt` initialises the linear head from the zero-shot text classifier, i.e. the
unit text embeddings scaled by the checkpoint's logit scale (100), so a head weight is about 2.9 on
average. AdamW at the tutorial's former single rate of 1e-4 moves each weight by at most about 1e-4
per step, so 16 steps leave the head at the zero-shot classifier (a 0.02 % displacement in the
review of the standalone tutorial, finding BIO-M2). The Philippine biodiversity capstone in this
repository fixed the same defect (its finding BC-M1, `tools/biodiversity_capstone.py`
`fit_head` / `head_change`); this module carries that fix to the package so the standalone
tutorial can use it without changing `pipeline.py`, which the capstone carries byte for byte.

The procedure: every rate in a predeclared grid starts from the same zero-shot initialisation and
seed and trains the head on image features the frozen tower computes once. Epoch 0 (the zero-shot
classifier itself) competes for every rate, so "no change" can legitimately win. The kept state is
the (rate, epoch) with the lowest validation cross-entropy; ties keep the earlier epoch, then the
smaller rate. The record reports how far the kept head moved from zero-shot and how many
predictions changed. Validation chooses; the test split never does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import DEFAULT_PROMPT_TEMPLATE, MAX_IMAGES_PER_CALL, _check_labels, decode_image
from .samples import validate_dataset

LEARNING_RATES: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
SELECTION_RULE = (
    "validation cross-entropy over every (learning rate, epoch) of the predeclared grid; epoch 0 is the "
    "zero-shot classifier; ties keep the earlier epoch, then the smaller rate; the test split is never used"
)


def fit_head(
    train_x: Any,
    train_y: Sequence[int],
    val_x: Any,
    val_y: Sequence[int],
    text: Any,
    scale: float,
    *,
    epochs: int = 4,
    learning_rates: Sequence[float] = LEARNING_RATES,
    batch_size: int = 8,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Fit a linear head on fixed features for every rate in the grid and select on validation.

    `train_x`/`val_x` are L2-normalised image features, `text` the unit text embeddings of the class
    prompts in class order and `scale` the logit scale. Returns the selected `head.weight`/`head.bias`
    tensors, the full history (one row per rate and epoch) and the selection record.
    """
    import torch

    rates = [float(r) for r in learning_rates]
    if not rates or any(r < 0 for r in rates):
        raise ValueError("learning_rates must be a non-empty list of non-negative rates")
    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    val_x = torch.as_tensor(val_x, dtype=torch.float32)
    train_y = torch.as_tensor(list(train_y), dtype=torch.long)
    val_y = torch.as_tensor(list(val_y), dtype=torch.long)
    initial = torch.as_tensor(text, dtype=torch.float32) * float(scale)
    loss_fn = torch.nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    best_key: tuple[float, int, int] | None = None
    best: dict[str, Any] = {}
    for rate_index, rate in enumerate(rates):
        torch.manual_seed(seed)
        head = torch.nn.Linear(train_x.shape[1], initial.shape[0])
        with torch.no_grad():
            head.weight.copy_(initial)
            head.bias.zero_()
        optimizer = torch.optim.AdamW(head.parameters(), lr=rate, weight_decay=weight_decay)
        for epoch in range(int(epochs) + 1):
            if epoch:
                for ix in torch.randperm(len(train_y)).split(int(batch_size)):
                    optimizer.zero_grad()
                    loss_fn(head(train_x[ix]), train_y[ix]).backward()
                    optimizer.step()
            with torch.no_grad():
                train_loss = float(loss_fn(head(train_x), train_y))
                val_logits = head(val_x)
                val_loss = float(loss_fn(val_logits, val_y))
                val_accuracy = float((val_logits.argmax(1) == val_y).float().mean())
            history.append(
                {
                    "learning_rate": rate,
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_accuracy, 4),
                }
            )
            key = (val_loss, epoch, rate_index)
            if best_key is None or key < best_key:
                best_key = key
                best = {"head." + k: v.detach().clone() for k, v in head.state_dict().items()}
    assert best_key is not None
    selection = {"learning_rate": rates[best_key[2]], "epoch": best_key[1], "val_loss": round(best_key[0], 6)}
    return best, history, selection


def head_change(features: Any, tensors: Mapping[str, Any], text: Any, scale: float) -> dict[str, Any]:
    """How far a head moved from its zero-shot initialisation, measured on the given features."""
    import numpy as np

    def array(value: Any) -> Any:
        data = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        return np.asarray(data, dtype=np.float64)

    weight, bias = array(tensors["head.weight"]), array(tensors["head.bias"])
    initial = array(text) * float(scale)
    feats = array(features)
    head_logits = feats @ weight.T + bias
    zero_shot_logits = feats @ initial.T
    return {
        "relative_weight_change": float(np.linalg.norm(weight - initial) / np.linalg.norm(initial)),
        "max_abs_weight_change": float(np.max(np.abs(weight - initial))),
        "max_abs_bias": float(np.max(np.abs(bias))),
        "max_abs_logit_change": float(np.max(np.abs(head_logits - zero_shot_logits))),
        "predictions_changed_vs_zero_shot": int(np.sum(head_logits.argmax(1) != zero_shot_logits.argmax(1))),
        "records_compared": int(len(head_logits)),
    }


def _features(pipe: Any, records: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    out: list[list[float]] = []
    for start in range(0, len(records), MAX_IMAGES_PER_CALL):
        chunk = records[start : start + MAX_IMAGES_PER_CALL]
        out.extend(pipe._image_embedder([decode_image(r["image_bytes"]) for r in chunk]))
    return out


def select_head(
    pipe: Any,
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
    *,
    classes: Sequence[str] | None = None,
    class_prompts: Mapping[str, str],
    template: str = DEFAULT_PROMPT_TEMPLATE,
    epochs: int = 4,
    learning_rates: Sequence[float] = LEARNING_RATES,
    batch_size: int = 8,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    """Head-only adaptation of `pipe` from the zero-shot classifier, with validation selection.

    Leaves `pipe` adapted exactly as `pipe.adapt(..., trainable_blocks=0)` would (same classifier
    structure, same exported tensor names), but with the selected head, and returns the adaptation
    record, which adds the grid, the selection and the head's distance from zero-shot.
    """
    if pipe.model is None or pipe.preprocess is None or pipe.weights_dir is None:
        raise RuntimeError("select_head requires a pipeline built by from_pretrained (no loaded base model)")
    if not 1 <= int(epochs) <= 50:
        raise ValueError("epochs must be in 1..50 (tutorial-scale adaptation)")
    if not 1 <= int(batch_size) <= MAX_IMAGES_PER_CALL:
        raise ValueError(f"batch_size must be in 1..{MAX_IMAGES_PER_CALL}")
    train_manifest = validate_dataset(train_records, classes=classes)
    class_list = list(train_manifest["classes"])
    validate_dataset(val_records, classes=class_list, min_records=2, min_per_class=1)
    missing = [c for c in class_list if c not in class_prompts]
    if missing:
        raise ValueError(f"class_prompts lacks an entry for classes {missing}")
    _check_labels({c: class_prompts[c] for c in class_list})

    import torch

    index = {c: i for i, c in enumerate(class_list)}
    text = pipe._text_embedder([template.format(class_prompts[c]) for c in class_list])
    train_x, val_x = _features(pipe, train_records), _features(pipe, val_records)
    train_y = [index[r["label"]] for r in train_records]
    val_y = [index[r["label"]] for r in val_records]
    tensors, history, selection = fit_head(
        train_x,
        train_y,
        val_x,
        val_y,
        text,
        pipe.logit_scale,
        epochs=epochs,
        learning_rates=learning_rates,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seed=seed,
    )
    change = head_change(train_x + val_x, tensors, text, pipe.logit_scale)

    clf = pipe._build_classifier(len(class_list))
    with torch.no_grad():
        clf.head.weight.copy_(tensors["head.weight"])
        clf.head.bias.copy_(tensors["head.bias"])
    for p in clf.parameters():
        p.requires_grad = False
    for p in clf.head.parameters():
        p.requires_grad = True
    clf = clf.to(pipe.device).eval()
    pipe.classes = class_list
    pipe.classifier_model = clf
    pipe._classifier = pipe._make_classifier(clf, pipe.preprocess, pipe.device)
    pipe.adaptation = {
        "method": (
            "gradient fine-tuning (AdamW) of a linear head over the L2-normalised image projection "
            "(frozen tower; features computed once), learning rate and epoch selected on validation"
        ),
        "head_initialisation": "zero-shot text classifier",
        "template": template,
        "classes": class_list,
        "epochs": int(selection["epoch"]),
        "max_epochs": int(epochs),
        "learning_rate": float(selection["learning_rate"]),
        "learning_rate_grid": [float(r) for r in learning_rates],
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay),
        "trainable_blocks": 0,
        "seed": int(seed),
        "precision": "float32",
        "trainable_parameters": int(sum(p.numel() for p in clf.head.parameters())),
        "total_parameters": int(sum(p.numel() for p in clf.parameters())),
        "trainable_parameter_names": ["head.bias", "head.weight"],
        "train_records": len(train_records),
        "val_records": len(val_records),
        "selection": SELECTION_RULE,
        "selected": dict(selection),
        "head_equals_zero_shot": selection["epoch"] == 0,
        "head_change": change,
        "history": history,
    }
    return dict(pipe.adaptation)
