"""Regression tests for the Notebook Review Framework v1 review of tutorials/bioclip2_biodiversity_colab.ipynb
(review PR #10, findings BIO-M1..M5 and BIO-m1..m6).

Everything here runs within CI's install budget (no torch): Section 4 of the notebook is executed verbatim
from the notebook JSON with the carried package functions and a fake `google.colab`, and the head-selection
test that needs torch skips cleanly without it.
"""

from __future__ import annotations

import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

import bioclip2_biodiversity_pipeline as pkg
from bioclip2_biodiversity_pipeline import (
    MIN_RECORDS,
    MIN_RECORDS_PER_CLASS,
    SAMPLE_CLASS_PROMPTS,
    generate_sample_dataset,
    head_change,
    load_byod_dataset,
    minimum_records_per_class,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "bioclip2_biodiversity_colab.ipynb"
SAMPLE = generate_sample_dataset()
BY_CLASS: dict[str, list[bytes]] = {}
for _r in SAMPLE:
    BY_CLASS.setdefault(_r["label"], []).append(_r["image_bytes"])
CLASSES = sorted(BY_CLASS)


def _nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells() -> list[str]:
    return ["".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "code"]


def _markdown() -> str:
    return "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")


def _cell(marker: str) -> str:
    hits = [s for s in _code_cells() if marker in s]
    assert len(hits) == 1, marker
    return hits[0]


def _records(per_class: int, classes=None) -> list[dict]:
    return [
        {"id": f"{c}-{k}", "image_bytes": BY_CLASS[c][k], "label": c}
        for c in (classes or CLASSES)
        for k in range(per_class)
    ]


def _write_folders(root: Path, per_class: int, classes=None, camera_names: bool = True) -> None:
    for c in classes or CLASSES:
        (root / c).mkdir(parents=True, exist_ok=True)
        for k in range(per_class):
            name = f"IMG_{k + 1:04d}.jpg" if camera_names else f"{c}_{k}.jpg"
            (root / c / name).write_bytes(BY_CLASS[c][k])


def _folder_zip(
    per_class: int, classes=None, enclosing: str | None = None, extra: dict | None = None
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for c in classes or CLASSES:
            for k in range(per_class):
                path = f"{c}/IMG_{k + 1:04d}.jpg"
                zf.writestr(path if enclosing is None else f"{enclosing}/{path}", BY_CLASS[c][k])
        for path, data in (extra or {}).items():
            zf.writestr(path, data)
    return buf.getvalue()


# --- BIO-M3: the folder layout loads from a relative upload directory ---------------------------------


def test_byod_directory_loads_from_a_relative_path(tmp_path, monkeypatch):
    """Failed on 2def984: the relative root was joined twice (work/byod/work/byod/...)."""
    monkeypatch.chdir(tmp_path)
    _write_folders(Path("work") / "byod", 7)
    records = load_byod_dataset(Path("work") / "byod")
    assert len(records) == 28
    assert {r["label"] for r in records} == set(CLASSES)
    assert records[0]["id"] == "chipping_sparrow/IMG_0001.jpg"


def test_byod_directory_accepts_one_enclosing_folder_and_ignores_macosx(tmp_path):
    _write_folders(tmp_path / "my_photos", 5)
    (tmp_path / "__MACOSX" / "my_photos").mkdir(parents=True)
    assert len(load_byod_dataset(tmp_path)) == 20


def test_byod_directory_without_label_folders_explains_the_layout(tmp_path):
    (tmp_path / "loose.jpg").write_bytes(BY_CLASS[CLASSES[0]][0])
    with pytest.raises(ValueError, match="one folder per label"):
        load_byod_dataset(tmp_path)


# --- BIO-M4: the stated minimum is the enforced one, and it is enforced by the split ---------------------


def test_minimum_records_per_class_matches_the_split_arithmetic():
    assert minimum_records_per_class(2) == 7
    assert [minimum_records_per_class(k) for k in (3, 4, 10)] == [5, 5, 5]
    for n_classes in (2, 3, 4):
        need = minimum_records_per_class(n_classes)
        for per_class in range(MIN_RECORDS_PER_CLASS, 12):
            records = _records(per_class, CLASSES[:n_classes])
            if per_class < need:
                # A dataset below the 8-record floor is refused by validate_dataset first; otherwise the
                # split names the per-class minimum.
                with pytest.raises(
                    ValueError,
                    match=f"supply at least {need} records per class|at least {MIN_RECORDS} are required",
                ):
                    split_dataset(records)
            else:
                train = split_dataset(records)["train"]
                # exactly what adaptation re-validates
                validate_dataset(train)
                assert len(train) >= MIN_RECORDS


@pytest.mark.parametrize("per_class", [3, 4])
def test_small_classes_are_refused_by_the_split_naming_the_training_split(per_class):
    with pytest.raises(ValueError, match="training split") as err:
        split_dataset(_records(per_class))
    assert "supply at least 5 records per class" in str(err.value)


def test_mpo_jpeg_is_refused_with_the_conversion(tmp_path):
    from PIL import Image

    _write_folders(tmp_path, 5)
    im = Image.open(io.BytesIO(BY_CLASS["song_sparrow"][0]))
    buf = io.BytesIO()
    im.save(buf, "MPO", save_all=True, append_images=[im.copy()])
    (tmp_path / "song_sparrow" / "IMG_9999.jpg").write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="MPO file.*Re-save it as a plain JPEG"):
        load_byod_dataset(tmp_path)


# --- BIO-m2: repeated camera names keep one file per record and the CSV reloads -------------------------


def test_write_dataset_csv_keeps_every_image_when_camera_names_repeat(tmp_path):
    _write_folders(tmp_path / "in", 7)
    records = load_byod_dataset(tmp_path / "in")
    csv_path = write_dataset_csv(records, tmp_path / "out" / "bioclip2_biodiversity_byod_dataset.csv")
    assert len(list((tmp_path / "out" / "images").iterdir())) == 28
    assert len(load_byod_dataset(csv_path)) == 28


def test_sample_dataset_csv_is_unchanged_by_the_collision_rule(tmp_path):
    path = write_dataset_csv(SAMPLE, tmp_path / "s.csv")
    names = [line.split(",")[1] for line in path.read_text(encoding="utf-8").splitlines()[1:]]
    assert names == [f"images/{r['file']}" for r in SAMPLE]


# --- Section 4 executed verbatim (BIO-M3, BIO-M4, BIO-m1, BIO-m2, BIO-m6) --------------------------------


def _section4(tmp_path, monkeypatch, *, upload=None, prompts=None, byod=True, path="", colab=True):
    monkeypatch.chdir(tmp_path)
    if colab:
        google, colab_mod, files = (
            types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files")
        )
        files.upload = lambda: dict(upload or {})
        colab_mod.files = files
        google.colab = colab_mod
        for name, mod in (("google", google), ("google.colab", colab_mod), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, mod)
    else:
        for name in ("google", "google.colab", "google.colab.files"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setitem(sys.modules, "google.colab", None)
    source = _cell("USE_BYOD = False")
    fields = {
        "USE_BYOD": "True" if byod else "False",
        "BYOD_PATH": repr(path),
        "BYOD_CLASS_PROMPTS": repr(json.dumps(prompts if prompts is not None else SAMPLE_CLASS_PROMPTS)),
    }
    for name, literal in fields.items():
        source, n = re.subn(
            rf"^{name} = .*$", lambda _m, n=name, v=literal: f"{n} = {v}", source, count=1, flags=re.M
        )
        assert n == 1, name
    ns = {name: getattr(pkg, name) for name in pkg.__all__}
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def test_section4_default_sample_path_writes_the_sample_csv(tmp_path, monkeypatch):
    ns = _section4(tmp_path, monkeypatch, byod=False)
    assert [len(ns[k]) for k in ("train_records", "val_records", "test_records")] == [28, 8, 12]
    assert ns["per_class_minimum"] == 5
    assert (tmp_path / "outputs" / "bioclip2_biodiversity_sample_dataset.csv").is_file()
    assert not (tmp_path / "outputs" / "byod").exists()


def test_section4_folder_zip_with_camera_names_runs_and_stays_out_of_the_sample_csv(tmp_path, monkeypatch):
    ns = _section4(tmp_path, monkeypatch, upload={"field.zip": _folder_zip(7)})
    assert ns["data_source"] == "BYOD (field.zip)"
    assert [len(ns[k]) for k in ("train_records", "val_records", "test_records")] == [16, 4, 8]
    assert not (tmp_path / "outputs" / "bioclip2_biodiversity_sample_dataset.csv").exists()
    assert len(list((tmp_path / "outputs" / "byod" / "images").iterdir())) == 28
    assert (
        len(load_byod_dataset(tmp_path / "outputs" / "byod" / "bioclip2_biodiversity_byod_dataset.csv")) == 28
    )


def test_section4_refuses_four_by_three_before_any_model(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="supply at least 5 records per class"):
        _section4(tmp_path, monkeypatch, upload={"a.zip": _folder_zip(3)})


def test_section4_second_upload_uses_only_the_second_upload(tmp_path, monkeypatch):
    first, second = CLASSES[:2], CLASSES[2:]
    _section4(
        tmp_path,
        monkeypatch,
        upload={"a.zip": _folder_zip(7, first)},
        prompts={c: SAMPLE_CLASS_PROMPTS[c] for c in first},
    )
    ns = _section4(
        tmp_path,
        monkeypatch,
        upload={"b.zip": _folder_zip(7, second)},
        prompts={c: SAMPLE_CLASS_PROMPTS[c] for c in second},
    )
    assert ns["data_source"] == "BYOD (b.zip)"
    assert ns["CLASSES"] == second
    byod_csv = tmp_path / "outputs" / "byod" / "bioclip2_biodiversity_byod_dataset.csv"
    rows = byod_csv.read_text(encoding="utf-8").splitlines()[1:]
    assert len(rows) == 14 and all(row.split("/")[0] in second for row in rows)
    assert len(list((tmp_path / "outputs" / "byod" / "images").iterdir())) == 14


def test_section4_refuses_extra_and_missing_prompt_keys(tmp_path, monkeypatch):
    two = CLASSES[:2]
    with pytest.raises(ValueError, match="must name exactly the labels found"):
        _section4(tmp_path, monkeypatch, upload={"a.zip": _folder_zip(7, two)}, prompts=SAMPLE_CLASS_PROMPTS)
    with pytest.raises(ValueError, match="must name exactly the labels found"):
        _section4(
            tmp_path,
            monkeypatch,
            upload={"a.zip": _folder_zip(7, two)},
            prompts={two[0]: "Spizella passerina"},
        )


def test_section4_zip_escape_and_symlink_are_refused_without_writing(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="escapes the upload directory"):
        _section4(
            tmp_path, monkeypatch, upload={"e.zip": _folder_zip(5, extra={"../byod_evil/escaped.txt": b"x"})}
        )
    assert not (tmp_path / "work" / "byod_evil").exists()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("song_sparrow/link.jpg")
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(ValueError, match="symbolic link"):
        _section4(tmp_path, monkeypatch, upload={"s.zip": buf.getvalue()})


def test_section4_upload_guards_are_actionable(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="Upload exactly one .zip file"):
        _section4(tmp_path, monkeypatch, upload={})
    with pytest.raises(RuntimeError, match="set BYOD_PATH"):
        _section4(tmp_path, monkeypatch, colab=False)
    with pytest.raises(ValueError, match="must be a JSON object"):
        source_prompts = "not json"
        monkeypatch.chdir(tmp_path)
        src = _cell("USE_BYOD = False")
        src = re.sub(r"^USE_BYOD = .*$", "USE_BYOD = True", src, count=1, flags=re.M)
        src = re.sub(
            r"^BYOD_CLASS_PROMPTS = .*$",
            lambda _m: f"BYOD_CLASS_PROMPTS = {source_prompts!r}",
            src,
            count=1,
            flags=re.M,
        )
        exec(compile(src, "<section 4>", "exec"), {name: getattr(pkg, name) for name in pkg.__all__})


def test_section4_byod_path_folder_works_without_colab(tmp_path, monkeypatch):
    _write_folders(tmp_path / "photos", 5)
    ns = _section4(tmp_path, monkeypatch, path=str(tmp_path / "photos"), colab=False)
    assert ns["data_source"] == "BYOD (photos)"
    assert len(ns["records"]) == 20


# --- BIO-M2: the head-selection record (NumPy part runs everywhere; fitting needs torch) ---------------


def test_head_change_reports_zero_for_the_zero_shot_head_and_counts_changes():
    import numpy as np

    rng = np.random.default_rng(0)
    text = rng.normal(size=(3, 8))
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    feats = rng.normal(size=(10, 8))
    same = head_change(feats, {"head.weight": text * 100.0, "head.bias": np.zeros(3)}, text, 100.0)
    assert same["relative_weight_change"] == 0.0 and same["predictions_changed_vs_zero_shot"] == 0
    moved = head_change(feats, {"head.weight": text[::-1] * 100.0, "head.bias": np.zeros(3)}, text, 100.0)
    assert moved["relative_weight_change"] > 0.5 and moved["predictions_changed_vs_zero_shot"] > 0


def test_fit_head_leaves_a_wrong_zero_shot_start_and_keeps_epoch_zero_when_nothing_helps():
    torch = pytest.importorskip("torch")
    import numpy as np

    rng = np.random.default_rng(1)
    centres = rng.normal(size=(3, 16))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    def draw(n):
        y = np.repeat(np.arange(3), n)
        x = centres[y] + 0.15 * rng.normal(size=(len(y), 16))
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), y

    train_x, train_y = draw(8)
    val_x, val_y = draw(4)
    # Names that carry no information: three nearly identical text directions, as with `species 0..2` (the
    # zero-shot start is close to guessing, as in the tutorial's Section 13).
    wrong_text = rng.normal(size=16) + 0.02 * rng.normal(size=(3, 16))
    wrong_text /= np.linalg.norm(wrong_text, axis=1, keepdims=True)
    tensors, history, selection = pkg.fit_head(train_x, train_y, val_x, val_y, wrong_text, 100.0, epochs=20)
    assert selection["learning_rate"] > 1e-4 and selection["epoch"] > 0
    epoch0 = next(h for h in history if h["epoch"] == 0)
    assert min(h["val_loss"] for h in history) < epoch0["val_loss"]
    assert head_change(train_x, tensors, wrong_text, 100.0)["predictions_changed_vs_zero_shot"] > 0
    assert len(history) == 3 * 21
    _t, _h, frozen = pkg.fit_head(
        train_x, train_y, val_x, val_y, wrong_text, 100.0, epochs=3, learning_rates=[0.0]
    )
    assert frozen["epoch"] == 0
    assert isinstance(_t["head.weight"], torch.Tensor)


# --- BIO-M1 / BIO-M5 / BIO-m3..m6: static notebook and record checks ------------------------------------


def test_isolated_runtime_replaces_the_in_kernel_install():
    kernel = [s for s in _code_cells() if "# dimer: kernel cell" in s]
    assert len(kernel) == 2
    install = next(s for s in kernel if "LOCK_TEXT = r" in s)
    for needed in (
        '"--managed-python"',
        '"--require-hashes"',
        '":all:"',
        "UV_SHA256",
        "LOCK_SHA256",
        "3.12.12",
    ):
        assert needed in install
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    assert "torch==2.14.0" in lock and "open-clip-torch==3.3.0" in lock


def test_section9_selects_on_validation_and_reports_the_head():
    s9 = _cell("EPOCHS = 20")
    assert "adapt_result = select_head(" in s9 and "learning_rates=LEARNING_RATES" in s9
    assert "template=PROMPT_TEMPLATE" in s9
    assert pkg.LEARNING_RATES == (1e-4, 1e-3, 1e-2)
    s10 = _cell("val_metrics = pipe.evaluate(val_records)")
    assert "'agreement_with_zero_shot': agreement" in s10


def test_learner_text_no_longer_draws_lessons_from_a_head_that_could_not_move():
    md = _markdown()
    for stale in (
        "the recorded reason the default is 0",
        "head-only adaptation preserves that",
        "On the sample it is zero",
        "at least 8 records and 3 per class",
        "{{",
        "}}",
        "about two minutes of model time",
    ):
        assert stale not in md, stale
    assert "Runtime → Run after" in md
    assert "## 13. Your turn — change one thing" in md


def test_release_records_do_not_call_the_two_pass_run_a_run_all_pass():
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    row = next(
        line
        for line in registry.splitlines()
        if "bioclip2_biodiversity_colab.ipynb" in line and line.startswith("|")
    )
    assert "Run all` PASS" not in row and "Run all PASS" not in row
    assert "Current status: **Candidate" in (ROOT / "STATUS.md").read_text(encoding="utf-8")
    verification = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "not a one-pass `Run all`" in verification
