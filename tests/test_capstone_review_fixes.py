"""Regression tests for the 2026-09-28 capstone review (BC-M1..M3, BC-m1..m15).

They exec the carried stage functions of tools/biodiversity_capstone.py and the generated notebook
source. Model libraries are replaced by small stubs, and synthetic features stand in for BioCLIP
and SigLIP outputs: these tests are software evidence only, never model-performance evidence.
Tests that fit the head with PyTorch skip when it is absent (lightweight CI); the rest need NumPy,
Pillow, matplotlib and safetensors only.
"""

import io
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import biodiversity_capstone as run  # noqa: E402
import biodiversity_core as core  # noqa: E402
import build_biodiversity_capstone as builder  # noqa: E402

CLASSES = ["c0", "c1", "c2", "c3"]


@pytest.fixture(scope="module")
def notebook():
    return builder.build()


def _cells(notebook, kind):
    return {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == kind}


def _manifest(n_probes=2):
    records = []
    for c, label in enumerate(CLASSES):
        for role, count in (("train", 8), ("validation", 3), ("test", 3)):
            for j in range(count):
                records.append(
                    {
                        "id": f"{role}_{label}_{j}",
                        "label": label,
                        "split": role,
                        "observer": f"{role}_{j % 2}",
                        "common_name": f"Common Bird {c}",
                        "scientific_name": f"Genus species{c}",
                    }
                )
    probes = [
        {"id": f"probe_{k}", "label": f"outside_{k}", "split": "probe", "observer": "p"}
        for k in range(n_probes)
    ]
    return {
        "classes": [
            {"label": label, "common_name": f"Common Bird {i}", "scientific_name": f"Genus species{i}"}
            for i, label in enumerate(CLASSES)
        ],
        "records": records,
        "probes": probes,
    }


def _features(records, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in records:
        base = np.zeros(768)
        if r["label"] in CLASSES:
            base[CLASSES.index(r["label"])] = 1.0
        vector = base + 0.35 * rng.normal(size=768) / np.sqrt(768)
        rows.append(vector / np.linalg.norm(vector))
    return np.asarray(rows, dtype=np.float32)


def _adapter(out, weight, bias):
    from safetensors.numpy import save_file

    folder = out / "adapter"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(
        {"head.weight": weight.astype(np.float32), "head.bias": bias.astype(np.float32)},
        str(folder / "adapter.safetensors"),
    )
    core.write_json(
        folder / "manifest.json",
        {
            "base_model": {"model_id": run.BIO_ID, "revision": run.BIO_REV},
            "classes": CLASSES,
            "prompts": {c: f"a photo of {c}." for c in CLASSES},
            "tensors": {"head.weight": [len(CLASSES), 768], "head.bias": [len(CLASSES)]},
            "bytes": (folder / "adapter.safetensors").stat().st_size,
            "sha256": core.sha256(folder / "adapter.safetensors"),
        },
    )


def evaluated_root(tmp_path, monkeypatch):
    """A complete post-adapt-head run directory built from synthetic features (NumPy only)."""
    manifest = _manifest()
    core.write_json(tmp_path / "data_manifest.json", manifest)
    core.write_json(tmp_path / "model_manifest.json", {"fixture": True})
    out = tmp_path / "outputs"
    out.mkdir()
    records = manifest["records"] + manifest["probes"]
    features = _features(records)
    text = np.eye(768, dtype=np.float32)[: len(CLASSES)]
    scale = 8.0
    np.save(out / "image_embeddings.npy", features)
    test_ix = [i for i, r in enumerate(records) if r["split"] == "test"][:2]
    np.save(out / "reload_probe_features.npy", features[test_ix])
    core.write_json(
        out / "embedding_identity.json",
        {"logit_scale": scale, "reload_probe_ids": [records[i]["id"] for i in test_ix]},
    )
    weight, bias = text * scale, np.zeros(len(CLASSES))
    _adapter(out, weight, bias)
    head = core.softmax(features @ weight.T + bias)
    np.save(out / "head_scores.npy", head)
    majority = np.zeros_like(head)
    majority[:, 0] = 1
    np.savez(out / "baseline_scores.npz", majority=majority, colour=head)
    np.savez(out / "siglip_scores.npz", siglip_common=head, siglip_scientific=head[:, ::-1])
    np.savez(out / "bioclip_scores.npz", bioclip_common=head, bioclip_scientific=head)
    np.save(out / "knn_scores.npy", head)
    val = [i for i, r in enumerate(records) if r["split"] == "validation"]
    y_val = np.array([CLASSES.index(records[i]["label"]) for i in val])
    policy = core.select_threshold(y_val, head[val])
    policy.update({"locked_before_test": True, "selection_split": "validation", "selected_epoch": 0})
    core.write_json(out / "selected_policy.json", policy)
    captured = {}
    monkeypatch.setattr(
        run,
        "contact",
        lambda root, recs, name, annotations=None, crop_boxes=False: captured.__setitem__(name, annotations),
    )
    run.evaluate(tmp_path)
    run.triage(tmp_path)
    return out, records, captured


def finished_root(tmp_path, monkeypatch):
    out, records, captured = evaluated_root(tmp_path, monkeypatch)
    core.write_json(out / "verification_reload.json", {"test_fixture": True})
    core.write_json(out / "prepare.json", {"test_fixture": True})
    core.write_json(
        out / "bootstrap.json",
        {
            "seconds": 5.0,
            "uv_download_seconds": 1.0,
            "environment_seconds": 2.0,
            "install_seconds": 2.0,
            "session_started_at": 0.0,
        },
    )
    for key, seconds in (("dataset", 3.0), ("siglip", 4.0), ("bioclip", 6.0)):
        core.write_json(out / f"download_{key}.json", {"downloaded_bytes": 10, "seconds": seconds})
    for stage in run.STAGES[:-1]:
        core.write_json(
            tmp_path / "receipts" / f"{stage}.json",
            {"seconds": 10.0, "resources": {"peak_vram_bytes": 1, "peak_host_ram_bytes": 2}},
        )
    run.report(tmp_path)
    return out, records, captured


# BC-M1 ---------------------------------------------------------------------------------------


def test_head_learning_rate_grid_is_predeclared():
    rates = run.CONFIG["learning_rates"]
    assert "learning_rate" not in run.CONFIG
    assert len(rates) >= 3 and max(rates) / min(rates) >= 100 and rates == sorted(rates)


def test_head_can_leave_zero_shot_when_validation_supports_it():
    pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    text = np.eye(768, dtype=np.float32)[:4]

    def make(label, wrong):
        v = (
            0.30 * text[label]
            + (0.32 * text[(label + 1) % 4] if wrong else 0)
            + 0.9 * rng.normal(size=768) / 27.7
        )
        return v / np.linalg.norm(v)

    train = [(make(c, k < 3), c) for c in range(4) for k in range(8)]
    val = [(make(c, k < 1), c) for c in range(4) for k in range(3)]
    tx, ty = np.array([f for f, _ in train], np.float32), np.array([y for _, y in train])
    vx, vy = np.array([f for f, _ in val], np.float32), np.array([y for _, y in val])
    tensors, history, selected = run.fit_head(tx, ty, vx, vy, text, 100.0)
    assert selected["learning_rate"] > min(run.CONFIG["learning_rates"])
    assert selected["epoch"] > 0
    epoch0 = next(r for r in history if r["epoch"] == 0)["validation_loss"]
    assert selected["validation_loss"] < epoch0
    change = run.head_change(np.concatenate([tx, vx]), tensors, text, 100.0)
    assert change["predictions_changed_vs_zero_shot"] > 0 and change["max_abs_logit_change"] > 1


def test_adapt_head_reports_selection_and_distance_from_zero_shot(tmp_path):
    pytest.importorskip("torch")
    manifest = _manifest(0)
    core.write_json(tmp_path / "data_manifest.json", manifest)
    out = tmp_path / "outputs"
    out.mkdir()
    np.save(out / "image_embeddings.npy", _features(manifest["records"]))
    np.save(out / "text_embeddings.npy", np.eye(768, dtype=np.float32)[:4])
    core.write_json(out / "embedding_identity.json", {"logit_scale": 8.0})
    run.adapt_head(tmp_path)
    training = core.read_json(out / "head_training.json")
    policy = core.read_json(out / "selected_policy.json")
    for key in (
        "selected_learning_rate",
        "selected_epoch",
        "head_equals_zero_shot",
        "max_abs_logit_change_train_validation",
        "predictions_changed_vs_zero_shot_train_validation",
    ):
        assert key in training
    assert all(not isinstance(v, (dict, list)) for v in training.values())  # renders in show_record
    assert policy["head_equals_zero_shot"] == (policy["selected_epoch"] == 0)
    assert training["selected_learning_rate"] in run.CONFIG["learning_rates"]


def test_notebook_explains_rate_selection_and_shows_head_record(notebook):
    md, code = _cells(notebook, "markdown"), _cells(notebook, "code")
    assert "epoch 0 of the head is exactly the BioCLIP scientific-name zero-shot classifier" in md["md-18"]
    assert "select" in md["md-18"] and "1e-4, 1e-3 and 1e-2" in md["md-18"]
    assert "show_record('head_training.json')" in code["code-19"]


# BC-M2 ---------------------------------------------------------------------------------------


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_siglip_stage_uses_documented_lowercase_64_token_text(tmp_path, monkeypatch):
    manifest = _manifest(0)
    core.write_json(tmp_path / "data_manifest.json", manifest)
    (tmp_path / "outputs").mkdir()
    calls = []

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Processor()

        def __call__(self, text, images, return_tensors, **kwargs):
            calls.append((list(text), dict(kwargs)))
            width = kwargs.get("max_length", 17)
            return Inputs(input_ids=np.zeros((len(text), width)), n=len(images))

    class Model:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Model()

        def float(self):
            return self

        def cuda(self):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, n):
            return types.SimpleNamespace(logits_per_image=np.zeros((n, input_ids.shape[0])))

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    fake_torch = types.SimpleNamespace(
        no_grad=NoGrad,
        sigmoid=lambda x: _Tensor(1 / (1 + np.exp(-x))),
        cuda=types.SimpleNamespace(empty_cache=lambda: None),
    )
    monkeypatch.setattr(run, "torch_runtime", lambda: fake_torch)
    monkeypatch.setattr(run, "stage_model", lambda root, key, record_download=True: tmp_path)
    monkeypatch.setattr(run, "images", lambda root, recs: [object() for _ in recs])
    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoModel=Model, AutoProcessor=Processor)
    )
    run.siglip_zero_shot(tmp_path)
    assert calls and all(
        kwargs.get("padding") == "max_length" and kwargs.get("max_length") == 64 for _, kwargs in calls
    )
    texts = {t for prompts, _ in calls for t in prompts}
    assert "This is a photo of common bird 0." in texts and "This is a photo of genus species0." in texts
    assert all(t[len("This is a photo of ") :] == t[len("This is a photo of ") :].lower() for t in texts)
    record = core.read_json(tmp_path / "outputs" / "siglip_preprocessing.json")
    assert record["label_text_lowercased"] is True and record["max_length"] == 64


# BC-M3 ---------------------------------------------------------------------------------------


def test_perfect_classifier_has_degenerate_macro_f1_interval():
    y = np.repeat(np.arange(4), 3)
    perfect = np.eye(4)[y]
    stratified = core.bootstrap(y, perfect, n_boot=2000, stratify=True)
    assert stratified["macro_f1_low"] == stratified["macro_f1_high"] == 1.0
    assert stratified == core.bootstrap(y, perfect, n_boot=2000, stratify=True)
    # The unstratified form shows the artefact the canonical path avoids.
    assert core.bootstrap(y, perfect, n_boot=2000)["macro_f1_low"] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        core.bootstrap(y, perfect, groups=list(range(12)), stratify=True)


def test_exported_intervals_are_species_stratified(tmp_path, monkeypatch):
    out, _, _ = evaluated_root(tmp_path, monkeypatch)
    rows = {r["system"]: r for r in run.read_csv(out / "metrics.csv")}
    head = rows["bioclip_head"]
    assert float(head["accuracy"]) == 1.0 and float(head["macro_f1_low"]) == 1.0
    assert "within each species" in head["bootstrap_method"]
    assert not any(key.startswith("observer_cluster_") for key in head)
    assert int(head["test_observers"]) == 2


# BC-m1 ---------------------------------------------------------------------------------------


def test_runtime_summary_reports_bootstrap_downloads_and_stages(tmp_path, monkeypatch):
    out, _, _ = finished_root(tmp_path, monkeypatch)
    rows = {r["step"]: r for r in run.read_csv(out / "runtime_summary.csv")}
    assert float(rows["bootstrap: install"]["seconds"]) == 2.0
    assert float(rows["download: dataset (included in its stage)"]["seconds"]) == 3.0
    assert rows["stage: siglip-zero-shot"]["peak_vram_bytes"] == "1"
    total = rows["total: bootstrap + stages before report"]
    assert float(total["seconds"]) == pytest.approx(1.0 + 2.0 + 2.0 + 10.0 * (len(run.STAGES) - 1))
    summary = core.read_json(out / "run_summary.json")
    assert summary["notebook_revision"] == run.NOTEBOOK_REVISION
    assert summary["timing"]["downloads"]["bioclip"]["seconds"] == 6.0
    assert summary["wall_clock_seconds_since_session_start"] > 0


def test_notebook_records_bootstrap_timing_and_renders_runtime(notebook):
    code = _cells(notebook, "code")
    assert "'install_seconds': install_seconds" in code["code-05"]
    assert "(ROOT / 'outputs' / 'bootstrap.json')" in code["code-05"]
    assert "show_table('runtime_summary.csv'" in code["code-32"]
    assert "show_record('download_dataset.json')" in code["code-07"]


# BC-m2 ---------------------------------------------------------------------------------------


def test_archive_verification_is_real_and_displayed(tmp_path, monkeypatch, notebook):
    out, _, _ = finished_root(tmp_path, monkeypatch)
    record = core.read_json(out / "archive_verification.json")
    assert (
        record["excluded_assets_absent"] and record["headline_metrics_recomputed_from_archived_predictions"]
    )
    assert "archive_verification.json" in core.read_json(out / "verification.json")["archive"]
    assert "show_record('archive_verification.json')" in _cells(notebook, "code")["code-32"]
    hashes = json.loads(zipfile.ZipFile(out / "results.zip").read("checksums.json"))
    with zipfile.ZipFile(out / "results.zip", "a") as bundle:
        bundle.writestr("figures/errors.png", b"photo-derived")
    with pytest.raises(ValueError):
        run.verify_archive(out / "results.zip", {**hashes, "figures/errors.png": "0" * 64}, CLASSES)


# BC-m3 ---------------------------------------------------------------------------------------


def test_repeat_and_reload_probe_use_matching_batches(tmp_path, monkeypatch):
    manifest = _manifest(0)
    core.write_json(tmp_path / "data_manifest.json", manifest)
    out = tmp_path / "outputs"
    out.mkdir()
    records = manifest["records"]
    base = {r["id"]: v for r, v in zip(records, _features(records), strict=True)}

    class BatchSensitivePipe:
        logit_scale = 10.0

        def embed_images(self, imgs):
            # Batch-size dependent float noise, as different GPU kernels can produce.
            vectors = [base[i] + 1e-5 * len(imgs) for i in imgs]
            return {"embeddings": [v / np.linalg.norm(v) for v in vectors]}

        def embed_texts(self, prompts):
            return {"embeddings": np.eye(768)[: len(prompts)]}

    np.savez(out / "siglip_scores.npz", siglip_common=np.full((len(records), 4), 0.25))
    monkeypatch.setattr(run, "bioclip_pipe", lambda root, record_download=True: BatchSensitivePipe())
    monkeypatch.setattr(run, "images", lambda root, recs: [r["id"] for r in recs])
    monkeypatch.setattr(run, "confusion_figure", lambda *a, **k: None)
    run.bioclip_zero_shot(tmp_path)
    identity = core.read_json(out / "embedding_identity.json")
    assert identity["cross_batch_size_max_abs_error"] > 0
    test_ids = [r["id"] for r in records if r["split"] == "test"][:2]
    assert identity["reload_probe_ids"] == test_ids
    expected = BatchSensitivePipe().embed_images(test_ids)["embeddings"]
    np.testing.assert_array_equal(
        np.load(out / "reload_probe_features.npy"), np.asarray(expected, np.float32)
    )


def test_reload_expectation_comes_from_the_matched_batch(tmp_path, monkeypatch):
    out, _, _ = evaluated_root(tmp_path, monkeypatch)
    tensors, _ = run.load_head(out / "adapter", CLASSES)
    expected = run.head_scores(np.load(out / "reload_probe_features.npy"), tensors)
    np.testing.assert_array_equal(np.load(out / "reload_expected.npy"), expected)
    assert "same two test records" in core.read_json(out / "reload_probe.json")["batch"]


# BC-m4 ---------------------------------------------------------------------------------------


def test_activity_uses_adjacent_validation_candidates(tmp_path, monkeypatch, notebook):
    out, _, _ = evaluated_root(tmp_path, monkeypatch)
    views = core.read_json(out / "activity_thresholds.json")
    candidates = sorted(
        {float(r["threshold"]) for r in run.read_csv(out / "risk_coverage.csv") if r["split"] == "validation"}
    )
    locked = views["canonical"]
    assert views["lower"] == max([c for c in candidates if c < locked], default=locked)
    assert views["higher"] == min([c for c in candidates if c > locked], default=locked)
    assert run.neighbour_thresholds([0.2, 0.5, 0.9], 0.5) == {"lower": 0.2, "canonical": 0.5, "higher": 0.9}
    run.triage(tmp_path, display_threshold=float(np.nextafter(1.0, 2.0)))  # largest candidate is accepted
    code = _cells(notebook, "code")["code-26"]
    assert "activity_thresholds.json" in code and "offset" not in code
    helpers = _cells(notebook, "code")["code-05"]
    namespace = {"Path": Path}
    exec(helpers[helpers.index("PRECISE_COLUMNS") : helpers.index("def show_table")], namespace)
    assert namespace["fmt"]("0.99991", "threshold") != namespace["fmt"]("0.99998", "threshold")
    assert namespace["fmt"](True, "x") == "True" and namespace["fmt"]("", "x") == "undefined"


# BC-m5 ---------------------------------------------------------------------------------------


def test_error_panel_names_categories_and_falls_back():
    correct = np.array([True, True, False, True])
    margins = np.array([0.9, 0.5, 0.1, 0.3])
    disagree = np.array([False, False, True, True])
    rows = run.error_panel_selection(correct, margins, disagree)
    picks = {category: pick for category, pick, _ in rows}
    assert picks["highest-margin correct"] == 0 and picks["highest-margin incorrect"] == 2
    assert picks["lowest margin"] == 3  # record 2 already shown; next-lowest margin is used
    assert picks["head and SigLIP (scientific) disagree"] is None  # both disagreeing records already shown
    assert len({p for p in picks.values() if p is not None}) == 3


def test_error_panel_annotations_and_csv(tmp_path, monkeypatch):
    out, _, captured = evaluated_root(tmp_path, monkeypatch)
    rows = run.read_csv(out / "error_panel.csv")
    assert [r["category"] for r in rows] == list(run.ERROR_CATEGORIES)
    shown = [r for r in rows if r["shown"] == "True"]
    assert len(captured["errors.png"]) == len(shown)
    for annotation, row in zip(captured["errors.png"], shown, strict=True):
        assert annotation.startswith(f"[{row['category']}]")


# BC-m6 / BC-m7 / BC-m8 -----------------------------------------------------------------------


def test_observer_concentration_and_audit_from_frozen_manifest():
    manifest = core.read_json(ROOT / "tools/biodiversity_data.json")
    records = manifest["records"] + manifest["probes"]
    roles = {r["role"]: r for r in run.observer_role_rows(records)}
    assert roles["test"]["observers"] == 4 and roles["test"]["largest_observer_share"] == 0.5
    checks = core.validate_manifest(manifest)
    audit = {r["check"]: r["value"] for r in run.audit_rows(manifest, records, checks)}
    assert audit["observation dates"] == "2010-04-19 to 2026-06-30"
    assert any(k.startswith("identified below species") for k in audit)
    assert sum(k.startswith("crop limitation") for k in audit) == 2
    assert run.centre_crop_box(1024, 621) == ((1024 - 621) / 2, 0.0, 621)


def test_learner_text_discloses_concentration_taxonomy_and_location_limits(notebook):
    md = _cells(notebook, "markdown")
    assert "the 12 test photographs come from 4 photographers, and one of them contributed 6" in md["md-06"]
    assert "*Sarcops calvus melanonotus*" in md["md-06"]
    assert (
        "must not reconstruct them" in md["md-06"]
        and "Endemic status does not make exact locations" in md["md-06"]
    )
    assert "does not count individuals" in md["md-35"]
    assert "dashed box" in md["md-08"].lower()
    code = _cells(notebook, "code")["code-07"]
    assert "observer_roles.csv" in code and "dataset_audit.csv" in code


# BC-m9 / BC-m11 / BC-m12 ---------------------------------------------------------------------


def test_probes_listed_for_every_system(tmp_path, monkeypatch):
    out, records, _ = evaluated_root(tmp_path, monkeypatch)
    rows = run.read_csv(out / "probe_predictions.csv")
    systems = {r["system"] for r in rows}
    assert systems == set(run.all_scores(tmp_path))
    assert len(rows) == len(systems) * sum(r["split"] == "probe" for r in records)
    assert all(r["accepted"] == "" for r in rows if r["system"] != "bioclip_head")


def test_majority_top2_is_undefined_and_parity_still_holds(tmp_path, monkeypatch):
    out, _, _ = finished_root(tmp_path, monkeypatch)
    rows = {r["system"]: r for r in run.read_csv(out / "metrics.csv")}
    assert rows["majority"]["top2_accuracy"] == "" and rows["colour"]["top2_accuracy"] != ""
    assert core.read_json(out / "verification.json")["prediction_metric_parity"]


def test_risk_rows_sorted_and_palette_colour_blind_safe(tmp_path, monkeypatch):
    out, _, _ = evaluated_root(tmp_path, monkeypatch)
    rows = run.read_csv(out / "risk_coverage.csv")
    for split in ("validation", "test"):
        values = [float(r["threshold"]) for r in rows if r["split"] == split]
        assert values == sorted(values)
    assert 'get_cmap("tab10")' not in (ROOT / "tools/biodiversity_capstone.py").read_text(encoding="utf-8")


# BC-m10 / BC-m13 / revision ------------------------------------------------------------------


def test_no_filelink_objectives_listed_and_revision_consistent(notebook):
    code, md = _cells(notebook, "code"), _cells(notebook, "markdown")
    assert not any("FileLink" in text for cell_id, text in code.items() if cell_id != "code-04")
    assert "files.download" in code["code-32"] and "DOWNLOAD_RESULTS = False" in code["code-32"]
    assert "**Learning objectives.**" in md["md-01"] and "\n9. " in md["md-01"]
    assert (
        notebook["metadata"]["dimer"]["notebook_revision"]
        == run.NOTEBOOK_REVISION
        == builder.NOTEBOOK_REVISION
    )
    assert run.NOTEBOOK_REVISION in md["md-00"]


# BC-m15 --------------------------------------------------------------------------------------


def _png(value):
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (40, 40), (value, 30, 60)).save(stream, format="PNG")
    return stream.getvalue()


def test_byod_accepts_one_enclosing_folder_and_explains_bad_csv(tmp_path, notebook):
    good = tmp_path / "folder.zip"
    with zipfile.ZipFile(good, "w") as archive:
        for index in range(2):
            archive.writestr(f"my_photos/c{index}/image.png", _png(index * 90))
    result = core.prepare_byod(tmp_path / "run", str(good), {"c0": "Species A", "c1": "Species B"}, True)
    assert sorted(c["label"] for c in result["classes"]) == ["c0", "c1"]
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("labels.csv", "image,label\na.png,c0\n")
        archive.writestr("a.png", _png(1))
    with pytest.raises(ValueError, match="lacks required column"):
        core.prepare_byod(tmp_path / "run2", str(bad), authorization=True)
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("deep/a/b/c.png", _png(2))
    with pytest.raises(ValueError, match="<label>/<image>"):
        core.prepare_byod(tmp_path / "run3", str(empty), authorization=True)
    assert "os.link(source, target)" in _cells(notebook, "code")["code-34"]
