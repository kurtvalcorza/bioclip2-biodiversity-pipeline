"""Regression tests for the 2026-10-12 review of the field-survey capstone.

Finding IDs are written "capstone BIO-..." because the 2026-10-02 review of
`bioclip2_biodiversity_colab.ipynb` used the same BIO-* IDs. NumPy/Pillow only: no model library.
"""

import ast
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import biodiversity_capstone as run  # noqa: E402
import biodiversity_core as core  # noqa: E402
import build_biodiversity_capstone as builder  # noqa: E402

from test_capstone_review_fixes import finished_root  # noqa: E402


@pytest.fixture(scope="module")
def notebook():
    return builder.build()


def _cells(notebook, kind):
    return {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == kind}


def _markdown(notebook):
    return "\n".join(_cells(notebook, "markdown").values())


def _png(value):
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (40, 40), (value, 30, 60)).save(stream, format="PNG")
    return stream.getvalue()


def _folder_zip(path, per_class, classes=("c0", "c1")):
    with zipfile.ZipFile(path, "w") as archive:
        for c, label in enumerate(classes):
            for j in range(per_class):
                archive.writestr(f"{label}/image{j}.png", _png(c * 100 + j))
    return path


# capstone BIO-M1 -----------------------------------------------------------------------------


def test_wilson_interval_keeps_width_where_the_stratified_bootstrap_collapses():
    perfect = core.wilson_interval(12, 12)
    assert perfect["high"] == 1.0 and perfect["low"] == pytest.approx(0.7575, abs=1e-3)
    half = core.wilson_interval(6, 12)
    assert half["low"] == pytest.approx(1 - half["high"]) and half["low"] < 0.5 < half["high"]
    assert core.wilson_interval(0, 12)["low"] == 0.0
    with pytest.raises(ValueError):
        core.wilson_interval(13, 12)


def test_metrics_add_wilson_and_per_metric_degeneracy_after_the_existing_columns(tmp_path, monkeypatch):
    out, _, _ = finished_root(tmp_path, monkeypatch)
    header = (out / "metrics.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[-4:] == [
        "accuracy_wilson_low",
        "accuracy_wilson_high",
        "accuracy_bootstrap_degenerate",
        "macro_f1_bootstrap_degenerate",
    ]
    assert header.index("paired_correctness_difference_vs_bioclip_scientific") == len(header) - 5
    for row in run.read_csv(out / "metrics.csv"):
        assert float(row["accuracy_wilson_low"]) < float(row["accuracy_wilson_high"])
        accuracy_flat = float(row["accuracy_low"]) == float(row["accuracy_high"])
        f1_flat = float(row["macro_f1_low"]) == float(row["macro_f1_high"])
        assert row["accuracy_bootstrap_degenerate"] == str(accuracy_flat)
        assert row["macro_f1_bootstrap_degenerate"] == str(f1_flat)


def test_constant_accuracy_with_varying_macro_f1_is_flagged_per_metric():
    """CX4: the recorded siglip_common pattern (accuracy 0.500-0.500, macro-F1 0.338-0.402)."""
    import numpy as np

    y = np.repeat(np.arange(4), 3)
    predicted = np.array([1, 1, 3, 1, 1, 1, 3, 3, 1, 3, 3, 3])  # species 0 and 2 always wrong
    scores = np.eye(4)[predicted] * 0.9 + 0.025
    interval = core.bootstrap(y, scores, n_boot=400, stratify=True)
    assert interval["accuracy_low"] == interval["accuracy_high"] == 0.5
    assert interval["macro_f1_low"] < interval["macro_f1_high"]
    assert run.bootstrap_degenerate(interval) == {
        "accuracy_bootstrap_degenerate": True,
        "macro_f1_bootstrap_degenerate": False,
    }


def test_markdown_calls_zero_width_degenerate_and_wilson_hypothetical(notebook):
    md = _cells(notebook, "markdown")
    assert "zero width" in md["md-21"] and "resampling distribution is degenerate" in md["md-21"]
    assert "hypothetical illustration for accuracy only" in md["md-21"] and "rule of three" in md["md-21"]
    assert "not this study's uncertainty" in md["md-21"] and "Wilson 95% interval" not in md["md-21"]
    assert (
        "resampling distribution is degenerate" in md["md-01"]
        and "A hypothetical illustration" in md["md-01"]
    )
    code = _cells(notebook, "code")["code-22"]
    assert "'accuracy_bootstrap_degenerate', 'macro_f1_bootstrap_degenerate'" in code
    assert "'accuracy_wilson_low', 'accuracy_wilson_high'" in code
    assert "stratified_interval_estimable" not in code


# capstone BIO-M2 (provisional option A: wording only) ----------------------------------------


def test_comparison_names_both_architectures_and_bounds_the_claim(notebook):
    md = _cells(notebook, "markdown")
    assert "ViT-B/16" in md["md-12"] and "ViT-L/14" in md["md-12"]
    assert "does not isolate the effect of biological pretraining" in md["md-12"]
    assert "does not isolate the effect of biological pretraining" in md["md-01"]
    assert "cannot be attributed" in md["md-14"]
    # CX7: the score function is not offered as a cause of a ranking gap.
    assert "change score scale and margins, not top-1 accuracy" in md["md-12"]
    assert "score function" not in md["md-14"]


# Codex follow-up CX1, CX2, CX5, CX6 ----------------------------------------------------------


def test_section_6_pca_shows_no_test_points_before_selection(notebook):
    import inspect

    source = inspect.getsource(run.representations)
    assert '("train", "o"), ("validation", "^"))' in source and '"test"' not in source
    assert "PCA(n_components=2" in source and ".fit(features[train])" in source
    assert "test photographs stay hidden until §8" in _cells(notebook, "markdown")["md-15"]


def test_referral_sections_demonstrate_rather_than_claim_benefit(notebook):
    md = _cells(notebook, "markdown")
    assert "**demonstrate and challenge** a locked referral rule" in md["md-24"]
    assert "matched review workload" in md["md-24"]
    assert "challenge the referral rule" in md["md-28"]


def test_worked_answers_do_not_overstate_mechanism(notebook):
    md = _cells(notebook, "markdown")
    assert "leaned on" not in md["md-11"] and "Background reliance is a hypothesis" in md["md-11"]
    assert (
        "not distinct or compact species clusters" in md["md-17"] and "In this sample, yes" not in md["md-17"]
    )


# capstone BIO-m1 -----------------------------------------------------------------------------

STAGE_SCRIPT = """
import sys
print('stage starting', flush=True)
stage = sys.argv[sys.argv.index('--stage') + 1]
if stage == 'refuse':
    raise ValueError('BYOD image format/dimensions refused')
if stage == 'crash':
    raise KeyError('boom')
print('stage complete', flush=True)
"""


def _run_stage(notebook, root):
    source = _cells(notebook, "code")["code-05"]
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "run_stage")
    namespace = {
        "ROOT": root,
        "PYTHON": Path(sys.executable),
        "ENV": dict(os.environ, PYTHONIOENCODING="utf-8"),
        "subprocess": __import__("subprocess"),
    }
    exec(ast.get_source_segment(source, node), namespace)
    (root / "capstone.py").write_text(STAGE_SCRIPT, encoding="utf-8")
    return namespace["run_stage"]


def test_run_stage_refusal_is_one_line_without_a_traceback(notebook, tmp_path, capsys):
    run_stage = _run_stage(notebook, tmp_path)
    assert run_stage("refuse", refusal="BYOD") is False
    shown = capsys.readouterr().out
    assert shown.count("BYOD refused: BYOD image format/dimensions refused") == 1
    assert "Traceback" not in shown and shown.count("stage starting") == 1
    assert "Traceback" in (tmp_path / "refuse.log").read_text(encoding="utf-8")


def test_run_stage_failure_prints_the_traceback_once_then_stops(notebook, tmp_path, capsys):
    run_stage = _run_stage(notebook, tmp_path)
    with pytest.raises(RuntimeError, match="KeyError: 'boom'"):
        run_stage("crash")
    shown = capsys.readouterr().out
    assert shown.count("Traceback (most recent call last):") == 1 and shown.count("stage starting") == 1
    # Without `refusal` (the canonical path) a ValueError still stops Run all.
    with pytest.raises(RuntimeError, match="ValueError: BYOD image"):
        run_stage("refuse")
    assert run_stage("ok") is True


def _byod_cell(notebook, tmp_path, *, zip_path, prompts, authorized=True):
    source = _cells(notebook, "code")["code-34"]
    params = {
        "USE_BYOD = False # @param {type:'boolean'}": "USE_BYOD = True",
        "BYOD_ZIP = '' # @param {type:'string'}": f"BYOD_ZIP = {str(zip_path)!r}",
        "BYOD_CLASS_PROMPTS_JSON = '{}' # @param {type:'string'}": f"BYOD_CLASS_PROMPTS_JSON = {prompts!r}",
        "BYOD_AUTHORIZED = False # @param {type:'boolean'}": f"BYOD_AUTHORIZED = {authorized}",
    }
    for old, new in params.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    canonical = tmp_path / "run"
    canonical.mkdir(exist_ok=True)
    calls = []
    namespace = {
        "json": json,
        "os": os,
        "shutil": __import__("shutil"),
        "subprocess": __import__("subprocess"),
        "uuid": __import__("uuid"),
        "Path": Path,
        "ROOT": canonical,
        "PYTHON": Path(sys.executable),
        "ENV": dict(os.environ, PYTHONIOENCODING="utf-8"),
        "CARRIED_FILES": {
            "biodiversity_core.py": (ROOT / "tools" / "biodiversity_core.py").read_text("utf-8")
        },
        "run_stage": lambda stage, *options, refusal=None: calls.append((stage, refusal)) or False,
        "show_table": lambda *a, **k: calls.append(("show_table",)),
        "show_path": lambda *a, **k: None,
    }
    exec(compile(source, "code-34", "exec"), namespace)
    assert namespace["ROOT"] == canonical
    return calls


@pytest.mark.parametrize(
    ("per_class", "prompts", "authorized", "message"),
    [
        (3, '{"c0": "Species a", "c1": "Species b"}', True, "BYOD needs at least 12 images per class"),
        (12, '{"c0": "Species a"}', True, "Supply a bounded scientific name for every class"),
        (12, "{not json", True, "the class prompts are not valid JSON"),
        (12, '{"c0": "Species a", "c1": "Species b"}', False, "confirm that you are authorised"),
    ],
)
def test_byod_refusals_print_one_message_and_no_traceback(
    notebook, tmp_path, capsys, per_class, prompts, authorized, message
):
    zip_path = _folder_zip(tmp_path / "upload.zip", per_class)
    calls = _byod_cell(notebook, tmp_path, zip_path=zip_path, prompts=prompts, authorized=authorized)
    shown = capsys.readouterr().out
    assert shown.count("BYOD refused:") == 1 and message in shown
    assert "Traceback" not in shown and "CalledProcessError" not in shown
    assert calls == []


def test_valid_byod_reaches_inference_with_clean_refusal_enabled(notebook, tmp_path, capsys):
    zip_path = _folder_zip(tmp_path / "upload.zip", 12)
    calls = _byod_cell(
        notebook, tmp_path, zip_path=zip_path, prompts='{"c0": "Species a", "c1": "Species b"}'
    )
    assert calls == [("byod-infer", "BYOD")]  # the stand-in refuses, so no table is shown
    assert "BYOD refused" not in capsys.readouterr().out


# capstone BIO-m2 -----------------------------------------------------------------------------


def test_inference_only_byod_enforces_the_stated_minimum(tmp_path, notebook):
    prompts = {"c0": "Species a", "c1": "Species b"}
    with pytest.raises(
        ValueError, match="at least 12 images per class in either layout; c0 has 11, c1 has 11"
    ):
        core.prepare_byod(tmp_path / "small", str(_folder_zip(tmp_path / "s.zip", 11)), prompts, True)
    result = core.prepare_byod(tmp_path / "ok", str(_folder_zip(tmp_path / "ok.zip", 12)), prompts, True)
    assert result["inference_only"] and len(result["records"]) == 24
    assert "at least twelve images per class in either layout" in _cells(notebook, "markdown")["md-33"]
    assert core.BYOD_MIN_PER_CLASS == 12


# capstone BIO-m3 -----------------------------------------------------------------------------


def test_threshold_reuse_and_grid_edge_are_disclosed(notebook):
    md = _cells(notebook, "markdown")
    assert "reuses the same validation photographs" in md["md-18"] and "optimistic" in md["md-18"]
    assert "edge of the grid or schedule" in md["md-20"]


# capstone BIO-m4 -----------------------------------------------------------------------------


def test_guided_layer_has_roadmap_glossary_and_labelled_worked_answers(notebook):
    md = _cells(notebook, "markdown")
    assert "**How to use this notebook.**" in md["md-00"] and "**Roadmap.**" in md["md-00"]
    assert "### Glossary" in md["md-01"]
    text = _markdown(notebook)
    answers = text.count("<summary>Worked answer:")
    assert answers >= 4
    # Every quoted canonical-run value is labelled with its source revision (REL13).
    assert (
        text.count("Canonical-run values, labelled per REL13: revision 0.2.2-candidate, blob 9d0097fb")
        == answers
    )


# capstone BIO-m5 -----------------------------------------------------------------------------


def test_one_conclusion_template_prefilled_with_run_values(tmp_path, monkeypatch, notebook):
    out, records, _ = finished_root(tmp_path, monkeypatch)
    text = (out / "conclusion.md").read_text(encoding="utf-8")
    for placeholder in (
        "[n]",
        "[number]",
        "[system]",
        "[coverage]",
        "[selective accuracy]",
        "[did / did not]",
    ):
        assert placeholder not in text
    for blank in ("[baselines]", "[failure pattern]", "[trade-off]", "[bounded use]", "[evidence]"):
        assert blank in text
    assert (
        f"sample of {len([r for r in records if r['split'] != 'probe'])} Philippine bird photographs" in text
    )
    assert "not reliably quantified here" in text and "selected head did equal" in text
    assert "Wilson" not in text and "does not show a review-workload benefit" in text
    assert "On this observer-disjoint sample" not in _markdown(notebook)


def _sample(byod=False):
    return {"byod": byod, "n_total": 56, "n_test": 12, "test_observers": 4, "largest_observer": 6}


def test_conclusion_reports_counts_and_marks_a_degenerate_interval():
    head = {"macro_f1": "1.0", "macro_f1_low": "1.0", "macro_f1_high": "1.0", "accuracy": "1.0"}
    policy = {"coverage": "1.0", "selective_accuracy": "1.0"}
    text = run.conclusion_text({"bioclip_head": head}, policy, False, _sample())
    assert "labelled 12 of 12 test photographs correctly" in text
    assert "from 4 held-out photographers (the largest share from one photographer: 6)" in text
    assert "1.000-1.000; degenerate, so this method gives no uncertainty for it" in text
    assert "did not equal" in text and "nationwide accuracy" in text


def test_byod_conclusion_does_not_claim_philippine_birds():
    """CX9: grouped BYOD also runs report()."""
    head = {"macro_f1": "0.8", "macro_f1_low": "0.6", "macro_f1_high": "0.9", "accuracy": "0.75"}
    policy = {"coverage": "0.5", "selective_accuracy": "1.0"}
    text = run.conclusion_text({"bioclip_head": head}, policy, True, _sample(byod=True))
    assert "Philippine bird" not in text and "56 user-supplied photographs" in text
    assert "as supplied by the user" in text and "degenerate" not in text


# capstone BIO-m6 -----------------------------------------------------------------------------


def test_status_text_names_the_real_open_gates(notebook):
    md0 = _cells(notebook, "markdown")["md-00"]
    assert "fresh hosted end-to-end run have been qualified" not in md0
    assert "BYOD" in md0 and "attribution review" in md0
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "fresh Colab T4 execution pending" not in readme and "attribution review" in readme
