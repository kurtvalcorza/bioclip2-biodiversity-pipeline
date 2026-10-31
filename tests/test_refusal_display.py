"""DAT19/DAT20: a refused BYOD input shows one clean line, ``Refused: <file>: <reason>``, with no traceback.

The router cell carries ml-worker's refusal channel (template ``dat20-router-refusal/1``) and Section 4 wraps its
whole BYOD branch so that every user-input problem raises ``BYODRefusal``. These tests drive both from the
notebook JSON with CI's dependencies only:

- Section 4 is executed verbatim (the review-fix harness) for each refusal path;
- the router's kernel loop is driven with a queued stub connection on every platform;
- on POSIX the real worker process runs Section 4, so the full display path is checked end to end.
"""
# ruff: noqa: E501  -- notebook source literals are matched verbatim

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from bioclip2_biodiversity_pipeline import SAMPLE_CLASS_PROMPTS
from test_bioclip2_biodiversity_colab_review_fixes import CLASSES, _folder_zip, _section4

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "bioclip2_biodiversity_colab.ipynb"
STEM = "bioclip2_biodiversity"

NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in NB["cells"]]
CODE = [i for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code"]


def _cell(marker: str) -> int:
    found = [i for i in CODE if marker in CELLS[i]]
    assert len(found) == 1, (marker, found)
    return found[0]


SEC4 = _cell("USE_BYOD = False  # @param")
ROUTER = _cell("_WORKER_SOURCE = r")
HOUSING = b"longitude,latitude,median_house_value\n1,2,3\n"


def refusing_section4_source(path: str, prompts: str = "{}") -> str:
    src = CELLS[SEC4]
    for name, literal in (("USE_BYOD", "True"), ("BYOD_PATH", repr(path)), ("BYOD_CLASS_PROMPTS", repr(prompts))):
        src, n = re.subn(rf"^{name} = .*$", lambda _m, n=name, v=literal: f"{n} = {v}", src, count=1, flags=re.M)
        assert n == 1, name
    return src


def _refusal(tmp_path, monkeypatch, **kwargs) -> str:
    with pytest.raises(ValueError) as info:
        _section4(tmp_path, monkeypatch, **kwargs)
    exc = info.value
    assert getattr(exc, "dimer_refusal", False) is True, repr(exc)
    lines = exc._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith("Refused: ") and "Traceback" not in lines[0], lines
    return lines[0]


# ---- each refusal path in Section 4 is one clean line naming the file ---------------------------------------


def test_acquisition_refusals(tmp_path, monkeypatch):
    missing = str(tmp_path / "missing.zip")
    assert _refusal(tmp_path, monkeypatch, path=missing).startswith(f"Refused: {missing}: BYOD_PATH does not exist")
    assert _refusal(tmp_path, monkeypatch, upload={}).startswith("Refused: (no file): upload exactly one .zip file (received 0)")
    two = {"a.zip": b"x", "b.zip": b"y"}
    assert _refusal(tmp_path, monkeypatch, upload=two).startswith("Refused: a.zip, b.zip: upload exactly one .zip file")
    assert "needs Google Colab" in _refusal(tmp_path, monkeypatch, colab=False)


def test_class_prompt_refusals_name_the_field(tmp_path, monkeypatch):
    line = _refusal(tmp_path, monkeypatch, upload={"a.zip": _folder_zip(7)}, prompts=["not", "a", "mapping"])
    assert line.startswith("Refused: BYOD_CLASS_PROMPTS: it must map each label")
    two = CLASSES[:2]
    line = _refusal(tmp_path, monkeypatch, upload={"a.zip": _folder_zip(7, two)}, prompts=SAMPLE_CLASS_PROMPTS)
    assert line.startswith("Refused: a.zip: BYOD_CLASS_PROMPTS must name exactly the labels found")


def test_content_and_contract_refusals_name_the_file(tmp_path, monkeypatch):
    bad = tmp_path / "housing.csv"
    bad.write_bytes(HOUSING)
    assert "missing required column" in _refusal(tmp_path, monkeypatch, path=str(bad))
    assert _refusal(tmp_path, monkeypatch, upload={"broken.zip": b"not a zip"}).startswith("Refused: broken.zip: ")
    assert "supply at least 5 records per class" in _refusal(tmp_path, monkeypatch, upload={"small.zip": _folder_zip(3)})
    line = _refusal(tmp_path, monkeypatch, upload={"e.zip": _folder_zip(5, extra={"../byod_evil/x.txt": b"x"})})
    assert line.startswith("Refused: e.zip: zip member escapes the upload directory")
    assert line.endswith("Fix the file, then select this cell and choose Runtime → Run after.")


def test_a_refusal_removes_the_earlier_runs_data_and_results(tmp_path, monkeypatch):
    ns = _section4(tmp_path, monkeypatch, upload={"good.zip": _folder_zip(7)})
    assert ns["train_records"] and (tmp_path / "outputs" / "byod").is_dir()
    (tmp_path / "outputs" / f"{STEM}_evaluation_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "outputs" / f"{STEM}_adapter").mkdir()
    monkeypatch.chdir(tmp_path)
    source = refusing_section4_source(str(tmp_path / "missing.zip"), json.dumps(SAMPLE_CLASS_PROMPTS))
    with pytest.raises(ValueError, match="does not exist"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not list((tmp_path / "outputs").glob(f"{STEM}_*")) and not (tmp_path / "outputs" / "byod").exists()
    for name in ("records", "splits", "train_records", "val_records", "test_records", "CLASSES", "data_source"):
        assert name not in ns, name


def test_the_sample_path_is_unchanged_and_a_failure_there_still_raises(tmp_path, monkeypatch):
    ns = _section4(tmp_path, monkeypatch, byod=False)
    assert [len(ns[k]) for k in ("train_records", "val_records", "test_records")] == [28, 8, 12]

    def broken():
        raise ValueError("internal sample check failed")

    ns["generate_sample_dataset"] = broken
    source = CELLS[SEC4]
    with pytest.raises(ValueError, match="internal sample check failed") as info:
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not getattr(info.value, "dimer_refusal", False)


WORKER_SETUP = [f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r})", "from bioclip2_biodiversity_pipeline import *"]
BAD_NAME, BAD_BYTES, BAD_RULE, DISCARDED = "housing.csv", HOUSING, "missing required column", "records"
_UNUSED = (io, zipfile)


# ---- the router: a refusal crosses the worker boundary as one line; a genuine error keeps its traceback -----


def _router_namespace() -> dict:
    """The router cell's definitions (everything before it starts the worker), with a stand-in IPython."""
    src = CELLS[ROUTER]
    head = src[: src.index("\nif SKIP_INSTALL:")]
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: None
    saved = sys.modules.get("IPython")
    sys.modules["IPython"] = ipython
    try:
        ns: dict = {"__name__": "__main__", "os": os, "subprocess": subprocess, "sys": sys}
        exec(compile(head, f"<cell {ROUTER}>", "exec"), ns)
    finally:
        if saved is None:
            sys.modules.pop("IPython", None)
        else:
            sys.modules["IPython"] = saved
    return ns


class _Queued:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _stub_runtime(ns, messages):
    runtime = object.__new__(ns["IsolatedRuntime"])
    runtime._recv, runtime._send, runtime._display = _Queued(messages), _Queued([]), lambda *a, **k: None
    return runtime


def test_router_carries_the_refusal_channel():
    src = CELLS[ROUTER]
    assert 'if getattr(exc, "dimer_refusal", False):' in src and '_send.send(("refusal", str(exc)))' in src
    assert 'elif kind == "refusal":' in src and "class IsolatedCellRefusal(Exception):" in src


def test_router_kernel_shows_a_refusal_as_one_line_and_an_error_with_its_traceback(capsys):
    ns = _router_namespace()
    with pytest.raises(ns["IsolatedCellRefusal"]) as info:
        _stub_runtime(ns, [("refusal", "data.csv: a column is missing")]).run("cell")
    assert info.value._render_traceback_() == ["Refused: data.csv: a column is missing"]
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    trace = "Traceback (most recent call last):\n  File x\nZeroDivisionError: division by zero\n"
    with pytest.raises(ns["IsolatedCellError"], match="ZeroDivisionError"):
        _stub_runtime(ns, [("error", trace, "ZeroDivisionError: division by zero")]).run("cell")
    assert capsys.readouterr().err == trace


@pytest.mark.skipif(sys.platform == "win32", reason="the router's worker uses pass_fds and POSIX sessions")
def test_real_worker_shows_the_section4_refusal_as_one_line_and_keeps_running(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ns = _router_namespace()
    runtime = ns["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        for source in WORKER_SETUP:
            runtime.run(source)
        bad = tmp_path / BAD_NAME
        bad.write_bytes(BAD_BYTES)
        capsys.readouterr()
        with pytest.raises(ns["IsolatedCellRefusal"]) as info:
            runtime.run(refusing_section4_source(str(bad)))
        lines = info.value._render_traceback_()
        captured = capsys.readouterr()
        assert len(lines) == 1 and lines[0].startswith(f"Refused: {BAD_NAME}: "), lines
        assert BAD_RULE in lines[0], lines
        assert "Traceback" not in captured.out + captured.err, captured
        runtime.run(f"print('still running', {DISCARDED!r} in globals())")
        assert capsys.readouterr().out == "still running False\n"
        with pytest.raises(ns["IsolatedCellError"]):
            runtime.run("1 / 0")
        assert "Traceback" in capsys.readouterr().err
    finally:
        runtime.close()
