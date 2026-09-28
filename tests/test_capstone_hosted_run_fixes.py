"""Regression tests for the first hosted Colab T4 run of the capstone (revision 0.2.0-candidate).

That run (recorded in docs/execution-evidence/2026-09-28/) stopped in the first stage:
Colab exports MPLBACKEND=module://matplotlib_inline.backend_inline, the bootstrap passed the
kernel environment to every stage process, and `import matplotlib` failed inside the isolated
environment, which has no matplotlib-inline.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_biodiversity_capstone as builder  # noqa: E402

COLAB_BACKEND = "module://matplotlib_inline.backend_inline"


def _code(cell_id):
    notebook = builder.build()
    return next("".join(c["source"]) for c in notebook["cells"] if c["id"] == cell_id)


def test_stage_module_imports_under_the_colab_inline_backend():
    pytest.importorskip("matplotlib")
    env = dict(os.environ, MPLBACKEND=COLAB_BACKEND)
    probe = (
        "import sys; sys.path.insert(0, 'tools'); import biodiversity_capstone, matplotlib; "
        "print(matplotlib.get_backend().lower())"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "agg"


def test_bootstrap_does_not_leak_kernel_settings_into_stage_processes():
    code = _code("code-05")
    import time

    namespace = {"os": os, "time": time}
    start = code.index("ENV = dict(os.environ")
    end = code.index("subprocess.run([str(UV), 'venv'")
    kernel = {"MPLBACKEND": COLAB_BACKEND, "PYTHONPATH": "/env/python", "HF_TOKEN": "x", "KEEP": "1"}
    saved = dict(os.environ)
    try:
        os.environ.update(kernel)
        exec(code[start:end], namespace)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    env = namespace["ENV"]
    assert env["MPLBACKEND"] == "Agg"
    assert "PYTHONPATH" not in env and "HF_TOKEN" not in env
    assert env["KEEP"] == "1" and env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_bootstrap_reports_progress_and_the_gpu_it_verified():
    code = _code("code-05")
    assert "Installing the hashed dependency lock" in code
    assert "torch.version.cuda" in code and "get_device_name(0)" in code
    assert "print(check.stdout.strip())" in code and "Environment ready in" in code


def test_every_stage_process_uses_the_sanitised_environment():
    code = _code("code-05") + _code("code-34")
    assert code.count("env=ENV") >= 5
    assert "env=os.environ" not in code
