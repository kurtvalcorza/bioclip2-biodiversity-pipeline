"""The capstone carrier is written as short string pieces, not one 374,826-character line.

A single very long notebook line can make the Colab editor unresponsive. The generator splits every
carried string into implicitly concatenated pieces; Python joins them back into identical text.
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_biodiversity_capstone as builder  # noqa: E402

NOTEBOOK = ROOT / "tutorials" / builder.NAME
MAX_LINE = 2000


def test_no_notebook_line_exceeds_the_limit():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    longest = max((len(line), cell["id"]) for cell in notebook["cells"] for line in cell["source"])
    assert longest[0] <= MAX_LINE, longest


def test_carried_literals_round_trip():
    samples = ["", "x" * 2500 + "\n", "first\nsecond\r\nthird 'quoted' \"double\" \\ é\n\nlast"]
    for value in samples:
        literal = builder.carried_literal(value)
        assert ast.literal_eval(literal) == value
        assert all(len(line) <= MAX_LINE for line in literal.splitlines())
    files = {f"file{i}.txt": value for i, value in enumerate(samples)}
    assert ast.literal_eval(builder.carried_dict_literal(files)) == files
