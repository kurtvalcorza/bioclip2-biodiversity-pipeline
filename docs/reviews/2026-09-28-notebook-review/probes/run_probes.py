
"""Run every review probe against the notebook. Python >= 3.10, standard library only.
Usage: python run_probes.py path/to/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb"""
import ast, json, pathlib, subprocess, sys
nb_path = pathlib.Path(sys.argv[1]).resolve(); here = pathlib.Path(__file__).parent
nb = json.loads(nb_path.read_text(encoding="utf-8"))
carrier = next(c for c in nb["cells"] if c.get("metadata", {}).get("dimer", {}).get("embedded_sources"))
files = ast.literal_eval(ast.parse("".join(carrier["source"])).body[0].value)
tmp = here / "_extracted"; tmp.mkdir(exist_ok=True); (tmp / "data_manifest.json").write_text(files["data_manifest.json"], encoding="utf-8")
jobs = [("p1_carrier_integrity.py", [str(nb_path)]), ("p2_sample_audit.py", [str(tmp / "data_manifest.json")]),
        ("p3_bootstrap_absent_class.py", [str(tmp / "data_manifest.json")]), ("p4_head_displacement.py", ["--contrast"]),
        ("p5_static_source_checks.py", [str(nb_path)]), ("p6_blank_control_digest.py", [str(tmp / "data_manifest.json")])]
for script, args in jobs:
    result = subprocess.run([sys.executable, str(here / script), *args], capture_output=True, text=True, encoding="utf-8")
    (here / script.replace(".py", ".out.json")).write_text(result.stdout, encoding="utf-8")
    print(script, "exit", result.returncode)
