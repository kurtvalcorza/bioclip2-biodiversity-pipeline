
"""Probe P5: source-level checks for specific review findings (stdlib only).
Each check reports the evidence line(s) it relies on. Source inspection only; nothing is executed."""
import ast, json, re, sys
nb = json.load(open(sys.argv[1], encoding="utf-8")); cells = nb["cells"]
carrier = next(c for c in cells if c.get("metadata", {}).get("dimer", {}).get("embedded_sources"))
tree = ast.parse("".join(carrier["source"])); files = ast.literal_eval(tree.body[0].value)
cap, core = files["capstone.py"], files["biodiversity_core.py"]
md = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
code = {c["id"]: "".join(c["source"]) for c in cells if c["cell_type"] == "code" and c is not carrier}
def lines(text, pattern):
    return [f"{i}: {l.strip()}" for i, l in enumerate(text.splitlines(), 1) if re.search(pattern, l)]
out = {}
# F-SigLIP: documented lowercasing and max_length=64
sig = cap[cap.index("def siglip_zero_shot"):cap.index("def bioclip_pipe")]
out["siglip_prompt_lines"] = lines(sig, r"This is a photo of|padding=|max_length|lower\(")
out["siglip_prompts_lowercased"] = ".lower()" in sig
out["siglip_max_length_64_passed"] = "max_length=64" in sig
out["siglip_class_names"] = [(c["common_name"], c["scientific_name"]) for c in json.loads(files["data_manifest.json"])["classes"]]
# F-Head: optimiser budget
m = re.search(r"CONFIG = \{(.*?)\n\}", cap, re.S); out["CONFIG"] = m.group(0)
out["head_init_and_steps"] = lines(cap, r"head.weight.copy_|split\(8\)|for epoch in range\(epochs \+ 1\)|if val_loss < best_loss")
# F-Bootstrap: macro-F1 counts absent classes as F1=0
out["macro_f1_zero_convention"] = lines(core, r"f1 = 2 \* tp|macro_f1\": float")
out["bootstrap_resampling"] = lines(core, r"rng.choice\(unique|default_rng")
# F-Runtime / show_record filter
out["show_record_filter"] = lines(code["code-05"], r"isinstance\(item, \(str, float, int, bool\)\)")
rs = cap[cap.index('out / "run_summary.json"'):cap.index('(out / "conclusion.md")')]
top_keys = re.findall(r'^\s{12}"(\w+)":', rs, re.M)
out["run_summary_top_level_keys"] = top_keys
out["run_summary_keys_rendered_by_show_record"] = [k for k in top_keys if k in ("status", "elapsed_seconds")]
out["elapsed_seconds_definition"] = lines(cap, r"elapsed_seconds|STAGES\[:-1\]")
out["bootstrap_seconds_written_outside_outputs"] = lines(code["code-05"], r"bootstrap.json")
out["dataset_download_timing_recorded"] = bool(re.search(r"dataset.*(seconds|download)", cap[cap.index("def prepare"):cap.index("def scalar_metrics")]))
# F-Archive verification record
ver = cap[cap.index("verification = {"):cap.index('core.write_json(out / "verification.json"')]
out["verification_json_keys"] = re.findall(r'"(\w+)":', ver)
out["archive_verification_excluded_from_zip"] = '"archive_verification.json"' in cap and "p.name not in" in cap
out["archive_verification_displayed_in_notebook"] = "archive_verification" in "".join(code.values())
# F-Activity
out["activity_offsets"] = lines(code["code-26"], r"offset =|display_threshold =")
out["show_table_number_format"] = lines(code["code-05"], r":\.4f")
# F-Reload batch composition
out["reload_expected_source"] = lines(cap, r"reload_expected|embed_images\(images\(root, selected\)\)|range\(0, len\(records\), 8\)|records\[:1\]")
out["parity_tolerances"] = lines(cap, r'"atol"|"rtol"') + lines(core, r"def parity|atol: float")
# F-Error panel categories
ev = cap[cap.index("def evaluate"):cap.index("def triage")]
out["error_panel_annotation_has_category_name"] = bool(re.search(r"(correct high|incorrect high|lowest margin|disagree)", ev.split("annotations = [")[1].split("]")[0], re.I))
out["error_panel_dedupe"] = lines(ev, r"for j in candidate\[:1\]|if int\(j\) not in chosen")
# F-Majority top-2
out["majority_scores"] = lines(cap, r"majority\[:, winner\] = 1")
# F-risk plot ordering
out["canonical_row_appended_after_sweep"] = lines(cap, r"rows.extend\(")
# F-Probes: only head displayed
out["probe_predictions_systems"] = lines(cap, r'probe_rows = prediction_rows')
# F-Spec 16 statements
out["md_mentions"] = {k: bool(re.search(p, md, re.I)) for k, p in {
    "endemic_status_vs_location_exposure": r"endemic[^.]*(location|coordinate)",
    "must_not_reconstruct_coordinates": r"reconstruct[^.]*(coordinate|location)",
    "does_not_count_individuals": r"count(ing)? individuals|does not count",
    "abundance_or_population": r"abundance|population",
    "learning_objectives_listed": r"learning objectives|by the end",
    "subspecies_record_disclosed": r"subspecies",
    "centre_crop_view_offered_to_learner": r"(show|display|view)[^.]*centre[- ]crop",
}.items()}
out["contact_sheet_shows_full_frame"] = lines(cap, r"ax.imshow\(im\)") + lines(cap, r"ImageOps.exif_transpose\(image\).convert\(\"RGB\"\).copy\(\)")
out["filelink_absolute_paths"] = lines("".join(code.values()), r"FileLink\(str\(")
out["disk_requirement"] = lines(code["code-03"], r"1024\*\*3")
out["pca_palette"] = lines(cap, r"get_cmap\(")
print(json.dumps(out, indent=2, ensure_ascii=False))
