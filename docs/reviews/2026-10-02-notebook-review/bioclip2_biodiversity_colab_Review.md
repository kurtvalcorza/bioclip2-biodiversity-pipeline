# BioCLIP 2 Species-Classification E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/bioclip2-biodiversity-pipeline`  
**Notebook:** `tutorials/bioclip2_biodiversity_colab.ipynb`  
**Reviewed commit:** `2def984808bede169e7626a2f6f962c60003a89c` (`main`, confirmed with `gh api repos/kurtvalcorza/bioclip2-biodiversity-pipeline/commits/main`)  
**Notebook Git blob:** `60e7655fe544c7438423aa5bcf4ae92ddcb40cd1`. This is the blob committed at `a43fdf5` and executed in the recorded Kaggle run at `22f2854`. Later commits added the separate capstone notebook and docs; generator `--check` exits 0 at the reviewed commit.  
**Finding prefix:** `BIO`  
**Scope note:** this review covers the original tutorial only. The Philippine capstone notebook in the same repository has its own review (`docs/reviews/2026-09-28-notebook-review/`) and is not assessed here.

## Executive assessment

The engineering is careful. The notebook carries its four modules byte for byte (generator `--check` and the release validator both exit 0). It digest-verifies a pinned 1.71 GB snapshot and builds the model from the pinned config with a strict load. It ships 48 licensed, provenance-tracked photographs, and it states score semantics plainly, including blank, noise and open-set probes. Majority and colour baselines are fitted on the training split only, and reload parity is asserted. A direct CPU execution of all 15 code cells, verbatim, with the real weights and the exact pins, reproduced the recorded numbers to four decimals: zero-shot 12/12, colour 0.25 / 0.1909, blank 0.5276, noise 0.8531, test 1.0 and reload parity 0.0.

Five problems stand in the way of `Ready for intended use`:

1. `Run all` in a fresh hosted runtime does not finish in one pass. The recorded qualification run itself needed "one expected interpreter restart" after the in-kernel pinned install, yet `tutorials/README.md` records `Run-all … PASS` (BIO-M1).
2. The default fine-tuning does not fine-tune. The head starts at the zero-shot text classifier scaled by 100, and `lr = 1e-4` for 16 AdamW steps moves it by **0.022 %**. The "adapted" model *is* the zero-shot model: argmax agreement is 100 % and the largest score difference is 5.8e-4. When zero-shot is wrong, the default adaptation learns nothing: with swapped names, test accuracy is 0.0 → 0.0 and loss stays flat at 9.8. The notebook's lessons, that head-only adaptation "preserves" zero-shot while unfreezing a block "degrades" it and that BYOD is where "the value of adaptation shows", rest on this artefact (BIO-M2).
3. The advertised BYOD layout, a zip of `<label>/<image>` folders, **always fails** in Section 4 with `image file not found: work\byod\work\byod\…`. The loader joins the relative upload root twice. Only the CSV/JSON layout works (BIO-M3).
4. The BYOD contract's stated limits are not the enforced ones. 12 records with 3 per class pass Sections 4–8 and fail in Section 9 with "dataset has 4 records". A second upload silently trains on the first upload's data, and several common inputs get unhelpful or late errors (BIO-M4).
5. The notebook is declared `GUIDED`, but most of the guided layer is absent, and about 9,900 lines of carried code (8,557 of them base64 photographs) are not labelled or collapsed as infrastructure (BIO-M5).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, and the opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main` |
| Intended audience | Not stated. The Prerequisites ask for knowledge of binomial names, zero-shot classification with a vision-language model, and accuracy / macro-F1 / AUROC |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU float32, CUDA when present; 1.71 GB download |
| Promised outcomes | One-pass `Run all` with no configuration edit; pinned install; four carried modules; staged, digest-verified snapshot; 48 embedded CC0 photographs, validation, 28/8/12 stratified split, three refusals; reproducible embeddings; zero-shot with common-name, open-set, blank and noise probes; majority and colour baselines; "bounded AdamW fine-tuning of a linear head initialised from the zero-shot classifier"; held-out accuracy / macro-F1 / AUROC with deltas; six held-out predictions; safetensors adapter with digest-verified reload parity; seven `outputs/` files; BYOD "through the same … cells" as a zip of `<label>/<image>` folders or a CSV/JSON/JSONL |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; carried modules from `src/bioclip2_biodiversity_pipeline/` @ `3ae1131` |

### Evidence actually obtained

- **Source inspection:** all 33 cells (15 code), the carried `pipeline.py` (`_check_images`, `decode_image`, `zero_shot_evaluate`, `_build_classifier`, `adapt`, `save_artifact`, `load_artifact`), `samples.py` (`validate_dataset`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv`), the generator and template, `tests/test_adaptation.py`, `README.md`, `STATUS.md`, `MODEL_CARD.md`, `tutorials/README.md` and `docs/release-verification.md`.
- **Documented execution evidence:** `docs/release-verification.md` and `MODEL_CARD.md`. The run was on Kaggle Tesla T4, 2026-09-18, commit `22f2854` / **blob `60e7655f`, the reviewed blob**: 15/15 code cells, 265.3 s, "after one expected interpreter restart". The archived executor output for that run was not located in the workspace, so the record was read, not the executed notebook. No Colab run of this blob is recorded. No BYOD run is recorded.
- **Direct execution (this review):** `run_probes.py` on Windows, CPython 3.12.10, in the repository's dedicated venv (`torch 2.14.0+cpu`, `open_clip 3.3.0`, `safetensors 0.8.0`, `pillow 12.3.0`, `numpy 2.5.3`, i.e. the notebook's exact pins), CPU only. It used the **real BioCLIP 2 weights** (SHA-256 `b7b2bf6f…`, equal to the manifest), hard-linked read-only into a scratch working directory. The notebook wrote its own manifest there and `stage_missing_files` fetched nothing, so the runtime was not clean and nothing was downloaded. Cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, which skips the install. Every code cell ran **verbatim from the notebook JSON** in one namespace. Only form-field literals were substituted, as an executor sets fields, and `google.colab.files.upload` was replaced by a fake for BYOD. Probes P2 and P3 also call the carried API directly, and are labelled as such. Static checks: JSON parse, compile of all 15 code cells, blob id, generator `--check` (exit 0), `tools/validate_release_assets.py` (exit 0, "PASS (model-card, identity-consistency, weight-facts, release-status, notebook+parity)").
- **Not verified:** a Colab run of any kind; a one-pass hosted `Run all`; the CUDA path; BYOD through the real upload widget; memory use (three ViT-L image towers are resident by Section 11); learner understanding.

## 2. Separate judgments

- **Technical correctness:** sound on the default path. Identity, digests, strict load, train-only baselines, split ownership, exports and reload parity all check out, and direct execution matched the record. Three technical defects: the restart-dependent install (BIO-M1); the zip-of-folders BYOD layout, which cannot load (BIO-M3); and the zip guard, which accepts a member that escapes into a sibling directory (BIO-m1).
- **Scientific validity:** the zero-shot, baseline and score-semantics material is careful and correct. The adaptation experiment is not valid as taught. The default head does not move, so "delta vs zero-shot 0.0", "head-only preserves", "one block degrades, therefore default 0" and "adaptation earns its keep on BYOD" are each an artefact of a learning rate that is about 1e-4 of the head's weight scale (BIO-M2).
- **Promise fulfilment:** the default capability list is delivered except "bounded fine-tuning", which runs but has no effect. The BYOD promise is not delivered for the advertised layout (BIO-M3) and is delivered unreliably for the working layout (BIO-M4). One-pass `Run all` is not delivered (BIO-M1).
- **Learner experience:** the prose is unusually honest about limits, and the probes teach score semantics well. But the notebook asks for nothing beyond running cells. The three optional experiments come without rerun instructions, and the notebook opens with about 9,900 lines of unlabelled infrastructure (BIO-M5, BIO-m3).
- **Spec conformance (2.2):** applicable `MUST`s fail: RUN1, RUN10 and ENV6 (BIO-M1); DAT10, DAT14 and REL12 (BIO-M3); DAT12 and DAT19 (BIO-M4); and §20 "extracted paths MUST resolve within the intended root" (BIO-m1). GDL1–GDL15 are largely unmet (SHOULD). Several EXE `SHOULD`s are unmet (BIO-m6). The declared spec is 2.0 (BIO-S1).

## 3. Promise and objective tracing

| Claim (where) | Implementation | Observable result (this review) | Learner interpretation |
|---|---|---|---|
| One-pass `Run all`, no intervention (cell 0) | cell 3 in-kernel `pip install` + stale-import guard | Kaggle record: restart needed after install; local: install skipped | **Not delivered** (BIO-M1) |
| Pinned, digest-verified model from pinned config (cells 12–13) | `stage_missing_files`, `verify_snapshot`, `from_pretrained` | 3 entries verified, strict load, `fetched: []` (pre-staged) | Delivered |
| 48 CC0 photos, validated, split 28/8/12 (cells 14–15) | `generate_sample_dataset`, `validate_dataset`, `split_dataset` | 48 / 4×12, digest `f8fc68cf…`, 28/8/12, contact sheet | Delivered |
| Refusals + acceptance (cells 16–17) | `validate_inputs` | garbage, BMP and 16×16 refused with the rule named | Delivered |
| Reproducible embeddings (cells 18–19) | `embed_images` ×3 | same batch identical; cross-batch below `1e-5`; CSV with ids | Delivered |
| Zero-shot + probes (cells 20–21) | `zero_shot_evaluate`, `zero_shot` | 12/12; blank → 0.5276, noise → 0.8531; open set → species 1.0 | Delivered, and well explained |
| Baselines on test (cells 22–23) | `majority_baseline`, `color_baseline` | 0.25 / 0.1 and 0.25 / 0.1909 | Delivered |
| "Bounded fine-tuning … initialised from the zero-shot classifier" (cells 24–25) | `adapt(lr=1e-4, epochs=4)`, head-only on cached features | head displaced 0.022 %; argmax = zero-shot on 12/12; max score Δ 5.8e-4 | **Runs, but nothing is learned** (BIO-M2) |
| Held-out evaluation with deltas (cells 26–27) | `evaluate`, report JSON | test 1.0, delta vs zero-shot 0.0 | The 0.0 delta is by construction, not a finding (BIO-M2) |
| Six held-out predictions; export; reload parity (cells 28–29) | `classify`, `save_artifact`, `from_artifact` | 6/6; 12,448-byte adapter; parity 0.0 | Delivered |
| Provenance export (cells 30–31) | `result.json` | written with identity, runtime and manifests | Delivered |
| `TRAINABLE_BLOCKS = 1` "took … 12/12 to 11/12" (cell 24) | same cell, field = 1 | test 0.9167; val 1.0 → 0.875; 14 tensors, 50.4 MB; parity 0.0 | The number reproduces; the explanation is confounded (BIO-M2) |
| BYOD zip of `<label>/<image>` folders "through the same cells" (cells 0, 1, 14) | cell 15 extract + `load_byod_dataset(dir)` | `FileNotFoundError … work\byod\work\byod\…` on every folder zip | **Not delivered** (BIO-M3) |
| BYOD CSV in a zip; "at least 8 records and 3 per class" (cell 1) | `load_byod_dataset(csv)` | 4×7 runs through Section 12 with parity 0.0; 4×3 and 4×4 fail in Section 9 | Partly delivered (BIO-M4) |

| Objective (cell 0) | Learner activity | Evidence it was exercised |
|---|---|---|
| "install the pinned runtime"; "inspect the carried … modules"; "stage and digest-verify" | run cells | Procedural only; nothing asks the learner to inspect anything |
| "validate images and split … without leakage" | read printed refusals | Shown, not practised |
| "classify zero-shot … and read what the scores do and do not mean" | read the probes | The best-taught objective, but no question checks the reading |
| "run a bounded fine-tuning whose starting point *is* the zero-shot classifier" | run cell 25 | Runs; the result cannot differ from zero-shot (BIO-M2) |
| "evaluate … on an independent test split"; "export … with verified parity" | run cells | Shown |
| (optional) `TRAINABLE_BLOCKS = 1`, prompt template, BYOD | edit and rerun, with no rerun instructions | The block experiment reproduces; the template one needs a code edit; folder BYOD fails |

## 4. Prioritized findings

### BIO-M1 — Major: fresh-runtime `Run all` needs a manual restart after the install cell, yet the registry records a `Run all` PASS

- **Cell/section:** Section 1 (cell 3); generator `tools/build_notebook.py:47-70` (`_INSTALL_GUARD`) and the Section 1 template.
- **Observed issue:** cell 3 pip-installs 10 pins into the running kernel and raises `RuntimeError(… 'Restart the runtime, then rerun from the top.')` when a pin replaces an already-imported distribution. The recorded qualification run (`docs/release-verification.md`, Kaggle T4, `22f2854` / `60e7655f`) says: "one expected interpreter restart after dependency installation; all 15 code cells completed". `tutorials/README.md` records, in its Run-all column, "exact commit `22f2854` / blob `60e7655f`: fresh Kaggle Tesla T4 `Run all` PASS, 15/15 code cells". The opening cell promises that Run all completes with "no configuration edit". The spec forbids exactly this manual restart (§1, RUN10, §25.7).
- **Consequence:** a learner's first `Run all` stops in cell 3 with a restart instruction. The release record calls a two-pass run a `Run all` PASS.
- **Evidence:** documented execution (the release-verification row, the MODEL_CARD "after one expected interpreter restart"); source inspection of cell 3. Whether Colab's preloaded NumPy/torch trigger the guard the same way is **not verified** (no Colab run), but the pins (`numpy==2.5.3`, `torch==2.14.0`) are newer than typical hosted images.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass. The setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`) and installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`). It then runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: **this repository's own** `tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb` (generator `tools/build_biodiversity_capstone.py:146-151`, lock `tools/biodiversity-requirements.lock`), and `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/build_notebook.py` / `tools/notebook_template.py`, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so that a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Colab (or Kaggle) runtime runs **Run all** once, with no restart and no intervention, through the final cell. The recorded run states "no restart". `tutorials/README.md` / `docs/release-verification.md` no longer call the 2026-09-18 two-pass run a Run-all PASS.
- **Spec:** RUN1, RUN10, ENV6, REL2, §25.7, §27 (MUST).

### BIO-M2 — Major: the default "fine-tuning" leaves the head at the zero-shot classifier, so the adaptation results and the lessons drawn from them are artefacts

- **Cell/section:** Sections 9–10 (cells 24–27), the Interpretation (cell 32), README "What the sample showed", MODEL_CARD deployment note. The code is the carried `pipeline.py` `adapt()` (head init `text_vecs * logit_scale`, `learning_rate=1e-4`) and the template's `LEARNING_RATE = 1e-4` (`tools/notebook_template.py:344`).
- **Observed issue:** the head is initialised to `100 × unit text embeddings`, a mean absolute weight of 2.86. AdamW at `lr = 1e-4` for 4 epochs × 4 batches can move each weight by about 1.6e-3, i.e. by about 0.05 %. Direct measurement:

  | Probe (P2/P3; CPU, real weights, the notebook's split and seed) | zero-shot test | adapted test | head displacement | note |
  |---|---|---|---|---|
  | default, scientific names (notebook path) | 1.0 | 1.0 | **0.022 %** | argmax = zero-shot on 12/12; max score Δ 5.8e-4 |
  | swapped names (each class gets another species' name), lr 1e-4 | 0.0 | **0.0** | 0.024 % | train loss 9.81 → 9.85; val 0.0 at every epoch |
  | neutral names `species k`, lr 1e-4 | 0.25 | **0.25** | 0.027 % | val 0.125 at every epoch |
  | neutral names, lr 1e-2 (direct API, not the notebook default) | 0.25 | 0.667 | 2.2 % | val 0.125 → 0.5 |

  The same `lr = 1e-4` applied to an unfrozen tower block is a far larger relative step (transformer block weights are typically of order 1e-2; their scale was not measured here). `TRAINABLE_BLOCKS = 1` drives the training loss to 6e-6 and test accuracy to 0.9167 (P4). The notebook's comparison "head-only preserves the zero-shot classifier; one block degrades it" therefore contrasts a head that cannot move with a block that moves a great deal. That comparison is "the recorded reason the default is 0" (cell 24, README, MODEL_CARD).
- **Consequence:** the central `E2E` demonstration of adaptation shows nothing. The `delta_vs_zero_shot 0.0` is guaranteed whatever the data. A learner is told that "the value of adaptation shows on a dataset where zero-shot is *not* saturated … which is what BYOD is for". On exactly such data the default configuration does not learn. The capstone in this repository fixed the same defect as BC-M1 (`docs/release-verification.md`, "head effectively equals zero-shot (lr 1e-4 × 80 steps on scale-100 weights)"). The original tutorial was not updated.
- **Evidence:** direct execution (P2, P3, P4 in `results.json`); source inspection of `adapt()`.
- **Recommended correction:** port the capstone's BC-M1 fix to `pipeline.adapt` and the template. Predeclare a head learning-rate grid (for example 1e-4 / 1e-3 / 1e-2) chosen on validation, keep epoch 0 (zero-shot) eligible so that "no change" can legitimately win, and report the head displacement and the zero-shot agreement in the history and in `evaluation_report.json`. Alternatively, parametrise the head as `logit_scale × W` with unit-scale `W`, so that one learning rate means the same thing for the head and the tower. Rewrite cells 24, 26 and 32, the README and the MODEL_CARD so that "preserves" and "degrades" are claimed only as the measured displacement supports. Add a model-backed or synthetic regression test asserting that the default configuration can move a head away from a wrong zero-shot initialisation.
- **Acceptance check:** (a) re-running P3's neutral-names case with the notebook's default adaptation settings raises validation or test accuracy above the epoch-0 value, or the notebook selects epoch 0 and says so. (b) Section 9/10 output prints the head displacement and the argmax agreement with zero-shot. (c) No learner-facing text attributes the default's choice to the head-versus-block comparison unless both are run at comparable relative step sizes.
- **Spec:** FT2 (real adaptation), FT4, EVAL14, UX1 (objective corresponds to what runs); scientific validity (framework dimension 3).

### BIO-M3 — Major: the advertised BYOD layout, a zip of `<label>/<image>` folders, always fails in Section 4

- **Cell/section:** Section 4 (cell 15); carried `samples.py` `load_byod_dataset` directory branch (`src/bioclip2_biodiversity_pipeline/samples.py`, the `if path.is_dir()` block and the later `img_path = base / img_path`).
- **Observed issue:** cell 15 extracts the upload under the **relative** `Path('work') / 'byod'` and passes that directory to `load_byod_dataset`. The directory branch records each image as `str(img)`, which already includes `work/byod/<label>/`. It then sets `base = path` and, because the stored path is relative, joins `base` again, giving `work/byod/work/byod/<label>/<file>`. Every folder-layout upload therefore raises `FileNotFoundError: record[0] (chipping_sparrow/chipping_sparrow_00.jpg) image file not found: work\byod\work\byod\chipping_sparrow\chipping_sparrow_00.jpg`. The unit test `tests/test_adaptation.py::test_byod_directory_layout` passes only because `tmp_path` is absolute.
- **Consequence:** the layout the opening, the Prerequisites and Section 4 offer first does not work, and the error message points at a path the user never created. A learner cannot reach validation, adaptation or evaluation on their own photographs except through the CSV route.
- **Evidence:** direct execution, 9 folder-layout probes (P5 B1–B8, including a 4×7 positive case), all failing at cell 15 with the doubled path. The CSV-in-zip layout loads (P5 B10; P6 C1 runs Sections 4–12 with reload parity 0.0). Source inspection of the loader and the test.
- **Recommended correction:** in `load_byod_dataset`'s directory branch, store the image path relative to the root (`img.relative_to(path)`), or resolve `path` first. Add a regression test that `monkeypatch.chdir(tmp_path)` and loads a *relative* directory. Regenerate the notebook. Record a BYOD run (positive and negative) in `docs/release-verification.md`.
- **Acceptance check:** with `USE_BYOD = True` and the sample prompts, a zip of 4 label folders × 7 photographs runs cells 15→31 without error and writes `result.json` with `data_source` naming the zip. The relative-directory regression test fails on `2def984` and passes after the fix.
- **Spec:** DAT10, DAT13, DAT14, REL12 (MUST); §25.9.

### BIO-M4 — Major: the BYOD contract's stated limits are not the enforced ones; bad or repeated uploads fail late or silently

- **Cell/section:** Prerequisites (cell 1), Section 4 (cell 15), Section 7 (cell 21), Section 9 (cell 25); carried `samples.py` (`split_dataset`, `validate_dataset`), `pipeline.adapt`, `pipeline.decode_image`.
- **Observed issue** (all via the working CSV-in-zip layout; direct execution P6, plus P5 B9):
  1. **Wrong minimum.** The Prerequisites say "at least 8 records and 3 per class". `adapt` re-validates the **training split** against those same minimums. 4 classes × 3 (12 records) pass Sections 4–8 and then fail in Section 9 with `ValueError: dataset has 4 records; at least 8 are required`, naming neither the split nor the real rule. 4 × 4 also fails (`fewer than 3 records each`). 4 × 5 and 2 × 7 pass Section 9. By the split arithmetic, the real minimum is 5 per class for 4 classes and 7 per class for 2 (2 × 6 leaves 6 training records; that probe hit item 3 first, so this case is arithmetic, not executed).
  2. **Silent stale data.** The upload directory is never cleared. Uploading `b.zip` (with `b/labels.csv`) after `a.zip` printed `data_source: 'BYOD (b.zip)'` but trained on `a/labels.csv`, i.e. upload A's classes, because `tables[0]` is the first table in sorted order.
  3. **Extra prompt entries crash Section 7.** The cell checks only for *missing* `BYOD_CLASS_PROMPTS` entries. A dict with extra species, such as the sample's four for a two-class upload, fails in `zero_shot_evaluate` with `classes ['dark_eyed_junco', 'white_throated_sparrow'] have fewer than 1 records each`.
  4. **MPO JPEGs refused.** A `.jpg` that Pillow reads as `MPO` (the multi-picture JPEG some cameras write) is rejected with `image format MPO is not accepted`, and the message gives no conversion advice. How common MPO is among learners' photos is **inferred**, not measured.
  5. **Enclosing folder and cancelled upload.** A zip with one enclosing folder gives `no <label>/<image> files found under work\byod`, which is moot until BIO-M3 is fixed, and the message gives no layout hint. A cancelled upload raises a bare `StopIteration`.
- **Consequence:** a field biologist with a small set, the likely BYOD user, satisfies the documented contract and fails four sections later, after the zero-shot run, with a message about "4 records" when they uploaded 12. A second attempt in the same session can silently evaluate the wrong data while labelling it as the new file.
- **Evidence:** direct execution, P6 C2, C4, C4b, C4c, C3, C5 and C7, and P5 B3 and B9.
- **Recommended correction:** derive the minimum from the split arithmetic and state it before upload (or check `len(train) >= MIN_RECORDS` and per-class training coverage in Section 4, naming the split and the required count per class). Clear `work/byod` before extracting. Reject extra or missing prompt keys with one message. Accept `MPO` as JPEG (read the first frame), or name the conversion. Accept one enclosing folder. Turn an empty upload into "no file was uploaded; run the cell again and choose a .zip". Do this in `samples.py`, `pipeline.decode_image` and the template, then regenerate.
- **Acceptance check:** (a) 4 × 3 is rejected **in Section 4** with a message stating the per-class minimum, and the documented minimum equals the smallest set that runs cells 15→31. (b) A second upload's `data_source` and classes are both upload B's. (c) A prompts dict with an extra key is rejected in Section 4. (d) The MPO probe either loads or names the conversion. (e) A cancelled upload prints an actionable message.
- **Spec:** DAT12, DAT19, VAL1, VAL6, UX10, REL12.

### BIO-M5 — Major: declared `GUIDED`, but most of the guided layer and any structured learner activity are absent; infrastructure is not labelled

- **Cell/section:** whole notebook; generator `tools/build_notebook.py` (`render`) and `tools/notebook_template.py`.
- **Observed issue:** no intended-learner statement, **How to use this notebook**, roadmap, Input → Model → Output contract, glossary, prediction prompt, interpretation checkpoint, worked answer, troubleshooting section or conclusion template. The objectives are mostly procedural ("install the pinned runtime; inspect the carried … modules; stage and digest-verify"). The four carried module cells (884 + 8,557 + 162 + 341 lines; `sample_data.py` is base64) sit between Section 1 and Section 3. They are not titled **Infrastructure**, not collapsed (`cellView: form` count 0) and not marked as safe to skip. The present elements: two "Look for" notes (Sections 1 and 4), clear stage prose with consequences, and three optional experiments listed in the final cell.
- **Consequence:** a self-paced learner who is new to zero-shot classification has to work out on their own what to notice, why the baselines matter and how to read the delta. Nothing checks understanding. The first screenfuls after the install are thousands of lines of code and base64 that look like prerequisite reading.
- **Evidence:** source inspection; P0 marker counts (`How to use` 0, `Glossary` 0, `Predict` 0, `Check your` 0, `Troubleshoot` 0, `Infrastructure` 0).
- **Recommended correction:** add the guided layer in the template, following the spec's 2.2 reference notebook (§25.13): audience and prerequisites, how-to-use, a roadmap, the Input → Model → Output contract, a short glossary (zero-shot, logit scale, macro-F1, AUROC, embedding), a prediction before zero-shot and before Section 10, "What to notice" after each principal stage, collapsible "Check your reasoning" answers, one **Predict → Change one thing → Run → Observe → Explain** activity (the prompt-vocabulary or learning-rate comparison fits; see BIO-M2), troubleshooting (download, memory, BYOD layout) and an evidence-based conclusion template. Title the carried cells `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL14 maps to a named cell. The four carried cells are titled Infrastructure and collapsed. At least one activity asks for a prediction before a result and gives a worked answer after it.
- **Spec:** GDL1–GDL15, UX5, UX8, UX9 (SHOULD); UX1 (MUST: objectives correspond to executed code; see BIO-M2 for the adaptation objective).

### BIO-m1 — Minor (fails a §20 MUST): the zip extraction guard is a string-prefix check and admits a sibling-directory escape

- **Cell/section:** Section 4 (cell 15); `tools/notebook_template.py:152`.
- **Observed issue:** `if not str(target).startswith(str(byod_root.resolve()))` accepts `work/byod_evil/…`, because that path starts with the string `…/work/byod`. A member named `../byod_evil/escaped.txt` was written to `work/byod_evil/escaped.txt`, outside the upload root. There is no expanded-size or member-count limit.
- **Consequence:** low in practice (the escape is limited to sibling names of `work/byod`), but the guard does not do what the notebook and README claim ("extracted with a path-traversal guard").
- **Evidence:** direct execution (P5 B5: `escaped_file_exists: true`).
- **Recommended correction:** use `target.is_relative_to(root)` (or `os.path.commonpath`), reject symlink entries, and cap the total expanded bytes and the member count.
- **Acceptance check:** the P5 B5 zip is rejected with a message naming the member, and nothing is written outside `work/byod`.
- **Spec:** §20 (MUST: extracted paths resolve within the intended root; SHOULD: size limits).

### BIO-m2 — Minor: under BYOD, user photographs are copied into `outputs/` under a "sample" name, colliding camera file names overwrite each other, and location metadata is kept

- **Cell/section:** Section 4 (cell 15) calls `write_dataset_csv(records, 'outputs/bioclip2_biodiversity_sample_dataset.csv')` unconditionally; carried `samples.write_dataset_csv`.
- **Observed issue:** with BYOD, the "sample dataset" CSV holds the user's 28 records, and their image bytes are copied unmodified into `outputs/images/`. Files are named by base name only, so camera names repeated across label folders (`IMG_0001.jpg` …) overwrite each other: 7 files for 28 records. The CSV then fails to reload (`record[7] … duplicates the image of 'chipping_sparrow-0'`). Bytes are copied unmodified, so any EXIF GPS block travels into `outputs/`, although the Prerequisites call location "the sensitive part of a biodiversity record". EXIF orientation is not applied when decoding (inferred effect on rotated phone photos; not measured).
- **Consequence:** a learner downloading `outputs/` gets a mislabelled, partly overwritten copy of their own data, possibly with coordinates.
- **Evidence:** direct execution (P6 C1 `sample_named_csv_holds_byod: 28`; C6 `outputs_images_IMG_files: 7`, reload error); source inspection for EXIF.
- **Recommended correction:** write the template only on the sample path, or write a `byod_dataset.csv` with label-qualified file names. Strip EXIF (re-encode) or warn before copying. Apply `ImageOps.exif_transpose` in `decode_image`.
- **Acceptance check:** under BYOD with repeated file names, `outputs/` holds one file per record and the CSV reloads. No file under `outputs/` named "sample" contains user data.
- **Spec:** DAT17, DAT18, OUT1.

### BIO-m3 — Minor: the optional experiments have no rerun instructions or controls, and sample-specific prose goes stale

- **Cell/section:** cell 24 (`TRAINABLE_BLOCKS` text), cell 26 ("On the sample it is zero"), cell 28 ("about 12 KB"), cell 32 (Optional experiments, Interpretation).
- **Observed issue:** "set `TRAINABLE_BLOCKS = 1` to watch the last tower block over-fit" does not say which cells to rerun. P4 found the right answer: Sections 9→12, after which the export is 14 tensors / 50.4 MB, not "about 12 KB". The prompt-template experiment requires editing code in Section 7; there is no template field. The Interpretation's "a perfect score" and Section 10's "On the sample it is zero" remain under BYOD or after an experiment. The `TRAINABLE_BLOCKS = 1` claim itself reproduces (0.9167, val 1.0 → 0.875).
- **Consequence:** a learner may rerun only cell 25 and read stale Section 10–12 outputs, or edit the wrong cell for the template.
- **Evidence:** direct execution (P4); source inspection.
- **Recommended correction:** add `PROMPT_TEMPLATE` as a form field used by Section 7 and by `adapt(template=…)`. State "rerun Sections 9–12" beside each experiment. Make result-dependent prose conditional or phrase it as "on the default sample run".
- **Acceptance check:** each optional experiment names its field and its rerun range. Following them reproduces the P4 numbers, and no printed or prose statement contradicts the run.
- **Spec:** GDL10, UX7, EXE1.

### BIO-m4 — Minor: runtime figures name no environment

- **Cell/section:** cell 0 ("On CPU the whole path takes about two minutes of model time after the download"), cell 1 ("about 0.3 s per image").
- **Observed issue:** neither figure names hardware. The recorded runs are 49.2 s (local CPU pre-flight) and 265.3 s (Kaggle T4 including the download). This review measured 42.9–46.9 s on a 24-thread CPU without the download.
- **Consequence:** small; a Colab CPU learner cannot tell whether a slow run is normal.
- **Evidence:** documented execution; direct execution (P1).
- **Recommended correction:** state measured times with their environment, label the rest as estimates, and give the download time separately.
- **Acceptance check:** every runtime figure in the notebook names its environment or is labelled an estimate.
- **Spec:** UX12 (MUST).

### BIO-m5 — Minor: doubled braces in the learner-facing data contract

- **Cell/section:** Prerequisites (cell 1); `tools/notebook_template.py:110`.
- **Observed issue:** "records are `{{id, image_bytes, label}}`". The template's format-string escape reaches the learner.
- **Acceptance check:** the rendered Prerequisites read `{id, image_bytes, label}`.
- **Spec:** DAT12.

### BIO-m6 — Minor: the BYOD interface is Colab-only, and the class-name mapping is not a form field

- **Cell/section:** Section 4 (cell 15).
- **Observed issue:** BYOD always calls `google.colab.files.upload()`, with no location field, so it cannot run in Jupyter (a stated supported runtime) or under an executor. `BYOD_CLASS_PROMPTS = {}` is a code literal, not a form field.
- **Acceptance check:** a `BYOD_PATH` field, when set, bypasses the upload dialog and `google.colab`. `BYOD_CLASS_PROMPTS` is settable as a form field (for example a JSON string).
- **Spec:** EXE1, EXE2, EXE5 (SHOULD); DAT16.

### Suggestions

- **BIO-S1:** update the declared `notebook_spec` from 2.0 to 2.2 in metadata, the opening cell and `NOTEBOOK_SOURCE` (`tools/build_notebook.py:31`), together with the validator assertion.
- **BIO-S2:** assert reload parity over the full score vectors of the whole test split, not only the label and top score of six images.
- **BIO-S3:** print a confusion table in Section 10. The sample has no errors, but a BYOD learner needs one, and it makes "explain an error" an available activity.
- **BIO-S4:** state the memory envelope. By Section 11 the base model, the adapted tower copy and the reloaded model (with its own tower copy) are all resident. This was not measured here.

## 5. Readiness

**Needs revision.**

- Open Majors: BIO-M1 (one-pass Run all), BIO-M2 (adaptation does not adapt), BIO-M3 (folder BYOD broken), BIO-M4 (BYOD contract and recovery), BIO-M5 (guided layer).
- Unresolved applicable `MUST`s: RUN1, RUN10, ENV6, REL2 (BIO-M1); DAT10, DAT14, REL12 (BIO-M3); DAT12, DAT19 (BIO-M4); §20 extraction root (BIO-m1); UX12 (BIO-m4).
- Remaining gates after the fixes: a one-pass hosted `Run all` record for the new blob; a recorded BYOD run with a positive folder upload and at least one clear rejection (REL12); and the maintainer's promotion decision. Current registry status is `Candidate`, which is consistent with this verdict.

## 6. Verified versus inferred

- **Verified by direct execution (CPU, real weights, exact pins, cells verbatim):** the default path (15/15, numbers equal to the record); the head displacement and the no-learning results (P2, P3); the `TRAINABLE_BLOCKS = 1` rerun (P4); the folder-layout BYOD failure (P5); the CSV-layout positive run and every BYOD negative listed (P6); the zip-guard escape (P5 B5).
- **Verified by documented evidence:** the restart in the Kaggle qualification run (record only; the executed notebook was not located).
- **Inferred, not verified:** that Colab triggers the restart guard; that MPO and EXIF-rotated photos are common among learners' uploads; memory headroom on a Colab CPU runtime; learner understanding (no learner observation).
- **Most likely to be wrong:** BIO-M2's framing as a defect of the default configuration rather than a deliberate "do no harm" choice. The notebook says head-only is what 28 images support. A reader could argue that a head pinned at zero-shot is acceptable. It is Major because the notebook teaches a measured comparison and a BYOD promise ("the value of adaptation shows" on non-saturated data) that the default settings cannot produce, as the neutral-names probe shows (0.25 → 0.25 at the default, 0.25 → 0.667 at lr 1e-2).
