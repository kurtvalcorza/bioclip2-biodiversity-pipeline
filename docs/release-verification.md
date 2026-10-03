# Release verification

`tutorials/bioclip2_biodiversity_colab.ipynb` (`E2E` / `GUIDED`, **standalone** carrier) is a **release candidate**
until the exact notebook revision has executed top-to-bottom, in one pass, in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are
necessary checks but are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the
durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- the tutorial registry in `tutorials/README.md`; `metadata.dimer` declares the `E2E` profile, spec `2.2`, a §3.3
  pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `sample_data.py`, `metrics.py`, `samples.py`,
  `head_selection.py`), each equal to its source after the generator's documented rewrites; the inline `MANIFEST`
  equal to the committed snapshot manifest and the inline `PINS` equal to the `pyproject.toml` runtime pins; the
  notebook byte-identical (on LF) to `tools/build_notebook.py` output for its recorded revision; `NOTEBOOK_SOURCE`
  recorded in exports;
- the isolated runtime (BIO-M1): exactly two `# dimer: kernel cell` cells — the install cell, which verifies the
  pinned `uv` wheel by size and SHA-256, creates a `--managed-python` 3.12.12 environment, refuses a non-Linux-x86_64
  platform and installs the carried hash lock (`tutorials/requirements-colab.lock.txt`, digest-checked) with
  `--require-hashes --only-binary :all:`, and the router that sends every later cell to one persistent worker in that
  environment; the generator checks that the lock pins every `pyproject.toml` pin and hashes every entry;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `BioClip2Pipeline.from_pretrained(weights_dir=...)`, `validate_dataset`, `split_dataset`,
  `minimum_records_per_class`, `write_dataset_csv`, `sample_provenance`, `validate_inputs`, `pipe.embed_images` with
  its same-batch and cross-batch assertions, `pipe.zero_shot_evaluate` with `PROMPT_TEMPLATE`, the open-set, blank and
  noise probes, `majority_baseline`, `color_baseline`, `select_head` over `LEARNING_RATES` with the selection and the
  head displacement printed, `pipe.adapt` for the optional unfrozen-block experiment, `pipe.evaluate` on both splits
  with the zero-shot and majority deltas and the agreement with zero-shot, `pipe.classify`, `pipe.save_artifact`,
  `BioClip2Pipeline.from_artifact` and the reload-parity assertion, and the Section 13 `fit_head` activity), the
  BYOD guards (fresh `work/byod`, the extraction root check and symlink refusal, the upload guard, exact prompt keys,
  the per-class minimum, `outputs/byod/`), the seven expected `outputs/` paths, the learner-facing statements (scores
  are not calibrated probabilities, closed-set zero-shot always answers, validation chooses and the test split is
  never looked at, embeddings are representations, no remote code, split by observation, site or photographer, CC0
  provenance, the BYOD minimum, Linux x86_64 only), the guided layer (who this is for, input → model → output, how
  to use, roadmap, predictions, what to notice, worked answers, the Section 13 activity, troubleshooting, glossary,
  conclusion, infrastructure labels), the absence of the stale learner text the review removed, and the gated-off
  BYOD default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the
  primary path, a mutable `revision='main'`, direct `open_clip` / `huggingface_hub` / `safetensors` / training use
  **outside the carried module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(` without
  `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, `tools/build_biodiversity_capstone.py
--check` and the offline unit suite (injected backends and temporary manifests, no weights and no model library —
only Pillow, because decoding is the image validation; tests that need `torch` skip; the embedded sample's dataset
digest is pinned). `tests/test_bioclip2_biodiversity_colab_review_fixes.py` executes the notebook's Section 4 verbatim
against the BYOD cases of the 2026-10-02 review. These are source/provenance and unit checks. They are **not**
execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab Linux x86_64 CPU or GPU runtime | The runtime the tutorial is written for; a clean one-pass top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh Linux x86_64 CPU or GPU container; the committed notebook executed verbatim with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence when it runs in one pass |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pins pre-installed and the two kernel cells skipped (`DIMER_NOTEBOOK_CI_PREINSTALLED=1`) | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new Linux x86_64 CPU or CUDA runtime (Colab, or a fresh-container executor
   above) with **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the
   working-directory snapshot `weights/bioclip-2/`; expect the isolated-environment install and a 1.71 GB download;
3. choose **Run all once** without editing implementation cells or restarting (form parameters at their defaults:
   `USE_BYOD = False`, `BYOD_PATH = ''`, `BYOD_CLASS_PROMPTS = '{}'`, `VAL_FRACTION = 0.2`, `TEST_FRACTION = 0.25`,
   `SEED = 42`, `PROMPT_TEMPLATE = 'a photo of {}.'`, `EPOCHS = 20`, `BATCH_SIZE = 8`, `TRAINABLE_BLOCKS = 0`,
   `BLOCK_EPOCHS = 4`, `BLOCK_LEARNING_RATE = 1e-4`, `ACTIVITY_NAMES = 'neutral'`); a run that needs a restart is not a
   one-pass `Run all` and is not promotion evidence;
4. verify that Section 1 reports the isolated Python 3.12.12, that the runtime cell reports
   `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in `metadata.dimer.generated_from`, and that
   the imported core package versions equal the inline `PINS` (= `pyproject.toml`): `torch==2.14.0`,
   `torchvision==0.29.0`, `open_clip_torch==3.3.0`, `timm==1.0.29`, `ftfy==6.3.1`, `regex==2026.9.10`,
   `huggingface-hub==1.32.0`, `safetensors==0.8.0`, `pillow==12.3.0`, `numpy==2.5.3`;
5. verify every default-path stage completes:
   - the five carried module cells execute with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(..., allow_download=True)`
     reporting the 3 entries fetched from `imageomics/bioclip-2` at the immutable revision, and `verify_snapshot`
     reporting 3 verified files before the model loads;
   - the dataset manifest printed with 48 records, classes `['chipping_sparrow', 'dark_eyed_junco', 'song_sparrow',
     'white_throated_sparrow']`, 12 each, 224×224, the ceilings, the digest `f8fc68cf251e0365…`, the provenance
     summary, the splits 28 / 8 / 12 with the per-class minimum 5, and the contact sheet;
   - three input-validation refusals (garbage bytes, BMP, 16×16) and one acceptance;
   - `pipe.embed_images` reporting 768-dimensional unit vectors, the same batch twice identical and a single-image
     versus in-batch difference below `1e-5`, and writing `outputs/bioclip2_biodiversity_embeddings.csv`;
   - `pipe.zero_shot_evaluate` on the test split (on the sample: accuracy 1.0, n = 12), the common-name variant, the
     open-set probe naming the species, and a confident label on the blank and noise images;
   - both trivial baselines on the test split (majority 0.25 / 0.1; colour 0.25 / 0.1909 in the recorded runs);
   - `select_head` printing one row per learning rate, the selected rate and epoch, `head_equals_zero_shot` and
     `head_change` (local CPU check of this version: 1e-2 at epoch 6, head moved 1.96 %, 0 predictions changed);
   - `pipe.evaluate` reporting validation and test metrics, the agreement with zero-shot, and writing
     `outputs/bioclip2_biodiversity_evaluation_report.json` with the three baselines, the deltas and the adaptation
     record;
   - `pipe.classify` on six held-out test images writing `outputs/bioclip2_biodiversity_predictions.csv`;
   - `pipe.save_artifact` writing `outputs/bioclip2_biodiversity_adapter/{adapter.safetensors,manifest.json}` (2
     tensors, 12,448 bytes), and `BioClip2Pipeline.from_artifact` reloading it with identical labels and a maximum
     absolute score difference below `1e-5`;
   - `outputs/bioclip2_biodiversity_result.json` written with `NOTEBOOK_SOURCE`, the model identity, the dataset
     manifest, the class prompts and template, the evaluation report, the predictions, the artifact manifest and the
     runtime versions;
   - the Section 13 activity printing a run-history row (local CPU check: neutral names 0.33 → 0.92 on the test
     split);
6. exercise the other journeys: a BYOD positive run (a zip of `<label>/<image>` folders, at least 5 photographs per
   class) through Section 13 and at least one clear BYOD rejection (REL12), and the `TRAINABLE_BLOCKS = 1` experiment
   with **Run after** from Section 9;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, open_clip, device), whether the model
   cache and the weights directory were clean, the outcome, produced outputs and the observed metrics (as
   observations, not a benchmark) in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `bioclip2_biodiversity_colab.ipynb` | `a43fdf5` / `60e7655f` | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `bioclip2_biodiversity_colab.ipynb` | `22f2854` / `60e7655f` | 2026-09-18 | Kaggle fresh GPU container (`gcr.io/kaggle-gpu-images/python@sha256:37c64f7…`, CPython 3.12.13, Tesla T4), strict serial executor v1 | All 15 code cells completed, but only after a **manual interpreter restart** that followed the in-kernel install. **Not a one-pass `Run all` and not promotion evidence** (review finding BIO-M1, 2026-10-02; this row previously read "supported clean-runtime qualification evidence"). Exact commit and fetched Git blob verified; empty Hub cache and no pre-staged snapshot. |

No one-pass hosted run of the current (isolated-runtime) notebook is recorded yet.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/bioclip2_biodiversity_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/bioclip2_biodiversity_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | `a43fdf5` / `60e7655f` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `open_clip 3.3.0`) | Default sample path (validate → split → reject probes → embed + reproducibility → zero-shot + probes → baselines → adapt → evaluate → classify → export → reload); weights pre-staged, so `stage_missing_files` fetched 0 of 3 entries and `verify_snapshot` verified all 3 | 49.2 s | **PASSED** — 15/15 code cells; zero-shot 12/12; head-only adaptation 3,076 params in 15.9 s; test accuracy/macro-F1/AUROC 1.0 (n=12) against majority 0.25/0.1 and colour 0.25/0.1909; delta vs zero-shot 0.0; 6/6 held-out; reload parity 0.0. Pre-flight; hosted clean-runtime run still required. The head was later shown not to move at its single rate of 1e-4 (BIO-M2) |
| 2026-09-18 | `22f2854` / `60e7655f` | Kaggle fresh GPU container, CPython 3.12.13, Tesla T4, `torch 2.14.0+cu130`, CUDA 13.0, `open_clip 3.3.0`; exact fetched blob verified; empty Hub cache | Default standalone path from an empty snapshot (install → **manual restart** → fetch and digest-verify → validate → split → reject probes → embed + reproducibility → zero-shot + probes → baselines → adapt → evaluate → classify → export → reload) | 265.3 s | Completed 15/15 code cells **after one manual restart**: not a one-pass `Run all`, not promotion evidence (BIO-M1). All 3 snapshot entries verified (1.71 GB staged); zero-shot 12/12; majority 0.25/0.1; colour 0.25/0.1909; head-only adaptation of 3,076 params in 3.18 s; test accuracy/macro-F1/AUROC 1.0 (n=12), delta vs zero-shot 0.0; 6/6 held-out; adapter reload parity 0.0 |
| 2026-10-03 | review-fix head of PR #10 (see below) | **Local check only** (Windows, CPython 3.12.12, `torch 2.11.0+cpu` — not the pinned 2.14.0 — `open_clip 3.3.0`, CPU float32, real weights pre-staged by hard link; the two kernel cells skipped with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`; learner cells executed verbatim from the notebook JSON; fake `google.colab` upload) | Default path; `TRAINABLE_BLOCKS = 1` re-run from Section 9; BYOD folder zip 4 × 7 (camera file names) re-run from Section 4; the BYOD contract matrix in Section 4 | 71 s (default learner cells) | Default 17/17 learner cells: zero-shot 12/12; majority 0.25 / 0.1; colour 0.25 / 0.1909; `select_head` chose 1e-2 at epoch 6 (val cross-entropy 0.0116 vs 0.0203 at epoch 0), head moved 1.96 %, 0 of 36 predictions changed; test accuracy / macro-F1 / AUROC 1.0, 12/12 equal to zero-shot, delta 0.0; 12,448-byte adapter, reload parity 0.0; Section 13 neutral names 0.33 → 0.92. `TRAINABLE_BLOCKS = 1`: test 0.9167 / 0.9143, val 1.0 → 0.875, 14 tensors, 50.4 MB, parity 0.0. BYOD 4 × 7 zip: Sections 4–13 ran, splits 16 / 4 / 8, selection kept epoch 0 (head equals zero-shot), test 1.0, parity 0.0, 28 files in `outputs/byod/images/`, the BYOD CSV reloads. BYOD refusals in Section 4: 4 × 3, 4 × 4, 2 × 6 (per-class minimum named), extra prompt keys, MPO, `../` escape (nothing written), symlink member, cancelled upload, no Colab and no path; accepted: 4 × 5, 2 × 7, one enclosing folder, a second upload (its own classes), `BYOD_PATH` folder without Colab. **Not hosted evidence** |

## Review fixes — `bioclip2_biodiversity_colab.ipynb` (review 2026-10-02, PR #10)

The review (`docs/reviews/2026-10-02-notebook-review/bioclip2_biodiversity_colab_Review.md`) found 5 Major and 6
Minor findings. All are fixed in the generator, the template and the package; the notebook is regenerated. Tests:
`tests/test_bioclip2_biodiversity_colab_review_fixes.py`.

| Finding | Fix | Evidence |
|---|---|---|
| BIO-M1 restart after the in-kernel install; two-pass run recorded as a `Run all` PASS | `tools/build_notebook.py` /2.1 with the fleet uv isolated runtime (pinned uv 0.12.15 wheel by size and SHA-256, managed CPython 3.12.12, hash lock `tutorials/requirements-colab.lock.txt` compiled from `pyproject.toml`, `--require-hashes --only-binary :all:`, one persistent worker); records above corrected; Linux x86_64 only | Validator and tests; the Linux install path was not executed locally — a one-pass hosted run is still required |
| BIO-M2 the head could not move at 1e-4 | `head_selection.py` (`select_head`, `fit_head`, `head_change`): the capstone's BC-M1 procedure — grid 1e-4 / 1e-3 / 1e-2, 20 epochs, validation cross-entropy selects (rate, epoch), epoch 0 eligible, displacement and agreement reported; lessons that relied on the immobile head rewritten. `pipeline.py` is unchanged because the capstone carries it byte for byte | Local real-weights check above; synthetic `fit_head` test (torch) |
| BIO-M3 folder BYOD joined the relative root twice | `load_byod_dataset` stores paths relative to the label root | Relative-directory test (fails on `2def984`); local BYOD 4 × 7 run |
| BIO-M4 stated BYOD limits differed from enforced ones; stale second upload; extra prompt keys; MPO; enclosing folder; cancelled upload | `split_dataset` refuses a class that leaves < 3 training records and names `minimum_records_per_class`; `work/byod` emptied per upload; exact prompt keys; MPO refused with the conversion; one enclosing folder accepted; upload guards | Section 4 executed verbatim in tests and in the local matrix above |
| BIO-M5 guided layer absent; infrastructure unlabelled | Audience, input → model → output, how to use, roadmap, observable objectives, predictions, what to notice, worked answers, Section 13 activity, troubleshooting, glossary, conclusion; Sections 1–3 labelled, carried code collapsed | Validator guided-layer markers |
| BIO-m1 string-prefix zip guard | Root check with `is_relative_to`, symlink refusal, 20,000-member / 2 GB cap, all before writing | Tests and local matrix |
| BIO-m2 user photos in a "sample" CSV, overwritten camera names, EXIF | BYOD data to `outputs/byod/` (replaced per upload), collision-safe names, EXIF GPS warning | Tests; local BYOD run (28 files, CSV reloads). EXIF orientation is not applied (`pipeline.py` unchanged) |
| BIO-m3 experiments without rerun ranges; stale sample prose | Each experiment names its field and **Run after** range; `PROMPT_TEMPLATE` field; result-dependent prose phrased as the default sample run | Validator stale-text list |
| BIO-m4 timings without an environment | Every figure names its environment or is labelled an estimate | Source inspection |
| BIO-m5 doubled braces | Single braces | Validator |
| BIO-m6 Colab-only BYOD; prompts not a field | `BYOD_PATH` field (zip, folder or table) without `google.colab`; `BYOD_CLASS_PROMPTS` JSON string field | Tests |

Suggestions BIO-S1..S4 are not addressed as such; S1 (spec 2.2) came with the generator update.


## Philippine biodiversity capstone — review fixes, hosted-run fixes and hosted default-path run, revisions 0.2.0- to 0.2.2-candidate (2026-09-28)

Scope: `tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`, a separate notebook
with its own evidence; nothing in this section changes the original tutorial's record above.
The review of PR head `771551b` (notebook blob `355f4e3f`) was a source inspection of an
unexecuted notebook. It is archived with its probes and a per-finding fix report in
`reviews/2026-09-28-notebook-review/`.

| Finding | Severity | Fix | Regression test (`tests/test_capstone_review_fixes.py`) |
|---|---|---|---|
| BC-M1 head effectively equals zero-shot (lr 1e-4 × 80 steps on scale-100 weights) | Major | Predeclared grid 1e-4/1e-3/1e-2; validation selects rate and epoch; displacement and zero-shot agreement exported and shown | `test_head_learning_rate_grid_is_predeclared`, `test_head_can_leave_zero_shot_when_validation_supports_it`, `test_adapt_head_reports_selection_and_distance_from_zero_shot`, `test_notebook_explains_rate_selection_and_shows_head_record` |
| BC-M2 SigLIP 2 text not lowercased, not padded to 64 tokens | Major | Documented convention applied and asserted; `siglip_preprocessing.json` | `test_siglip_stage_uses_documented_lowercase_64_token_text` |
| BC-M3 macro-F1 bootstrap counts an absent species as F1 = 0 | Major | Species-stratified record bootstrap; observer-cluster interval removed | `test_perfect_classifier_has_degenerate_macro_f1_interval`, `test_exported_intervals_are_species_stratified` |
| BC-m1 runtime/resources hidden, total understated | Minor | Bootstrap timings, dataset download, `runtime_summary.csv` displayed | `test_runtime_summary_reports_bootstrap_downloads_and_stages`, `test_notebook_records_bootstrap_timing_and_renders_runtime` |
| BC-m2 archive checks missing from the record | Minor | `verify_archive` (CRC, digests, paths, exclusions, metric recomputation) shown in §11 | `test_archive_verification_is_real_and_displayed` |
| BC-m3 parity across different batch compositions | Minor | Same-batch repeat; reload probe embedded as one batch; cross-batch difference recorded | `test_repeat_and_reload_probe_use_matching_batches`, `test_reload_expectation_comes_from_the_matched_batch` |
| BC-m4 threshold activity offsets ±0.1, 4-decimal display | Minor | Adjacent validation candidates; 10-significant-digit thresholds; accepted counts | `test_activity_uses_adjacent_validation_candidates` |
| BC-m5 error-panel categories unlabelled, dropped silently | Minor | Named categories, fallback selection, `error_panel.csv` | `test_error_panel_names_categories_and_falls_back`, `test_error_panel_annotations_and_csv` |
| BC-m6 observer concentration undisclosed | Minor | `observer_roles.csv`; manifest-computed prose | `test_observer_concentration_and_audit_from_frozen_manifest`, `test_learner_text_discloses_concentration_taxonomy_and_location_limits` |
| BC-m7 audit display incomplete; crop not shown | Minor | `dataset_audit.csv`, crop boxes on the contact sheet, crop limitations surfaced | same two tests as BC-m6 |
| BC-m8 §16 statements; subspecies record undisclosed in the notebook | Minor | Location, reconstruction and counting statements; subspecies disclosure | `test_learner_text_discloses_concentration_taxonomy_and_location_limits` |
| BC-m9 probes shown for the head only | Minor | All eight systems listed per probe | `test_probes_listed_for_every_system` |
| BC-m10 `FileLink` downloads | Minor | Files-panel paths; optional `DOWNLOAD_RESULTS` | `test_no_filelink_objectives_listed_and_revision_consistent` |
| BC-m11 majority top-2 is a tie-break artefact | Minor | Reported as undefined; parity check updated | `test_majority_top2_is_undefined_and_parity_still_holds` |
| BC-m12 risk plot doubles back; `tab10` palette | Minor | Sorted rows, separate canonical marker, Okabe-Ito palette | `test_risk_rows_sorted_and_palette_colour_blind_safe` |
| BC-m13 objectives not listed; head config not shown | Minor | Nine objectives in §1; `head_training.json` shown | `test_no_filelink_objectives_listed_and_revision_consistent`, `test_notebook_explains_rate_selection_and_shows_head_record` |
| BC-m14 specification drift | Minor | Disk and BYOD-group deviations were already recorded in `biodiversity-capstone.md`; folder-map and learning-rate deviations added | documentation only |
| BC-m15 BYOD re-download, enclosing folder, bad CSV | Minor | Snapshot hard links; enclosing folder accepted; actionable CSV/layout errors | `test_byod_accepts_one_enclosing_folder_and_explains_bad_csv` |

Suggestions: BC-S1 (post-release subset) is implemented as a descriptive audit row, and BC-S3
(network hosts) in §2. BC-S2 (verify the synthetic blank by decoded pixels) and BC-S4 (branch
Colab badge) are not implemented: the blank's byte digest is part of the frozen manifest, and
the badge must point at `main` once merged.

Offline evidence for this revision (2026-09-28, Windows, CPython 3.12.14, CPU torch 2.13.0,
numpy 2.5.3, Pillow 12.3.0): 99 tests pass; with PyTorch hidden, as in CI, 94 pass and 5 skip;
the 22 new tests all fail on `771551b`. Ruff, `validate_release_assets.py` and both generator
parity checks pass. No model weights, SigLIP tokenizer or iNaturalist photograph was loaded,
so none of the fixes has been observed with real models. The learning-rate grid in
particular has been exercised only on synthetic features.

### Maintainer-supplied Colab execution of revision 0.2.0-candidate — 2026-09-28

| Item | Value |
|---|---|
| File | `execution-evidence/2026-09-28/DIMER_Philippine_Biodiversity_Field_Survey_Capstone_0.2.0-candidate.ipynb`, copied byte for byte |
| SHA-256 | `3189f3e009a419ca06c629cd7786265b4f838de16e8082b9162e8d34b2d1d081` (454,876 bytes) |
| Source match | All 36 cells have the same ids and sources as notebook blob `9774379b` at `5131aa2`; no form parameter was changed |
| Runtime | Google Colab, Tesla T4 (printed by the preflight cell); notebook metadata `accelerator: GPU`, `gpuType: T4` |
| Executed | Code cells 1–4 in order (execution counts 1–4): preflight, carrier verification ("Embedded implementation, dependencies and manifests verified."), environment bootstrap, then §3 |
| Result | **FAILED in the first stage.** `run_stage('prepare')` exited 1 because `import matplotlib` raised `ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value`. The remaining 10 code cells did not run; no metrics were produced. |
| Cause | Colab exports `MPLBACKEND=module://matplotlib_inline.backend_inline`; the bootstrap copied the kernel environment into every stage process, and the isolated environment has no matplotlib-inline. The same code is present at `771551b`, so the defect predates the review fixes. |
| Positive evidence | The bootstrap cell completed without error, and its `check=True` probe (`assert torch.cuda.is_available()` inside the isolated environment) passed. The locked CUDA 13 torch wheels therefore installed and saw the T4 on this Colab image. Versions were not captured, because the probe's output went to the process stream, not the notebook. |
| Evidence boundary | Saved outputs were inspected; execution was not independently repeated. |
| Fixed in | revision 0.2.1-candidate: stage processes force `MPLBACKEND=Agg` (in the bootstrap environment and in `capstone.py` before `import matplotlib`) and drop `PYTHONPATH`, `PYTHONHOME` and `PYTHONSTARTUP`. The bootstrap now prints progress and the Python, torch, CUDA and GPU it verified. Tests: `tests/test_capstone_hosted_run_fixes.py` (4 tests; the two behavioural ones fail on `5131aa2`). |
| Open | Everything from §3 onward, i.e. the whole evidence list below. |

### Maintainer-supplied Colab execution of revision 0.2.1-candidate — 2026-09-28

| Item | Value |
|---|---|
| File | `execution-evidence/2026-09-28/DIMER_Philippine_Biodiversity_Field_Survey_Capstone_0.2.1-candidate.ipynb`, copied byte for byte |
| SHA-256 | `92fcfab72b5a609d84b2570a64af62d0b8e41d1532e26675a96e0eb0917d792e` (457,070 bytes) |
| Source match | All 36 cells have the same ids and sources as notebook blob `26f290f3` at `ebaed9e`; no form parameter was changed |
| Runtime | Google Colab, Tesla T4. The bootstrap printed: Python 3.12.12, torch 2.11.0+cu130, CUDA 13.0, GPU Tesla T4. Environment ready in 69 s (uv 1 s, venv 2 s, install 63 s). |
| Executed | Code cells 1–4 in order (execution counts 1–4) |
| Result | **FAILED in `prepare`** with `ValueError: Photo size/hash mismatch: blank-control`. The remaining 10 code cells did not run; no metrics were produced. |
| What passed first | The 0.2.1 environment fix worked: `capstone.py` imported, and `prepare` validated the manifest and exclusions. `validate_images` checks records before probes, and the blank control is the last probe, so all 60 iNaturalist photographs (56 canonical, 4 probes) were downloaded and passed the size, SHA-256, decode, dimension, GPS-EXIF, duplicate-pixel and dHash checks on Colab before the failure. |
| Cause | The synthetic blank control is generated at runtime by encoding a constant 224 × 224 PNG. Its manifest byte digest came from the freezing machine's encoder (reproducible there with zlib-ng at level 6), and the Pillow wheel on Colab encoded different bytes. The 2026-09-28 review had flagged this as suggestion BC-S2. |
| Evidence boundary | Saved outputs were inspected; execution was not independently repeated. |
| Fixed in | revision 0.2.2-candidate: synthetic probes are verified by the SHA-256 of their decoded pixels (`verify_cached`), and later stages re-verify the cache the same way. Photographs keep byte-exact verification. The frozen manifest is unchanged. Tests: three added to `tests/test_capstone_hosted_run_fixes.py`. |

Offline check of the full stage chain for 0.2.2-candidate (not hosted evidence): the carried
`capstone.py` ran all nine stages, the threshold activity at `lower` and `higher`, the
fresh-process reload and the report. Each stage ran in its own process with Colab's
`MPLBACKEND` set. The inputs were a synthetic photo cache (random images with the manifest's
dimensions; record digests rewritten) and deterministic stub encoders in place of BioCLIP and
SigLIP. The head fit, selection, evaluation, triage, reload parity (max error 0.0) and archive
verification ran unmodified. This is software evidence only.

### Maintainer-supplied Colab execution of revision 0.2.2-candidate — 2026-09-28

| Item | Value |
|---|---|
| File | `execution-evidence/2026-09-28/DIMER_Philippine_Biodiversity_Field_Survey_Capstone_0.2.2-candidate.ipynb`, copied byte for byte |
| SHA-256 | `bfecf50b8db3ae5c5158afdea64448cc5594c7b886ed63037865d0e6eee98398` (9,018,568 bytes) |
| Source match | All 36 cell ids are in order and all sources are identical to notebook blob `fd96c48d` at `f8fc8e3`, except one line Colab inserted at the top of the collapsed carrier cell `code-04`: the comment `# @title`, which adds no code. No form parameter was changed. |
| Runtime | Google Colab, Tesla T4. Isolated Python 3.12.12, torch 2.11.0+cu130, CUDA 13.0. Environment ready in 57 s (install 54 s). Wall clock from the first cell to the report: 287 s. |
| Executed | **All 14 code cells, execution counts 1–14 in order, no errors** (a single Run all). All nine stages completed with receipts. |
| Downloads | 60 photographs (23,918,875 bytes) in 97 s, including validation; SigLIP 2 1,539,456,760 bytes in 21 s; BioCLIP 2 1,710,538,981 bytes in 16 s. The synthetic blank was generated and passed decoded-pixel verification. |
| Peak VRAM / host RAM | SigLIP stage 1.58 GB / 2.49 GB; BioCLIP stage 1.83 GB / 4.27 GB; reload 1.75 GB / 4.25 GB; the head fit ran on CPU (0.70 GB host RAM). |
| Preprocessing | SigLIP labels lowercased, padded to 64 tokens (the stage asserts N × 64 input ids). |
| Head selection (validation only) | lr 0.01 and epoch 20 selected, both the largest values in the grid and the schedule. Validation cross-entropy 0.018 against 0.235 at epoch 0. `head_equals_zero_shot` False. Maximum logit change 3.48; 2 of 44 train and validation predictions differ from zero-shot. |
| Referral policy | The locked threshold of 0.6202 (the smallest validation margin) gives 100% validation coverage at 100% selective accuracy. On test: 12/12 accepted, 0 incorrect accepted. |
| Reload and archive | Fresh process (PID 2790 → 2839); max absolute error 0.0; identical labels and referrals. `results.zip`: 44 members, 65,619 bytes; CRC, digests, paths, exclusions and 8-system metric recomputation all passed. |
| Evidence boundary | Saved outputs were inspected; execution was not independently repeated. |

Held-out test results, 12 photographs from 4 photographers (species-stratified bootstrap, 2,000 draws):

| System | Accuracy | Macro-F1 | Macro-F1 95% interval | Top-2 | Paired difference vs BioCLIP scientific |
|---|---:|---:|---|---:|---:|
| majority | 0.250 | 0.100 | 0.100–0.100 | undefined | −0.750 |
| colour | 0.500 | 0.439 | 0.259–0.617 | 0.917 | −0.500 |
| SigLIP 2, common names | 0.500 | 0.364 | 0.338–0.402 | 0.667 | −0.500 |
| SigLIP 2, scientific names | 0.333 | 0.270 | 0.188–0.351 | 0.500 | −0.667 |
| BioCLIP 2, common names | 0.917 | 0.914 | 0.667–1.000 | 1.000 | −0.083 |
| BioCLIP 2, scientific names (canonical zero-shot) | 1.000 | 1.000 | 1.000–1.000 | 1.000 | 0.000 |
| BioCLIP 2 5-NN | 1.000 | 1.000 | 1.000–1.000 | 1.000 | 0.000 |
| **BioCLIP 2 head (primary)** | **1.000** | **1.000** | 1.000–1.000 | 1.000 | 0.000 |

Validation (12 photographs): SigLIP common 0.750 / 0.652, SigLIP scientific 0.500 / 0.375, BioCLIP
common 1.000 / 1.000, BioCLIP scientific 0.917 / 0.914, 5-NN 1.000 / 1.000, colour 0.417 / 0.411,
majority 0.250 / 0.100 (accuracy / macro-F1). Closed-set probes: the head named Philippine
Pied-Fantail for all five probes. It accepted the flowerpecker (margin 0.911), the butterfly (0.997)
and the snail (0.995), and referred the sparrow (0.195) and the blank control (0.526).

Journeys:

| Journey | Verdict |
|---|---|
| Clean default (Run all, defaults) | **PASS** on this runtime |
| Threshold activity at `lower` / `higher` | Not assessed in this run: only `canonical` was executed. The recorded neighbours are 0 (lower) and 0.999889 (higher). Assessed in the two follow-up runs below. |
| §11 re-run after the activity | Not assessed in this run; assessed below |
| BYOD (grouped or inference-only) | Not assessed in this run (default off) |

Observations for the maintainer, none of which blocks the default path:

1. Test performance is at ceiling for three BioCLIP systems (1.0 with degenerate intervals). This run therefore cannot distinguish the head from zero-shot on test; the head changed only 2 train and validation predictions.
2. The selected learning rate and epoch are the largest in the grid and the schedule, and validation loss was still falling at epoch 20.
3. One creator's name, 呂一起 (Lyu yi-chi), is rendered with missing glyphs in the photo panels (Matplotlib's DejaVu Sans lacks CJK). The romanised name and the licence remain legible, and `attribution.csv` is exact.
4. Each check above applies to this run only; the tiny sample means small changes can move these numbers.

Evidence still required before this capstone can leave Candidate:

1. a fresh Colab T4 **Run all** of the exact 0.2.2-candidate (or later) commit with default settings, recording the notebook
   blob, all 14 code cells, runtime, peak RAM/VRAM and the observed metrics;
2. confirmation in that run that the SigLIP text is 64 tokens, that the selected learning rate
   and epoch and `head_equals_zero_shot` are reported, and that reload parity passes;
3. the threshold activity run at `lower` and `higher` followed by the §11 cell, and at least
   one BYOD run (grouped or inference-only);
4. the maintainer's review of photo attribution and use.

Capstone executions so far: the 0.2.0- and 0.2.1-candidate runs failed in `prepare`, and the 0.2.2-candidate run passed the clean default path. Items 1 and 2 of the list above are met for notebook blob `fd96c48d`. Items 3 and 4 remained open after that run; see the follow-up runs below.

### Maintainer-supplied Colab executions of 0.2.2-candidate at `THRESHOLD_VIEW = lower` and `higher` — 2026-09-28

| Item | `lower` run | `higher` run |
|---|---|---|
| File | `execution-evidence/2026-09-28/DIMER_Philippine_Biodiversity_Field_Survey_Capstone_0.2.2-candidate_threshold-lower.ipynb`, byte for byte | `…_0.2.2-candidate_threshold-higher.ipynb`, byte for byte |
| SHA-256 | `3f8ac6285d6de1dd3601a5ecedfefc8fd573cc32bfa2e52b1e6d966a2ae2a8cb` (9,014,662 bytes) | `bcd545e838d910983bc92f3245c7e2f3729e8b09c837d2a469d2fae016bace29` (10,356,135 bytes) |
| Source match | All 36 cell ids and sources identical to blob `fd96c48d` except Colab's `# @title` line in `code-04` and the single form value in `code-26` | same |
| Form change | `THRESHOLD_VIEW = 'lower'` | `THRESHOLD_VIEW = 'higher'` |
| Executed | All 14 code cells, execution counts 1–14, no errors; §11 (`code-32`) ran after the activity | same |
| Environment | Fresh install: ready after uv 1 s, venv 2 s, install 59 s | Warm runtime: the existing isolated environment was reused (venv 0 s, install 8 s) and verified, so a repeated Run all in a used runtime also passes |
| Activity | Display threshold 0: 12/12 accepted, selective accuracy 1.000, 0 incorrect accepted (same as the locked 0.6202) | Display threshold 0.999889479: 7/12 accepted, selective accuracy 1.000, 5 referred, 0 incorrect accepted |
| Activity is display-only | `display only` True; locked threshold reported as 0.6202107991 in both | same |
| Unchanged results | Every model output is identical to the canonical run: after normalising the run directory and timings, the outputs of cells `code-07` to `code-25` and `code-29` match the canonical run exactly; adapter SHA-256 `3cd707f4…46dd` in all three runs | same |
| §11 after the activity | Fresh-process reload (PID 2036 → 2084) max error 0.0, identical labels and referrals; `results.zip` 44 members, 65,581 bytes, all checks including 8-system metric recomputation passed | Reload PID 3634 → 3683, max error 0.0; `results.zip` 44 members, 65,613 bytes, all checks passed |
| Evidence boundary | Saved outputs were inspected; execution was not independently repeated | same |

The archive sizes differ from the canonical run's 65,619 bytes only because `activity.json` records
the display view. In the `higher` run the activity figure shows referred photographs, and Matplotlib
warned about the three missing CJK glyphs (observation 3 above).

Journeys after these runs:

| Journey | Verdict |
|---|---|
| Clean default (Run all, defaults) | **PASS** (canonical run) |
| Threshold activity at `lower` / `higher` | **PASS**: display-only, official metrics unchanged |
| §11 after the activity | **PASS** in both runs (as part of Run all, after `code-26`) |
| Repeated Run all in a used runtime | **PASS** (`higher` run reused the environment) |
| BYOD (grouped or inference-only) | Not assessed (optional, default off) |

Items 1–3 of the evidence list above are met for blob `fd96c48d`, except the BYOD part of item 3.
Open before the capstone can leave Candidate: one BYOD run and the maintainer's attribution
review (item 4).

### Notebook source layout change (2026-10-02)

The carrier cell `code-04` held all carried files on one 374,826-character line, which can make the Colab editor
unresponsive. The generator now writes each carried string as short concatenated string pieces (longest notebook
line 1,176 characters). Python joins the pieces into the same text, so every carried implementation, dependency and
manifest file is unchanged byte for byte. The only carried value that changes is `source.json`, whose
`generator_sha256` records the edited generator. The notebook blob changes from `fd96c48d` to `9d0097fb`. The hosted
runs above executed blob `fd96c48d`; blob `9d0097fb` was re-run on 2026-10-03 (below).

### Colab CLI execution of revision `5c316dc` (blob `9d0097fb`) — 2026-10-03

| Item | Record |
| --- | --- |
| File | [`execution-evidence/2026-10-03/DIMER_Philippine_Biodiversity_Field_Survey_Capstone_5c316dc_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_Philippine_Biodiversity_Field_Survey_Capstone_5c316dc_colab-cli-t4.ipynb), SHA-256 `e24c7137641d8cc559ba2f0ee172ec1543ff4f2b06f0bcf43089423774ca0b28`, byte-for-byte copy of the CLI's output notebook |
| Executor | Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session, driven by the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`). Every code cell ran in order in one kernel; this is not a browser Run all. The CLI does not record `execution_count`, so order is evidenced by its `Executing cell k/14` log |
| Source match | Downloaded from GitHub at `5c316dc`; git blob `9d0097fbe07e` checked before the session was created; every cell equals the PR head |
| Result | **PASSED**: 14/14 code cells, no error output; wall 236.6 s (notebook clock 226.3 s; stages 152.1 s) |
| Equivalence | Compared with the 2026-09-28 default run of blob `fd96c48d` (`0.2.2-candidate`): 12 of 14 code-cell outputs are identical after removing paths and timings; the other two differ only in timings, process ids and the results bundle size (65,612 vs 65,619 bytes). The carrier split changed no runtime behaviour |
| Boundary | Saved outputs were inspected. The threshold activity variants, BYOD and the Files-panel download were not exercised. Status remains **Candidate** |

## Current status

The notebook source passes all static checks, including generator parity (`--check` OK), and exact candidate commit
`22f2854` / notebook blob `60e7655f` completed the supported clean-runtime procedure in a fresh Kaggle Tesla T4
container. The run started with an empty Hub cache and snapshot, downloaded and digest-verified the full 1.71 GB
model snapshot, completed all 15 code cells, and preserved hashed outputs. The clean-runtime execution gate is
therefore satisfied for that exact candidate. The repository remains **Candidate** until the maintainer explicitly
approves promotion to `Release-grade`; a later code or notebook change requires qualification of the new exact revision.
