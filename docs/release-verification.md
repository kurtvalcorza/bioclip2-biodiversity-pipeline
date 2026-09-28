# Release verification

`tutorials/bioclip2_biodiversity_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `sample_data.py`, `metrics.py`, `samples.py`), each equal to
  its source after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot
  manifest and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `BioClip2Pipeline.from_pretrained(weights_dir=...)`, `validate_dataset`, `split_dataset`, `write_dataset_csv`,
  `sample_provenance`, `validate_inputs`, `pipe.embed_images` with its same-batch and cross-batch assertions,
  `pipe.zero_shot_evaluate`, the open-set, blank and noise probes, `majority_baseline`, `color_baseline`,
  `pipe.adapt` with `class_prompts` and its explicit hyperparameters, `pipe.evaluate` on both the validation and
  the test split with the zero-shot and majority deltas, `pipe.classify`, `pipe.save_artifact`,
  `BioClip2Pipeline.from_artifact` and the reload-parity assertion), the seven expected `outputs/` paths, the
  learner-facing statements (scores are not calibrated probabilities, closed-set zero-shot always answers,
  validation is monitoring only, embeddings are representations, no remote code, split by observation, site or
  photographer, CC0 provenance) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any
  `git clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `open_clip` / `huggingface_hub` / `safetensors` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected backends and temporary manifests, no
weights and no model library — only Pillow, because decoding is the image validation; the embedded sample's dataset
digest is pinned). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/bioclip-2/` (the standalone path writes the manifest itself and stages every listed file, so
   the directory may not be seeded); expect a 1.71 GB download;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `VAL_FRACTION = 0.2`, `TEST_FRACTION = 0.25`, `SEED = 42`, `BYOD_CLASS_PROMPTS = {}`,
   `EPOCHS = 4`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 8`, `TRAINABLE_BLOCKS = 0`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `open_clip_torch==3.3.0`, `timm==1.0.29`,
   `ftfy==6.3.1`, `regex==2026.9.10`, `huggingface-hub==1.32.0`, `safetensors==0.8.0`, `pillow==12.3.0`,
   `numpy==2.5.3`;
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the four carried module cells execute (defining `BioClip2Pipeline`, `verify_snapshot`, `stage_missing_files`,
     `decode_image`, `image_digest`, `validate_inputs`, `SAMPLE_RECORDS`, `SAMPLE_IMAGES_B64`,
     `classification_metrics`, `majority_baseline`, `color_baseline`, `validate_dataset`, `split_dataset`,
     `generate_sample_dataset`, `sample_provenance`, `write_dataset_csv`, `load_byod_dataset`) with no import of the
     repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(..., allow_download=True)`
     reporting the 3 entries fetched from `imageomics/bioclip-2` at the immutable revision, and `verify_snapshot`
     reporting 3 verified files before the model loads;
   - the dataset manifest printed with 48 records, classes `['chipping_sparrow', 'dark_eyed_junco', 'song_sparrow',
     'white_throated_sparrow']`, 12 each, 224×224, the ceilings, the digest
     `f8fc68cf251e0365…`, the provenance summary (48 images, CC0-1.0, `location_data: not collected`), the splits
     28 / 8 / 12, and the contact sheet;
   - three input-validation refusals (garbage bytes, BMP, 16×16) and one acceptance;
   - `pipe.embed_images` reporting 768-dimensional unit vectors, the same batch twice identical, a single-image
     versus in-batch difference below `1e-5` (observed ≈1e-7), and writing `outputs/bioclip2_biodiversity_embeddings.csv`;
   - `pipe.zero_shot_evaluate` on the test split (on the sample: accuracy 1.0, n = 12), the common-name variant, the
     open-set probe naming the species, and a confident label on the blank and noise images;
   - both trivial baselines reported on the test split (majority accuracy 0.25 on the balanced split; the colour
     baseline near chance);
   - `pipe.adapt` reporting `head_initialisation: zero-shot text classifier`, 3,076 trainable of 303,969,284
     parameters, an epoch-0 validation entry and a four-epoch history;
   - `pipe.evaluate` reporting validation and test accuracy, macro-F1, AUROC and per-class rows, and writing
     `outputs/bioclip2_biodiversity_evaluation_report.json` with the three baselines and the deltas against
     zero-shot and majority;
   - `pipe.classify` on six held-out test images writing `outputs/bioclip2_biodiversity_predictions.csv` with
     per-class scores;
   - `pipe.save_artifact` writing `outputs/bioclip2_biodiversity_adapter/{adapter.safetensors,manifest.json}` (2
     tensors, about 12 KB), and `BioClip2Pipeline.from_artifact` reloading it with identical labels and a maximum
     absolute score difference below `1e-5` (the cell asserts both);
   - `outputs/bioclip2_biodiversity_result.json` written with `NOTEBOOK_SOURCE`, the model identity, revision and
     licence, `remote_code_executed: false`, the dataset manifest and sample provenance, the class prompts, the
     evaluation report, the predictions, the artifact manifest and the runtime versions;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, open_clip, device), the model
   identifier and immutable revision, whether the model cache and the weights directory were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `bioclip2_biodiversity_colab.ipynb` | `a43fdf5` / `60e7655f` | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `bioclip2_biodiversity_colab.ipynb` | `22f2854` / `60e7655f` | 2026-09-18 | Kaggle fresh GPU container (`gcr.io/kaggle-gpu-images/python@sha256:37c64f7…`, CPython 3.12.13, Tesla T4), strict serial executor v1 | **PASS — supported clean-runtime qualification evidence.** Exact commit and fetched Git blob verified; empty Hub cache and no pre-staged snapshot; one expected interpreter restart after dependency installation; all 15 code cells completed. Repository remains Candidate pending an explicit maintainer promotion decision. |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/bioclip2_biodiversity_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/bioclip2_biodiversity_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | `a43fdf5` / `60e7655f` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `open_clip 3.3.0`) | Default sample path (validate → split → reject probes → embed + reproducibility → zero-shot + probes → baselines → adapt → evaluate → classify → export → reload); weights pre-staged, so `stage_missing_files` fetched 0 of 3 entries and `verify_snapshot` verified all 3 | 49.2 s | **PASSED** — 15/15 code cells; zero-shot 12/12; head-only adaptation 3,076 params in 15.9 s; test accuracy/macro-F1/AUROC 1.0 (n=12) against majority 0.25/0.1 and colour 0.25/0.1909; delta vs zero-shot 0.0; 6/6 held-out; reload parity 0.0. Pre-flight; hosted clean-runtime run still required |
| 2026-09-18 | `22f2854` / `60e7655f` | Kaggle fresh GPU container, CPython 3.12.13, Tesla T4, `torch 2.14.0+cu130`, CUDA 13.0, `open_clip 3.3.0`; exact fetched blob verified; empty Hub cache | Default standalone path from an empty snapshot (install → expected restart → fetch and digest-verify → validate → split → reject probes → embed + reproducibility → zero-shot + probes → baselines → adapt → evaluate → classify → export → reload) | 265.3 s | **PASSED** — 15/15 code cells after one expected restart; all 3 snapshot entries verified (1.71 GB staged); zero-shot 12/12; majority 0.25/0.1; colour 0.25/0.1909; head-only adaptation of 3,076 params in 3.18 s; test accuracy/macro-F1/AUROC 1.0 (n=12), delta vs zero-shot 0.0; 6/6 held-out; adapter reload parity 0.0; preserved outputs include the evaluation, predictions, embeddings, sample data, adapter, result, contact sheet and embedded images with hashes recorded by the executor. This is qualification evidence, not a maintainer promotion decision. |


## Philippine biodiversity capstone — review fixes and hosted-run fixes, revisions 0.2.0- to 0.2.2-candidate (2026-09-28)

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

Evidence still required before this capstone can leave Candidate:

1. a fresh Colab T4 **Run all** of the exact 0.2.2-candidate (or later) commit with default settings, recording the notebook
   blob, all 14 code cells, runtime, peak RAM/VRAM and the observed metrics;
2. confirmation in that run that the SigLIP text is 64 tokens, that the selected learning rate
   and epoch and `head_equals_zero_shot` are reported, and that reload parity passes;
3. the threshold activity run at `lower` and `higher` followed by the §11 cell, and at least
   one BYOD run (grouped or inference-only);
4. the maintainer's review of photo attribution and use.

The capstone executions so far are the failed 0.2.0- and 0.2.1-candidate runs recorded above; the next run must use revision 0.2.2-candidate or later.

## Current status

The notebook source passes all static checks, including generator parity (`--check` OK), and exact candidate commit
`22f2854` / notebook blob `60e7655f` completed the supported clean-runtime procedure in a fresh Kaggle Tesla T4
container. The run started with an empty Hub cache and snapshot, downloaded and digest-verified the full 1.71 GB
model snapshot, completed all 15 code cells, and preserved hashed outputs. The clean-runtime execution gate is
therefore satisfied for that exact candidate. The repository remains **Candidate** until the maintainer explicitly
approves promotion to `Release-grade`; a later code or notebook change requires qualification of the new exact revision.
