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
weights; the embedded sample's dataset digest is pinned). These are source/provenance and unit checks. They are
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
| `bioclip2_biodiversity_colab.ipynb` | __LOCAL_ROW__ | | | |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/bioclip2_biodiversity_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/bioclip2_biodiversity_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| __LOCAL_EXEC__ | | | | | |

## Current status

The notebook source is complete and passes all static checks, including the generator parity checks (`--check` OK).
The repository stays at **Candidate** until a Colab or fresh-container run of the exact release revision is recorded
above.
