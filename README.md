# BioCLIP 2 Biodiversity Pipeline

DIMER-oriented pipeline for **BioCLIP 2** (`imageomics/bioclip-2`, ViT-L/14), pinned to an immutable Hugging Face revision. The repository exposes zero-shot species classification from taxonomic names, L2-normalised organism image embeddings, a labelled-image-dataset contract with explicit ceilings, bounded fine-tuning of a species-classification head that starts from the zero-shot classifier, held-out classification metrics with three baselines, and a safetensors adapter artifact that is digest-verified before it is loaded.

## Upstream alignment

- Model: `imageomics/bioclip-2`
- Revision: `2957b322090f9cb17ae72c71981c7218a28d81e0`
- Upstream weight license: MIT
- Upstream task: zero-shot image classification of organisms (CLIP trained on TreeOfLife-200M); this repository uses both towers for zero-shot classification and embeddings, and the image tower for supervised species classification
- Runtime: `open_clip_torch` (the upstream `library_name`), **not** `transformers` — the model is built from the pinned `open_clip_config.json` and the weights are loaded strictly; no remote code
- Repository adaptation: **E2E** (bounded fine-tuning of a linear head, optionally with the last *n* image-tower blocks, with a portable safetensors adapter)
- Newer upstream checkpoint: the pinned README names `imageomics/bioclip-2.5-vith14` as `new_version`; it is a different architecture and is not exercised here

## Quick start

```python
from bioclip2_biodiversity_pipeline import (
    SAMPLE_CLASS_PROMPTS, BioClip2Pipeline, generate_sample_dataset, split_dataset,
)

pipe = BioClip2Pipeline.from_pretrained()          # verifies the snapshot under weights/ first
records = generate_sample_dataset()                # 48 embedded CC0 iNaturalist photos, 4 sparrow species
emb = pipe.embed_images([records[0]["image_bytes"]])
print(emb["dimension"], len(emb["embeddings"][0]))   # 768 768

zs = pipe.zero_shot([records[0]["image_bytes"]], SAMPLE_CLASS_PROMPTS)   # no training
print(zs["predictions"][0]["label"], round(zs["predictions"][0]["score"], 3))

splits = split_dataset(records)
pipe.adapt(splits["train"], splits["validation"], class_prompts=SAMPLE_CLASS_PROMPTS)  # head from zero-shot
print(pipe.evaluate(splits["test"])["accuracy"])
print(pipe.classify([records[0]["image_bytes"]])["predictions"][0]["label"])
```

`embed_images()`, `zero_shot()` and `classify()` take 1..32 images (`MAX_IMAGES_PER_CALL`) as JPEG/PNG/WEBP bytes, file paths or PIL images, with a shorter side of at least 32 px and a longer side of at most 8,192 px; images are resized so the shorter side is 224 px and centre-cropped by the CLIP transform, so a small subject in a wide frame may be cropped out. `zero_shot()` takes 2..64 labels — taxon names, or a `{label: taxon name}` mapping — and a prompt template (`"a photo of {}."` by default, BioCLIP's convention; scientific names are the recommended minimum). Validation is **decoding, not biology**: nothing checks that an image shows an organism or that a label names a real taxon. `classify()` requires a prior `adapt()` or `from_artifact()`. Datasets are `{id, image_bytes, label}` records: at least 8 rows and 3 per class, at most 5,000 rows and 50 classes, unique ids and unique image bytes. Because adaptation re-validates the training split, `split_dataset()` refuses a class that would leave fewer than 3 training rows: with the default 20 % / 25 % validation / test fractions that means at least 5 rows per class (7 with only two classes), which `minimum_records_per_class()` computes.

## What the sample showed, and the default that follows

Zero-shot from scientific names scored 48/48 on the sample. `adapt()` initialises its linear head from that zero-shot classifier (unit text embeddings × the logit scale of 100), and at its default single learning rate of 1e-4 four epochs move the head by about 0.02 %: the "adapted" head stays the zero-shot classifier whatever the data (review finding BIO-M2; the capstone's BC-M1). The tutorial therefore fine-tunes the head with `select_head()`: a predeclared grid of 1e-4, 1e-3 and 1e-2 for up to 20 epochs on features the frozen tower computes once, with the rate and epoch chosen by validation cross-entropy and epoch 0 (zero-shot itself) eligible. In a local CPU check it chose 1e-2 at epoch 6 on the sample, moved the head by about 2 %, changed no prediction and kept the test split at 12/12; with uninformative class names (`species 0` … `species 3`) the same procedure took test accuracy from 0.33 to 0.92. Unfreezing **one** of the 24 image-tower blocks with `adapt(trainable_blocks=1)` at 1e-4 for four epochs (12.6 million parameters, 28 training images) took the test split to 11/12. That run uses one fixed rate and no selection, so it is not a like-for-like comparison and does not rank the two methods; `adapt()` keeps `trainable_blocks=0` as its default and is unchanged, because the capstone carries `pipeline.py` byte for byte.

## Weights layout

```
weights/bioclip-2/   open_clip_config.json  open_clip_model.safetensors  README.md  dimer-base-manifest.json
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` (byte size + SHA-256 of every manifest entry; staging fetches only absent entries, only at the pinned revision, and only with `allow_download=True`), asserts the pinned config's architecture fields against the package constants, builds `open_clip.model.CLIP` from that config, and loads the safetensors with `strict=True`. The upstream `open_clip_pytorch_model.bin` is deliberately absent from the manifest and is never loaded; the upstream HF tokenizer files are not staged because the loader uses open_clip's bundled CLIP BPE (verified to produce identical token ids — see `docs/WEIGHTS.md`). `.safetensors` files are git-ignored; the Git repository does not vendor the checkpoint.

## Sample data

`src/bioclip2_biodiversity_pipeline/sample_data.py` embeds 48 photographs (12 each of Chipping Sparrow, Dark-eyed Junco, Song Sparrow and White-throated Sparrow) from research-grade iNaturalist observations whose **photo** licence code is CC0 1.0, centre-cropped to 224×224 JPEG, about 627 KB in total. Each carries its observation URL, photo id and observer login; no location was collected. The four species were chosen a priori so that colour is a weak cue (all four are streaked grey-brown birds) and the colour baseline sits at chance. The images are unfiltered beyond licence and quality grade, which is what field data looks like.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the trained tensors — with the default head-only setting, a 4×768 weight and a 4-entry bias, about 12 KB; plus any unfrozen tower blocks) and a `manifest.json` recording the artifact format, the exact base model id and revision, the class order, the tensor names, the file size and SHA-256, and the full adaptation configuration including how the head was initialised. `BioClip2Pipeline.from_artifact(dir)` re-verifies the base snapshot, then checks the artifact manifest, the base identity and every digest **before** deserialising, and refuses any tensor the classifier architecture does not have.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: injected backends and temporary manifests, never the weights. Pillow is used to decode the embedded sample, and the sample's dataset digest is pinned.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/bioclip2-biodiversity-pipeline/blob/main/tutorials/bioclip2_biodiversity_colab.ipynb)

`tutorials/bioclip2_biodiversity_colab.ipynb` is declared `E2E` / `GUIDED` and is **standalone** (DIMER Notebook Specification 2.2 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 5 package modules (`pipeline.py`, `sample_data.py`, `samples.py`, `metrics.py`, `head_selection.py`) verbatim in dependency order — the sample photographs travel inside it, which is why it is about 1 MB — together with the pinned model identity, the snapshot manifest and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable. Do not edit it by hand; change the package or the template and regenerate (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI). Its default path validates the 48 embedded photographs, splits them 28/8/12 stratified by class, stages and digest-verifies the 1.71 GB snapshot, extracts embeddings and asserts they are reproducible, classifies the test split zero-shot from scientific names with blank/noise/open-set probes, measures majority-class and colour baselines, fine-tunes the head from the zero-shot classifier with a validation-selected learning rate and epoch and reports how far it moved, evaluates accuracy/macro-F1/AUROC on the held-out split with the delta and the agreement against zero-shot, classifies six held-out images, exports the safetensors adapter, verifies reload parity, and ends with a change-one-thing activity on uninformative class names. Section 1 builds an isolated Python 3.12.12 environment with `uv` from the hash-locked `tutorials/requirements-colab.lock.txt`, so **Run all** needs no restart; the notebook therefore supports **Linux x86_64 runtimes only** (Google Colab, Kaggle, Linux Jupyter). BYOD (a zip of `<label>/<image>` folders, or CSV/JSON/JSONL of image paths, by upload or `BYOD_PATH`, with scientific names in the `BYOD_CLASS_PROMPTS` JSON field; at least 5 photographs per class with the default split) is optional and gated off by default. See `tutorials/README.md`.

The separate [Philippine Biodiversity Field Survey capstone](tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb)
adds a Philippine field-photo investigation with BioCLIP 2 and SigLIP 2, observer-disjoint evaluation,
baselines, head adaptation, a validation-selected referral policy, and verified artifact export/reload.
It is a standalone `E2E` / `WORKSHOP` notebook under DIMER Notebook Specification 2.2.
Its status is **Candidate: fresh Colab T4 execution pending**. The original tutorial qualification
below does not qualify this new capstone. See [capstone build notes](docs/biodiversity-capstone.md).

## Release status

**Candidate — hosted BYOD and experiment runs pending.** The 2026-09-18 Kaggle Tesla T4 run of commit `22f2854` / notebook blob `60e7655f` completed all 15 code cells only after a manual interpreter restart that followed the in-kernel install, so it is **not a one-pass `Run all` and not promotion evidence** (review finding BIO-M1). The current notebook installs into an isolated environment and needs no restart; commit `bcce1a8` / blob `5226299a` ran the default path in one pass with no restart and 0 errors on a fresh Colab Tesla T4 session on 2026-10-03 (BYOD and the `TRAINABLE_BLOCKS = 1` experiment not yet exercised on a hosted runtime). See `docs/release-verification.md`. `Release-grade` needs the remaining hosted journeys and an explicit maintainer decision.

## Licensing

- Upstream weights: MIT (`imageomics/bioclip-2`), staged unmodified from the pinned revision.
- Sample photographs: CC0 1.0 (per-photo licence codes from the iNaturalist API, recorded in `sample_data.py`).
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
