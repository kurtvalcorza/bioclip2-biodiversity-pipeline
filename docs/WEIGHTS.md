# Weight provenance, sample-data provenance and DIMER hosting

This repository pins **one** snapshot with its own `dimer-base-manifest.json`, and embeds one sample dataset whose provenance is recorded per image.

## BioCLIP 2 weights

- Upstream: `imageomics/bioclip-2`
- Immutable revision: `2957b322090f9cb17ae72c71981c7218a28d81e0`
- Weight format: SafeTensors (`open_clip_model.safetensors`, 1,710,517,724 bytes), in open_clip's checkpoint layout (`visual.*`, `transformer.*`, `token_embedding.*`, `logit_scale`, …)
- Upstream weight license: MIT (`license: mit` in the pinned upstream README front matter and in the Hub repository metadata)
- Local layout: `weights/bioclip-2/` holds the 3 manifest entries (`open_clip_config.json`, `open_clip_model.safetensors`, upstream `README.md`; 1,710,538,981 bytes total) with byte size and SHA-256 for each. `verify_snapshot()` in `src/bioclip2_biodiversity_pipeline/pipeline.py` checks all of them before any load and refuses a manifest that omits the config or the weights; `stage_missing_files(allow_download=True)` fetches only absent entries at the pinned revision. `.safetensors` files are git-ignored; the Git repository does not vendor the checkpoint.
- Cross-check: the manifest's `open_clip_model.safetensors` digest `b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef` equals the `oid sha256` of the Hub LFS pointer at the pinned revision (`https://huggingface.co/imageomics/bioclip-2/raw/2957b322090f9cb17ae72c71981c7218a28d81e0/open_clip_model.safetensors`).

## How the model is built (and why not from the registry)

`open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')` — the upstream quick-start — resolves files from the Hub at `main` and picks the architecture from the Hub config at load time. This pipeline instead reads the **pinned** `open_clip_config.json`, asserts its architecture fields (`embed_dim` 768, image size 224, 24 vision layers, context length 77, vocabulary 49,408) against the package constants before any model library is imported, constructs `open_clip.model.CLIP(embed_dim, vision_cfg, text_cfg, quick_gelu=False)` directly, and calls `load_state_dict(..., strict=True)` on the pinned safetensors — verified: 0 missing, 0 unexpected keys, 427,616,513 parameters. `quick_gelu` is absent from the pinned config and defaults to `False`, which is correct for the LAION-2B lineage (OpenAI's original CLIP weights would need `True`); `tests/test_pipeline.py` asserts the key stays absent. The image transform is built from the config's `preprocess_cfg` (bicubic shortest-side resize to 224, centre crop, CLIP mean/std).

## The tokenizer: bundled BPE, verified against the shipped `tokenizer.json`

The upstream repository ships HF tokenizer files (`tokenizer.json`, `tokenizer_config.json` declaring `CLIPTokenizer`, `vocab.json`, `merges.txt`, `special_tokens_map.json`). open_clip's `SimpleTokenizer` carries the same CLIP BPE vocabulary inside the package. The loader uses the bundled one, so the HF files are **not manifest entries**. Parity was verified on 2026-09-18 with the `tokenizers` library reading the upstream `tokenizer.json` at the pinned revision: for `a photo of Danaus plexippus.`, `a photo of Cardinalis cardinalis.`, a 7-rank taxonomic string and `a photo of the monarch butterfly`, the token ids were identical (after stripping open_clip's zero padding), e.g. `[49406, 320, 1125, 539, 2257, 7630, …]` for the first. This is a verified equivalence for the prompt style the pipeline uses, not a proof over the whole vocabulary; a DIMER profile that tokenizes with the HF files instead would be well-advised to re-run the check on its own prompts.

## Files deliberately not staged

The upstream repository at the pinned revision also carries `open_clip_pytorch_model.bin` (1,710,639,510 bytes), the five HF tokenizer files above (`tokenizer.json` 2,224,081 bytes, `vocab.json` 862,328, `merges.txt` 524,619, `special_tokens_map.json` 588, `tokenizer_config.json` 705) and `.gitattributes` (1,519 bytes). None is listed in the manifest and none is fetched or loaded: SafeTensors carries the same parameters without executing a pickle, and the tokenizer is bundled. A DIMER profile upload for this model should use `open_clip_model.safetensors` plus `open_clip_config.json`, and must not upload the `.bin` twin.

## Sample photographs

`src/bioclip2_biodiversity_pipeline/sample_data.py` embeds 48 JPEG images (627,446 bytes of image data; 949,467 bytes as a Python module) with a provenance record per image. They were fetched once on 2026-09-18 from the iNaturalist API (`GET https://api.inaturalist.org/v1/observations` with `taxon_name`, `photo_license=cc0`, `quality_grade=research`, ordered by votes), keeping for each of four species at most 12 research-grade observations whose community taxon is exactly the species, **one per observer**, and only the first photo of each observation whose own `license_code` is `cc0`. The `medium`-size rendition was centre-cropped to a square and resized to 224×224 (bicubic), saved as JPEG quality 85. Recorded per image: file, label, scientific and common name, observation id and URL, photo id, licence code, observer login, observation date. **Not recorded, deliberately:** `place_guess`, coordinates or any location field — the upstream authors identify location disclosure as the poaching risk, and a tutorial dataset has no need of it. Observer logins are recorded as courtesy attribution; CC0 requires none. The set was not curated for image quality or subject framing, and was not reviewed for bystanders. Its identity is the dataset digest `f8fc68cf251e03655e343fd4ab3f49556248350d0d77ba4261698602e4a33cc6`, pinned by `tests/test_adaptation.py`, so any edit to the module fails the suite.

## DIMER hosting

- MIT permits use, modification, redistribution and commercial use subject to preservation of the licence and copyright notice. DIMER may mirror the pinned snapshot in its model store under those terms; the weights would be redistributed unmodified.
- Loader trust boundary: no `trust_remote_code`, no registry lookup, no Hub access on the snapshot path. `open_clip_torch==3.3.0` is the runtime (with `timm`, `ftfy` and `regex` as its own dependencies); a DIMER runtime for this model is an open_clip runtime, not a transformers one.
- Serving shape: embeddings and zero-shot classification need both towers and no adapter; an adapted profile needs the image tower plus a 12 KB head (head-only) or the head plus any unfrozen blocks. Zero-shot scores are relative to the supplied label set — a hosted profile should surface that in its contract.
- Compute: on the recorded CPU, verification 1.0 s, load 6.6 s, about 0.3 s per image for embedding or zero-shot, head-only adaptation on cached features in seconds; unfreezing tower blocks is where the "GPU-heavy" label in the fleet inventory becomes real, and where the sample showed degradation with few images.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
