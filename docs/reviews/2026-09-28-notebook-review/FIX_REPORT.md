# Fix report — Philippine Biodiversity Field-Survey Capstone, revision 0.2.0-candidate

Date: 2026-09-28. Scope: the findings in `DIMER_Philippine_Biodiversity_Capstone_Notebook_Review.md`
(this folder), which reviewed PR head `771551b`, notebook blob `355f4e3f`. All changes are in the
generator and its sources: `tools/build_biodiversity_capstone.py`, `tools/biodiversity_capstone.py`
and `tools/biodiversity_core.py`. The notebook is regenerated, never hand-edited. The frozen data
manifest, exclusions, model manifest and dependency lock are unchanged byte for byte.

## Evidence boundary

- **Done:** source changes; 22 new regression tests (all fail on `771551b`, all pass here); the
  full suite (99 passed; 94 passed and 5 skipped with PyTorch hidden, as in CI); ruff;
  release-asset validation; generator parity; and an exact materialisation of the regenerated
  carrier, with every hash verified and `capstone` / `biodiversity_core` importing. Logs are in
  `test_logs.md`.
- **Synthetic only:** every stage test uses synthetic unit vectors or stub model objects. The
  learning-rate selection (BC-M1) has not been exercised on BioCLIP embeddings; SigLIP's
  64-token padding (BC-M2) has been checked against a stub processor, not the pinned tokenizer.
- **Not done:** no real-model CPU pre-flight. The process that ran the tests could not reach
  Hugging Face or the iNaturalist bucket, so no weights, tokenizer or photograph was loaded.
  No hosted run exists. Readiness stays **Verification pending**, and the status stays
  **Candidate**.

## Per finding

| ID | Verified against source | Fix | Test(s) |
|---|---|---|---|
| BC-M1 | Confirmed: `CONFIG.learning_rate` 1e-4, 80 AdamW steps, head = logit scale × text, and epoch 0 eligible. `pipeline.adapt` has the same default. | `CONFIG.learning_rates = [1e-4, 1e-3, 1e-2]`. `fit_head` fits each rate from the same seed and start and selects (rate, epoch) by minimum validation cross-entropy (ties: earlier epoch, then smaller rate). It returns a selection record. `head_change` reports weight, bias and logit displacement and the prediction changes against zero-shot on train+validation. `selected_policy.json` gains `selected_learning_rate` and `head_equals_zero_shot`. `head_training.json` is displayed in §7. The loss plot shows every rate. The §7, §8 and §12 prose explains that epoch 0 is zero-shot. | 4 tests; the synthetic case moves the head, with its selected rate above 1e-4 and its selected epoch above 0 |
| BC-M2 | Confirmed: capitalised names, no `max_length`. | `SIGLIP_TEMPLATE` plus lowercased label text; `processor(..., padding="max_length", max_length=64)`; the stage refuses input IDs that are not N × 64; `siglip_preprocessing.json` is exported and shown; §5 prose. | stub-processor test |
| BC-M3 | Confirmed by port (probe P3). | `core.bootstrap(..., stratify=True)` resamples within each reference species. `evaluate` uses it and drops the `observer_cluster_*` columns, adding `bootstrap_method`, `test_records` and `test_observers`. The §8 prose explains the artefact. | a perfect classifier gives [1, 1], and the unstratified form still gives 0.75 (documents the artefact); an exported-metrics test |
| BC-m1 | Confirmed. | Bootstrap timings (uv, environment, install, session start) go to `outputs/bootstrap.json`. `prepare` writes `download_dataset.json`. `report` writes `runtime_summary.csv` (bootstrap, downloads, per-stage seconds and peak RAM/VRAM, total) and adds `timing`, `notebook_revision` and `wall_clock_seconds_since_session_start` to `run_summary.json`. §3 and §11 display them. | 2 tests |
| BC-m2 | Confirmed. | `verify_archive` reopens the ZIP and checks CRCs, member digests, safe paths, excluded suffixes and prefixes, that the only safetensors member is the adapter, and the size. It recomputes headline metrics from the archived predictions. `verification.json` names the record, and §11 displays `archive_verification.json`. | writes a figure into the ZIP and asserts refusal |
| BC-m3 | Confirmed (a design risk; the GPU behaviour itself is not verified). | The repeat check re-embeds the first batch of 8 exactly and records the cross-batch-size difference descriptively. The reload probe (first two test records) is embedded as one batch in `bioclip-zero-shot`. `evaluate` derives `reload_expected.npy` from those features, so the fresh-process reload repeats the same computation. | a stub pipe with batch-dependent noise, which fails on `771551b` |
| BC-m4 | Confirmed. | `activity_thresholds.json` holds the adjacent distinct validation candidates, and code-26 uses them. Thresholds display to 10 significant digits, and the activity table shows accepted counts. The display bound allows the `nextafter(1)` candidate. | 1 test |
| BC-m5 | Confirmed. | `error_panel_selection` names the four categories and falls back to the next candidate. Annotations start with `[category]`, and `error_panel.csv` reports empty categories. | 2 tests |
| BC-m6 | Confirmed (test: 4 photographers, one contributing 6 of 12; validation: 5, one contributing 6). | `observer_roles.csv`; prose computed from the manifest in §3; the conclusion template asks for the number of test photographers. | 2 tests |
| BC-m7 | Confirmed. | `dataset_audit.csv` (dates, dimensions, licences, refusal checks, the finer-taxon record, crop limitations, the post-release count); dashed centre-crop boxes on the contact sheet; §3 prose. | same tests as BC-m6 |
| BC-m8 | Partly correct. The subspecies rule *is* recorded as a resolved specification detail in `docs/biodiversity-capstone.md`, so the review's "not recorded as a spec deviation" was inaccurate. The notebook disclosure and the three §16 statements were missing. | The subspecies record is disclosed in §3, generated from the manifest. The statements on location exposure, not reconstructing coordinates, and not counting individuals are added to §3 and §12. | 1 test |
| BC-m9 | Confirmed. | `probe_predictions.csv` lists every predeclared system for each probe; `accepted` applies to the head only. | 1 test |
| BC-m10 | The source pattern is confirmed; Colab behaviour itself was not tested. | `FileLink` is replaced by Files-panel paths (`show_path`), with an optional `DOWNLOAD_RESULTS` form in §11. | 1 test |
| BC-m11 | Confirmed. | `UNSCORED_SYSTEMS = {"majority"}`: its top-2 is exported empty and shown as undefined, and the report's parity check accepts the undefined value. | 1 test |
| BC-m12 | Confirmed. | Risk rows are sorted; the canonical marker is drawn separately from the sweep; the Okabe-Ito palette replaces `tab10`. | 1 test |
| BC-m13 | Confirmed. | The nine objectives are listed in §1, and `head_training.json` is shown in §7. | 2 tests |
| BC-m14 | Partly correct. The 12 GiB disk requirement and the ≥ 6 BYOD groups were already documented (resolved details 4 and 6). | Details 7 (folder layout and label map) and 8 (learning-rate grid) added. | documentation |
| BC-m15 | Confirmed. | The BYOD cell hard-links the verified snapshots (copying as a fallback; stages re-hash them). One enclosing top-level folder is accepted. Missing `labels.csv` columns and an empty layout raise actionable errors. | 1 test |
| BC-S1 | Suggestion. | Implemented as a descriptive audit row: photographs observed after 2025-05. | covered by the BC-m6/m7 test |
| BC-S2 | Suggestion. | Not implemented: the blank-control byte digest belongs to the frozen manifest. | — |
| BC-S3 | Suggestion. | Network hosts listed in §2. | — |
| BC-S4 | Suggestion. | Not implemented: the badge must target `main` once merged. | — |

The notebook revision is `0.2.0-candidate` (`metadata.dimer.notebook_revision`, the §0 header
and `run_summary.json`). The cell count (36) and cell ids are unchanged.

## Pre-release gates

1. A fresh Colab T4 Run all of the exact head commit with defaults.
2. That run must show `siglip_preprocessing.json` with 64 tokens, the selected rate, epoch and
   `head_equals_zero_shot`, the runtime table, and passing reload and archive verification.
3. The threshold activity run at `lower` and `higher`, followed by §11, and one BYOD run.
4. The maintainer's attribution review.

## Addendum — first hosted run (0.2.0-candidate) and revision 0.2.1-candidate

The maintainer's Colab T4 run of `5131aa2` matched its source exactly. It stopped in `prepare`:
the isolated environment inherited Colab's `MPLBACKEND=module://matplotlib_inline.backend_inline`,
and `import matplotlib` failed. Revision 0.2.1-candidate forces the `Agg` backend for stage
processes, drops the kernel's `PYTHONPATH`, `PYTHONHOME` and `PYTHONSTARTUP`, and makes the
bootstrap print its progress and the GPU it verified. The run is recorded in
`../../release-verification.md`; the tests are in `tests/test_capstone_hosted_run_fixes.py`.

## Addendum — second hosted run (0.2.1-candidate) and revision 0.2.2-candidate

The run of `ebaed9e` matched its source exactly. The plotting-backend fix held, and all 60
iNaturalist photographs passed their checks on Colab. `prepare` then refused the synthetic blank
control, because its PNG bytes differed from the manifest digest: this was the review's
suggestion BC-S2. Revision 0.2.2-candidate verifies synthetic probes by decoded pixels
(`core.verify_cached`), so BC-S2 is now implemented. An offline stage-chain check (synthetic
photos, stub encoders) ran all nine stages, the activity and the report; see
`offline-stage-chain-0.2.2/`.
