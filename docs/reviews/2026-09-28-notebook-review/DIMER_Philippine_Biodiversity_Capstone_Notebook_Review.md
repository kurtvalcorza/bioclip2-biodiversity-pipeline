# Notebook Review — Philippine Biodiversity Field-Survey Capstone (BioCLIP 2)

Review date: 2026-09-28 · Framework: Notebook Review Framework v1 · Requirements baseline: DIMER Notebook Spec 2.2 and `PHILIPPINE_BIODIVERSITY_FIELD_SURVEY_CAPSTONE_SPEC.md` (2026-09-27)

## 1. Review contract

| Item | Value |
|---|---|
| Notebook | `tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb` (uploaded copy) |
| File SHA-256 | `fa2be8830eb9a15de1f1e9946ea7761950776415329ec567caedbeb55c569894` (410,113 bytes) |
| Cells | 36 (22 markdown, 14 code). **No code cell has been executed; the file has no outputs.** |
| Declared identity | Spec 2.2 · profile E2E · mode WORKSHOP · standalone · release status Candidate |
| Generator | `build_biodiversity_capstone.py/1`, generator SHA-256 `96ab1814…cf93`, base revision `d19beaa93c9d` of `bioclip2-biodiversity-pipeline` |
| Carried implementation | 12 files embedded in cell `code-04`; every file matches `CARRIED_HASHES` and the notebook metadata digests (probe P1) |
| Lock | 65 packages, all hashed; torch 2.11.0 / torchvision 0.26.0 on CUDA 13 wheels, open-clip-torch 3.3.0, transformers 4.57.6 |
| Audience | Self-paced learners with basic Python and Colab; no ecology or ML background required |
| Supported runtime | Fresh Google Colab T4, Run all |
| Promised outcomes | The nine learning objectives in capstone spec §2.1, plus a verified results bundle and reloadable head |

### Evidence basis

| Basis | Used in this review |
|---|---|
| Source inspection | Every learner-facing cell, and the carried `capstone.py`, `biodiversity_core.py`, `pipeline.py`, both manifests and the lock |
| Direct execution | **Standard-library probes only (P1–P6)**. The review sandbox could not start a NumPy- or PyTorch-capable kernel, so no carried code was executed. P3 and P4 re-implement the carried logic in pure Python, and each script says so in its header. |
| Documented execution evidence | None. The uploaded notebook has no outputs, and no hosted run exists. |
| Learner observation | None |
| Not verified | Everything that needs the GPU run: model loading, downloads, all metrics, timings, reload parity, the ZIP, figures and the BYOD path |

Saved outputs were not inspected because none exist. This review makes no claim that the notebook runs.

## 2. Separate judgments

**Technical correctness — good design, with verification defects.** The integrity architecture is strong:
- carried-source digests and fully hashed dependencies;
- a strict snapshot manifest, and remote-code-free model construction;
- stage receipts that bind source, configuration and prerequisite products, and invalidate downstream stages;
- a reload in a new process ID (fresh PID);
- a safetensors-only adapter loader with tensor-scope checks;
- refusal probes that exercise the real validators.

The defects are in what the verification records contain and display (BC-m1, BC-m2), and in two parity checks whose tolerances may be too tight for the conditions they compare (BC-m3).

**Promise fulfilment — partly met.** The comparison, split, triage and closed-set structure is all present. Two central promises are weakened:
- the "fit a small head" stage cannot materially move away from zero-shot under its configured optimiser budget (BC-M1);
- the general-model comparator does not run under its documented text preprocessing (BC-M2).

**Scientific validity — the design is sound, but three results would mislead.** The locking order is correctly enforced in code: prompt, epoch and threshold are fixed before test. The following would mislead:
- the head-versus-zero-shot comparison (BC-M1);
- the SigLIP-versus-BioCLIP comparison (BC-M2);
- the confidence intervals on the primary endpoint (BC-M3).

Test-set observer concentration is also not disclosed (BC-m6).

**Learner experience — strong scaffolding.** Every section carries predict, what-to-notice and interpret prompts, and the limits of a closed-set model are explained carefully. The gaps:
- the threshold activity may produce views that cannot be told apart (BC-m4);
- the error panel does not name its categories (BC-m5);
- several audit and configuration facts that the prose points to are never displayed (BC-m7, BC-m13).

**Specification conformance — unmet applicable MUSTs.**

| Requirement | Finding |
|---|---|
| Capstone §7.2: documented upstream preprocessing | BC-M2 |
| §2 and §13.2: measured download and total runtime reporting | BC-m1 |
| §11 and §12.4: verification record contents | BC-m2 |
| §16: ethical-boundary statements | BC-m8 |
| §5.4: exact-species taxon rule (one subspecies-level record, not disclosed) | BC-m8 |
| §10, sections 3, 8 and 10: required content | BC-m5, BC-m7, BC-m9 |
| §18: acceptance items 14–16 | Open by design |

## 3. Findings

Severity follows the framework: Blocker, Major, Minor, Suggestion. Spec references are to the capstone spec unless marked otherwise.

### Major

#### BC-M1 — The "adapted head" is effectively the zero-shot classifier under the configured optimiser budget

**Where:**
- §7 (`code-19`, `md-18`);
- `capstone.py` `CONFIG` (line 39: `learning_rate` 1e-4, `weight_decay` 0.01, batch 8, 20 epochs);
- `fit_head` (lines 533 and 540).

**Observed:**
- The head is initialised as `logit_scale × text embeddings`, with logit scale ≈ 100 (pipeline default and checkpoint value). Epoch 0 is therefore numerically the BioCLIP scientific-name zero-shot classifier.
- Training runs 32 images in batches of 8 for 20 epochs, which is 80 AdamW steps at lr 1e-4.
- Adam moves each weight by at most about lr per step. The total displacement is therefore ≲ 8×10⁻³ per element, against weights of magnitude ≈ 100 × 0.036 ≈ 3.6 (about 0.2%).
- For unit-norm image embeddings, Cauchy–Schwarz bounds each class logit's change by √768 × 8×10⁻³ + 8×10⁻³ ≈ 0.23. Any pairwise logit gap therefore changes by ≲ 0.46.
- A prediction can only change where the zero-shot cosine gap between the top classes is below ≈ 0.005.
- Epoch 0 is also eligible for selection. If validation loss does not fall, the exported head *is* the zero-shot classifier.

**Consequence:** §7 and §8 teach adaptation. They include the prompt "adaptation can redistribute errors", the primary endpoint is the head's macro-F1, and the conclusion compares the head with zero-shot. Under this budget, "the head did not beat zero-shot" is largely determined by the step size before any data is seen. Learners would draw a conclusion about adaptation from an optimiser artefact. The notebook's line "A head that fails to beat zero-shot is still an informative result" is not true in this configuration. Objective 2 (distinguish zero-shot, 5-NN and a fitted head) is not exercised meaningfully.

**Evidence:**
- Source inspection and the analytic bound above.
- Probe P4 is a pure-Python re-implementation of `fit_head` on **synthetic** unit-norm embeddings in which zero-shot is wrong on 25% of training images, with a cosine gap of 0.02.
  - At lr 1e-4: maximum weight change 0.0073, maximum logit change 0.083, **0 prediction flips**, and validation cross-entropy 0.873 → 0.865 over 20 epochs.
  - Contrast run at lr 1e-2: logit change 7.65, 9 flips, training accuracy 0.75 → 1.00, validation accuracy 0.83 → 0.92.
- Real BioCLIP embeddings were not tested.

**Context:** capstone spec §7.3 asks for the repository's head-only semantics "where compatible". `pipeline.adapt` uses the same lr 1e-4 default. Its docstring records a saturated zero-shot classifier on the tutorial sample. That default suits a tutorial that demonstrates the mechanics of adaptation; it does not suit a capstone that asks learners to evaluate adaptation.

**Correction:** use one of these:
- **(a)** Predeclare a small learning-rate grid (for example 1e-4, 1e-3, 1e-2). Select the rate with epoch by minimum validation cross-entropy, before test, and record the grid in `adapter/manifest.json`.
- **(b)** Reparameterise the head as `W = logit_scale · (T + Δ)` with unit-scale Δ, so the learning rate is relative to the text prior.

In either case, export and display: parameter displacement ‖ΔW‖, maximum logit change, the selected epoch, and per-record agreement with BioCLIP scientific zero-shot. If epoch 0 wins, tell the learner explicitly that the exported head equals zero-shot. Amend capstone spec §7.3 to record the choice.

**Acceptance check:**
- A CPU test on fixture embeddings shows that the selected configuration can change predictions when the validation evidence supports it.
- The hosted run records the learning-rate selection, displacement and agreement count.
- The prose in §7 and §8 describes what happened.

#### BC-M2 — The SigLIP 2 comparator does not run under its documented text preprocessing

**Where:**
- §5 (`code-13`);
- `capstone.py` line 373: `prompts = [f"This is a photo of {c[field]}." ...]`;
- line 379: `processor(..., padding="max_length", ...)`.

**Observed:**
- Candidate names are passed with their original capitalisation: "Philippine Pied-Fantail", "Rhipidura nigritorquis" and so on. No `.lower()` is applied.
- `max_length=64` is not passed.
- The Transformers SigLIP 2 documentation says the model was trained on lowercased text, and says to pass `padding="max_length"` with `max_length=64`, because that is how the model was trained.
- SigLIP's text tower pools a fixed token position. The effective sequence length therefore changes the text embedding. Whether the pinned `tokenizer_config.json` fixes 64 is **not verified**.

**Consequence:**
- Objective 3 (biology-specific versus general model) rests on this comparison. The capstone spec requires each model's documented preprocessing (§7.2 MUST).
- A BioCLIP advantage may be partly a preprocessing artefact.
- The "Will scientific names help both models equally?" experiment is confounded, because the SigLIP scientific-name prompts carry a capitalised genus and the common names are title-case.

**Evidence:** source inspection; probe P5 (`siglip_prompts_lowercased: false`, `siglip_max_length_64_passed: false`); Transformers SigLIP 2 model documentation.

**Correction:** lowercase the SigLIP label text, keeping the documented template "This is a photo of {label}.", and pass `max_length=64`. Record the preprocessing in `model_manifest.json` or `run_summary.json`. State in §5 that each model uses its own documented convention.

**Acceptance check:**
- A unit test asserts the lowercased prompts and `max_length=64` in the SigLIP stage, for example by exec'ing `siglip_zero_shot` with a recording processor stub.
- The hosted run records a token length of 64.

#### BC-M3 — Bootstrap intervals for the primary endpoint are dominated by class-absence artefacts

**Where:**
- §8 (`code-22`, `md-21`);
- `biodiversity_core.metrics`, line 406: F1 = 0 when support and predictions are both zero, averaged over all classes;
- `bootstrap`, line 434;
- `capstone.evaluate`, lines 718 and 721.

**Observed:**
- Record resampling of 12 test photographs (3 per species) omits at least one species in **12.5%** of resamples (exact inclusion–exclusion: 0.1252).
- An absent species contributes F1 = 0 to the macro average, so the resample's macro-F1 falls to ≤ 0.75 even when every prediction is correct.
- The observer-cluster variant resamples only **4 test observers**, so there are just 256 equally likely draws.

**Evidence (probe P3):**
- A classifier that is correct on every test photograph, on the real test layout, gets a record-bootstrap 95% macro-F1 interval of **[0.75, 1.00]**, while its accuracy interval is [1.00, 1.00].
- The exact observer-cluster enumeration gives **[0.50, 1.00]**. Its distribution is 93.4% at 1.0, 5.9% at 0.5, and 0.4% each at 0.75 and 0.25.
- P3 is a pure-Python port. The carried code's NumPy generator would produce different individual draws, but the absence probability is a property of the design.

**Consequence:** the conclusion template asks learners to report "[macro-F1 with interval]". The lower bound mostly reflects how often a species goes missing from a resample, not classifier uncertainty. `md-21` describes the intervals as the uncertainty of this sample, which they do not measure.

**Correction:**
- Use a **stratified record bootstrap**: resample within each species, keeping per-species support at 3.
- Either drop the observer-cluster interval, or label it as not interpretable with 4 clusters and do not export it beside the primary interval.
- Add the unit test from capstone spec §12.2 ("bootstrap … null handling"): a perfect classifier yields a macro-F1 interval of [1, 1].

**Acceptance check:** that unit test passes. The `metrics.csv` columns and the `md-21` wording describe the stratified procedure.

### Minor

#### BC-m1 — Runtime and resource reporting is incomplete and mostly hidden (§2 and §13.2 MUST; `md-02` promise)

**Observed:**
- `show_record('run_summary.json')` renders only scalar keys. Of the eight top-level keys, only `status` and `elapsed_seconds` appear (P5); environment, stages, timings and RAM/VRAM are hidden.
- `elapsed_seconds` sums the receipts for the stages before `report` only (line 1026). It excludes the uv/Python/environment bootstrap, which is written to `ROOT/bootstrap.json` outside `outputs/` and never summarised, and it excludes the report stage.
- Dataset download bytes and seconds are not measured (P5: `dataset_download_timing_recorded: false`).
- Model download records exist as `download_*.json` but are not surfaced.

**Correction:** measure the dataset fetch in `prepare`. Fold `bootstrap.json` and the download records into `run_summary.json` with a true wall-clock total. Render a flat runtime table: download, per-stage, total, and peak RAM/VRAM per stage.

**Acceptance check:** the hosted run shows every §13.2 field in the notebook output.

#### BC-m2 — The verification record omits the archive checks (§11 and §12.4)

**Observed:**
- `verification.json` (line 1003) carries CSV, reload and dataset checks, but no ZIP CRC or member-hash result.
- `archive_verification.json` is written after the ZIP, excluded from it (line 1050), and never displayed (P5).
- "Excluded assets absent" is recorded as a literal `True`, justified by the suffix filter, rather than by checking the ZIP members.

**Correction:**
- Write the archive results into `verification.json`, or into a displayed sidecar.
- Assert that no member has an image or weight suffix, that no member path is under `cache/` or `weights/`, and that the size ceiling holds.
- Display the result in `code-32`.

#### BC-m3 — The parity checks compare different batch compositions at float32 noise-level tolerances (conditional blocker)

**Observed:**
- `reload_expected.npy` is the head scores computed from embeddings produced in batches of 8 (`bioclip_zero_shot`). The reload re-embeds the two probe images as one batch of 2 (line 908), then requires softmax all-close at atol 1e-6 and rtol 1e-5.
- The repeat-embedding check (line 414) compares a batch of 1 with a batch of 8 under the same tolerances.
- The logit scale is 100, so an embedding difference of ~1e-6 can move a non-saturated probability by ~1e-5.
- Whether a T4 produces batch-invariant ViT-L float32 outputs at this level is **not verified**. If it does not, Run all stops at §6 or §11.

**Correction:**
- Compute the expected probe values with the same batch composition as the reload, or compare the head on the stored embeddings (head parity) separately from embedding parity.
- Justify the tolerances from a measured maximum error.

**Acceptance check:** the hosted run passes and records the observed maximum absolute error.

#### BC-m4 — The threshold activity may not produce a visible change (§9, `code-26`)

**Observed:**
- The display thresholds are fixed offsets of ±0.1 from the locked threshold, clamped to [0, 1]. They are not taken from the observed margin distribution.
- With logit scale 100, head softmax margins are often close to 1. "higher" can clamp to 1.0 and refer nearly everything, and "lower" can match the canonical decisions.
- `show_table` prints four decimals, so thresholds such as 0.99991 and 0.99998 both show as `1.0000`.
- **Verification pending** on the real distribution.

**Correction:**
- Take the display values as the adjacent distinct candidate thresholds from the validation sweep (previous, locked, next).
- Show accepted and referred counts.
- Print thresholds with enough precision to tell them apart.

#### BC-m5 — Error-panel categories are unlabelled and can silently drop (§8; §10 item 8)

**Observed:**
- The `errors.png` titles carry truth, top-2, margin and decision, but not the category name (P5).
- "Incorrect high confidence" is the highest-margin incorrect case, whatever its margin.
- The de-duplication at line 756 takes only `candidate[:1]`. When the same record wins two categories, the second category disappears even though other candidates exist.
- `error_categories.json` is not displayed.

**Correction:** label each panel with its category. Fall back to the next candidate. Display the availability record ("category absent" rather than silence).

#### BC-m6 — Observer concentration in the held-out roles is not disclosed (§3 and §8; scientific interpretation)

**Observed (P2):**

| Role | Photographs | Observers | Concentration |
|---|---:|---:|---|
| Test | 12 | 4 | One observer contributes 6 of 12, across all four species |
| Validation | 12 | 5 | One observer contributes 6 of 12 |

`dataset_summary.csv` reports observers per species, 2–3 per role, which hides this cross-species concentration. The split is spec-conformant.

**Consequence:** learners are told that observer grouping strengthens evaluation. The effective held-out sample is four photographers, one of whom dominates.

**Correction:** display the per-role observer totals and the largest observer share, and reference them in `md-08`, `md-21` and the conclusion limits.

#### BC-m7 — The §3 audit display is incomplete, and crop geometry is not shown (§10 section 3)

**Observed:**
- Dates, dimensions and the coordinate-absence check (`precise_locations_absent`, nested in `prepare.json`) are not rendered.
- `md-08` asks learners to inspect crop geometry, but `contact()` shows full frames (line 116), and SigLIP's fixed-resolution processor resizes without cropping.
- The builder's recorded crop limitations are not surfaced: the fantail is hard to locate in the centre crop, and the outside-species sparrow is small in the crop.

**Correction:**
- Render a flat audit table: counts, observers, date range, dimension range, licence counts, and forbidden-field check results.
- Draw each model's visible region on the contact sheet, or add a crop view.
- Surface `visual_review.crop_limitations`.

#### BC-m8 — Required ethical-boundary statements are missing, and a taxonomy deviation is undisclosed (§16 MUST; §5.4)

**Observed (P5):** no markdown states that:
- endemic status does not justify exposing exact locations;
- the notebook must not reconstruct coordinates;
- the model does not count individuals.

Abundance and conservation status are covered in `md-35`.

Separately, one coleto training record (`inat-344854106-628387309`) is identified at subspecies level (*Sarcops calvus melanonotus*) and accepted as `verified_subspecies_ancestor`. §5.4 requires the taxon to resolve "exactly" to the target species. The inclusion is defensible, but it is neither disclosed in the notebook nor recorded as a spec deviation.

**Correction:** add the three statements to `md-06` or `md-35`. Disclose the subspecies record, or amend §5.4 to allow verified subspecies descendants.

#### BC-m9 — The closed-set probe section shows only the head (§10 section 10: "every model still returns a candidate label")

**Observed:** zero-shot and 5-NN scores for the probes are already computed (every stage scores all records, probes included), but `probe_predictions.csv` holds only `bioclip_head` (line 839).

**Correction:** add each predeclared score-producing system to the probe table, descriptively and unscored.

#### BC-m10 — Download links probably do not work in Colab (§11, completion)

**Observed:** `FileLink(str(path))` is used with absolute paths throughout, including for `results.zip`. In Colab, the reliable routes are the Files panel or `google.colab.files.download`. **Not verified.**

**Correction:** print the Files-panel path. Optionally offer a guarded `files.download` in a separate optional cell.

#### BC-m11 — The majority baseline reports a meaningless top-2 accuracy

**Observed:** majority scores are one-hot (line 158). Top-2 is therefore decided by the stable tie-break: the winner plus class index 0. The winner is `coleto`, on a lexical tie at 8 training images each. §8.2 limits top-2 to score-producing systems.

**Correction:** report top-2 as undefined for the majority baseline, and exclude it from the paired and top-2 displays.

#### BC-m12 — Figure defects (§14)

**Observed:**
- In `risk_coverage.png`, the test canonical row is appended after the sorted sweep (line 800), so the test line doubles back to the canonical point.
- The PCA uses `tab10` (line 511), which is not a colour-blind-safe palette; §14 requires one. The shapes by split are correct.

**Correction:** plot the sweep and the canonical marker separately, and use a colour-blind-safe qualitative palette.

#### BC-m13 — Learning objectives and configuration are referenced but not shown

**Observed:**
- The nine objectives in capstone spec §2.1 are never stated (P5: `learning_objectives_listed: false`), so learners have no completion criteria to check against.
- `md-18` says "Read the exported configuration for trainable tensors, parameter count, learning rate and weight decay", but `adapter/manifest.json` is neither displayed nor linked.

**Correction:** add a short objectives list to §1. Render the adapter training block in `code-19`.

#### BC-m14 — Specification drift that should be reconciled

| Item | Capstone spec | Notebook | Notes |
|---|---|---|---|
| Free disk | "approximately 5 GB" (§2) | Requires and enforces ≥ 12 GiB | No measured basis recorded, although §13.3 asks for one |
| BYOD grouped evaluation | ≥ 3 observer groups per class (§15) | ≥ 6 groups and ≥ 6/3/3 images | The notebook's rule follows from two observers per role. It is documented in `md-33` and is the better rule. |
| BYOD folder layout | Accepts a folder layout | Additionally requires a JSON label → scientific-name map | |

**Correction:** amend the spec, or record these as deliberate deviations.

#### BC-m15 — Friction on the BYOD reuse path (source inspection only; BYOD was not executed)

**Observed:**
- A grouped BYOD run re-downloads both model snapshots (≈ 3.2 GB) into a new root, because weights live under `ROOT/weights` (line 317).
- A folder ZIP with a top-level directory, the usual result of zipping a folder, has three path parts. It is rejected as "BYOD has no images", with no hint about the layout (line 148).
- A `labels.csv` missing a required column raises a bare `KeyError`.

**Correction:** reuse the verified snapshots through a digest-checked link or copy. Strip a single common top-level directory. Validate the CSV header with an actionable message.

### Suggestions

- **BC-S1 — Describe the post-release subset.** 11 of the 56 photographs were observed in 2026, after BioCLIP 2's 2025 release (P2). They cannot be in its training set. A descriptive "post-release subset" row can illustrate the overlap boundary without claiming to resolve it.
- **BC-S2 — Verify the synthetic blank by decoded pixels, not encoded bytes.** P6 reproduces the manifest digest (675 bytes) only with zlib-ng 1.3.1 at level 6 and per-row adaptive filtering. The canonical run therefore depends on the PNG encoder build in the pinned Pillow wheel.
- **BC-S3 — List every network host in `md-02`:** PyPI, the uv managed-Python download, Hugging Face and the iNaturalist open-data S3 bucket.
- **BC-S4 — Point the Colab badge at the branch under review.** It targets `main`, where the notebook does not yet exist; while it is a Candidate, provide a branch link for reviewers.

## 4. Promise-to-evidence tracing

| Promise or objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Observer-disjoint evaluation (objective 1) | Global observer roles; validator refuses leakage; refusal probe | `dataset_summary.csv`, refusal table | `md-08` explains why | Implemented. The effective held-out sample is not shown (BC-m6). |
| Zero-shot vs 5-NN vs fitted head (objective 2) | All three are implemented | Metrics tables | `md-17`, `md-20` | The head barely differs from zero-shot (BC-M1) |
| Biology-specific vs general model (objective 3) | SigLIP then BioCLIP, sequential processes | Validation tables | `md-14` | Unequal preprocessing (BC-M2) |
| Metrics and confusion matrix (objective 4) | Correct metric code; 8 confusion matrices | `metrics.csv`, figures | `md-23` | Intervals are misleading (BC-M3); top-2 for majority (BC-m11) |
| Closed-set behaviour (objective 5) | Probe set: 2 outside birds including *Dicaeum australe*, 2 non-birds, 1 blank | `probe_predictions.csv` | `md-30` | Head only (BC-m9) |
| Validation-only referral policy (objective 6) | `select_threshold` follows capstone spec §9.3; locked before test in code | `selected_policy.json`, risk–coverage | `md-18`, `md-24` | Correct. The activity may be uninformative (BC-m4). |
| Error inspection (objective 7) | Four diagnostic categories | `errors.png` | `md-23` | Categories unlabelled (BC-m5) |
| Export and reload parity (objective 8) | Digest- and scope-checked safetensors; new process ID | `verification_reload.json` | `md-31` | Tolerance risk (BC-m3) |
| Evidence-bounded conclusion (objective 9) | Template in `md-35` and `conclusion.md` | — | Template | Depends on BC-M1 and BC-M3 |
| Measured runtime and resources (§2) | Stage receipts | `run_summary.json` shows 2 fields | — | BC-m1 |

## 5. Journeys

- **First-time learner (source walk).**
  - Orientation, the Input → System → Output table, the predict prompts and the closed-set explanations are clear.
  - Barriers: objectives are not listed (BC-m13); the prose points to configuration and audit details that are never shown (BC-m7, BC-m13); the error panel is hard to read (BC-m5).
- **Clean default: not verified.** No execution evidence exists. Source-level risks:
  - BC-m3, the parity tolerances;
  - the CUDA 13 wheel set on the Colab T4 driver (not verified; an external report from March 2026 describes Colab T4 runtimes with CUDA 13 PyTorch);
  - the reachability of the 60 iNaturalist S3 assets.
- **Active learning.**
  - The threshold activity is correctly display-only, and exports use the locked threshold. Its informativeness is not verified (BC-m4).
  - The prompt-vocabulary comparison is confounded (BC-M2).
- **Reuse and recovery.**
  - Stage receipts give clear rerun semantics: changing a prerequisite invalidates downstream stages.
  - Fourteen refusal probes run on the real validators and are displayed: ten manifest mutations, observer leakage, changed bytes, changed digest and path traversal.
  - BYOD was not executed; its source-level friction is in BC-m15.

## 6. Readiness decision

**Needs revision.**

- Three Major findings affect the capstone's central comparisons and its primary-endpoint uncertainty: BC-M1, BC-M2 and BC-M3.
- The notebook has never been executed.
- After revision it remains **Verification pending** until capstone spec §18 items 14–16 are met:
  - a fresh Colab T4 Run all with recorded commit, blob, timing, RAM/VRAM and metrics;
  - the threshold activity and one BYOD run exercised;
  - the maintainer's attribution review.
- Release status stays **Candidate**.

## 7. Probe inventory (`DIMER_Philippine_Biodiversity_Capstone_Review_Probes.zip`)

All probes use Python ≥ 3.10 and the standard library only. Run them with `python run_probes.py <notebook.ipynb>`.

| Probe | What it establishes | Basis |
|---|---|---|
| P1 `p1_carrier_integrity.py` | Notebook digest, that nothing has been executed, carried-file hashes, manifest and exclusions digests, lock pins | Direct execution (stdlib) |
| P2 `p2_sample_audit.py` | Split counts, observer roles and concentration, duplicates, dHash distance (minimum 11), licences, taxon rank, hosts, bytes, dates, forbidden keys | Direct execution (stdlib) |
| P3 `p3_bootstrap_absent_class.py` | BC-M3: class-absence probability and the interval for a perfect classifier | Port of the carried logic |
| P4 `p4_head_displacement.py` | BC-M1: AdamW displacement under `CONFIG`, with an lr 1e-2 contrast | Port of the carried logic, synthetic data |
| P5 `p5_static_source_checks.py` | Line-level evidence for BC-M2 and BC-m1 to BC-m13 | Source inspection |
| P6 `p6_blank_control_digest.py` | BC-S2: the blank-control PNG digest depends on the encoder | Direct execution (stdlib) |
