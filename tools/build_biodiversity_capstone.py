"""Generate the standalone Philippine biodiversity scientific capstone."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def carried_files() -> dict[str, str]:
    files = {
        "bioclip2_biodiversity_pipeline/__init__.py": '"""Isolated capstone carrier for the unchanged pipeline module."""\n'
    }
    for name in ("pipeline.py", "metrics.py"):
        files["bioclip2_biodiversity_pipeline/" + name] = (
            ROOT / "src/bioclip2_biodiversity_pipeline" / name
        ).read_text(encoding="utf-8")
    for target, source in {
        "capstone.py": "tools/biodiversity_capstone.py",
        "biodiversity_core.py": "tools/biodiversity_core.py",
        "data_manifest.json": "tools/biodiversity_data.json",
        "model_manifest.json": "tools/biodiversity_models.json",
        "exclusions.json": "tools/biodiversity_exclusions.json",
        "requirements.txt": "tools/biodiversity-requirements.lock",
        "licenses/code.txt": "LICENSE",
    }.items():
        # Exclusions have a byte digest in the data manifest: preserve original newlines.
        files[target] = (
            (ROOT / source).read_bytes().decode("utf-8")
            if target == "exclusions.json"
            else (ROOT / source).read_text(encoding="utf-8")
        )
    for path in sorted((ROOT / "weights").glob("*/dimer-base-manifest.json")):
        files[path.relative_to(ROOT).as_posix()] = path.read_text(encoding="utf-8")
    return files


PREFLIGHT = r"""
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile

SESSION_START = time.perf_counter()
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('Use a fresh Google Colab Linux x86-64 T4 runtime for this notebook.')
try:
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
except FileNotFoundError as exc:
    raise RuntimeError('Select Runtime > Change runtime type > T4 GPU, then Run all.') from exc
if gpu.returncode or 'T4' not in gpu.stdout:
    raise RuntimeError('The canonical run requires a T4 GPU. Change the runtime before Run all.')
print('Device:', gpu.stdout.strip())
ROOT = Path.cwd() / 'outputs' / 'philippine_biodiversity' / uuid.uuid4().hex[:12]
ROOT.mkdir(parents=True)
if shutil.disk_usage(ROOT).free < 12 * 1024**3:
    raise RuntimeError('At least 12 GiB free disk is required for dependencies, two model snapshots and outputs.')
print('Run directory:', ROOT)
"""

BOOTSTRAP = r"""
UV_URL = 'https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
UV_SHA256 = 'aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60'
for attempt in range(3):
    try:
        with urllib.request.urlopen(UV_URL, timeout=90) as response:
            wheel = response.read(20081405)
        break
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        if attempt == 2:
            raise
        time.sleep(2 ** attempt)
if len(wheel) != 20081404 or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
    raise RuntimeError('uv wheel size/hash mismatch')
with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
    member = next(name for name in archive.namelist() if name.endswith('.data/scripts/uv'))
    UV = ROOT / 'uv'
    UV.write_bytes(archive.read(member))
UV.chmod(0o700)
ENV = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
ENV.pop('HF_TOKEN', None)
ENV.pop('HUGGING_FACE_HUB_TOKEN', None)
subprocess.run([str(UV), 'venv', '--managed-python', '--python', '3.12.12', str(ROOT / 'env')], env=ENV, check=True)
PYTHON = ROOT / 'env/bin/python'
subprocess.run([str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:',
                '--index-url', 'https://pypi.org/simple', '-r', str(ROOT / 'requirements.txt')], env=ENV, check=True)
subprocess.run([str(PYTHON), '-c', 'import sys,torch; print(sys.version); print(torch.__version__); assert torch.cuda.is_available()'], env=ENV, check=True)
BOOTSTRAP_SECONDS = time.perf_counter() - SESSION_START
(ROOT / 'bootstrap.json').write_text(json.dumps({'seconds': BOOTSTRAP_SECONDS, 'gpu': gpu.stdout.strip()}), encoding='utf-8')

def run_stage(stage, *options):
    log = ROOT / (stage + '.log')
    print('Running', stage, 'in a separate process. Log:', log, flush=True)
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.Popen([str(PYTHON), '-u', str(ROOT / 'capstone.py'), '--root', str(ROOT), '--stage', stage, *options],
                                  env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in result.stdout:
            output.write(line)
            output.flush()
            if line.strip() and len(line) < 2000:
                print(line.rstrip(), flush=True)
        result.wait()
    if result.returncode:
        print(log.read_text(encoding='utf-8')[-7000:])
        raise RuntimeError(f'{stage} failed with exit {result.returncode}. Preserve the diagnostics.')
"""

HELPERS = r"""
import csv
from IPython.display import FileLink, Image, Markdown, display

def show_table(name, limit=20, columns=None):
    path = ROOT / 'outputs' / name
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        available = reader.fieldnames
        rows = list(reader)
    if not available or not rows:
        raise RuntimeError('Expected nonempty evidence table: ' + name)
    if columns is not None:
        missing = set(columns) - set(available)
        if missing:
            raise RuntimeError(f'Missing columns in {name}: {missing}')
    else:
        preferred = ['system', 'label', 'common_name', 'role', 'split', 'count', 'images', 'observers', 'accuracy', 'macro_f1',
                     'top2_accuracy', 'precision', 'recall', 'f1', 'support', 'predicted', 'threshold', 'coverage', 'selective_accuracy',
                     'referred', 'incorrect_accepted', 'canonical']
        columns = [column for column in preferred if column in available] or available[:8]
    def clean(value):
        if value is None or value == '':
            return 'undefined'
        try:
            return f'{float(value):.4f}' if '.' in value or 'e-' in value else value
        except (ValueError, TypeError):
            return str(value).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join(clean(row[column]) for column in columns) + ' |' for row in rows[:limit]]
    display(Markdown('\n'.join(lines)))
    display(FileLink(str(path)))
    if len(rows) > limit:
        print(f'Showing {limit} of {len(rows)} records. Open the CSV for all rows and fields.')

def show_figures(pattern, limit=8):
    paths = sorted((ROOT / 'outputs' / 'figures').glob(pattern))
    if not paths:
        raise RuntimeError('Expected learning figure is missing: ' + pattern)
    for path in paths[:limit]:
        print(path.stem.replace('_', ' '))
        display(Image(filename=str(path)))

def show_record(name):
    path = ROOT / 'outputs' / name
    value = json.loads(path.read_text(encoding='utf-8'))
    rows = [(key, item) for key, item in value.items() if isinstance(item, (str, float, int, bool)) or item is None]
    lines = ['| Evidence | Value |', '| --- | --- |']
    lines += ['| ' + str(key).replace('_', ' ') + ' | ' + str(item).replace('|', '/') + ' |' for key, item in rows]
    display(Markdown('\n'.join(lines)))
    if name == 'activity.json':
        columns = ['threshold', 'coverage', 'selective_accuracy', 'referred', 'incorrect_accepted']
        comparison = value['comparison']
        lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
        lines += ['| ' + ' | '.join(str(row.get(column, 'undefined')) for column in columns) + ' |' for row in comparison]
        display(Markdown('\n'.join(lines)))
    display(FileLink(str(path)))
"""


def build() -> dict:
    files = carried_files()
    dataset = json.loads(files["data_manifest.json"])
    species_text = "; ".join(
        f"{item['common_name']} (*{item['scientific_name']}*)" for item in dataset["classes"]
    )
    target = ROOT / "tutorials" / NAME
    revision = (
        json.loads(target.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]["base_revision"]
        if target.exists()
        else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )
    source = {
        "repository": "kurtvalcorza/bioclip2-biodiversity-pipeline",
        "base_revision": revision,
        "generator": "build_biodiversity_capstone.py/1",
        "generator_sha256": sha(Path(__file__).read_text(encoding="utf-8")),
        "files": {name: sha(text) for name, text in files.items()},
    }
    files["source.json"] = json.dumps(source, indent=2)
    cells = []

    def md(text: str) -> None:
        cells.append(
            {
                "cell_type": "markdown",
                "id": f"md-{len(cells):02d}",
                "metadata": {},
                "source": text.strip().splitlines(True),
            }
        )

    def code(text: str, **metadata: object) -> None:
        ast.parse(text)
        cells.append(
            {
                "cell_type": "code",
                "id": f"code-{len(cells):02d}",
                "metadata": metadata,
                "execution_count": None,
                "outputs": [],
                "source": text.strip().splitlines(True),
            }
        )

    md("""# Mapping Similarity, Not Certainty: Philippine Biodiversity Field-Survey Triage

[![DIMER](https://img.shields.io/badge/DIMER-Applied_AI-165b80)](https://training.dimer1.asti.dost.gov.ph/signin)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/bioclip2-biodiversity-pipeline/blob/main/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb)

**Profile:** E2E · **Mode:** WORKSHOP · **Notebook standard:** 2.2 · **Release status:** Candidate.

**Scientific question:** Can a biological foundation model help triage photographs of four Philippine endemic bird species while referring uncertain cases for expert review?

This optional, modular capstone is for research and education. Basic Python and Colab familiarity are enough; earlier classification and representation activities are helpful. You will audit a local sample, compare general and biological representations, fit a small head, freeze a referral policy, evaluate held-out observers and verify your exported classifier.

**AI Assistance Disclosure:** Generative AI assisted this notebook's code and technical writing under maintainer direction. The maintainer remains responsible for reviewing implementation, validating results and release decisions. AI assistance is not independent verification, provider endorsement or release approval.

Photographs retain their individual CC0 or CC BY terms and creator attribution. BioCLIP 2 weights use MIT terms; SigLIP 2 uses Apache-2.0. This is a frozen educational sample, not an official iNaturalist benchmark. Candidate status remains until the sample review and a fresh hosted end-to-end run have been qualified.
""")
    md(
        """## 1. Orient: what will this system decide?

The candidate list is SPECIES_LIST. The frozen manifest, rather than a live search, fixes the sample and exact taxonomy. Every final species must pass the same licensing, observer and split feasibility gates; reserve substitutions are documented in the dataset manifest.

| Stage | Input → system → output |
| --- | --- |
| Data | Licensed photographs and community-supported labels → integrity and split checks → a frozen observer-disjoint sample |
| Models | Images and candidate names → SigLIP 2 or BioCLIP 2 → model-specific embeddings and class scores |
| Adaptation | Training embeddings → a fitted linear head → four candidate scores |
| Triage | Scores and a validation-selected threshold → accept or refer → a proposed label and review decision |
| Evidence | Held-out predictions → metrics and artifact checks → a reproducible results bundle |

An **embedding** is a vector representing an image or label. **Zero-shot** classification compares images with label text without fitting to our training photographs. A **linear head** fits a small mapping from frozen image vectors to candidate scores. Neither foundation model checks that the image contains a bird or that its true species is in the supplied list.

Before running, predict which pair will be hardest to separate. Name a factor such as framing, plumage, occlusion, lighting or background. Keep this prediction to compare with the error panels later.

Completion notes and written answers are optional personal learning aids, **not required submissions**.
""".replace("SPECIES_LIST", species_text)
    )
    md("""## 2. Set up the reproducible runtime

Select **Runtime → Change runtime type → T4 GPU**, start a fresh runtime and select **Run all**. The default path needs no login, token, upload, repository clone, DIMER service or manual restart. Infrastructure cells can be collapsed.

The notebook creates an isolated Python 3.12 environment from a fully hashed lock and carries its implementation. Public assets are pinned by revision, byte count and SHA-256. It needs **at least 12 GiB free disk** for CUDA dependencies, both model snapshots and outputs; final measured download time, runtime and peak RAM/VRAM are reported by the run, not promised in advance. SigLIP and BioCLIP run in separate processes and are never resident simultaneously. Defaults use float32 and bounded batches.

**Input:** a fresh T4 runtime. **System:** verified carrier plus locked dependencies. **Output:** a unique run directory and an isolated executable. Integrity failures stop execution; do not disable checks to continue.
""")
    code(PREFLIGHT)
    code(
        "CARRIED_FILES = "
        + repr(files)
        + "\nCARRIED_HASHES = "
        + repr({name: sha(text) for name, text in files.items()})
        + r"""
for name, text in CARRIED_FILES.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')
    if hashlib.sha256(path.read_bytes()).hexdigest() != CARRIED_HASHES[name]:
        raise RuntimeError('Carried file integrity failure: ' + name)
print('Embedded implementation, dependencies and manifests verified.')
""",
        cellView="form",
        dimer={"embedded_sources": True},
    )
    code(BOOTSTRAP + HELPERS, cellView="form")
    md("""## 3. Inspect the Philippine field-photo sample

The target design is 56 photographs: 14 per species, with **8 train / 3 validation / 3 test** per species (32/12/12 overall). Observers are assigned globally to one role, including observers who photographed several species. Each species must have at least two observers in each role. Photo IDs, observation IDs, byte hashes and perceptual duplicates are checked before inference.

**Research Grade** means community-supported identification under iNaturalist's rules; it is not infallible ground truth. A balanced label table does not imply balanced life stages, framing, equipment, backgrounds or sites. This dataset records no precise coordinates or locality fields. Contact sheets show training examples with attribution; withheld test predictions remain hidden until choices are locked.

**Input:** immutable image/label/licence records. **System:** bounded downloads and validation. **Output:** counts, attributions, a contact sheet and validation-only baselines. Predict whether observer grouping will make the comparison harder than randomly assigning photographs.
""")
    code(
        "run_stage('prepare')\nshow_table('dataset_summary.csv')\nshow_record('prepare.json')\nshow_figures('contact_train.png')\nshow_table('attribution.csv', limit=8)\nshow_table('refusal_probe_results.csv')"
    )
    md("""**What to notice:** inspect crop geometry as well as the original framing. The organism may occupy very different fractions of each photograph. Read observer counts alongside image counts. Look for one case where background or crop could help or hurt identification.

<details><summary>Why keep observers separate?</summary>Pictures by one observer can share camera characteristics, editing style and places. Placing that observer on both sides of the split could reward these shortcuts. Global grouping reduces this within-sample leakage; it does not prove pretraining independence.</details>

**Pretraining boundary:** BioCLIP's TreeOfLife pretraining and SigLIP's web-scale pretraining may include these photographs or related content. Exact overlap is unresolved. Holding out our observers protects head fitting and model selection inside this capstone; it cannot establish novel species, sites or images relative to pretraining.
""")
    md("""## 4. Start with simple baselines

The **majority baseline** chooses the most frequent training label, with a fixed lexical tie-break. The **colour nearest-centroid baseline** measures coarse RGB features in the model-visible centre crop, fits a class centroid using training images and selects the nearest one. These ask whether label balance or simple appearance already explains a result.

Only validation results are shown here. **Accuracy** is the fraction correct; **macro-F1** averages each species' F1 equally. F1 combines precision (how often a predicted species is right) and recall (how many examples of that species were found). For comparable tables, the declared zero-division convention reports precision as zero when a class has no predictions; mathematically its denominator is empty. Read predicted counts and support before interpreting averages.

Predict: will the colour baseline separate the species, or mostly exploit backgrounds?
""")
    code("show_table('validation_baselines.csv')\nshow_figures('baseline_validation.png')")
    md(
        """**Interpret:** use the validation confusion counts to identify a weak class. A simple baseline succeeding is useful evidence; the biological model is not required to win. Keep all methods and all selected records in the final comparison."""
    )
    md("""## 5. Compare general and biological zero-shot systems

**Input:** the same images and either common or scientific names. **System:** pinned SigLIP 2 followed by pinned BioCLIP 2. **Output:** candidate rankings and validation metrics under four predeclared conditions.

SigLIP 2 is the general-purpose comparator. BioCLIP 2 is trained for biological representations. **BioCLIP scientific-name zero-shot is the canonical zero-shot reference** regardless of which prompt appears better. Their score magnitudes are not directly comparable or calibrated probabilities of taxonomic truth.

Predict whether scientific names will help both models equally. Change only the prompt vocabulary: keep images, split, checkpoints and preprocessing fixed. Models run sequentially, with process exit releasing memory between them.
""")
    code(
        "run_stage('siglip-zero-shot')\nshow_table('validation_siglip.csv')\nrun_stage('bioclip-zero-shot')\nshow_table('validation_bioclip.csv')\nshow_table('validation_per_class.csv')\nshow_figures('zero_shot_validation.png')"
    )
    md("""**Compare:** which prompt condition changes validation rankings, and for which species? Treat a difference from twelve validation photographs as a small-sample observation. Do not choose the canonical prompt from eventual test performance.

<details><summary>Why might scientific names behave differently?</summary>A biological model's training text may align taxonomic names closely with organism imagery; a general model may have encountered more common-name descriptions. This is a hypothesis about representation, not a guarantee for any species.</details>""")
    md("""## 6. Explore the frozen representation

**Input:** cached BioCLIP image vectors. **System:** L2-normalisation, train-only five-nearest-neighbour voting and a two-dimensional PCA projection fitted on train only. **Output:** validation predictions and a view of the embedding geometry.

For **5-NN**, only training images can be neighbours; votes sum non-negative cosine similarities, with fixed class-order ties. No gradients are needed. **PCA** compresses vectors to two coordinates for display; it can hide separation or invent apparent clusters. Split shapes and species colours are visual aids, not additional evaluation evidence.

Predict whether images of one species cluster together even when their observers differ.
""")
    code("run_stage('representations')\nshow_table('validation_knn.csv')\nshow_figures('pca.png')")
    md(
        """**What to notice:** compare validation 5-NN results with zero-shot results. Neighbour voting tests how this local training set is arranged in the representation. A plausible PCA cluster does not establish accuracy, independence or a species' geographic distribution."""
    )
    md("""## 7. Fit a small head and lock the policy

Both BioCLIP towers stay frozen. A linear head starts from scientific-name text embeddings and is fitted on training vectors using float32 AdamW, seed 42 and at most 20 epochs. This is **head-only adaptation**, not full-model fine-tuning. The selected epoch has the lowest validation cross-entropy; exact ties keep the earlier epoch. Read the exported configuration for trainable tensors, parameter count, learning rate and weight decay.

Before any test metrics are revealed, lock the referral rule using **validation only**. Confidence is the largest minus second-largest head softmax score. Consider every distinct validation margin, requiring at least 50% coverage. Among thresholds reaching 80% selective accuracy choose greatest coverage; if none reaches it, choose greatest selective accuracy then coverage. Remaining ties choose the higher threshold. Acceptance uses the declared threshold comparison consistently.

The 80% teaching target is not a deployment guarantee. With twelve validation records, one decision can materially change a percentage. Predict whether decreasing training loss will always improve validation loss.
""")
    code("run_stage('adapt-head')\nshow_record('selected_policy.json')\nshow_figures('loss.png')")
    md(
        """**Interpret:** locate the selected epoch rather than assuming the final epoch is best. A widening train/validation gap can indicate overfitting even with a frozen backbone. The head and validation threshold are now fixed. The next section must preserve negative or inconclusive results."""
    )
    md("""## 8. Reveal the observer-disjoint test results

**Input:** test photographs excluded from head fitting and all capstone selection. **System:** the eight predeclared systems with frozen settings. **Output:** the primary head macro-F1 endpoint, comparisons, per-species counts and uncertainty intervals.

The systems are majority, colour, SigLIP common/scientific names, BioCLIP common/scientific names, BioCLIP 5-NN and the BioCLIP head. Top-2 accuracy asks whether the true label occurs among the two highest scores; it does not mean either is verified. Accuracy and macro-F1 intervals use at least 2,000 deterministic bootstrap resamples of test records. These intervals omit pretraining overlap, selection bias and label error; correlated photographs can make record resampling optimistic.

Predict whether head adaptation improves every species or redistributes errors. Do not revise the head, prompt, k or threshold after reading this section.
""")
    code(
        "run_stage('evaluate')\nshow_table('metrics.csv')\nshow_table('metrics.csv', columns=['system', 'accuracy_low', 'accuracy_high', 'macro_f1_low', 'macro_f1_high', 'paired_correctness_difference_vs_bioclip_scientific'])\nshow_table('per_class_metrics.csv', limit=32)\nshow_figures('confusion_test_*.png')\nshow_figures('errors.png')"
    )
    md("""**What to notice:** pair each average with support and predicted counts. Read confusion counts and row-normalised percentages. The error panel uses declared diagnostic categories: correct high confidence, incorrect high confidence, lowest margin and model disagreement. A category may be empty; do not manufacture a failure or success to fill it.

Compare the head with the canonical BioCLIP scientific-name reference using paired per-record correctness. Identify one plausible explanation and one alternative, such as crop, background, community label uncertainty or pretraining overlap. These twelve test images do not estimate nationwide identification performance.""")
    md("""## 9. Change one thing: the review threshold

**Predict → change one thing → run → observe → explain.** First run the canonical triage policy. It reports acceptance coverage, accuracy among accepted cases, incorrect accepted cases, referrals and per-species referral rates. Primary classification metrics still include every test record.

Then use the form below to display a lower, canonical or higher threshold around the locked value. Only the display decision changes. Weights, scores, records and canonical exports remain fixed. The test risk-coverage curve is a diagnostic, not permission to retune the reported policy.

Predict how a higher threshold changes coverage. Does it necessarily remove all incorrect accepted cases?
""")
    code(
        "run_stage('triage')\nshow_table('risk_coverage.csv', limit=40)\nshow_table('referral_by_species.csv')\nshow_figures('risk_coverage.png')"
    )
    code(
        r"""# @title Display-only review-threshold activity
THRESHOLD_VIEW = 'canonical' # @param ['lower', 'canonical', 'higher']
policy = json.loads((ROOT / 'outputs' / 'selected_policy.json').read_text(encoding='utf-8'))
threshold = float(policy['threshold'])
offset = {'lower': -0.1, 'canonical': 0.0, 'higher': 0.1}[THRESHOLD_VIEW]
display_threshold = min(1.0, max(0.0, threshold + offset))
print('Locked threshold:', threshold, '; display-only threshold:', display_threshold)
run_stage('triage', '--display-threshold', str(display_threshold))
show_record('activity.json')
show_figures('activity.png')
""",
        cellView="form",
    )
    md(
        """**Explain:** compare how many records were accepted and how many accepted labels were wrong. A high margin may still accompany a confident mistake. Referral is a workflow policy, not calibrated uncertainty or a reliable detector of unfamiliar organisms."""
    )
    md("""## 10. Challenge the closed candidate list

The separate probe set includes at least two bird species outside the candidate list, two non-bird Philippine organisms and a synthetic blank control. These records never enter fitting, threshold selection or primary metrics.

**Input:** unsuitable or outside-list images. **System:** the same four-class head and locked policy. **Output:** forced candidate labels plus descriptive referral behaviour. Predict whether all unsuitable images will be referred.
""")
    code(
        "show_table('probe_predictions.csv', columns=['id', 'truth', 'prediction', 'margin', 'accepted'])\nshow_figures('probes.png')"
    )
    md(
        """**Interpret:** a four-class head must return one of its four labels even when none is correct. Some probes may be accepted confidently. Record that failure rather than scoring an outside-list label as an ordinary four-class test example. Expert review and a broader validation design remain necessary for field use."""
    )
    md("""## 11. Verify the head and results bundle

The adapter contains only the declared head tensors in safetensors format. Reload verifies the base model/revision, class order, byte count, digest, tensor names, shapes and finiteness before use. A **new process ID** reconstructs BioCLIP and reproduces a fixed test probe, checking score tolerance, labels, meaningful top-2 ordering and canonical referral decisions.

The report reopens CSVs, recomputes headline metrics, verifies archive CRC and every member digest, and refuses unsafe paths. The results ZIP includes derived evidence, manifests, the small head and licences. Source photographs, image contact sheets, model weights, package caches, raw API payloads and precise locations are excluded. Complete attribution stays available beside locally displayed images.
""")
    code(
        "run_stage('reload')\nshow_record('verification_reload.json')\nrun_stage('report')\nshow_record('verification.json')\nshow_record('run_summary.json')\ndisplay(FileLink(str(ROOT / 'outputs' / 'results.zip')))\ndisplay(Markdown((ROOT / 'outputs' / 'conclusion.md').read_text(encoding='utf-8')))"
    )
    md("""### Optional: explore your own photographs

This form is disabled by default and runs in a separate directory. Upload your ZIP through Colab's Files panel, then enter its path. Acceptable layouts are `labels.csv` with `id,image,label,scientific_name,observer_group`, or `<label>/<image>` folders plus a JSON mapping of labels to scientific names. Use 2–20 classes, at least twelve images per class and no duplicate images. No location fields are required or retained.

Grouped evaluation requires at least six distinct observer groups per class so train, validation and test can each contain two observers, with a feasible global assignment and at least 6/3/3 images. This is stricter than merely supplying three groups per class. If groups are absent, only descriptive inference is permitted; the notebook refuses to present an equivalent held-out adaptation score. Set the rights confirmation only for photographs you are authorised to process. BYOD images are never added to the canonical archive.
""")
    code(
        r"""# @title Optional BYOD (default off)
USE_BYOD = False # @param {type:'boolean'}
BYOD_ZIP = '' # @param {type:'string'}
BYOD_CLASS_PROMPTS_JSON = '{}' # @param {type:'string'}
BYOD_AUTHORIZED = False # @param {type:'boolean'}
if not USE_BYOD:
    print('Optional BYOD is off. The canonical capstone is complete.')
else:
    if not BYOD_AUTHORIZED:
        raise ValueError('Confirm that you are authorised to process the supplied photographs.')
    zip_path = Path(BYOD_ZIP).expanduser().resolve()
    if not zip_path.is_file():
        raise ValueError('Upload a ZIP in the Colab Files panel and enter its actual path.')
    prompts = json.loads(BYOD_CLASS_PROMPTS_JSON)
    if not isinstance(prompts, dict):
        raise ValueError('Class prompts must be a JSON object mapping labels to scientific names.')
    canonical_root = ROOT
    byod_root = ROOT.parent / (ROOT.name + '-byod-' + uuid.uuid4().hex[:6])
    byod_root.mkdir(parents=True)
    for name, text in CARRIED_FILES.items():
        target = byod_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8', newline='\n')
    prepare_script = ('import json,sys; from pathlib import Path; import biodiversity_core as core; '
                      'core.prepare_byod(Path(sys.argv[1]),Path(sys.argv[2]),'
                      'class_prompts=json.loads(sys.argv[3]),authorization=True)')
    subprocess.run([str(PYTHON), '-c', prepare_script, str(byod_root), str(zip_path), json.dumps(prompts)],
                   cwd=byod_root, env=ENV, check=True)
    manifest = json.loads((byod_root / 'data_manifest.json').read_text(encoding='utf-8'))
    try:
        ROOT = byod_root
        if manifest.get('inference_only'):
            run_stage('byod-infer')
            show_table('byod_predictions.csv')
            print('Ungrouped BYOD: descriptive inference only; no held-out metric.')
        else:
            for stage in ('prepare', 'siglip-zero-shot', 'bioclip-zero-shot', 'representations',
                          'adapt-head', 'evaluate', 'triage', 'reload', 'report'):
                run_stage(stage)
            show_table('metrics.csv')
            display(FileLink(str(ROOT / 'outputs' / 'results.zip')))
        print('Separate BYOD outputs:', ROOT / 'outputs')
    finally:
        ROOT = canonical_root
""",
        cellView="form",
    )
    md("""## 12. Conclude with evidence

Complete this optional record in your own words:

> On this observer-disjoint sample of [n] Philippine bird photographs, [system] achieved [macro-F1 with interval] compared with [baselines]. The locked review policy accepted [coverage] with [selective accuracy], while [failure pattern] remained. Changing only the display threshold showed [trade-off]. This supports [bounded use], but does not establish pretraining independence, open-set recognition, nationwide accuracy or expert-level identification. We would next collect [specific evidence].

Research Grade labels may be wrong. Candidate scores are not taxonomic truth. Observer grouping does not remove unresolved foundation-model pretraining overlap. One small frozen sample cannot establish population abundance, conservation status, geographic occurrence or operational performance. A head that fails to beat zero-shot is still an informative result. The strongest next study uses more independent observers, species and labelled cases, with the evaluation plan fixed before test inspection.

## 13. Continue exploring DIMER

Browse the [DIMER model repository](https://training.dimer1.asti.dost.gov.ph/signin) and explore the full [BioCLIP 2 pipeline notebook](https://github.com/kurtvalcorza/bioclip2-biodiversity-pipeline/blob/main/tutorials/bioclip2_biodiversity_colab.ipynb) or [SigLIP 2 pipeline](https://github.com/kurtvalcorza/siglip2-vision-language-pipeline) for deeper data preparation, supported adaptation, evaluation, inference and artifact contracts. Completion records are not required submissions.

## Troubleshooting and sources

| Observation | Response |
| --- | --- |
| Missing T4 or insufficient disk | Start a fresh T4 runtime with the stated free-space requirement. |
| Asset hash, licence or split refusal | Stop and retain diagnostics; do not weaken validation or substitute photos. |
| No method improves on zero-shot | Preserve results and examine sample size, representations and errors. |
| High confidence on a non-bird | Record the closed-set failure; confidence does not prove candidate membership. |
| Reload or CSV parity fails | Retain outputs and resolve the mismatch before release. |
| Undefined metric | Read its denominator and support; do not replace it with a favourable number. |

Sources: [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2), [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224), [iNaturalist developer guidance](https://www.inaturalist.org/pages/developers), [iNaturalist open data](https://github.com/inaturalist/inaturalist-open-data), [DIMER notebook standard](https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md). Exact immutable model and data identities are carried in this notebook and exported in the run.
""")
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "accelerator": "GPU",
            "colab": {"name": NAME, "gpuType": "T4", "provenance": []},
            "dimer": {
                "notebook_spec": "2.2",
                "profile": "E2E",
                "notebook_profile": "E2E",
                "pedagogical_mode": "WORKSHOP",
                "notebook_mode": "WORKSHOP",
                "standalone": True,
                "requires_dimer_worker": False,
                "release_status": "Candidate",
                "generated_from": source,
            },
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "tutorials" / NAME
    content = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            raise SystemExit("Biodiversity capstone differs from generated sources")
        print("Biodiversity capstone parity: PASS")
    else:
        target.write_text(content, encoding="utf-8", newline="\n")
        print(target)


if __name__ == "__main__":
    main()
