"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, sample_data.py, samples.py, metrics.py), and the model pin/stage/verify cells
are produced by the generator from repository sources so they cannot drift from the package.

This template configures an E2E species-classification workflow: the pinned BioCLIP 2 snapshot is
digest-verified, 48 embedded CC0 iNaturalist photographs of four sparrow species are validated and
split, image embeddings and zero-shot species classification are run, trivial baselines are
measured, a bounded head fine-tuning initialised from the zero-shot classifier runs in the kernel,
and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "bioclip2-biodiversity-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/bioclip2_biodiversity_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-imageomics%2Fbioclip--2-ffcc4d?style=flat",
        "https://huggingface.co/imageomics/bioclip-2",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-Imageomics%2Fbioclip--2-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/Imageomics/bioclip-2",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2505.23883-b31b1b.svg", "https://arxiv.org/abs/2505.23883"),
]

TEMPLATE = {
    "package": "bioclip2_biodiversity_pipeline",
    "repo_name": REPO,
    "stem": "bioclip2_biodiversity",
    "notebook_name": "bioclip2_biodiversity_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned BioCLIP 2 snapshot (3 files, 1.71 GB), builds the CLIP model from the pinned config and loads the weights "
        "strictly, decodes the 48 CC0 iNaturalist sparrow photographs carried inside this notebook (no dataset download), "
        "validates the image and dataset contracts, splits them into stratified train/validation/test sets, shows what the "
        "input validation rejects, computes image embeddings and checks they are reproducible, classifies the test split "
        "zero-shot from scientific names (the model's own prior), measures a majority-class and a colour baseline, runs a "
        "bounded AdamW fine-tuning of a linear head initialised from the zero-shot classifier, evaluates accuracy, macro-F1 and "
        "AUROC on the held-out test split, classifies six held-out images as new inputs, exports the adapter as safetensors with "
        "a manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. The default path needs no "
        "repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC "
        "2.0 §5). On CPU the whole path takes about two minutes of model time after the download."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled organism photographs as a zip of `<label>/<image>` folders, or a zip holding a CSV (`id,image,label`), a JSON "
        "array or a JSONL file with image paths relative to it. They pass through the same image validation, stratified split, "
        "zero-shot and trivial baselines, adaptation, held-out evaluation, inference, artifact export and reload-parity cells as "
        "the sample. Supply the scientific name of each class in `BYOD_CLASS_PROMPTS` so zero-shot and the head initialisation "
        "can use it. The expected layout and the image ceilings are stated in the Prerequisites and in Section 4, and uploaded "
        "files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "BioClip2Pipeline",
    "weights_key": "bioclip-2",
    "modules": ["pipeline.py", "sample_data.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "open_clip", "safetensors"],
    "title": "BioCLIP 2 — DIMER E2E species-classification tutorial (standalone)",
    "badges": BADGES,
    "capability": "zero-shot species classification, organism image embeddings and bounded species-classification fine-tuning",
    "intro": (
        "BioCLIP 2 is a vision-language foundation model for organismal biology: a CLIP model (ViT-L/14 image tower, 12-layer "
        "text tower, 768-dimensional joint space) trained by the Imageomics Institute on TreeOfLife-200M — about 214 million "
        "images covering 952 thousand taxa — with hierarchical contrastive learning over taxonomic names. Because images and "
        "taxon names share one space, it can classify a photograph against **any** list of species names with no training at "
        "all, and the upstream paper reports that this scaled training also produces representations that separate traits and "
        "ecological attributes it was never told about.\n\n"
        "This tutorial exercises three things on real field data. The dataset is 48 photographs — 12 each of four North American "
        "sparrow species — from research-grade iNaturalist observations whose photos carry the CC0 licence, carried inside this "
        "notebook. The four species were chosen **before any result was seen** to make colour a weak cue: all four are streaked "
        "grey-brown birds, so a colour baseline has little to use and any separation comes from finer structure. First the model "
        "classifies them zero-shot from scientific names; then a linear head is fine-tuned starting *from* that zero-shot "
        "classifier, so every number after training is directly comparable to the number before it."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, sample, dataset and metrics modules; stage and digest-verify "
        "an immutable 1.71 GB snapshot and build the model from its pinned config rather than a library registry; validate "
        "images and split a labelled dataset without leakage; extract L2-normalised image embeddings and confirm they are "
        "reproducible; classify zero-shot from taxonomic names and read what the scores do and do not mean; measure "
        "majority-class and colour baselines; run a bounded fine-tuning whose starting point *is* the zero-shot classifier; "
        "evaluate accuracy, macro-F1 and AUROC on an independent test split; classify held-out images; and export a "
        "safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "the published BioCLIP 2 benchmarks (NABirds, Rare Species, Meta-Album, NeWT, FishNet and the rest), open-set or "
        "hierarchical (genus/family) prediction, image-to-image retrieval at scale, the newer `imageomics/bioclip-2.5-vith14` "
        "checkpoint, and any claim that a photograph shows what its label says. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU is enough — image embedding is about 0.3 s per image and the default head-only fine-tuning runs on features computed once — and CUDA is used automatically when present. The snapshot download is 1.71 GB.",
        "- **Knowledge:** what a scientific (binomial) name is, what zero-shot classification with a vision-language model means, and how accuracy, macro-F1 and AUROC differ.",
        "- **No remote code:** the checkpoint is a plain safetensors state dict loaded strictly into `open_clip`'s own CLIP module, built from the pinned `open_clip_config.json`. Nothing from the Hub is executed.",
        "- **Data contract:** records are `{{id, image_bytes, label}}` in memory; images are JPEG, PNG or WEBP with a shorter side of at least 32 px and a longer side of at most 8192 px, decoded to RGB and centre-cropped to 224x224 by the CLIP transform; unique ids and unique image bytes; at least 8 records and 3 per class, 2..50 classes. BYOD accepts a zip of `<label>/<image>` folders or a CSV/JSON/JSONL of image paths.",
        "- **Validation is decoding, not biology:** nothing checks that a photograph shows an organism, that it fills the frame, or that a label names a real taxon. A photo of a rock is embedded and classified without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. Location metadata is the sensitive part of a biodiversity record — the sample carries none, and a BYOD set should not either. The default path uploads nothing.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample photographs, validation and split\n\n"
                "The default dataset is carried inside this notebook: 48 JPEG photographs, 224x224, decoded from base64 in the "
                "`sample_data` module above, each traceable to its iNaturalist observation (the observation URL, photo id and "
                "observer login are recorded; the CC0 licence code was checked per photo when the set was built; no location was "
                "collected). `validate_dataset` decodes every image — that is the check — and reports sizes, class counts, the "
                "ceilings and a dataset digest before any model runs. `split_dataset` shuffles within each class and cuts 20 % "
                "validation / 25 % test; the sample keeps one photo per observer within a species, which is what makes a random "
                "split defensible here.\n\n"
                "Look for: 48 images, four classes, 12 each, splits 28/8/12, the contact sheet, and a written "
                "`outputs/{stem}_sample_dataset.csv` with an `images/` folder beside it — the BYOD shape. The photographs are "
                "unfiltered beyond licence and quality grade: some birds are small in the frame, one is held in a hand. That is "
                "what field data looks like."
            ),
            "code": (
                "import io\n"
                "import json\n"
                "import os\n"
                "import zipfile\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "VAL_FRACTION = 0.2  # @param {{type:\"number\"}}\n"
                "TEST_FRACTION = 0.25  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n"
                "BYOD_CLASS_PROMPTS = {{}}  # e.g. {{'song_sparrow': 'Melospiza melodia'}}: one scientific name per BYOD label\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_root = Path('work') / 'byod'\n"
                "    byod_root.mkdir(parents=True, exist_ok=True)\n"
                "    if file_name.lower().endswith('.zip'):\n"
                "        with zipfile.ZipFile(io.BytesIO(payload)) as zf:\n"
                "            for member in zf.infolist():\n"
                "                target = (byod_root / member.filename).resolve()\n"
                "                if not str(target).startswith(str(byod_root.resolve())):\n"
                "                    raise ValueError('zip member escapes the upload directory: ' + member.filename)\n"
                "                if not member.is_dir():\n"
                "                    target.parent.mkdir(parents=True, exist_ok=True)\n"
                "                    target.write_bytes(zf.read(member))\n"
                "        tables = sorted(p for p in byod_root.rglob('*') if p.suffix.lower() in ('.csv', '.json', '.jsonl'))\n"
                "        byod_source = tables[0] if tables else byod_root\n"
                "    else:\n"
                "        byod_source = byod_root / file_name\n"
                "        byod_source.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_source)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    class_prompts = dict(BYOD_CLASS_PROMPTS)\n"
                "else:\n"
                "    records = generate_sample_dataset()\n"
                "    data_source = 'embedded CC0 iNaturalist sample (' + str(SAMPLE_SIZE) + ' photographs, four sparrow species)'\n"
                "    class_prompts = dict(SAMPLE_CLASS_PROMPTS)\n\n"
                "dataset_manifest = validate_dataset(records)\n"
                "CLASSES = dataset_manifest['classes']\n"
                "missing_prompts = [c for c in CLASSES if c not in class_prompts]\n"
                "if missing_prompts:\n"
                "    raise ValueError('supply a scientific name for every class in BYOD_CLASS_PROMPTS: ' + str(missing_prompts))\n"
                "splits = split_dataset(records, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(records, 'outputs/{stem}_sample_dataset.csv')\n\n"
                "print({{'data_source': data_source, 'n_records': dataset_manifest['n_records'], 'classes': CLASSES, 'class_counts': dataset_manifest['class_counts']}})\n"
                "print({{'class_prompts': class_prompts, 'validation': dataset_manifest['validation']}})\n"
                "print({{'image_width': dataset_manifest['image_width'], 'image_height': dataset_manifest['image_height'], 'ceilings': dataset_manifest['ceilings'], 'digest': dataset_manifest['digest'][:16] + '...'}})\n"
                "print({{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}})\n"
                "if not USE_BYOD:\n"
                "    print({{'provenance': sample_provenance()}})\n\n"
                "from PIL import Image\n\n"
                "thumb, per_row = 96, 12\n"
                "shown = records[:48]\n"
                "sheet = Image.new('RGB', (per_row * thumb, ((len(shown) + per_row - 1) // per_row) * thumb), 'white')\n"
                "for i, r in enumerate(shown):\n"
                "    sheet.paste(decode_image(r['image_bytes']).resize((thumb, thumb)), ((i % per_row) * thumb, (i // per_row) * thumb))\n"
                "sheet.save('outputs/{stem}_contact_sheet.png')\n"
                "try:\n"
                "    from IPython.display import display\n"
                "    display(sheet)\n"
                "except ImportError:\n"
                "    print('contact sheet written to outputs/{stem}_contact_sheet.png')"
            ),
        },
        {
            "md": (
                "## 5. What the input validation accepts and refuses\n\n"
                "`validate_inputs` decodes each image with Pillow, records its size and SHA-256, and raises exactly what "
                "`embed_images` or `classify` would raise. The ceilings are the checkpoint's and the runtime's, not biology's: a "
                "shorter side of at least `MIN_IMAGE_SIDE` px, a longer side of at most `MAX_IMAGE_SIDE` px, JPEG/PNG/WEBP, at "
                "most `MAX_IMAGES_PER_CALL` images per call and `MAX_LABELS` labels per zero-shot call. The cell shows three "
                "refusals — undecodable bytes, an unsupported format, a too-small image — and one acceptance. What it cannot "
                "show is a *semantic* refusal, because there is none: the schema says so in words."
            ),
            "code": (
                "print({{'max_images_per_call': MAX_IMAGES_PER_CALL, 'image_side_pixels': [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE], 'accepted_formats': list(ACCEPTED_FORMATS), 'max_labels': MAX_LABELS, 'image_size': IMAGE_SIZE}})\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n\n"
                "def encoded(image, fmt):\n"
                "    buf = io.BytesIO()\n"
                "    image.save(buf, fmt)\n"
                "    return buf.getvalue()\n\n"
                "probes = {{\n"
                "    'garbage bytes': b'not an image',\n"
                "    'BMP file': encoded(Image.new('RGB', (64, 64), (200, 30, 30)), 'BMP'),\n"
                "    '16x16 image': encoded(Image.new('RGB', (16, 16)), 'JPEG'),\n"
                "}}\n"
                "for name, payload in probes.items():\n"
                "    try:\n"
                "        validate_inputs([payload])\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})\n\n"
                "input_manifest = validate_inputs([r['image_bytes'] for r in test_records[:4]], names=[r['id'] for r in test_records[:4]])\n"
                "print({{'verdict': input_manifest['verdict'], 'n_images': input_manifest['n_images'], 'requires_remote_code': input_manifest['requires_remote_code'], 'first': input_manifest['inputs'][0]}})"
            ),
        },
        {
            "md": (
                "## 6. Image embeddings (representation, not prediction)\n\n"
                "`pipe.embed_images` runs the verified image tower and returns one L2-normalised 768-dimensional vector per "
                "image — a point in the joint image/text space. Embeddings are representations: they carry no label and no metric "
                "of their own; a downstream task is what gives them meaning (EVAL9). The cell embeds the eight validation images, "
                "writes them with their ids to `outputs/{stem}_embeddings.csv` (OUT4), and prints the mean cosine similarity "
                "within and between species as an inspection, not an evaluation.\n\n"
                "It also checks reproducibility the honest way: calling the same batch twice returns identical vectors, and "
                "embedding one of those images on its own differs by at most ~1e-7 — float accumulation order changes with the "
                "batch, which is noise, not stochasticity. A reader who needs bit-identical embeddings across batch sizes must "
                "fix the batch composition."
            ),
            "code": (
                "import csv\n"
                "import math\n\n"
                "embed_records = val_records[:8]\n"
                "embedding_result = pipe.embed_images([r['image_bytes'] for r in embed_records], names=[r['id'] for r in embed_records])\n"
                "vectors = embedding_result['embeddings']\n"
                "print({{'n_images': embedding_result['n_images'], 'dimension': embedding_result['dimension'], 'pooling': embedding_result['pooling'], 'norm_of_first': round(math.sqrt(sum(x * x for x in vectors[0])), 6)}})\n\n"
                "repeat = pipe.embed_images([r['image_bytes'] for r in embed_records], names=[r['id'] for r in embed_records])['embeddings']\n"
                "single = pipe.embed_images([embed_records[0]['image_bytes']])['embeddings'][0]\n"
                "cross_batch = max(abs(a - b) for a, b in zip(single, vectors[0]))\n"
                "print({{'same_batch_twice_identical': repeat == vectors, 'single_vs_batch_max_abs_diff': cross_batch, 'note': 'batch composition changes float accumulation order; this is noise at the 1e-7 level, not stochastic inference'}})\n"
                "assert repeat == vectors\n"
                "assert cross_batch < 1e-5\n\n"
                "def cosine(a, b):\n"
                "    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))\n\n"
                "within, between = [], []\n"
                "for i in range(len(embed_records)):\n"
                "    for j in range(i + 1, len(embed_records)):\n"
                "        (within if embed_records[i]['label'] == embed_records[j]['label'] else between).append(cosine(vectors[i], vectors[j]))\n"
                "print({{'mean_cosine_within_species': round(sum(within) / len(within), 4) if within else None, 'mean_cosine_between_species': round(sum(between) / len(between), 4) if between else None, 'note': 'inspection only; embeddings are unlabelled representations'}})\n\n"
                "with open('outputs/{stem}_embeddings.csv', 'w', encoding='utf-8', newline='') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['id', 'label', 'sha256'] + [f'dim_{{k}}' for k in range(embedding_result['dimension'])])\n"
                "    for r, meta, vec in zip(embed_records, embedding_result['images'], vectors):\n"
                "        writer.writerow([r['id'], r['label'], meta['sha256']] + [f'{{x:.6f}}' for x in vec])\n"
                "print('wrote outputs/{stem}_embeddings.csv')"
            ),
        },
        {
            "md": (
                "## 7. Zero-shot species classification — the model's own prior\n\n"
                "`pipe.zero_shot` embeds one prompt per label (`a photo of <scientific name>.` by default, BioCLIP's convention), "
                "embeds the images, and takes a softmax over the scaled cosine similarities. No training is involved: this is "
                "what the pretrained model already knows about these species, and it is the baseline the fine-tuning in Section 9 "
                "must be read against. `zero_shot_evaluate` scores the whole test split this way.\n\n"
                "Three probes make the score semantics concrete (UNC1–UNC4). The scores are a softmax over **the label set you "
                "supplied and nothing else**: a blank grey image and a noise image still receive a confident label, because "
                "closed-set zero-shot always answers. An open-set probe — the same sparrow photo against `a rock`, `a car`, its "
                "species and `a domestic cat` — shows the model choosing the species. And common names are tried alongside "
                "scientific names, because which vocabulary the text tower knows best is an empirical question on your taxa."
            ),
            "code": (
                "baseline_zero_shot = pipe.zero_shot_evaluate(test_records, class_prompts)\n"
                "print({{k: baseline_zero_shot[k] for k in ('baseline', 'template', 'n', 'accuracy', 'macro_f1', 'auroc')}})\n"
                "for cls_name, row in baseline_zero_shot['per_class'].items():\n"
                "    print({{'class': cls_name, 'prompt': DEFAULT_PROMPT_TEMPLATE.format(class_prompts[cls_name]), **row}})\n\n"
                "if not USE_BYOD:\n"
                "    common = pipe.zero_shot_evaluate(test_records, SAMPLE_COMMON_NAMES)\n"
                "    print({{'with_common_names': {{k: common[k] for k in ('accuracy', 'macro_f1')}}, 'note': 'same images, common names in the prompt instead of scientific names'}})\n\n"
                "example = test_records[0]\n"
                "single = pipe.zero_shot([example['image_bytes']], class_prompts, names=[example['id']])\n"
                "print({{'id': example['id'], 'true_label': example['label'], 'predicted': single['predictions'][0]['label'], 'scores': {{k: round(v, 4) for k, v in single['predictions'][0]['scores'].items()}}, 'logit_scale': round(single['logit_scale'], 2)}})\n"
                "print({{'decision_rule': single['decision_rule']}})\n\n"
                "open_set = pipe.zero_shot([example['image_bytes']], ['a rock', 'a car', class_prompts[example['label']], 'a domestic cat'])\n"
                "print({{'open_set_probe': {{k: round(v, 4) for k, v in open_set['predictions'][0]['scores'].items()}}}})\n\n"
                "blank = encoded(Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128)), 'JPEG')\n"
                "import random\n"
                "rnd = random.Random(0)\n"
                "noise = encoded(Image.frombytes('RGB', (IMAGE_SIZE, IMAGE_SIZE), bytes(rnd.getrandbits(8) for _ in range(IMAGE_SIZE * IMAGE_SIZE * 3))), 'JPEG')\n"
                "for name, payload in (('blank grey image', blank), ('uniform noise image', noise)):\n"
                "    p = pipe.zero_shot([payload], class_prompts)['predictions'][0]\n"
                "    print({{'probe': name, 'label': p['label'], 'score': round(p['score'], 4), 'note': 'closed-set zero-shot always answers; a confident label is not evidence of a bird'}})"
            ),
        },
        {
            "md": (
                "## 8. Trivial baselines on the test split\n\n"
                "Two predictors that know nothing about birds set the floor (EVAL10/EVAL11). `majority_baseline` predicts the most "
                "frequent training class — 0.25 on a balanced four-class split. `color_baseline` reduces each image to six numbers "
                "(mean and standard deviation of R, G and B on a 32x32 thumbnail), fits one centroid per class on the **training "
                "split only** (SPL8) and assigns each test image to the nearest centroid.\n\n"
                "Colour is the confounder worth ruling out in a species task, and the four species were chosen to make it weak. "
                "On your own data, read this baseline first: if colour already separates your classes — a red bird versus a "
                "yellow one, or two species photographed against different backgrounds — a model can score well without having "
                "seen any structure."
            ),
            "code": (
                "baseline_majority = majority_baseline(train_records, test_records, CLASSES)\n"
                "print({{k: baseline_majority[k] for k in ('baseline', 'predicted_label', 'accuracy', 'macro_f1')}})\n"
                "baseline_color = color_baseline(train_records, test_records, CLASSES)\n"
                "print({{k: baseline_color[k] for k in ('baseline', 'accuracy', 'macro_f1')}})\n"
                "print({{'centroids_rgb_mean': {{c: v[:3] for c, v in baseline_color['centroids'].items()}}, 'note': 'six-number colour space; fitted on the training split only'}})"
            ),
        },
        {
            "md": (
                "## 9. Bounded fine-tuning, starting from the zero-shot classifier\n\n"
                "`pipe.adapt` copies the image tower, puts a linear head over its L2-normalised projection, and — because "
                "`class_prompts` supplies a scientific name per class — **initialises that head from the zero-shot text "
                "classifier** (the text embeddings scaled by the checkpoint's logit scale). Epoch 0 in the history is therefore "
                "the zero-shot classifier evaluated on the validation split, and every later epoch is comparable to it. AdamW then "
                "runs with the hyperparameters below (FT4/FT6): tutorial values, not production settings. Validation metrics are "
                "**monitoring only**; the final epoch's weights are kept (EVAL14). Training loss going down is optimisation "
                "evidence, not task-quality evidence (FT7) — Section 10 is where quality is measured.\n\n"
                "`TRAINABLE_BLOCKS = 0` trains the head alone on features the frozen tower computes once, which is what 28 "
                "training images support. Setting it to 1 unfreezes the last of the 24 tower blocks: on the sample that took a "
                "saturated zero-shot classifier from 12/12 to 11/12 on the test split — 12.6 million parameters pulled around by "
                "28 images — which is the recorded reason the default is 0."
            ),
            "code": (
                "import time\n\n"
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_BLOCKS = 0  # @param {{type:\"integer\"}}\n\n"
                "started = time.perf_counter()\n"
                "adapt_result = pipe.adapt(\n"
                "    train_records,\n"
                "    val_records,\n"
                "    classes=CLASSES,\n"
                "    class_prompts=class_prompts,\n"
                "    epochs=EPOCHS,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    trainable_blocks=TRAINABLE_BLOCKS,\n"
                "    seed=SEED,\n"
                ")\n"
                "adapt_seconds = round(time.perf_counter() - started, 2)\n"
                "print({{'method': adapt_result['method'], 'head_initialisation': adapt_result['head_initialisation'], 'trainable_parameters': adapt_result['trainable_parameters'], 'total_parameters': adapt_result['total_parameters'], 'precision': adapt_result['precision'], 'device': pipe.device, 'seconds': adapt_seconds}})\n"
                "for step in adapt_result['history']:\n"
                "    print(step)"
            ),
        },
        {
            "md": (
                "## 10. Held-out evaluation\n\n"
                "`pipe.evaluate` classifies every image of a split with the adapted head and reports `accuracy`, `macro_f1` (the "
                "unweighted mean of per-class F1, which exposes a model that ignores a class), per-class precision/recall/F1 with "
                "support, and `auroc` (macro one-vs-rest, ranking quality independent of the argmax). The **test split** was never "
                "used for training or monitoring, so its numbers are the independent evidence (SPL6/SPL7). These are tutorial "
                "metrics on a 12-image split (EVAL6): one holdout, no dispersion estimate. The report — with the zero-shot, "
                "majority and colour baselines and the deltas against them — is written to `outputs/{stem}_evaluation_report.json`.\n\n"
                "Read the delta against **zero-shot** first. On the sample it is zero, because the pretrained model already "
                "separates these four species perfectly and head-only adaptation preserves that; the value of adaptation shows on "
                "a dataset where zero-shot is *not* saturated — a local morph, a poorly named taxon, a camera-trap angle — which is "
                "what BYOD is for."
            ),
            "code": (
                "val_metrics = pipe.evaluate(val_records)\n"
                "test_metrics = pipe.evaluate(test_records)\n"
                "print({{'split': 'validation', **{{k: val_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "print({{'split': 'test', **{{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "for cls_name, row in test_metrics['per_class'].items():\n"
                "    print({{'class': cls_name, **row}})\n\n"
                "evaluation_report = {{\n"
                "    'task': 'species classification (bounded fine-tuning of a BioCLIP 2 head initialised from zero-shot)',\n"
                "    'evidence': 'tutorial sample-sanity metrics on one stratified holdout of 48 field photographs; not a benchmark',\n"
                "    'estimation': 'single train/validation/test split, seed ' + str(SEED) + ', no dispersion estimate',\n"
                "    'data_source': data_source,\n"
                "    'dataset_digest': dataset_manifest['digest'],\n"
                "    'classes': CLASSES,\n"
                "    'class_prompts': class_prompts,\n"
                "    'splits': {{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}},\n"
                "    'baselines': {{'zero_shot': baseline_zero_shot, 'majority': baseline_majority, 'color': baseline_color}},\n"
                "    'validation_metrics': val_metrics,\n"
                "    'test_metrics': test_metrics,\n"
                "    'delta_vs_zero_shot': {{k: round(test_metrics[k] - baseline_zero_shot[k], 4) for k in ('accuracy', 'macro_f1')}},\n"
                "    'delta_vs_majority': {{k: round(test_metrics[k] - baseline_majority[k], 4) for k in ('accuracy', 'macro_f1')}},\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k != 'trainable_parameter_names'}},\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "print({{'delta_vs_zero_shot': evaluation_report['delta_vs_zero_shot'], 'delta_vs_majority': evaluation_report['delta_vs_majority'], 'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 11. Inference on held-out images, artifact export and fresh reload\n\n"
                "`pipe.classify` returns, per image, the argmax `label`, its `score`, the full `scores` dictionary in class order "
                "and the image digest. The scores are softmax outputs of a head trained on a few dozen images — **not calibrated "
                "probabilities** (UNC2); the only decision rule is argmax (UNC3). The six inputs here are held-out test images the "
                "head never saw in training.\n\n"
                "`pipe.save_artifact` writes the trained tensors — with the default head-only setting, the 4x768 weight and the "
                "4-entry bias, about 12 KB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the "
                "base model id and revision, the class order, the tensor names, the file size and SHA-256, and the adaptation "
                "configuration including how the head was initialised (OUT8). `BioClip2Pipeline.from_artifact` re-verifies the "
                "base snapshot, checks the artifact manifest and digests **before** deserialising, rebuilds the classifier and "
                "overlays the tensors — a fresh object from files, not the in-memory model (VER2). The cell asserts identical "
                "labels and scores within `1e-5` (VER4)."
            ),
            "code": (
                "new_records = test_records[:6]\n"
                "new_source = 'six held-out test-split images (never used for training or monitoring)'\n"
                "inference_result = pipe.classify([r['image_bytes'] for r in new_records], names=[r['id'] for r in new_records])\n"
                "predictions = inference_result['predictions']\n"
                "print({{'new_source': new_source, 'decision_rule': inference_result['decision_rule']}})\n"
                "n_match = 0\n"
                "for p, r in zip(predictions, new_records):\n"
                "    n_match += p['label'] == r['label']\n"
                "    print({{'id': p['id'], 'predicted': p['label'], 'score': round(p['score'], 4), 'true_label': r['label']}})\n"
                "print({{'matches': n_match, 'of': len(new_records), 'note': 'sanity check on held-out images, not an evaluation'}})\n\n"
                "with open('outputs/{stem}_predictions.csv', 'w', encoding='utf-8', newline='') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['id', 'sha256', 'predicted_label', 'score'] + [f'score_{{c}}' for c in CLASSES])\n"
                "    for p in predictions:\n"
                "        writer.writerow([p['id'], p['sha256'], p['label'], f\"{{p['score']:.6f}}\"] + [f\"{{p['scores'][c]:.6f}}\" for c in CLASSES])\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "pipe.save_artifact(artifact_dir, metadata={{'data_source': data_source, 'dataset_digest': dataset_manifest['digest'], 'test_metrics': {{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "with open(artifact_dir / ARTIFACT_MANIFEST_NAME, encoding='utf-8') as f:\n"
                "    artifact_manifest = json.load(f)\n"
                "print({{'format': artifact_manifest['format'], 'base_model': artifact_manifest['base_model'], 'requires_remote_code': artifact_manifest['requires_remote_code'], 'n_tensors': len(artifact_manifest['tensors']), 'tensors': artifact_manifest['tensors'], 'files': artifact_manifest['files']}})\n\n"
                "reloaded_pipe = BioClip2Pipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR)\n"
                "reloaded_result = reloaded_pipe.classify([r['image_bytes'] for r in new_records], names=[r['id'] for r in new_records])\n"
                "max_score_diff = 0.0\n"
                "for before, after in zip(predictions, reloaded_result['predictions']):\n"
                "    assert before['id'] == after['id'] and before['label'] == after['label'], f'reload parity failure on {{before[\"id\"]}}'\n"
                "    max_score_diff = max(max_score_diff, abs(before['score'] - after['score']))\n"
                "assert max_score_diff < 1e-5, f'reload score drift {{max_score_diff}}'\n"
                "print({{'reload_parity': 'PASS', 'labels_equal': True, 'max_abs_score_diff': max_score_diff}})"
            ),
        },
        {
            "md": (
                "## 12. Result export and provenance\n\n"
                "The last output, `outputs/{stem}_result.json`, gathers what a reader needs to interpret the files above: the "
                "notebook source revision, the model id, immutable revision and licence, the dataset source, digest and (for the "
                "sample) its iNaturalist provenance summary, the class prompts, the adaptation configuration, the zero-shot and "
                "trivial baselines and held-out metrics, the held-out predictions, the artifact manifest, the reload-parity "
                "result, and the runtime versions and device (OUT6/OUT7). No credential is involved anywhere in this notebook, so "
                "none can leak into it (OUT10)."
            ),
            "code": (
                "import platform\n\n"
                "result_payload = {{\n"
                "    'task': 'species classification adaptation (BioCLIP 2)',\n"
                "    'pipeline_class': 'BioClip2Pipeline',\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'remote_code_executed': False,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'data_source': data_source,\n"
                "    'sample_provenance': None if USE_BYOD else sample_provenance(),\n"
                "    'dataset_manifest': dataset_manifest,\n"
                "    'class_prompts': class_prompts,\n"
                "    'embedding_summary': {{'n_images': embedding_result['n_images'], 'dimension': embedding_result['dimension'], 'pooling': embedding_result['pooling'], 'single_vs_batch_max_abs_diff': cross_batch}},\n"
                "    'evaluation_report': evaluation_report,\n"
                "    'inference': {{'new_source': new_source, 'decision_rule': inference_result['decision_rule'], 'predictions': predictions}},\n"
                "    'artifact_format': ARTIFACT_FORMAT,\n"
                "    'artifact_format_version': ARTIFACT_FORMAT_VERSION,\n"
                "    'artifact_manifest': artifact_manifest,\n"
                "    'reload_parity': {{'labels_equal': True, 'max_abs_score_diff': max_score_diff}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'open_clip': open_clip.__version__,\n"
                "        'safetensors': safetensors.__version__,\n"
                "        'device': pipe.device,\n"
                "        'precision': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The pretrained model separates four similarly coloured sparrow species from their scientific names alone, and a head "
        "fine-tuned from that starting point keeps the separation on an independent split. That is the claim: BioCLIP 2's joint "
        "space already encodes fine-grained taxonomic structure that colour does not explain, and the adaptation contract "
        "preserves it rather than destroying it. The test split has 12 photographs, the metrics come from one seeded holdout with "
        "no dispersion estimate, and the four species are common, well-photographed North American birds — the easy end of "
        "biodiversity. So a perfect score says the contract works on this sample, not that BioCLIP 2 identifies rare, cryptic or "
        "poorly represented taxa, and not that it reproduces any published benchmark.\n\n"
        "Three things to carry to real data. **Zero-shot first:** run `zero_shot_evaluate` on your labelled set before you "
        "train anything — if it is already saturated, adaptation has nothing to add, and if it is poor, check the prompt "
        "vocabulary (scientific versus common names, full taxonomic strings) before blaming the model. **Splits:** photographs of "
        "one individual, one photographer or one site leak across a random split; split by observation, site or photographer. "
        "**Unfreezing the tower:** with a few dozen images, even one unfrozen block degraded a perfect zero-shot classifier here; "
        "unfreeze only with hundreds of images per class and a validation split you trust.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model, build it from its pinned config, validate the demonstrated "
        "dataset contract, classify zero-shot, execute bounded fine-tuning, evaluate against trivial baselines on an independent "
        "split, and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish "
        "benchmark superiority, production fitness, or that any photograph shows what its label says.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_BLOCKS = 1` to watch the last tower block "
        "over-fit 28 images; change the prompt template in `pipe.zero_shot` (for example `'a photo of the bird {{}}.'`) and compare; "
        "or bring your own labelled set through BYOD with scientific names in `BYOD_CLASS_PROMPTS` and read the zero-shot and colour "
        "baselines first.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/bioclip2-biodiversity-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/bioclip2-biodiversity-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/Imageomics/bioclip-2\n"
        "- Gu, J., Stevens, S., Campolongo, E. G., et al. (2025). BioCLIP 2: Emergent Properties from Scaling Hierarchical "
        "Contrastive Learning. NeurIPS 2025. arXiv:2505.23883. https://arxiv.org/abs/2505.23883\n"
        "- Sample photographs: iNaturalist research-grade observations with CC0 1.0 photo licences; per-image observation URLs "
        "are recorded in the carried `sample_data` module."
    ),
}
