"""Freeze licensed Philippine bird photos without retaining sensitive API fields.

Run with Python, numpy, scipy, Pillow installed. No model is loaded. Discovery is
live; --audit-only verifies the already frozen files without reselection.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/data-cache"
AUDIT = ROOT / "outputs/dataset-audit"
SALT = "dimer-ph-birds-v1:"
VERSION = "1.0"
MAX_BYTES = 20_000_000
LICENCES = {
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
}
SPECIES = [
    ("philippine_pied_fantail", "Rhipidura nigritorquis", 339682),
    ("philippine_bulbul", "Hypsipetes philippinus", 144916),
    ("coleto", "Sarcops calvus", 15002),
    ("philippine_serpent_eagle", "Spilornis holospilus", 5160),
]
PROBE_SPECIES = [
    ("outside_bird_flowerpecker", "Dicaeum australe", "outside_candidate_bird"),
    ("outside_bird_sparrow", "Passer montanus", "outside_candidate_bird"),
    ("nonbird_butterfly", "Danaus chrysippus", "non_bird"),
    ("nonbird_snail", "Lissachatina fulica", "non_bird"),
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def fetch(url: str, ceiling: int = MAX_BYTES) -> bytes:
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "DIMER-educational-data-freezer/1.0"}
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read(ceiling + 1)
            if len(data) > ceiling:
                raise ValueError("asset exceeds byte ceiling")
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def api(route: str, **query: object) -> dict:
    url = "https://api.inaturalist.org/v1/" + route
    if query:
        url += "?" + urllib.parse.urlencode(query)
    return json.loads(fetch(url, 30_000_000))


def taxon_record(name: str, taxon_id: int | None = None) -> dict:
    results = (
        api(f"taxa/{taxon_id}")["results"] if taxon_id else api("taxa", q=name, rank="species")["results"]
    )
    exact = [r for r in results if r["name"] == name and r["rank"] == "species" and r.get("is_active", True)]
    if len(exact) != 1:
        raise ValueError(f"accepted exact taxonomy unresolved: {name}")
    row = exact[0]
    return {
        "taxon_id": row["id"],
        "scientific_name": row["name"],
        "common_name": row.get("preferred_common_name", row["name"]),
    }


def discover(label: str, taxon: dict) -> list[dict]:
    rows = []
    verified_infraspecies = {}
    for page in range(1, 31):
        result = api(
            "observations",
            place_id=6873,
            taxon_id=taxon["taxon_id"],
            quality_grade="research",
            photo_license="cc0,cc-by",
            photos="true",
            per_page=200,
            page=page,
            order_by="id",
            order="asc",
        )
        for observation in result["results"]:
            if observation.get("quality_grade") != "research" or observation.get("captive") is not False:
                continue
            observed_taxon = observation.get("taxon", {})
            exact = observed_taxon.get("id") == taxon["taxon_id"]
            if not exact:
                # A verified subspecies resolves unambiguously to the target
                # species; a genus-level or unrelated identification does not.
                if observed_taxon.get("rank") not in {"subspecies", "variety", "form"}:
                    continue
                observed_id = observed_taxon["id"]
                if observed_id not in verified_infraspecies:
                    verified_infraspecies[observed_id] = api(f"taxa/{observed_id}")["results"][0]
                verified = verified_infraspecies[observed_id]
                species_ancestors = [a for a in verified.get("ancestors", []) if a["rank"] == "species"]
                if not verified.get("is_active") or [a["id"] for a in species_ancestors] != [
                    taxon["taxon_id"]
                ]:
                    continue
            observer = observation.get("user", {}).get("login")
            if not observer:
                continue
            photos = [
                p
                for p in observation.get("photos", [])
                if p.get("license_code") in LICENCES and p.get("attribution") and not p.get("hidden")
            ]
            if not photos:
                continue
            photo = min(photos, key=lambda p: digest((SALT + str(p["id"])).encode()))
            url = photo["url"].replace("/square.", "/large.")
            if urllib.parse.urlsplit(url).hostname not in {
                "inaturalist-open-data.s3.amazonaws.com",
                "static.inaturalist.org",
            }:
                continue
            rows.append(
                {
                    "id": f"inat-{observation['id']}-{photo['id']}",
                    "label": label,
                    **taxon,
                    "observer": observer,
                    "observation_id": observation["id"],
                    "observation_url": f"https://www.inaturalist.org/observations/{observation['id']}",
                    "photo_id": photo["id"],
                    "url": url,
                    "license_code": photo["license_code"],
                    "license_url": LICENCES[photo["license_code"]],
                    "attribution": photo["attribution"],
                    "observed_on": observation.get("observed_on"),
                    "selection_rank": digest((SALT + str(photo["id"])).encode()),
                    "quality_grade": "research",
                    "wild": True,
                    "observed_taxon_id": observed_taxon["id"],
                    "observed_scientific_name": observed_taxon["name"],
                    "observed_taxon_rank": observed_taxon["rank"],
                    "taxonomic_resolution": "exact_species" if exact else "verified_subspecies_ancestor",
                    "species_ancestor_id": taxon["taxon_id"],
                }
            )
        if page * 200 >= result["total_results"]:
            break
    else:
        raise ValueError("query exceeds bounded pagination")
    return sorted(rows, key=lambda r: r["selection_rank"])


def perceptual_hash(image: Image.Image) -> str:
    grey = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = grey[:, 1:] > grey[:, :-1]
    return f"{int(''.join('1' if b else '0' for b in bits.ravel()), 2):016x}"


def materialise(row: dict) -> dict:
    path = CACHE / f"{row['id']}.img"
    data = path.read_bytes() if path.exists() else fetch(row["url"])
    if not data or len(data) > MAX_BYTES:
        raise ValueError("invalid image bytes")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.getexif().get(34853):
            raise ValueError("source image contains GPS EXIF")
        if image.format not in {"JPEG", "PNG", "WEBP"} or min(image.size) < 32 or max(image.size) > 8192:
            raise ValueError("unsupported image format or dimensions")
        upright = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(upright)
        if float(array.std()) < 1:
            raise ValueError("blank image")
        row = {
            **row,
            "sha256": digest(data),
            "bytes": len(data),
            "width": image.width,
            "height": image.height,
            "media_type": Image.MIME[image.format],
            "perceptual_hash": perceptual_hash(upright),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return row


def deduplicate(rows: list[dict], exclusions: list[dict]) -> list[dict]:
    accepted = []
    for row in sorted(rows, key=lambda r: r["selection_rank"]):
        duplicate = next(
            (
                a
                for a in accepted
                if row["observation_id"] == a["observation_id"]
                or row["photo_id"] == a["photo_id"]
                or row["sha256"] == a["sha256"]
                or (int(row["perceptual_hash"], 16) ^ int(a["perceptual_hash"], 16)).bit_count() <= 4
            ),
            None,
        )
        if duplicate:
            exclusions.append(
                {"id": row["id"], "reason": "duplicate_or_near_duplicate", "matching_id": duplicate["id"]}
            )
        else:
            accepted.append(row)
    return accepted


def choose_split(rows: list[dict], classes: list[dict]) -> tuple[list[dict], dict]:
    """Exact deterministic MILP: counts, group separation, <=3 and >=2 observers.

    Lexicographic objectives: maximum represented class-observer-role units,
    then hash-rank sum; HiGHS deterministic ordering resolves remaining ties.
    This records an explicit refinement of naive role-first hash selection,
    which can consume three same-observer validation photos and violate the
    two-observer rule despite available capacity.
    """
    observers = sorted({r["observer"] for r in rows})
    roles = ["train", "validation", "test"]
    targets = [8, 3, 3]
    labels = [c["label"] for c in classes]
    # x(record, role), y(observer, role), z(class, observer, role).
    variables = [("x", i, s) for i in range(len(rows)) for s in range(3)]
    variables += [("y", o, s) for o in observers for s in range(3)]
    pairs = sorted({(r["label"], r["observer"]) for r in rows})
    variables += [("z", (c, o), s) for c, o in pairs for s in range(3)]
    ix = {key: i for i, key in enumerate(variables)}
    constraints, lower, upper = [], [], []

    def add(terms: dict, lo: float, hi: float) -> None:
        constraints.append({ix[k]: v for k, v in terms.items()})
        lower.append(lo)
        upper.append(hi)

    for i, row in enumerate(rows):
        add({("x", i, s): 1 for s in range(3)}, 0, 1)
        for s in range(3):
            add({("x", i, s): 1, ("y", row["observer"], s): -1}, -np.inf, 0)
    for o in observers:
        add({("y", o, s): 1 for s in range(3)}, 0, 1)
    for c in labels:
        for s, count in enumerate(targets):
            add({("x", i, s): 1 for i, r in enumerate(rows) if r["label"] == c}, count, count)
            add({("z", (cc, o), s): 1 for cc, o in pairs if cc == c}, 2, np.inf)
    for c, o in pairs:
        for s in range(3):
            terms = {("x", i, s): 1 for i, r in enumerate(rows) if r["label"] == c and r["observer"] == o}
            add({**terms, ("z", (c, o), s): -3}, -np.inf, 0)
            add({**terms, ("z", (c, o), s): -1}, 0, np.inf)
    matrix = lil_matrix((len(constraints), len(variables)), dtype=float)
    for j, terms in enumerate(constraints):
        for i, value in terms.items():
            matrix[j, i] = value
    objective = np.array([-1.0 if v[0] == "z" else 0 for v in variables])
    limits = LinearConstraint(matrix.tocsr(), lower, upper)
    options = {"time_limit": 90, "mip_rel_gap": 0.0}
    first = milp(
        objective,
        integrality=np.ones(len(variables)),
        bounds=Bounds(0, 1),
        constraints=limits,
        options=options,
    )
    if not first.success:
        raise ValueError(f"split capacity gate failed: {first.message}")
    maximum = int(round(-first.fun))
    fixed = LinearConstraint(objective[None, :], -maximum, -maximum)
    second_objective = np.array(
        [
            (v[1] + 1)
            if v[0] == "x"
            else (observers.index(v[1]) + 1) * (v[2] + 1) / 100000
            if v[0] == "y"
            else 0
            for v in variables
        ],
        dtype=float,
    )
    second = milp(
        second_objective,
        integrality=np.ones(len(variables)),
        bounds=Bounds(0, 1),
        constraints=[limits, fixed],
        options=options,
    )
    if not second.success:
        raise ValueError(f"split rank optimisation failed: {second.message}")
    chosen = [
        {**rows[i], "split": roles[s]}
        for i in range(len(rows))
        for s in range(3)
        if second.x[ix[("x", i, s)]] > 0.5
    ]
    proof = {
        "algorithm": "scipy.optimize.milp / HiGHS",
        "optimal": True,
        "max_class_observer_role_units": maximum,
        "secondary_rank_objective": second.fun,
        "selection_refinement": (
            "Joint record/group optimisation guarantees exact counts and maximises represented photographers "
            "before minimum hash-rank sum; deterministic sorted variables, HiGHS tie handling. "
            "This replaces role-first greedy truncation, which may violate the two-observer minimum. "
            "It does not assert lexical optimality among equal rank-sum assignments."
        ),
    }
    return sorted(chosen, key=lambda r: (r["label"], roles.index(r["split"]), r["selection_rank"])), proof


def validate(rows: list[dict], classes: list[dict]) -> dict:
    summary = {}
    for key in ["id", "photo_id", "observation_id", "sha256"]:
        assert len({r[key] for r in rows}) == len(rows), key
    role_sets = [{r["observer"] for r in rows if r["split"] == s} for s in ["train", "validation", "test"]]
    assert not any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i))
    for c in classes:
        summary[c["label"]] = {}
        for role, expected in [("train", 8), ("validation", 3), ("test", 3)]:
            subset = [r for r in rows if r["label"] == c["label"] and r["split"] == role]
            counts = {o: sum(r["observer"] == o for r in subset) for o in {r["observer"] for r in subset}}
            assert len(subset) == expected and len(counts) >= 2 and max(counts.values()) <= 3
            summary[c["label"]][role] = {"images": len(subset), "observers": len(counts)}
    for i, row in enumerate(rows):
        assert row["license_code"] in LICENCES and row["attribution"]
        data = (CACHE / f"{row['id']}.img").read_bytes()
        assert len(data) == row["bytes"] and digest(data) == row["sha256"]
        assert all(
            (int(row["perceptual_hash"], 16) ^ int(other["perceptual_hash"], 16)).bit_count() > 4
            for other in rows[:i]
        )
    return summary


def contact_sheet(rows: list[dict], name: str) -> None:
    tile_w, tile_h = 300, 360
    sheet = Image.new("RGB", (tile_w * 4, tile_h * ((len(rows) + 3) // 4)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x, y = index % 4 * tile_w, index // 4 * tile_h
        with Image.open(CACHE / f"{row['id']}.img") as image:
            image = ImageOps.fit(
                ImageOps.exif_transpose(image).convert("RGB"), (280, 280), method=Image.Resampling.BICUBIC
            )
        sheet.paste(image, (x + 10, y))
        text = (
            f"{row['id']}\n{row['scientific_name']}\n"
            f"{row.get('split', 'probe')} | {row['observer']} | {row['license_code']}"
        )
        draw.multiline_text((x + 8, y + 283), text, fill="black", spacing=4)
    sheet.save(AUDIT / name)


def blank_probe() -> dict:
    stream = io.BytesIO()
    Image.new("RGB", (224, 224), (128, 128, 128)).save(stream, format="PNG")
    payload = stream.getvalue()
    row = {
        "id": "blank-control",
        "synthetic": "blank",
        "label": "blank_control",
        "scientific_name": "Not applicable",
        "common_name": "Constant gray control",
        "observer": "DIMER",
        "split": "probe",
        "probe_type": "blank",
        "taxon_id": None,
        "photo_id": None,
        "observation_id": None,
        "observation_url": "",
        "url": "",
        "observed_on": None,
        "sha256": digest(payload),
        "bytes": len(payload),
        "width": 224,
        "height": 224,
        "media_type": "image/png",
        "license_code": "cc0",
        "license_url": LICENCES["cc0"],
        "attribution": "DIMER generated constant gray control (CC0)",
        "perceptual_hash": "0000000000000000",
    }
    (CACHE / "blank-control.img").write_bytes(payload)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--review-complete",
        action="store_true",
        help="Record composition-only visual review after all contact sheets were inspected",
    )
    args = parser.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "tools/biodiversity_data.json"
    if args.review_complete:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("manifest_sha256")
        manifest["probes"] = [p for p in manifest["probes"] if p.get("synthetic") != "blank"] + [
            blank_probe()
        ]
        manifest.pop("generated_probe", None)
        manifest["visual_review"] = {
            "status": "composition_only_reviewed",
            "scope": (
                "Fixed centre-crop contact sheets visually inspected; downloaded originals machine decoded "
                "and hash checked. No manual full-frame inspection or model predictions consulted."
            ),
            "limitations": "Not expert taxonomic verification. Difficult/small birds retained.",
            "crop_limitations": [
                {
                    "observation_id": 217651776,
                    "note": "Fantail small and difficult to locate in centre crop.",
                },
                {"observation_id": 339451045, "note": "Outside-species sparrow small in centre crop."},
            ],
            "maintainer_attribution_review": "pending",
        }
        manifest["manifest_sha256"] = digest(canonical(manifest))
        write_json(manifest_path, manifest)
        contact_sheet(manifest["probes"], "probes.jpg")
        summary = validate(manifest["records"], manifest["classes"])
        write_json(
            ROOT / "outputs/birds_data_return.json",
            {
                "status": "frozen_composition_reviewed",
                "records": len(manifest["records"]),
                "probes": len(manifest["probes"]),
                "manifest_sha256": manifest["manifest_sha256"],
                "bytes": sum(r["bytes"] for r in manifest["records"] + manifest["probes"]),
                "split_summary": summary,
                "selection_proof": manifest["selection_proof"],
                "model_inference_performed": False,
            },
        )
        return
    if args.audit_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.pop("manifest_sha256")
        assert digest(canonical(manifest)) == expected
        summary = validate(manifest["records"], manifest["classes"])
        write_json(AUDIT / "audit.json", summary)
        print(json.dumps(summary, indent=2))
        return
    exclusions_path = ROOT / "tools/biodiversity_exclusions.json"
    prior = (
        json.loads(exclusions_path.read_text(encoding="utf-8"))
        if exclusions_path.exists()
        else {"records": []}
    )
    manual = [r for r in prior["records"] if r.get("review_type") == "manual"]
    exclusions = manual.copy()
    rejected = {r["id"] for r in manual}
    classes, rows, discovery = [], [], {}
    for label, name, taxon_id in SPECIES:
        taxon = taxon_record(name, taxon_id)
        classes.append({"label": label, **taxon})
        candidates = discover(label, taxon)
        discovery[label] = {
            "eligible_observations": len(candidates),
            "eligible_observers": len({r["observer"] for r in candidates}),
        }
        if len(candidates) < 14 or discovery[label]["eligible_observers"] < 8:
            raise ValueError(f"feasibility gate failed: {label}: {discovery[label]}")
        rows.extend(r for r in candidates if r["id"] not in rejected)
        print(label, discovery[label], flush=True)
    write_json(AUDIT / "discovery.json", discovery)
    usable = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [(r, pool.submit(materialise, r)) for r in rows]
        for row, job in jobs:
            try:
                usable.append(job.result())
            except Exception as error:
                exclusions.append(
                    {"id": row["id"], "reason": "decode_or_download_failure", "detail": str(error)[:150]}
                )
    usable = deduplicate(usable, exclusions)
    write_json(AUDIT / "eligible_candidates.json", usable)
    write_json(exclusions_path, {"script_version": VERSION, "records": exclusions})
    chosen, proof = choose_split(usable, classes)
    probes = []
    for label, name, kind in PROBE_SPECIES:
        taxon = taxon_record(name)
        candidates = discover(label, taxon)
        for row in candidates:
            if row["id"] in rejected:
                continue
            try:
                materialised = materialise(row)
                if len(deduplicate(chosen + probes + [materialised], [])) != len(chosen + probes) + 1:
                    continue
                probes.append({**materialised, "split": "probe", "probe_type": kind})
                break
            except Exception:
                continue
        else:
            raise ValueError(f"no usable probe: {name}")
    exclusion_record = {"script_version": VERSION, "records": sorted(exclusions, key=lambda r: r["id"])}
    write_json(exclusions_path, exclusion_record)
    probes.append(blank_probe())
    manifest = {
        "name": "Philippine Endemic Bird Field-Photo Sample v1",
        "version": VERSION,
        "frozen_at": datetime.now(UTC).isoformat(),
        "script_version": VERSION,
        "source": "iNaturalist",
        "place_id": 6873,
        "country": "Philippines",
        "query": {
            "endpoint": "https://api.inaturalist.org/v1/observations",
            "place_id": 6873,
            "quality_grade": "research",
            "photo_license": "cc0,cc-by",
            "photos": True,
            "order_by": "id",
            "order": "asc",
        },
        "selection_salt": SALT,
        "duplicate_policy": "64-bit dHash Hamming <=4 excluded globally before selection",
        "license_note": (
            "Individual photos retain CC0 or CC BY licences and creator attribution; "
            "no source photos redistributed in results archive."
        ),
        "exclusions_sha256": digest(exclusions_path.read_bytes()),
        "classes": classes,
        "discovery": discovery,
        "selection_proof": proof,
        "records": chosen,
        "probes": probes,
        "pretraining_overlap": (
            "Unresolved for BioCLIP 2 and SigLIP 2; split isolates only capstone fitting and selection."
        ),
        "visual_review": (
            "pending: centre-crop contact sheets must be inspected before maintainer qualification"
        ),
    }
    manifest["manifest_sha256"] = digest(canonical(manifest))
    assert sum(r["bytes"] for r in chosen + probes) <= 250_000_000
    write_json(manifest_path, manifest)
    summary = validate(chosen, classes)
    write_json(AUDIT / "audit.json", summary)
    for c in classes:
        contact_sheet([r for r in chosen if r["label"] == c["label"]], c["label"] + ".jpg")
    contact_sheet(probes, "probes.jpg")
    write_json(
        ROOT / "outputs/birds_data_return.json",
        {
            "status": "frozen_pending_visual_review",
            "records": len(chosen),
            "probes": len(probes),
            "manifest_sha256": manifest["manifest_sha256"],
            "bytes": sum(r["bytes"] for r in chosen + probes),
            "split_summary": summary,
            "selection_proof": proof,
            "exclusions": len(exclusions),
        },
    )
    print(json.dumps({"selected": len(chosen), "probes": len(probes), "proof": proof}, indent=2))


if __name__ == "__main__":
    main()
