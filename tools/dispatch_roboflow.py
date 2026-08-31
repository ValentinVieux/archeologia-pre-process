"""Dispatch des exports Roboflow (COCO zips) vers l'arborescence data_regions_v2.

Usage :
    .venv\\Scripts\\python.exe tools\\dispatch_roboflow.py <attribution.json> <dossier_zips> <staging_dir>

- attribution.json : produit par l'analyse d'attribution (image -> zone par dataset).
- Écrit dans <staging_dir> (local, rapide) : à copier ensuite vers le Drive (robocopy).
- Ne modifie JAMAIS les zips sources. Applique les résolutions d'ambiguïté documentées.

Structure produite par zone :
    <region>/<zone_id>/training/roboflow/<dataset>/{train,valid,test}/  (+ _annotations.coco.json filtré)
    <region>/<zone_id>/training/roboflow/<dataset>/upload_manifest.yaml (filename -> tags, prêt pour ré-upload)
    _a_trier/<dataset>/...  pour les images inattribuables.
"""
from __future__ import annotations

import collections
import json
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

import yaml

REGION_OF_ZONE = {
    "55_verdun": "grand_est",
    "57_fenetrange": "grand_est",
    "54_foret_de_haye": "grand_est",
    "70_vosges_saonoises": "bourgogne_franche_comte",
    "25_foret_de_chailluz": "bourgogne_franche_comte",
    "25_haut_doubs": "bourgogne_franche_comte",
    "xx_mont_de_la_croix": "bourgogne_franche_comte",
    "30_ales_garrigues_ne": "occitanie",
    "30_la_capelle_et_masmolene": "occitanie",
    "78_rambouillet": "ile_de_france",
    "77_fontainebleau": "ile_de_france",
    "78_saint_germain_marly": "ile_de_france",
    "77_bataille_de_la_marne": "ile_de_france",
    "28_dreux": "centre_val_de_loire",
    "41_blois": "centre_val_de_loire",
    "42_hautes_chaumes_forez": "auvergne_rhone_alpes",
    "44_loire_atlantique_2020": "pays_de_la_loire",
    "35_anomalie_lidar_bretagne": "bretagne",
}

LHD_RE = re.compile(r"LHD_FXX_(\d{4})_(\d{4})")

# Boîtes englobantes étendues (km Lambert-93) pour rattacher les dalles hors index exact.
# Justification : la campagne « 78 » (output 78_foret_domaine_regional) dépasse la forêt de
# Rambouillet ; les datasets d'Alès (HG+G) couvrent un large quart NE du Gard.
EXTENDED_BBOX = {
    "78_rambouillet": (595, 630, 6820, 6875),
    "30_ales_garrigues_ne": (770, 860, 6290, 6380),
}


def resolve(zone: str | None, base: str) -> tuple[str | None, str]:
    """Applique les règles de résolution -> (zone_finale|None, methode)."""
    if zone and not zone.startswith("AMBIGU:"):
        return zone, "index_dalles"
    if zone and zone.startswith("AMBIGU:"):
        zones = zone[7:].split("|")
        if set(zones) == {"30_ales_garrigues_ne", "30_la_capelle_et_masmolene"}:
            return "30_ales_garrigues_ne", "ambigu_resolu_campagne_ales"
        return None, "ambigu_non_resolu"
    m = LHD_RE.search(base)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        hits = [z for z, (x0, x1, y0, y1) in EXTENDED_BBOX.items()
                if x0 <= x <= x1 and y0 <= y <= y1]
        if len(hits) == 1:
            return hits[0], "bbox_etendue"
    return None, "inattribuable"


def filter_coco(coco: dict, keep: set[str]) -> dict:
    imgs = [i for i in coco.get("images", []) if i["file_name"] in keep]
    ids = {i["id"] for i in imgs}
    return {
        **{k: v for k, v in coco.items() if k not in ("images", "annotations")},
        "images": imgs,
        "annotations": [a for a in coco.get("annotations", []) if a["image_id"] in ids],
    }


def main() -> None:
    attribution_path, zips_dir, staging = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))

    for slug, data in attribution.items():
        zpath = zips_dir / data["zip"]
        print(f"=== {slug} ({zpath.name})")
        # zone finale par image
        dest: dict[str, tuple[str | None, str]] = {}
        for name, zone in data["images"].items():
            base = name.rsplit("/", 1)[-1]
            dest[name] = resolve(zone, base)

        per_zone_split: dict = collections.defaultdict(lambda: collections.defaultdict(set))
        methods: dict = collections.defaultdict(collections.Counter)
        for name, (zone, method) in dest.items():
            split = name.split("/")[0]
            per_zone_split[zone][split].add(name.rsplit("/", 1)[-1])
            methods[zone][method] += 1

        with zipfile.ZipFile(zpath) as z:
            cocos = {}
            for n in z.namelist():
                if n.endswith("_annotations.coco.json"):
                    cocos[n.split("/")[0]] = json.loads(z.read(n))

            for zone, splits in per_zone_split.items():
                if zone is None:
                    zdir = staging / "_a_trier" / slug
                else:
                    zdir = staging / REGION_OF_ZONE[zone] / zone / "training" / "roboflow" / slug
                manifest_rows = []
                for split, keep in sorted(splits.items()):
                    outdir = zdir / split
                    outdir.mkdir(parents=True, exist_ok=True)
                    coco = filter_coco(cocos[split], keep)
                    (outdir / "_annotations.coco.json").write_text(
                        json.dumps(coco, ensure_ascii=False), encoding="utf-8")
                    for fname in sorted(keep):
                        target = outdir / fname
                        if not target.exists():
                            with z.open(f"{split}/{fname}") as src:
                                target.write_bytes(src.read())
                        manifest_rows.append({"filename": fname, "split": split})
                    print(f"    {zone or '_a_trier'} / {split}: {len(keep)} images, "
                          f"{len(coco['annotations'])} annotations")
                tags = [zone, REGION_OF_ZONE[zone]] if zone else []
                (zdir / "upload_manifest.yaml").write_text(yaml.safe_dump({
                    "source_export": data["zip"],
                    "dataset": slug,
                    "dispatched": str(date.today()),
                    "zone": zone,
                    "region": REGION_OF_ZONE.get(zone),
                    "tags": tags,
                    "attribution_methods": dict(methods[zone]),
                    "images": manifest_rows,
                }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("Dispatch termine ->", staging)


if __name__ == "__main__":
    main()
