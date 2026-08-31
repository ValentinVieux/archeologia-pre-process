"""Construit le corpus d'entraînement multi-zones : fusion de datasets découpés
(sorties de slice_zone) en un corpus COCO unique par split, classes canoniques
(fusions appliquées, ex. voie -> parcellaire), provenance de zone conservée sur
chaque image, splits spatiaux préservés tels quels.

Usage : python build_corpus.py <config.yaml> <dossier_des_datasets> [--out <dossier>]

Entrées de `datasets:` — un NOM (dataset entier) ou un mapping restreint :
    datasets:
      - enclos_fr_bretagne_ld648_v1                # entier
      - {nom: enclos_ie_sligo_ld648_v1,            # restreint (ex. corpus graduel)
         splits: [train], tuiles: [a.png, b.png]}
La restriction est tracée dans corpus_manifest.yaml ; la recette complète = la
config (commitée). Le manifeste porte aussi la provenance (date, outil+commit,
sha1 de la config, raster/gpkg/tuile_px des sources, sha1 des COCO produits).
"""
import argparse
import datetime as _dt
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slice_zone import _refuser_drive

SPLITS = ("train", "valid", "test")


def _git_head_court():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=Path(__file__).resolve().parents[1],
                           capture_output=True, text=True)
        return r.stdout.strip() or "?"
    except Exception:
        return "?"


def specs_datasets(cfg):
    """Normalise cfg['datasets'] : noms nus ET mappings restreints {nom, splits, tuiles}."""
    specs = []
    for entree in cfg["datasets"]:
        if isinstance(entree, str):
            specs.append({"nom": entree, "splits": None, "tuiles": None})
        elif isinstance(entree, dict) and entree.get("nom"):
            specs.append({"nom": entree["nom"],
                          "splits": list(entree["splits"]) if entree.get("splits") else None,
                          "tuiles": set(entree["tuiles"]) if entree.get("tuiles") else None})
        else:
            sys.exit(f"datasets : entrée invalide {entree!r} (nom nu ou mapping avec nom:)")
    return specs


def construire(config_path, dossier_datasets, out_dir):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    classes = list(cfg["classes"])
    fusions = dict(cfg.get("fusions") or {})
    for source, cible in fusions.items():
        if cible not in classes:
            sys.exit(f"fusion {source}->{cible} : cible absente de classes")
    dossier_datasets = Path(dossier_datasets)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    categories = [{"id": i + 1, "name": c, "supercategory": "none"}
                  for i, c in enumerate(classes)]
    cat_id = {c: i + 1 for i, c in enumerate(classes)}

    manifest = {"corpus": cfg["corpus"],
                "genere_le": _dt.date.today().isoformat(),
                "outil": "tools/build_corpus.py",
                "outil_commit": _git_head_court(),
                "config_sha1": hashlib.sha1(
                    Path(config_path).read_bytes()).hexdigest(),
                "classes": classes, "fusions": fusions,
                "datasets": {}, "splits": {}}
    for cle in ("gsd_m", "rvt", "notes"):
        if cfg.get(cle) is not None:
            manifest[cle] = cfg[cle]
    for split in SPLITS:
        coco_out = {"images": [], "annotations": [], "categories": categories}
        (out_dir / split).mkdir(parents=True)
        prochain_img, prochain_ann = 1, 1
        noms_vus = set()
        for spec in specs_datasets(cfg):
            nom_ds = spec["nom"]
            if spec["splits"] and split not in spec["splits"]:
                continue
            racine = dossier_datasets / nom_ds
            coco_p = racine / split / "_annotations.coco.json"
            if not coco_p.exists():
                sys.exit(f"{nom_ds}/{split} : _annotations.coco.json absent")
            split_man = yaml.safe_load((racine / "split_manifest.yaml")
                                       .read_text(encoding="utf-8"))
            zone = split_man.get("zone") or nom_ds
            coco = json.loads(coco_p.read_text(encoding="utf-8"))
            noms_cats = {c["id"]: c["name"] for c in coco["categories"]}
            remap_img = {}
            for im in coco["images"]:
                nom = im["file_name"]
                if spec["tuiles"] is not None and nom not in spec["tuiles"]:
                    continue
                if nom in noms_vus:
                    sys.exit(f"collision de nom d'image : {nom}")
                noms_vus.add(nom)
                remap_img[im["id"]] = prochain_img
                coco_out["images"].append({**im, "id": prochain_img,
                                           "zone": zone, "dataset": nom_ds})
                shutil.copy2(racine / split / nom, out_dir / split / nom)
                prochain_img += 1
            stats = Counter()
            for a in coco["annotations"]:
                if a["image_id"] not in remap_img:
                    continue  # image hors restriction (spec['tuiles'])
                brut = noms_cats[a["category_id"]]
                final = fusions.get(brut, brut)
                if final not in cat_id:
                    sys.exit(f"{nom_ds}/{split} : classe inattendue '{brut}'")
                coco_out["annotations"].append(
                    {**a, "id": prochain_ann, "image_id": remap_img[a["image_id"]],
                     "category_id": cat_id[final]})
                stats[final] += 1
                prochain_ann += 1
            cfg_src = split_man.get("config") or {}
            entree = manifest["datasets"].setdefault(
                nom_ds, {"zone": zone, "splits": {},
                         "split_manifest_sha1": hashlib.sha1(
                             (racine / "split_manifest.yaml")
                             .read_bytes()).hexdigest(),
                         # provenance amont, reprise du split_manifest (audit 2026-08-31)
                         "raster": cfg_src.get("raster"),
                         "gpkg": cfg_src.get("gpkg"),
                         "tuile_px": cfg_src.get("tuile_px"),
                         "genere_le_dataset": split_man.get("genere_le")})
            if spec["splits"] or spec["tuiles"] is not None:
                entree["restriction"] = {
                    "splits": spec["splits"],
                    "tuiles": (len(spec["tuiles"]) if spec["tuiles"] is not None else None)}
            entree["splits"][split] = {"images": len(remap_img),
                                       "annotations": dict(stats)}
        texte_coco = json.dumps(coco_out)
        (out_dir / split / "_annotations.coco.json").write_text(
            texte_coco, encoding="utf-8")
        manifest["splits"][split] = {
            "images": len(coco_out["images"]),
            "coco_sha1": hashlib.sha1(texte_coco.encode("utf-8")).hexdigest(),
            "annotations": dict(Counter(
                classes[a["category_id"] - 1]
                for a in coco_out["annotations"]))}
        print(f"{split} : {len(coco_out['images'])} images, "
              f"{len(coco_out['annotations'])} annotations")
    (out_dir / "corpus_manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("datasets")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else Path("corpus") / cfg["corpus"]
    for chemin, nom in ((args.datasets, "datasets"), (out, "--out")):
        _refuser_drive(chemin, nom)
    out = construire(args.config, args.datasets, out)
    print(f"Sorties :\n  {out}\n  {out / 'corpus_manifest.yaml'}")


if __name__ == "__main__":
    main()
