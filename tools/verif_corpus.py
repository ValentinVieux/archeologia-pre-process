"""Contrôleur indépendant du corpus multi-zones (boucle de vérification).

Usage : python verif_corpus.py <config.yaml> <dossier_des_datasets> <corpus>
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("config", help="configs/corpus_*.yaml (recette du corpus)")
_ap.add_argument("dossier", help="dossier des datasets sources (ex. datasets)")
_ap.add_argument("corpus", help="dossier du corpus construit par build_corpus")
_a = _ap.parse_args()
config, dossier, corpus = Path(_a.config), Path(_a.dossier), Path(_a.corpus)
cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
classes = list(cfg["classes"])
fusions = dict(cfg.get("fusions") or {})
SPLITS = ("train", "valid", "test")

# normalisation INDÉPENDANTE des entrées datasets (noms nus + mappings restreints)
specs = []
for _e in cfg["datasets"]:
    if isinstance(_e, str):
        specs.append((_e, None, None))
    else:
        specs.append((_e["nom"],
                      set(_e["splits"]) if _e.get("splits") else None,
                      set(_e["tuiles"]) if _e.get("tuiles") else None))

total_images = 0
for split in SPLITS:
    coco_c = json.loads((corpus / split / "_annotations.coco.json")
                        .read_text(encoding="utf-8"))
    assert [c["name"] for c in coco_c["categories"]] == classes, \
        f"{split} : catégories {[c['name'] for c in coco_c['categories']]}"
    cats_c = {c["id"]: c["name"] for c in coco_c["categories"]}

    # attendu recalculé depuis les sources
    attendu_images = 0
    attendu_annos = Counter()
    sha_sources = {}
    for nom_ds, splits_ds, tuiles_ds in specs:
        if splits_ds and split not in splits_ds:
            continue
        racine = dossier / nom_ds
        coco_s = json.loads((racine / split / "_annotations.coco.json")
                            .read_text(encoding="utf-8"))
        noms_cats = {c["id"]: c["name"] for c in coco_s["categories"]}
        ids_retenus = set()
        for im in coco_s["images"]:
            if tuiles_ds is not None and im["file_name"] not in tuiles_ds:
                continue
            ids_retenus.add(im["id"])
            attendu_images += 1
            sha_sources[im["file_name"]] = racine / split / im["file_name"]
        for a in coco_s["annotations"]:
            if a["image_id"] not in ids_retenus:
                continue
            brut = noms_cats[a["category_id"]]
            attendu_annos[fusions.get(brut, brut)] += 1

    assert len(coco_c["images"]) == attendu_images, \
        f"{split} : {len(coco_c['images'])} images vs {attendu_images}"
    noms = [im["file_name"] for im in coco_c["images"]]
    assert len(noms) == len(set(noms)), f"{split} : noms d'images en double"
    assert all("zone" in im and "dataset" in im for im in coco_c["images"]), \
        f"{split} : provenance absente"
    obtenu = Counter(cats_c[a["category_id"]] for a in coco_c["annotations"])
    assert obtenu == attendu_annos, \
        f"{split} : annotations {dict(obtenu)} vs {dict(attendu_annos)}"

    # ids d'annotations -> images existantes, et bornes de catégories
    ids_img = {im["id"] for im in coco_c["images"]}
    assert all(a["image_id"] in ids_img for a in coco_c["annotations"]), \
        f"{split} : annotation orpheline"

    # intégrité des fichiers : tailles pour tous, sha1 sur échantillon
    for im in coco_c["images"]:
        src = sha_sources[im["file_name"]]
        dst = corpus / split / im["file_name"]
        assert dst.exists() and dst.stat().st_size == src.stat().st_size, \
            f"{split}/{im['file_name']} : fichier absent ou taille divergente"
    import random
    for im in random.Random(42).sample(coco_c["images"],
                                       min(60, len(coco_c["images"]))):
        src = sha_sources[im["file_name"]]
        dst = corpus / split / im["file_name"]
        assert (hashlib.sha1(dst.read_bytes()).hexdigest()
                == hashlib.sha1(src.read_bytes()).hexdigest()), \
            f"{split}/{im['file_name']} : contenu divergent"
    total_images += len(coco_c["images"])

man = yaml.safe_load((corpus / "corpus_manifest.yaml").read_text(encoding="utf-8"))
assert man["classes"] == classes and set(man["datasets"]) == {s[0] for s in specs}
par_zone = defaultdict(int)
for nom_ds, e in man["datasets"].items():
    for split in SPLITS:
        par_zone[e["zone"]] += e["splits"].get(split, {}).get("images", 0)
assert sum(par_zone.values()) == total_images
# sha1 des COCO publiés dans le manifeste (garde anti-fuite : contenu figé)
for split in SPLITS:
    sha_pub = man["splits"][split].get("coco_sha1")
    if sha_pub:
        reel = hashlib.sha1((corpus / split / "_annotations.coco.json")
                            .read_bytes()).hexdigest()
        assert sha_pub == reel, f"{split} : coco_sha1 du manifeste != contenu réel"

print(f"vérification corpus : CONFORME — {total_images} images, "
      f"{len(classes)} classes canoniques, zones : {dict(par_zone)}")
