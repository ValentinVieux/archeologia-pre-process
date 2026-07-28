"""Contrôleur indépendant du corpus multi-zones (boucle de vérification).

Usage : python verif_corpus.py <config.yaml> <dossier_des_datasets> <corpus>
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

config, dossier, corpus = (Path(a) for a in sys.argv[1:4])
cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
classes = list(cfg["classes"])
fusions = dict(cfg.get("fusions") or {})
SPLITS = ("train", "valid", "test")

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
    for nom_ds in cfg["datasets"]:
        racine = dossier / nom_ds
        coco_s = json.loads((racine / split / "_annotations.coco.json")
                            .read_text(encoding="utf-8"))
        noms_cats = {c["id"]: c["name"] for c in coco_s["categories"]}
        attendu_images += len(coco_s["images"])
        for a in coco_s["annotations"]:
            brut = noms_cats[a["category_id"]]
            attendu_annos[fusions.get(brut, brut)] += 1
        for im in coco_s["images"]:
            sha_sources[im["file_name"]] = racine / split / im["file_name"]

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
assert man["classes"] == classes and set(man["datasets"]) == set(cfg["datasets"])
par_zone = defaultdict(int)
for nom_ds, e in man["datasets"].items():
    for split in SPLITS:
        par_zone[e["zone"]] += e["splits"][split]["images"]
assert sum(par_zone.values()) == total_images

print(f"vérification corpus : CONFORME — {total_images} images, "
      f"{len(classes)} classes canoniques, zones : {dict(par_zone)}")
