"""Contrôle de cohérence inter-zones d un corpus (au-delà de verif_dataset / verif_corpus) : classes, masques,
tailles par zone, splits, doublons d images, négatifs non vides, tuiles 648, annotations tronquées.
Usage : .venv/Scripts/python.exe tools/verif_coherence_corpus.py <corpus> <dossier_datasets>   (exit 1 = NON CONFORME)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

CORPUS = Path(sys.argv[1])
DATASETS = Path(sys.argv[2])
pb = []


def coco(p):
    return json.load(open(p, encoding="utf-8"))


def zone_of(img_name, index):
    return index.get(img_name, "?")


# --- index image -> zone/dataset depuis les datasets sources
index, ds_of = {}, {}
for d in sorted(DATASETS.glob("depressions_*_ld648_v1")):
    man = yaml.safe_load(open(d / "split_manifest.yaml", encoding="utf-8"))
    for split in ("train", "valid", "test"):
        c = coco(d / split / "_annotations.coco.json")
        for im in c["images"]:
            if im["file_name"] in index:
                pb.append(f"image en double entre datasets : {im['file_name']} ({index[im['file_name']]} et {man['zone']})")
            index[im["file_name"]] = man["zone"]
            ds_of[im["file_name"]] = d.name

print("== classes")
cats = {}
for split in ("train", "valid", "test"):
    c = coco(CORPUS / split / "_annotations.coco.json")
    cats[split] = [(k["id"], k["name"], k.get("supercategory")) for k in c["categories"]]
    print(f"  {split}: {cats[split]}")
if len({json.dumps(v) for v in cats.values()}) != 1:
    pb.append("catégories différentes entre splits")

print("== masques / boîtes / tailles par zone (px, tuiles 648)")
stats = defaultdict(lambda: dict(n=0, sans_seg=0, seg_bbox=0, aires=[], diams=[], tronquees=0, sommets=[]))
seen = Counter()
for split in ("train", "valid", "test"):
    c = coco(CORPUS / split / "_annotations.coco.json")
    ims = {im["id"]: im for im in c["images"]}
    for im in c["images"]:
        seen[im["file_name"]] += 1
        if (im["width"], im["height"]) != (648, 648):
            pb.append(f"{im['file_name']} : {im['width']}x{im['height']} px")
    for a in c["annotations"]:
        z = zone_of(ims[a["image_id"]]["file_name"], index)
        s = stats[z]
        s["n"] += 1
        seg = a.get("segmentation")
        if not seg or not seg[0] or len(seg[0]) < 6:
            s["sans_seg"] += 1
            continue
        xs, ys = seg[0][0::2], seg[0][1::2]
        # masque = rectangle exact de la boîte ?
        if len(seg[0]) in (8, 10) and {round(v, 1) for v in xs} <= {round(a["bbox"][0], 1), round(a["bbox"][0] + a["bbox"][2], 1)}:
            s["seg_bbox"] += 1
        s["aires"].append(a["area"])
        s["diams"].append((a["bbox"][2] + a["bbox"][3]) / 2)
        s["sommets"].append(len(seg[0]) // 2)
        x0, y0, w, h = a["bbox"]
        if x0 <= 0.5 or y0 <= 0.5 or x0 + w >= 647.5 or y0 + h >= 647.5:
            s["tronquees"] += 1
for z, s in stats.items():
    d = np.array(s["diams"])
    print(f"  {z:45s} n={s['n']:5d} sans_seg={s['sans_seg']} masque=boîte={s['seg_bbox']} "
          f"diam px p05/med/p95={np.percentile(d,5):.0f}/{np.median(d):.0f}/{np.percentile(d,95):.0f} "
          f"sommets med={int(np.median(s['sommets']))} tronquées(bord)={s['tronquees']} ({100*s['tronquees']/s['n']:.0f} %)")
    if s["sans_seg"]:
        pb.append(f"{z} : {s['sans_seg']} annotations sans masque")
dup = [k for k, v in seen.items() if v > 1]
if dup:
    pb.append(f"{len(dup)} images présentes dans plusieurs splits du corpus : {dup[:5]}")

print("== splits par zone (images) et blocs")
for d in sorted(DATASETS.glob("depressions_*_ld648_v1")):
    man = yaml.safe_load(open(d / "split_manifest.yaml", encoding="utf-8"))
    blocs = man.get("blocs") or man.get("blocks") or {}
    n = {s: len(coco(d / s / "_annotations.coco.json")["images"]) for s in ("train", "valid", "test")}
    src = man.get("split_depuis")
    print(f"  {d.name:45s} {n} split_depuis={Path(src).name if src else None} blocs={len(blocs)}")

print("== négatifs : contenu des tuiles sans annotation (moyenne / écart-type des pixels)")
for split in ("train", "valid", "test"):
    c = coco(CORPUS / split / "_annotations.coco.json")
    annotes = {a["image_id"] for a in c["annotations"]}
    vides, plats = 0, []
    for im in c["images"]:
        if im["id"] in annotes:
            continue
        vides += 1
        a = np.asarray(Image.open(CORPUS / split / im["file_name"]).convert("L"), dtype=np.float32)
        if a.std() < 2 or (a == 0).mean() > 0.5 or (a == 255).mean() > 0.5:
            plats.append(im["file_name"])
    print(f"  {split}: {vides} négatifs, {len(plats)} plats/vides {plats[:3]}")
    if plats:
        pb.append(f"{split} : {len(plats)} tuiles négatives plates ou vides")

print("== bandes / mode des images (échantillon)")
modes = Counter()
for split in ("train",):
    c = coco(CORPUS / split / "_annotations.coco.json")
    for im in c["images"][:50]:
        modes[Image.open(CORPUS / split / im["file_name"]).mode] += 1
print("  ", dict(modes))

print()
if pb:
    print("NON CONFORME :")
    for p in pb:
        print("  -", p)
    sys.exit(1)
print("CONFORME : cohérence inter-zones OK")
