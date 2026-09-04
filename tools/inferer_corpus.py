"""Inférence d'un modèle RF-DETR (seg ou détection) sur les tuiles d'un corpus COCO.

Produit les détections au-dessus d'un plancher, étiquetées `retenu` quand le score
atteint le seuil F1-max par classe du metriques_eval.json canonique (source unique
des seuils, règle 2026-08-31). Sorties dans <out> :

  detections.json  — détections en coordonnées PIXEL par tuile (consommé par
                     tools/review_detections, l'app locale de revue humaine)
  detections.gpkg  — mêmes détections géoréférencées EPSG:2154 (couche `detections`,
                     géométrie = contour du masque prédit, ou bbox en détection)
  resume.json      — provenance (poids, seuils, plancher) + comptes par split/classe

Géoréférencement : nom de tuile <dataset>_rRRRR_cCCCC.png + grille du manifest
versionné manifests/split/<dataset>.yaml (fenêtre pixel (col*tuile, row*tuile)
depuis grille.origine, axe y vers le bas) — même convention que tools/slice_zone.py.

Le plancher est volontairement SOUS les seuils F1-max (défaut 0,15) : les zones à
scores écrasés (Chailluz, cf. mémoire chantier) restent visibles dans la revue via
retenu=false. venv GPU requis (rfdetr/torch) : D:\\veille_irlande\\venv_adaf.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import yaml

RE_TUILE = re.compile(r"_r(\d+)_c(\d+)\.png$")


def iou_bbox(a, b):
    # même définition que tools/courbes_eval.py (xyxy)
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def charger_seuils(chemin):
    with open(chemin, encoding="utf-8") as f:
        me = json.load(f)
    modeles = me.get("modeles", {})
    if len(modeles) != 1:
        sys.exit(f"ERREUR : {chemin} porte {len(modeles)} modèles — préciser un "
                 "metriques_eval.json à modèle unique (celui du run évalué).")
    nom, m = next(iter(modeles.items()))
    seuils = {c: v["seuil_f1max"] for c, v in m["par_classe"].items()}
    return nom, seuils, m["global"]["seuil_f1max"], me


def masque_vers_poly_px(masque):
    """Plus grande composante du masque bool -> liste [[x,y],...] simplifiée (px)."""
    from rasterio import features
    from shapely.geometry import shape
    meilleur = None
    for geom, val in features.shapes(masque.astype(np.uint8), mask=masque):
        p = shape(geom)
        if meilleur is None or p.area > meilleur.area:
            meilleur = p
    if meilleur is None:
        return None
    p = meilleur.simplify(0.75)
    return [[round(x, 1), round(y, 1)] for x, y in p.exterior.coords]


def grille_dataset(dossier_manifests, dataset, cache={}):
    if dataset not in cache:
        chemin = os.path.join(dossier_manifests, f"{dataset}.yaml")
        with open(chemin, encoding="utf-8") as f:
            man = yaml.safe_load(f)
        g = man["grille"]
        cache[dataset] = (g["origine"][0], g["origine"][1],
                         g["gsd_m_px"][0], g["gsd_m_px"][1], g["tuile_px"])
    return cache[dataset]


def px_vers_l93(pts, grille, row, col):
    ox, oy, gx, gy, tuile = grille
    x0, y0 = ox + col * tuile * gx, oy - row * tuile * gy
    return [(x0 + x * gx, y0 - y * gy) for x, y in pts]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus", help="dossier corpus (train/valid/test + _annotations.coco.json)")
    ap.add_argument("poids", help="checkpoint .pth du modèle")
    ap.add_argument("out", help="dossier de sortie")
    ap.add_argument("--metriques", required=True,
                    help="metriques_eval.json canonique (source des seuils F1-max par classe)")
    ap.add_argument("--plancher", type=float, default=0.15,
                    help="score minimal conservé (défaut 0,15 ; retenu = seuil F1-max classe)")
    ap.add_argument("--resolution", type=int, default=648)
    ap.add_argument("--manifests", default=os.path.join(os.path.dirname(__file__), "..",
                                                        "manifests", "split"),
                    help="dossier des manifests de split versionnés (grilles de géoréf)")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--limite", type=int, default=0,
                    help="debug : n images max par split (0 = toutes)")
    args = ap.parse_args()

    from PIL import Image
    from pycocotools.coco import COCO
    from rfdetr import RFDETR

    nom_modele, seuils, seuil_global, me = charger_seuils(args.metriques)
    os.makedirs(args.out, exist_ok=True)

    modele = RFDETR.from_checkpoint(args.poids, resolution=args.resolution)
    seg = bool(getattr(modele.model_config, "segmentation_head", False))
    modele.optimize_for_inference()
    print(f"{nom_modele} : tâche {'segmentation' if seg else 'détection'}, "
          f"résolution {args.resolution}, plancher {args.plancher}, seuils {seuils}")

    tuiles, lignes_gpkg, decal = {}, [], None
    compte = {"images": 0, "detections": 0, "retenues": 0}
    for split in args.splits:
        dossier = os.path.join(args.corpus, split)
        coco = COCO(os.path.join(dossier, "_annotations.coco.json"))
        cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        noms = [c["name"] for c in coco.loadCats(sorted(coco.getCatIds()))]
        ids = coco.getImgIds()[:args.limite or None]
        for n_fait, img_id in enumerate(ids):
            info = coco.loadImgs(img_id)[0]
            m = RE_TUILE.search(info["file_name"])
            if not m:
                sys.exit(f"ERREUR : nom de tuile inattendu {info['file_name']}")
            row, col = int(m.group(1)), int(m.group(2))
            grille = grille_dataset(args.manifests, info["dataset"])
            im = Image.open(os.path.join(dossier, info["file_name"])).convert("RGB")
            d = modele.predict(im, threshold=args.plancher)
            if seg and len(d) and d.mask is None:
                sys.exit("ERREUR : modèle segmentation sans masques prédits.")
            if decal is None and len(d):
                decal = 1 if (int(d.class_id.min()) >= 1
                              and int(d.class_id.max()) >= len(noms)) else 0
            gts = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            gt_boxes = [(cats[a["category_id"]],
                         (a["bbox"][0], a["bbox"][1],
                          a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]))
                        for a in gts]
            pris = [False] * len(gt_boxes)
            dets = []
            for i in (np.argsort(-d.confidence) if len(d) else []):
                idx = int(d.class_id[i]) - (decal or 0)
                if not (0 <= idx < len(noms)):
                    continue
                classe = noms[idx]
                score = float(d.confidence[i])
                bbox = [round(float(v), 1) for v in d.xyxy[i]]
                poly = masque_vers_poly_px(d.mask[i].astype(bool)) if seg else None
                if poly is None:
                    poly = [[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                            [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]
                meilleur, mi = 0.0, -1
                for j, (cg, bg) in enumerate(gt_boxes):
                    if pris[j] or cg != classe:
                        continue
                    iou = iou_bbox(tuple(bbox), bg)
                    if iou > meilleur:
                        meilleur, mi = iou, j
                if meilleur >= 0.5:
                    pris[mi] = True
                uid = f"{split}:{info['file_name']}:{len(dets)}"
                seuil = seuils.get(classe, seuil_global)
                det = {"uid": uid, "classe": classe, "score": round(score, 4),
                       "retenu": score >= seuil,
                       "bbox_px": bbox, "poly_px": poly,
                       "gt_apparie": meilleur >= 0.5,
                       "iou_gt": round(meilleur, 3) if meilleur >= 0.5 else None}
                dets.append(det)
                lignes_gpkg.append({**{k: det[k] for k in
                                       ("uid", "classe", "score", "retenu",
                                        "gt_apparie", "iou_gt")},
                                    "split": split, "zone": info.get("zone", ""),
                                    "dataset": info["dataset"],
                                    "tuile": info["file_name"],
                                    "coords": px_vers_l93(poly, grille, row, col)})
                compte["detections"] += 1
                compte["retenues"] += det["retenu"]
            if dets:
                tuiles[f"{split}/{info['file_name']}"] = {
                    "split": split, "zone": info.get("zone", ""),
                    "dataset": info["dataset"], "n_gt": len(gt_boxes),
                    "detections": dets}
            compte["images"] += 1
            if (n_fait + 1) % 200 == 0:
                print(f"  {split} {n_fait + 1}/{len(ids)} — "
                      f"{compte['detections']} détections ({compte['retenues']} retenues)")
        print(f"{split} : {len(ids)} images faites")

    meta = {"outil": "tools/inferer_corpus.py", "genere_le": datetime.now().isoformat(),
            "modele": nom_modele, "poids": os.path.abspath(args.poids),
            "tache": "segmentation" if seg else "detection",
            "resolution": args.resolution, "plancher": args.plancher,
            "seuils_f1max": seuils, "class_offset": decal or 0,
            "metriques_source": os.path.abspath(args.metriques),
            "corpus": os.path.abspath(args.corpus), "splits": args.splits,
            "comptes": compte}
    with open(os.path.join(args.out, "detections.json"), "w", encoding="utf-8") as f:
        json.dump({"_meta": meta, "tuiles": tuiles}, f, ensure_ascii=False)

    import geopandas as gpd
    from shapely.geometry import Polygon
    gdf = gpd.GeoDataFrame(
        [{k: v for k, v in l.items() if k != "coords"} for l in lignes_gpkg],
        geometry=[Polygon(l["coords"]) for l in lignes_gpkg], crs="EPSG:2154")
    gdf.to_file(os.path.join(args.out, "detections.gpkg"), layer="detections")

    par = {}
    for l in lignes_gpkg:
        cle = (l["split"], l["zone"], l["classe"])
        par.setdefault(cle, [0, 0])
        par[cle][0] += 1
        par[cle][1] += l["retenu"]
    meta["par_split_zone_classe"] = [
        {"split": s, "zone": z, "classe": c, "detections": n, "retenues": r}
        for (s, z, c), (n, r) in sorted(par.items())]
    with open(os.path.join(args.out, "resume.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"TERMINÉ : {compte['images']} images, {compte['detections']} détections "
          f"dont {compte['retenues']} retenues (offset classes {decal or 0}) -> {args.out}")


if __name__ == "__main__":
    main()
