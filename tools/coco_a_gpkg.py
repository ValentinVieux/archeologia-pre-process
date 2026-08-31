"""Reconstruit un GPKG géoréférencé depuis un payload Roboflow dispatché (COCO par split).

Pour les zones dont on n'a QUE les annotations COCO (pas de vecteurs source), ex.
57_fenetrange : chaque bbox redevient un polygone EPSG:2154 traçable vers son
annotation d'origine (uid = split:annotation_id), éditable dans QGIS. Optionnellement
géoréférence aussi les tuiles jpg en GeoTIFF pour la revue.

Convention d'emprise (vérifiée sur la grille WFS IGNF_MNT-LIDAR-HD:dalle, champ
coordonnees_NW) : LHD_FXX_XXXX_YYYY = coin NORD-OUEST en km Lambert-93, dalle de
1 km ; incertitude résiduelle <= 0,25 m (grille WMS décalée d'un demi-pixel MNT).

Usage :
  .venv\\Scripts\\python.exe tools\\coco_a_gpkg.py <payload> <sortie> --classes c1 [c2 ...] [--rasters]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
from shapely.affinity import scale as agrandir
from shapely.geometry import Point, box

LHD_RE = re.compile(r"LHD_FXX_(\d{4})_(\d{4})")
SPLITS = ("train", "valid", "test")
COTE_M = 1000.0  # dalle LHD 1 km


def emprise_dalle(file_name: str) -> tuple[float, float] | None:
    """(xmin, ymax) Lambert-93 du coin NW, ou None si nom non LHD."""
    m = LHD_RE.search(file_name)
    if not m:
        return None
    return int(m.group(1)) * 1000.0, int(m.group(2)) * 1000.0


def lire_split(payload: Path, split: str) -> dict | None:
    p = payload / split / "_annotations.coco.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def extraire(payload: Path, classes: list[str], forme: str = "bbox",
             fusion: dict[str, str] | None = None) -> gpd.GeoDataFrame:
    lignes, hors_grille = [], 0
    fusion = fusion or {}
    for split in SPLITS:
        coco = lire_split(payload, split)
        if coco is None:
            continue
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        images = {i["id"]: i for i in coco["images"]}
        for a in coco["annotations"]:
            classe = fusion.get(cats.get(a["category_id"]), cats.get(a["category_id"]))
            if classe not in classes:
                continue
            im = images[a["image_id"]]
            nw = emprise_dalle(im["file_name"])
            if nw is None:
                hors_grille += 1
                continue
            xmin_d, ymax_d = nw
            px = COTE_M / im["width"]  # m/pixel
            x, y, w, h = a["bbox"]
            if forme == "ellipse":
                # ellipse inscrite dans la bbox (objets quasi circulaires : masque approché)
                geom = agrandir(Point(xmin_d + (x + w / 2) * px, ymax_d - (y + h / 2) * px).buffer(1.0, 32),
                                w * px / 2, h * px / 2)
            else:
                geom = box(xmin_d + x * px, ymax_d - (y + h) * px,
                           xmin_d + (x + w) * px, ymax_d - y * px)
            lignes.append({
                "uid": f"{split}:{a['id']}",
                "split": split,
                "annotation_id": a["id"],
                "image_id": a["image_id"],
                "file_name": im["file_name"],
                "tuile": f"{int(xmin_d / 1000):04d}_{int(ymax_d / 1000):04d}",
                "classe": classe,
                "largeur_m": round(w * px, 2),
                "hauteur_m": round(h * px, 2),
                "geometry": geom,
            })
    if hors_grille:
        print(f"ATTENTION : {hors_grille} annotation(s) sur images hors grille LHD, ignorées")
    return gpd.GeoDataFrame(lignes, crs="EPSG:2154")


def georeferencer_rasters(payload: Path, sortie: Path) -> int:
    import rasterio
    from rasterio.transform import from_origin

    dossier = sortie / "rasters"
    dossier.mkdir(parents=True, exist_ok=True)
    vus: dict[str, str] = {}
    n = 0
    for split in SPLITS:
        for jpg in sorted((payload / split).glob("*.jpg")):
            nw = emprise_dalle(jpg.name)
            if nw is None:
                print(f"ATTENTION : {jpg.name} hors grille LHD, non géoréférencée")
                continue
            xmin_d, ymax_d = nw
            dalle = f"LHD_FXX_{int(xmin_d / 1000):04d}_{int(ymax_d / 1000):04d}_LD"
            if dalle in vus:
                print(f"ATTENTION : dalle {dalle} en double ({jpg.name} vs {vus[dalle]}), première conservée")
                continue
            vus[dalle] = jpg.name
            with rasterio.open(jpg) as src:
                data = src.read()
                profil = {
                    "driver": "GTiff", "count": src.count, "dtype": data.dtype,
                    "width": src.width, "height": src.height,
                    "crs": "EPSG:2154",
                    "transform": from_origin(xmin_d, ymax_d, COTE_M / src.width, COTE_M / src.height),
                    "compress": "lzw", "tiled": True,
                }
            with rasterio.open(dossier / f"{dalle}.tif", "w", **profil) as dst:
                dst.write(data)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("payload", type=Path, help="dossier du payload dispatché (train/valid/test)")
    ap.add_argument("sortie", type=Path, help="dossier de sortie")
    ap.add_argument("--classes", nargs="+", required=True, help="classes (APRÈS fusion) à extraire, une couche GPKG chacune")
    ap.add_argument("--rasters", action="store_true", help="géoréférencer aussi les jpg en GeoTIFF")
    ap.add_argument("--forme", choices=["bbox", "ellipse"], default="bbox",
                    help="ellipse = ellipse inscrite dans la bbox (masques approchés d'objets circulaires)")
    ap.add_argument("--fusion", action="append", default=[], metavar="SRC=DST",
                    help="renommage/fusion de classes COCO, ex. charbonniere_vosges=charbonniere")
    a = ap.parse_args()
    fusion = dict(f.split("=", 1) for f in a.fusion)

    if not any((a.payload / s / "_annotations.coco.json").is_file() for s in SPLITS):
        sys.exit(f"aucun _annotations.coco.json dans {a.payload}\\{{train,valid,test}}")
    a.sortie.mkdir(parents=True, exist_ok=True)

    gdf = extraire(a.payload, a.classes, forme=a.forme, fusion=fusion)
    if gdf.empty:
        sys.exit(f"aucune annotation des classes {a.classes} trouvée")
    gpkg = a.sortie / "annotations_coco.gpkg"
    if gpkg.exists():
        gpkg.unlink()  # régénération complète, le COCO source fait foi
    for classe in a.classes:
        couche = gdf[gdf["classe"] == classe].drop(columns=["classe"])
        if couche.empty:
            print(f"  {classe} : 0 annotation, couche omise")
            continue
        couche.to_file(gpkg, layer=classe, driver="GPKG")
        parts = ", ".join(f"{s} {n}" for s, n in couche["split"].value_counts().items())
        print(f"  {classe} : {len(couche)} entités ({parts})")

    if a.rasters:
        n = georeferencer_rasters(a.payload, a.sortie)
        print(f"  rasters : {n} GeoTIFF dans {a.sortie / 'rasters'}")
    print(f"Sorties : {gpkg}")


if __name__ == "__main__":
    main()
