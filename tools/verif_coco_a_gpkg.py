"""Contrôleur indépendant de coco_a_gpkg.py : re-dérive tout depuis le COCO source.

Vérifie sans importer l'outil (boucle de vérification, règle 2026-07-27) :
  1. comptes par couche/split GPKG == comptes recalculés depuis les COCO ;
  2. unicité des uid ; CRS EPSG:2154 ;
  3. géométries : recalcul indépendant de CHAQUE bbox (coin NW du nom de dalle,
     m/px = 1000/width) et comparaison aux bornes du GPKG à 1 cm près ;
  4. rasters : un GeoTIFF par dalle des splits, transform/CRS/dimensions conformes.

Usage :
  .venv\\Scripts\\python.exe tools\\verif_coco_a_gpkg.py <payload> <sortie>
Verdict final : CONFORME / NON CONFORME (exit 1).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import pyogrio

ERREURS: list[str] = []


def ko(msg: str) -> None:
    ERREURS.append(msg)
    print(f"  KO : {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("payload", type=Path)
    ap.add_argument("sortie", type=Path)
    ap.add_argument("--fusion", action="append", default=[], metavar="SRC=DST",
                    help="mêmes fusions de classes que coco_a_gpkg (recalcul indépendant)")
    a = ap.parse_args()
    fusion = dict(f.split("=", 1) for f in a.fusion)
    gpkg = a.sortie / "annotations_coco.gpkg"
    if not gpkg.is_file():
        sys.exit(f"NON CONFORME : {gpkg} absent")

    # référence indépendante : (classe, split, uid) -> bornes attendues
    attendu: dict[str, tuple] = {}
    dalles: set[str] = set()
    for split in ("train", "valid", "test"):
        p = a.payload / split / "_annotations.coco.json"
        if not p.is_file():
            continue
        coco = json.load(open(p, encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        images = {i["id"]: i for i in coco["images"]}
        for im in coco["images"]:
            m = re.search(r"LHD_FXX_(\d{4})_(\d{4})", im["file_name"])
            if m:
                dalles.add(f"LHD_FXX_{m.group(1)}_{m.group(2)}_LD")
        for ann in coco["annotations"]:
            im = images[ann["image_id"]]
            m = re.search(r"LHD_FXX_(\d{4})_(\d{4})", im["file_name"])
            if not m:
                continue
            x0, y1 = int(m.group(1)) * 1000, int(m.group(2)) * 1000
            r = 1000.0 / im["width"]
            bx, by, bw, bh = ann["bbox"]
            attendu[f"{split}:{ann['id']}"] = (
                fusion.get(cats[ann["category_id"]], cats[ann["category_id"]]), split,
                x0 + bx * r, y1 - (by + bh) * r, x0 + (bx + bw) * r, y1 - by * r,
            )

    couches = [c for c, _ in pyogrio.list_layers(gpkg)]
    print(f"couches : {couches}")
    uids_vus: set[str] = set()
    for couche in couches:
        gdf = gpd.read_file(gpkg, layer=couche)
        if gdf.crs is None or gdf.crs.to_epsg() != 2154:
            ko(f"{couche} : CRS {gdf.crs} != EPSG:2154")
        doublons = set(gdf["uid"]) & uids_vus
        if doublons or gdf["uid"].duplicated().any():
            ko(f"{couche} : uid non uniques")
        uids_vus |= set(gdf["uid"])

        ref = {u: v for u, v in attendu.items() if v[0] == couche}
        par_split = gdf["split"].value_counts().to_dict()
        ref_split: dict[str, int] = {}
        for v in ref.values():
            ref_split[v[1]] = ref_split.get(v[1], 0) + 1
        if par_split != ref_split:
            ko(f"{couche} : comptes gpkg {par_split} != coco {ref_split}")
        else:
            print(f"  OK comptes {couche} : {par_split}")

        n_geo_ko = 0
        for _, row in gdf.iterrows():
            v = ref.get(row["uid"])
            if v is None:
                n_geo_ko += 1
                continue
            b = row.geometry.bounds
            if any(abs(b[i] - v[2 + i]) > 0.01 for i in range(4)):
                n_geo_ko += 1
        if n_geo_ko:
            ko(f"{couche} : {n_geo_ko} géométrie(s) discordantes (> 1 cm) ou uid inconnus")
        else:
            print(f"  OK géométries {couche} : {len(gdf)}/{len(gdf)} recalculées identiques")

    rasters = a.sortie / "rasters"
    if rasters.is_dir():
        import rasterio
        tifs = {p.stem: p for p in rasters.glob("*.tif")}
        manquantes = dalles - set(tifs)
        if manquantes:
            ko(f"rasters : {len(manquantes)} dalle(s) sans GeoTIFF, ex. {sorted(manquantes)[:3]}")
        for stem, p in tifs.items():
            m = re.match(r"LHD_FXX_(\d{4})_(\d{4})_LD", stem)
            with rasterio.open(p) as src:
                t = src.transform
                ok = (m and src.crs and src.crs.to_epsg() == 2154
                      and abs(t.c - int(m.group(1)) * 1000) < 0.01
                      and abs(t.f - int(m.group(2)) * 1000) < 0.01
                      and abs(t.a * src.width - 1000) < 0.01
                      and abs(-t.e * src.height - 1000) < 0.01)
            if not ok:
                ko(f"raster {p.name} : géoréférencement non conforme")
        if not manquantes:
            print(f"  OK rasters : {len(tifs)} GeoTIFF conformes à la grille")

    if ERREURS:
        print(f"\nNON CONFORME ({len(ERREURS)} erreur(s))")
        sys.exit(1)
    print("\nCONFORME")


if __name__ == "__main__":
    main()
