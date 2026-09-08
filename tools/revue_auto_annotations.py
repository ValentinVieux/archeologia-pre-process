"""Revue AUTOMATIQUE des couches d'annotations d'une zone avant découpe (promu le 2026-09-08,
règle des deux usages — utilisé sur 8 zones le 2026-09-07). Sans GPU.

Pour chaque couche polygone demandée : effectif, diamètre moyen (largeur+hauteur)/2 min/médiane/
max, boîtes < --diam-min et > --diam-max, allongement > 3, géométries invalides, DOUBLONS
(paires IoU des boîtes >= 0,5), boîtes hors emprise du raster, boîtes posées sur du NoData
(échantillon), et signature de taille fixe (> 80 % des boîtes au même diamètre à 0,1 m).
Verdict par couche : RAS / À VOIR (avec les compteurs). Ne modifie rien.

    .venv/Scripts/python.exe tools/revue_auto_annotations.py <gpkg> <raster> [--couches c1 c2] [--diam-min 3] [--diam-max 60]
"""
import argparse
import sys

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.windows import from_bounds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpkg")
    ap.add_argument("raster", help="LD/MNT de la zone (VRT ou tif), même CRS")
    ap.add_argument("--couches", nargs="*", default=None, help="défaut : toutes les couches polygones")
    ap.add_argument("--diam-min", type=float, default=3.0)
    ap.add_argument("--diam-max", type=float, default=60.0)
    ap.add_argument("--echantillon", type=int, default=200, help="boîtes testées sur le NoData")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    a_voir = 0
    with rasterio.open(a.raster) as src:
        nod = src.nodata if src.nodata is not None else 255
        for lay, geom_type in pyogrio.list_layers(a.gpkg):
            if a.couches and lay not in a.couches:
                continue
            if "Polygon" not in str(geom_type):
                continue
            df = gpd.read_file(a.gpkg, layer=lay)
            if df.empty:
                print(f"  {lay}: vide"); continue
            if df.crs is None or src.crs is None or df.crs.to_epsg() != src.crs.to_epsg():
                print(f"  {lay}: CRS {df.crs} ≠ raster {src.crs} — À VOIR"); a_voir += 1; continue
            b = df.geometry.bounds
            w, h = b.maxx - b.minx, b.maxy - b.miny
            d = (w + h) / 2
            allonge = np.maximum(w, h) / np.maximum(np.minimum(w, h), 0.01)
            invalides = int((~df.geometry.is_valid).sum() + df.geometry.is_empty.sum())
            env = df.geometry.envelope
            sidx = env.sindex
            dup = 0
            for i, g in enumerate(env):
                for j in sidx.query(g, predicate="intersects"):
                    if j <= i:
                        continue
                    inter = g.intersection(env.iloc[j]).area
                    if inter / (g.area + env.iloc[j].area - inter) >= 0.5:
                        dup += 1
            hors = int((~((b.minx >= src.bounds.left) & (b.maxx <= src.bounds.right)
                          & (b.miny >= src.bounds.bottom) & (b.maxy <= src.bounds.top))).sum())
            idx = rng.choice(len(df), min(a.echantillon, len(df)), replace=False)
            nod_n = 0
            for i in idx:
                try:
                    arr = src.read(1, window=from_bounds(*df.geometry.iloc[i].bounds, transform=src.transform),
                                   boundless=True, fill_value=nod)
                    nod_n += arr.size == 0 or (arr == nod).mean() > 0.5
                except Exception:
                    nod_n += 1
            mode = np.round(d, 1).mode() if hasattr(np.round(d, 1), "mode") else None
            part_fixe = float((np.round(d, 1) == np.round(d, 1).mode()[0]).mean())
            petits, gros, longs = int((d < a.diam_min).sum()), int((d > a.diam_max).sum()), int((allonge > 3).sum())
            alertes = []
            if petits: alertes.append(f"{petits} < {a.diam_min:g} m")
            if gros: alertes.append(f"{gros} > {a.diam_max:g} m")
            if longs: alertes.append(f"{longs} allongées (>3)")
            if invalides: alertes.append(f"{invalides} invalides")
            if dup: alertes.append(f"{dup} paires doublons (IoU>=0,5)")
            if hors: alertes.append(f"{hors} hors raster")
            if nod_n: alertes.append(f"{nod_n}/{len(idx)} sur NoData")
            if part_fixe > 0.8: alertes.append(f"taille FIXE ({100 * part_fixe:.0f} % à {np.round(d, 1).mode()[0]:.1f} m — boîtes synthétiques ?)")
            verdict = "RAS" if not alertes else "À VOIR : " + " ; ".join(alertes)
            a_voir += bool(alertes)
            print(f"  {lay:22s} n={len(df):5d} | diam {d.min():5.1f}/{d.median():5.1f}/{d.max():6.1f} m | {verdict}")
    print("VERDICT :", "RAS" if not a_voir else f"{a_voir} couche(s) À VOIR (diagnostiquer avant découpe : dedup_annotations.py, corriger_georef_dalles.py, revue humaine)")
    sys.exit(0 if not a_voir else 2)


if __name__ == "__main__":
    main()
