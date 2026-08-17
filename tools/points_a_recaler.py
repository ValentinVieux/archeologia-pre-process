"""Couche des points a recaler d'un secteur : points d'un GPKG source sur donnees
valides du LD (garde-fou : aucun NoData a moins de 2xRmax=20 m du point), score de
contraste local (p90-p10 sur fenetre 120 m), tries decroissant. Valide sur la
campagne irlandaise (9 secteurs, 2026-08).

Usage : python tools\points_a_recaler.py <ld.tif> <sortie.gpkg>
        [--smr <points.gpkg>] [--couches enclosure ringfort]
"""
import argparse
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows

sys.stdout.reconfigure(encoding="utf-8")
BORD = 20   # 2 x Rmax : jamais annoter un monument coupe au bord
D = 60      # fenetre de score 120 m

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("ld"); ap.add_argument("sortie")
ap.add_argument("--smr", default="D:/veille_irlande/irlande_smr_enclos.gpkg")
ap.add_argument("--couches", nargs="+", default=["enclosure", "ringfort"])
a = ap.parse_args()
SMR = a.smr
ld, sortie = a.ld, a.sortie
src = rasterio.open(ld)
frames = []
for cls in a.couches:
    g = gpd.read_file(SMR, layer=cls).to_crs(src.crs)
    g["classe"] = cls
    frames.append(g[["classe", "ENTITY_ID", "COUNTY", "TOWNLAND", "WEBSITE_LINK", "geometry"]])
pts = pd.concat(frames).reset_index(drop=True)
b = src.bounds
pts = pts[(pts.geometry.x > b.left) & (pts.geometry.x < b.right)
          & (pts.geometry.y > b.bottom) & (pts.geometry.y < b.top)].copy()
rows = []
for i, p in pts.iterrows():
    r0, c0 = src.index(p.geometry.x, p.geometry.y)
    a = src.read(1, window=rasterio.windows.Window(c0 - D, r0 - D, 2 * D, 2 * D),
                 boundless=True, fill_value=255).astype("float32")
    a[a == 255] = np.nan
    centre = a[D - BORD:D + BORD + 1, D - BORD:D + BORD + 1]
    if not np.isfinite(centre).all():
        rows.append((i, np.nan, True))
        continue
    rows.append((i, round(float(np.nanpercentile(a, 90) - np.nanpercentile(a, 10)), 1), False))
res = pd.DataFrame(rows, columns=["idx", "score_dn", "exclu"]).set_index("idx")
pts = pts.join(res)
n_exclus = int(res.exclu.sum())
pts = pts[~pts.exclu].drop(columns=["exclu"])
pts["statut"] = ""
pts["note"] = ""
pts = pts.sort_values("score_dn", ascending=False)
gpd.GeoDataFrame(pts, crs=src.crs).to_file(sortie, layer="points_a_recaler", driver="GPKG")
print(f"{len(pts)} points -> {sortie} | exclus bord/hors donnees : {n_exclus} | "
      f"{dict(pts.classe.value_counts())} | score_dn mediane {pts.score_dn.median():.0f}")
