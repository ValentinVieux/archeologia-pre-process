"""Dédoublonne les annotations d'un GPKG (toutes couches polygones) : deux entités de même
couche dont les boîtes ont un IoU >= --iou sont le même objet ; on garde la première (ordre
de la couche) et on supprime les autres, en traçant `doublon_de` (uid ou fid gardé) dans une
couche `<couche>_doublons` de sortie.

Cause typique (2026-09-07) : dalles LD « à marge » (km + 50 m) — un objet dans la marge est
annoté dans DEUX images voisines ; après correction du géoréférencement les deux annotations
coïncident (avant, elles étaient à ~100 m l'une de l'autre). Le rapport distingue les paires
inter-dalles (champ `tuile` différent) des paires intra-dalle (vrais doublons de saisie).

    .venv/Scripts/python.exe tools/dedup_annotations.py <gpkg_entree> <gpkg_sortie> [--iou 0.5] [--couches c1 c2]
"""
import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pyogrio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpkg_entree")
    ap.add_argument("gpkg_sortie")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--couches", nargs="*", default=None, help="défaut : toutes les couches")
    a = ap.parse_args()
    if os.path.exists(a.gpkg_sortie):
        sys.exit(f"refus : {a.gpkg_sortie} existe déjà")
    for lay, _ in pyogrio.list_layers(a.gpkg_entree):
        if a.couches and lay not in a.couches:
            continue
        df = gpd.read_file(a.gpkg_entree, layer=lay)
        if df.empty:
            df.to_file(a.gpkg_sortie, layer=lay, driver="GPKG"); continue
        cle = "uid" if "uid" in df.columns else None
        boites = df.geometry.envelope
        sidx = boites.sindex
        mort = np.zeros(len(df), dtype=bool)
        garde_de = [None] * len(df)
        inter_d, intra_d = 0, 0
        for i in range(len(df)):
            if mort[i]:
                continue
            bi = boites.iloc[i]
            for j in sidx.query(bi, predicate="intersects"):
                if j <= i or mort[j]:
                    continue
                bj = boites.iloc[j]
                inter = bi.intersection(bj).area
                if inter / (bi.area + bj.area - inter) >= a.iou:
                    mort[j] = True
                    garde_de[j] = str(df[cle].iloc[i]) if cle else str(i)
                    if "tuile" in df.columns and df["tuile"].iloc[i] != df["tuile"].iloc[j]:
                        inter_d += 1
                    else:
                        intra_d += 1
        gardes = df[~mort].copy()
        doublons = df[mort].copy()
        doublons["doublon_de"] = [g for g, m in zip(garde_de, mort) if m]
        gardes.to_file(a.gpkg_sortie, layer=lay, driver="GPKG")
        if len(doublons):
            doublons.to_file(a.gpkg_sortie, layer=f"{lay}_doublons", driver="GPKG")
        print(f"  {lay}: {len(df)} -> {len(gardes)} gardées, {len(doublons)} doublons supprimés "
              f"(inter-dalles {inter_d}, intra-dalle {intra_d})")


if __name__ == "__main__":
    main()
