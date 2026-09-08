"""Contrôleur indépendant (SANS GPU, .venv) de tools/sam_polygones_bbox.py.

Vérifie, depuis les fichiers produits : même nombre d'entités que la couche d'entrée,
mêmes attributs d'origine, CRS identique, chaque polygone contient le centre de sa
boîte et tient dans la boîte élargie de --marge m, diam_eq_m recalculé, repli bbox
byte-équivalent à la boîte d'origine. Verdict CONFORME / NON CONFORME (exit 1).

    .venv/Scripts/python.exe tools/verif_sam_polygones_bbox.py <gpkg_entree> <couche> <gpkg_sortie> [--couche-sortie X] [--marge 20]
"""
import argparse
import sys

import geopandas as gpd
import numpy as np
from shapely.geometry import box


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpkg_entree")
    ap.add_argument("couche")
    ap.add_argument("gpkg_sortie")
    ap.add_argument("--couche-sortie", default=None, help="défaut : <couche>_sam")
    ap.add_argument("--marge", type=float, default=20.0, help="RÉPÉTER la marge du producteur (m par côté)")
    a = ap.parse_args()

    src = gpd.read_file(a.gpkg_entree, layer=a.couche)
    out = gpd.read_file(a.gpkg_sortie, layer=a.couche_sortie or f"{a.couche}_sam")
    pb = []
    if len(src) != len(out):
        pb.append(f"effectif : entrée {len(src)} vs sortie {len(out)}")
    if src.crs != out.crs:
        pb.append(f"CRS : {src.crs} vs {out.crs}")
    manquants = [c for c in src.columns if c != "geometry" and c not in out.columns]
    if manquants:
        pb.append(f"attributs d'origine perdus : {manquants}")
    n = min(len(src), len(out))
    hors, centre_out, diam_faux, repli_faux, sam = 0, 0, 0, 0, 0
    for g_in, (_, r) in zip(src.geometry.iloc[:n], out.iloc[:n].iterrows()):
        b = g_in.bounds
        elargie = box(b[0] - a.marge, b[1] - a.marge, b[2] + a.marge, b[3] + a.marge)
        g = r.geometry
        if not elargie.buffer(0.5).contains(g):
            hors += 1
        if not g.contains(box(*b).centroid):
            centre_out += 1
        if abs(2 * np.sqrt(g.area / np.pi) - r["diam_eq_m"]) > 0.2:
            diam_faux += 1
        if r["methode"] == "bbox_repli" and not g.equals_exact(box(*b), 0.01):
            repli_faux += 1
        sam += r["methode"] == "sam"
    if hors:
        pb.append(f"{hors} polygones débordent de la boîte élargie")
    if centre_out:
        pb.append(f"{centre_out} polygones ne contiennent pas le centre de leur boîte")
    if diam_faux:
        pb.append(f"{diam_faux} diam_eq_m incohérents")
    if repli_faux:
        pb.append(f"{repli_faux} replis bbox différents de la boîte d'origine")
    # --- qualité du résultat (leçon 2026-09-07 : des replis injustifiés étaient passés « CONFORMES ») ---
    o = out.iloc[:n]
    bx = o.geometry.bounds
    remplissage = o.geometry.area / ((bx.maxx - bx.minx) * (bx.maxy - bx.miny))
    rect_sam = o[(o["methode"] == "sam") & (remplissage > 0.95)]
    replis = o[o["methode"] == "bbox_repli"]
    if "motif" not in o.columns:
        pb.append("champ motif absent : producteur antérieur au 2026-09-07, relancer")
    else:
        sans_motif = replis[replis["motif"].isin(["ok", "", None])]
        if len(sans_motif):
            pb.append(f"{len(sans_motif)} replis sans motif")
        print("motifs des replis :", replis["motif"].value_counts().to_dict() if len(replis) else "aucun repli")
    if len(rect_sam):
        pb.append(f"{len(rect_sam)} polygones « sam » quasi rectangulaires (remplissage > 95 %) : fid_source {rect_sam.get('fid_source', rect_sam.index).tolist()[:10]}")
    if n and len(replis) / n > 0.05:
        pb.append(f"taux de replis {100 * len(replis) / n:.1f} % > 5 % : à diagnostiquer avant livraison")
    print(f"{n} entités — sam {sam}, repli {n - sam} ; hors boîte élargie {hors} ; centre absent {centre_out}")
    print(f"remplissage de la boîte (sam) : quartiles {np.percentile(remplissage[o['methode'] == 'sam'], [25, 50, 75]).round(2).tolist() if sam else '-'} ; "
          f"sam_score médian {o.loc[o['methode'] == 'sam', 'sam_score'].median():.2f}" if sam else "aucun masque sam")
    if len(replis):
        print("replis (fid_source) :", replis.get("fid_source", replis.index).tolist()[:20])
    if pb:
        print("NON CONFORME :\n  " + "\n  ".join(pb))
        sys.exit(1)
    print("CONFORME")


if __name__ == "__main__":
    main()
