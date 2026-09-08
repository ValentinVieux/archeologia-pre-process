"""Corrige le géoréférencement des annotations issues de dalles LD « à marge » (2026-09-06).

Défaut : une dalle LHD de w px est géoréférencée comme couvrant 1 000 m depuis son coin
NW (px = 1000 / w). Or les dalles de 2 201 px couvrent w × 0,5 m (km + ~50 m de marge de
chaque côté) : compression de 10 % vers le NW, 0 au centre, ±50 m aux bords, héritée par
les annotations (coco_a_gpkg). Correction déterministe, par entité, d'après le champ
`tuile` (coin NW en km) et la largeur réelle de l'image de la dalle :

    m = (w × 0,5 − 1000) / 2 ;  x' = X0 − m + (x − X0) × w × 0,5 / 1000 ;  y' = Y0 + m − (Y0 − y) × w × 0,5 / 1000

(w = 2000 → identité). largeur_m / hauteur_m sont recalculés. Toutes les couches du GPKG
sont traitées ; sortie = nouveau GPKG (l'entrée n'est jamais modifiée).

    .venv/Scripts/python.exe tools/corriger_georef_dalles.py <gpkg_entree> <dossier_dalles> <gpkg_sortie> [--gsd 0.5] [--cote 1000]

<dossier_dalles> : les images/rasters des dalles (LHD_FXX_XXXX_YYYY_*.tif|jpg), pour lire w.
"""
import argparse
import glob
import os
import re
import sys

import geopandas as gpd
import pyogrio
from shapely.affinity import scale, translate


def largeurs_dalles(dossier):
    """{ 'XXXX_YYYY': largeur px } depuis les rasters/images du dossier (en-tête seulement)."""
    import rasterio
    w = {}
    for f in glob.glob(os.path.join(dossier, "LHD_FXX_*")):
        m = re.match(r"LHD_FXX_(\d{4})_(\d{4})", os.path.basename(f))
        if not m or not f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png")):
            continue
        with rasterio.open(f) as s:
            w[f"{m.group(1)}_{m.group(2)}"] = s.width
    return w


def corriger(geom, tuile, w, gsd, cote):
    x0, y0 = int(tuile[:4]) * 1000.0, int(tuile[5:9]) * 1000.0
    s = w * gsd / cote            # 1,1005 pour 2 201 px
    m = (w * gsd - cote) / 2.0    # marge réelle (m) : 50,25 pour 2 201 px
    if abs(s - 1.0) < 1e-9:
        return geom
    g = scale(geom, xfact=s, yfact=s, origin=(x0, y0))
    return translate(g, xoff=-m, yoff=m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpkg_entree")
    ap.add_argument("dossier_dalles")
    ap.add_argument("gpkg_sortie")
    ap.add_argument("--gsd", type=float, default=0.5, help="taille réelle du pixel des dalles (m)")
    ap.add_argument("--cote", type=float, default=1000.0, help="côté nominal de la dalle (m)")
    a = ap.parse_args()
    if os.path.exists(a.gpkg_sortie):
        sys.exit(f"refus : {a.gpkg_sortie} existe déjà")
    w = largeurs_dalles(a.dossier_dalles)
    print(f"{len(w)} dalles, largeurs : { {k: sum(1 for v in w.values() if v == k) for k in set(w.values())} }")
    for lay, _ in pyogrio.list_layers(a.gpkg_entree):
        gdf = gpd.read_file(a.gpkg_entree, layer=lay)
        manq = sorted(set(gdf["tuile"]) - set(w))
        if manq:
            sys.exit(f"[{lay}] dalles sans image dans {a.dossier_dalles} : {manq[:10]}")
        gdf["geometry"] = [corriger(g, t, w[t], a.gsd, a.cote) for g, t in zip(gdf.geometry, gdf["tuile"])]
        b = gdf.geometry.bounds
        if "largeur_m" in gdf:
            gdf["largeur_m"] = (b.maxx - b.minx).round(2)
        if "hauteur_m" in gdf:
            gdf["hauteur_m"] = (b.maxy - b.miny).round(2)
        n_corr = sum(1 for t in gdf["tuile"] if w[t] * a.gsd != a.cote)
        gdf.to_file(a.gpkg_sortie, layer=lay, driver="GPKG")
        print(f"  {lay}: {len(gdf)} entités, {n_corr} corrigées (dalles à marge), {len(gdf) - n_corr} inchangées")


if __name__ == "__main__":
    main()
