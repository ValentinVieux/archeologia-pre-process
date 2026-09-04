"""Assemble les GPKG d'entraînement des zones SPÉCIALES du corpus fours/charbonnières.

Trois zones ne sont pas de simples conversions COCO (celles-là passent par
coco_a_gpkg.py --forme ellipse --fusion) :
  - 25_besancon_chailluz : points sans taille -> cercles rayon constant 5 m
    (diamètre médian Blois 9,6 m) + auto-labels circular_depression ;
  - 41_blois : points AVEC rayon/diamètre par entité -> cercles fidèles
    + auto-labels circular_depression ;
  - 78_rambouillet : charbonnières = ellipses des bboxes COCO (tailles réelles
    par instance, fusion charbonniere_rambouillet->charbonniere), dépressions =
    polygones du GPKG v2, points de dépressions sans emprise -> couche
    `depression_pts_ignorer` (cercles 15 m, ignorer: true dans slice_zone).

Les auto-labels (auto_label_depressions.py, bboxes) sont convertis en ellipses
inscrites. S'ils manquent : avertissement, couche omise (re-lancer après).

Usage :
  .venv\\Scripts\\python.exe tools\\build_gpkg_ponctuelles.py <zone> <sortie.gpkg>
      [--auto-labels <gpkg>] [--payload <dossier COCO>] [--source <gpkg source>]
zone ∈ {chailluz, blois, rambouillet}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent))
from coco_a_gpkg import extraire  # noqa: E402  (ellipses depuis COCO, convention NW vérifiée)

RAYON_CHAILLUZ_M = 5.0    # diamètre 10 m ~ médiane Blois (9,6 m)
RAYON_IGNORER_M = 15.0    # points de dépression sans emprise : zone à exclure des négatifs
CERCLE_SEGS = 32


def ellipses_auto_labels(chemin: Path) -> gpd.GeoDataFrame | None:
    if not chemin or not Path(chemin).is_file():
        print("ATTENTION : auto-labels absents, couche circular_depression omise")
        return None
    g = gpd.read_file(chemin)  # bboxes de auto_label_depressions.py
    from shapely.affinity import scale as agrandir
    from shapely.geometry import Point

    def ellipse(geom):
        x0, y0, x1, y1 = geom.bounds
        return agrandir(Point((x0 + x1) / 2, (y0 + y1) / 2).buffer(1.0, CERCLE_SEGS),
                        (x1 - x0) / 2, (y1 - y0) / 2)

    g["geometry"] = g.geometry.map(ellipse)
    return g[["score", "geometry"]].set_crs("EPSG:2154", allow_override=True)


def ecrire(gpkg: Path, couches: dict[str, gpd.GeoDataFrame | None]) -> None:
    gpkg = Path(gpkg)
    gpkg.parent.mkdir(parents=True, exist_ok=True)
    if gpkg.exists():
        gpkg.unlink()  # régénération complète, les sources font foi
    for nom, gdf in couches.items():
        if gdf is None or gdf.empty:
            continue
        gdf.to_file(gpkg, layer=nom, driver="GPKG")
        print(f"  {nom} : {len(gdf)}")
    print(f"Sorties : {gpkg}")


def zone_chailluz(a) -> dict:
    g = gpd.read_file(a.source, layer="charbonnieres")
    g["geometry"] = g.geometry.buffer(RAYON_CHAILLUZ_M, CERCLE_SEGS)
    return {"charbonniere": g[["geometry"]],
            "circular_depression": ellipses_auto_labels(a.auto_labels)}


def zone_blois(a) -> dict:
    g = gpd.read_file(a.source, layer="charbonniere")
    rayons = (g["diamètre"] / 2).fillna(g["rayon"]).fillna(RAYON_CHAILLUZ_M)
    g["geometry"] = [p.buffer(max(float(r), 1.0), CERCLE_SEGS)
                     for p, r in zip(g.geometry, rayons)]
    return {"charbonniere": g[["geometry"]],
            "circular_depression": ellipses_auto_labels(a.auto_labels)}


def zone_rambouillet(a) -> dict:
    charb = extraire(Path(a.payload), ["charbonniere"], forme="ellipse",
                     fusion={"charbonniere_rambouillet": "charbonniere"})
    charb = charb[["uid", "split", "geometry"]]
    dep = gpd.read_file(a.source, layer="circular_depression")[["geometry"]]
    pts = gpd.read_file(a.source, layer="circular_depression_pts")
    pts["geometry"] = pts.geometry.buffer(RAYON_IGNORER_M, CERCLE_SEGS)
    return {"charbonniere": charb, "circular_depression": dep,
            "depression_pts_ignorer": pts[["geometry"]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zone", choices=["chailluz", "blois", "rambouillet"])
    ap.add_argument("sortie", type=Path)
    ap.add_argument("--source", help="GPKG source (couches de la zone)")
    ap.add_argument("--payload", help="payload COCO dispatché (rambouillet)")
    ap.add_argument("--auto-labels", help="GPKG de auto_label_depressions.py (chailluz/blois)")
    a = ap.parse_args()
    couches = {"chailluz": zone_chailluz, "blois": zone_blois,
               "rambouillet": zone_rambouillet}[a.zone](a)
    ecrire(a.sortie, couches)


if __name__ == "__main__":
    main()
