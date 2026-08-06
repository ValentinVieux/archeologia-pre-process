"""Contrôleur indépendant des propositions de polygones du corpus Irlande.

Vérifie un GPKG produit par proposer_polygones_irlande.py contre sa couche de
points d'entrée : compte (une proposition par point, ni plus ni moins), CRS
EPSG:2154, géométries valides SANS TROU (post-traitement acté), chaque point à
l'intérieur (ou à moins de 5 m) de son polygone, champs requis présents, aires
dans le gabarit [50 m², 20 000 m²]. Ne modifie RIEN. Verdict final : CONFORME
ou liste des écarts.

Usage : .venv\\Scripts\\python.exe tools\\verif_polygones_irlande.py
            <points.gpkg> <propositions.gpkg> [--couche-points points_a_recaler]
"""
import argparse
import sys

import geopandas as gpd

CHAMPS = {"ENTITY_ID", "classe", "methode", "echelle", "sc_cercle", "sc_sam",
          "accord", "a_verifier", "verdict", "note"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("points")
    ap.add_argument("propositions")
    ap.add_argument("--couche-points", default="points_a_recaler")
    a = ap.parse_args()

    ecarts = []
    pts = gpd.read_file(a.points, layer=a.couche_points)
    props = gpd.read_file(a.propositions, layer="propositions")

    if props.crs is None or props.crs.to_epsg() != 2154:
        ecarts.append(f"CRS = {props.crs}, attendu EPSG:2154")
    if pts.crs != props.crs:
        pts = pts.to_crs(props.crs)

    manquants = CHAMPS - set(props.columns)
    if manquants:
        ecarts.append(f"champs manquants : {sorted(manquants)}")

    ids_pts = set(pts.ENTITY_ID.dropna())
    ids_props = set(props.ENTITY_ID.dropna())
    if ids_pts - ids_props:
        ecarts.append(f"{len(ids_pts - ids_props)} points sans proposition : {sorted(ids_pts - ids_props)[:5]}...")
    if ids_props - ids_pts:
        ecarts.append(f"{len(ids_props - ids_pts)} propositions orphelines")
    if props.ENTITY_ID.dropna().duplicated().any():
        ecarts.append("ENTITY_ID dupliqués dans les propositions")

    invalides = (~props.geometry.is_valid).sum()
    if invalides:
        ecarts.append(f"{invalides} géométries invalides")
    trous = sum(len(g.interiors) for g in props.geometry if g.geom_type == "Polygon")
    if trous:
        ecarts.append(f"{trous} trous intérieurs (post-traitement non appliqué ?)")
    aires = props.geometry.area
    hors = ((aires < 50) | (aires > 20000)).sum()
    if hors:
        ecarts.append(f"{hors} aires hors gabarit [50, 20000] m²")

    pts_i = pts.set_index("ENTITY_ID")
    loin = 0
    for _, r in props.dropna(subset=["ENTITY_ID"]).iterrows():
        if r.ENTITY_ID in pts_i.index:
            p = pts_i.loc[r.ENTITY_ID].geometry
            p = p.iloc[0] if hasattr(p, "iloc") else p
            if r.geometry.distance(p) > 5.0:
                loin += 1
    if loin:
        ecarts.append(f"{loin} propositions à plus de 5 m de leur point")

    if ecarts:
        print("NON CONFORME :")
        for e in ecarts:
            print("  -", e)
        sys.exit(1)
    print(f"vérification propositions : CONFORME — {len(props)} polygones, "
          f"{int(props.a_verifier.sum())} drapeaux a_verifier, CRS 2154, zéro trou")


if __name__ == "__main__":
    main()
