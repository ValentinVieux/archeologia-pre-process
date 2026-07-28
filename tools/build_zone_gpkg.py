"""Construit le GPKG d'entités d'une zone (training/vecteurs) depuis sa livraison
auditée, piloté par un YAML de mapping (configs/vecteurs_<zone>.yaml).

Généralisation de build_fontainebleau_gpkg.py (règle de trois : Haye, Fontainebleau,
Rambouillet). Une couche par entité ; quand une entité existe en points ET dans une
autre géométrie, les points partent en couche `_pts`. Géométries invalides réparées
par make_valid (compte rapporté), attributs + type_source conservés, aucune
transformation de géométrie.

Format du YAML de mapping :
    gpkg_nom: <nom>.gpkg
    crs: 2154                      # CRS attendu des sources (contrôle strict)
    sources:
      <fichier.shp>:
        champ: <nom du champ de classe>
        valeurs: { "<valeur brute>": <entity_id>, ... }   # aliases.yaml fait foi
        exclure: ["<valeur ignorée>", ...]                # optionnel (ex. "Autre")

Usage :
    .venv\\Scripts\\python.exe tools\\build_zone_gpkg.py <mapping.yaml> <dossier_source> [--out <dossier>]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import yaml
from shapely import make_valid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mapping", help="configs/vecteurs_<zone>.yaml")
    p.add_argument("source", help="dossier de la livraison (copie locale)")
    p.add_argument("--out", default=None, help="dossier de sortie (défaut : source)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.mapping).read_text(encoding="utf-8"))
    source = Path(args.source)
    sortie = Path(args.out) if args.out else source
    sortie.mkdir(parents=True, exist_ok=True)
    gpkg = sortie / cfg["gpkg_nom"]
    if gpkg.exists():
        gpkg.unlink()  # reconstruit intégralement (sorties régénérables)

    # géométries par entité pour décider des suffixes _pts (conflit points/autre)
    geoms_par_entite = {}
    donnees = {}
    for fichier, spec in cfg["sources"].items():
        gdf = gpd.read_file(source / fichier)
        if gdf.crs is None or gdf.crs.to_epsg() != cfg["crs"]:
            sys.exit(f"{fichier} : CRS {gdf.crs} inattendu (EPSG:{cfg['crs']} requis)")
        valeurs = set(gdf[spec["champ"]].dropna())
        exclues = set(spec.get("exclure", []))
        for v in exclues & valeurs:
            n = int((gdf[spec["champ"]] == v).sum())
            print(f"  exclu : {fichier}/{v} — {n} entité(s) (ignorée par l'audit)")
        inconnues = valeurs - set(spec["valeurs"]) - exclues
        if inconnues:
            sys.exit(f"{fichier} : valeurs non mappées {sorted(inconnues)} — "
                     "compléter le mapping (aliases.yaml fait foi) ou re-auditer")
        donnees[fichier] = (gdf, spec)
        est_point = gdf.geom_type.iloc[0] in ("Point", "MultiPoint") if len(gdf) else False
        for entite in spec["valeurs"].values():
            geoms_par_entite.setdefault(entite, set()).add(
                "point" if est_point else "autre")

    total, reparees, nulles_ecartees = 0, 0, 0
    recap = Counter()
    details = {}
    couches_ecrites = set()
    for fichier, (gdf, spec) in donnees.items():
        est_point = gdf.geom_type.iloc[0] in ("Point", "MultiPoint") if len(gdf) else False
        for brut, entite in spec["valeurs"].items():
            sel = gdf[gdf[spec["champ"]] == brut].copy()
            n_nulles = int((sel.geometry.isna() | sel.geometry.is_empty).sum())
            if n_nulles:  # géométries vidées par l'édition (cas Haye parcellaire v2)
                sel = sel[sel.geometry.notna() & ~sel.geometry.is_empty]
                nulles_ecartees += n_nulles
                print(f"  attention : {fichier}/{brut} — {n_nulles} géométrie(s) "
                      "nulle(s) écartée(s)")
            if sel.empty:
                continue
            n_invalides = int((~sel.geometry.is_valid).sum())
            if n_invalides:
                sel["geometry"] = sel.geometry.apply(
                    lambda g: g if g.is_valid else make_valid(g))
                reparees += n_invalides
            sel["type_source"] = brut
            conflit = geoms_par_entite[entite] == {"point", "autre"}
            couche = f"{entite}_pts" if (est_point and conflit) else entite
            mode = "a" if couche in couches_ecrites else "w"
            sel.to_file(gpkg, layer=couche, driver="GPKG", mode=mode)
            couches_ecrites.add(couche)
            recap[couche] += len(sel)
            details.setdefault(couche, []).append(f"{brut} {len(sel)}")
            total += len(sel)

    for couche in sorted(recap):
        print(f"  {couche:26s} {recap[couche]:6d}  ({' + '.join(details[couche])})")
    print(f"\n{gpkg}  —  {len(recap)} couches, {total} entités, "
          f"{reparees} géométrie(s) réparée(s), {nulles_ecartees} nulle(s) écartée(s), "
          f"EPSG:{cfg['crs']}")
    print(f"Sorties : {gpkg}")


if __name__ == "__main__":
    main()
