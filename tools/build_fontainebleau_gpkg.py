"""Construit training/vecteurs/fontainebleau_entites_l93.gpkg depuis la livraison
Digitalisation ONF (Digit_Ligne/Point/Polygone, EPSG:2154 natif).

Mappings entité <- valeurs du champ Type validés par l'audit du 2026-07-27
(taxonomy/aliases.yaml, source_dataset 77_fontainebleau). Une couche par entité et
par type de géométrie : quand une entité existe en points ET en polygones (mare,
butte), les points vont dans une couche séparée suffixée `_pts` — aucune
transformation de géométrie dans ce GPKG de référence. Les géométries invalides
sont réparées par make_valid (compte rapporté).

Usage :
    .venv\\Scripts\\python.exe tools\\build_fontainebleau_gpkg.py <dossier_digitalisation> [--out <dossier>]
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
from shapely import make_valid

# valeur brute du champ Type -> id d'entité (aliases.yaml, décisions 2026-07-22/27)
MAPPINGS = {
    "Digit_Ligne.shp": {
        "parcellaire": "parcellaire",
        # split talus/fosse (décision utilisateur 2026-07-27) : deux classes
        # distinctes, la fusion talus_fosse ne vaut que pour les sources qui ne
        # distinguent pas (Haye fossébutte)
        "fosse": "fosse",
        "talus": "talus",
        "cheminement": "chemin_creux",
        "tranchee": "tranchees_et_boyaux",
    },
    "Digit_Point.shp": {
        "depression": "circular_depression",
        "mare": "mare",
        "Trous d'obus": "cratere",
        "butte": "butte",
    },
    "Digit_Polygone.shp": {
        "chaos rocheux": "chaos_rocheux",
        "mare": "mare",
        "zone extraction": "extraction",
        "sable": "zone_sableuse",
        "zone perturbee": "zone_perturbee",
        "indice archeo": "indice_archeo",
        "forage": "forage",
        "butte": "butte",
        "enclos": "enclos",
        "amenagement militaire": "amenagement_militaire",
        "parquet": "parquet",
    },
}
# entités présentes dans plusieurs fichiers : les points partent en couche `_pts`
SUFFIXE_POINTS = {"mare", "butte"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="dossier contenant Digit_Ligne/Point/Polygone.shp")
    p.add_argument("--out", default=None, help="dossier de sortie (défaut : source)")
    args = p.parse_args()
    source = Path(args.source)
    sortie = Path(args.out) if args.out else source
    sortie.mkdir(parents=True, exist_ok=True)
    gpkg = sortie / "fontainebleau_entites_l93.gpkg"
    if gpkg.exists():
        gpkg.unlink()  # reconstruit intégralement (sorties régénérables)

    total, reparees = 0, 0
    recap = []
    for fichier, mapping in MAPPINGS.items():
        gdf = gpd.read_file(source / fichier)
        if gdf.crs is None or gdf.crs.to_epsg() != 2154:
            sys.exit(f"{fichier} : CRS {gdf.crs} inattendu (EPSG:2154 requis)")
        inconnues = set(gdf["Type"].dropna()) - set(mapping)
        if inconnues:
            sys.exit(f"{fichier} : valeurs Type non auditées {sorted(inconnues)} — "
                     "relancer /audit-dataset avant de reconstruire")
        est_point = fichier == "Digit_Point.shp"
        for brut, entite in mapping.items():
            sel = gdf[gdf["Type"] == brut].copy()
            if sel.empty:
                continue
            n_invalides = int((~sel.geometry.is_valid).sum())
            if n_invalides:
                sel["geometry"] = sel.geometry.apply(
                    lambda g: g if g.is_valid else make_valid(g))
                reparees += n_invalides
            sel["type_source"] = brut
            couche = (f"{entite}_pts" if est_point and entite in SUFFIXE_POINTS
                      else entite)
            mode = "a" if couche in [r[0] for r in recap] else "w"
            sel.to_file(gpkg, layer=couche, driver="GPKG", mode=mode)
            recap.append((couche, brut, len(sel)))
            total += len(sel)

    couches = {}
    for couche, brut, n in recap:
        couches.setdefault(couche, []).append((brut, n))
    for couche in sorted(couches):
        details = " + ".join(f"{brut} {n}" for brut, n in couches[couche])
        print(f"  {couche:24s} {sum(n for _, n in couches[couche]):6d}  ({details})")
    print(f"\n{gpkg}  —  {len(couches)} couches, {total} entités, "
          f"{reparees} géométrie(s) réparée(s), EPSG:2154")
    print(f"Sorties : {gpkg}")


if __name__ == "__main__":
    main()
