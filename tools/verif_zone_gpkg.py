"""Vérification générique d'un GPKG d'entités contre sa livraison (mapping YAML).

Usage : python verif_zone_gpkg.py <mapping.yaml> <dossier_source> <gpkg>
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pyogrio
import yaml

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("mapping", help="configs/vecteurs_<zone>.yaml (mapping de la livraison)")
_ap.add_argument("source", help="dossier de la livraison auditée")
_ap.add_argument("gpkg", help="GPKG d'entités produit par build_zone_gpkg")
_a = _ap.parse_args()
mapping, source, gpkg = Path(_a.mapping), Path(_a.source), Path(_a.gpkg)
cfg = yaml.safe_load(mapping.read_text(encoding="utf-8"))

sources = {}
geoms_par_entite = {}
for fichier, spec in cfg["sources"].items():
    gdf = gpd.read_file(source / fichier)
    sources[fichier] = gdf
    est_point = gdf.geom_type.dropna().iloc[0] in ("Point", "MultiPoint")
    entites = ([spec["entite"]] if "entite" in spec else spec["valeurs"].values())
    for entite in entites:
        geoms_par_entite.setdefault(entite, set()).add("point" if est_point else "autre")

attendu = {}  # couche -> [(fichier, valeur, n_attendu_hors_nulles)]
for fichier, spec in cfg["sources"].items():
    gdf = sources[fichier]
    est_point = gdf.geom_type.dropna().iloc[0] in ("Point", "MultiPoint")
    if "entite" in spec:  # couche entière -> type_source = nom de fichier (stem)
        paires = [(Path(fichier).stem, spec["entite"], gdf)]
    else:
        paires = [(brut, entite, gdf[gdf[spec["champ"]] == brut])
                  for brut, entite in spec["valeurs"].items()]
    for brut, entite, sel in paires:
        n = int((sel.geometry.notna() & ~sel.geometry.is_empty).sum())
        if n == 0:
            continue
        conflit = geoms_par_entite[entite] == {"point", "autre"}
        couche = f"{entite}_pts" if (est_point and conflit) else entite
        attendu.setdefault(couche, []).append((fichier, brut, n))

couches_gpkg = {nom for nom, _ in pyogrio.list_layers(gpkg)}
assert couches_gpkg == set(attendu), f"couches divergentes : {couches_gpkg ^ set(attendu)}"

total = 0
for couche, entrees in sorted(attendu.items()):
    d = gpd.read_file(gpkg, layer=couche)
    n_attendu = sum(n for _, _, n in entrees)
    assert len(d) == n_attendu, f"{couche} : {len(d)} vs {n_attendu} attendus"
    assert d.crs.to_epsg() == cfg["crs"], f"{couche} : CRS {d.crs}"
    assert d.geometry.notna().all() and d.geometry.is_valid.all(), f"{couche} : géom"
    assert set(d["type_source"]) == {b for _, b, _ in entrees}, f"{couche} : type_source"
    for fichier, brut, _ in entrees:  # 1re géométrie valide retrouvée à l'identique
        src = sources[fichier]
        spec_f = cfg["sources"][fichier]
        masque = (src.geometry.notna() & ~src.geometry.is_empty
                  & src.geometry.is_valid)
        if "champ" in spec_f:
            masque &= src[spec_f["champ"]] == brut
        cand = src[masque]
        if cand.empty:
            continue
        g0 = cand.geometry.iloc[0]
        sel = d[d["type_source"] == brut]
        assert any(g0.equals(g) for g in sel.geometry.iloc[:3000]), \
            f"{couche}/{brut} : géométrie source introuvable"
    total += len(d)

print(f"vérification GPKG : CONFORME — {len(attendu)} couches, {total} entités, "
      f"EPSG:{cfg['crs']}")
