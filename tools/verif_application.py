"""Contrôleur indépendant de l'application des décisions (boucle de vérif).

Usage : python verif_application.py <gpkg_source> <gpkg_recale>
            <decisions.yaml> <gpkg_final> [<gpkg_reference>]
(gpkg_reference : si l'application a utilisé --recale-depuis, les décisions
'recale' doivent porter la géométrie de CETTE version — celle revue.)
"""
import sys
from pathlib import Path

import geopandas as gpd
import pyogrio
import yaml
from shapely import wkt

source, recale, decisions_p, final = (Path(a) for a in sys.argv[1:5])
decisions = yaml.safe_load(decisions_p.read_text(encoding="utf-8")) or {}
reference = {}
if len(sys.argv) > 5:
    ref_p = Path(sys.argv[5])
    for couche_r in (n for n, _ in pyogrio.list_layers(str(ref_p))):
        gdf_r = gpd.read_file(ref_p, layer=couche_r)
        if "id_recalage" not in gdf_r.columns:
            continue
        for _, l in gdf_r.iterrows():
            reference[l["id_recalage"]] = l.geometry

couches_src = {n for n, _ in pyogrio.list_layers(str(source))}
couches_rec = {n for n, _ in pyogrio.list_layers(str(recale))}
couches_fin = {n for n, _ in pyogrio.list_layers(str(final))}
assert couches_fin == couches_src, f"couches : {couches_fin ^ couches_src}"

total, exclues_total = 0, 0
for couche in sorted(couches_src):
    f = gpd.read_file(final, layer=couche)
    assert (f.crs.to_epsg() == 2154), f"{couche} : CRS {f.crs}"
    if couche not in couches_rec:  # couche copiée verbatim
        s = gpd.read_file(source, layer=couche)
        assert len(f) == len(s), f"{couche} : {len(f)} vs {len(s)}"
        total += len(f)
        continue
    r = gpd.read_file(recale, layer=couche)
    par_id = {l["id_recalage"]: l for _, l in r.iterrows()}
    exclues = {i for i, d in decisions.items()
               if d["decision"] == "exclue" and i in par_id}
    assert len(f) == len(r) - len(exclues), \
        f"{couche} : {len(f)} vs {len(r)} - {len(exclues)} exclues"
    assert not (set(f["id_recalage"]) & exclues), f"{couche} : exclue présente"
    assert f["geom_origine"].notna().all(), f"{couche} : geom_origine perdue"
    for _, l in f.iterrows():
        rec_l = par_id[l["id_recalage"]]
        assert l["geom_origine"] == rec_l["geom_origine"], \
            f"{couche}/{l['id_recalage']} : geom_origine modifiée"
        d = decisions.get(l["id_recalage"], {})
        decision = d.get("decision", "auto")
        if decision == "auto" and l["decision_humaine"] == "auto_original":
            decision = "auto_original"  # défaut de couche, pas de décision humaine
        assert l["decision_humaine"] == decision, \
            f"{couche}/{l['id_recalage']} : décision {l['decision_humaine']}"
        if decision == "editee":
            attendu = wkt.loads(d["geometrie_editee"])
        elif decision in ("original", "auto_original"):
            attendu = wkt.loads(rec_l["geom_origine"])
        elif decision == "recale" and l["id_recalage"] in reference:
            attendu = reference[l["id_recalage"]]
        else:
            attendu = rec_l.geometry
        assert l.geometry.equals(attendu), \
            f"{couche}/{l['id_recalage']} : géométrie ≠ décision '{decision}'"
    total += len(f)
    exclues_total += len(exclues)

print(f"vérification application : CONFORME — {len(couches_fin)} couches, "
      f"{total} entités, {exclues_total} exclues retirées")
