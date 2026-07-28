"""Applique les décisions de revue au GPKG recalé → GPKG final d'entraînement
(spec recalage §5). Les couches non recalées de la source sont copiées telles
quelles ; sur les couches recalées : editee → géométrie humaine, original →
géométrie d'origine, exclue → retirée (comptée), recale/non décidée → recalé.
`geom_origine` est conservée sur chaque ligne (exigence : ne jamais perdre le
tracé d'origine) ; seule la géométrie active part en découpe/upload.

Usage : python appliquer_decisions.py <gpkg_source> <gpkg_recale>
            <decisions.yaml> [--out <gpkg_final>]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pyogrio
import yaml
from shapely import wkt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slice_zone import _refuser_drive


def appliquer(gpkg_source, gpkg_recale, decisions_path, out):
    decisions = yaml.safe_load(Path(decisions_path).read_text(encoding="utf-8")) or {}
    couches_recalees = {n for n, _ in pyogrio.list_layers(str(gpkg_recale))}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    comptes = {}
    for couche, _ in pyogrio.list_layers(str(gpkg_source)):
        if couche not in couches_recalees:  # couche non recalée : verbatim
            gpd.read_file(gpkg_source, layer=couche).to_file(out, layer=couche,
                                                             driver="GPKG")
            comptes[couche] = "copiée"
            continue
        gdf = gpd.read_file(gpkg_recale, layer=couche)
        geoms, decs, garder = [], [], []
        for _, l in gdf.iterrows():
            d = decisions.get(l["id_recalage"], {})
            decision = d.get("decision", "auto")
            if decision == "exclue":
                garder.append(False)
                geoms.append(None)
                decs.append(decision)
                continue
            garder.append(True)
            decs.append(decision)
            if decision == "editee":
                geoms.append(wkt.loads(d["geometrie_editee"]))
            elif decision == "original":
                geoms.append(wkt.loads(l["geom_origine"]))
            else:  # recale accepté ou non décidé (auto_ok/sans_signal)
                geoms.append(l.geometry)
        gdf = gdf.assign(decision_humaine=decs)
        gdf.geometry = geoms
        final = gdf[garder].copy()
        if final.empty:
            sys.exit(f"{couche} : toutes les lignes exclues — décision humaine "
                     "explicite requise pour retirer une couche entière")
        final.to_file(out, layer=couche, driver="GPKG")
        comptes[couche] = dict(Counter(decs))
    return out, comptes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("recale")
    ap.add_argument("decisions")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(Path(args.recale).with_name(
        Path(args.recale).stem.replace("_recale", "_final") + ".gpkg"))
    for chemin, nom in ((args.source, "source"), (args.recale, "recalé"),
                        (out, "--out")):
        _refuser_drive(chemin, nom)
    out, comptes = appliquer(args.source, args.recale, args.decisions, out)
    for couche, c in comptes.items():
        print(f"{couche} : {c}")
    print(f"Sorties :\n  {out}")


if __name__ == "__main__":
    main()
