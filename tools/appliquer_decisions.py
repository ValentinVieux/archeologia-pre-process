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


def appliquer(gpkg_source, gpkg_recale, decisions_path, out, gpkg_reference=None,
              defaut_original=()):
    """gpkg_reference : version du GPKG recalé sur laquelle la revue a été
    faite — les décisions 'recale' y prennent leur géométrie (l'humain a
    validé CETTE version ; les re-runs d'algo n'engagent que le non décidé).
    defaut_original : couches dont les lignes NON décidées reviennent à la
    géométrie d'origine (recalage jugé inadapté en revue, ex. voies larges) —
    marquées decision_humaine='auto_original'."""
    decisions = yaml.safe_load(Path(decisions_path).read_text(encoding="utf-8")) or {}
    couches_recalees = {n for n, _ in pyogrio.list_layers(str(gpkg_recale))}
    reference = {}
    if gpkg_reference:
        for couche in (n for n, _ in pyogrio.list_layers(str(gpkg_reference))):
            for _, l in gpd.read_file(gpkg_reference, layer=couche).iterrows():
                if "id_recalage" in l:
                    reference[l["id_recalage"]] = l.geometry
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
            decision = d.get("decision",
                             "auto_original" if couche in defaut_original
                             else "auto")
            if decision == "exclue":
                garder.append(False)
                geoms.append(None)
                decs.append(decision)
                continue
            garder.append(True)
            decs.append(decision)
            if decision == "editee":
                geoms.append(wkt.loads(d["geometrie_editee"]))
            elif decision in ("original", "auto_original"):
                geoms.append(wkt.loads(l["geom_origine"]))
            elif decision == "recale" and l["id_recalage"] in reference:
                geoms.append(reference[l["id_recalage"]])  # version validée
            else:  # recale accepté (sans référence) ou non décidé
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
    ap.add_argument("--recale-depuis", default=None, dest="reference",
                    help="GPKG recalé sur lequel la revue a été faite : les "
                         "décisions 'recale' y prennent leur géométrie")
    ap.add_argument("--defaut-original", default="", dest="defaut_original",
                    help="couches (séparées par des virgules) dont les lignes "
                         "NON décidées reviennent à la géométrie d'origine")
    args = ap.parse_args()
    out = args.out or str(Path(args.recale).with_name(
        Path(args.recale).stem.replace("_recale", "_final") + ".gpkg"))
    for chemin, nom in ((args.source, "source"), (args.recale, "recalé"),
                        (out, "--out")):
        _refuser_drive(chemin, nom)
    if args.reference:
        _refuser_drive(args.reference, "--recale-depuis")
    out, comptes = appliquer(
        args.source, args.recale, args.decisions, out, args.reference,
        {c for c in args.defaut_original.split(",") if c})
    for couche, c in comptes.items():
        print(f"{couche} : {c}")
    print(f"Sorties :\n  {out}")


if __name__ == "__main__":
    main()
