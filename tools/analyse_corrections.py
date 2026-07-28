"""Boucle d'amélioration du recalage : confronte les décisions humaines de l'app
(recalage_decisions_<zone>.yaml) au GPKG recalé, typologie des corrections et
suggestions de paramètres par couche (spec §4).

Usage : python analyse_corrections.py <decisions.yaml> <gpkg_recale> [--out <yaml>]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import yaml
from shapely import wkt


def _distances(g1, g2, n=30):
    """Distances de n points curvilignes de g1 à g2."""
    return np.array([g1.interpolate(t, normalized=True).distance(g2)
                     for t in np.linspace(0.02, 0.98, n)])


def typologie_edition(editee, recale, origine):
    """Classe une correction manuelle par rapport au recalé et à l'origine."""
    d_rec = _distances(editee, recale)
    d_ori = _distances(editee, origine)
    if d_rec.mean() < 0.5:
        return "micro_retouche", float(d_rec.mean())
    if d_ori.mean() < d_rec.mean():  # l'utilisateur est reparti de l'origine
        return "recalage_nuisible", float(d_rec.mean())
    if d_rec.std() < 0.75:  # décalage quasi constant : le signal était plus loin
        return "translation_residuelle", float(d_rec.mean())
    return "retouche_forme", float(d_rec.mean())


def analyser(decisions_path, gpkg_path):
    decisions = yaml.safe_load(Path(decisions_path).read_text(encoding="utf-8")) or {}
    lignes = {}
    for couche in (n for n, _ in pyogrio.list_layers(str(gpkg_path))):
        gdf = gpd.read_file(gpkg_path, layer=couche)
        if "id_recalage" not in gdf.columns:
            continue
        for _, l in gdf.iterrows():
            lignes[l["id_recalage"]] = (couche, l)

    par_couche = {}
    inconnues = []
    for id_, d in sorted(decisions.items()):
        if id_ not in lignes:
            inconnues.append(id_)
            continue
        couche, l = lignes[id_]
        r = par_couche.setdefault(couche, {"decisions": Counter(),
                                           "typologie": Counter(),
                                           "distances_edition_m": [],
                                           "original_malgre_nets": 0})
        r["decisions"][d["decision"]] += 1
        if d["decision"] == "editee":
            typ, dist = typologie_edition(wkt.loads(d["geometrie_editee"]),
                                          l.geometry, wkt.loads(l["geom_origine"]))
            r["typologie"][typ] += 1
            r["distances_edition_m"].append(round(dist, 2))
        elif d["decision"] == "original" and l["pts_nets_pct"] >= 60:
            r["original_malgre_nets"] += 1

    rapport = {"decisions_total": len(decisions),
               "ids_inconnus": inconnues, "couches": {}, "suggestions": []}
    for couche, r in sorted(par_couche.items()):
        dists = r["distances_edition_m"]
        rapport["couches"][couche] = {
            "decisions": dict(r["decisions"]), "typologie": dict(r["typologie"]),
            "distance_edition_mediane_m": (round(float(np.median(dists)), 2)
                                           if dists else None),
        }
        n_ed = r["decisions"]["editee"]
        sug = rapport["suggestions"]
        if n_ed and r["typologie"]["translation_residuelle"] / n_ed > 0.3:
            sug.append(f"{couche} : décalage résiduel constant fréquent — "
                       "augmenter fenetre_m ou vérifier la polarité")
        if n_ed and r["typologie"]["recalage_nuisible"] / n_ed > 0.2:
            sug.append(f"{couche} : capture probable d'une structure voisine — "
                       "réduire fenetre_m ou relever seuil_ambiguite")
        if r["original_malgre_nets"] >= 5:
            sug.append(f"{couche} : {r['original_malgre_nets']} lignes nettes "
                       "rejetées — relever seuil_contraste ou poids_derivee")
        rejets = r["decisions"]["original"] + r["decisions"]["exclue"]
        total = sum(r["decisions"].values())
        if total >= 20 and rejets / total < 0.05:
            sug.append(f"{couche} : {rejets}/{total} rejets seulement — les "
                       "seuils a_revoir peuvent être assouplis (moins de revue)")
    return rapport


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("decisions")
    ap.add_argument("gpkg")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rapport = analyser(args.decisions, args.gpkg)
    out = Path(args.out) if args.out else \
        Path(args.decisions).with_name("analyse_corrections.yaml")
    out.write_text(yaml.safe_dump(rapport, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    print(yaml.safe_dump(rapport, allow_unicode=True, sort_keys=False))
    print(f"Sorties :\n  {out}")


if __name__ == "__main__":
    main()
