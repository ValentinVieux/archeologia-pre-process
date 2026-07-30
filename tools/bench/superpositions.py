"""Analyse des superpositions INTER-CLASSES des détections.

Question posée : faut-il laisser les polygones de classes différentes se superposer ?
La campagne a mesuré que désactiver `remove_overlaps` gagne +0,007 de F1 longueur, mais
ce chiffre ne dit pas CE QU'ON VOIT — or l'observation de terrain est que deux classes
tracent parfois la même structure, ce qui donne deux polygones à valider pour un seul
objet.

Il faut distinguer deux situations que la métrique confond :

  DOUBLON      deux classes décrivent LE MÊME objet (fortement emboîtés, mêmes axes).
               C'est de la confusion de classe, et l'archéologue valide deux fois.
  CROISEMENT   deux structures réelles se coupent (un chemin traverse un parcellaire).
               Les supprimer serait une PERTE d'information archéologique.

Le critère de séparation est l'IoS (intersection / aire du plus petit) combiné à l'écart
d'orientation des axes principaux : un doublon est emboîté ET colinéaire ; un croisement
se touche peu ET change de direction.

    python -m tools.bench.superpositions --visuel /out/bench/visuel
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Seuils de classement, choisis puis VERIFIES par la distribution empirique (cf. sortie).
IOS_DOUBLON = 0.50      # au-dela, le plus petit est majoritairement dans le plus grand
ANGLE_DOUBLON = 30.0    # ecart d'azimut des axes principaux, en degres


def azimut(geom) -> float:
    """Orientation de l'axe principal, en degrés dans [0, 180)."""
    try:
        rect = geom.minimum_rotated_rectangle
        xs, ys = rect.exterior.coords.xy
    except Exception:
        return float("nan")
    pts = list(zip(xs, ys))[:4]
    if len(pts) < 4:
        return float("nan")
    cotes = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    (ax, ay), (bx, by) = max(cotes, key=lambda c: (c[1][0] - c[0][0]) ** 2
                                                  + (c[1][1] - c[0][1]) ** 2)
    return math.degrees(math.atan2(by - ay, bx - ax)) % 180.0


def ecart_angulaire(a: float, b: float) -> float:
    if a != a or b != b:
        return float("nan")
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def analyser(gpkg: Path, prefixe: str) -> dict:
    import geopandas as gpd
    import pyogrio
    from shapely.strtree import STRtree

    couches = [n for n, _ in pyogrio.list_layers(gpkg) if n.startswith(prefixe)]
    if not couches:
        return {}
    geoms, classes, confs = [], [], []
    for c in couches:
        g = gpd.read_file(gpkg, layer=c)
        cl = c[len(prefixe):]
        for geom, conf in zip(g.geometry, g.get("confiance", [None] * len(g))):
            if geom is None or geom.is_empty:
                continue
            geoms.append(geom)
            classes.append(cl)
            confs.append(conf)
    if not geoms:
        return {}

    arbre = STRtree(geoms)
    paires: List[dict] = []
    vus = set()
    for i, g in enumerate(geoms):
        for j in arbre.query(g):
            j = int(j)
            if j <= i or classes[j] == classes[i]:
                continue          # meme classe : deja fusionnee par merge_adjacent
            cle = (i, j)
            if cle in vus:
                continue
            vus.add(cle)
            inter = g.intersection(geoms[j])
            if inter.is_empty or inter.area <= 0:
                continue
            pa, pb = g.area, geoms[j].area
            ios = inter.area / max(min(pa, pb), 1e-9)
            paires.append({
                "classes": tuple(sorted((classes[i], classes[j]))),
                "ios": ios,
                "iou": inter.area / max(pa + pb - inter.area, 1e-9),
                "aire_inter_m2": inter.area,
                "aire_min_m2": min(pa, pb),
                "angle": ecart_angulaire(azimut(g), azimut(geoms[j])),
                "conf_min": min([c for c in (confs[i], confs[j]) if c is not None],
                                default=None),
            })

    for p in paires:
        colineaire = p["angle"] == p["angle"] and p["angle"] <= ANGLE_DOUBLON
        p["type"] = ("doublon" if (p["ios"] >= IOS_DOUBLON and colineaire)
                     else "croisement")
    return {"n_polygones": len(geoms), "paires": paires,
            "aire_totale_m2": float(sum(g.area for g in geoms))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visuel", default=os.environ.get("BENCH_OUT", "/out/bench") + "/visuel")
    ap.add_argument("--prefixe", default="nouveau_")
    ap.add_argument("--out", default=os.environ.get("BENCH_OUT", "/out/bench"))
    a = ap.parse_args()

    racine = Path(a.visuel)
    total: List[dict] = []
    par_mos: Dict[str, dict] = {}
    for gpkg in sorted(racine.glob("*/comparatif.gpkg")):
        zone = gpkg.parent.name
        r = analyser(gpkg, a.prefixe)
        if not r:
            continue
        par_mos[zone] = {"n_polygones": r["n_polygones"],
                         "n_paires": len(r["paires"]),
                         "n_doublons": sum(1 for p in r["paires"] if p["type"] == "doublon"),
                         "aire_totale_m2": r["aire_totale_m2"],
                         "aire_doublons_m2": sum(p["aire_inter_m2"] for p in r["paires"]
                                                 if p["type"] == "doublon")}
        for p in r["paires"]:
            p["mosaique"] = zone
        total.extend(r["paires"])

    if not total:
        print(f"aucune superposition inter-classes pour le prefixe {a.prefixe!r}.")
        print("Ce n'est PAS forcement une bonne nouvelle : si la configuration exportee")
        print("avait `remove_overlaps` active, la strategie `difference` les a deja toutes")
        print("supprimees — y compris les croisements reels. Verifier la config exportee")
        print("(visuel/index.json) avant d'en conclure quoi que ce soit sur le modele.")
        return 0

    n = len(total)
    dbl = [p for p in total if p["type"] == "doublon"]
    crx = [p for p in total if p["type"] == "croisement"]
    n_poly = sum(v["n_polygones"] for v in par_mos.values())
    aire_tot = sum(v["aire_totale_m2"] for v in par_mos.values())
    aire_dbl = sum(p["aire_inter_m2"] for p in dbl)

    print(f"SUPERPOSITIONS INTER-CLASSES — prefixe {a.prefixe!r}\n")
    print(f"  {n_poly} polygones, {n} paires de classes differentes qui se recouvrent")
    print(f"    doublons   (IoS>={IOS_DOUBLON} et axes a moins de {ANGLE_DOUBLON}deg) : "
          f"{len(dbl):>4}  ({100*len(dbl)/n:.0f} %)")
    print(f"    croisements                                          : {len(crx):>4}  "
          f"({100*len(crx)/n:.0f} %)")
    print(f"  aire recouverte par les doublons : {aire_dbl/1e4:.2f} ha "
          f"soit {100*aire_dbl/max(aire_tot,1e-9):.2f} % de l'aire detectee totale")

    print(f"\n  distribution de l'IoS (verifie que le seuil {IOS_DOUBLON} separe vraiment) :")
    bornes = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
    for lo, hi in zip(bornes, bornes[1:]):
        k = sum(1 for p in total if lo <= p["ios"] < hi)
        print(f"    [{lo:.1f} ; {hi:.1f})  {k:>4}  {'#' * int(40 * k / n)}")

    # Ce que `remove_overlaps` (strategie `difference`) ferait concretement : rogner le
    # polygone le moins confiant de CHAQUE paire, croisements compris. C'est le chiffre
    # qui decide, plus que le F1 : on repare 9 doublons en abimant 98 croisements reels.
    aire_crx = sum(p["aire_inter_m2"] for p in crx)
    print(f"\n  ce que `remove_overlaps` (strategie `difference`) rognerait :")
    print(f"    sur des DOUBLONS    (utile)   {len(dbl):>4} paires, {aire_dbl/1e4:>6.2f} ha")
    print(f"    sur des CROISEMENTS (nuisible){len(crx):>4} paires, {aire_crx/1e4:>6.2f} ha")
    if len(dbl):
        print(f"    soit {len(crx)/len(dbl):.0f} croisements reels abimes pour 1 doublon corrige")

    print(f"\n  couples de classes les plus concernes :")
    for (c1, c2), k in Counter(p["classes"] for p in total).most_common(8):
        kd = sum(1 for p in total if p["classes"] == (c1, c2) and p["type"] == "doublon")
        print(f"    {c1:<14} x {c2:<14} {k:>4} paires, dont {kd:>3} doublons")

    print(f"\n  par mosaique :")
    print(f"    {'mosaique':<44}{'polys':>7}{'paires':>8}{'doublons':>10}{'% aire':>8}")
    for z, v in sorted(par_mos.items()):
        pct = 100 * v["aire_doublons_m2"] / max(v["aire_totale_m2"], 1e-9)
        print(f"    {z:<44}{v['n_polygones']:>7}{v['n_paires']:>8}"
              f"{v['n_doublons']:>10}{pct:>8.2f}")

    p = Path(a.out) / f"superpositions_{a.prefixe.rstrip('_')}.json"
    p.write_text(json.dumps({
        "prefixe": a.prefixe, "seuils": {"ios": IOS_DOUBLON, "angle_deg": ANGLE_DOUBLON},
        "n_polygones": n_poly, "n_paires": n,
        "n_doublons": len(dbl), "n_croisements": len(crx),
        "part_aire_doublons": aire_dbl / max(aire_tot, 1e-9),
        "par_mosaique": par_mos,
        "couples": {f"{a_}|{b_}": k for (a_, b_), k in
                    Counter(p_["classes"] for p_ in total).items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
