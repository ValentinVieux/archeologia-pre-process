"""Staging du GPKG d'entités v3 d'une zone après revue humaine (skill /revue-annotations, § Dépôt).

Depuis le GPKG de revue (édité dans QGIS) : chaque couche revue (--couches, syntaxe `c`, `c=couche_ref`
si la couche a été renommée pendant la revue, `c=` sans référence) est recopiée avec `decision_humaine`
∈ {inchangé, modifié, ajouté}, `uid` et `geom_origine` (WKT de la géométrie de référence), jugées
contre la couche de référence (--reference = dernière version déposée) : par `uid` quand les deux
côtés en portent un (uid vides -> géométrie ; une paire à uid est « modifié » quel que soit son
déplacement), sinon BOÎTE CONTRE BOÎTE : bounds identiques (±0,01 m) = inchangé ; recouvrement de
boîtes (IoU ≥ 0,5, appariement glouton) = modifié ; sinon ajouté (uid `revue:<fid>`). Les entités de
référence non retrouvées vont dans `<c>_supprimees` (motif `supprime_qgis`), avec les entités d'une
éventuelle couche `<c>_doublons` de la revue (motif `doublon`) — jamais dans une couche d'entraînement.
Les fid du livrable sont réattribués 1..N : les couches --sam sont recopiées avec `fid_source`
REMAPPÉ sur ces nouveaux fid (polygones sans boîte écartés et comptés). --copier <gpkg>:<couche>
recopie une couche d'un autre GPKG. Schémas préservés (un entier à NULL reste entier), sortie
EPSG:2154, refus d'écraser, refus de G:, couches de la revue non reprises listées.

    .venv/Scripts/python.exe tools/deposer_revue_zone.py <gpkg_revue> <sortie_v3.gpkg>
        --couches c1 c2=ref2 [--reference <gpkg_v2>] [--sam c1_sam] [--copier <gpkg>:<couche> ...]
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slice_zone import _refuser_drive  # noqa: E402

EPSG = 2154
TOL = 0.01  # m, égalité de bounds
IOU_MIN = 0.5


def vide(x):
    """NULL GPKG lu None, NaN (colonne mixte, pandas 3) ou chaîne vide."""
    return x is None or (isinstance(x, float) and x != x) or x == ""


def parser_couche(spec, couches_ref, reference_donnee):
    """'c' | 'c=ref' | 'c=' -> (c, couche de référence ou None) ; refuse une couche introuvable."""
    c, sep, r = spec.partition("=")
    if sep:
        if r and r not in couches_ref:
            sys.exit(f"{spec} : couche {r} absente de la référence {sorted(couches_ref)}")
        return c, (r or None)
    if reference_donnee and c not in couches_ref:
        sys.exit(f"couche {c} absente de la référence {sorted(couches_ref)} : préciser {c}=<couche_ref> "
                 f"(renommée pendant la revue) ou {c}= (sans référence)")
    return c, (c if c in couches_ref else None)


def lire(gpkg, couche, nom, fid_as_index=False):
    """Couche en EPSG:2154, entiers à NULL recastés (pyogrio les lit en float64)."""
    g = pyogrio.read_dataframe(gpkg, layer=couche, fid_as_index=fid_as_index)
    if g.crs is None:
        sys.exit(f"{nom} : CRS absent")
    if g.crs.to_epsg() != EPSG:
        g = g.to_crs(EPSG)
    info = pyogrio.read_info(gpkg, layer=couche)
    for col, dt in zip(info["fields"], info["dtypes"]):
        dt = str(dt)
        if dt.startswith("int") and col in g.columns and g[col].dtype.kind == "f":
            g[col] = g[col].astype("Int64" if dt == "int64" else "Int32")
    return g


def bounds_egaux(a, b):
    return all(abs(x - y) <= TOL for x, y in zip(a, b))


def iou_boites(a, b):
    ba, bb = box(*a.bounds), box(*b.bounds)
    u = ba.union(bb).area
    return ba.intersection(bb).area / u if u else 0.0


def apparier(revue, ref):
    """-> listes (decision, uid, geom_origine) alignées sur `revue`, et fids de `ref` non retrouvés.
    Les uid existants de la revue sont conservés ; ceux qui manquent reçoivent ref:<fid>/revue:<fid>."""
    n = len(revue)
    uid_revue = list(revue["uid"]) if "uid" in revue.columns else [None] * n
    decision, uid, origine = ["ajouté"] * n, [None] * n, [None] * n
    pos = {f: i for i, f in enumerate(revue.index)}
    if ref is None or ref.empty:
        return decision, [u if not vide(u) else f"revue:{f}" for u, f in zip(uid_revue, revue.index)], origine, []
    libres = set(ref.index)

    def prendre(i, r, d):
        decision[i], origine[i] = d, ref.geometry[r].wkt
        uid[i] = uid_revue[i] if not vide(uid_revue[i]) else f"ref:{r}"
        libres.discard(r)

    # 1) uid partagé (uid vides exclus : ils passent par la géométrie)
    if "uid" in ref.columns:
        par_uid = {}
        for r, u in zip(ref.index, ref["uid"]):
            if not vide(u):
                par_uid.setdefault(u, []).append(r)
        for f in revue.index:
            u = uid_revue[pos[f]]
            if vide(u):
                continue
            for r in par_uid.get(u, []):
                if r in libres:
                    prendre(pos[f], r, "inchangé" if bounds_egaux(revue.geometry[f].bounds, ref.geometry[r].bounds) else "modifié")
                    break
    # 2) bounds identiques à TOL près (index spatial, pas d'arrondi)
    sidx = ref.sindex
    for f in revue.index:
        i = pos[f]
        if origine[i] is not None:
            continue
        g = revue.geometry[f]
        for k in sidx.query(box(*g.bounds).buffer(TOL), predicate="intersects"):
            r = ref.index[k]
            if r in libres and bounds_egaux(g.bounds, ref.geometry[r].bounds):
                prendre(i, r, "inchangé")
                break
    # 3) recouvrement de boîtes, glouton par IoU décroissant
    candidats = []
    for f in revue.index:
        if origine[pos[f]] is not None:
            continue
        g = revue.geometry[f]
        for k in sidx.query(box(*g.bounds), predicate="intersects"):
            r = ref.index[k]
            if r in libres:
                v = iou_boites(g, ref.geometry[r])
                if v >= IOU_MIN:
                    candidats.append((v, f, r))
    for v, f, r in sorted(candidats, reverse=True):
        i = pos[f]
        if origine[i] is not None or r not in libres:
            continue
        prendre(i, r, "modifié")
    for f in revue.index:
        i = pos[f]
        if origine[i] is None:
            uid[i] = uid_revue[i] if not vide(uid_revue[i]) else f"revue:{f}"
    return decision, uid, origine, sorted(libres)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpkg_revue")
    ap.add_argument("sortie")
    ap.add_argument("--couches", nargs="+", required=True, help="couches revues : c | c=couche_ref (renommée) | c= (sans référence)")
    ap.add_argument("--reference", default=None, help="GPKG de référence (dernière version déposée)")
    ap.add_argument("--sam", nargs="*", default=[], help="couches de polygones SAM (<couche>_sam), fid_source remappé")
    ap.add_argument("--copier", nargs="*", default=[], metavar="GPKG:COUCHE", help="couches recopiées d'un autre GPKG")
    a = ap.parse_args()
    out = Path(a.sortie)
    _refuser_drive(out, "sortie")
    if out.exists():
        sys.exit(f"{out} existe déjà : refus d'écraser")
    couches_revue = {n for n, _ in pyogrio.list_layers(a.gpkg_revue)}
    couches_ref = {n for n, _ in pyogrio.list_layers(a.reference)} if a.reference else set()
    couches = [parser_couche(s, couches_ref, a.reference is not None) for s in a.couches]
    for c in [c for c, _ in couches] + a.sam:
        if c not in couches_revue:
            sys.exit(f"couche {c} absente du GPKG de revue {sorted(couches_revue)}")
    for spec in a.copier:
        if spec.rpartition(":")[2] in couches_revue:
            sys.exit(f"{spec} : la couche existe aussi dans la revue — la passer en --couches, pas en --copier")

    bilan, remaps, reprises = [], {}, set()
    for c, cr in couches:
        revue = lire(a.gpkg_revue, c, f"{c} (revue)", fid_as_index=True)
        ref = lire(a.reference, cr, f"{cr} (référence)", fid_as_index=True) if cr else None
        decision, uid, origine, supprimes = apparier(revue, ref)
        remaps[c] = {f: i + 1 for i, f in enumerate(revue.index)}  # fid revue -> fid livrable (couche neuve)
        sortie = revue.reset_index(drop=True)
        sortie["uid"] = uid
        sortie["decision_humaine"] = decision
        sortie["geom_origine"] = origine
        sortie.to_file(out, layer=c, driver="GPKG")
        reprises.add(c)
        n = {d: decision.count(d) for d in ("inchangé", "modifié", "ajouté")}
        ligne = f"{c}: {len(sortie)} (inchangé {n['inchangé']}, modifié {n['modifié']}, ajouté {n['ajouté']})" + (f" [référence {cr}]" if cr else " [sans référence]")
        morceaux, d = [], None
        if f"{c}_doublons" in couches_revue:
            d = lire(a.gpkg_revue, f"{c}_doublons", f"{c}_doublons")
            d["decision_humaine"], d["motif"] = "supprimé", "doublon"
            reprises.add(f"{c}_doublons")
            if ref is not None and "uid" in d.columns and "uid" in ref.columns:
                # un doublon retiré de la couche n'est pas un « supprimé dans QGIS » (Vosges 2026-09-08 : compté deux fois)
                uids_doublons = set(d["uid"].dropna())
                supprimes = [r for r in supprimes if ref["uid"][r] not in uids_doublons]
        if supprimes:
            s = ref.loc[supprimes].reset_index(drop=True)
            s["decision_humaine"], s["motif"] = "supprimé", "supprime_qgis"
            morceaux.append(s)
        if d is not None:
            morceaux.append(d)
        if morceaux:
            s = gpd.GeoDataFrame(pd.concat(morceaux, ignore_index=True), crs=f"EPSG:{EPSG}")
            s.to_file(out, layer=f"{c}_supprimees", driver="GPKG")
            ligne += f" ; {len(s)} -> {c}_supprimees ({', '.join(f'{v} {k}' for k, v in s['motif'].value_counts().items())})"
        bilan.append(ligne)
    for c in a.sam:
        base = c[:-4] if c.endswith("_sam") else c
        if base not in remaps:
            sys.exit(f"{c} : sa couche de boîtes {base} doit être dans --couches")
        g = lire(a.gpkg_revue, c, c)
        cible = [None if vide(f) else remaps[base].get(int(f)) for f in g["fid_source"]]
        garde = [t is not None for t in cible]
        ecartes = [None if vide(f) else int(f) for f, k in zip(g["fid_source"], garde) if not k]
        g = g[garde].copy()
        g["fid_source"] = [t for t in cible if t is not None]
        g.to_file(out, layer=c, driver="GPKG")
        reprises.add(c)
        bilan.append(f"{c}: {len(g)} (fid_source remappé sur les fid du livrable)"
                     + (f" ; {len(ecartes)} polygones ÉCARTÉS sans boîte (fid_source revue {ecartes[:10]})" if ecartes else ""))
    for spec in a.copier:
        gp, _, c = spec.rpartition(":")
        g = lire(gp, c, spec)
        g.to_file(out, layer=c, driver="GPKG")
        bilan.append(f"{c}: {len(g)} recopiée de {gp} (schéma préservé)")
    print(f"écrit {out} (EPSG:{EPSG})")
    for b in bilan:
        print("  " + b)
    oubliees = sorted(couches_revue - reprises)
    if oubliees:
        print(f"  couches de la revue NON reprises : {oubliees}")


if __name__ == "__main__":
    main()
