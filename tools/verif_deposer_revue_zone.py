"""Contrôleur indépendant (SANS GPU) de tools/deposer_revue_zone.py — mêmes arguments, plus --marge.

Recalcule depuis les FICHIERS : couches présentes = attendues, EPSG:2154 partout ; chaque couche
revue = mêmes géométries que la revue DANS LE MÊME ORDRE (fid livrable i = i-ème entité de la revue),
schéma attributaire de la revue préservé (+ uid/decision_humaine/geom_origine) ; `decision_humaine`
valide, `uid` non vide et unique ; `geom_origine` présente ssi inchangé/modifié et ÉGALE à une
géométrie de la référence (c'est elle qui désigne l'entité de référence, prise au plus une fois) ;
si la référence porte ce uid, il doit désigner la même entité ; inchangé = bounds identiques
(±0,01 m) ; modifié = bounds différents, et IoU ≥ 0,5 sauf paire à uid partagé ; inchangés recomptés
indépendamment (appariement injectif par tolérance) ; inchangé + modifié + supprime_qgis = effectif
de référence ; `_supprimees` : motif ∈ {supprime_qgis, doublon}, supprime_qgis ⊂ référence non
appariée, doublon = couche `<c>_doublons` de la revue ; SAM : polygones de la revue moins ceux sans
boîte, `fid_source` = fid LIVRABLE de la boîte qui les contient (élargie de --marge, RÉPÉTER celle
du producteur SAM), remap cohérent avec l'ordre de la revue ; recopiées : effectif, géométries,
schéma et valeurs identiques à la source. Verdict CONFORME / NON CONFORME (exit 1).

    .venv/Scripts/python.exe tools/verif_deposer_revue_zone.py <gpkg_revue> <sortie_v3.gpkg>
        --couches c1 c2=ref2 [--reference <gpkg_v2>] [--sam c1_sam] [--copier <gpkg>:<couche> ...] [--marge 0]
"""
import argparse
import hashlib
import sys

import pyogrio
from shapely import wkt
from shapely.geometry import box

TOL = 0.01


def vide(x):
    return x is None or (isinstance(x, float) and x != x) or x == ""


def sha(geoms):
    return hashlib.sha256(b"".join(sorted(g.wkb for g in geoms))).hexdigest()[:16]


def bounds_egaux(a, b):
    return all(abs(x - y) <= TOL for x, y in zip(a, b))


def iou_boites(a, b):
    ba, bb = box(*a.bounds), box(*b.bounds)
    u = ba.union(bb).area
    return ba.intersection(bb).area / u if u else 0.0


def crs_ok(df, nom, pb):
    if df.crs is None or df.crs.to_epsg() != 2154:
        pb.append(f"{nom} : CRS {df.crs} ≠ EPSG:2154")


def schema(gpkg, couche):
    i = pyogrio.read_info(gpkg, layer=couche)
    return list(zip(list(i["fields"]), [str(d) for d in i["dtypes"]]))


def parser_couche(spec, couches_ref, reference_donnee):
    c, sep, r = spec.partition("=")
    if sep:
        return c, (r if r in couches_ref else None), bool(r) and r not in couches_ref
    return c, (c if c in couches_ref else None), reference_donnee and c not in couches_ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpkg_revue")
    ap.add_argument("sortie")
    ap.add_argument("--couches", nargs="+", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--sam", nargs="*", default=[])
    ap.add_argument("--copier", nargs="*", default=[], metavar="GPKG:COUCHE")
    ap.add_argument("--marge", type=float, default=0.0, help="marge (m par côté) du producteur SAM, tolérance de tenue dans la boîte")
    a = ap.parse_args()
    pb = []
    presentes = {n for n, _ in pyogrio.list_layers(a.sortie)}
    couches_revue = {n for n, _ in pyogrio.list_layers(a.gpkg_revue)}
    couches_ref = {n for n, _ in pyogrio.list_layers(a.reference)} if a.reference else set()
    couches = []
    for s in a.couches:
        c, cr, faux = parser_couche(s, couches_ref, a.reference is not None)
        if faux:
            pb.append(f"{s} : couche de référence introuvable dans {sorted(couches_ref)}")
        couches.append((c, cr))
    attendues = {c for c, _ in couches} | set(a.sam) | {s.rpartition(":")[2] for s in a.copier}
    remaps = {}

    for c, cr in couches:
        avant = len(pb)
        rev = pyogrio.read_dataframe(a.gpkg_revue, layer=c, fid_as_index=True)
        v3 = pyogrio.read_dataframe(a.sortie, layer=c, fid_as_index=True)
        crs_ok(v3, c, pb)
        if len(v3) != len(rev):
            pb.append(f"{c} : effectif {len(v3)} vs revue {len(rev)}")
        elif list(v3.index) != list(range(1, len(rev) + 1)) or any(not g.equals_exact(h, 1e-6) for g, h in zip(v3.geometry, rev.geometry)):
            pb.append(f"{c} : fid ou ordre des géométries ≠ revue (attendu fid 1..N dans l'ordre de la revue)")
        for col in ("uid", "decision_humaine", "geom_origine"):
            if col not in v3.columns:
                pb.append(f"{c} : colonne {col} absente")
        sr, sv = schema(a.gpkg_revue, c), schema(a.sortie, c)
        ajoutees = [(n, "object") for n in ("uid", "decision_humaine", "geom_origine") if n not in dict(sr)]  # réécrites en place si déjà là (génération suivante)
        if sv[:len(sr)] != sr or [n for n, _ in sv[len(sr):]] != [n for n, _ in ajoutees]:
            pb.append(f"{c} : schéma ≠ revue + colonnes de décision ({sv} vs {sr + ajoutees})")
        if len(pb) > avant:
            continue
        remaps[c] = {f: i + 1 for i, f in enumerate(rev.index)}
        uids = list(v3["uid"])
        if any(vide(u) for u in uids):
            pb.append(f"{c} : {sum(vide(u) for u in uids)} uid vides")
        if len(set(uids)) != len(uids):
            pb.append(f"{c} : uid non uniques ({len(uids) - len(set(uids))} doublons)")
        decs = list(v3["decision_humaine"])
        mauvaises = set(decs) - {"inchangé", "modifié", "ajouté"}
        if mauvaises:
            pb.append(f"{c} : decision_humaine inconnue {mauvaises}")
        n = {d: decs.count(d) for d in ("inchangé", "modifié", "ajouté")}
        ref = pyogrio.read_dataframe(a.reference, layer=cr, fid_as_index=True) if cr else None
        ref_uid = {}
        if ref is not None and "uid" in ref.columns:
            for r, u in zip(ref.index, ref["uid"]):
                if not vide(u):
                    ref_uid.setdefault(u, []).append(r)
        pris = set()
        for f, d, u, go, g in zip(v3.index, decs, uids, v3["geom_origine"], v3.geometry):
            if d == "ajouté":
                if not vide(go):
                    pb.append(f"{c} fid {f} : ajouté avec geom_origine")
                elif u != f"revue:{rev.index[f - 1]}" and vide(rev["uid"].iloc[f - 1] if "uid" in rev.columns else None):
                    pb.append(f"{c} fid {f} : uid {u!r} attendu revue:{rev.index[f - 1]}")
                continue
            if ref is None:
                pb.append(f"{c} fid {f} : {d} sans référence"); continue
            if vide(go):
                pb.append(f"{c} fid {f} : {d} sans geom_origine"); continue
            g0 = wkt.loads(go)
            cands = [ref.index[k] for k in ref.sindex.query(box(*g0.bounds).buffer(TOL), predicate="intersects")
                     if ref.index[k] not in pris and g0.equals_exact(ref.geometry[ref.index[k]], 1e-6)]
            if not cands:
                pb.append(f"{c} fid {f} : geom_origine ne correspond à aucune entité de référence libre"); continue
            par_uid = u in ref_uid
            r = next((k for k in cands if k in ref_uid.get(u, [])), None) if par_uid else cands[0]
            if r is None:
                pb.append(f"{c} fid {f} : uid {u!r} désigne une autre entité que geom_origine"); continue
            pris.add(r)
            if d == "inchangé" and not bounds_egaux(g.bounds, ref.geometry[r].bounds):
                pb.append(f"{c} fid {f} : inchangé mais bounds différents de la référence {r}")
            if d == "modifié" and (bounds_egaux(g.bounds, ref.geometry[r].bounds) or (not par_uid and iou_boites(g, ref.geometry[r]) < 0.5)):
                pb.append(f"{c} fid {f} : modifié mais bounds identiques, ou IoU < 0,5 sans uid partagé")
        sup = pyogrio.read_dataframe(a.sortie, layer=f"{c}_supprimees") if f"{c}_supprimees" in presentes else None
        doublons = pyogrio.read_dataframe(a.gpkg_revue, layer=f"{c}_doublons") if f"{c}_doublons" in couches_revue else None
        if sup is not None:
            attendues.add(f"{c}_supprimees")
            crs_ok(sup, f"{c}_supprimees", pb)
            if "motif" not in sup.columns or set(sup["motif"]) - {"supprime_qgis", "doublon"}:
                pb.append(f"{c}_supprimees : motif absent ou inconnu")
        n_sq = int((sup["motif"] == "supprime_qgis").sum()) if sup is not None and "motif" in sup.columns else 0
        n_db = int((sup["motif"] == "doublon").sum()) if sup is not None and "motif" in sup.columns else 0
        if doublons is None and n_db:
            pb.append(f"{c}_supprimees : {n_db} doublons sans couche {c}_doublons dans la revue")
        if doublons is not None and (n_db != len(doublons) or sha(sup[sup['motif'] == 'doublon'].geometry) != sha(doublons.geometry)):
            pb.append(f"{c}_supprimees : doublons ≠ couche {c}_doublons de la revue ({n_db} vs {len(doublons)})")
        if ref is None:
            if n["ajouté"] != len(v3):
                pb.append(f"{c} : sans référence, tout doit être ajouté ({n})")
            if n_sq:
                pb.append(f"{c}_supprimees : {n_sq} supprime_qgis sans référence")
        else:
            libres, n_inch = set(ref.index), 0  # recompte injectif des inchangés par tolérance
            for g in v3.geometry:
                for k in ref.sindex.query(box(*g.bounds).buffer(TOL), predicate="intersects"):
                    r = ref.index[k]
                    if r in libres and bounds_egaux(g.bounds, ref.geometry[r].bounds):
                        libres.discard(r); n_inch += 1
                        break
            if n_inch != n["inchangé"]:
                pb.append(f"{c} : inchangés recomptés {n_inch} vs déclarés {n['inchangé']}")
            # les doublons portant un uid de la référence sont retirés de la couche sans être « supprimés dans QGIS »
            n_db_ref = int(doublons["uid"].isin(ref["uid"]).sum()) if doublons is not None and "uid" in doublons.columns and "uid" in ref.columns else 0
            if n["inchangé"] + n["modifié"] + n_sq + n_db_ref != len(ref):
                pb.append(f"{c} : inchangé {n['inchangé']} + modifié {n['modifié']} + supprime_qgis {n_sq} + doublons de référence {n_db_ref} ≠ référence {len(ref)}")
            if n_sq:
                wk_ref = {ref.geometry[r].wkb for r in ref.index if r not in pris}
                if any(g.wkb not in wk_ref for g in sup[sup["motif"] == "supprime_qgis"].geometry):
                    pb.append(f"{c}_supprimees : supprime_qgis ∉ référence non appariée")
        print(f"{c}: {len(v3)} — {n}" + (f" [référence {cr}]" if cr else " [sans référence]")
              + (f" ; supprimées {len(sup)} (supprime_qgis {n_sq}, doublon {n_db})" if sup is not None else ""))

    for c in a.sam:
        avant = len(pb)
        base = c[:-4] if c.endswith("_sam") else c
        src = pyogrio.read_dataframe(a.gpkg_revue, layer=c)
        v3 = pyogrio.read_dataframe(a.sortie, layer=c)
        crs_ok(v3, c, pb)
        if base not in remaps:
            pb.append(f"{c} : couche de boîtes {base} non vérifiée (absente de --couches ou en défaut)"); continue
        boites_rev = pyogrio.read_dataframe(a.gpkg_revue, layer=base, fid_as_index=True)
        boites_v3 = pyogrio.read_dataframe(a.sortie, layer=base, fid_as_index=True)
        attendu = []  # (géométrie, fid livrable attendu) dans l'ordre de la revue, orphelins écartés
        for f, g in zip(src["fid_source"], src.geometry):
            if not vide(f) and int(f) in boites_rev.index:
                attendu.append((g, remaps[base][int(f)]))
        n_orphelins = len(src) - len(attendu)
        if len(v3) != len(attendu) or any(not g.equals_exact(h, 1e-6) for g, (h, _) in zip(v3.geometry, attendu)):
            pb.append(f"{c} : effectif {len(v3)} vs revue {len(src)} moins {n_orphelins} sans boîte, ou géométries/ordre différents")
        if len(pb) > avant:
            continue
        faux_remap, hors = 0, 0
        for f, g, (_, cible) in zip(v3["fid_source"], v3.geometry, attendu):
            if vide(f) or int(f) != cible:
                faux_remap += 1
            elif int(f) not in boites_v3.index or not box(*boites_v3.geometry[int(f)].bounds).buffer(a.marge + 0.5).contains(g):
                hors += 1
        if faux_remap or hors:
            pb.append(f"{c} : {faux_remap} fid_source mal remappés, {hors} polygones hors de leur boîte livrable (marge {a.marge} m)")
        print(f"{c}: {len(v3)} (sam ; {n_orphelins} sans boîte écartés)")

    for spec in a.copier:
        gp, _, c = spec.rpartition(":")
        src = pyogrio.read_dataframe(gp, layer=c)
        v3 = pyogrio.read_dataframe(a.sortie, layer=c)
        crs_ok(v3, c, pb)
        if len(v3) != len(src) or sha(v3.geometry) != sha(src.geometry):
            pb.append(f"{c} : géométries ≠ source {gp} ({len(v3)} vs {len(src)})"); continue
        if schema(gp, c) != schema(a.sortie, c):
            pb.append(f"{c} : schéma ≠ source ({schema(gp, c)} vs {schema(a.sortie, c)})")
        elif not src.drop(columns="geometry").reset_index(drop=True).equals(v3.drop(columns="geometry").reset_index(drop=True)):
            pb.append(f"{c} : valeurs d'attributs ≠ source")
        print(f"{c}: {len(v3)} (recopiée)")

    if presentes != attendues:
        pb.append(f"couches présentes {sorted(presentes)} ≠ attendues {sorted(attendues)}")
    if pb:
        print("NON CONFORME :\n  " + "\n  ".join(pb[:40]))
        sys.exit(1)
    print("CONFORME")


if __name__ == "__main__":
    main()
