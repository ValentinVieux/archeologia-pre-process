"""Contrôleur indépendant de l'éval outillée (boucle de vérification, SANS GPU).

Usage : python verif_courbes_eval.py <dossier_eval>
(<dossier_eval> contient metriques_eval.json + appariements.json — la sortie de
tools/courbes_eval.py)

Recalcule depuis appariements.json, par une implémentation VOLONTAIREMENT
différente (python pur, recomptage brut par seuil, enveloppe de précision par
boucle arrière — zéro code partagé avec courbes_eval), les métriques publiées
dans metriques_eval.json : global, par_classe, par_zone, par_zone_classe (bloc
additif du 2026-09-03 : absent = AVERTISSEMENT, pas de non-conformité — le
compléter par tools/completer_metriques_eval.py). Égalité exacte exigée après
arrondi à 4 décimales. Contrôles de cohérence : schémas, empreinte _meta vs
champs publiés, seuil_f1max ∈ grille, IoU des matches ≥ 0,5, sommes n_gt (par
classe = global ; par zone × classe = par zone).

L'inférence elle-même est couverte par l'autocontrôle de chargement de
courbes_eval (rappel plancher >= 0,30) ; CE contrôleur garantit que les CHIFFRES
publiés — dont les seuils de production du plugin — découlent bien des
appariements. Verdict CONFORME requis avant tout dépôt/seuil/dashboard.
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import median

RAPPORT: list[str] = []
AVERTISSEMENTS: list[str] = []


def probleme(msg: str) -> None:
    RAPPORT.append(msg)


def zone_classe_local(sous, classe, s0, s_classe):
    """Bloc zone × classe attendu (recomptage brut ; R/P None quand dénominateur nul)."""
    tp, fp, ngt = compte(sous, s0, classe)
    tp_cl, fp_cl, _ = compte(sous, s_classe, classe)
    tp_max = sum(1 for e in sous for m in e["matches"] if m[2] == classe)  # TOUS les matches
    return {"n_gt": ngt, "tp": tp, "fp": fp,
            "R": round(tp / ngt, 4) if ngt else None,
            "P": round(tp / (tp + fp), 4) if tp + fp else None,
            "R_seuil_classe": round(tp_cl / ngt, 4) if ngt else None,
            "fp_seuil_classe": fp_cl,
            "R_max": round(tp_max / ngt, 4) if ngt else None}


def grille_locale(plancher: float) -> list[float]:
    seuils, k = [], 0
    while True:
        s = round(plancher + 0.005 * k, 3)
        if s > 0.9505:
            break
        seuils.append(s)
        k += 1
    return seuils


def compte(enregs, seuil, classe=None):
    """(tp, fp, n_gt) par recomptage brut — pas de tri, pas de cumsum."""
    tp = fp = ngt = 0
    for e in enregs:
        for m in e["matches"]:
            if (classe is None or m[2] == classe) and m[0] >= seuil:
                tp += 1
        for f in e["fps"]:
            if (classe is None or f[1] == classe) and f[0] >= seuil:
                fp += 1
        ngt += (sum(1 for c in e["gt_classes"] if c == classe) if classe
                else e["n_gt"])
    return tp, fp, ngt


def prf_local(enregs, seuil, classe=None):
    tp, fp, ngt = compte(enregs, seuil, classe)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / ngt if ngt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, ngt


def ap50_local(enregs, classe=None):
    """AP toutes-points, enveloppe par boucle arrière (implémentation jumelle)."""
    dets = []
    for e in enregs:
        dets += [(m[0], 1) for m in e["matches"] if classe is None or m[2] == classe]
        dets += [(f[0], 0) for f in e["fps"] if classe is None or f[1] == classe]
    _, _, ngt = compte(enregs, 0.0, classe)
    if not dets or not ngt:
        return 0.0
    dets.sort(key=lambda t: -t[0])
    rr, pp, tp, fp = [], [], 0, 0
    for conf, est_tp in dets:
        tp += est_tp
        fp += 1 - est_tp
        rr.append(tp / ngt)
        pp.append(tp / (tp + fp))
    for j in range(len(pp) - 2, -1, -1):  # enveloppe monotone décroissante
        if pp[j] < pp[j + 1]:
            pp[j] = pp[j + 1]
    ap, r_prec = 0.0, 0.0
    for r, p in zip(rr, pp):
        ap += (r - r_prec) * p
        r_prec = r
    return ap


def bloc_local(enregs, plancher, classe=None):
    seuils = grille_locale(plancher)
    meilleur_f, meilleur_i = -1.0, 0
    for i, s in enumerate(seuils):
        f = prf_local(enregs, s, classe)[2]
        if f > meilleur_f:  # strictement > : le PREMIER max gagne (comme argmax)
            meilleur_f, meilleur_i = f, i
    s0 = seuils[meilleur_i]
    p0, r0, f0, ngt = prf_local(enregs, s0, classe)
    ious = [m[1] for e in enregs for m in e["matches"]
            if m[0] >= s0 and (classe is None or m[2] == classe)]
    return {"seuil_f1max": s0, "F1": round(f0, 4), "P": round(p0, 4),
            "R": round(r0, 4), "AP50": round(ap50_local(enregs, classe), 4),
            "n_gt": ngt,
            "iou_median": round(median(ious), 4) if ious else None}


def comparer(nom, attendu, recalcule):
    for cle, val in recalcule.items():
        pub = attendu.get(cle)
        if pub != val:
            probleme(f"{nom}.{cle} : publié {pub!r} != recalculé {val!r}")
    for cle in attendu:
        if cle not in recalcule:
            probleme(f"{nom}.{cle} : clé publiée non recalculable")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dossier", help="dossier d'éval (metriques_eval.json + appariements.json)")
    a = ap.parse_args()
    dossier = Path(a.dossier)

    metriques = json.loads((dossier / "metriques_eval.json").read_text(encoding="utf-8"))
    cache = json.loads((dossier / "appariements.json").read_text(encoding="utf-8"))
    meta = cache.pop("_meta", None)

    if not str(metriques.get("schema", "")).startswith("metriques_eval/"):
        probleme(f"schema metriques inattendu : {metriques.get('schema')!r}")
    plancher = float(metriques.get("plancher", 0.05))
    grille = grille_locale(plancher)

    # cohérence empreinte <-> publication
    if meta is None:
        print("note : cache sans _meta (adoption legacy) — cohérence d'empreinte sautée,")
        print("       provenance publiée :", metriques.get("provenance_cache"))
        if metriques.get("provenance_cache") != "adoptee_sans_empreinte":
            probleme("cache sans _meta mais provenance_cache != 'adoptee_sans_empreinte'")
    else:
        if meta.get("plancher") != plancher:
            probleme(f"_meta.plancher {meta.get('plancher')} != metriques.plancher {plancher}")
        if meta.get("tache") != metriques.get("tache"):
            probleme(f"_meta.tache {meta.get('tache')} != metriques.tache {metriques.get('tache')}")
        if (meta.get("fusion") or {}) != (metriques.get("fusion") or {}):
            probleme("_meta.fusion != metriques.fusion")
        if sorted(meta.get("modeles", {})) != sorted(metriques.get("modeles", {})):
            probleme("modèles de _meta != modèles publiés")

    if sorted(cache) != sorted(metriques.get("modeles", {})):
        probleme(f"modèles du cache {sorted(cache)} != publiés {sorted(metriques.get('modeles', {}))}")

    n_zone_classe = 0
    for nom, dd in cache.items():
        enregs = dd["enregs"]
        publie = metriques["modeles"].get(nom)
        if publie is None:
            continue
        # sanité du cache
        for e in enregs:
            for m in e["matches"]:
                if m[1] < 0.5:
                    probleme(f"{nom} : match à IoU {m[1]} < 0,5 dans le cache")
                    break
        if publie.get("class_offset") != dd.get("decal"):
            probleme(f"{nom}.class_offset publié {publie.get('class_offset')} != cache {dd.get('decal')}")

        # global
        g = bloc_local(enregs, plancher)
        if g["seuil_f1max"] not in grille:
            probleme(f"{nom} : seuil_f1max recalculé {g['seuil_f1max']} hors grille")
        comparer(f"{nom}.global", publie.get("global", {}), g)

        # par_classe (l'univers des classes = celui de la publication)
        for cl, bloc_pub in (publie.get("par_classe") or {}).items():
            comparer(f"{nom}.par_classe.{cl}", bloc_pub, bloc_local(enregs, plancher, cl))

        # sommes n_gt : global == somme des classes publiées
        somme = sum(b.get("n_gt", 0) for b in (publie.get("par_classe") or {}).values())
        if somme != g["n_gt"]:
            probleme(f"{nom} : somme n_gt par_classe {somme} != global {g['n_gt']}")

        # par_zone au seuil F1-max global
        s0 = g["seuil_f1max"]
        zones_pub = publie.get("par_zone") or {}
        zones_reelles = sorted({e.get("zone", "") for e in enregs if e.get("zone")})
        if zones_reelles and sorted(zones_pub) != zones_reelles:
            probleme(f"{nom} : zones publiées {sorted(zones_pub)} != cache {zones_reelles}")
        for z, bloc_pub in zones_pub.items():
            sous = [e for e in enregs if e.get("zone") == z]
            tp, fp, ngt = compte(sous, s0)
            attendu = {"P": round(tp / (tp + fp), 4) if tp + fp else 1.0,
                       "R": round(tp / ngt, 4) if ngt else 0.0, "n_gt": ngt}
            comparer(f"{nom}.par_zone.{z}", bloc_pub, attendu)

        # par_zone_classe (2026-09-03) : même seuil s0, seuil de classe publié, R_max
        zc_pub = publie.get("par_zone_classe")
        if zones_pub and zc_pub is None:
            AVERTISSEMENTS.append(f"{nom} : par_zone_classe absent (éval antérieure au "
                                  "2026-09-03 : lancer completer_metriques_eval.py)")
        elif zc_pub is not None:
            n_zone_classe += 1
            classes_pub = sorted(publie.get("par_classe") or {})
            if sorted(zc_pub) != sorted(zones_pub):
                probleme(f"{nom} : zones de par_zone_classe {sorted(zc_pub)} != par_zone")
            for z, par_cl in zc_pub.items():
                sous = [e for e in enregs if e.get("zone") == z]
                if sorted(par_cl) != classes_pub:
                    probleme(f"{nom}.par_zone_classe.{z} : classes {sorted(par_cl)} "
                             f"!= par_classe {classes_pub}")
                for cl, bloc_pub in par_cl.items():
                    s_cl = (publie.get("par_classe") or {}).get(cl, {}).get("seuil_f1max")
                    if s_cl is None:
                        continue  # déjà signalé (classe hors par_classe)
                    comparer(f"{nom}.par_zone_classe.{z}.{cl}", bloc_pub,
                             zone_classe_local(sous, cl, s0, s_cl))
                somme = sum(b.get("n_gt", 0) for b in par_cl.values())
                if somme != (zones_pub.get(z) or {}).get("n_gt"):
                    probleme(f"{nom}.par_zone_classe.{z} : somme n_gt {somme} "
                             f"!= par_zone {(zones_pub.get(z) or {}).get('n_gt')}")

    for av in AVERTISSEMENTS:
        print("AVERTISSEMENT —", av)
    if RAPPORT:
        print(f"NON CONFORME — {len(RAPPORT)} divergence(s) :")
        for r in RAPPORT:
            print("  -", r)
        sys.exit(1)
    n_modeles = len(cache)
    print(f"CONFORME — {n_modeles} modèle(s), grille {len(grille)} seuils, "
          "global + par_classe + par_zone recalculés à l'identique"
          + (f" + par_zone_classe ({n_zone_classe} modèle(s))" if n_zone_classe else ""))


if __name__ == "__main__":
    main()
