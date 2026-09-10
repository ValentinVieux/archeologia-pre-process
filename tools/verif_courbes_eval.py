"""Contrôleur indépendant de l'éval outillée (boucle de vérification, SANS GPU).

Usage : python verif_courbes_eval.py <dossier_eval>
(<dossier_eval> contient metriques_eval.json + appariements.json — la sortie de
tools/courbes_eval.py)

Recalcule depuis appariements.json, par une implémentation VOLONTAIREMENT
différente (python pur, recomptage brut par seuil, enveloppe de précision par
boucle arrière — zéro code partagé avec courbes_eval), les métriques publiées
dans metriques_eval.json : global, par_classe, par_zone, par_zone_classe (bloc
additif du 2026-09-03) et etude_seuil de global/par_classe (bloc additif du
2026-09-09 : F2-max, plateaux F1, FP/image, seuil proposé, tableau) — un bloc
additif absent = AVERTISSEMENT, pas de non-conformité (le compléter par
tools/completer_metriques_eval.py). Égalité exacte exigée après
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
    tp, fp, ngt, tpr = compte(sous, s0, classe)
    _, fp_cl, _, tpr_cl = compte(sous, s_classe, classe)
    tp_max = sum(1 for e in sous for m in e["matches"] if m[2] == classe)  # TOUS les matches
    return {"n_gt": ngt, "tp": tp, "fp": fp,
            "R": round(tpr / ngt, 4) if ngt else None,
            "P": round(tp / (tp + fp), 4) if tp + fp else None,
            "R_seuil_classe": round(tpr_cl / ngt, 4) if ngt else None,
            "fp_seuil_classe": fp_cl,
            "R_max": round(tp_max / ngt, 4) if ngt else None,
            "bandes": bandes_local(sous, classe)}


def grille_locale(plancher: float) -> list[float]:
    seuils, k = [], 0
    while True:
        s = round(plancher + 0.005 * k, 3)
        if s > 0.9505:
            break
        seuils.append(s)
        k += 1
    return seuils


def vrais(e):
    """Prédictions vraies (côté précision) : tps_pred au critère couverture, sinon matches."""
    return e["tps_pred"] if "tps_pred" in e else e["matches"]


def compte(enregs, seuil, classe=None):
    """(tp_precision, fp, n_gt, tp_rappel) par recomptage brut — pas de tri, pas de cumsum."""
    tp = fp = ngt = tpr = 0
    for e in enregs:
        for m in e["matches"]:
            if (classe is None or m[2] == classe) and m[0] >= seuil:
                tpr += 1
        for t in vrais(e):
            if (classe is None or t[2] == classe) and t[0] >= seuil:
                tp += 1
        for f in e["fps"]:
            if (classe is None or f[1] == classe) and f[0] >= seuil:
                fp += 1
        ngt += (sum(1 for c in e["gt_classes"] if c == classe) if classe
                else e["n_gt"])
    return tp, fp, ngt, tpr


def prf_local(enregs, seuil, classe=None):
    tp, fp, ngt, tpr = compte(enregs, seuil, classe)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tpr / ngt if ngt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, ngt


def ap50_local(enregs, classe=None):
    """AP toutes-points, enveloppe par boucle arrière (implémentation jumelle)."""
    dets = []
    couverture = any("tps_pred" in e for e in enregs)
    for e in enregs:
        dets += [(t[0], 1) for t in vrais(e) if classe is None or t[2] == classe]
        dets += [(f[0], 0) for f in e["fps"] if classe is None or f[1] == classe]
    _, _, ngt, _ = compte(enregs, 0.0, classe)
    if not dets or not ngt:
        return 0.0
    dets.sort(key=lambda t: -t[0])
    # couverture : rappel = annotations retrouvées à une confiance >= celle du rang
    # (liste triée + bisect, stdlib — pas de numpy ici)
    from bisect import bisect_left
    mconfs = sorted(m[0] for e in enregs for m in e["matches"] if classe is None or m[2] == classe)
    rr, pp, tp, fp = [], [], 0, 0
    for conf, est_tp in dets:
        tp += est_tp
        fp += 1 - est_tp
        rr.append(((len(mconfs) - bisect_left(mconfs, conf)) if couverture else tp) / ngt)
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


def etude_local(enregs, plancher, classe=None):
    """Bloc etude_seuil attendu (2026-09-09) — implémentation jumelle, boucles pures."""
    seuils = grille_locale(plancher)
    n_img = len(enregs)
    lignes = []  # (seuil, tp, fp, p, r, f1, f2)
    for s in seuils:
        tp, fp, ngt, tpr = compte(enregs, s, classe)
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tpr / ngt if ngt else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f2 = 5 * p * r / (4 * p + r) if p + r else 0.0
        lignes.append((s, tp, fp, p, r, f1, f2))
    i1 = i2 = 0
    for i, l in enumerate(lignes):
        if l[5] > lignes[i1][5]:
            i1 = i
        if l[6] > lignes[i2][6]:
            i2 = i
    f1max = lignes[i1][5]

    def plateau(frac):
        idx = [i for i, l in enumerate(lignes) if l[5] >= frac * f1max]
        return idx[0], idx[-1]

    lo98, hi98 = plateau(0.98)
    lo95, hi95 = plateau(0.95)
    ip = i2 if i2 > lo95 else lo95

    def fp_img(i):
        return round(lignes[i][2] / n_img, 4) if n_img else None

    tp_add = lignes[ip][1] - lignes[i1][1]
    fp_add = lignes[ip][2] - lignes[i1][2]
    tableau = [{"seuil": l[0], "P": round(l[3], 4), "R": round(l[4], 4), "F1": round(l[5], 4),
                "F2": round(l[6], 4), "fp_img": fp_img(i)}
               for i, l in enumerate(lignes) if abs(l[0] * 20 - round(l[0] * 20)) < 1e-6]
    bandes = bandes_local(enregs, classe)
    return {"seuil_f2max": lignes[i2][0], "F2": round(lignes[i2][6], 4),
            "P_f2": round(lignes[i2][3], 4), "R_f2": round(lignes[i2][4], 4),
            "plateau_f1_98": [lignes[lo98][0], lignes[hi98][0]],
            "plateau_f1_95": [lignes[lo95][0], lignes[hi95][0]],
            "R_max": round(lignes[0][4], 4), "n_images": n_img,
            "fp_par_image": {"f1max": fp_img(i1), "f2max": fp_img(i2), "propose": fp_img(ip)},
            "seuil_propose": lignes[ip][0],
            "P_propose": round(lignes[ip][3], 4), "R_propose": round(lignes[ip][4], 4),
            "precision_marginale": (round(tp_add / (tp_add + fp_add), 4)
                                    if tp_add + fp_add else None),
            "tableau": tableau,
            "bandes": bandes,
            "fiabilite_proposee": fiabilite_local(bandes, lignes[ip][0])}


def bandes_local(enregs, classe=None):
    """Tranches FINES de 0,01 sur [0,05 ; 1] (dernière inclusive) — recomptage brut."""
    out = []
    for k in range(5, 100):
        lo, hi = round(0.01 * k, 2), round(0.01 * (k + 1), 2)
        tp = fp = 0
        for e in enregs:
            for m in vrais(e):
                if (classe is None or m[2] == classe) and m[0] >= lo - 1e-9 \
                        and (k == 99 or m[0] < hi - 1e-9):
                    tp += 1
            for f in e["fps"]:
                if (classe is None or f[1] == classe) and f[0] >= lo - 1e-9 \
                        and (k == 99 or f[0] < hi - 1e-9):
                    fp += 1
        out.append({"lo": lo, "hi": hi, "tp": tp, "fp": fp})
    return out


NIVEAUX = (("possible", 0.35), ("probable", 0.60), ("quasi_certain", 0.85))


def fenetres_local(bandes, debut=None):
    """Fenêtres de 0,05 alignées sur la grille depuis les tranches fines, première
    fenêtre PARTIELLE dès `debut` (jumelle de courbes_eval.agreger_bandes, sans numpy)."""
    fen = {}
    for b in bandes:
        if debut is not None and b["lo"] < debut - 1e-9:
            continue
        lo = round(int(b["lo"] / 0.05 + 1e-9) * 0.05, 2)
        hi = round(lo + 0.05, 2)
        if debut is not None and lo < debut - 1e-9:
            lo = float(debut)
        f = fen.setdefault(lo, {"lo": lo, "hi": hi, "tp": 0, "fp": 0})
        f["tp"] += b["tp"]
        f["fp"] += b["fp"]
        if b["hi"] > f["hi"]:
            f["hi"] = b["hi"]
    return [fen[k] for k in sorted(fen)]


def fiabilite_local(bandes, seuil):
    """Catégories de fiabilité au seuil donné (implémentation jumelle de
    courbes_eval.coupures_fiabilite + fiabilite_par_classe) : coupures sur les
    fenêtres de 0,05, effectifs sur les tranches fines."""
    fines = bandes
    utiles = [b for b in fenetres_local(bandes, seuil) if b["lo"] >= seuil - 1e-9 and b["tp"] + b["fp"] > 0]
    coupures, dernier = [], seuil
    for cle, niveau in NIVEAUX:
        trouve = None
        for i, b in enumerate(utiles):
            if b["lo"] < dernier - 1e-9:
                continue
            suiv = utiles[i + 1] if i + 1 < len(utiles) else {"tp": 0, "fp": 0}
            tp, fp = b["tp"] + suiv["tp"], b["fp"] + suiv["fp"]
            if tp + fp and tp / (tp + fp) >= niveau:
                trouve = b["lo"]
                break
        if trouve is not None:
            coupures.append((cle, niveau, trouve))
            dernier = trouve
    bornes = [("douteux", 0.0, float(seuil))] + [(c, g, float(s)) for c, g, s in coupures]
    bornes = [b for i, b in enumerate(bornes)
              if (bornes[i + 1][2] if i + 1 < len(bornes) else 1.01) - b[2] >= 1e-9]
    while True:
        cats = []
        for i, (cle, garanti, debut) in enumerate(bornes):
            fin = bornes[i + 1][2] if i + 1 < len(bornes) else 1.01
            tp = fp = 0
            for b in fines:
                if debut - 1e-9 <= b["lo"] < fin - 1e-9:
                    tp += b["tp"]
                    fp += b["fp"]
            cats.append({"categorie": cle, "seuil": debut, "garanti": float(garanti),
                         "mesure": round(tp / (tp + fp), 3) if tp + fp >= 30 else None,
                         "n": tp + fp})
        # fusion vers le bas : effectif < 30 ou mesure sous le niveau garanti (la plus haute d'abord)
        a_fusionner = None
        for i in range(len(cats) - 1, 0, -1):
            c = cats[i]
            if c["n"] < 30 or (c["mesure"] is not None and c["mesure"] < c["garanti"]):
                a_fusionner = i
                break
        if a_fusionner is None:
            break
        del bornes[a_fusionner]
    if len(cats) > 1 and cats[0]["n"] == 0:  # catégorie basse vide : la suivante démarre au seuil
        cats[1]["seuil"] = cats[0]["seuil"]
        cats = cats[1:]
    return cats


def comparer(nom, attendu, recalcule):
    for cle, val in recalcule.items():
        pub = attendu.get(cle)
        if pub != val:
            probleme(f"{nom}.{cle} : publié {pub!r} != recalculé {val!r}")
    for cle in attendu:
        if cle not in recalcule:
            probleme(f"{nom}.{cle} : clé publiée non recalculable")


def comparer_bloc(nom, attendu, enregs, plancher, classe=None):
    """global / par_classe : métriques standard, puis etude_seuil (additif 2026-09-09 :
    absent = avertissement, présent = recalculé à l'identique)."""
    attendu = dict(attendu or {})
    etude_pub = attendu.pop("etude_seuil", None)
    comparer(nom, attendu, bloc_local(enregs, plancher, classe))
    if etude_pub is None:
        AVERTISSEMENTS.append(f"{nom} : etude_seuil absent (éval antérieure au 2026-09-09 : "
                              "lancer completer_metriques_eval.py)")
        return 0
    if "bandes" not in etude_pub:
        # bloc du 2026-09-09 matin, sans la table de calibrage (fiabilité A+D) :
        # additif lui aussi — avertissement, complétable sans GPU.
        AVERTISSEMENTS.append(f"{nom} : etude_seuil sans bandes/fiabilite_proposee "
                              "(lancer completer_metriques_eval.py)")
        attendu_et = etude_local(enregs, plancher, classe)
        attendu_et.pop("bandes"); attendu_et.pop("fiabilite_proposee")
        comparer(f"{nom}.etude_seuil", etude_pub, attendu_et)
        return 1
    comparer(f"{nom}.etude_seuil", etude_pub, etude_local(enregs, plancher, classe))
    return 1


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
        if meta.get("critere", "iou") != metriques.get("critere", "iou"):
            probleme(f"_meta.critere {meta.get('critere', 'iou')} != metriques.critere "
                     f"{metriques.get('critere', 'iou')}")
        if sorted(meta.get("modeles", {})) != sorted(metriques.get("modeles", {})):
            probleme("modèles de _meta != modèles publiés")

    if sorted(cache) != sorted(metriques.get("modeles", {})):
        probleme(f"modèles du cache {sorted(cache)} != publiés {sorted(metriques.get('modeles', {}))}")

    n_zone_classe = n_etude = 0
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
        n_etude += comparer_bloc(f"{nom}.global", publie.get("global", {}), enregs, plancher)

        # par_classe (l'univers des classes = celui de la publication)
        for cl, bloc_pub in (publie.get("par_classe") or {}).items():
            n_etude += comparer_bloc(f"{nom}.par_classe.{cl}", bloc_pub, enregs, plancher, cl)

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
            tp, fp, ngt, tpr = compte(sous, s0)
            attendu = {"P": round(tp / (tp + fp), 4) if tp + fp else 1.0,
                       "R": round(tpr / ngt, 4) if ngt else 0.0, "n_gt": ngt}
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
                    attendu_zc = zone_classe_local(sous, cl, s0, s_cl)
                    if "bandes" not in bloc_pub:  # bloc antérieur au 2026-09-09 soir : additif
                        attendu_zc.pop("bandes")
                        AVERTISSEMENTS.append(f"{nom}.par_zone_classe.{z}.{cl} : bandes absentes "
                                              "(lancer completer_metriques_eval.py)")
                    comparer(f"{nom}.par_zone_classe.{z}.{cl}", bloc_pub, attendu_zc)
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
          + (f" + par_zone_classe ({n_zone_classe} modèle(s))" if n_zone_classe else "")
          + (f" + etude_seuil ({n_etude} bloc(s))" if n_etude else ""))


if __name__ == "__main__":
    main()
