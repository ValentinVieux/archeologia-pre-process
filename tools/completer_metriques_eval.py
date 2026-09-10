"""Complète un metriques_eval.json antérieur avec les blocs additifs manquants.

- `par_zone_classe` (2026-09-03) : pour chaque modèle publié ayant des zones, détail
  ZONE × CLASSE (n_gt, tp, fp, R, P au seuil F1-max global ; R_seuil_classe et
  fp_seuil_classe au seuil F1-max de la classe ; R_max = rappel avec TOUS les
  matches du cache) ;
- `etude_seuil` (2026-09-09) dans `global` et chaque `par_classe[c]` : F2-max,
  plateaux F1 ≥ 98 %/95 %, R_max, FP par image, seuil proposé, précision marginale,
  tableau au pas 0,05 — les indicateurs du CHOIX du seuil de production.

Recalcul depuis appariements.json par les fonctions PARTAGÉES de courbes_eval
(par_zone_classe, etude_seuil) — donc sans GPU ni réinférence, et à l'identique
de ce que courbes_eval écrit désormais nativement. Tout le reste du fichier est
conservé tel quel (contrôle par relecture : toutes les autres clés égales à
l'original) ; un champ racine `complete_le` {"par_zone_classe"|"etude_seuil":
<date ISO>, "outil": ...} trace l'opération. Refuse si rien ne manque (sauf
--forcer : recalcul de tout).

Usage (.venv, SANS GPU) :
  .venv\\Scripts\\python.exe tools\\completer_metriques_eval.py <dossier_eval> [--out <dossier>] [--forcer]
<dossier_eval> contient metriques_eval.json + appariements.json (sortie de
courbes_eval). --out : dossier où écrire le metriques_eval.json complété (défaut :
<dossier_eval>, réécrit en place — jamais directement sur G: : copie locale puis
re-dépôt, cf. CLAUDE.md). Enchaîner tools/verif_courbes_eval.py sur le résultat.
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from courbes_eval import etude_seuil, par_zone_classe  # noqa: E402


def _blocs(m):
    """[(classe|None, bloc)] : global puis chaque classe."""
    return [(None, m["global"])] + list(m["par_classe"].items())


def completer(metriques, cache, forcer=False):
    """Ajoute les blocs additifs manquants (par_zone_classe, etude_seuil) + complete_le
    au dict metriques (en place). Retourne le nombre de modèles complétés.
    `cache` = appariements.json (sans _meta)."""
    modeles = metriques["modeles"]
    manque_zc = any("par_zone" in m and ("par_zone_classe" not in m
                                         or any("bandes" not in b for z in m["par_zone_classe"].values()
                                                for b in z.values()))
                    for m in modeles.values())
    # etude_seuil absent, OU présent sans la table de calibrage (bandes, 2026-09-09 soir)
    manque_et = any("bandes" not in (b.get("etude_seuil") or {})
                    for m in modeles.values() for _, b in _blocs(m))
    if not forcer and not (manque_zc or manque_et):
        sys.exit("ERREUR : par_zone_classe et etude_seuil déjà présents — rien à compléter "
                 "(--forcer pour recalculer).")
    plancher = float(metriques.get("plancher", 0.05))
    n, fait = 0, set()
    for nom, m in modeles.items():
        if nom not in cache:
            sys.exit(f"ERREUR : modèle publié {nom} absent de appariements.json.")
        enregs = cache[nom]["enregs"]
        touche = False
        # pas de zones dans le COCO : pas de bloc par_zone_classe (comme courbes_eval)
        zc_incomplet = "par_zone_classe" in m and any(
            "bandes" not in b for z in m["par_zone_classe"].values() for b in z.values())
        if "par_zone" in m and (forcer or "par_zone_classe" not in m or zc_incomplet):
            m["par_zone_classe"] = par_zone_classe(
                enregs, m["global"]["seuil_f1max"],
                {c: b["seuil_f1max"] for c, b in m["par_classe"].items()}, list(m["par_classe"]))
            fait.add("par_zone_classe"); touche = True
        for cl, b in _blocs(m):
            if forcer or "bandes" not in (b.get("etude_seuil") or {}):
                b["etude_seuil"] = etude_seuil(enregs, plancher, cl)
                fait.add("etude_seuil"); touche = True
        n += touche
    horodatage = datetime.now().isoformat(timespec="seconds")
    trace = dict(metriques.get("complete_le") or {})
    trace.update({k: horodatage for k in fait})
    trace["outil"] = "tools/completer_metriques_eval.py"
    metriques["complete_le"] = trace
    return n


def sans_bloc(metriques):
    """Copie du dict sans les blocs additifs ni complete_le (pour la relecture)."""
    d = json.loads(json.dumps(metriques))
    d.pop("complete_le", None)
    for m in d["modeles"].values():
        m.pop("par_zone_classe", None)
        for _, b in _blocs(m):
            b.pop("etude_seuil", None)
    return d


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")  # les refus (sys.exit) portent des accents
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dossier", help="dossier d'éval (metriques_eval.json + appariements.json)")
    ap.add_argument("--out", default=None, help="dossier de sortie (défaut : en place)")
    ap.add_argument("--forcer", action="store_true", help="recalculer un bloc déjà présent")
    a = ap.parse_args()

    src = os.path.join(a.dossier, "metriques_eval.json")
    original = json.load(open(src, encoding="utf-8"))
    metriques = json.loads(json.dumps(original))
    cache = json.load(open(os.path.join(a.dossier, "appariements.json"), encoding="utf-8"))
    cache.pop("_meta", None)
    n = completer(metriques, cache, a.forcer)

    out = a.out or a.dossier
    os.makedirs(out, exist_ok=True)
    chemin = os.path.join(out, "metriques_eval.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(metriques, f, ensure_ascii=False, indent=1)
    relu = json.load(open(chemin, encoding="utf-8"))
    assert sans_bloc(relu) == sans_bloc(original), "relecture : contenu hors bloc modifié"
    assert all(("par_zone_classe" in m) == ("par_zone" in m) for m in relu["modeles"].values())
    assert all("etude_seuil" in b for m in relu["modeles"].values() for _, b in _blocs(m))
    print(f"{n} modèle(s) complété(s) -> {chemin}")
    for nom, m in relu["modeles"].items():
        for z, par_cl in (m.get("par_zone_classe") or {}).items():
            print(f"  {nom} / {z} : " + ", ".join(
                f"{c} n_gt {b['n_gt']} R {b['R']} R_max {b['R_max']}" for c, b in par_cl.items()))
        for cl, b in _blocs(m):
            et = b["etude_seuil"]
            print(f"  {nom} / {cl or 'global'} : F1-max @ {b['seuil_f1max']}, F2-max @ "
                  f"{et['seuil_f2max']}, plateau 95 % {et['plateau_f1_95']}, proposé "
                  f"{et['seuil_propose']} (P {et['P_propose']} / R {et['R_propose']}, FP/img "
                  f"{et['fp_par_image']['propose']}, précision marginale {et['precision_marginale']})")


if __name__ == "__main__":
    main()
