"""Complète un metriques_eval.json antérieur au 2026-09-03 avec le bloc par_zone_classe.

Pour chaque modèle publié ayant des zones, recalcule le détail ZONE × CLASSE
(n_gt, tp, fp, R, P au seuil F1-max global ; R_seuil_classe et fp_seuil_classe au
seuil F1-max de la classe ; R_max = rappel avec TOUS les matches du cache) depuis
appariements.json, par la fonction PARTAGÉE courbes_eval.par_zone_classe — donc
sans GPU ni réinférence, et à l'identique de ce que courbes_eval écrit désormais
nativement. Tout le reste du fichier est conservé tel quel (contrôle par
relecture : toutes les autres clés égales à l'original) ; un champ racine
`complete_le` {"par_zone_classe": <date ISO>, "outil": ...} trace l'opération.
Refuse si le bloc existe déjà (sauf --forcer : recalcul).

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
from courbes_eval import par_zone_classe  # noqa: E402


def completer(metriques, cache, forcer=False):
    """Ajoute par_zone_classe + complete_le au dict metriques (en place). Retourne
    le nombre de modèles complétés. `cache` = appariements.json (sans _meta)."""
    modeles = metriques["modeles"]
    if not forcer and any("par_zone_classe" in m for m in modeles.values()):
        sys.exit("ERREUR : par_zone_classe déjà présent — rien à compléter "
                 "(--forcer pour recalculer).")
    n = 0
    for nom, m in modeles.items():
        if nom not in cache:
            sys.exit(f"ERREUR : modèle publié {nom} absent de appariements.json.")
        if "par_zone" not in m:
            continue  # pas de zones dans le COCO : pas de bloc (comme courbes_eval)
        m["par_zone_classe"] = par_zone_classe(
            cache[nom]["enregs"], m["global"]["seuil_f1max"],
            {c: b["seuil_f1max"] for c, b in m["par_classe"].items()}, list(m["par_classe"]))
        n += 1
    metriques["complete_le"] = {"par_zone_classe": datetime.now().isoformat(timespec="seconds"),
                                "outil": "tools/completer_metriques_eval.py"}
    return n


def sans_bloc(metriques):
    """Copie du dict sans par_zone_classe ni complete_le (pour la relecture)."""
    d = json.loads(json.dumps(metriques))
    d.pop("complete_le", None)
    for m in d["modeles"].values():
        m.pop("par_zone_classe", None)
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
    print(f"{n} modèle(s) complété(s) -> {chemin}")
    for nom, m in relu["modeles"].items():
        for z, par_cl in (m.get("par_zone_classe") or {}).items():
            print(f"  {nom} / {z} : " + ", ".join(
                f"{c} n_gt {b['n_gt']} R {b['R']} R_max {b['R_max']}" for c, b in par_cl.items()))


if __name__ == "__main__":
    main()
