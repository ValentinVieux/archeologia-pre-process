"""Contrôle de clôture d'un dossier de brouillon de chantier.

Un chantier d'entraînement produit un dossier de travail local (`D:\\brouillons\\<chantier>`)
qui n'a AUCUNE vocation à survivre : tout ce qui compte est déposé sur le Drive
(data_regions_v2, model-training), commité dans le repo, ou installé dans le plugin.

Cet outil répond à la seule question qui autorise la suppression : *quel fichier du
brouillon n'a de jumeau nulle part ailleurs ?* Il indexe les destinations pérennes par
(nom de fichier, taille en octets) puis agrège le verdict par dossier du brouillon.

Il ne supprime RIEN (décision humaine, cf. CLAUDE.md § Brouillons de chantier) : il
imprime un état, écrit `cloture.json` dans le brouillon, et sort 0 si tout est déposé,
2 sinon.

    python tools\\cloturer_chantier.py D:\\brouillons\\ponctuelles_2cl_v3
    python tools\\cloturer_chantier.py <brouillon> --cible <racine> [--cible ...] [--json <fichier>]

Angles morts assumés : un fichier renommé au dépôt (`annotations_v3_dedup.gpkg` ->
`<zone>_entites_l93_v3.gpkg`) sort SANS JUMEAU alors qu'il est déposé — c'est voulu,
au producteur de le justifier ligne à ligne. Inversement, un jumeau de même nom et
même taille mais de contenu différent passerait : les gros livrables sont vérifiés par
leurs propres contrôleurs (verif_dataset, verif_corpus, verif_courbes_eval) en amont.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

# Destinations pérennes standard (cf. CLAUDE.md § Stockage Drive, § Rangement model-training).
CIBLES_DEFAUT = [
    r"G:\Mon Drive\Archeologia\Archeologia_Shared\model-training",
    r"G:\Mon Drive\Archeologia\Archeologia_Shared\data\data_regions_v2",
    r"G:\Mon Drive\Colab Notebooks",
    os.path.expandvars(
        r"%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\archeologia-pipeline\data\models"
    ),
    # Le repo : configs/manifests commités + les datasets/corpus intermédiaires encore locaux.
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "manifests"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus"),
]

GO = 2 ** 30


def indexer(racines: list[str]) -> dict[tuple[str, int], list[str]]:
    """(nom, taille) -> chemins, sur toutes les destinations pérennes lisibles."""
    index: dict[tuple[str, int], list[str]] = defaultdict(list)
    for racine in racines:
        if not os.path.isdir(racine):
            print(f"  cible ABSENTE (ignorée) : {racine}", file=sys.stderr)
            continue
        n0 = len(index)
        for dossier, _, fichiers in os.walk(racine):
            for nom in fichiers:
                chemin = os.path.join(dossier, nom)
                try:
                    index[(nom, os.path.getsize(chemin))].append(chemin)
                except OSError:
                    pass
        print(f"  indexée : {racine}  (+{len(index) - n0} entrées)", file=sys.stderr)
    return index


def auditer(brouillon: str, index: dict) -> dict:
    """Verdict par dossier du brouillon : combien de fichiers sans jumeau, et lesquels."""
    par_dossier: dict[str, dict] = {}
    for dossier, _, fichiers in os.walk(brouillon):
        rel = os.path.relpath(dossier, brouillon)
        if not fichiers:
            continue
        etat = par_dossier.setdefault(
            rel, {"n": 0, "octets": 0, "n_sans_jumeau": 0, "octets_sans_jumeau": 0, "exemples": []}
        )
        for nom in fichiers:
            chemin = os.path.join(dossier, nom)
            try:
                taille = os.path.getsize(chemin)
            except OSError:
                continue
            etat["n"] += 1
            etat["octets"] += taille
            if not index.get((nom, taille)):
                etat["n_sans_jumeau"] += 1
                etat["octets_sans_jumeau"] += taille
                if len(etat["exemples"]) < 12:
                    etat["exemples"].append(nom)
    return dict(sorted(par_dossier.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("brouillon", help="dossier de brouillon à clôturer")
    ap.add_argument("--cible", action="append", default=[], metavar="RACINE",
                    help="destination pérenne supplémentaire (répétable)")
    ap.add_argument("--sans-defaut", action="store_true",
                    help="n'indexer QUE les --cible (sans les destinations standard)")
    ap.add_argument("--json", metavar="FICHIER", help="rapport JSON (défaut : <brouillon>\\cloture.json)")
    args = ap.parse_args()

    brouillon = os.path.abspath(args.brouillon)
    if not os.path.isdir(brouillon):
        print(f"ERREUR : {brouillon} n'est pas un dossier", file=sys.stderr)
        return 1

    print("Indexation des destinations pérennes...", file=sys.stderr)
    cibles = (([] if args.sans_defaut else CIBLES_DEFAUT) + args.cible)
    index = indexer(cibles)
    if not index:
        print("ERREUR : aucune destination pérenne lisible (Drive démonté ? cf. Get-PSDrive G)",
              file=sys.stderr)
        return 1

    par_dossier = auditer(brouillon, index)
    total = sum(d["n"] for d in par_dossier.values())
    total_o = sum(d["octets"] for d in par_dossier.values())
    reste = sum(d["n_sans_jumeau"] for d in par_dossier.values())
    reste_o = sum(d["octets_sans_jumeau"] for d in par_dossier.values())

    print(f"\n=== {brouillon} — {total} fichiers, {total_o / GO:.2f} Go ===")
    for rel, d in par_dossier.items():
        etat = "deposé" if d["n_sans_jumeau"] == 0 else f"{d['n_sans_jumeau']}/{d['n']} SANS JUMEAU"
        print(f"  {rel:<58} {d['n']:>6} f {d['octets'] / GO:>7.2f} Go  {etat}"
              + (f"  [{d['octets_sans_jumeau'] / GO:.2f} Go]" if d["n_sans_jumeau"] else ""))
        if d["exemples"]:
            print(f"       ex. : {', '.join(d['exemples'][:6])}")

    rapport = args.json or os.path.join(brouillon, "cloture.json")
    with open(rapport, "w", encoding="utf-8") as f:
        json.dump({"brouillon": brouillon, "cibles": cibles,
                   "total_fichiers": total, "total_octets": total_o,
                   "sans_jumeau_fichiers": reste, "sans_jumeau_octets": reste_o,
                   "par_dossier": par_dossier}, f, ensure_ascii=False, indent=1)

    if reste:
        print(f"\nÀ DÉPOSER (ou à justifier comme régénérable) : {reste} fichiers, {reste_o / GO:.2f} Go")
        print(f"Rapport : {rapport}")
        return 2
    print(f"\nCLÔTURABLE : tout a un jumeau pérenne.\nRapport : {rapport}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
