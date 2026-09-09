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
    python tools\\cloturer_chantier.py <brouillon> [<brouillon> ...] --cible <racine> [--json <fichier>]

Plusieurs brouillons se passent d'un coup : l'indexation des destinations pérennes est
la partie coûteuse (~45 min sur un Drive qui synchronise) et n'est faite qu'une fois.

Tout ce qui reste SANS JUMEAU doit être déposé OU justifié : un fichier
`cloture-ignore.txt` à la racine du brouillon liste, un motif glob par ligne (style
.gitignore, `#` = commentaire), ce qui n'a délibérément pas de copie pérenne —
régénérable, journal, intermédiaire périmé. Chaque motif DOIT porter sa justification
en commentaire de fin de ligne : c'est elle qui rend le verdict CLÔTURABLE lisible.

Angles morts assumés : un fichier renommé au dépôt (`annotations_v3_dedup.gpkg` ->
`<zone>_entites_l93_v3.gpkg`) sort SANS JUMEAU alors qu'il est déposé — c'est voulu,
au producteur de le justifier ligne à ligne. Inversement, un jumeau de même nom et
même taille mais de contenu différent passerait : les gros livrables sont vérifiés par
leurs propres contrôleurs (verif_dataset, verif_corpus, verif_courbes_eval) en amont.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from collections import defaultdict

IGNORE = "cloture-ignore.txt"

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


def motifs_ignores(brouillon: str) -> list[str]:
    """Motifs glob de <brouillon>/cloture-ignore.txt (chemins relatifs, slashs avant)."""
    chemin = os.path.join(brouillon, IGNORE)
    if not os.path.exists(chemin):
        return []
    motifs = []
    for ligne in open(chemin, encoding="utf-8"):
        motif = ligne.split("#")[0].strip().replace("\\", "/")
        if motif:
            motifs.append(motif)
    return motifs


def auditer(brouillon: str, index: dict, motifs: list[str]) -> dict:
    """Verdict par dossier : fichiers sans jumeau, dont ceux couverts par cloture-ignore."""
    par_dossier: dict[str, dict] = {}
    for dossier, _, fichiers in os.walk(brouillon):
        rel = os.path.relpath(dossier, brouillon)
        if not fichiers:
            continue
        etat = par_dossier.setdefault(
            rel, {"n": 0, "octets": 0, "n_sans_jumeau": 0, "octets_sans_jumeau": 0,
                  "n_justifies": 0, "octets_justifies": 0, "exemples": []}
        )
        for nom in fichiers:
            chemin = os.path.join(dossier, nom)
            try:
                taille = os.path.getsize(chemin)
            except OSError:
                continue
            etat["n"] += 1
            etat["octets"] += taille
            if index.get((nom, taille)):
                continue
            relatif = os.path.relpath(chemin, brouillon).replace("\\", "/")
            if nom == IGNORE or any(fnmatch.fnmatch(relatif, m) for m in motifs):
                etat["n_justifies"] += 1
                etat["octets_justifies"] += taille
                continue
            etat["n_sans_jumeau"] += 1
            etat["octets_sans_jumeau"] += taille
            if len(etat["exemples"]) < 12:
                etat["exemples"].append(nom)
    return dict(sorted(par_dossier.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("brouillon", nargs="+", help="dossier(s) de brouillon à clôturer")
    ap.add_argument("--cible", action="append", default=[], metavar="RACINE",
                    help="destination pérenne supplémentaire (répétable)")
    ap.add_argument("--sans-defaut", action="store_true",
                    help="n'indexer QUE les --cible (sans les destinations standard)")
    ap.add_argument("--json", metavar="FICHIER",
                    help="rapport JSON d'un brouillon unique (défaut : <brouillon>\\cloture.json)")
    args = ap.parse_args()

    brouillons = [os.path.abspath(b) for b in args.brouillon]
    for b in brouillons:
        if not os.path.isdir(b):
            print(f"ERREUR : {b} n'est pas un dossier", file=sys.stderr)
            return 1
    if args.json and len(brouillons) > 1:
        print("ERREUR : --json ne vaut que pour un brouillon unique", file=sys.stderr)
        return 1

    print("Indexation des destinations pérennes...", file=sys.stderr)
    cibles = (([] if args.sans_defaut else CIBLES_DEFAUT) + args.cible)
    index = indexer(cibles)
    if not index:
        print("ERREUR : aucune destination pérenne lisible (Drive démonté ? cf. Get-PSDrive G)",
              file=sys.stderr)
        return 1

    return max(auditer_et_rapporter(b, index, cibles, args.json) for b in brouillons)


def auditer_et_rapporter(brouillon: str, index: dict, cibles: list[str], json_force: str | None) -> int:
    motifs = motifs_ignores(brouillon)
    print(f"  {IGNORE} : {len(motifs)} motifs" if motifs else f"  {IGNORE} : absent", file=sys.stderr)

    par_dossier = auditer(brouillon, index, motifs)
    total = sum(d["n"] for d in par_dossier.values())
    total_o = sum(d["octets"] for d in par_dossier.values())
    reste = sum(d["n_sans_jumeau"] for d in par_dossier.values())
    reste_o = sum(d["octets_sans_jumeau"] for d in par_dossier.values())
    just = sum(d["n_justifies"] for d in par_dossier.values())
    just_o = sum(d["octets_justifies"] for d in par_dossier.values())

    print(f"\n=== {brouillon} — {total} fichiers, {total_o / GO:.2f} Go ===")
    for rel, d in par_dossier.items():
        if d["n_sans_jumeau"]:
            etat = f"{d['n_sans_jumeau']}/{d['n']} SANS JUMEAU  [{d['octets_sans_jumeau'] / GO:.2f} Go]"
        elif d["n_justifies"]:
            etat = f"deposé ou justifié ({d['n_justifies']} justifiés)"
        else:
            etat = "deposé"
        print(f"  {rel:<58} {d['n']:>6} f {d['octets'] / GO:>7.2f} Go  {etat}")
        if d["exemples"]:
            print(f"       ex. : {', '.join(d['exemples'][:6])}")

    rapport = json_force or os.path.join(brouillon, "cloture.json")
    with open(rapport, "w", encoding="utf-8") as f:
        json.dump({"brouillon": brouillon, "cibles": cibles, "motifs_ignores": motifs,
                   "total_fichiers": total, "total_octets": total_o,
                   "sans_jumeau_fichiers": reste, "sans_jumeau_octets": reste_o,
                   "justifies_fichiers": just, "justifies_octets": just_o,
                   "par_dossier": par_dossier}, f, ensure_ascii=False, indent=1)

    print(f"\nJustifiés par {IGNORE} : {just} fichiers, {just_o / GO:.2f} Go")
    if reste:
        print(f"À DÉPOSER (ou à justifier dans {IGNORE}) : {reste} fichiers, {reste_o / GO:.2f} Go")
        print(f"Rapport : {rapport}")
        return 2
    print(f"CLÔTURABLE : tout est déposé ou justifié.\nRapport : {rapport}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
