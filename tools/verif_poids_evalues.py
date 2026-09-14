"""Les poids déployés sont-ils CEUX QU'ON A MESURÉS ?

Garde-fou contre le bug best_ema/best_total, qui a livré deux modèles sur des poids jamais
évalués (enclos ie/fr en 2026-08, crateres_seg_ld_v1 le 2026-09-11, tranchees_seg_ld_v1 le
2026-09-10 — découvert le 2026-09-14) : la section d'évaluation du notebook mesure
`checkpoint_best_ema.pth` tandis que le packaging copie `checkpoint_best_total.pth` dans
`package/weights/best.pth`. Rien ne comparait les deux.

Chaque `metriques_eval.json` enregistre, dans `modeles.<nom>.poids`, le checkpoint RÉELLEMENT
mesuré. On le retrouve sur disque et on compare son empreinte à celle de `weights/best.pth`.

Seuls les dossiers `evaluation*` sont contrôlés : un `comparaison_*` mesure d'autres poids par
construction (baseline, checkpoint écarté), c'est sa raison d'être.

    .venv\\Scripts\\python.exe tools\\verif_poids_evalues.py <data/models du plugin>

Verdict par modèle : CONFORME / SUSPECT (les poids livrés ne sont pas ceux mesurés) /
INDÉTERMINABLE (le checkpoint mesuré est introuvable sur ce poste — on ne conclut pas).
Sortie 2 si au moins un SUSPECT ; 0 sinon (les INDÉTERMINABLE sont signalés, pas bloquants,
sauf --strict).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Les évals tournent dans Colab : leurs chemins sont montés sous /content/drive/MyDrive.
PREFIXES_DRIVE = ("/content/drive/MyDrive/", "/content/drive/My Drive/")
RACINE_DRIVE = "G:/Mon Drive/"


def empreinte(chemin: Path, taille_bloc: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as flux:
        for bloc in iter(lambda: flux.read(taille_bloc), b""):
            h.update(bloc)
    return h.hexdigest()


def candidats(poids: str, run_drive: Path | None) -> list[Path]:
    """Chemins plausibles pour le checkpoint mesuré, du plus sûr au plus indirect."""
    chemins: list[Path] = []
    for prefixe in PREFIXES_DRIVE:
        if poids.startswith(prefixe):
            chemins.append(Path(RACINE_DRIVE + poids[len(prefixe):]))
    chemins.append(Path(poids))
    # repli : même nom de fichier dans le run Drive du modèle (les copies locales D:\... des
    # brouillons sont supprimées à la clôture, le run, lui, est pérenne)
    if run_drive is not None:
        chemins.append(run_drive / Path(poids).name)
    return chemins


def run_drive_du_modele(dossier: Path, model_training: Path | None) -> Path | None:
    """Le run Drive d'où vient le modèle (sert de repli quand le chemin mesuré a disparu).

    Deux sources : le champ `poids` d'une éval lancée dans Colab, ou — quand toutes les évals
    pointent sur un brouillon local purgé (cas ponctuelles_2cl_det_ld_v1, D:\\entrainement_*
    supprimé à la clôture) — le run homonyme sous la racine model-training passée en --drive.
    """
    for evaluation in sorted(dossier.glob("entrainement/evaluation*/metriques_eval.json")):
        for modele in json.loads(evaluation.read_text(encoding="utf-8")).get("modeles", {}).values():
            poids = modele.get("poids", "")
            for prefixe in PREFIXES_DRIVE:
                if poids.startswith(prefixe):
                    return Path(RACINE_DRIVE + poids[len(prefixe):]).parent
    if model_training is not None:
        # glob BORNÉ (profondeur fixe) : un rglob sur le Drive ne finit jamais
        for motif in ("*/*/runs/training", "*/runs/training"):
            for runs in model_training.glob(motif):
                candidat = runs / dossier.name
                if candidat.is_dir():
                    return candidat
    return None


def controler(dossier: Path, model_training: Path | None) -> tuple[str, list[str]]:
    best = dossier / "weights" / "best.pth"
    if not best.exists():
        return "INDÉTERMINABLE", ["weights/best.pth absent"]

    evaluations = sorted(dossier.glob("entrainement/evaluation*/metriques_eval.json"))
    if not evaluations:
        return "INDÉTERMINABLE", ["aucune évaluation outillée (entrainement/evaluation*/)"]

    reference = empreinte(best)
    run = run_drive_du_modele(dossier, model_training)
    verdicts, lignes = [], []
    for evaluation in evaluations:
        donnees = json.loads(evaluation.read_text(encoding="utf-8"))
        for nom, modele in donnees.get("modeles", {}).items():
            poids = modele.get("poids", "")
            mesure = next((c for c in candidats(poids, run) if c.exists()), None)
            ou = evaluation.parent.name
            if mesure is None:
                verdicts.append("INDÉTERMINABLE")
                lignes.append(f"{ou}: checkpoint mesuré introuvable — {poids}")
                continue
            if empreinte(mesure) == reference:
                verdicts.append("CONFORME")
                lignes.append(f"{ou}: {Path(poids).name} == weights/best.pth")
            else:
                verdicts.append("SUSPECT")
                lignes.append(f"{ou}: MESURÉ {Path(poids).name} ({empreinte(mesure)[:12]}) "
                              f"!= LIVRÉ best.pth ({reference[:12]}) — source {mesure}")
    if "SUSPECT" in verdicts:
        return "SUSPECT", lignes
    if "INDÉTERMINABLE" in verdicts and "CONFORME" not in verdicts:
        return "INDÉTERMINABLE", lignes
    return "CONFORME", lignes


def main() -> int:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("racine", type=Path,
                         help="dossier data/models du plugin, ou un dossier de modèle unique")
    parseur.add_argument("--strict", action="store_true",
                         help="sortir en erreur aussi sur INDÉTERMINABLE")
    parseur.add_argument("--drive", type=Path, default=None,
                         help="racine model-training du Drive, pour retrouver le checkpoint "
                              "mesuré quand le brouillon local qui le portait a été purgé")
    args = parseur.parse_args()

    racine: Path = args.racine
    if (racine / "weights").is_dir():
        modeles = [racine]
    else:
        modeles = sorted(p for p in racine.iterdir() if p.is_dir() and (p / "weights").is_dir())
    if not modeles:
        print(f"aucun modèle sous {racine}", file=sys.stderr)
        return 1

    compte = {"CONFORME": 0, "SUSPECT": 0, "INDÉTERMINABLE": 0}
    for modele in modeles:
        verdict, lignes = controler(modele, args.drive)
        compte[verdict] += 1
        marque = {"CONFORME": "  ok  ", "SUSPECT": " SUSP ", "INDÉTERMINABLE": " ???? "}[verdict]
        print(f"[{marque}] {modele.name}")
        for ligne in lignes:
            print(f"           {ligne}")

    print(f"\n{len(modeles)} modèle(s) : {compte['CONFORME']} CONFORME, "
          f"{compte['SUSPECT']} SUSPECT, {compte['INDÉTERMINABLE']} INDÉTERMINABLE")
    if compte["SUSPECT"]:
        print("\nUn SUSPECT signifie que les chiffres publiés (model_card, registre, dashboard) "
              "ne décrivent PAS les poids installés. Remesurer les poids livrés avec "
              "tools/courbes_eval.py avant de rien conclure sur la qualité du modèle.")
        return 2
    if compte["INDÉTERMINABLE"] and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
