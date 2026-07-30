"""Les noms de classes doivent venir DU MODÈLE, pas du COCO de test.

Le comparatif fait tourner deux modèles de taxonomies différentes sur le même corpus :
l'ancien a 3 classes (chemin_creux, parcellaire, talus_fosse) avec class_offset 1, le
nouveau en a 5 dans un autre ordre. `_geo_postprocess` nommait les couches d'après
`Corpus.noms_classes`, c'est-à-dire la taxonomie du corpus v2 — l'ancien modèle aurait
donc produit des géométries correctes sous des étiquettes fausses, et toute ventilation
par classe aurait été silencieusement inversée.

Ce test vérifie les deux garde-fous :
  1. `noms_classes_modele` lit bien la table du modèle chargé ;
  2. le mappage canonique couvre chaque classe de chaque modèle, sans trou ni collision.

    python tests\\test_noms_classes_modele.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PLUGIN = Path(os.environ.get(
    "ARCHEO_PLUGIN",
    r"C:\Users\valen\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\archeologia-pipeline"))
sys.path.insert(0, str(PLUGIN / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.bench.__main__ import noms_classes_modele          # noqa: E402
from tools.bench.mosaic import (                              # noqa: E402
    CANONIQUES, CANONIQUE_PAR_MODELE, COUCHES, COUCHES_CANONIQUES, canonique_pour,
)

ATTENDU = {
    "lineaires_seg_v2_1":
        ["parcellaire", "talus", "fosse", "talus_fosse", "chemin_creux"],
    "formes_lineaires_ld_a15_rmin10_rm_rfdetr_seg_1":
        ["chemin_creux", "parcellaire", "talus_fosse"],
}


def main() -> int:
    echecs: list[str] = []

    for dossier, noms in ATTENDU.items():
        onnx = PLUGIN / "data" / "models" / dossier / "weights" / "best.onnx"
        if not onnx.exists():
            echecs.append(f"{dossier} : best.onnx absent ({onnx})")
            continue

        lus = noms_classes_modele(str(onnx))
        if lus != noms:
            echecs.append(f"{dossier} : noms lus {lus} != attendus {noms}")
        else:
            print(f"OK   {dossier:<48} {lus}")

        # Le piège que ce test existe pour attraper : nommer avec l'AUTRE taxonomie.
        autre = next(v for k, v in ATTENDU.items() if k != dossier)
        if lus == autre:
            echecs.append(f"{dossier} : noms de l'autre modele !")

        canon = canonique_pour(str(onnx))
        manquants = set(range(len(lus))) - set(canon)
        if manquants:
            echecs.append(f"{dossier} : classes sans mappage canonique {sorted(manquants)}")
        hors = {c for c in canon.values() if c not in CANONIQUES}
        if hors:
            echecs.append(f"{dossier} : classes canoniques inconnues {hors}")
        surplus = set(canon) - set(range(len(lus)))
        if surplus:
            echecs.append(f"{dossier} : mappage d'ids inexistants {sorted(surplus)}")
        print(f"     canonique : { {lus[i]: c for i, c in sorted(canon.items())} }")

    # Chaque classe canonique doit être atteignable par les DEUX modèles, sinon la
    # comparaison par classe serait vide d'un côté.
    for cl in CANONIQUES:
        for dossier, canon in CANONIQUE_PAR_MODELE.items():
            if cl not in set(canon.values()):
                echecs.append(f"{dossier} ne peut produire aucune classe canonique {cl!r}")

    # Et chaque couche GT utilisée doit avoir une cible canonique, sinon sa longueur
    # disparaîtrait de la ventilation par classe sans avertissement.
    for zone, couches in COUCHES.items():
        for couche in couches:
            if couche not in COUCHES_CANONIQUES:
                echecs.append(f"{zone} : couche GT {couche!r} sans classe canonique")
    print(f"\nOK   {len(COUCHES_CANONIQUES)} couches GT mappees, "
          f"{len(CANONIQUES)} classes canoniques")

    if echecs:
        print("\nECHEC :")
        for e in echecs:
            print("  -", e)
        return 1
    print("\nTOUT PASSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
