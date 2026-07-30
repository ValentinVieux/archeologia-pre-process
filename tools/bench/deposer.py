"""Prépare le dépôt Drive du banc d'inférence : staging local, puis robocopy.

Règles du projet respectées :
  - JAMAIS d'écriture directe sur G: — on constitue un dossier local complet, puis
    `robocopy /E /MT:16` (aucun /MIR : rien n'est supprimé côté Drive).
  - aucun fichier existant du Drive n'est déplacé ni supprimé.

Optimisation faite au passage : les fonds LD sont écrits par cv2 en 3 canaux alors que le
Local Dominance est en niveaux de gris — les trois canaux sont identiques. On les repasse
en 1 canal, ce qui divise leur poids par ~3 sans perdre un seul niveau de gris.

    python tools/bench/deposer.py --bench D:\\pipeline_results\\bench --staging D:\\bench\\depot
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MESURES_UTILES = (
    "comparatif_modeles.json",
    "superpositions_nouveau.json",
    "seuils_par_classe.json",
    "aires_par_classe.json",
    "e0_plafond_rappel.json",
    "niveau_b_e7_ancien_modele__formes_lineaires_ld_a15_rmin10_rm_rfdetr_seg_1.json",
    "niveau_b_e7_nouveau_modele__lineaires_seg_v2_1.json",
    "niveau_b_e_niveaub.json",
    "niveau_b_e5_fusion.json",
    "niveau_b_e6_aire_min.json",
)


def fond_en_niveaux_de_gris(src: Path, dst: Path) -> tuple[float, float]:
    """Recopie le fond LD en 1 canal. Retourne (Mo avant, Mo apres)."""
    import cv2
    import numpy as np
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    avant = src.stat().st_size / 1e6
    if img is None:
        shutil.copy2(src, dst)
        return avant, avant
    if img.ndim == 3:
        # Vérifie que les canaux sont bien identiques avant d'en jeter deux.
        if not (np.array_equal(img[..., 0], img[..., 1])
                and np.array_equal(img[..., 1], img[..., 2])):
            shutil.copy2(src, dst)
            return avant, avant
        img = img[..., 0]
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return avant, dst.stat().st_size / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--staging", required=True)
    a = ap.parse_args()
    bench, st = Path(a.bench), Path(a.staging)
    if st.exists():
        shutil.rmtree(st)
    (st / "mesures").mkdir(parents=True)
    (st / "controle_visuel").mkdir(parents=True)

    shutil.copy2(bench / "report.html", st / "rapport.html")
    print(f"rapport.html  {(st / 'rapport.html').stat().st_size/1e6:.1f} Mo")

    n = 0
    for nom in MESURES_UTILES:
        src = bench / nom
        if src.exists():
            shutil.copy2(src, st / "mesures" / nom)
            n += 1
    print(f"mesures/      {n} fichiers JSON")

    tot_avant = tot_apres = 0.0
    for d in sorted((bench / "visuel").iterdir()):
        if not d.is_dir():
            continue
        cible = st / "controle_visuel" / d.name
        (cible / "extraits").mkdir(parents=True)
        for f in ("comparatif.gpkg", "fond_LD.pgw", "fond_LD.prj"):
            if (d / f).exists():
                shutil.copy2(d / f, cible / f)
        if (d / "fond_LD.png").exists():
            av, ap_ = fond_en_niveaux_de_gris(d / "fond_LD.png", cible / "fond_LD.png")
            tot_avant += av
            tot_apres += ap_
        for j in sorted((d / "extraits").glob("*.jpg")):
            shutil.copy2(j, cible / "extraits" / j.name)
        print(f"  {d.name:<44} gpkg + fond + "
              f"{len(list((cible / 'extraits').glob('*.jpg')))} extraits")
    print(f"fonds LD : {tot_avant:.0f} Mo -> {tot_apres:.0f} Mo "
          f"(-{100*(1-tot_apres/max(tot_avant,1e-9)):.0f} %, 3 canaux identiques -> 1)")

    total = sum(f.stat().st_size for f in st.rglob("*") if f.is_file()) / 1e6
    print(f"\nstaging pret : {st}  ({total:.0f} Mo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
