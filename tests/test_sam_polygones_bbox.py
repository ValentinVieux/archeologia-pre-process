"""Auto-test de tools/sam_polygones_bbox.py (choix du masque, SANS GPU) :
.venv\\Scripts\\python.exe tests\\test_sam_polygones_bbox.py

Défaut reproduit (Blois 2026-09-08, 5 replis « centre_absent ») : dans les fonds plats, le
masque brut de SAM est moucheté (dizaines à centaines de trous) ; juger la plausibilité sur
UN pixel (le centre) rejetait des masques corrects. Le choix doit se faire sur le polygone
nettoyé (composante principale, trous comblés), comme la géométrie livrée.
"""
import sys
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sam_polygones_bbox import choisir_masque  # noqa: E402

N, RES = 256, 0.5
TFM = from_origin(0.0, N * RES, RES, RES)  # 256 px × 0,5 m ; origine NW (0, 128)
BOITE_PX = [78.0, 78.0, 178.0, 178.0]  # 100 px = 50 m, centre pixel (128, 128)
BOITE = box(39.0, 39.0, 89.0, 89.0)  # même boîte en m ; centroïde (64, 64)
YY, XX = np.mgrid[:N, :N]
R = np.hypot(YY - 128, XX - 128)


def cuvette_mouchetee():
    m = R <= 40  # disque de 40 px = 20 m de rayon
    m[(YY % 4 == 0) & (XX % 4 == 0)] = False  # mouchetage : 1 px sur 16
    m[127:130, 127:130] = False  # trou de 3 px au centre
    return m


def main():
    cuvette, anneau, miette = cuvette_mouchetee(), (R >= 35) & (R <= 40), R <= 3
    assert not cuvette[128, 128], "le centre doit tomber dans un trou (ancien critère)"

    # anneau mieux noté que la cuvette : il ne contient pas le centre, la cuvette doit gagner
    poly, score, motif = choisir_masque(np.stack([anneau, cuvette, miette]), np.array([0.9, 0.7, 0.8]),
                                        BOITE_PX, TFM, BOITE, 0.10, 1.00)
    assert motif == "ok" and score == 0.7, (motif, score)
    assert poly.contains(BOITE.centroid) and len(poly.interiors) == 0
    ratio = poly.area / BOITE.area  # disque de 20 m dans 50 × 50 m : ~0,50
    assert 0.42 < ratio < 0.58, ratio
    assert BOITE.buffer(0.01).contains(poly)

    # aucun candidat plausible : repli motivé
    poly, score, motif = choisir_masque(np.stack([anneau, miette]), np.array([0.9, 0.8]),
                                        BOITE_PX, TFM, BOITE, 0.10, 1.00)
    assert poly is None and score == 0.0, (poly, score)
    assert motif.startswith("masques_rejetes:") and "centre_absent" in motif and "aire_" in motif, motif
    print("OK test_sam_polygones_bbox")


if __name__ == "__main__":
    main()
