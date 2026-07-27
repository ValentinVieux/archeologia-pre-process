"""Découpeur de tuiles d'entraînement à split spatial par blocs.

Spec : docs/superpowers/specs/2026-07-27-slice-zone-design.md (décisions D1-D4 du
2026-07-27). Remède au split aléatoire documenté dans docs/fuite_spatiale_train_test.html :
tuiles jointives sans chevauchement, split par blocs géographiques équilibré par classe,
tracé dans split_manifest.yaml.

Usage :
    .venv\\Scripts\\python.exe tools\\slice_zone.py <dataset_config.yaml> [--out D] [--seed N]
"""
import math
import random
from collections import Counter

from rasterio.windows import Window, bounds as fenetre_bounds

ORDRE_SPLITS = ("train", "valid", "test")  # ordre de départage des égalités


# ---------------------------------------------------------------------------
# Noyau géométrique
# ---------------------------------------------------------------------------

def grille_tuiles(transform, largeur_px, hauteur_px, tuile_px):
    """Grille de tuiles pleines, jointives, sans chevauchement.

    Les tuiles partielles de bord sont écartées (pas de padding : pas de bords
    noirs artificiels dans le dataset). Ordre (row, col).
    """
    tuiles = []
    for row in range(hauteur_px // tuile_px):
        for col in range(largeur_px // tuile_px):
            fenetre = Window(col * tuile_px, row * tuile_px, tuile_px, tuile_px)
            tuiles.append({
                "row": row,
                "col": col,
                "fenetre": fenetre,
                "bounds": fenetre_bounds(fenetre, transform),
            })
    return tuiles


def bloc_de(bounds, bloc_m):
    """Id du bloc contenant le CENTRE de la tuile (grille de bloc_m alignée sur 0)."""
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    return (math.floor(cx / bloc_m), math.floor(cy / bloc_m))


def affecter_splits(annos_par_bloc, cibles, seed):
    """Affectation gloutonne des blocs aux splits, équilibrée par classe.

    Blocs triés par richesse (total annotations) décroissante, départage par mélange
    seedé ; chaque bloc va au split de plus grand déficit pondéré
    Σ_c (1/total_c) * (part_cible_s * total_c - deja_alloue[s][c]).
    Les classes rares pèsent ainsi autant que les abondantes. Les blocs sans
    annotation ne sont pas affectés (ils restent hors split ; les tuiles vides des
    blocs affectés servent de vivier de négatifs).
    """
    parts = {s: cibles[s] / sum(cibles.values()) for s in cibles}
    total_par_classe = Counter()
    for c in annos_par_bloc.values():
        total_par_classe.update(c)

    rng = random.Random(seed)
    blocs = [b for b, c in annos_par_bloc.items() if sum(c.values()) > 0]
    rng.shuffle(blocs)  # départage des égalités de richesse
    blocs.sort(key=lambda b: -sum(annos_par_bloc[b].values()))

    alloue = {s: Counter() for s in cibles}
    affectation = {}
    for b in blocs:
        deficits = {}
        for s in cibles:
            deficits[s] = sum(
                (parts[s] * total_par_classe[c] - alloue[s][c]) / total_par_classe[c]
                for c in total_par_classe
            )
        meilleur = max(sorted(deficits, key=lambda s: ORDRE_SPLITS.index(s)),
                       key=lambda s: deficits[s])
        affectation[b] = meilleur
        alloue[meilleur].update(annos_par_bloc[b])
    return affectation
