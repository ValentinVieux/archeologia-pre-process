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
from shapely import make_valid
from shapely.geometry import box as boite

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


# ---------------------------------------------------------------------------
# Entités -> polygones COCO
# ---------------------------------------------------------------------------

def preparer_entites(gdf, buffer_m):
    """Géométries d'une couche -> liste de polygones prêts à rasteriser.

    buffer_m : largeur TOTALE pour les lignes (buffer de buffer_m/2), rayon pour les
    points, None pour les polygones (inchangés). MultiX explosés, vides écartées,
    invalides réparées (make_valid).
    """
    polys = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if buffer_m is not None:
            rayon = buffer_m / 2 if "LineString" in geom.geom_type else buffer_m
            geom = geom.buffer(rayon)
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.geom_type == "Polygon":
            polys.append(geom)
        elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
            polys.extend(g for g in geom.geoms if g.geom_type == "Polygon")
    return polys


def polygone_vers_coco(poly, bounds, tuile_px):
    """Anneau extérieur d'un polygone -> coordonnées pixels COCO [x1,y1,x2,y2,...].

    Origine au coin haut-gauche de la tuile, y vers le bas, arrondi 2 décimales.
    Les anneaux intérieurs (trous) sont ignorés — rarissimes sur nos entités
    bufferisées, et le format polygone COCO ne les représente pas.
    """
    minx, _, _, maxy = bounds
    sx = tuile_px / (bounds[2] - bounds[0])
    sy = tuile_px / (bounds[3] - bounds[1])
    anneau = []
    for x, y in list(poly.exterior.coords)[:-1]:
        anneau.extend([round((x - minx) * sx, 2), round((maxy - y) * sy, 2)])
    return [anneau] if len(anneau) >= 6 else []


def annotations_tuile(polys_par_classe, bounds, tuile_px):
    """Clip des polygones à la tuile -> annotations {classe, segmentation, bbox_px, aire_px}."""
    tuile_geo = boite(*bounds)
    sx = tuile_px / (bounds[2] - bounds[0])
    sy = tuile_px / (bounds[3] - bounds[1])
    annos = []
    for classe, polys in polys_par_classe.items():
        for p in polys:
            if not p.intersects(tuile_geo):
                continue
            clip = p.intersection(tuile_geo)
            morceaux = ([clip] if clip.geom_type == "Polygon"
                        else [g for g in getattr(clip, "geoms", ())
                              if g.geom_type == "Polygon"])
            for m in morceaux:
                segmentation = polygone_vers_coco(m, bounds, tuile_px)
                if not segmentation:
                    continue
                mnx, mny, mxx, mxy = m.bounds
                annos.append({
                    "classe": classe,
                    "segmentation": segmentation,
                    "bbox_px": [round((mnx - bounds[0]) * sx, 2),
                                round((bounds[3] - mxy) * sy, 2),
                                round((mxx - mnx) * sx, 2),
                                round((mxy - mny) * sy, 2)],
                    "aire_px": round(m.area * sx * sy, 2),
                })
    return annos
