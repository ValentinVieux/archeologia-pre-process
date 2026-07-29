"""Niveau B — mosaïques géoréférencées.

Le niveau A (tuiles 648 isolées) ne peut pas mesurer ce qui se passe ENTRE les tuiles :
fusion inter-tuiles sur du contenu réellement différent, et tout le post-traitement géo.
Il faut un raster large.

Les tuiles de test étant jointives, une mosaïque de tuiles contiguës est une découpe
fidèle du raster LD source. La vérité terrain vient des GPKG v2 recalés : ce sont des
LIGNES, donc rasterisées à 1 px elles SONT la ligne de centre — plus aucune
approximation par squelettisation d'un buffer, contrairement au niveau A.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .data import GSD_M, PAS_M, TUILE_PX, Tuile

# Couches GPKG -> classe du modèle, d'après configs/lineaires_<zone>_ld_648_v2.yaml.
# `voie` est fusionnée en `parcellaire` au niveau corpus (corpus_lineaires_v2.yaml).
# Les couches `ignorer:` sont volontairement absentes : elles ne sont pas annotées.
COUCHES = {
    "54_foret_de_haye":       {"parcellaire": 0, "talus_fosse": 3},
    "77_fontainebleau":       {"parcellaire": 0, "talus": 1, "fosse": 2, "chemin_creux": 4},
    "78_rambouillet":         {"parcellaire": 0, "talus": 1, "fosse": 2, "talus_fosse": 3,
                               "chemin_creux": 4, "voie": 0},
    "78_saint_germain_marly": {"parcellaire": 0, "voie": 0, "talus": 1, "fosse": 2},
    "41_blois":               {"parcellaire": 0, "talus_fosse": 3, "chemin_creux": 4},
}
IGNOREES = {"54_foret_de_haye": ["rempart"],
            "77_fontainebleau": ["tranchees_et_boyaux"],
            "78_rambouillet": ["amenagement_militaire"],
            "78_saint_germain_marly": ["amenagement_militaire"],
            "41_blois": []}


class Mosaique:
    def __init__(self, tuiles: Sequence[Tuile]):
        self.tuiles = sorted(tuiles, key=lambda t: (t.row, t.col))
        self.zone = self.tuiles[0].zone
        self.row0 = min(t.row for t in self.tuiles)
        self.col0 = min(t.col for t in self.tuiles)
        self.n_lig = max(t.row for t in self.tuiles) - self.row0 + 1
        self.n_col = max(t.col for t in self.tuiles) - self.col0 + 1
        self.w = self.n_col * TUILE_PX
        self.h = self.n_lig * TUILE_PX
        b = [t.bounds for t in self.tuiles]
        self.xmin = min(x[0] for x in b)
        self.ymax = max(x[3] for x in b)
        self.xmax = self.xmin + self.n_col * PAS_M
        self.ymin = self.ymax - self.n_lig * PAS_M

    @property
    def id(self) -> str:
        return f"{self.zone}_r{self.row0:04d}_c{self.col0:04d}_{self.n_lig}x{self.n_col}"

    def px(self, t: Tuile) -> Tuple[int, int]:
        return (t.col - self.col0) * TUILE_PX, (t.row - self.row0) * TUILE_PX

    def pgw(self) -> str:
        # Convention COIN (pas de décalage demi-pixel) pour coller à conversion_shp.py
        # l.1113, qui fait x_geo = x_origin + x_px * pixel_width sans recentrage.
        return f"{GSD_M}\n0.0\n0.0\n{-GSD_M}\n{self.xmin}\n{self.ymax}\n"

    def geo(self, x_px: np.ndarray, y_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.xmin + x_px * GSD_M, self.ymax - y_px * GSD_M

    def construire(self, dossier_tuiles: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne (raster RGB, masque de validité au pixel).

        Les trous (tuiles absentes du split) sont remplis par le gris MÉDIAN de la
        mosaïque : le noir créerait des arêtes artificielles qu'un détecteur de
        structures linéaires prendrait pour du signal.
        """
        from PIL import Image
        canvas = np.zeros((self.h, self.w, 3), np.uint8)
        valide = np.zeros((self.h, self.w), bool)
        gris = []
        for t in self.tuiles:
            x, y = self.px(t)
            a = np.asarray(Image.open(dossier_tuiles / t.nom).convert("RGB"))
            canvas[y:y + TUILE_PX, x:x + TUILE_PX] = a
            valide[y:y + TUILE_PX, x:x + TUILE_PX] = True
            gris.append(int(np.median(a[..., 0])))
        if not valide.all():
            canvas[~valide] = int(np.median(gris))
        return canvas, valide

    def meta(self) -> dict:
        return {"id": self.id, "zone": self.zone, "n_tuiles": len(self.tuiles),
                "grille": [self.n_lig, self.n_col], "taille_px": [self.h, self.w],
                "bounds_l93": [self.xmin, self.ymin, self.xmax, self.ymax],
                "aire_km2": (self.n_lig * PAS_M) * (self.n_col * PAS_M) / 1e6,
                "taux_remplissage": round(len(self.tuiles) / (self.n_lig * self.n_col), 4),
                "tuiles": [t.nom for t in self.tuiles]}


def gt_lignes(mosaique: Mosaique, gpkg: Path,
              classes: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, Dict[int, float]]:
    """Rasterise les LIGNES de vérité terrain (1 px) dans la grille de la mosaïque.

    Une ligne rasterisée à 1 px est déjà la ligne de centre : pas de squelettisation,
    donc pas d'artefact de bout de buffer.
    """
    import geopandas as gpd
    from shapely.geometry import box

    emprise = box(mosaique.xmin, mosaique.ymin, mosaique.xmax, mosaique.ymax)
    skel = np.zeros((mosaique.h, mosaique.w), np.uint8)
    longueurs: Dict[int, float] = {}
    couches = COUCHES.get(mosaique.zone, {})
    for nom_couche, class_id in couches.items():
        try:
            gdf = gpd.read_file(gpkg, layer=nom_couche)
        except Exception:
            continue
        if gdf.empty:
            continue
        if classes is not None and class_id not in classes:
            continue
        gdf = gdf[gdf.intersects(emprise)]
        if gdf.empty:
            continue
        gdf = gdf.clip(emprise)
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            parties = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
            for part in parties:
                if part.geom_type != "LineString" or part.length == 0:
                    continue
                xs, ys = np.asarray(part.coords).T
                px = np.round((xs - mosaique.xmin) / GSD_M).astype(np.int32)
                py = np.round((mosaique.ymax - ys) / GSD_M).astype(np.int32)
                cv2.polylines(skel, [np.stack([px, py], axis=1)], False, 1, 1)
                longueurs[class_id] = longueurs.get(class_id, 0.0) + part.length
    return skel.astype(bool), longueurs


def choisir(composantes: Sequence[Sequence[Tuile]], par_zone: int = 1,
            max_tuiles: int = 64) -> List[Mosaique]:
    """La plus grande composante de chaque zone, plafonnée en taille.

    Le plafond n'est pas cosmétique : dans l'accumulation du plugin, chaque instance
    porte une carte de probabilité à la taille de la tuile (648^2 float32 = 1,7 Mo) et
    l'agrandit à chaque fusion. Des milliers d'instances font sauter la mémoire.
    """
    vues: Dict[str, int] = {}
    out = []
    for comp in sorted(composantes, key=lambda c: -len(c)):
        z = comp[0].zone
        if vues.get(z, 0) >= par_zone or len(comp) > max_tuiles:
            continue
        vues[z] = vues.get(z, 0) + 1
        out.append(Mosaique(comp))
    return out
