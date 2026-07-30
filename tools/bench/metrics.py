"""Métriques du banc.

PRIMAIRE : F1 longueur @ tau (méthode buffer de Wiedemann, standard de l'extraction de
réseaux linéaires). La GT est un buffer de 7 m ARBITRAIRE autour d'une ligne : un
décalage latéral de 2,35 m suffit à faire tomber l'IoU masque sous 0,5, donc toute
métrique IoU note surtout l'erreur résiduelle de digitalisation et la largeur fabriquée
du buffer. La métrique longueur est invariante à la largeur et à la fragmentation par la
grille de tuilage.

SECONDAIRES (reportées, jamais optimisées) : F1 instance @ IoU boîte (continuité avec
evaluation_results.json), F1 instance @ IoU masque (le chiffre honnête), P/R/F1 pixel,
FP/km², indice de fragmentation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

GSD_M = 0.5  # m/pixel — LD 0,5 m


# --------------------------------------------------------------------------------------
# Rastérisation
# --------------------------------------------------------------------------------------

def masque_coco(anns: Sequence[dict], h: int, w: int,
                classes: Optional[Sequence[int]] = None) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for a in anns:
        if classes is not None and a["category_id"] not in classes:
            continue
        for poly in a["segmentation"]:
            pts = np.asarray(poly, np.float64).reshape(-1, 2)
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def masque_detections(dets: Sequence[dict], h: int, w: int,
                      classes: Optional[Sequence[int]] = None) -> np.ndarray:
    """Détections du décodeur (polygones normalisés par la taille image)."""
    m = np.zeros((h, w), np.uint8)
    for d in dets:
        if classes is not None and d["class_id"] not in classes:
            continue
        pts = np.asarray(d["polygon"], np.float64).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        for hole in d.get("polygon_holes", []):
            hp = np.asarray(hole, np.float64).reshape(-1, 2)
            hp[:, 0] *= w
            hp[:, 1] *= h
            cv2.fillPoly(m, [np.round(hp).astype(np.int32)], 0)
    return m.astype(bool)


# --------------------------------------------------------------------------------------
# Squelette et longueurs
# --------------------------------------------------------------------------------------

_VOISINS = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)


def _elaguer(skel: np.ndarray, min_px: int) -> np.ndarray:
    """Retire les barbules : branches terminales plus courtes que min_px.

    Les extrémités de buffer coupées par la grille de tuilage produisent des barbules
    diagonales aux coins — sans élagage elles comptent comme de la longueur prédite (ou
    GT) qui n'existe pas.
    """
    if min_px <= 0 or not skel.any():
        return skel
    s = skel.astype(np.uint8)
    voisinage = cv2.filter2D(s, -1, _VOISINS, borderType=cv2.BORDER_CONSTANT)
    n_voisins = np.where(s > 0, voisinage - 10, 0)
    noeuds = (n_voisins >= 3) & (s > 0)          # points de branchement
    extremites = (n_voisins == 1) & (s > 0)

    branches = s.copy()
    branches[noeuds] = 0
    lab, n = ndimage.label(branches, structure=np.ones((3, 3), int))
    if n == 0:
        return skel
    tailles = np.bincount(lab.ravel())
    a_extremite = np.zeros(n + 1, bool)
    a_extremite[np.unique(lab[extremites])] = True
    a_extremite[0] = False
    petites = (tailles < min_px) & a_extremite
    petites[0] = False
    return skel & ~petites[lab]


def squelette(masque: np.ndarray, elagage_px: int = 20) -> np.ndarray:
    if not masque.any():
        return np.zeros_like(masque, bool)
    return _elaguer(skeletonize(masque), elagage_px)


_ORTH = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8)
_DIAG = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], np.uint8)


def carte_longueur(skel: np.ndarray) -> np.ndarray:
    """Longueur (m) portée par CHAQUE pixel du squelette.

    `carte_longueur(s).sum()` vaut `longueur_m(s)` : c'est la même formule, simplement
    non sommée. Sert à ventiler une longueur par tuile de façon EXACTEMENT additive —
    découper le squelette en tranches perdrait les arêtes qui franchissent les bords de
    tuile, et la somme des morceaux serait inférieure au total.
    """
    s = skel.astype(np.uint8)
    if not s.any():
        return np.zeros(skel.shape, np.float64)
    n_o = (cv2.filter2D(s, -1, _ORTH, borderType=cv2.BORDER_CONSTANT) * s).astype(np.float64)
    n_d = (cv2.filter2D(s, -1, _DIAG, borderType=cv2.BORDER_CONSTANT) * s).astype(np.float64)
    return (n_o / 2.0 + (n_d / 2.0) * np.sqrt(2.0)) * GSD_M


def ccq_decompose(pred_masque: np.ndarray, gt: "CoteGT",
                  tuiles: Sequence[Tuple[int, int, int, int]],
                  tau_m: float = 5.0, elagage_px: int = 20) -> List[Dict[str, float]]:
    """CCQ ventilé par tuile, sans aucune squelettisation supplémentaire.

    Avec 3 mosaïques on n'a que 3 unités : impossible de produire un intervalle de
    confiance. Or la métrique est additive en longueur : les squelettes et cartes de
    distance sont calculés une fois à l'échelle de la mosaïque, puis la longueur est
    répartie par tuile. On passe de 3 à ~86 unités, donc à un bootstrap apparié utile.

    `tuiles` : emprises (y0, y1, x0, x1) en pixels de mosaïque.
    """
    sk_p = squelette(pred_masque, elagage_px)
    tau_px = tau_m / GSD_M
    vide = np.zeros(pred_masque.shape, bool)
    if gt.skel.any() and sk_p.any():
        dt_p = ndimage.distance_transform_edt(~sk_p).astype(np.float32)
        ok_g = gt.skel & (dt_p <= tau_px)
        ok_p = sk_p & (gt.dt <= tau_px)
    else:
        ok_g = ok_p = vide

    c_gt, c_pred = carte_longueur(gt.skel), carte_longueur(sk_p)
    c_okg = np.where(ok_g, c_gt, 0.0)
    c_okp = np.where(ok_p, c_pred, 0.0)

    parts = []
    for y0, y1, x0, x1 in tuiles:
        z = (slice(y0, y1), slice(x0, x1))
        parts.append({
            "len_gt_m": float(c_gt[z].sum()),
            "len_pred_m": float(c_pred[z].sum()),
            "len_tp_gt_m": float(c_okg[z].sum()),
            "len_tp_pred_m": float(c_okp[z].sum()),
            "n_seg_pred": 0, "n_seg_gt": 0,
        })
    return parts


def longueur_m(skel: np.ndarray) -> float:
    """Longueur d'un squelette en mètres.

    Un pas diagonal vaut sqrt(2) pixel : compter les pixels sous-estimerait de ~8 %.
    On compte les ARÊTES du squelette (chaque pixel relié à son voisin), moitié pour
    éviter le double comptage.
    """
    if not skel.any():
        return 0.0
    s = skel.astype(np.uint8)
    orth = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8)
    diag = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], np.uint8)
    n_o = float((cv2.filter2D(s, -1, orth, borderType=cv2.BORDER_CONSTANT) * s).sum())
    n_d = float((cv2.filter2D(s, -1, diag, borderType=cv2.BORDER_CONSTANT) * s).sum())
    return (n_o / 2.0 + (n_d / 2.0) * np.sqrt(2.0)) * GSD_M


# --------------------------------------------------------------------------------------
# MÉTRIQUE PRIMAIRE — CCQ / F1 longueur
# --------------------------------------------------------------------------------------

class CoteGT:
    """Côté GT du calcul CCQ, précalculé UNE fois par image.

    Le squelette de la GT et sa carte de distance ne dépendent d'aucun paramètre de
    config : les recalculer par config coûtait 25x le budget de la campagne.
    """

    __slots__ = ("skel", "dt", "longueur", "n_composantes")

    @classmethod
    def depuis_squelette(cls, skel: np.ndarray) -> "CoteGT":
        """Niveau B : les lignes du GPKG rasterisées à 1 px SONT la ligne de centre.
        Aucune squelettisation, donc aucun artefact de bout de buffer."""
        o = cls.__new__(cls)
        o.skel = skel
        o.dt = (ndimage.distance_transform_edt(~skel).astype(np.float32)
                if skel.any() else None)
        o.longueur = longueur_m(skel)
        o.n_composantes = (ndimage.label(skel, structure=np.ones((3, 3), int))[1]
                           if skel.any() else 0)
        return o

    def __init__(self, gt_masque: np.ndarray, elagage_px: int = 20):
        self.skel = squelette(gt_masque, elagage_px)
        # float32 suffit : la carte sert uniquement à comparer à une tolérance de 10 px,
        # et scipy la rend en float64, soit 141 Mo sur une mosaïque de 4536x3888 — x4 avec
        # les cartes par classe canonique.
        self.dt = (ndimage.distance_transform_edt(~self.skel).astype(np.float32)
                   if self.skel.any() else None)
        self.longueur = longueur_m(self.skel)
        self.n_composantes = (ndimage.label(self.skel, structure=np.ones((3, 3), int))[1]
                              if self.skel.any() else 0)


def ccq_prepare(pred_masque: np.ndarray, gt: CoteGT, tau_m: float = 5.0,
                elagage_px: int = 20) -> Dict[str, float]:
    """CCQ avec le côté GT déjà calculé. Même résultat que ccq(), ~2x plus rapide."""
    sk_p = squelette(pred_masque, elagage_px)
    tau_px = tau_m / GSD_M
    len_p = longueur_m(sk_p)
    len_g = gt.longueur

    if gt.skel.any() and sk_p.any():
        dt_p = ndimage.distance_transform_edt(~sk_p).astype(np.float32)
        comp = longueur_m(gt.skel & (dt_p <= tau_px)) / len_g if len_g else 0.0
        corr = longueur_m(sk_p & (gt.dt <= tau_px)) / len_p if len_p else 0.0
    else:
        comp = 0.0 if gt.skel.any() else float("nan")
        corr = 0.0 if sk_p.any() else float("nan")

    f1 = 0.0
    if comp == comp and corr == corr and (comp + corr) > 0:
        f1 = 2 * comp * corr / (comp + corr)
    n_comp_p = (ndimage.label(sk_p, structure=np.ones((3, 3), int))[1]
                if sk_p.any() else 0)
    return {"completude": comp, "correction": corr, "f1_len": f1,
            "len_gt_m": len_g, "len_pred_m": len_p,
            "len_tp_gt_m": comp * len_g if comp == comp else 0.0,
            "len_tp_pred_m": corr * len_p if corr == corr else 0.0,
            "n_seg_pred": n_comp_p, "n_seg_gt": gt.n_composantes}


def ccq(pred_masque: np.ndarray, gt_masque: np.ndarray, tau_m: float = 5.0,
        elagage_px: int = 20, valide: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Complétude / Correction / Qualité en longueur de linéaire.

    complétude = longueur de GT à moins de tau d'une prédiction / longueur GT
    correction = longueur prédite à moins de tau d'une GT / longueur prédite
    """
    if valide is not None:
        pred_masque = pred_masque & valide
        gt_masque = gt_masque & valide

    sk_p = squelette(pred_masque, elagage_px)
    sk_g = squelette(gt_masque, elagage_px)
    tau_px = tau_m / GSD_M

    len_g = longueur_m(sk_g)
    len_p = longueur_m(sk_p)

    if sk_g.any() and sk_p.any():
        dt_p = ndimage.distance_transform_edt(~sk_p).astype(np.float32)
        dt_g = ndimage.distance_transform_edt(~sk_g).astype(np.float32)
        comp = longueur_m(sk_g & (dt_p <= tau_px)) / len_g if len_g else 0.0
        corr = longueur_m(sk_p & (dt_g <= tau_px)) / len_p if len_p else 0.0
    else:
        comp = 0.0 if sk_g.any() else float("nan")
        corr = 0.0 if sk_p.any() else float("nan")

    f1 = 0.0
    if comp == comp and corr == corr and (comp + corr) > 0:
        f1 = 2 * comp * corr / (comp + corr)

    return {"completude": comp, "correction": corr, "f1_len": f1,
            "len_gt_m": len_g, "len_pred_m": len_p,
            "len_tp_gt_m": comp * len_g if comp == comp else 0.0,
            "len_tp_pred_m": corr * len_p if corr == corr else 0.0}


def agreger_frag(parts: Sequence[Dict[str, float]]) -> float:
    """Indice de fragmentation global : longueur moyenne d'un segment détecté rapportée
    à celle d'un segment GT. Une config qui gagne en complétude en éclatant les lignes
    en 40 morceaux n'est pas une amélioration."""
    lp = sum(p["len_pred_m"] for p in parts); np_ = sum(p.get("n_seg_pred", 0) for p in parts)
    lg = sum(p["len_gt_m"] for p in parts); ng = sum(p.get("n_seg_gt", 0) for p in parts)
    if not np_ or not ng or not lg:
        return float("nan")
    return (lp / np_) / (lg / ng)


def agreger_ccq(parts: Sequence[Dict[str, float]], tau_m: float = 5.0) -> Dict[str, float]:
    """Agrège des CCQ par tuile en pondérant par la LONGUEUR, pas en moyennant des ratios.

    Moyenner des ratios donnerait le même poids à une tuile de 20 m et à une tuile de 2 km.
    """
    lg = sum(p["len_gt_m"] for p in parts)
    lp = sum(p["len_pred_m"] for p in parts)
    tg = sum(p["len_tp_gt_m"] for p in parts)
    tp = sum(p["len_tp_pred_m"] for p in parts)
    comp = tg / lg if lg else float("nan")
    corr = tp / lp if lp else float("nan")
    f1 = 2 * comp * corr / (comp + corr) if (comp == comp and corr == corr and comp + corr) else 0.0
    return {"completude": comp, "correction": corr, "f1_len": f1,
            "len_gt_m": lg, "len_pred_m": lp, "tau_m": tau_m}


def indice_fragmentation(pred_masque: np.ndarray, gt_masque: np.ndarray) -> float:
    """Longueur moyenne d'un segment détecté / longueur moyenne d'un segment GT.

    Garde-fou : une config qui gagne en complétude en éclatant les lignes en 40 morceaux
    n'est pas une amélioration.
    """
    def moy(m):
        sk = squelette(m)
        if not sk.any():
            return 0.0
        lab, n = ndimage.label(sk, structure=np.ones((3, 3), int))
        if n == 0:
            return 0.0
        return longueur_m(sk) / n
    mg = moy(gt_masque)
    return (moy(pred_masque) / mg) if mg else float("nan")


# --------------------------------------------------------------------------------------
# Métriques pixel
# --------------------------------------------------------------------------------------

def metriques_pixel(pred: np.ndarray, gt: np.ndarray,
                    valide: Optional[np.ndarray] = None) -> Dict[str, float]:
    if valide is not None:
        pred, gt = pred & valide, gt & valide
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if (p == p and r == r and p + r) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": f1, "iou": iou}


# --------------------------------------------------------------------------------------
# Métriques instance
# --------------------------------------------------------------------------------------

def _iou_boites(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a [N,4], b [M,4] en xyxy -> IoU [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _iou_masques(ma: List[np.ndarray], mb: List[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(ma), len(mb)))
    for i, a in enumerate(ma):
        for j, b in enumerate(mb):
            inter = np.count_nonzero(a & b)
            if inter:
                out[i, j] = inter / np.count_nonzero(a | b)
    return out


def apparier(pred_cls: np.ndarray, gt_cls: np.ndarray, iou: np.ndarray,
             seuil: float, conf: Optional[np.ndarray] = None,
             class_aware: bool = True, ordre: str = "conf") -> Tuple[np.ndarray, np.ndarray]:
    """Appariement glouton 1-1. Retourne (index gt apparié par pred, ou -1 ; idem inverse).

    ordre="conf" : convention COCO, monotone en seuil -> une seule passe donne toute la
    courbe P/R. ordre="iou" : convention du notebook d'entraînement, à n'utiliser que
    pour reproduire evaluation_results.json.
    """
    n_p, n_g = len(pred_cls), len(gt_cls)
    app_p = np.full(n_p, -1, int)
    app_g = np.full(n_g, -1, int)
    if n_p == 0 or n_g == 0:
        return app_p, app_g

    m = iou.copy()
    if class_aware:
        m[pred_cls[:, None] != gt_cls[None, :]] = 0.0
    m[m < seuil] = 0.0

    if ordre == "conf":
        ordre_p = np.argsort(-conf) if conf is not None else np.arange(n_p)
        for i in ordre_p:
            libres = np.nonzero((app_g < 0) & (m[i] > 0))[0]
            if len(libres):
                j = libres[np.argmax(m[i][libres])]
                app_p[i], app_g[j] = j, i
    else:
        paires = [(m[i, j], i, j) for i in range(n_p) for j in range(n_g) if m[i, j] > 0]
        for _, i, j in sorted(paires, key=lambda t: -t[0]):
            if app_p[i] < 0 and app_g[j] < 0:
                app_p[i], app_g[j] = j, i
    return app_p, app_g


def pr_par_seuil(scores_tp: np.ndarray, n_gt: int,
                 seuils: Sequence[float]) -> List[Dict[str, float]]:
    """Courbe P/R depuis (score, est_tp) accumulés — appariement conf-décroissant."""
    out = []
    for s in seuils:
        sel = scores_tp[:, 0] >= s
        tp = int(scores_tp[sel, 1].sum())
        fp = int(sel.sum()) - tp
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / n_gt if n_gt else float("nan")
        f1 = 2 * p * r / (p + r) if (p == p and r == r and p + r) else 0.0
        out.append({"seuil": s, "precision": p, "recall": r, "f1": f1,
                    "tp": tp, "fp": fp, "fn": n_gt - tp})
    return out
