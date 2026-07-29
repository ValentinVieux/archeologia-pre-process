"""Port PARAMÉTRÉ de `_run_rfdetr_seg_with_sahi` (plugin archeologia-pipeline).

C'est le seul fork du banc : tout le reste (prétraitement, géoréférencement,
post-traitement géo) importe le plugin verbatim. Toute dérive avec la production
entre donc ici, et uniquement ici.

CONTRAT : avec `Params()` (tous les défauts), ce module doit reproduire la sortie du
plugin **octet pour octet**. C'est ce que vérifie tests/test_parity_plugin.py.
Chaque paramètre non-défaut correspond à un axe de la matrice d'expériences.

Correspondance ligne à ligne avec le plugin (computer_vision_onnx.py) :
  l. 366-367  SNAP / MASK_IOU_MERGE_THRESHOLD   -> Params.snap / mask_iou_merge
  l. 408      np.maximum                        -> Params.fusion
  l. 427-429  sigmoid + max/argmax              -> Params.decode
  l. 472-480  sigmoid -> resize float           -> Params.mask_interp / mask_logit_space
  l. 550      prob >= 0.5                       -> Params.mask_cutoff
  l. 566      area < 10                         -> Params.min_area_px
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------------------
# Paramètres
# --------------------------------------------------------------------------------------

@dataclass
class Params:
    """Tous les défauts == constantes du plugin. Ne JAMAIS changer un défaut ici."""

    # --- décodage des logits ---
    decode: str = "argmax"           # argmax | topk | multilabel
    topk: int = 300                  # decode=topk : num_select sur (requête x classe)
    confidence: float = 0.3          # seuil global
    confidence_par_classe: Optional[Dict[int, float]] = None  # surcharge par classe
    score_rule: str = "raw"          # raw | mask_mean | raw_x_mask_mean
    class_offset: int = 0
    # Le plugin calcule n_real = max(1, n_cls - class_offset). Avec class_offset=0 et
    # 6 colonnes de logits pour 5 classes, ça vaut 6 : une colonne 5 gagnante passerait
    # en `classe_6` fantôme. Renseigner n_classes borne au vrai nombre de classes.
    # (Mesuré : la colonne 5 ne gagne jamais, sigmoïde max 0,0048 — défaut latent.)
    n_classes: Optional[int] = None

    # --- masques ---
    mask_cutoff: float = 0.5
    mask_interp: str = "linear"      # linear | cubic | nearest
    mask_logit_space: bool = False   # True = resize(logit) puis >0 (voie officielle rfdetr)

    # --- découpage SAHI (lu par le harnais, pas par decode.run) ---
    # ATTENTION : sur une image 648 avec slice=648, get_slice_bboxes renvoie QUATRE
    # fenêtres identiques (stride 519 < 648, la fenêtre débordante est ramenée en
    # arrière sur [0,0,648,648]). Ces passes en double ne sont pas neutres : elles
    # refont la recherche de clé d'instance et fusionnent des instances qu'une passe
    # unique garde séparées. L'overlap est donc un axe actif MÊME au niveau tuile.
    sahi_slice: int = 648
    sahi_overlap: float = 0.2
    # True = supprime les fenêtres STRICTEMENT identiques renvoyées par get_slice_bboxes.
    # C'est le correctif propre des passes redondantes (4 fenêtres identiques sur une
    # tuile 648) : contrairement à overlap=0 il ne change rien au recouvrement utile
    # sur un grand raster, il n'enlève que le calcul refait à l'identique.
    sahi_dedup: bool = False

    # None = heuristique du plugin `boxes.max() <= 1.0`. MESURÉ : elle bascule sur
    # 50,7 % des tuiles (dépassement médian 0,0157, une boîte qui mord hors de l'image),
    # et le plugin conclut alors « boîtes en pixels absolus ». Il ne multiplie plus par
    # 648, donc gcx = int(0,98) = 0 et TOUTES les clés d'instance s'écrasent sur
    # (classe, 0, 0, 0, 0). True force la lecture normalisée (RF-DETR sort toujours du
    # cxcywh normalisé).
    boxes_normalisees: Optional[bool] = None

    # --- accumulation inter-tuiles ---
    snap: int = 16
    mask_iou_merge: float = 0.15
    fusion: str = "max"              # max | mean

    # --- suppression de doublons (absente du plugin) ---
    nms_mask_iou: Optional[float] = None      # None = pas de NMS
    nms_class_agnostic: bool = True

    # --- extraction des contours ---
    min_area_px: float = 10.0
    morph_close: int = 0             # 0 = aucune ; sinon taille du noyau (3,5,7)
    morph_open: int = 0
    simplify_px: float = 0.0         # Douglas-Peucker sur le contour, en pixels
    compactness_max: Optional[float] = None   # rejette 4*pi*A/P^2 > seuil (formes rondes)

    # --- post-traitement géo (niveau B uniquement ; défauts = args.yaml + model_card) ---
    geo_merge: bool = True
    geo_merge_buffer_m: float = 0.5
    geo_remove_overlaps: bool = True
    geo_overlap_strategy: str = "difference"   # difference | relation
    geo_min_area_m2: float = 0.0

    def cle(self) -> dict:
        return asdict(self)


_INTERP = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "nearest": cv2.INTER_NEAREST}


@dataclass
class SliceOut:
    """Sorties brutes du modèle pour une tuile, dans l'espace de la mosaïque.

    boxes/logits/masks sont alignés sur les K requêtes retenues au moment du cache
    (score >= plancher). Comme tout seuil balayé est >= ce plancher, les requêtes
    écartées ne peuvent pas produire de détection : la troncature est sans effet.
    `logits_full` garde les 200 requêtes pour les analyses de calibration et de
    plafond de rappel, qui elles regardent sous le plancher.
    """
    start_x: int
    start_y: int
    slice_w: int
    slice_h: int
    boxes: np.ndarray    # [K, 4] cxcywh, espace modèle
    logits: np.ndarray   # [K, C]
    masks: np.ndarray    # [K, Mh, Mw] logits de masque
    qidx: Optional[np.ndarray] = None        # indices d'origine des K requêtes
    logits_full: Optional[np.ndarray] = None  # [200, C]


def _cut(p: "Params") -> float:
    """Seuil de binarisation dans l'espace courant (probabilité ou logit)."""
    return 0.0 if p.mask_logit_space else p.mask_cutoff


# --------------------------------------------------------------------------------------
# Sélection des requêtes
# --------------------------------------------------------------------------------------

def _selection(logits: np.ndarray, p: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(indices de requête, class_id, score) selon la stratégie de décodage."""
    scores = 1.0 / (1.0 + np.exp(-logits))          # sigmoid [K, C] — comme le plugin

    if p.decode == "argmax":
        # Plugin : une détection par requête, argmax sur les classes.
        q = np.arange(len(scores))
        return q, scores.argmax(axis=1), scores.max(axis=1)

    if p.decode == "topk":
        # rfdetr officiel : top-k sur (requête x classe) aplati -> une requête peut
        # produire plusieurs hypothèses de classe.
        flat = scores.reshape(-1)
        k = min(p.topk, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]
        return idx // scores.shape[1], idx % scores.shape[1], flat[idx]

    if p.decode == "multilabel":
        # Toute paire (requête, classe) au-dessus du seuil.
        q, c = np.nonzero(scores >= _seuil_min(p))
        return q, c, scores[q, c]

    raise ValueError(f"decode inconnu : {p.decode}")


def _seuil_min(p: Params) -> float:
    if p.confidence_par_classe:
        return min([p.confidence, *p.confidence_par_classe.values()])
    return p.confidence


def _seuil(p: Params, class_id: int) -> float:
    if p.confidence_par_classe and class_id in p.confidence_par_classe:
        return p.confidence_par_classe[class_id]
    return p.confidence


# --------------------------------------------------------------------------------------
# Accumulation par instance (fidèle au plugin)
# --------------------------------------------------------------------------------------

def _inst_iou(inst: dict, sy: int, sx: int, ey: int, ex: int,
              new_vals: np.ndarray, cutoff: float) -> float:
    bb = inst["bbox"]
    iy0, ix0 = max(sy, bb[0]), max(sx, bb[1])
    iy1, ix1 = min(ey, bb[2]), min(ex, bb[3])
    if iy0 >= iy1 or ix0 >= ix1:
        return 0.0
    patch = inst["prob"][iy0 - bb[0]:iy1 - bb[0], ix0 - bb[1]:ix1 - bb[1]]
    cnt = inst.get("cnt")
    if cnt is not None:
        # En fusion « moyenne », `prob` est une SOMME de votes : la comparer telle quelle
        # au seuil rendrait la fusion d'instances d'autant plus permissive qu'une zone a
        # déjà reçu de contributions. Il faut normaliser avant de seuiller.
        patch = patch / np.maximum(
            cnt[iy0 - bb[0]:iy1 - bb[0], ix0 - bb[1]:ix1 - bb[1]], 1.0)
    e = patch >= cutoff
    n = new_vals[iy0 - sy:iy1 - sy, ix0 - sx:ix1 - sx] >= cutoff
    return np.count_nonzero(e & n) / max(np.count_nonzero(e | n), 1)


def _agrandir(inst: dict, sy: int, sx: int, ey: int, ex: int) -> None:
    bb = inst["bbox"]
    ny0, nx0 = min(bb[0], sy), min(bb[1], sx)
    ny1, nx1 = max(bb[2], ey), max(bb[3], ex)
    if ny0 == bb[0] and nx0 == bb[1] and ny1 == bb[2] and nx1 == bb[3]:
        return
    oy, ox = bb[0] - ny0, bb[1] - nx0
    oh, ow = bb[2] - bb[0], bb[3] - bb[1]
    for champ in ("prob", "cnt"):
        if inst.get(champ) is None:
            continue
        neuf = np.zeros((ny1 - ny0, nx1 - nx0), dtype=np.float32)
        neuf[oy:oy + oh, ox:ox + ow] = inst[champ]
        inst[champ] = neuf
    inst["bbox"] = [ny0, nx0, ny1, nx1]


def _fusionner(inst: dict, sy: int, sx: int, ey: int, ex: int,
               new_vals: np.ndarray, conf: float, p: Params) -> None:
    _agrandir(inst, sy, sx, ey, ex)
    bb = inst["bbox"]
    ry, rx = sy - bb[0], sx - bb[1]
    rh, rw = ey - sy, ex - sx
    zone = (slice(ry, ry + rh), slice(rx, rx + rw))
    if p.fusion == "max":
        inst["prob"][zone] = np.maximum(inst["prob"][zone], new_vals)
    elif p.fusion == "mean":
        # Moyenne pondérée par le vote — ce que fait déjà la voie SegFormer du plugin.
        inst["prob"][zone] += new_vals
        inst["cnt"][zone] += 1.0
    else:
        raise ValueError(f"fusion inconnue : {p.fusion}")
    inst["conf"] = max(inst["conf"], conf)


def _cle_instance(instance_maps: dict, base_key: tuple, sy: int, sx: int, ey: int, ex: int,
                  new_vals: np.ndarray, p: Params) -> tuple:
    """Reproduit exactement la recherche de clé du plugin (l. 484-520)."""
    cut = _cut(p)
    if base_key in instance_maps:
        if _inst_iou(instance_maps[base_key], sy, sx, ey, ex, new_vals, cut) >= p.mask_iou_merge:
            return base_key
        for suffix in range(1, 100):
            cand = base_key + (suffix,)
            if cand not in instance_maps:
                break
            if _inst_iou(instance_maps[cand], sy, sx, ey, ex, new_vals, cut) >= p.mask_iou_merge:
                return cand
        for suffix in range(1, 100):
            cand = base_key + (suffix,)
            if cand not in instance_maps:
                return cand
        return base_key
    for suffix in range(1, 100):
        cand = base_key + (suffix,)
        if cand not in instance_maps:
            break
        if _inst_iou(instance_maps[cand], sy, sx, ey, ex, new_vals, cut) >= p.mask_iou_merge:
            return cand
    return base_key


def accumuler(slices: Sequence[SliceOut], orig_w: int, orig_h: int,
              model_w: int, model_h: int, p: Params) -> Dict[tuple, dict]:
    """Étape 1 : accumulation des cartes de probabilité par instance."""
    instance_maps: Dict[tuple, dict] = {}
    n_real: Optional[int] = None
    interp = _INTERP[p.mask_interp]

    for sl in slices:
        if sl.logits.size == 0:
            continue
        q_idx, cls_idx, scores_sel = _selection(sl.logits, p)

        n_cls = sl.logits.shape[1]
        if n_real is None:
            n_real = (p.n_classes if p.n_classes is not None
                      else max(1, n_cls - p.class_offset))

        end_x = min(sl.start_x + sl.slice_w, orig_w)
        end_y = min(sl.start_y + sl.slice_h, orig_h)
        actual_w, actual_h = end_x - sl.start_x, end_y - sl.start_y
        scale_x, scale_y = sl.slice_w / model_w, sl.slice_h / model_h
        boxes_norm = (p.boxes_normalisees if p.boxes_normalisees is not None
                      else (bool(sl.boxes.max() <= 1.0) if sl.boxes.size else True))

        for q, c, sc in zip(q_idx, cls_idx, scores_sel):
            class_id = int(c) - p.class_offset
            if class_id < 0 or class_id >= n_real:
                continue

            # Filtre avant le masque quand le score n'en dépend pas : c'est l'ordre du
            # plugin, et ça évite d'interpoler 200 masques au lieu de 20.
            if p.score_rule == "raw" and float(sc) < _seuil(p, class_id):
                continue

            mask_logit = sl.masks[q].astype(np.float32)
            if p.mask_logit_space:
                # Voie officielle rfdetr : interpoler le LOGIT puis seuiller à 0.
                mask_slice = cv2.resize(mask_logit, (sl.slice_w, sl.slice_h), interpolation=interp)
                new_vals = mask_slice[:actual_h, :actual_w]
                bin_ref = new_vals > 0.0
            else:
                # Voie plugin : sigmoid puis interpoler la PROBABILITÉ.
                mask_prob = 1.0 / (1.0 + np.exp(-mask_logit))
                mask_slice = cv2.resize(mask_prob, (sl.slice_w, sl.slice_h), interpolation=interp)
                new_vals = mask_slice[:actual_h, :actual_w]
                bin_ref = new_vals >= p.mask_cutoff

            confidence = float(sc)
            if p.score_rule != "raw":
                aire = int(np.count_nonzero(bin_ref))
                moy = float(new_vals[bin_ref].mean()) if aire else 0.0
                if p.mask_logit_space:
                    moy = float(1.0 / (1.0 + np.exp(-moy)))
                confidence = moy if p.score_rule == "mask_mean" else confidence * moy

            if confidence < _seuil(p, class_id):
                continue

            cx, cy, w, h = sl.boxes[q]
            if boxes_norm:
                cx, cy, w, h = cx * model_w, cy * model_h, w * model_w, h * model_h
            gcx = int(sl.start_x + cx * scale_x)
            gcy = int(sl.start_y + cy * scale_y)
            gw, gh = int(w * scale_x), int(h * scale_y)

            base_key = (class_id,
                        round(gcx / p.snap) * p.snap, round(gcy / p.snap) * p.snap,
                        round(gw / p.snap) * p.snap, round(gh / p.snap) * p.snap)

            # Toujours passer par la recherche complète : même quand base_key est absente,
            # le plugin inspecte les clés suffixées et peut fusionner dans l'une d'elles.
            key = _cle_instance(instance_maps, base_key, sl.start_y, sl.start_x,
                                end_y, end_x, new_vals, p)

            if key not in instance_maps:
                instance_maps[key] = {
                    "prob": new_vals.copy(),
                    "cnt": np.ones_like(new_vals) if p.fusion == "mean" else None,
                    "bbox": [sl.start_y, sl.start_x, end_y, end_x],
                    "conf": confidence,
                    "class_id": class_id,
                }
            else:
                _fusionner(instance_maps[key], sl.start_y, sl.start_x, end_y, end_x,
                           new_vals, confidence, p)

    return instance_maps


# --------------------------------------------------------------------------------------
# NMS masque (absent du plugin)
# --------------------------------------------------------------------------------------

def _nms(instances: List[dict], p: Params) -> List[dict]:
    if p.nms_mask_iou is None or len(instances) < 2:
        return instances
    ordre = sorted(range(len(instances)), key=lambda i: -instances[i]["conf"])
    garde: List[int] = []
    for i in ordre:
        a = instances[i]
        ba, ma = a["bbox"], a["_bin"]
        rejete = False
        for j in garde:
            b = instances[j]
            if not p.nms_class_agnostic and a["class_id"] != b["class_id"]:
                continue
            bb = b["bbox"]
            iy0, ix0 = max(ba[0], bb[0]), max(ba[1], bb[1])
            iy0, ix0 = int(iy0), int(ix0)
            iy1, ix1 = int(min(ba[2], bb[2])), int(min(ba[3], bb[3]))
            if iy0 >= iy1 or ix0 >= ix1:
                continue
            pa = ma[iy0 - ba[0]:iy1 - ba[0], ix0 - ba[1]:ix1 - ba[1]]
            pb = b["_bin"][iy0 - bb[0]:iy1 - bb[0], ix0 - bb[1]:ix1 - bb[1]]
            inter = np.count_nonzero(pa & pb)
            if not inter:
                continue
            union = np.count_nonzero(ma) + np.count_nonzero(b["_bin"]) - inter
            if inter / max(union, 1) >= p.nms_mask_iou:
                rejete = True
                break
        if not rejete:
            garde.append(i)
    return [instances[i] for i in sorted(garde)]


# --------------------------------------------------------------------------------------
# Extraction des polygones
# --------------------------------------------------------------------------------------

def _binariser(inst: dict, p: Params) -> np.ndarray:
    prob = inst["prob"]
    if p.fusion == "mean" and inst.get("cnt") is not None:
        prob = prob / np.maximum(inst["cnt"], 1.0)
    seuil = 0.0 if p.mask_logit_space else p.mask_cutoff
    binaire = (prob > seuil if p.mask_logit_space else prob >= seuil).astype(np.uint8)
    if p.morph_open:
        k = np.ones((p.morph_open, p.morph_open), np.uint8)
        binaire = cv2.morphologyEx(binaire, cv2.MORPH_OPEN, k)
    if p.morph_close:
        k = np.ones((p.morph_close, p.morph_close), np.uint8)
        binaire = cv2.morphologyEx(binaire, cv2.MORPH_CLOSE, k)
    return binaire


def extraire(instance_maps: Dict[tuple, dict], orig_w: int, orig_h: int,
             p: Params) -> List[Dict]:
    """Étape 2 : cartes d'instance -> détections (polygones normalisés)."""
    instances = []
    for inst in instance_maps.values():
        binaire = _binariser(inst, p)
        if not binaire.any():
            continue
        inst = dict(inst)
        inst["_bin"] = binaire.astype(bool)
        instances.append(inst)

    instances = _nms(instances, p)

    detections: List[Dict] = []
    for inst in instances:
        binaire = inst["_bin"].astype(np.uint8)
        off_y, off_x = inst["bbox"][0], inst["bbox"][1]
        contours, hierarchy = cv2.findContours(binaire, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or hierarchy is None:
            continue
        hierarchy = hierarchy[0]

        for cidx, contour in enumerate(contours):
            if hierarchy[cidx][3] != -1:
                continue
            if p.simplify_px > 0:
                contour = cv2.approxPolyDP(contour, p.simplify_px, True)
            area = cv2.contourArea(contour)
            if area < p.min_area_px or len(contour) < 3:
                continue
            if p.compactness_max is not None:
                perim = cv2.arcLength(contour, True)
                if perim > 0 and (4.0 * np.pi * area) / (perim * perim) > p.compactness_max:
                    continue

            polygon = []
            for pt in contour:
                polygon.extend([float(pt[0][0] + off_x) / orig_w,
                                float(pt[0][1] + off_y) / orig_h])

            holes = []
            child = hierarchy[cidx][2]
            while child != -1:
                hc = contours[child]
                if cv2.contourArea(hc) >= p.min_area_px and len(hc) >= 3:
                    holes.append([v for pt in hc for v in
                                  (float(pt[0][0] + off_x) / orig_w,
                                   float(pt[0][1] + off_y) / orig_h)])
                child = hierarchy[child][0]

            x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
            det = {
                "class_id": inst["class_id"],
                "confidence": inst["conf"],
                "polygon": polygon,
                "bbox": [float(x_b + off_x), float(y_b + off_y),
                         float(x_b + off_x + w_b), float(y_b + off_y + h_b)],
                "area": float(area),
            }
            if holes:
                det["polygon_holes"] = holes
            detections.append(det)

    return detections


def run(slices: Sequence[SliceOut], orig_w: int, orig_h: int,
        model_w: int, model_h: int, p: Params) -> List[Dict]:
    """Chaîne complète : accumulation + extraction. Équivalent de _run_rfdetr_seg_with_sahi."""
    return extraire(accumuler(slices, orig_w, orig_h, model_w, model_h, p),
                    orig_w, orig_h, p)
