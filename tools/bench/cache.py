"""Passe forward ONNX + cache des sorties brutes.

Idée directrice du banc : une passe forward par (poids, transformation d'entrée), puis
TOUTES les expériences de décodage / seuil / post-traitement sont un rejeu CPU à zéro
passe forward. Le cache stocke les logits de masque à leur résolution NATIVE (162x162)
en float16 : suréchantillonner au moment du cache figerait l'interpolation et coûterait
16x le disque.

float16 sur les logits de masque -> erreur de probabilité < 2.5e-4, incapable de faire
basculer une décision `prob >= seuil` sauf à 2.5e-4 du seuil. Le test de parité tourne
cache désactivé, donc la revendication bit-à-bit porte bien sur le code, pas le stockage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .decode import SliceOut

# --------------------------------------------------------------------------------------
# TTA — le groupe diédral, sans algèbre à la main
# --------------------------------------------------------------------------------------
# Chaque variante est une op sur tableau. Le mapping de coordonnées inverse est obtenu en
# appliquant la MÊME op à des images de coordonnées : aucun risque d'erreur de signe.

TTA_VARIANTES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "id":        lambda a: a,
    "hflip":     lambda a: a[:, ::-1],
    "vflip":     lambda a: a[::-1, :],
    "rot180":    lambda a: a[::-1, ::-1],
    "rot90":     lambda a: np.rot90(a, 1),
    "rot270":    lambda a: np.rot90(a, 3),
    "transpose": lambda a: a.T,
    "antitrans": lambda a: np.rot90(a, 1)[:, ::-1],
}

# Inverse de chaque variante (le groupe est fermé ; rot90 <-> rot270, le reste est involutif)
_INV = {"id": "id", "hflip": "hflip", "vflip": "vflip", "rot180": "rot180",
        "rot90": "rot270", "rot270": "rot90", "transpose": "transpose",
        "antitrans": "antitrans"}


def _grille_inverse(n: int, variante: str) -> Tuple[np.ndarray, np.ndarray]:
    """Coordonnées ORIGINALES normalisées à chaque pixel de l'espace transformé."""
    f = TTA_VARIANTES[variante]
    xs = np.tile((np.arange(n) + 0.5) / n, (n, 1))
    ys = np.tile(((np.arange(n) + 0.5) / n).reshape(-1, 1), (1, n))
    return np.ascontiguousarray(f(xs)), np.ascontiguousarray(f(ys))


def appliquer_tta(img: np.ndarray, variante: str) -> np.ndarray:
    return np.ascontiguousarray(TTA_VARIANTES[variante](img))


def defaire_tta_masques(masks: np.ndarray, variante: str) -> np.ndarray:
    """Ramène des masques [K, M, M] de l'espace transformé vers l'espace original."""
    if variante == "id":
        return masks
    inv = TTA_VARIANTES[_INV[variante]]
    return np.ascontiguousarray(np.stack([inv(m) for m in masks]))


def defaire_tta_boites(boxes: np.ndarray, variante: str, n: int = 512) -> np.ndarray:
    """Ramène des boîtes cxcywh normalisées vers l'espace original.

    Transforme les deux coins et reprend l'enveloppe axis-aligned : exact, car les 8
    transformations du groupe envoient un rectangle axis-aligned sur un rectangle
    axis-aligned.
    """
    if variante == "id" or boxes.size == 0:
        return boxes
    gx, gy = _grille_inverse(n, variante)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def conv(xn, yn):
        c = np.clip((xn * n).astype(int), 0, n - 1)
        r = np.clip((yn * n).astype(int), 0, n - 1)
        return gx[r, c], gy[r, c]

    ax, ay = conv(x1, y1)
    bx, by = conv(x2, y2)
    nx1, nx2 = np.minimum(ax, bx), np.maximum(ax, bx)
    ny1, ny2 = np.minimum(ay, by), np.maximum(ay, by)
    return np.stack([(nx1 + nx2) / 2, (ny1 + ny2) / 2, nx2 - nx1, ny2 - ny1], axis=1)


# --------------------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------------------

def charger_session(model_path: str, device: str = "auto", tf32: bool = False):
    """Charge la session ONNX via le code du plugin, puis VÉRIFIE le provider.

    onnxruntime retombe SILENCIEUSEMENT sur CPU si l'EP CUDA ne charge pas — ce qui
    corromprait la clé de cache. On refuse plutôt que de mesurer sans le savoir.

    tf32=False : MESURÉ ICI — avec les options CUDA par défaut, TF32 est actif et les
    sorties divergent du CPU de ~2-5 % en erreur relative médiane (jusqu'à 1.4e+02 sur
    les logits de masque), ce qui change le nombre de détections. `use_tf32=0` ramène
    l'écart à 1.7e-02. Comme la production tourne sur le binaire CPU, le banc doit
    reproduire le CPU : TF32 reste désactivé par défaut.
    """
    import onnxruntime as ort
    from pipeline.cv.computer_vision_onnx import _load_onnx_model

    if device == "gpu":
        dispo = ort.get_available_providers()
        if "CUDAExecutionProvider" not in dispo:
            raise RuntimeError(f"--device gpu demandé mais CUDA absent : {dispo}")
        opts = {} if tf32 else {"use_tf32": "0"}
        session = ort.InferenceSession(
            model_path, providers=[("CUDAExecutionProvider", opts), "CPUExecutionProvider"])
        provider = session.get_providers()[0]
        if provider != "CUDAExecutionProvider":
            raise RuntimeError(f"repli silencieux sur {provider!r}")
        info = session.get_inputs()[0]
        return session, info.name, info.shape, {}, f"{provider}{'' if tf32 else '+notf32'}"

    session, input_name, input_shape, meta = _load_onnx_model(model_path)
    provider = session.get_providers()[0]
    if device == "cpu" and provider != "CPUExecutionProvider":
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        provider = "CPUExecutionProvider"
    return session, input_name, input_shape, meta, provider


def forward(session, input_name: str, img_rgb: np.ndarray, model_hw: Tuple[int, int],
            variante: str = "id") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Une passe forward. Retourne (boxes, logits, masks) dans l'espace ORIGINAL.

    Le prétraitement est celui du plugin, importé verbatim.
    """
    from PIL import Image
    from pipeline.cv.computer_vision_onnx import _preprocess_image

    arr = appliquer_tta(img_rgb, variante) if variante != "id" else img_rgb
    tensor = _preprocess_image(Image.fromarray(arr), model_hw, "rfdetr")
    out = session.run(None, {input_name: tensor})
    boxes, logits, masks = out[0][0], out[1][0], out[2][0]
    if variante != "id":
        boxes = defaire_tta_boites(boxes, variante)
        masks = defaire_tta_masques(masks, variante)
    return boxes.astype(np.float32), logits.astype(np.float32), masks.astype(np.float32)


# --------------------------------------------------------------------------------------
# Cache disque
# --------------------------------------------------------------------------------------

def _sha_fichier(path: Path, n: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()


def cle_cache(model_path: str, ort_version: str, provider: str, prepro: str,
              input_hw: Tuple[int, int], variante: str, plancher: float) -> str:
    brut = "|".join([_sha_fichier(Path(model_path))[:16], ort_version, provider,
                     prepro, f"{input_hw[0]}x{input_hw[1]}", variante, f"{plancher:.4f}"])
    return hashlib.sha256(brut.encode()).hexdigest()[:12]


class Cache:
    def __init__(self, racine: Path, cle: str, meta: Optional[dict] = None):
        self.dir = Path(racine) / cle
        self.meta_path = self.dir / "meta.json"
        if meta is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            # `split` est informatif, pas discriminant : la clé ne dépend que du moteur et
            # du prétraitement, et aucun nom de tuile n'est partagé entre splits (vérifié).
            # Plusieurs splits peuvent donc légitimement cohabiter dans un même cache.
            informatifs = {"n_unites", "split"}
            if self.meta_path.exists():
                ancien = json.loads(self.meta_path.read_text(encoding="utf-8"))
                divergent = {k: (ancien.get(k), v) for k, v in meta.items()
                             if k not in informatifs and ancien.get(k) != v}
                if divergent:
                    raise RuntimeError(f"meta.json du cache diverge : {divergent}")
                splits = sorted(set(str(ancien.get("split", "")).split(",")) |
                                {str(meta.get("split", ""))} - {""})
                ancien["split"] = ",".join(s for s in splits if s)
                self.meta_path.write_text(json.dumps(ancien, indent=2), encoding="utf-8")
            else:
                self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def chemin(self, unite: str) -> Path:
        return self.dir / f"{unite}.npz"

    def existe(self, unite: str) -> bool:
        return self.chemin(unite).exists()

    def ecrire(self, unite: str, boxes: np.ndarray, logits: np.ndarray,
               masks: np.ndarray, plancher: float) -> int:
        scores = 1.0 / (1.0 + np.exp(-logits))
        garde = np.nonzero(scores.max(axis=1) >= plancher)[0].astype(np.int16)
        p = self.chemin(unite)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p,                                   # non compressé : fp16 ~ incompressible
                 boxes=boxes[garde].astype(np.float32),
                 logits=logits[garde].astype(np.float32),
                 masks=masks[garde].astype(np.float16),
                 qidx=garde,
                 boxes_full=boxes.astype(np.float32),
                 logits_full=logits.astype(np.float32))
        return len(garde)

    def lire(self, unite: str, start_x: int = 0, start_y: int = 0,
             slice_w: int = 648, slice_h: int = 648) -> SliceOut:
        d = np.load(self.chemin(unite))
        return SliceOut(
            start_x=start_x, start_y=start_y, slice_w=slice_w, slice_h=slice_h,
            boxes=d["boxes"], logits=d["logits"],
            masks=d["masks"].astype(np.float32),
            qidx=d["qidx"], logits_full=d["logits_full"],
        )

    def lire_brut(self, unite: str) -> dict:
        d = np.load(self.chemin(unite))
        return {k: d[k] for k in d.files}
