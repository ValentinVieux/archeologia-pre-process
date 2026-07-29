"""E1 — le chemin du plugin (ONNX + argmax) contre la référence (PyTorch + top-k officiel).

Le bloc `test` d'evaluation_results.json (F1 .331) N'A PAS été produit par le plugin :
il vient de `model.predict()` en PyTorch, avec le post-traitement top-k officiel de
rfdetr et un appariement sur IoU de BOÎTE, sans SAHI. Le chemin réellement exécuté en
production n'a jamais été mesuré. Ce script quantifie l'écart entre les deux moteurs.

Trois des quatre causes candidates sont déjà des axes du balayage (gratuits depuis le
cache) : argmax vs top-k, sigmoid->resize vs resize(logit)->0, et le n_real à 6. Il ne
reste ici que la numérique ONNX vs PyTorch, qui exige torch.

Lancement (venv du plugin, qui a torch 2.10 + rfdetr 1.8.3) :
  & "$PLUGIN\\dev\\runner_onnx\\.venv_onnx\\Scripts\\python.exe" tools/bench/e1_onnx_vs_torch.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PLUGIN = Path(os.environ.get(
    "ARCHEO_PLUGIN",
    r"C:\Users\valen\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\archeologia-pipeline"))
DATA = Path(os.environ.get("BENCH_DATA", r"D:\bench\data\test"))
MODELE = PLUGIN / "data" / "models" / "lineaires_seg_v2_1"
N_TUILES = int(os.environ.get("E1_N", "24"))
CONF = 0.3

sys.path.insert(0, str(PLUGIN / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / u) if u else float("nan")


def main() -> int:
    import cv2
    import torch
    from PIL import Image
    from pipeline.cv.computer_vision_onnx import _load_onnx_model, _preprocess_image
    from tools.bench.data import Corpus
    from tools.bench.decode import Params, SliceOut, run as decoder

    corpus = Corpus(DATA)
    ids = sorted(corpus.images)
    images = [corpus.images[i] for i in ids[::max(1, len(ids) // N_TUILES)][:N_TUILES]]

    session, input_name, shape, _ = _load_onnx_model(str(MODELE / "weights" / "best.onnx"))
    mw, mh = int(shape[3]), int(shape[2])

    from rfdetr.detr import RFDETRSegLarge
    modele = RFDETRSegLarge(pretrain_weights=str(MODELE / "weights" / "best.pth"),
                            resolution=mw)
    interne = getattr(getattr(modele, "model", None), "model", None)
    if interne is None:
        print("[!] structure rfdetr inattendue :", type(modele), dir(modele)[:20])
        return 2
    interne.eval()

    print(f"{len(images)} tuiles, conf={CONF}\n")
    print(f"{'tuile':<42}{'n_onnx':>7}{'n_torch':>8}{'IoU union':>11}"
          f"{'d_logits':>10}{'d_masques':>11}")
    ious, d_log, d_msk, n_o, n_t = [], [], [], 0, 0
    for img in images:
        pil = Image.open(corpus.chemin(img)).convert("RGB")
        w, h = pil.size
        t = _preprocess_image(pil, (mw, mh), "rfdetr")
        o = session.run(None, {input_name: t})
        with torch.no_grad():
            sortie = interne(torch.from_numpy(t))
        # Le graphe ONNX a ses sorties MÉLANGÉES (export_to_onnx.py:827 inverse les noms) :
        # on aligne par forme, pas par nom.
        par_forme = {}
        for k, v in (sortie.items() if isinstance(sortie, dict) else enumerate(sortie)):
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            par_forme[arr.shape[1:]] = arr[0]
        boxes_t = par_forme.get(o[0][0].shape)
        logits_t = par_forme.get(o[1][0].shape)
        masks_t = par_forme.get(o[2][0].shape)
        if logits_t is None or masks_t is None:
            print(f"  [!] formes torch {list(par_forme)} vs onnx "
                  f"{[o[i][0].shape for i in range(3)]}")
            return 2

        d_log.append(float(np.abs(o[1][0] - logits_t).max()))
        d_msk.append(float(np.abs(o[2][0] - masks_t).max()))

        p = Params(confidence=CONF, class_offset=0)
        mk = lambda b, l, m: [SliceOut(0, 0, w, h, b.astype(np.float32),
                                       l.astype(np.float32), m.astype(np.float32))
                              for _ in range(4)]
        d_onnx = decoder(mk(o[0][0], o[1][0], o[2][0]), w, h, mw, mh, p)
        d_torch = decoder(mk(boxes_t if boxes_t is not None else o[0][0], logits_t, masks_t),
                          w, h, mw, mh, p)

        def union(dets):
            m = np.zeros((h, w), np.uint8)
            for d in dets:
                pts = np.asarray(d["polygon"], np.float64).reshape(-1, 2)
                pts[:, 0] *= w
                pts[:, 1] *= h
                cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
            return m.astype(bool)

        i = iou(union(d_onnx), union(d_torch))
        ious.append(i)
        n_o += len(d_onnx)
        n_t += len(d_torch)
        print(f"{img['file_name']:<42}{len(d_onnx):>7}{len(d_torch):>8}{i:>11.4f}"
              f"{d_log[-1]:>10.2e}{d_msk[-1]:>11.2e}")

    print(f"\nTOTAL  detections ONNX={n_o}  torch={n_t}  "
          f"(ecart {100*abs(n_o-n_t)/max(n_t,1):.1f} %)")
    print(f"IoU median des unions de masques : {np.nanmedian(ious):.4f}")
    print(f"ecart max logits  : median {np.median(d_log):.2e}  max {max(d_log):.2e}")
    print(f"ecart max masques : median {np.median(d_msk):.2e}  max {max(d_msk):.2e}")
    print("\nLecture : un IoU proche de 1 et des ecarts ~1e-4 signifient que l'export ONNX")
    print("est fidele et que la difference avec evaluation_results.json vient du DECODAGE")
    print("(argmax vs top-k) et de l'appariement (IoU boite), pas du moteur.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
