"""Parité de DÉCISION PyTorch vs ONNX d'un modèle RF-DETR de DÉTECTION sur N tuiles d'un split
(promu le 2026-09-08 ; complète la porte de parité de l'export, qui ne teste que 2 images).

Même prétraitement que la porte du plugin (`dev/runner_onnx/export_to_onnx.py` :
`_load_test_image`). Pour chaque tuile : détections au plancher et au seuil déployé,
appariement glouton (même classe, boîte à --tol-boite près en coordonnées normalisées, score à
--tol-score % près). Verdict CONFORME si toutes les tuiles ont des décisions identiques aux deux
seuils ; sinon NON CONFORME (exit 1) avec les tuiles divergentes.

ATTENTION : venv_onnx du PLUGIN (torch + onnxruntime + rfdetr), `PYTHONIOENCODING=utf-8` :
    <plugin>/dev/runner_onnx/.venv_onnx/Scripts/python.exe tools/verif_parite_onnx.py <best.pth> <best.onnx> <dossier_tuiles> --resolution 672 --seuil 0.40 [--n 120] [--plugin <chemin plugin>]
"""
import argparse
import glob
import os
import pathlib
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pth")
    ap.add_argument("onnx")
    ap.add_argument("tuiles", help="dossier de tuiles png/jpg (ex. corpus/<nom>/test)")
    ap.add_argument("--resolution", type=int, required=True)
    ap.add_argument("--seuil", type=float, required=True, help="seuil déployé (confidence_default du model_card)")
    ap.add_argument("--plancher", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--tol-boite", type=float, default=1e-3)
    ap.add_argument("--tol-score", type=float, default=5.0)
    ap.add_argument("--plugin", default=os.path.expandvars(r"%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\archeologia-pipeline"))
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.plugin, "dev", "runner_onnx"))
    import export_to_onnx as ex  # noqa: E402
    import onnxruntime as ort  # noqa: E402
    import torch  # noqa: E402
    from rfdetr import RFDETR  # noqa: E402

    m = RFDETR.from_checkpoint(a.pth, resolution=a.resolution)
    net = m.model.model; net.eval(); net.export()
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    tuiles = sorted(glob.glob(os.path.join(a.tuiles, "*.png")) + glob.glob(os.path.join(a.tuiles, "*.jpg")))
    rng = np.random.default_rng(0)
    ech = [tuiles[i] for i in rng.choice(len(tuiles), min(a.n, len(tuiles)), replace=False)]

    def dets(boxes, logits, t):
        s = 1 / (1 + np.exp(-logits)); idx = np.where(s.max(1) >= t)[0]
        return [(int(s[i].argmax()), float(s[i].max()), boxes[i]) for i in idx]

    def apparie(A, B):
        if len(A) != len(B):
            return False, []
        pris = [False] * len(B); ecarts = []
        for c, s, b in A:
            cand = [(np.abs(b - b2).max(), j) for j, (c2, s2, b2) in enumerate(B)
                    if not pris[j] and c2 == c and np.abs(b - b2).max() <= a.tol_boite
                    and abs(s - s2) / max(s, 1e-6) * 100 <= a.tol_score]
            if not cand:
                return False, ecarts
            e, j = min(cand); pris[j] = True; ecarts.append(abs(s - B[j][1]) / max(s, 1e-6) * 100)
        return True, ecarts

    divergentes, n_det, ecarts = {a.plancher: [], a.seuil: []}, {a.plancher: 0, a.seuil: 0}, []
    for t in ech:
        x, _ = ex._load_test_image(a.resolution, "rfdetr", pathlib.Path(t))
        with torch.no_grad():
            out = net(torch.from_numpy(x))
        items = list(out.values()) if isinstance(out, dict) else list(out)
        pb, pl = items[0][0].numpy(), items[1][0].numpy()
        ob, ol = [o[0] for o in sess.run(None, {inp: x})]
        for th in divergentes:
            A, B = dets(pb, pl, th), dets(ob, ol, th)
            ok, e = apparie(A, B); n_det[th] += len(A); ecarts += e
            if not ok:
                divergentes[th].append(os.path.basename(t))
    for th in divergentes:
        print(f"seuil {th:.2f} : {len(ech) - len(divergentes[th])}/{len(ech)} tuiles identiques, {n_det[th]} détections PyTorch"
              + (f" — divergentes : {divergentes[th][:10]}" if divergentes[th] else ""))
    if ecarts:
        print(f"écart de score sur les appariées : médian {np.median(ecarts):.3f} %, max {max(ecarts):.2f} %")
    if any(divergentes.values()):
        print("NON CONFORME : décisions différentes entre PyTorch et ONNX"); sys.exit(1)
    print("CONFORME : décisions identiques PyTorch / ONNX aux deux seuils")


if __name__ == "__main__":
    main()
