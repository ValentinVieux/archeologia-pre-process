"""Parité de DÉCISION PyTorch vs ONNX d'un modèle RF-DETR (détection OU segmentation d'instances)
sur N tuiles d'un split (promu le 2026-09-08, segmentation ajoutée le 2026-09-09 ; complète la
porte de parité de l'export, qui ne teste que 2 images).

Même prétraitement que la porte du plugin (`dev/runner_onnx/export_to_onnx.py` :
`_load_test_image`). Pour chaque tuile : détections au plancher et au seuil déployé,
appariement glouton (même classe, boîte à --tol-boite près en coordonnées normalisées, score à
--tol-score % près) ; en segmentation, les masques binarisés (signe du logit, règle de la porte)
de chaque paire doivent avoir un IoU >= --tol-masque AU SEUIL DÉPLOYÉ (au plancher : rapporté).
Au plancher, une détection dont le score chevauche le plancher à --tol-score % près (arrondi bf16)
est retirée des deux côtés avant appariement et rapportée comme « tolérée » (2026-09-09 :
depressions_grandes_seg_ld_v1, 2 tuiles à 0,050). Verdict CONFORME si toutes les tuiles ont des
décisions identiques aux deux seuils ; sinon NON CONFORME (exit 1) avec les tuiles divergentes.

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
    ap.add_argument("--tol-masque", type=float, default=0.99,
                    help="segmentation : IoU minimal des masques binarisés (signe du logit) de chaque paire appariée")
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

    # Sorties reconnues PAR FORME, jamais par nom : l'exportateur du plugin nomme les sorties seg
    # dans l'ordre inverse des tenseurs (constaté 2026-09-09 : « pred_masks » = boîtes [1,Q,4],
    # « pred_boxes » = logits [1,Q,C], « pred_logits » = masques [1,Q,h,w]) ; le runner lit par
    # position. boîtes = 3D à 4 colonnes ; masques = 4D ; logits = l'autre 3D. En segmentation la
    # décision de masque = signe du logit (même règle que la porte de l'export : masques binarisés).
    noms_onnx = [o.name for o in sess.get_outputs()]
    formes_onnx = [o.shape for o in sess.get_outputs()]
    seg = any(len(f) == 4 for f in formes_onnx)
    print(f"sorties ONNX : {list(zip(noms_onnx, formes_onnx))} — tâche {'segmentation' if seg else 'détection'} (appariement par forme)")

    def par_forme(tenseurs, quoi):
        """quoi = 'box' | 'logit' | 'mask' ; tenseurs = liste de tableaux (numpy ou torch) sans lot."""
        for t in tenseurs:
            nd, last = t.ndim, t.shape[-1]
            if quoi == "mask" and nd == 3:
                return t
            if quoi == "box" and nd == 2 and last == 4:
                return t
            if quoi == "logit" and nd == 2 and last != 4:
                return t
        raise KeyError(quoi)

    def dets(boxes, logits, t):
        s = 1 / (1 + np.exp(-logits)); idx = np.where(s.max(1) >= t)[0]
        return [(int(s[i].argmax()), float(s[i].max()), boxes[i], int(i)) for i in idx]

    def apparie(A, B, mA=None, mB=None):
        """Appariement glouton classe + boîte + score ; en seg, IoU des masques binarisés des appariées."""
        pris = [False] * len(B); ecarts = []; ious = []; non_a = []
        for c, s, b, i in A:
            cand = [(np.abs(b - b2).max(), j) for j, (c2, s2, b2, _) in enumerate(B)
                    if not pris[j] and c2 == c and np.abs(b - b2).max() <= a.tol_boite
                    and abs(s - s2) / max(s, 1e-6) * 100 <= a.tol_score]
            if not cand:
                non_a.append(s); continue
            e, j = min(cand); pris[j] = True; ecarts.append(abs(s - B[j][1]) / max(s, 1e-6) * 100)
            if mA is not None:
                p, o = mA[i] > 0, mB[B[j][3]] > 0
                u = np.logical_or(p, o).sum()
                ious.append((1.0 if u == 0 else float(np.logical_and(p, o).sum() / u), s))
        non_b = [B[j][1] for j in range(len(B)) if not pris[j]]
        if non_a or non_b:
            motif = (f"{len(A)} PyTorch / {len(B)} ONNX ; non appariées PyTorch {[round(s, 4) for s in non_a]}"
                     f" / ONNX {[round(s, 4) for s in non_b]}")
            return False, ecarts, ious, motif, non_a + non_b
        return True, ecarts, ious, "", []

    divergentes, n_det, ecarts, ious = {a.plancher: [], a.seuil: []}, {a.plancher: 0, a.seuil: 0}, [], {a.plancher: [], a.seuil: []}
    tolerees = []  # chevauchements du plancher (rapportés, non comptés comme divergences)
    for t in ech:
        x, _ = ex._load_test_image(a.resolution, "rfdetr", pathlib.Path(t))
        with torch.no_grad():
            out = net(torch.from_numpy(x))
        tv = [v[0].numpy() for v in (out.values() if isinstance(out, dict) else out)]
        to = [v[0] for v in sess.run(None, {inp: x})]
        pb, pl = par_forme(tv, "box"), par_forme(tv, "logit")
        ob, ol = par_forme(to, "box"), par_forme(to, "logit")
        pm = par_forme(tv, "mask") if seg else None
        om = par_forme(to, "mask") if seg else None
        for th in divergentes:
            A, B = dets(pb, pl, th), dets(ob, ol, th)
            ok, e, iu, motif, orphelines = apparie(A, B, pm, om); n_det[th] += len(A); ecarts += e; ious[th] += iu
            if not ok and th == a.plancher:
                # Chevauchement du plancher : une détection dont le score est à --tol-score % du
                # plancher peut passer d'un côté et pas de l'autre (arrondi bf16) sans que la décision
                # diffère à un seuil utile. Tolérée si TOUTES les non appariées sont dans cette bande.
                lim = th * (1 + a.tol_score / 100)
                if all(s < lim for s in orphelines):
                    tolerees.append(f"{os.path.basename(t)} : {motif} — toutes sous {lim:.4f}")
                    ok, motif = True, ""
            faibles = [(v, s) for v, s in iu if v < a.tol_masque]
            if not ok:
                divergentes[th].append((os.path.basename(t), motif))
            elif faibles and th == a.seuil:  # l'IoU de masque n'est exigé qu'au seuil déployé ; au plancher il est rapporté
                divergentes[th].append((os.path.basename(t), "masques IoU " + ", ".join(f"{v:.3f} (score {s:.3f})" for v, s in faibles)))
    for th in divergentes:
        print(f"seuil {th:.3f} : {len(ech) - len(divergentes[th])}/{len(ech)} tuiles identiques, {n_det[th]} détections PyTorch")
        for nom, motif in divergentes[th][:10]:
            print(f"   divergente {nom} : {motif}")
        if ious[th]:
            v = [x for x, _ in ious[th]]
            print(f"   masques binarisés des appariées : IoU min {min(v):.4f}, médian {np.median(v):.4f}, "
                  f"identiques (IoU 1,0) {sum(x == 1.0 for x in v)}/{len(v)}")
    for ligne in tolerees[:10]:
        print(f"   chevauchement du plancher toléré — {ligne}")
    if ecarts:
        print(f"écart de score sur les appariées : médian {np.median(ecarts):.3f} %, max {max(ecarts):.2f} %")
    if divergentes[a.seuil]:
        print("NON CONFORME : décisions différentes entre PyTorch et ONNX AU SEUIL DÉPLOYÉ"); sys.exit(1)
    if divergentes[a.plancher]:
        # Le plancher n'est pas un seuil d'exploitation : on le rapporte sans le confondre avec le seuil
        # déployé (exit 2 = décision humaine, à consigner dans NOTE-export-onnx.md du modèle).
        print(f"CONFORME AU SEUIL DÉPLOYÉ {a.seuil:.3f} (décisions et masques identiques) ; "
              f"{len(divergentes[a.plancher])} tuile(s) divergente(s) au plancher {a.plancher:.2f} — à consigner")
        sys.exit(2)
    print("CONFORME : décisions identiques PyTorch / ONNX aux deux seuils")


if __name__ == "__main__":
    main()
