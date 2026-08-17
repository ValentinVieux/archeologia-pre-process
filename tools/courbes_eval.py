"""Courbes d'évaluation standard des modèles de segmentation (style YOLO/RF-DETR).

Pour 1 à N modèles RF-DETR-Seg évalués sur le MÊME jeu COCO (splits valid+test),
produit les planches de référence avec courbes SUPERPOSÉES et point de
fonctionnement F1-max annoté sur chaque panneau :
  - courbes_seuils_pr.png : Précision/confiance, Rappel/confiance, F1/confiance,
    courbe Précision-Rappel (AP@0,5 toutes-points) ;
  - f1_par_classe.png (si >1 classe après fusion) ;
  - zones_et_masques.png (rappel par zone + histogramme IoU des TP, si champ
    `zone` dans le COCO) ;
  - appariements.json : appariements bruts mis en CACHE — toute relance sans
    supprimer ce fichier ne refait PAS l'inférence (re-rendu seul).

Protocole (doctrine maison, cf. docs/rapport_test_adaf.html et rétrospective
v2_1) : inférence au plancher 0,05, appariement glouton par classe à IoU
masque >= 0,5, métriques par balayage post-hoc du seuil — jamais de seuil fixe
(scores IA-BCE écrasés). RÈGLE : ces courbes sont générées pour TOUT nouveau
modèle entraîné, et déposées dans data/models/<modele>/ du plugin quand la
comparaison concerne un modèle installé.

Usage (venv_adaf OBLIGATOIRE — torch/rfdetr/matplotlib) :
  D:\\veille_irlande\\venv_adaf\\Scripts\\python.exe tools\\courbes_eval.py
      --coco <dossier avec valid/ et test/>
      --modele "nom=poids.pth@resolution" [--modele ...]
      --out <dossier de sortie>
      [--fusion talus=talus_fosse --fusion fosse=talus_fosse]
      [--titre "..."] [--plancher 0.05]
Les classes de chaque modèle sont lues dans le sidecar weights/best.json voisin
du .pth s'il existe, sinon supposées identiques aux catégories du COCO.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#888888", "#c1272d", "#1f77b4", "#2c6e49", "#7c4a1e"]


def inferer(modeles, coco_dir, fusion, plancher):
    from PIL import Image
    from pycocotools.coco import COCO
    from rfdetr import RFDETRSegLarge

    donnees = {}
    for nom, cfg in modeles.items():
        modele = RFDETRSegLarge(pretrain_weights=cfg["poids"], resolution=cfg["resolution"])
        modele.optimize_for_inference()
        decal = None
        enregs = []
        for split in ("valid", "test"):
            chemin = os.path.join(coco_dir, split, "_annotations.coco.json")
            if not os.path.exists(chemin):
                continue
            coco = COCO(chemin)
            cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
            for img_id in coco.getImgIds():
                info = coco.loadImgs(img_id)[0]
                im = Image.open(os.path.join(coco_dir, split, info["file_name"])).convert("RGB")
                d = modele.predict(im, threshold=plancher)
                n_pred = 0 if d.mask is None else len(d)
                if decal is None and n_pred:
                    # vieux exports : background en colonne 0 -> ids 1..N
                    decal = 1 if (int(d.class_id.min()) >= 1
                                  and int(d.class_id.max()) >= len(cfg["noms"])) else 0
                gts = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
                masques, cls_gt = [], []
                for a in gts:
                    c = fusion.get(cats[a["category_id"]], cats[a["category_id"]])
                    masques.append(coco.annToMask(a).astype(bool))
                    cls_gt.append(c)
                pris = [False] * len(masques)
                matches, fps = [], []
                for i in (np.argsort(-d.confidence) if n_pred else []):
                    idx = int(d.class_id[i]) - (decal or 0)
                    if not (0 <= idx < len(cfg["noms"])):
                        continue
                    c = fusion.get(cfg["noms"][idx], cfg["noms"][idx])
                    m = d.mask[i].astype(bool)
                    meilleur, mi = 0.0, -1
                    for j, g in enumerate(masques):
                        if pris[j] or cls_gt[j] != c:
                            continue
                        inter = np.logical_and(m, g).sum()
                        if inter:
                            iou = inter / np.logical_or(m, g).sum()
                            if iou > meilleur:
                                meilleur, mi = iou, j
                    if meilleur >= 0.5:
                        pris[mi] = True
                        matches.append([float(d.confidence[i]), float(meilleur), c])
                    else:
                        fps.append([float(d.confidence[i]), c])
                enregs.append({"split": split, "zone": info.get("zone", ""),
                               "n_gt": len(masques), "gt_classes": cls_gt,
                               "matches": matches, "fps": fps})
        donnees[nom] = {"decal": decal, "enregs": enregs}
        print(f"{nom} : {sum(len(e['matches']) for e in enregs)} TP potentiels, offset {decal}")
        del modele
        import gc
        import torch
        gc.collect(); torch.cuda.empty_cache()
    return donnees


def prf(enregs, s, classe=None):
    tp = sum(sum(1 for m in e["matches"] if m[0] >= s and (classe is None or m[2] == classe))
             for e in enregs)
    fp = sum(sum(1 for f in e["fps"] if f[0] >= s and (classe is None or f[1] == classe))
             for e in enregs)
    ngt = sum((sum(1 for c in e["gt_classes"] if c == classe) if classe else e["n_gt"])
              for e in enregs)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / ngt if ngt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def planche_principale(donnees, titre, sortie):
    seuils = np.arange(0.05, 0.96, 0.01)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(titre, fontsize=12)
    for k, (nom, dd) in enumerate(donnees.items()):
        enregs, c = dd["enregs"], PALETTE[k % len(PALETTE)]
        P = np.array([prf(enregs, s)[0] for s in seuils])
        R = np.array([prf(enregs, s)[1] for s in seuils])
        F = np.array([prf(enregs, s)[2] for s in seuils])
        i = int(np.argmax(F))
        s0, p0, r0, f0 = seuils[i], P[i], R[i], F[i]
        for ax, Y, y0, texte, dy in ((axes[0][0], P, p0, f"P={p0:.2f} @ {s0:.2f}", -12),
                                     (axes[0][1], R, r0, f"R={r0:.2f} @ {s0:.2f}", 8)):
            ax.plot(seuils, Y, color=c, label=nom)
            ax.scatter([s0], [y0], color=c, zorder=5)
            ax.annotate(texte, (s0, y0), textcoords="offset points",
                        xytext=(8, dy + 12 * k * np.sign(dy)), fontsize=8, color=c)
        axes[1][0].plot(seuils, F, color=c, label=f"{nom} — max {f0:.3f} @ {s0:.2f}")
        axes[1][0].scatter([s0], [f0], color=c, zorder=5)
        axes[1][0].annotate(f"F1={f0:.3f}\n@ {s0:.2f}", (s0, f0), textcoords="offset points",
                            xytext=(8, 6), fontsize=8, color=c)
        couples = sorted({(prf(enregs, s)[1], prf(enregs, s)[0])
                          for s in np.arange(0.02, 0.99, 0.005)})
        rr = np.array([0.0] + [x[0] for x in couples] + [couples[-1][0]])
        pp = np.array([1.0] + [x[1] for x in couples] + [0.0])
        for j in range(len(pp) - 2, -1, -1):
            pp[j] = max(pp[j], pp[j + 1])
        ap = float(np.trapz(pp[:-1], rr[:-1]))
        axes[1][1].plot(rr[:-1], pp[:-1], color=c, label=f"{nom} — AP@0.5 {ap:.3f}")
        axes[1][1].scatter([r0], [p0], color=c, zorder=5, marker="*", s=120)
        axes[1][1].annotate(f"F1max: P={p0:.2f}, R={r0:.2f}", (r0, p0), textcoords="offset points",
                            xytext=(8, 6 + 12 * k), fontsize=8, color=c)
    for ax, t in ((axes[0][0], "Précision vs confiance"), (axes[0][1], "Rappel vs confiance"),
                  (axes[1][0], "F1 vs confiance"), (axes[1][1], "Courbe Précision-Rappel (★ = F1-max)")):
        ax.set_title(t); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)


def planche_classes(donnees, classes, sortie):
    seuils = np.arange(0.05, 0.96, 0.01)
    fig, axes = plt.subplots(1, len(classes), figsize=(5.3 * len(classes), 4.5))
    axes = np.atleast_1d(axes)
    fig.suptitle("F1 vs confiance par classe", fontsize=12)
    for k, cl in enumerate(classes):
        for j, (nom, dd) in enumerate(donnees.items()):
            F = [prf(dd["enregs"], s, cl)[2] for s in seuils]
            i = int(np.argmax(F))
            axes[k].plot(seuils, F, color=PALETTE[j % len(PALETTE)],
                         label=f"{nom} — max {F[i]:.3f} @ {seuils[i]:.2f}")
            axes[k].scatter([seuils[i]], [F[i]], color=PALETTE[j % len(PALETTE)], zorder=5)
        axes[k].set_title(cl); axes[k].grid(alpha=0.3); axes[k].set_ylim(0, 1)
        axes[k].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)


def planche_zones(donnees, sortie):
    seuils = np.arange(0.05, 0.96, 0.01)
    zones = sorted({e.get("zone", "") for dd in donnees.values() for e in dd["enregs"]
                    if e.get("zone")})
    if not zones:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    larg = 0.8 / len(donnees)
    for k, (nom, dd) in enumerate(donnees.items()):
        enregs, c = dd["enregs"], PALETTE[k % len(PALETTE)]
        F = [prf(enregs, s)[2] for s in seuils]
        s0 = seuils[int(np.argmax(F))]
        rappels = []
        for z in zones:
            sous = [e for e in enregs if e["zone"] == z]
            tp = sum(sum(1 for m in e["matches"] if m[0] >= s0) for e in sous)
            ngt = sum(e["n_gt"] for e in sous)
            rappels.append(tp / ngt if ngt else 0)
        axes[0].bar(np.arange(len(zones)) + (k - (len(donnees) - 1) / 2) * larg, rappels, larg,
                    color=c, label=f"{nom} (seuil {s0:.2f})")
        ious = [m[1] for e in enregs for m in e["matches"] if m[0] >= s0]
        if ious:
            axes[1].hist(ious, bins=np.arange(0.5, 1.01, 0.05), alpha=0.6, color=c,
                         label=f"{nom} — IoU médian {np.median(ious):.3f}")
    axes[0].set_xticks(np.arange(len(zones)))
    axes[0].set_xticklabels(zones, rotation=20, ha="right", fontsize=8)
    axes[0].set_title("Rappel par zone (au seuil F1-optimal)"); axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3, axis="y"); axes[0].legend(fontsize=8)
    axes[1].set_title("Qualité des masques appariés (IoU des TP)")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", required=True, help="dossier contenant valid/ et test/")
    ap.add_argument("--modele", action="append", required=True,
                    help='"nom=poids.pth@resolution" (répétable)')
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion", action="append", default=[], help="classe_source=classe_cible")
    ap.add_argument("--titre", default=None)
    ap.add_argument("--plancher", type=float, default=0.05)
    a = ap.parse_args()

    modeles = {}
    for spec in a.modele:
        nom, reste = spec.split("=", 1)
        poids, resolution = reste.rsplit("@", 1)
        sidecar = os.path.join(os.path.dirname(poids), "best.json")
        noms = None
        if os.path.exists(sidecar):
            noms = json.load(open(sidecar, encoding="utf-8")).get("class_names")
        modeles[nom] = {"poids": poids, "resolution": int(resolution), "noms": noms}
    fusion = dict(f.split("=", 1) for f in a.fusion)

    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "appariements.json")
    if os.path.exists(cache):
        donnees = json.load(open(cache, encoding="utf-8"))
        print("cache reutilise :", cache)
    else:
        # classes par defaut = categories du COCO
        coco_valid = json.load(open(os.path.join(a.coco, "valid", "_annotations.coco.json"),
                                    encoding="utf-8"))
        cats_coco = [c["name"] for c in coco_valid["categories"]]
        for cfg in modeles.values():
            if cfg["noms"] is None:
                cfg["noms"] = cats_coco
        donnees = inferer(modeles, a.coco, fusion, a.plancher)
        json.dump(donnees, open(cache, "w", encoding="utf-8"))

    base_coco = os.path.basename(a.coco.rstrip("/\\"))
    titre = a.titre or f"Évaluation {base_coco} — IoU masque ≥ 0,5"
    planche_principale(donnees, titre, os.path.join(a.out, "courbes_seuils_pr.png"))
    classes = sorted({m[2] for dd in donnees.values() for e in dd["enregs"] for m in e["matches"]}
                     | {c for dd in donnees.values() for e in dd["enregs"] for c in e["gt_classes"]})
    if len(classes) > 1:
        planche_classes(donnees, classes, os.path.join(a.out, "f1_par_classe.png"))
    planche_zones(donnees, os.path.join(a.out, "zones_et_masques.png"))
    print("planches ->", a.out)


if __name__ == "__main__":
    main()
