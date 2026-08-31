"""Éval outillée standard des modèles RF-DETR — détection bbox ET segmentation.

Pour 1 à N modèles évalués sur le MÊME jeu COCO, produit :
  - metriques_eval.json : sortie CANONIQUE machine-readable (schéma metriques_eval/1)
    — seuils F1-max global + par classe, P/R/F1, AP@50 toutes-points, par zone,
    provenance (poids, résolution, dataset). C'est LA source des seuils du
    model_card (confidence_default + confidence_per_class) et du dashboard
    (tools/tableau_modeles.py) ;
  - courbes_seuils_pr.png : P/confiance, R/confiance, F1/confiance, courbe P-R
    (AP@0,5 toutes-points), courbes SUPERPOSÉES et point F1-max annoté ;
  - f1_par_classe.png (si >1 classe après fusion) ;
  - zones_et_masques.png (rappel par zone + histogramme IoU des TP, si champ
    `zone` dans le COCO) ;
  - appariements.json : appariements bruts en CACHE, avec empreinte de
    provenance `_meta` — relance à empreinte identique = re-rendu sans
    inférence ; empreinte différente = erreur ; cache legacy sans empreinte
    accepté seulement avec --adopter-cache.

Protocole (doctrine maison, cf. docs/rapport_test_adaf.html) : inférence au
plancher 0,05, appariement glouton par confiance décroissante, class-aware,
IoU >= 0,5 (MASQUE en segmentation, BBOX en détection), métriques par balayage
post-hoc du seuil (grille pas 0,005) — jamais de seuil fixe. Convention :
P = 1,0 quand TP+FP = 0. Autocontrôle de chargement : rappel au plancher
< 0,30 (global ou --classe-controle) = abandon SANS écrire le cache.

La tâche (détection/segmentation) est auto-détectée depuis le checkpoint
(RFDETR.from_checkpoint) ; --tache la force et toute contradiction est une
ERREUR (un checkpoint det chargé en seg ne donne plus silencieusement 0
prédiction). Tous les modèles d'un même run doivent être de la même tâche.

Usage (venv_adaf OBLIGATOIRE — torch/rfdetr/matplotlib) :
  D:\\veille_irlande\\venv_adaf\\Scripts\\python.exe tools\\courbes_eval.py
      --coco <dossier avec valid/ et test/ | dossier d'UN split>
      --modele "nom=poids.pth@resolution" [--modele ...]
      --out <dossier de sortie>
      [--tache detection|segmentation]
      [--fusion talus=talus_fosse --fusion fosse=talus_fosse]
      [--titre "..."] [--plancher 0.05]
      [--classe-controle <classe>] [--sans-autocontrole] [--adopter-cache]
Les classes de chaque modèle sont lues dans le sidecar best.json voisin du
.pth s'il existe, sinon supposées identiques aux catégories du COCO.
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np

PALETTE = ["#888888", "#c1272d", "#1f77b4", "#2c6e49", "#7c4a1e"]
SCHEMA_METRIQUES = "metriques_eval/1"
SCHEMA_CACHE = "appariements/2"
RAPPEL_MIN = 0.30


def grille(plancher):
    """Grille de balayage unique du seuil de confiance (pas 0,005)."""
    return np.round(np.arange(plancher, 0.951, 0.005), 3)


def chemin_norm(p):
    """Chemin absolu à slashs avant (piège \\v -> tabulation verticale)."""
    return os.path.abspath(p).replace("\\", "/")


def iou_bbox(a, b):
    """IoU de deux boîtes xyxy."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if not inter:
        return 0.0
    aire_a = (a[2] - a[0]) * (a[3] - a[1])
    aire_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aire_a + aire_b - inter)


def inferer(modeles, splits, fusion, plancher, tache_forcee):
    from PIL import Image
    from pycocotools.coco import COCO
    from rfdetr import RFDETR

    donnees, tache = {}, tache_forcee
    for nom, cfg in modeles.items():
        try:
            modele = RFDETR.from_checkpoint(cfg["poids"], resolution=cfg["resolution"])
        except ValueError as e:
            sys.exit(f"ERREUR chargement {nom} : {e}\n(vieux checkpoint sans model_name : "
                     "renommer le fichier au nom canonique rfdetr, cf. doc from_checkpoint)")
        seg = bool(getattr(modele.model_config, "segmentation_head", False))
        tache_modele = "segmentation" if seg else "detection"
        if tache is None:
            tache = tache_modele
        if tache_modele != tache:
            sys.exit(f"ERREUR : {nom} est un modèle {tache_modele}, le run est {tache} "
                     "(--tache forcée ou autre modèle) — on ne mélange pas les tâches.")
        modele.optimize_for_inference()
        decal = None
        enregs = []
        for etiquette, dossier in splits:
            coco = COCO(os.path.join(dossier, "_annotations.coco.json"))
            cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
            for img_id in coco.getImgIds():
                info = coco.loadImgs(img_id)[0]
                im = Image.open(os.path.join(dossier, info["file_name"])).convert("RGB")
                d = modele.predict(im, threshold=plancher)
                n_pred = len(d)
                if seg and n_pred and d.mask is None:
                    sys.exit(f"ERREUR : {nom} (segmentation) prédit sans masques.")
                if decal is None and n_pred:
                    # vieux exports : background en colonne 0 -> ids 1..N
                    decal = 1 if (int(d.class_id.min()) >= 1
                                  and int(d.class_id.max()) >= len(cfg["noms"])) else 0
                gts = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
                objs, cls_gt = [], []
                for a in gts:
                    c = fusion.get(cats[a["category_id"]], cats[a["category_id"]])
                    if seg:
                        objs.append(coco.annToMask(a).astype(bool))
                    else:
                        x, y, w, h = a["bbox"]
                        objs.append((x, y, x + w, y + h))
                    cls_gt.append(c)
                pris = [False] * len(objs)
                matches, fps = [], []
                for i in (np.argsort(-d.confidence) if n_pred else []):
                    idx = int(d.class_id[i]) - (decal or 0)
                    if not (0 <= idx < len(cfg["noms"])):
                        continue
                    c = fusion.get(cfg["noms"][idx], cfg["noms"][idx])
                    obj = d.mask[i].astype(bool) if seg else tuple(map(float, d.xyxy[i]))
                    meilleur, mi = 0.0, -1
                    for j, g in enumerate(objs):
                        if pris[j] or cls_gt[j] != c:
                            continue
                        if seg:
                            inter = np.logical_and(obj, g).sum()
                            iou = inter / np.logical_or(obj, g).sum() if inter else 0.0
                        else:
                            iou = iou_bbox(obj, g)
                        if iou > meilleur:
                            meilleur, mi = iou, j
                    if meilleur >= 0.5:
                        pris[mi] = True
                        matches.append([float(d.confidence[i]), float(meilleur), c])
                    else:
                        fps.append([float(d.confidence[i]), c])
                enregs.append({"split": etiquette, "zone": info.get("zone", ""),
                               "n_gt": len(objs), "gt_classes": cls_gt,
                               "matches": matches, "fps": fps})
        donnees[nom] = {"decal": decal, "enregs": enregs}
        print(f"{nom} : {sum(len(e['matches']) for e in enregs)} TP potentiels, "
              f"offset {decal}, tâche {tache}")
        del modele
        import gc
        import torch
        gc.collect(); torch.cuda.empty_cache()
    return donnees, tache


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


def ap50(enregs, classe=None):
    """AP@0,5 toutes-points par rang de confiance. Retourne (ap, rappels, précisions)."""
    scores = []
    for e in enregs:
        scores += [(m[0], 1) for m in e["matches"] if classe is None or m[2] == classe]
        scores += [(f[0], 0) for f in e["fps"] if classe is None or f[1] == classe]
    ngt = sum((sum(1 for c in e["gt_classes"] if c == classe) if classe else e["n_gt"])
              for e in enregs)
    if not scores or not ngt:
        return 0.0, np.array([0.0]), np.array([1.0])
    scores.sort(key=lambda t: -t[0])
    tp = np.cumsum([s[1] for s in scores])
    fp = np.cumsum([1 - s[1] for s in scores])
    rr = tp / ngt
    pp = tp / (tp + fp)
    for j in range(len(pp) - 2, -1, -1):  # enveloppe de précision monotone
        pp[j] = max(pp[j], pp[j + 1])
    ap = float(np.sum((rr - np.concatenate(([0.0], rr[:-1]))) * pp))
    return ap, rr, pp


def bloc_metriques(enregs, plancher, classe=None):
    """Métriques standard d'un modèle (ou d'une classe) : point F1-max + AP50."""
    seuils = grille(plancher)
    vals = [prf(enregs, s, classe) for s in seuils]
    i = int(np.argmax([v[2] for v in vals]))
    s0 = float(seuils[i])
    p0, r0, f0 = vals[i]
    ngt = sum((sum(1 for c in e["gt_classes"] if c == classe) if classe else e["n_gt"])
              for e in enregs)
    ious = [m[1] for e in enregs for m in e["matches"]
            if m[0] >= s0 and (classe is None or m[2] == classe)]
    return {"seuil_f1max": s0, "F1": round(f0, 4), "P": round(p0, 4), "R": round(r0, 4),
            "AP50": round(ap50(enregs, classe)[0], 4), "n_gt": int(ngt),
            "iou_median": round(float(np.median(ious)), 4) if ious else None}


def classes_presentes(donnees):
    return sorted({m[2] for dd in donnees.values() for e in dd["enregs"] for m in e["matches"]}
                  | {f[1] for dd in donnees.values() for e in dd["enregs"] for f in e["fps"]}
                  | {c for dd in donnees.values() for e in dd["enregs"] for c in e["gt_classes"]})


def resumer(donnees, meta_modeles, tache, dataset, fusion, plancher, provenance_cache):
    """Construit le dict metriques_eval/1 — LA sortie canonique de l'outil."""
    classes = classes_presentes(donnees)
    resume = {
        "schema": SCHEMA_METRIQUES,
        "genere_le": datetime.now().isoformat(timespec="seconds"),
        "outil": "tools/courbes_eval.py",
        "tache": tache,
        "iou": {"type": "masque" if tache == "segmentation" else "bbox", "seuil": 0.5},
        "appariement": "glouton conf decroissante, class-aware",
        "plancher": plancher,
        "grille": {"min": plancher, "max": 0.95, "pas": 0.005},
        "p_sans_prediction": 1.0,
        "dataset": dataset,
        "fusion": fusion,
        "provenance_cache": provenance_cache,
        "modeles": {},
    }
    for nom, dd in donnees.items():
        enregs = dd["enregs"]
        bloc_global = bloc_metriques(enregs, plancher)
        s0 = bloc_global["seuil_f1max"]
        zones = sorted({e.get("zone", "") for e in enregs if e.get("zone")})
        par_zone = {}
        for z in zones:
            sous = [e for e in enregs if e.get("zone") == z]
            tp = sum(sum(1 for m in e["matches"] if m[0] >= s0) for e in sous)
            fp = sum(sum(1 for f in e["fps"] if f[0] >= s0) for e in sous)
            ngt = sum(e["n_gt"] for e in sous)
            par_zone[z] = {"P": round(tp / (tp + fp), 4) if tp + fp else 1.0,
                           "R": round(tp / ngt, 4) if ngt else 0.0, "n_gt": int(ngt)}
        info = meta_modeles.get(nom, {})
        resume["modeles"][nom] = {
            "poids": info.get("poids"),
            "resolution": info.get("resolution"),
            "class_offset": dd.get("decal"),
            "global": bloc_global,
            "par_classe": {cl: bloc_metriques(enregs, plancher, cl) for cl in classes},
        }
        if par_zone:
            resume["modeles"][nom]["par_zone"] = par_zone
    return resume


# ---------------------------------------------------------------- planches ---

def planche_principale(donnees, titre, plancher, sortie):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seuils = grille(plancher)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(titre, fontsize=12)
    for k, (nom, dd) in enumerate(donnees.items()):
        enregs, c = dd["enregs"], PALETTE[k % len(PALETTE)]
        vals = [prf(enregs, s) for s in seuils]
        P, R, F = (np.array([v[j] for v in vals]) for j in range(3))
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
        ap, rr, pp = ap50(enregs)
        axes[1][1].plot(rr, pp, color=c, label=f"{nom} — AP@0.5 {ap:.3f}")
        axes[1][1].scatter([r0], [p0], color=c, zorder=5, marker="*", s=120)
        axes[1][1].annotate(f"F1max: P={p0:.2f}, R={r0:.2f}", (r0, p0), textcoords="offset points",
                            xytext=(8, 6 + 12 * k), fontsize=8, color=c)
    for ax, t in ((axes[0][0], "Précision vs confiance"), (axes[0][1], "Rappel vs confiance"),
                  (axes[1][0], "F1 vs confiance"), (axes[1][1], "Courbe Précision-Rappel (★ = F1-max)")):
        ax.set_title(t); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)


def planche_classes(donnees, classes, plancher, sortie):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seuils = grille(plancher)
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


def planche_zones(donnees, tache, plancher, sortie):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seuils = grille(plancher)
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
    qual = "masques" if tache == "segmentation" else "boîtes"
    axes[1].set_title(f"Qualité des {qual} appariés (IoU des TP)")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    return True


# --------------------------------------------------------- cache/empreinte ---

def construire_meta(plancher, coco, fusion, modeles, tache=None):
    meta = {"schema": SCHEMA_CACHE, "tache": tache, "plancher": plancher,
            "coco": chemin_norm(coco), "fusion": fusion, "modeles": {}}
    for nom, cfg in modeles.items():
        taille = os.path.getsize(cfg["poids"]) if os.path.exists(cfg["poids"]) else None
        meta["modeles"][nom] = {"poids": chemin_norm(cfg["poids"]),
                                "resolution": cfg["resolution"], "taille_octets": taille}
    return meta


def meta_divergence(attendu, cache):
    """Première divergence entre l'empreinte attendue (CLI) et celle du cache, ou None."""
    if cache.get("plancher") != attendu["plancher"]:
        return f"plancher {cache.get('plancher')} != {attendu['plancher']}"
    if cache.get("coco", "").casefold() != attendu["coco"].casefold():
        return f"coco {cache.get('coco')} != {attendu['coco']}"
    if cache.get("fusion") != attendu["fusion"]:
        return f"fusion {cache.get('fusion')} != {attendu['fusion']}"
    if sorted(cache.get("modeles", {})) != sorted(attendu["modeles"]):
        return f"modèles {sorted(cache.get('modeles', {}))} != {sorted(attendu['modeles'])}"
    for nom, att in attendu["modeles"].items():
        cac = cache["modeles"][nom]
        if cac.get("poids", "").casefold() != att["poids"].casefold():
            return f"{nom} : poids {cac.get('poids')} != {att['poids']}"
        if cac.get("resolution") != att["resolution"]:
            return f"{nom} : résolution {cac.get('resolution')} != {att['resolution']}"
        if (att["taille_octets"] is not None and cac.get("taille_octets") is not None
                and cac["taille_octets"] != att["taille_octets"]):
            return f"{nom} : taille des poids {cac['taille_octets']} != {att['taille_octets']}"
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", required=True,
                    help="dossier avec valid/ et test/, OU dossier d'un seul split")
    ap.add_argument("--modele", action="append", required=True,
                    help='"nom=poids.pth@resolution" (répétable)')
    ap.add_argument("--out", required=True)
    ap.add_argument("--tache", choices=["detection", "segmentation"], default=None,
                    help="forcer la tâche (défaut : auto depuis le checkpoint)")
    ap.add_argument("--fusion", action="append", default=[], help="classe_source=classe_cible")
    ap.add_argument("--titre", default=None)
    ap.add_argument("--plancher", type=float, default=0.05)
    ap.add_argument("--classe-controle", default=None,
                    help="autocontrôle du rappel plancher sur cette classe (défaut : global)")
    ap.add_argument("--sans-autocontrole", action="store_true")
    ap.add_argument("--adopter-cache", action="store_true",
                    help="accepter un appariements.json legacy sans empreinte _meta")
    a = ap.parse_args()

    modeles = {}
    for spec in a.modele:
        nom, reste = spec.split("=", 1)
        if nom == "_meta":
            sys.exit("ERREUR : nom de modèle '_meta' interdit (réservé au cache).")
        poids, resolution = reste.rsplit("@", 1)
        sidecar = os.path.join(os.path.dirname(poids), "best.json")
        noms = None
        if os.path.exists(sidecar):
            noms = json.load(open(sidecar, encoding="utf-8")).get("class_names")
        modeles[nom] = {"poids": poids, "resolution": int(resolution), "noms": noms}
    fusion = dict(f.split("=", 1) for f in a.fusion)

    if os.path.exists(os.path.join(a.coco, "_annotations.coco.json")):
        splits = [(os.path.basename(a.coco.rstrip("/\\")), a.coco)]
    else:
        splits = [(s, os.path.join(a.coco, s)) for s in ("valid", "test")
                  if os.path.exists(os.path.join(a.coco, s, "_annotations.coco.json"))]
    if not splits:
        sys.exit(f"ERREUR : aucun _annotations.coco.json trouvé sous {a.coco} "
                 "(ni directement, ni dans valid/ ou test/).")

    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "appariements.json")
    meta_attendu = construire_meta(a.plancher, a.coco, fusion, modeles)
    provenance_cache = "calculee"

    if os.path.exists(cache):
        donnees = json.load(open(cache, encoding="utf-8"))
        meta_cache = donnees.pop("_meta", None)
        if meta_cache is None:
            if not a.adopter_cache:
                sys.exit(f"ERREUR : cache sans empreinte de provenance : {cache}\n"
                         "Cache legacy — relancer avec --adopter-cache pour l'accepter "
                         "(rétrofit), ou le supprimer pour recalculer.")
            provenance_cache = "adoptee_sans_empreinte"
            tache = a.tache or "segmentation"  # les caches legacy viennent du chemin seg
            print(f"AVERTISSEMENT : cache adopté sans empreinte ({cache}), "
                  f"tâche supposée {tache}")
            if sorted(donnees) != sorted(modeles):
                sys.exit(f"ERREUR : modèles du cache {sorted(donnees)} != CLI "
                         f"{sorted(modeles)} — nommer les --modele comme dans le cache.")
        else:
            div = meta_divergence(meta_attendu, meta_cache)
            if div:
                sys.exit(f"ERREUR : le cache {cache} vient d'une autre provenance ({div}).\n"
                         "Supprimer le cache ou changer --out.")
            tache = meta_cache["tache"]
            if a.tache and a.tache != tache:
                sys.exit(f"ERREUR : cache en tâche {tache}, --tache {a.tache} demandé.")
        print("cache reutilise :", cache)
    else:
        # classes par défaut = catégories du premier split
        coco0 = json.load(open(os.path.join(splits[0][1], "_annotations.coco.json"),
                               encoding="utf-8"))
        cats_coco = [c["name"] for c in coco0["categories"]]
        for cfg in modeles.values():
            if cfg["noms"] is None:
                cfg["noms"] = cats_coco
        donnees, tache = inferer(modeles, splits, fusion, a.plancher, a.tache)
        if not a.sans_autocontrole:
            for nom, dd in donnees.items():
                _, r, _ = prf(dd["enregs"], a.plancher, a.classe_controle)
                if r < RAPPEL_MIN:
                    cible = a.classe_controle or "global"
                    sys.exit(f"ERREUR autocontrôle : rappel plancher de {nom} ({cible}) = "
                             f"{r:.3f} < {RAPPEL_MIN} — chargement/offset suspect. Cache NON "
                             "écrit. (--sans-autocontrole pour un modèle légitimement faible)")
        meta_attendu["tache"] = tache
        json.dump({"_meta": meta_attendu, **donnees}, open(cache, "w", encoding="utf-8"))

    # ------- sortie canonique metriques_eval.json (validée par relecture) -------
    n_img = len(next(iter(donnees.values()))["enregs"])
    n_gt = sum(e["n_gt"] for e in next(iter(donnees.values()))["enregs"])
    dataset = {"chemin": chemin_norm(a.coco), "splits": [s for s, _ in splits],
               "n_images": n_img, "n_gt": int(n_gt)}
    resume = resumer(donnees, meta_attendu["modeles"], tache, dataset, fusion,
                     a.plancher, provenance_cache)
    chemin_metriques = os.path.join(a.out, "metriques_eval.json")
    with open(chemin_metriques, "w", encoding="utf-8") as f:
        json.dump(resume, f, ensure_ascii=False, indent=1)
    relu = json.load(open(chemin_metriques, encoding="utf-8"))
    assert relu["schema"] == SCHEMA_METRIQUES and relu["modeles"], "relecture invalide"
    print("metriques ->", chemin_metriques)
    for nom, m in resume["modeles"].items():
        g = m["global"]
        print(f"  {nom} : F1 {g['F1']} @ {g['seuil_f1max']} (P {g['P']} / R {g['R']}, "
              f"AP50 {g['AP50']})")
        for cl, b in m["par_classe"].items():
            print(f"    {cl} : F1 {b['F1']} @ {b['seuil_f1max']} (n_gt {b['n_gt']})")

    # ------------------------------------------------------------- planches ---
    base_coco = os.path.basename(a.coco.rstrip("/\\"))
    iou_lib = "masque" if tache == "segmentation" else "bbox"
    titre = a.titre or f"Évaluation {base_coco} — IoU {iou_lib} ≥ 0,5"
    planche_principale(donnees, titre, a.plancher, os.path.join(a.out, "courbes_seuils_pr.png"))
    classes = classes_presentes(donnees)
    if len(classes) > 1:
        planche_classes(donnees, classes, a.plancher, os.path.join(a.out, "f1_par_classe.png"))
    planche_zones(donnees, tache, a.plancher, os.path.join(a.out, "zones_et_masques.png"))
    print("planches ->", a.out)


if __name__ == "__main__":
    main()
