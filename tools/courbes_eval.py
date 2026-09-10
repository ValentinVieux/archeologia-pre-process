"""Éval outillée standard des modèles RF-DETR — détection bbox ET segmentation.

Pour 1 à N modèles évalués sur le MÊME jeu COCO, produit :
  - metriques_eval.json : sortie CANONIQUE machine-readable (schéma metriques_eval/1)
    — seuils F1-max global + par classe, P/R/F1, AP@50 toutes-points, par zone,
    par zone × classe, provenance (poids, résolution, dataset). C'est LA source
    des seuils du model_card (confidence_default + confidence_per_class) et du
    dashboard (tools/tableau_modeles.py). Par modèle : `global` et
    `par_classe[c]` (seuil_f1max, F1, P, R, AP50, n_gt, iou_median, et depuis le
    2026-09-09 le bloc `etude_seuil` : F2-max, plateaux F1 ≥ 98 %/95 %, R_max,
    FP par image, seuil_propose = max(F2-max, bas du plateau 95 %), précision
    marginale, tableau au pas 0,05 — de quoi CHOISIR le seuil de production sous
    le F1-max, par lecture, cf. CLAUDE.md « Choix du seuil de production » ; et
    `bandes` (tp/fp par tranche de 0,05 = table de calibrage) + `fiabilite_proposee`
    (catégories douteux/possible/probable/quasi_certain définies par la part de
    vrais objets, coupures par classe — proposition A+D du 2026-09-09) ;
    `par_zone[z]` (P, R, n_gt au seuil F1-max global) ; `par_zone_classe[z][c]`
    (2026-09-03, additif : n_gt, tp, fp, R, P au seuil F1-max GLOBAL,
    R_seuil_classe + fp_seuil_classe au seuil F1-max de LA classe, R_max =
    rappel avec TOUS les matches du cache ; R/P null quand n_gt / tp+fp = 0 —
    toutes les classes de par_classe figurent dans chaque zone). Une éval
    antérieure au 2026-09-03 se complète sans GPU par
    tools/completer_metriques_eval.py ;
  - courbes_seuils_pr.png : P/confiance, R/confiance, F1/confiance, courbe P-R
    (AP@0,5 toutes-points), courbes SUPERPOSÉES et point F1-max annoté ;
  - f1_par_classe.png (si >1 classe après fusion) ;
  - zones_et_masques.png (rappel par zone + histogramme IoU des TP, si champ
    `zone` dans le COCO) ;
  - appariements.json : appariements bruts en CACHE, avec empreinte de
    provenance `_meta` — relance à empreinte identique = re-rendu sans
    inférence ; empreinte différente = erreur ; cache legacy sans empreinte
    accepté seulement avec --adopter-cache. Cache par MODÈLE (2026-09-02) :
    `--reprendre-de <dossier_eval>` (répétable) adopte les appariements des
    modèles déjà évalués ailleurs à provenance identique (run + par-modèle) ;
    seuls les modèles encore manquants passent à l'inférence — ajouter un
    concurrent à une comparaison ne réinfère plus les anciens.

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

# Critère de vérité d'une prédiction (2026-09-09, structures LINÉAIRES) :
# - "iou" (défaut, doctrine) : appariement 1-1 glouton, IoU masque/boîte >= 0,5 ;
# - "couverture" (segmentation seulement) : une prédiction est VRAIE si >= 50 % de sa
#   surface tombe sur l'union des masques annotés de sa classe ; une annotation est
#   RETROUVÉE dès que l'union des prédictions de sa classe (par confiance décroissante)
#   couvre >= 50 % de sa surface — le score d'appariement est la confiance à laquelle
#   c'est atteint. Tolère fragmentation et décalage latéral (tampon de 7 m des
#   annotations), que l'IoU 1-1 compte à tort comme faux positifs sur du linéaire.
#   Les enregistrements portent alors `tps_pred` (prédictions vraies, côté précision)
#   en plus de `matches` (annotations retrouvées, côté rappel).
CRITERES = ("iou", "couverture")
SEUIL_COUVERTURE = 0.5


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


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _bbox_chevauche(a, b):
    return a is not None and b is not None and a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]


def apparier_couverture(preds, objs, cls_gt):
    """Critère « couverture » (segmentation) — fonction pure.

    `preds` : [(conf, classe, masque bool)] par confiance DÉCROISSANTE ; `objs` :
    masques annotés ; `cls_gt` : leurs classes. Retourne (matches, fps, tps_pred) :
    matches = [[conf d'atteinte de 50 % de couverture, couverture finale, classe]] par
    annotation retrouvée ; tps_pred = [[conf, couverture, classe]] par prédiction vraie ;
    fps = [[conf, classe]] par prédiction fausse.
    """
    union, bb_gt = {}, [_bbox(g) for g in objs]
    for g, c in zip(objs, cls_gt):
        union[c] = g if c not in union else (union[c] | g)
    tps_pred, fps, bb_pred = [], [], []
    for conf, c, mask in preds:
        aire = int(mask.sum())
        cov = float((mask & union[c]).sum()) / aire if (c in union and aire) else 0.0
        if cov >= SEUIL_COUVERTURE:
            tps_pred.append([float(conf), round(cov, 4), c])
        else:
            fps.append([float(conf), c])
        bb_pred.append(_bbox(mask))
    matches = []
    for j, (g, c) in enumerate(zip(objs, cls_gt)):
        aire_g = int(g.sum())
        if not aire_g:
            continue
        acc, trouve = np.zeros_like(g), None
        for (conf, cp, mask), bb in zip(preds, bb_pred):
            if cp != c or not _bbox_chevauche(bb, bb_gt[j]):
                continue
            acc |= (mask & g)
            if acc.sum() / aire_g >= SEUIL_COUVERTURE:
                trouve = float(conf)
                break
        if trouve is not None:
            matches.append([trouve, round(float(acc.sum()) / aire_g, 4), c])
    return matches, fps, tps_pred


def inferer(modeles, splits, fusion, plancher, tache_forcee, critere="iou"):
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
        if critere == "couverture" and not seg:
            sys.exit(f"ERREUR : critère couverture = segmentation seulement ({nom} est en détection).")
        modele.optimize_for_inference()
        # class_offset du sidecar best.json = vérité (leçon 2026-09-08 : run_rf_detr_1 prédit
        # parfois la catégorie 0 « entites » héritée de Roboflow -> l'heuristique sur la
        # première image tombait à 0 et perdait la classe four) ; heuristique sinon.
        decal = cfg.get("decal_sidecar")
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
                if critere == "couverture":
                    preds = []
                    for i in (np.argsort(-d.confidence) if n_pred else []):
                        idx = int(d.class_id[i]) - (decal or 0)
                        if not (0 <= idx < len(cfg["noms"])):
                            continue
                        c = fusion.get(cfg["noms"][idx], cfg["noms"][idx])
                        preds.append((float(d.confidence[i]), c, d.mask[i].astype(bool)))
                    matches, fps, tps_pred = apparier_couverture(preds, objs, cls_gt)
                    enregs.append({"split": etiquette, "zone": info.get("zone", ""),
                                   "n_gt": len(objs), "gt_classes": cls_gt,
                                   "matches": matches, "fps": fps, "tps_pred": tps_pred})
                    continue
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


def vrais_pred(e):
    """Prédictions VRAIES d'un enregistrement (côté précision) : `tps_pred` au critère
    couverture, sinon les appariements 1-1 (`matches`, critère IoU)."""
    return e["tps_pred"] if "tps_pred" in e else e["matches"]


def comptes(enregs, s, classe=None):
    """(tp_precision, fp, n_gt, tp_rappel) au seuil s — recomptage brut du cache.
    Critère IoU : tp_precision == tp_rappel (appariement 1-1)."""
    tp_r = sum(sum(1 for m in e["matches"] if m[0] >= s and (classe is None or m[2] == classe))
               for e in enregs)
    tp_p = sum(sum(1 for t in vrais_pred(e) if t[0] >= s and (classe is None or t[2] == classe))
               for e in enregs)
    fp = sum(sum(1 for f in e["fps"] if f[0] >= s and (classe is None or f[1] == classe))
             for e in enregs)
    ngt = sum((sum(1 for c in e["gt_classes"] if c == classe) if classe else e["n_gt"])
              for e in enregs)
    return tp_p, fp, ngt, tp_r


def prf(enregs, s, classe=None):
    tp_p, fp, ngt, tp_r = comptes(enregs, s, classe)
    p = tp_p / (tp_p + fp) if tp_p + fp else 1.0
    r = tp_r / ngt if ngt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# Choix du seuil de production (règle utilisateur 2026-09-09) : le F1-max pèse un oubli
# comme un faux positif ; en prospection l'oubli ne se rattrape pas, le faux positif se
# rejette en quelques secondes sur le RVT. Le seuil déployé se choisit donc SOUS le
# F1-max, en ÉTUDIANT la courbe — jamais par formule aveugle. L'outil publie de quoi
# étudier : F2-max, plateaux F1 (≥ 98 % / 95 % du max), rappel max au plancher, FP par
# image, précision marginale des détections ajoutées, tableau au pas 0,05 ; et un
# `seuil_propose` = max(F2-max, bas du plateau 95 %) qui n'est qu'un POINT DE DÉPART.
PLATEAUX_F1 = (0.98, 0.95)
PAS_TABLEAU = 0.05

# Fiabilité affichée dans QGIS (décision utilisateur 2026-09-09, propositions A + D) : les
# catégories « douteux / possible / probable / très probable » sont définies par la part de
# VRAIS objets mesurée dans la tranche de score (précision locale), pas par le score. Le
# vocabulaire est donc stable entre modèles ; ce sont les COUPURES de score qui bougent,
# par classe. Niveaux garantis (part de vrais sur le banc) et effectif minimal pour
# publier une mesure.
NIVEAUX_FIABILITE = {"possible": 0.35, "probable": 0.60, "quasi_certain": 0.85}
CATEGORIES_FIABILITE = ("douteux", "possible", "probable", "quasi_certain")
N_MIN_MESURE = 30
PAS_BANDE = 0.01      # table de calibrage fine (un seuil déployé à 0,29 ou 0,245 tombe juste)
PAS_COUPURE = 0.05    # les coupures se décident sur des fenêtres de 0,05 (lissées deux à deux)


def f2(p, r):
    return 5 * p * r / (4 * p + r) if p + r else 0.0


def bandes_confiance(enregs, classe=None, pas=PAS_BANDE):
    """Comptes (tp, fp) par tranche de score de `pas` (0,01) sur [0,05 ; 1] — la table
    de calibrage d'un modèle (ou d'une classe). Dernière tranche inclusive de 1.
    tp = prédictions VRAIES (côté précision : `tps_pred` au critère couverture)."""
    tp_conf = [t[0] for e in enregs for t in vrais_pred(e) if classe is None or t[2] == classe]
    fp_conf = [f[0] for e in enregs for f in e["fps"] if classe is None or f[1] == classe]
    out = []
    n = int(round(1.0 / pas))
    for k in range(int(round(0.05 / pas)), n):
        lo, hi = round(pas * k, 2), round(pas * (k + 1), 2)
        if k == n - 1:
            dans = lambda c: c >= lo - 1e-9  # noqa: E731
        else:
            dans = lambda c: lo - 1e-9 <= c < hi - 1e-9  # noqa: E731
        out.append({"lo": lo, "hi": hi, "tp": sum(1 for c in tp_conf if dans(c)),
                    "fp": sum(1 for c in fp_conf if dans(c))})
    return out


def agreger_bandes(bandes, pas=PAS_COUPURE, debut=None):
    """Regroupe des tranches fines en fenêtres de `pas` alignées sur la grille
    (lo multiple de `pas`) — des tranches déjà à ce pas restent inchangées.
    `debut` (le seuil déployé, rarement sur la grille) : les tranches sous `debut`
    sont ignorées et la première fenêtre est PARTIELLE [debut ; prochaine ligne de
    grille[ — une coupure peut ainsi tomber AU seuil (les détections juste au-dessus
    du seuil ne sont pas condamnées à « douteux » par la grille)."""
    fen = {}
    for b in bandes:
        if debut is not None and b["lo"] < debut - 1e-9:
            continue
        lo = round(np.floor(b["lo"] / pas + 1e-9) * pas, 2)
        hi = round(lo + pas, 2)  # prochaine ligne de grille
        if debut is not None and lo < debut - 1e-9:
            lo = float(debut)     # fenêtre partielle [debut ; ligne de grille[
        f = fen.setdefault(lo, {"lo": lo, "hi": hi, "tp": 0, "fp": 0})
        f["tp"] += b["tp"]; f["fp"] += b["fp"]
        f["hi"] = max(f["hi"], b["hi"])
    return [fen[k] for k in sorted(fen)]


def coupures_fiabilite(bandes, seuil, niveaux=NIVEAUX_FIABILITE):
    """Coupures de score des catégories de fiabilité au-dessus de `seuil` :
    première fenêtre de 0,05 (la première partielle, dès le seuil) dont la précision
    LISSÉE sur deux fenêtres consécutives atteint le niveau ; coupures monotones ;
    None si jamais atteint."""
    bandes = agreger_bandes(bandes, debut=seuil)
    ub = [b for b in bandes if b["lo"] >= seuil - 1e-9 and b["tp"] + b["fp"] > 0]
    lisse = []
    for i, b in enumerate(ub):
        n = ub[i + 1] if i + 1 < len(ub) else {"tp": 0, "fp": 0}
        tp, fp = b["tp"] + n["tp"], b["fp"] + n["fp"]
        lisse.append((b["lo"], tp / (tp + fp) if tp + fp else None))
    out, dernier = {}, seuil
    for cle in CATEGORIES_FIABILITE[1:]:
        c = next((lo for lo, p in lisse
                  if lo >= dernier - 1e-9 and p is not None and p >= niveaux[cle]), None)
        out[cle] = c
        if c is not None:
            dernier = c
    return out


def fiabilite_par_classe(bandes, seuil, coupures, niveaux=NIVEAUX_FIABILITE, n_min=N_MIN_MESURE):
    """Catégories de fiabilité d'une classe : [{categorie, seuil, garanti, mesure, n}]
    du plus douteux au plus sûr. `mesure` = part de vrais objets dans les tranches
    de la catégorie. C'est le bloc `thresholds.fiabilite.par_classe[c]` du
    model_card du plugin.

    Garde-fous (petits corpus) : une catégorie vide (coupure égale au seuil) est
    omise ; une catégorie de moins de `n_min` détections, ou dont la part mesurée
    n'atteint pas son niveau garanti, perd sa coupure et FUSIONNE dans la catégorie
    du dessous (l'étiquette sous-estime, jamais l'inverse) — répété jusqu'à
    stabilité. La catégorie basse (« douteux », rien de garanti) peut rester sous
    `n_min` : sa mesure est alors None. Les effectifs se comptent sur les tranches
    FINES (0,01) dont le bas est >= début de catégorie : un seuil à 0,29 compte dès
    0,29, un seuil à 0,245 dès 0,25.
    """
    bornes = [("douteux", float(seuil))] + [
        (cle, float(coupures[cle])) for cle in CATEGORIES_FIABILITE[1:]
        if coupures.get(cle) is not None]
    bornes = [(c, s) for i, (c, s) in enumerate(bornes)
              if (bornes[i + 1][1] if i + 1 < len(bornes) else 1.01) - s >= 1e-9]

    def construire(bornes):
        cats = []
        for i, (cle, debut) in enumerate(bornes):
            fin = bornes[i + 1][1] if i + 1 < len(bornes) else 1.01
            tp = sum(b["tp"] for b in bandes if debut - 1e-9 <= b["lo"] < fin - 1e-9)
            fp = sum(b["fp"] for b in bandes if debut - 1e-9 <= b["lo"] < fin - 1e-9)
            n = tp + fp
            cats.append({"categorie": cle, "seuil": debut, "garanti": float(niveaux.get(cle, 0.0)),
                         "mesure": round(tp / n, 3) if n >= n_min else None, "n": n})
        return cats

    while True:
        cats = construire(bornes)
        fusion = next((i for i in range(len(cats) - 1, 0, -1)
                       if cats[i]["n"] < n_min
                       or (cats[i]["mesure"] is not None and cats[i]["mesure"] < cats[i]["garanti"])),
                      None)
        if fusion is None:
            break
        bornes = bornes[:fusion] + bornes[fusion + 1:]
    # catégorie basse sans AUCUNE détection au banc (sous la finesse des tranches, ex.
    # seuil 0,245 et coupure 0,25) : la suivante démarre au seuil, pas de catégorie vide
    if len(cats) > 1 and cats[0]["n"] == 0:
        cats[1]["seuil"] = cats[0]["seuil"]
        cats = cats[1:]
    return cats


def etude_seuil(enregs, plancher, classe=None):
    """Indicateurs du choix du seuil (fonction pure, sans GPU) — cf. commentaire ci-dessus.

    Toutes les valeurs découlent des comptes (tp, fp) par seuil de la grille ; arrondi
    4 décimales. `plateau_f1_95`/`_98` = [min, max] des seuils où F1 ≥ 95 % / 98 % du
    F1-max. `precision_marginale` = part de vrais objets parmi les détections AJOUTÉES
    en descendant du F1-max au seuil proposé (null si rien n'est ajouté). `tableau` =
    une ligne par seuil multiple de 0,05 (seuil, P, R, F1, F2, fp_img).
    """
    seuils = grille(plancher)
    n_img = len(enregs)
    cpt = [comptes(enregs, s, classe) for s in seuils]
    ngt = cpt[0][2] if cpt else 0

    P, R, F1, F2 = [], [], [], []
    for tp, fp, _, tp_r in cpt:
        p, r = (tp / (tp + fp) if tp + fp else 1.0), (tp_r / ngt if ngt else 0.0)
        P.append(p); R.append(r)
        F1.append(2 * p * r / (p + r) if p + r else 0.0)
        F2.append(f2(p, r))
    i1 = int(np.argmax(F1))
    i2 = int(np.argmax(F2))
    plateaux = {}
    for frac in PLATEAUX_F1:
        ok = [i for i, f in enumerate(F1) if f >= frac * F1[i1]]
        plateaux[frac] = (int(ok[0]), int(ok[-1]))
    lo95 = plateaux[0.95][0]
    ip = max(i2, lo95)  # F2-max, remonté au bas du plateau 95 % s'il est dessous

    def fp_img(i):
        return round(cpt[i][1] / n_img, 4) if n_img else None

    tp_add = cpt[ip][0] - cpt[i1][0]
    fp_add = cpt[ip][1] - cpt[i1][1]
    marg = round(tp_add / (tp_add + fp_add), 4) if tp_add + fp_add else None
    tableau = []
    for i, s in enumerate(seuils):
        if abs(s / PAS_TABLEAU - round(s / PAS_TABLEAU)) > 1e-6:
            continue
        tableau.append({"seuil": float(s), "P": round(P[i], 4), "R": round(R[i], 4),
                        "F1": round(F1[i], 4), "F2": round(F2[i], 4), "fp_img": fp_img(i)})
    bandes = bandes_confiance(enregs, classe)
    seuil_propose = float(seuils[ip])
    return {
        "seuil_f2max": float(seuils[i2]), "F2": round(F2[i2], 4),
        "P_f2": round(P[i2], 4), "R_f2": round(R[i2], 4),
        "plateau_f1_98": [float(seuils[plateaux[0.98][0]]), float(seuils[plateaux[0.98][1]])],
        "plateau_f1_95": [float(seuils[lo95]), float(seuils[plateaux[0.95][1]])],
        "R_max": round(R[0], 4),
        "n_images": n_img,
        "fp_par_image": {"f1max": fp_img(i1), "f2max": fp_img(i2), "propose": fp_img(ip)},
        "seuil_propose": seuil_propose,
        "P_propose": round(P[ip], 4), "R_propose": round(R[ip], 4),
        "precision_marginale": marg,
        "tableau": tableau,
        # table de calibrage + catégories de fiabilité PROPOSÉES au seuil proposé
        # (le model_card porte celles recalculées au seuil RETENU, cellule 11bis)
        "bandes": bandes,
        "fiabilite_proposee": fiabilite_par_classe(
            bandes, seuil_propose, coupures_fiabilite(bandes, seuil_propose)),
    }


def ap50(enregs, classe=None):
    """AP@0,5 toutes-points par rang de confiance. Retourne (ap, rappels, précisions).
    Critère couverture : précision par rang sur les prédictions (`tps_pred` + fps),
    rappel par rang = annotations retrouvées à une confiance >= celle du rang."""
    couverture = any("tps_pred" in e for e in enregs)
    scores = []
    for e in enregs:
        scores += [(t[0], 1) for t in vrais_pred(e) if classe is None or t[2] == classe]
        scores += [(f[0], 0) for f in e["fps"] if classe is None or f[1] == classe]
    ngt = sum((sum(1 for c in e["gt_classes"] if c == classe) if classe else e["n_gt"])
              for e in enregs)
    if not scores or not ngt:
        return 0.0, np.array([0.0]), np.array([1.0])
    scores.sort(key=lambda t: -t[0])
    tp = np.cumsum([s[1] for s in scores])
    fp = np.cumsum([1 - s[1] for s in scores])
    if couverture:
        mconf = np.sort(np.array([m[0] for e in enregs for m in e["matches"]
                                  if classe is None or m[2] == classe]))
        rr = np.array([len(mconf) - np.searchsorted(mconf, s[0], side="left") for s in scores]) / ngt
    else:
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
            "iou_median": round(float(np.median(ious)), 4) if ious else None,
            "etude_seuil": etude_seuil(enregs, plancher, classe)}


def imprimer_etude(nom, bloc, indent="  "):
    """Tableau d'étude du seuil (stdout) — ce que l'analyste LIT avant de choisir."""
    et = bloc["etude_seuil"]
    print(f"{indent}{nom} : F1-max {bloc['F1']} @ {bloc['seuil_f1max']} | F2-max {et['F2']} @ "
          f"{et['seuil_f2max']} | plateau F1>=95% [{et['plateau_f1_95'][0]} ; "
          f"{et['plateau_f1_95'][1]}] | R_max {et['R_max']} | proposé {et['seuil_propose']} "
          f"(P {et['P_propose']} / R {et['R_propose']}, FP/img {et['fp_par_image']['propose']} "
          f"vs {et['fp_par_image']['f1max']} au F1-max, précision marginale "
          f"{et['precision_marginale']})")
    print(f"{indent}  seuil   P      R      F1     F2     FP/img")
    for l in et["tableau"]:
        print(f"{indent}  {l['seuil']:.2f}   {l['P']:.3f}  {l['R']:.3f}  {l['F1']:.3f}  "
              f"{l['F2']:.3f}  {l['fp_img']}")
    if et.get("fiabilite_proposee"):
        print(f"{indent}  fiabilité proposée au seuil {et['seuil_propose']} : " + " · ".join(
            f"{c['categorie']} dès {c['seuil']} (garanti >= {int(c['garanti'] * 100)} %, mesuré "
            f"{'—' if c['mesure'] is None else int(round(c['mesure'] * 100))} % sur {c['n']})"
            for c in et["fiabilite_proposee"]))


def classes_presentes(donnees):
    return sorted({m[2] for dd in donnees.values() for e in dd["enregs"] for m in e["matches"]}
                  | {f[1] for dd in donnees.values() for e in dd["enregs"] for f in e["fps"]}
                  | {c for dd in donnees.values() for e in dd["enregs"] for c in e["gt_classes"]})


def par_zone_classe(enregs, seuil_global, seuils_classe, classes):
    """Détail ZONE × CLASSE d'un modèle (fonction pure, sans GPU) — 2026-09-03.

    Par zone puis par classe (toutes les `classes`, n_gt 0 possible) :
    n_gt ; tp/fp/R/P au `seuil_global` (F1-max global, comme par_zone) ;
    R_seuil_classe/fp_seuil_classe au seuil de LA classe (`seuils_classe[c]`) ;
    R_max = rappel avec TOUS les matches du cache (inférence au plancher) —
    ce que le modèle retrouve au mieux. R null si n_gt == 0, P null si
    tp + fp == 0 (PAS la convention P = 1,0 de par_zone : ici l'absence de
    prédiction n'est pas un mérite). Arrondi 4 décimales.
    """
    def ratio(num, den):
        return round(num / den, 4) if den else None

    out = {}
    for z in sorted({e.get("zone", "") for e in enregs if e.get("zone")}):
        sous = [e for e in enregs if e.get("zone") == z]
        out[z] = {}
        for c in classes:
            sc = seuils_classe[c]
            ngt = sum(1 for e in sous for g in e["gt_classes"] if g == c)
            confs_tp = [m[0] for e in sous for m in e["matches"] if m[2] == c]      # rappel
            confs_tpp = [t[0] for e in sous for t in vrais_pred(e) if t[2] == c]   # précision
            confs_fp = [f[0] for e in sous for f in e["fps"] if f[1] == c]
            tp = sum(1 for v in confs_tpp if v >= seuil_global)
            tp_r = sum(1 for v in confs_tp if v >= seuil_global)
            fp = sum(1 for v in confs_fp if v >= seuil_global)
            out[z][c] = {
                "n_gt": ngt, "tp": tp, "fp": fp,
                "R": ratio(tp_r, ngt), "P": ratio(tp, tp + fp),
                "R_seuil_classe": ratio(sum(1 for v in confs_tp if v >= sc), ngt),
                "fp_seuil_classe": sum(1 for v in confs_fp if v >= sc),
                "R_max": ratio(len(confs_tp), ngt),
                # table de calibrage de la zone (2026-09-09) : permet de calibrer la
                # fiabilité sur les seules zones à annotation exhaustive
                "bandes": bandes_confiance(sous, c),
            }
    return out


def resumer(donnees, meta_modeles, tache, dataset, fusion, plancher, provenance_cache,
            critere="iou"):
    """Construit le dict metriques_eval/1 — LA sortie canonique de l'outil."""
    classes = classes_presentes(donnees)
    resume = {
        "schema": SCHEMA_METRIQUES,
        "genere_le": datetime.now().isoformat(timespec="seconds"),
        "outil": "tools/courbes_eval.py",
        "tache": tache,
        "critere": critere,
        "iou": {"type": "masque" if tache == "segmentation" else "bbox", "seuil": 0.5},
        "appariement": ("couverture >= 0,5 de l'union des masques annotés, class-aware"
                        if critere == "couverture" else "glouton conf decroissante, class-aware"),
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
            tp, fp, ngt, tp_r = comptes(sous, s0)
            par_zone[z] = {"P": round(tp / (tp + fp), 4) if tp + fp else 1.0,
                           "R": round(tp_r / ngt, 4) if ngt else 0.0, "n_gt": int(ngt)}
        info = meta_modeles.get(nom, {})
        par_classe = {cl: bloc_metriques(enregs, plancher, cl) for cl in classes}
        resume["modeles"][nom] = {
            "poids": info.get("poids"),
            "resolution": info.get("resolution"),
            "class_offset": dd.get("decal"),
            "global": bloc_global,
            "par_classe": par_classe,
        }
        if par_zone:
            resume["modeles"][nom]["par_zone"] = par_zone
            resume["modeles"][nom]["par_zone_classe"] = par_zone_classe(
                enregs, s0, {cl: b["seuil_f1max"] for cl, b in par_classe.items()}, classes)
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
        # étude du seuil : plateau F1 >= 95 % (bande) et seuil proposé (carré)
        et = etude_seuil(enregs, plancher)
        lo, hi = et["plateau_f1_95"]
        axes[1][0].axvspan(lo, hi, color=c, alpha=0.07)
        sp = et["seuil_propose"]
        axes[1][0].scatter([sp], [F[int(np.abs(seuils - sp).argmin())]], color=c, marker="s",
                           zorder=5, s=36)
        axes[1][0].annotate(f"proposé {sp:.2f}\n(F2-max {et['seuil_f2max']:.2f})", (sp, 0.02 + 0.06 * k),
                            fontsize=7, color=c)
        ap, rr, pp = ap50(enregs)
        axes[1][1].plot(rr, pp, color=c, label=f"{nom} — AP@0.5 {ap:.3f}")
        axes[1][1].scatter([r0], [p0], color=c, zorder=5, marker="*", s=120)
        axes[1][1].annotate(f"F1max: P={p0:.2f}, R={r0:.2f}", (r0, p0), textcoords="offset points",
                            xytext=(8, 6 + 12 * k), fontsize=8, color=c)
    for ax, t in ((axes[0][0], "Précision vs confiance"), (axes[0][1], "Rappel vs confiance"),
                  (axes[1][0], "F1 vs confiance (bande = plateau F1 ≥ 95 %, ■ = seuil proposé)"),
                  (axes[1][1], "Courbe Précision-Rappel (★ = F1-max)")):
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

def construire_meta(plancher, coco, fusion, modeles, tache=None, critere="iou"):
    meta = {"schema": SCHEMA_CACHE, "tache": tache, "plancher": plancher, "critere": critere,
            "coco": chemin_norm(coco), "fusion": fusion, "modeles": {}}
    for nom, cfg in modeles.items():
        taille = os.path.getsize(cfg["poids"]) if os.path.exists(cfg["poids"]) else None
        meta["modeles"][nom] = {"poids": chemin_norm(cfg["poids"]),
                                "resolution": cfg["resolution"], "taille_octets": taille}
    return meta


def divergence_run(attendu, cache):
    """Divergence des paramètres de RUN (plancher/coco/fusion) entre empreintes, ou None."""
    if cache.get("plancher") != attendu["plancher"]:
        return f"plancher {cache.get('plancher')} != {attendu['plancher']}"
    if cache.get("coco", "").casefold() != attendu["coco"].casefold():
        return f"coco {cache.get('coco')} != {attendu['coco']}"
    if cache.get("fusion") != attendu["fusion"]:
        return f"fusion {cache.get('fusion')} != {attendu['fusion']}"
    if cache.get("critere", "iou") != attendu.get("critere", "iou"):
        return f"critère {cache.get('critere', 'iou')} != {attendu.get('critere', 'iou')}"
    return None


def divergence_modele(nom, att, cac):
    """Divergence de l'empreinte PAR MODÈLE (poids/résolution/taille), ou None."""
    if cac.get("poids", "").casefold() != att["poids"].casefold():
        return f"{nom} : poids {cac.get('poids')} != {att['poids']}"
    if cac.get("resolution") != att["resolution"]:
        return f"{nom} : résolution {cac.get('resolution')} != {att['resolution']}"
    if (att["taille_octets"] is not None and cac.get("taille_octets") is not None
            and cac["taille_octets"] != att["taille_octets"]):
        return f"{nom} : taille des poids {cac['taille_octets']} != {att['taille_octets']}"
    return None


def meta_divergence(attendu, cache):
    """Première divergence entre l'empreinte attendue (CLI) et celle du cache, ou None."""
    div = divergence_run(attendu, cache)
    if div:
        return div
    if sorted(cache.get("modeles", {})) != sorted(attendu["modeles"]):
        return f"modèles {sorted(cache.get('modeles', {}))} != {sorted(attendu['modeles'])}"
    for nom, att in attendu["modeles"].items():
        div = divergence_modele(nom, att, cache["modeles"][nom])
        if div:
            return div
    return None


def reprendre_modeles(sources, meta_attendu, modeles, donnees, tache):
    """Cache par MODÈLE : adopte depuis d'autres sorties d'éval les appariements
    des modèles du CLI encore manquants. Exigences : empreinte de RUN identique
    (plancher/coco/fusion), tâche compatible, empreinte PAR MODÈLE identique
    (même nom, mêmes poids, même résolution). Un modèle présent dans la source
    sous le bon nom mais à empreinte divergente = ERREUR (jamais de réinférence
    silencieuse de ce que l'appelant croyait repris) ; une source qui n'apporte
    rien est tolérée (modèles déjà couverts ou absents). Retourne la tâche."""
    for dossier in sources:
        chemin = os.path.join(dossier, "appariements.json")
        if not os.path.exists(chemin):
            sys.exit(f"ERREUR : --reprendre-de {dossier} : appariements.json absent.")
        brut = json.load(open(chemin, encoding="utf-8"))
        meta_src = brut.pop("_meta", None)
        if meta_src is None:
            sys.exit(f"ERREUR : --reprendre-de {dossier} : cache sans empreinte "
                     "(legacy) — inutilisable pour la reprise par modèle.")
        div = divergence_run(meta_attendu, meta_src)
        if div:
            sys.exit(f"ERREUR : --reprendre-de {dossier} : autre provenance ({div}).")
        if tache is not None and meta_src.get("tache") not in (None, tache):
            sys.exit(f"ERREUR : --reprendre-de {dossier} : tâche "
                     f"{meta_src.get('tache')} != {tache}.")
        tache = tache or meta_src.get("tache")
        pris = []
        for nom in modeles:
            if nom in donnees or nom not in brut:
                continue
            div = divergence_modele(nom, meta_attendu["modeles"][nom],
                                    meta_src.get("modeles", {}).get(nom, {}))
            if div:
                sys.exit(f"ERREUR : --reprendre-de {dossier} : {div}.")
            donnees[nom] = brut[nom]
            pris.append(nom)
        print(f"repris de {dossier} : {', '.join(pris) if pris else 'rien'}")
    return tache


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
    ap.add_argument("--critere", choices=list(CRITERES), default="iou",
                    help="vérité d'une prédiction : iou (1-1 >= 0,5, défaut) ou couverture "
                         "(>= 50 %% de la prédiction sur les masques annotés — structures "
                         "LINÉAIRES, segmentation seulement)")
    ap.add_argument("--classe-controle", default=None,
                    help="autocontrôle du rappel plancher sur cette classe (défaut : global)")
    ap.add_argument("--sans-autocontrole", action="store_true")
    ap.add_argument("--adopter-cache", action="store_true",
                    help="accepter un appariements.json legacy sans empreinte _meta")
    ap.add_argument("--reprendre-de", action="append", default=[], metavar="DOSSIER",
                    help="sortie d'une éval précédente dont appariements.json fournit "
                         "les appariements des modèles du CLI encore manquants (cache "
                         "par modèle : empreintes de run et par-modèle vérifiées, seuls "
                         "les modèles restants sont inférés) — répétable")
    a = ap.parse_args()

    modeles = {}
    for spec in a.modele:
        nom, reste = spec.split("=", 1)
        if nom == "_meta":
            sys.exit("ERREUR : nom de modèle '_meta' interdit (réservé au cache).")
        poids, resolution = reste.rsplit("@", 1)
        sidecar = os.path.join(os.path.dirname(poids), "best.json")
        noms, decal_sidecar = None, None
        if os.path.exists(sidecar):
            sc = json.load(open(sidecar, encoding="utf-8"))
            noms, decal_sidecar = sc.get("class_names"), sc.get("class_offset")
        modeles[nom] = {"poids": poids, "resolution": int(resolution), "noms": noms,
                        "decal_sidecar": decal_sidecar}
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
    meta_attendu = construire_meta(a.plancher, a.coco, fusion, modeles, critere=a.critere)
    provenance_cache = "calculee"
    donnees, tache, modifie = {}, a.tache, False

    if os.path.exists(cache):
        brut = json.load(open(cache, encoding="utf-8"))
        meta_cache = brut.pop("_meta", None)
        if meta_cache is None:
            if not a.adopter_cache:
                sys.exit(f"ERREUR : cache sans empreinte de provenance : {cache}\n"
                         "Cache legacy — relancer avec --adopter-cache pour l'accepter "
                         "(rétrofit), ou le supprimer pour recalculer.")
            provenance_cache = "adoptee_sans_empreinte"
            tache = a.tache or "segmentation"  # les caches legacy viennent du chemin seg
            print(f"AVERTISSEMENT : cache adopté sans empreinte ({cache}), "
                  f"tâche supposée {tache}")
            if sorted(brut) != sorted(modeles):
                sys.exit(f"ERREUR : modèles du cache {sorted(brut)} != CLI "
                         f"{sorted(modeles)} — nommer les --modele comme dans le cache.")
            donnees = brut
        else:
            div = divergence_run(meta_attendu, meta_cache)
            if div:
                sys.exit(f"ERREUR : le cache {cache} vient d'une autre provenance ({div}).\n"
                         "Supprimer le cache ou changer --out.")
            # cache par MODÈLE : le cache de --out peut couvrir un sous-ensemble
            # du CLI (les manquants seront repris ailleurs ou inférés) ; un modèle
            # du cache absent du CLI reste une erreur (la sortie l'écraserait).
            for nom in brut:
                if nom not in modeles:
                    sys.exit(f"ERREUR : le cache {cache} contient '{nom}' absent du CLI — "
                             "reprendre ce modèle au CLI ou changer --out.")
                div = divergence_modele(nom, meta_attendu["modeles"][nom],
                                        meta_cache.get("modeles", {}).get(nom, {}))
                if div:
                    sys.exit(f"ERREUR : le cache {cache} vient d'une autre provenance "
                             f"({div}).\nSupprimer le cache ou changer --out.")
            donnees = brut
            tache = meta_cache["tache"]
            if a.tache and a.tache != tache:
                sys.exit(f"ERREUR : cache en tâche {tache}, --tache {a.tache} demandé.")
        print(f"cache reutilise : {cache} ({len(donnees)} modèle(s))")

    if a.reprendre_de:
        if provenance_cache == "adoptee_sans_empreinte":
            sys.exit("ERREUR : --reprendre-de est incompatible avec un cache legacy adopté.")
        n_avant = len(donnees)
        tache = reprendre_modeles(a.reprendre_de, meta_attendu, modeles, donnees, tache)
        modifie = modifie or len(donnees) > n_avant

    manquants = {nom: cfg for nom, cfg in modeles.items() if nom not in donnees}
    if manquants:
        # classes par défaut = catégories du premier split
        coco0 = json.load(open(os.path.join(splits[0][1], "_annotations.coco.json"),
                               encoding="utf-8"))
        cats_coco = [c["name"] for c in coco0["categories"]]
        for cfg in manquants.values():
            if cfg["noms"] is None:
                cfg["noms"] = cats_coco
        infere, tache = inferer(manquants, splits, fusion, a.plancher, tache, critere=a.critere)
        if not a.sans_autocontrole:
            for nom, dd in infere.items():
                _, r, _ = prf(dd["enregs"], a.plancher, a.classe_controle)
                if r < RAPPEL_MIN:
                    cible = a.classe_controle or "global"
                    sys.exit(f"ERREUR autocontrôle : rappel plancher de {nom} ({cible}) = "
                             f"{r:.3f} < {RAPPEL_MIN} — chargement/offset suspect. Cache NON "
                             "écrit. (--sans-autocontrole pour un modèle légitimement faible)")
        donnees.update(infere)
        modifie = True

    donnees = {nom: donnees[nom] for nom in modeles}  # ordre du CLI (légendes des planches)
    if modifie and provenance_cache == "calculee":
        meta_attendu["tache"] = tache
        json.dump({"_meta": meta_attendu, **donnees}, open(cache, "w", encoding="utf-8"))

    # ------- sortie canonique metriques_eval.json (validée par relecture) -------
    n_img = len(next(iter(donnees.values()))["enregs"])
    n_gt = sum(e["n_gt"] for e in next(iter(donnees.values()))["enregs"])
    dataset = {"chemin": chemin_norm(a.coco), "splits": [s for s, _ in splits],
               "n_images": n_img, "n_gt": int(n_gt)}
    resume = resumer(donnees, meta_attendu["modeles"], tache, dataset, fusion,
                     a.plancher, provenance_cache, critere=a.critere)
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
    print("\nÉTUDE DU SEUIL DE PRODUCTION — à lire avant de fixer confidence_default /"
          " confidence_per_class (le seuil proposé n'est qu'un point de départ) :")
    for nom, m in resume["modeles"].items():
        imprimer_etude(f"{nom} (global)", m["global"])
        if len(m["par_classe"]) > 1:
            for cl, b in m["par_classe"].items():
                imprimer_etude(f"{nom} / {cl}", b, indent="    ")

    # ------------------------------------------------------------- planches ---
    base_coco = os.path.basename(a.coco.rstrip("/\\"))
    iou_lib = "masque" if tache == "segmentation" else "bbox"
    titre = a.titre or (f"Évaluation {base_coco} — couverture ≥ 0,5 (linéaires)"
                        if a.critere == "couverture" else f"Évaluation {base_coco} — IoU {iou_lib} ≥ 0,5")
    planche_principale(donnees, titre, a.plancher, os.path.join(a.out, "courbes_seuils_pr.png"))
    classes = classes_presentes(donnees)
    if len(classes) > 1:
        planche_classes(donnees, classes, a.plancher, os.path.join(a.out, "f1_par_classe.png"))
    planche_zones(donnees, tache, a.plancher, os.path.join(a.out, "zones_et_masques.png"))
    print("planches ->", a.out)


if __name__ == "__main__":
    main()
