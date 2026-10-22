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
#
# - "axe" (2026-10-22, audit des linéaires — segmentation seulement) : la vérité par la
#   DISTANCE À L'AXE, en mètres, au lieu d'un rapport de surfaces (le masque annoté est un
#   tampon de largeur conventionnelle, ses bords ne garantissent rien ; seul l'axe est
#   fiable). Unité vraie si >= 50 % des pixels de son squelette sont à <= t du squelette de
#   l'union des masques annotés de sa classe ; annotation retrouvée dès que >= 50 % de son
#   squelette est à <= t d'une unité prédite (par confiance décroissante). À t = 3,5 m
#   (demi-largeur du tampon de 7 m) il reproduit la couverture à 0,01 près ; t = 5 m par
#   défaut. Les enregistrements portent `unites` et `annotations` (part de squelette à <= t
#   pour t − 1,5 / t / t + 1,5 m, distance médiane, longueurs) d'où découle la vue
#   principale matches/fps/tps_pred ; le fichier publie P_len/R_len (en longueur), les
#   fausses par distance, la sensibilité des coupures à la tolérance.
#
# Unité (`--unite`) : « fragment » (historique : une prédiction = une unité) ou « polygone »
# (2026-10-22) : les prédictions d'une image sont fusionnées comme le plugin les affiche
# (même classe, contact à <= 1 px, en deux passes : >= seuil de la classe entre elles, puis
# les autres entre elles), confiance = moyenne pondérée par l'aire. Précision, bandes et
# fiabilité comptent alors des polygones ; `bandes_fragments` garde le comptage par fragment.
CRITERES = ("iou", "couverture", "axe")
UNITES = ("fragment", "polygone")
SEUIL_COUVERTURE = 0.5
TOLERANCE_DEFAUT = 5.0     # m
ECART_SENSIBILITE = 1.5    # m : coupures recalculées à t ± 1,5 m
I_T = 1                    # index de la tolérance principale dans tolerances_m(t)
BORNES_DISTANCE = (5, 10, 30)   # fausses_par_distance : d_max en m, puis null (au-delà / inconnue)


def tolerances_m(t):
    """[t − 1,5 ; t ; t + 1,5] en mètres — la tolérance principale au milieu (I_T)."""
    return [round(t - ECART_SENSIBILITE, 3), float(t), round(t + ECART_SENSIBILITE, 3)]


def px(metres, gsd):
    """Mètres → pixels entiers (arrondi), même convention que l'audit du 2026-10-22."""
    return int(round(metres / gsd))


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


def _fenetre(bb, marge, forme):
    """Tranches (lignes, colonnes) de la boîte `bb` (y0, y1, x0, x1 inclusifs) élargie de `marge`."""
    return (slice(max(0, bb[0] - marge), min(forme[0], bb[1] + 1 + marge)),
            slice(max(0, bb[2] - marge), min(forme[1], bb[3] + 1 + marge)))


def _squelette(mask, bb, forme):
    """Squelette (skimage) calculé sur la fenêtre de la boîte, replacé dans l'image."""
    from skimage.morphology import skeletonize
    sk = np.zeros(forme, bool)
    w = _fenetre(bb, 1, forme)
    sk[w] = skeletonize(mask[w])
    return sk


def _distance_vers(S):
    """Distance (px) de chaque pixel au pixel True le plus proche de S (inf si S vide)."""
    from scipy import ndimage as ndi
    if not S.any():
        return np.full(S.shape, np.inf, np.float32)
    return ndi.distance_transform_edt(~S).astype(np.float32)


def apparier_axe(unites, objs, cls_gt, tolerances, gsd, rappel=True):
    """Critère « axe » (segmentation, linéaires) — fonction pure sur masques.

    `unites` : [(conf, classe, masque bool, extra)] par confiance DÉCROISSANTE (fragments ou
    polygones ; extra = {conf_max, n_membres} d'un polygone, {} sinon) ; `objs`/`cls_gt` :
    masques annotés et leurs classes ; `tolerances` : liste en mètres, la principale à
    l'index I_T ; `gsd` : mètres par pixel. Une unité est VRAIE si >= 50 % des pixels de
    son squelette sont à <= t du squelette de l'union des masques annotés de sa classe ;
    une annotation est RETROUVÉE dès que >= 50 % de son squelette est à <= t d'un squelette
    prédit, les unités étant prises par confiance décroissante (score = confiance qui fait
    franchir la barre). Retourne (matches, fps, tps_pred, unites_rec, annotations_rec) :
    les trois premiers = vue principale à la tolérance I_T (même forme qu'au critère
    couverture) ; unites_rec = [[conf, classe, len_px, med_m, [part à <= t par tolérance],
    conf_max, n_membres]] (med_m None sans annotation de la classe dans l'image) ;
    annotations_rec = [[classe, len_px, [bp par tolérance]]], bp = [[conf, part cumulée]]
    à chaque hausse (vide, et matches vide, si `rappel` est faux).
    """
    forme = objs[0].shape if objs else (unites[0][2].shape if unites else None)
    if forme is None:
        return [], [], [], [], []
    tols_px = [px(t, gsd) for t in tolerances]
    marge = max(tols_px) + 1
    bb_u = [_bbox(m) for _, _, m, _ in unites]
    bb_g = [_bbox(g) for g in objs]
    unites_rec, annotations_rec = [None] * len(unites), [None] * len(objs)
    for c in sorted({cu for _, cu, _, _ in unites} | set(cls_gt)):
        idx_u = [i for i, (_, cu, _, _) in enumerate(unites) if cu == c]
        idx_g = [j for j, cg in enumerate(cls_gt) if cg == c]
        U = np.zeros(forme, bool)
        for j in idx_g:
            U |= objs[j]
        d_axe = _distance_vers(_squelette(U, _bbox(U), forme)) if U.any() else None
        sk_u, d_u = {}, {}
        for i in idx_u:
            conf, _, m, extra = unites[i]
            extra_rec = [float(extra.get("conf_max", conf)), int(extra.get("n_membres", 1))]
            if bb_u[i] is None:
                unites_rec[i] = [float(conf), c, 0, None, [0.0] * len(tols_px)] + extra_rec
                continue
            sk = _squelette(m, bb_u[i], forme)
            sk_u[i] = sk
            n = int(sk.sum())
            if d_axe is None or not n:
                fracs, med = [0.0] * len(tols_px), None
            else:
                da = d_axe[sk]
                fracs = [round(float((da <= p).mean()), 4) for p in tols_px]
                med = round(float(np.median(da)) * gsd, 2)
            unites_rec[i] = [float(conf), c, n, med, fracs] + extra_rec
        for j in idx_g:
            if not rappel:
                break
            if bb_g[j] is None:
                annotations_rec[j] = [c, 0, [[] for _ in tols_px]]
                continue
            ys, xs = np.nonzero(_squelette(objs[j], bb_g[j], forme))
            n_g = len(ys)
            bps = [[] for _ in tols_px]
            couvert = [np.zeros(n_g, bool) for _ in tols_px]
            for i in idx_u:
                if i not in sk_u or not n_g:
                    continue
                bb = bb_u[i]
                if not _bbox_chevauche((bb[0] - marge, bb[1] + marge, bb[2] - marge, bb[3] + marge), bb_g[j]):
                    continue
                if i not in d_u:
                    w = _fenetre(bb, marge, forme)
                    d_u[i] = (w, _distance_vers(sk_u[i][w]))
                (wy, wx), di = d_u[i]
                dedans = (ys >= wy.start) & (ys < wy.stop) & (xs >= wx.start) & (xs < wx.stop)
                if not dedans.any():
                    continue
                d = np.full(n_g, np.inf, np.float32)
                d[dedans] = di[ys[dedans] - wy.start, xs[dedans] - wx.start]
                for k, p in enumerate(tols_px):
                    nouveau = (d <= p) & ~couvert[k]
                    if nouveau.any():
                        couvert[k] |= nouveau
                        bps[k].append([float(unites[i][0]), round(float(couvert[k].mean()), 4)])
            annotations_rec[j] = [c, n_g, bps]
    if not rappel:
        annotations_rec = []
    matches, fps, tps_pred = vue_principale(unites_rec, annotations_rec)
    return matches, fps, tps_pred, unites_rec, annotations_rec


def vue_principale(unites_rec, annotations_rec):
    """(matches, fps, tps_pred) à la tolérance principale — la forme que lisent comptes,
    prf, bandes_confiance, ap50 ; dérivée des unités et annotations (donc recalculable
    par le contrôleur sans masque)."""
    tps = [[u[0], u[4][I_T], u[1]] for u in unites_rec if u[4][I_T] >= SEUIL_COUVERTURE]
    fps = [[u[0], u[1]] for u in unites_rec if u[4][I_T] < SEUIL_COUVERTURE]
    matches = []
    for a in annotations_rec:
        trouve = next((x for x in a[2][I_T] if x[1] >= SEUIL_COUVERTURE), None)
        if trouve is not None:
            matches.append([trouve[0], trouve[1], a[0]])
    return matches, fps, tps


def _composantes(masques):
    """Groupes d'indices de masques (même image) qui se touchent ou se chevauchent à
    <= 1 px : composantes 8-connexes de l'union, puis pontage par les pixels vides
    voisins de >= 2 composantes (un pixel vide commun = contact à 1 px, diagonales
    comprises — un fragment lui-même non connexe relie ses propres composantes)."""
    if not masques:
        return []
    if len(masques) == 1:
        return [[0]]
    from scipy import ndimage as ndi
    union = np.zeros_like(masques[0])
    for m in masques:
        union |= m
    lab, n = ndi.label(union, np.ones((3, 3), bool))
    parent = list(range(n + 1))

    def racine(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unir(a, b):
        ra, rb = racine(a), racine(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    H, W = lab.shape
    pad = np.pad(lab, 1)
    vois = np.stack([pad[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    v = vois[:, lab == 0]                                   # (8, pixels vides)
    vmin = np.where(v == 0, n + 1, v).min(axis=0)
    for s in range(8):
        ok = (vmin <= n) & (v[s] > vmin)
        if ok.any():
            for a, b in np.unique(np.stack([vmin[ok], v[s][ok]]), axis=1).T:
                unir(int(a), int(b))
    etiq = []
    for m in masques:
        labs = np.unique(lab[m])
        labs = labs[labs > 0]
        for l in labs[1:]:
            unir(int(labs[0]), int(l))
        etiq.append(int(labs[0]) if len(labs) else None)
    groupes = {}
    for k, l in enumerate(etiq):
        groupes.setdefault(("vide", k) if l is None else racine(l), []).append(k)
    return list(groupes.values())


def fusionner_polygones(preds, seuils_classe):
    """Unité « polygone » : fusion des prédictions d'une image comme le plugin (même
    classe, contact ou chevauchement à <= 1 px), en DEUX passes : (a) les fragments
    >= seuil de la classe entre eux, (b) ceux sous le seuil entre eux, jamais avec (a).
    Confiance = moyenne pondérée par l'aire des membres ; extra = conf_max, n_membres.
    `preds` : [(conf, classe, masque)] ; retourne [(conf, classe, masque, extra)] par
    confiance décroissante."""
    out = []
    for c in sorted({cl for _, cl, _ in preds}):
        if c not in seuils_classe:
            sys.exit(f"ERREUR : unité polygone — pas de seuil de classe pour « {c} » "
                     "(--seuil-classe, model_card ou F1-max).")
        for passe in (True, False):
            sel = [i for i, (conf, cl, _) in enumerate(preds)
                   if cl == c and (conf >= seuils_classe[c]) == passe]
            for groupe in _composantes([preds[i][2] for i in sel]):
                membres = [sel[k] for k in groupe]
                aires = np.array([preds[i][2].sum() for i in membres], float)
                confs = np.array([preds[i][0] for i in membres], float)
                masque = np.zeros_like(preds[membres[0]][2])
                for i in membres:
                    masque |= preds[i][2]
                conf = float(np.average(confs, weights=aires)) if aires.sum() else float(confs.max())
                out.append((conf, c, masque, {"conf_max": float(confs.max()), "n_membres": len(membres)}))
    out.sort(key=lambda t: -t[0])
    return out


def apparier_unites(preds, objs, cls_gt, critere, params):
    """Enregistrement d'une image aux critères couverture / axe (segmentation), en unité
    fragment ou polygone. `preds` : [(conf, classe, masque)] par confiance décroissante ;
    `params` : tolerances, gsd, unite, seuils_classe."""
    unite = params.get("unite", "fragment")
    if unite == "polygone":
        unites = fusionner_polygones(preds, params["seuils_classe"])
    else:
        unites = [(conf, c, m, {}) for conf, c, m in preds]
    if critere == "couverture":
        matches, fps, tps = apparier_couverture([(conf, c, m) for conf, c, m, _ in unites], objs, cls_gt)
        return {"matches": matches, "fps": fps, "tps_pred": tps}
    tols, gsd = params["tolerances"], params["gsd"]
    matches, fps, tps, urec, arec = apparier_axe(unites, objs, cls_gt, tols, gsd)
    rec = {"matches": matches, "fps": fps, "tps_pred": tps, "unites": urec, "annotations": arec}
    if unite == "polygone":
        frec = apparier_axe([(conf, c, m, {}) for conf, c, m in preds], objs, cls_gt, tols, gsd, rappel=False)[3]
        rec["fragments"] = [[u[0], u[1], u[4][I_T]] for u in frec]
    return rec


def apparier_iou(preds, objs, cls_gt, seg):
    """Critère « iou » : appariement 1-1 glouton par confiance décroissante, class-aware,
    IoU masque (seg) ou boîte >= 0,5. `preds` : [(conf, classe, masque | boîte xyxy)]."""
    pris = [False] * len(objs)
    matches, fps = [], []
    for conf, c, obj in preds:
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
            matches.append([float(conf), float(meilleur), c])
        else:
            fps.append([float(conf), c])
    return {"matches": matches, "fps": fps}


def _rle(mask):
    from pycocotools import mask as cocomask
    r = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": r["size"], "counts": r["counts"].decode("ascii")}


def _masque(rle):
    from pycocotools import mask as cocomask
    return cocomask.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(bool)


def lire_masques(dossier, etiquette, coco):
    """Masques prédits mis en cache (--masques-cache ; format de l'audit du 2026-10-22 :
    preds_<split>.json, une entrée par image dans l'ordre coco.getImgIds(), prédictions
    au plancher par confiance décroissante, RLE COCO, classe = nom du modèle avant
    fusion) — ou None si le fichier n'existe pas."""
    chemin = os.path.join(dossier, f"preds_{etiquette}.json")
    if not os.path.exists(chemin):
        return None
    entrees = json.load(open(chemin, encoding="utf-8"))
    attendu = [coco.loadImgs(i)[0]["file_name"] for i in coco.getImgIds()]
    if [e["file_name"] for e in entrees] != attendu:
        sys.exit(f"ERREUR : {chemin} ne suit pas les images de {etiquette} (ordre ou contenu).")
    return entrees


def charger_modele(nom, cfg, tache, critere):
    """Charge un checkpoint RF-DETR ; retourne (modele, seg, tache) après les contrôles de tâche."""
    from rfdetr import RFDETR
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
    if critere != "iou" and not seg:
        sys.exit(f"ERREUR : critère {critere} = segmentation seulement ({nom} est en détection).")
    modele.optimize_for_inference()
    return modele, seg, tache


def inferer(modeles, splits, fusion, plancher, tache_forcee, critere="iou", params=None,
            masques_cache=None):
    """Inférence (ou relecture du cache de masques) + appariement de chaque image, par
    modèle. `params` (critères couverture/axe) : unite, seuils_classe, tolerances, gsd.
    `masques_cache` : dossier des preds_<split>.json — lus s'ils existent (aucun GPU
    nécessaire), écrits sinon."""
    from pycocotools.coco import COCO

    params = params or {}
    donnees, tache = {}, tache_forcee
    for nom, cfg in modeles.items():
        modele, seg = None, None
        # class_offset du sidecar best.json = vérité (leçon 2026-09-08 : run_rf_detr_1 prédit
        # parfois la catégorie 0 « entites » héritée de Roboflow -> l'heuristique sur la
        # première image tombait à 0 et perdait la classe four) ; heuristique sinon.
        decal = cfg.get("decal_sidecar")
        enregs = []
        for etiquette, dossier in splits:
            coco = COCO(os.path.join(dossier, "_annotations.coco.json"))
            cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
            entrees = lire_masques(masques_cache, etiquette, coco) if masques_cache else None
            a_ecrire = None
            if entrees is None:
                if modele is None:
                    modele, seg, tache = charger_modele(nom, cfg, tache, critere)
                if masques_cache:
                    a_ecrire = []
            else:
                seg, tache = True, tache or "segmentation"
                if tache != "segmentation":
                    sys.exit(f"ERREUR : cache de masques ({etiquette}) = segmentation, run {tache}.")
                print(f"{nom} : masques de {etiquette} lus dans {masques_cache} ({len(entrees)} images)")
            for k, img_id in enumerate(coco.getImgIds()):
                info = coco.loadImgs(img_id)[0]
                if entrees is not None:
                    preds = [(float(p["conf"]), fusion.get(p["cls"], p["cls"]), _masque(p["rle"]))
                             for p in entrees[k]["preds"]]
                    preds.sort(key=lambda t: -t[0])
                else:
                    from PIL import Image
                    im = Image.open(os.path.join(dossier, info["file_name"])).convert("RGB")
                    d = modele.predict(im, threshold=plancher)
                    n_pred = len(d)
                    if seg and n_pred and d.mask is None:
                        sys.exit(f"ERREUR : {nom} (segmentation) prédit sans masques.")
                    if decal is None and n_pred:
                        # vieux exports : background en colonne 0 -> ids 1..N
                        decal = 1 if (int(d.class_id.min()) >= 1
                                      and int(d.class_id.max()) >= len(cfg["noms"])) else 0
                    preds, bruts = [], []
                    for i in (np.argsort(-d.confidence) if n_pred else []):
                        idx = int(d.class_id[i]) - (decal or 0)
                        if not (0 <= idx < len(cfg["noms"])):
                            continue
                        obj = d.mask[i].astype(bool) if seg else tuple(map(float, d.xyxy[i]))
                        preds.append((float(d.confidence[i]), fusion.get(cfg["noms"][idx], cfg["noms"][idx]), obj))
                        if a_ecrire is not None:
                            bruts.append({"conf": float(d.confidence[i]), "cls": cfg["noms"][idx], "rle": _rle(obj)})
                    if a_ecrire is not None:
                        a_ecrire.append({"split": etiquette, "zone": info.get("zone", ""), "image_id": img_id,
                                         "file_name": info["file_name"], "preds": bruts})
                objs, cls_gt = [], []
                for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
                    c = fusion.get(cats[a["category_id"]], cats[a["category_id"]])
                    if seg:
                        objs.append(coco.annToMask(a).astype(bool))
                    else:
                        x, y, w, h = a["bbox"]
                        objs.append((x, y, x + w, y + h))
                    cls_gt.append(c)
                if critere == "iou":
                    rec = apparier_iou(preds, objs, cls_gt, seg)
                else:
                    rec = apparier_unites(preds, objs, cls_gt, critere, params)
                enregs.append({"split": etiquette, "zone": info.get("zone", ""),
                               "n_gt": len(objs), "gt_classes": cls_gt, **rec})
                if k % 200 == 0 and k:
                    print(f"  {nom} / {etiquette} : {k}/{len(coco.getImgIds())}", flush=True)
            if a_ecrire is not None:
                os.makedirs(masques_cache, exist_ok=True)
                with open(os.path.join(masques_cache, f"preds_{etiquette}.json"), "w", encoding="utf-8") as f:
                    json.dump(a_ecrire, f)
                print(f"{nom} : masques de {etiquette} écrits dans {masques_cache}")
        donnees[nom] = {"decal": decal, "enregs": enregs}
        print(f"{nom} : {sum(len(e['matches']) for e in enregs)} TP potentiels, "
              f"offset {decal}, tâche {tache}")
        if modele is not None:
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


def longueurs(enregs, s, classe=None):
    """(P_len, R_len) au seuil s, critère axe : part de la longueur de squelette prédit
    (unités >= s) à <= t d'un axe annoté ; part de la longueur d'axe annoté à <= t d'une
    unité >= s. P_len None sans unité, R_len 0 sans annotation."""
    num_p = den_p = num_r = den_r = 0.0
    for e in enregs:
        for u in e["unites"]:
            if (classe is None or u[1] == classe) and u[0] >= s:
                den_p += u[2]
                num_p += u[2] * u[4][I_T]
        for a in e["annotations"]:
            if classe is None or a[0] == classe:
                den_r += a[1]
                f = 0.0
                for conf, frac in a[2][I_T]:
                    if conf < s:
                        break
                    f = frac
                num_r += a[1] * f
    return (round(num_p / den_p, 4) if den_p else None, round(num_r / den_r, 4) if den_r else 0.0)


def vue_fragments(enregs):
    """Enregistrements vus PAR FRAGMENT (unité polygone) : [{tps_pred, fps}] pour
    bandes_confiance — vrai = part de squelette >= 0,5 à la tolérance principale."""
    return [{"tps_pred": [[f[0], f[2], f[1]] for f in e.get("fragments", []) if f[2] >= SEUIL_COUVERTURE],
             "fps": [[f[0], f[1]] for f in e.get("fragments", []) if f[2] < SEUIL_COUVERTURE]}
            for e in enregs]


def bandes_tolerance(enregs, k, classe=None):
    """Bandes (tp/fp par 0,01) avec la vérité des unités prise à la tolérance d'index k."""
    vue = [{"tps_pred": [[u[0], u[4][k], u[1]] for u in e["unites"] if u[4][k] >= SEUIL_COUVERTURE],
            "fps": [[u[0], u[1]] for u in e["unites"] if u[4][k] < SEUIL_COUVERTURE]} for e in enregs]
    return bandes_confiance(vue, classe)


def sensibilite(enregs, classe, seuil, tolerances):
    """Coupures et catégories de fiabilité de la classe au `seuil`, recalculées à chaque
    tolérance (t − 1,5 / t / t + 1,5 m) : ce qui bouge quand la tolérance bouge.
    {"<t>": {n_vrais, coupures, fiabilite}}."""
    out = {}
    for k, t in enumerate(tolerances):
        b = bandes_tolerance(enregs, k, classe)
        cp = coupures_fiabilite(b, seuil)
        out[str(t)] = {"n_vrais": sum(x["tp"] for x in b), "coupures": cp,
                       "fiabilite": fiabilite_par_classe(b, seuil, cp)}
    return out


def fausses_par_distance(enregs, classe, seuil, bornes=BORNES_DISTANCE):
    """Histogramme des unités FAUSSES (>= seuil) de la classe par distance médiane à l'axe
    annoté le plus proche : [{d_max: 5, n}, {d_max: 10, n}, {d_max: 30, n}, {d_max: null, n}]
    — la dernière = au-delà de la dernière borne, ou aucune annotation de la classe dans
    l'image (la part « loin de tout » est celle qui mérite une relecture d'annotation)."""
    n = [0] * (len(bornes) + 1)
    for e in enregs:
        for u in e["unites"]:
            if u[1] != classe or u[0] < seuil or u[4][I_T] >= SEUIL_COUVERTURE:
                continue
            n[next((i for i, b in enumerate(bornes) if u[3] is not None and u[3] <= b), len(bornes))] += 1
    return [{"d_max": b, "n": c} for b, c in zip(list(bornes) + [None], n)]


def etude_seuil(enregs, plancher, classe=None, seuil_classe=None):
    """Indicateurs du choix du seuil (fonction pure, sans GPU) — cf. commentaire ci-dessus.

    Toutes les valeurs découlent des comptes (tp, fp) par seuil de la grille ; arrondi
    4 décimales. `plateau_f1_95`/`_98` = [min, max] des seuils où F1 ≥ 95 % / 98 % du
    F1-max. `precision_marginale` = part de vrais objets parmi les détections AJOUTÉES
    en descendant du F1-max au seuil proposé (null si rien n'est ajouté). `tableau` =
    une ligne par seuil multiple de 0,05 (seuil, P, R, F1, F2, fp_img ; + P_len, R_len au
    critère axe). Unité polygone : `bandes_fragments` (comptage par fragment) à côté des
    `bandes` (polygones), et `sous_seuil` = fenêtres de 0,05 des bandes de polygones sous
    `seuil_classe` (ceux de la seconde passe de fusion) quand ce seuil est connu.
    """
    seuils = grille(plancher)
    n_img = len(enregs)
    cpt = [comptes(enregs, s, classe) for s in seuils]
    avec_long = any("unites" in e for e in enregs)
    avec_frag = any("fragments" in e for e in enregs)
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
        ligne = {"seuil": float(s), "P": round(P[i], 4), "R": round(R[i], 4),
                 "F1": round(F1[i], 4), "F2": round(F2[i], 4), "fp_img": fp_img(i)}
        if avec_long:
            ligne["P_len"], ligne["R_len"] = longueurs(enregs, s, classe)
        tableau.append(ligne)
    bandes = bandes_confiance(enregs, classe)
    seuil_propose = float(seuils[ip])
    extra = {}
    if avec_frag:
        extra["bandes_fragments"] = bandes_confiance(vue_fragments(enregs), classe)
        if seuil_classe is not None:
            extra["sous_seuil"] = agreger_bandes([b for b in bandes if b["lo"] < seuil_classe - 1e-9])
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
        **extra,
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


def bloc_metriques(enregs, plancher, classe=None, seuil_classe=None):
    """Métriques standard d'un modèle (ou d'une classe) : point F1-max + AP50 ;
    `seuil_classe` (unité polygone) ne sert qu'au bloc sous_seuil de l'étude."""
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
            "etude_seuil": etude_seuil(enregs, plancher, classe, seuil_classe)}


def imprimer_etude(nom, bloc, indent="  "):
    """Tableau d'étude du seuil (stdout) — ce que l'analyste LIT avant de choisir."""
    et = bloc["etude_seuil"]
    print(f"{indent}{nom} : F1-max {bloc['F1']} @ {bloc['seuil_f1max']} | F2-max {et['F2']} @ "
          f"{et['seuil_f2max']} | plateau F1>=95% [{et['plateau_f1_95'][0]} ; "
          f"{et['plateau_f1_95'][1]}] | R_max {et['R_max']} | proposé {et['seuil_propose']} "
          f"(P {et['P_propose']} / R {et['R_propose']}, FP/img {et['fp_par_image']['propose']} "
          f"vs {et['fp_par_image']['f1max']} au F1-max, précision marginale "
          f"{et['precision_marginale']})")
    long_ = "P_len" in (et["tableau"][0] if et["tableau"] else {})
    print(f"{indent}  seuil   P      R      F1     F2     FP/img" + ("   P_len  R_len" if long_ else ""))
    for l in et["tableau"]:
        suite = ""
        if long_:
            p_len = "—    " if l["P_len"] is None else f"{l['P_len']:.3f}"
            suite = f"   {p_len}  {l['R_len']:.3f}"
        print(f"{indent}  {l['seuil']:.2f}   {l['P']:.3f}  {l['R']:.3f}  {l['F1']:.3f}  "
              f"{l['F2']:.3f}  {l['fp_img']}" + suite)
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
            if any("fragments" in e for e in sous):   # unité polygone : comptage par fragment
                out[z][c]["bandes_fragments"] = bandes_confiance(vue_fragments(sous), c)
    return out


def appariement_texte(critere, tolerance=None):
    """Ligne `appariement` de l'en-tête, en clair (lue par completer_metriques_eval)."""
    if critere == "axe":
        return (f"axe : >= 50 % du squelette de l'unité à <= {tolerance} m du squelette des masques "
                f"annotés de sa classe ; annotation retrouvée dès que >= 50 % de son squelette est "
                f"à <= {tolerance} m d'une unité prédite, class-aware")
    if critere == "couverture":
        return "couverture >= 0,5 de l'union des masques annotés, class-aware"
    return "glouton conf decroissante, class-aware"


def resumer(donnees, meta_modeles, tache, dataset, fusion, plancher, provenance_cache,
            critere="iou", tolerance=None, unite="fragment", seuils_classe=None, gsd=None):
    """Construit le dict metriques_eval/1 — LA sortie canonique de l'outil.
    Critère axe : `tolerance` (m) et `gsd` (m/px) en tête, blocs `fausses_par_distance` et
    `sensibilite` par modèle (au seuil de classe connu, sinon au F1-max de la classe) ;
    unité polygone : `seuils_classe` (ceux des deux passes de fusion) en tête."""
    if critere == "axe" and tolerance is None:
        raise ValueError("critère axe : tolérance requise")
    classes = classes_presentes(donnees)
    resume = {
        "schema": SCHEMA_METRIQUES,
        "genere_le": datetime.now().isoformat(timespec="seconds"),
        "outil": "tools/courbes_eval.py",
        "tache": tache,
        "critere": critere,
        "unite": unite,
        "iou": {"type": "masque" if tache == "segmentation" else "bbox", "seuil": 0.5},
        "appariement": appariement_texte(critere, tolerance),
        "plancher": plancher,
        "grille": {"min": plancher, "max": 0.95, "pas": 0.005},
        "p_sans_prediction": 1.0,
        "dataset": dataset,
        "fusion": fusion,
        "provenance_cache": provenance_cache,
        "modeles": {},
    }
    if critere == "axe":
        resume["tolerance_m"] = float(tolerance)
        resume["gsd_m"] = gsd
    if seuils_classe:
        resume["seuils_classe"] = seuils_classe
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
        par_classe = {cl: bloc_metriques(enregs, plancher, cl, (seuils_classe or {}).get(cl))
                      for cl in classes}
        resume["modeles"][nom] = {
            "poids": info.get("poids"),
            "resolution": info.get("resolution"),
            "class_offset": dd.get("decal"),
            "global": bloc_global,
            "par_classe": par_classe,
        }
        if critere == "axe":
            seuils = {cl: (seuils_classe or {}).get(cl, par_classe[cl]["seuil_f1max"]) for cl in classes}
            tols = tolerances_m(tolerance)
            resume["modeles"][nom]["fausses_par_distance"] = {
                cl: {"seuil": seuils[cl], "histogramme": fausses_par_distance(enregs, cl, seuils[cl])}
                for cl in classes}
            resume["modeles"][nom]["sensibilite"] = {
                cl: {"seuil": seuils[cl], "tolerances": sensibilite(enregs, cl, seuils[cl], tols)}
                for cl in classes}
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

def construire_meta(plancher, coco, fusion, modeles, tache=None, critere="iou", tolerance=None,
                    unite="fragment", seuils_classe=None, gsd=None):
    """Empreinte de provenance du cache ; tolérance + gsd au critère axe, unité + seuils de
    classe en unité polygone (absents sinon : les caches antérieurs restent lisibles)."""
    meta = {"schema": SCHEMA_CACHE, "tache": tache, "plancher": plancher, "critere": critere,
            "coco": chemin_norm(coco), "fusion": fusion, "modeles": {}}
    if critere == "axe":
        meta["tolerance"] = float(tolerance)
        meta["gsd"] = gsd
    if unite != "fragment":
        meta["unite"] = unite
        meta["seuils_classe"] = seuils_classe
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
    if cache.get("tolerance") != attendu.get("tolerance"):
        return f"tolérance {cache.get('tolerance')} != {attendu.get('tolerance')}"
    if cache.get("gsd") != attendu.get("gsd"):
        return f"gsd {cache.get('gsd')} != {attendu.get('gsd')}"
    if cache.get("unite", "fragment") != attendu.get("unite", "fragment"):
        return f"unité {cache.get('unite', 'fragment')} != {attendu.get('unite', 'fragment')}"
    if cache.get("seuils_classe") != attendu.get("seuils_classe"):
        return f"seuils de classe {cache.get('seuils_classe')} != {attendu.get('seuils_classe')}"
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


def lire_gsd(coco):
    """gsd_m du corpus_manifest.yaml du corpus (dossier donné ou son parent), sinon None."""
    import yaml
    for d in (coco, os.path.dirname(os.path.abspath(coco.rstrip("/\\")))):
        chemin = os.path.join(d, "corpus_manifest.yaml")
        if os.path.exists(chemin):
            v = (yaml.safe_load(open(chemin, encoding="utf-8")) or {}).get("gsd_m")
            return float(v) if v is not None else None
    return None


def resoudre_seuils(cli, poids, classes):
    """Seuils de classe des deux passes de fusion (unité polygone) : --seuil-classe (doit
    couvrir toutes les classes), sinon thresholds.confidence_per_class (+ confidence_default)
    du model_card.yaml voisin des poids, sinon None = F1-max (première passe en fragments).
    Retourne (dict | None, source)."""
    if cli:
        seuils = {}
        for spec in cli:
            c, v = spec.split("=", 1)
            seuils[c] = float(v)
        manquent = [c for c in classes if c not in seuils]
        if manquent:
            sys.exit(f"ERREUR : --seuil-classe manquant pour {manquent}.")
        return seuils, "cli"
    carte = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(poids))), "model_card.yaml")
    if os.path.exists(carte):
        import yaml
        th = (yaml.safe_load(open(carte, encoding="utf-8")) or {}).get("thresholds") or {}
        par, defaut = th.get("confidence_per_class") or {}, th.get("confidence_default")
        seuils = {c: float(par.get(c, defaut)) for c in classes if par.get(c, defaut) is not None}
        if len(seuils) == len(classes):
            return seuils, "model_card"
        print(f"AVERTISSEMENT : {carte} ne couvre pas toutes les classes — seuils au F1-max")
    return None, "f1max"


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
                    help="vérité d'une prédiction : iou (1-1 >= 0,5, défaut), couverture "
                         "(>= 50 %% de la prédiction sur les masques annotés) ou axe (>= 50 %% du "
                         "squelette à <= --tolerance m de l'axe annoté) — structures LINÉAIRES, "
                         "segmentation seulement")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_DEFAUT,
                    help="critère axe : distance admise à l'axe annoté, en mètres (défaut 5,0)")
    ap.add_argument("--unite", choices=list(UNITES), default="fragment",
                    help="unité jugée : fragment (une prédiction, défaut) ou polygone (prédictions "
                         "fusionnées comme le plugin, critère axe seulement)")
    ap.add_argument("--seuil-classe", action="append", default=[], metavar="CLASSE=SEUIL",
                    help="unité polygone : seuil qui sépare les deux passes de fusion (répétable ; "
                         "défaut : confidence_per_class du model_card.yaml voisin des poids, sinon F1-max)")
    ap.add_argument("--gsd", type=float, default=None,
                    help="mètres par pixel (défaut : gsd_m du corpus_manifest.yaml du corpus)")
    ap.add_argument("--masques-cache", default=None, metavar="DOSSIER",
                    help="cache des masques prédits (preds_<split>.json, format de l'audit du "
                         "2026-10-22) : lus s'ils existent (sans GPU), écrits sinon — un seul modèle")
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
    if a.unite == "polygone" and a.critere != "axe":
        sys.exit("ERREUR : --unite polygone exige --critere axe.")
    if (a.unite == "polygone" or a.masques_cache) and len(modeles) != 1:
        sys.exit("ERREUR : --unite polygone et --masques-cache = un seul --modele.")

    if os.path.exists(os.path.join(a.coco, "_annotations.coco.json")):
        splits = [(os.path.basename(a.coco.rstrip("/\\")), a.coco)]
    else:
        splits = [(s, os.path.join(a.coco, s)) for s in ("valid", "test")
                  if os.path.exists(os.path.join(a.coco, s, "_annotations.coco.json"))]
    if not splits:
        sys.exit(f"ERREUR : aucun _annotations.coco.json trouvé sous {a.coco} "
                 "(ni directement, ni dans valid/ ou test/).")
    # classes par défaut = catégories du premier split
    coco0 = json.load(open(os.path.join(splits[0][1], "_annotations.coco.json"), encoding="utf-8"))
    cats_coco = [c["name"] for c in coco0["categories"]]
    for cfg in modeles.values():
        if cfg["noms"] is None:
            cfg["noms"] = cats_coco

    gsd = None
    if a.critere == "axe":
        gsd = a.gsd if a.gsd is not None else lire_gsd(a.coco)
        if gsd is None:
            sys.exit("ERREUR : critère axe — gsd_m introuvable dans corpus_manifest.yaml (donner --gsd).")
    tolerance = a.tolerance if a.critere == "axe" else None
    seuils_classe, source_seuils = None, None
    if a.unite == "polygone":
        classes_modele = sorted({fusion.get(n, n) for cfg in modeles.values() for n in cfg["noms"]})
        seuils_classe, source_seuils = resoudre_seuils(
            a.seuil_classe, next(iter(modeles.values()))["poids"], classes_modele)

    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "appariements.json")
    if a.unite == "polygone" and seuils_classe is None:
        # F1-max : adopté d'un cache déjà calculé à cette provenance, sinon première passe
        # en unité fragment (les masques sont relus du cache s'il est donné)
        if os.path.exists(cache):
            m0 = json.load(open(cache, encoding="utf-8")).get("_meta") or {}
            if m0.get("seuils_classe_source") == "f1max" and m0.get("seuils_classe"):
                seuils_classe = m0["seuils_classe"]
        if seuils_classe is None:
            print("unité polygone sans seuil connu : première passe en fragments pour le F1-max par classe")
            frag, _ = inferer(modeles, splits, fusion, a.plancher, a.tache, critere=a.critere,
                              params={"unite": "fragment", "tolerances": tolerances_m(tolerance), "gsd": gsd},
                              masques_cache=a.masques_cache)
            enr = next(iter(frag.values()))["enregs"]
            seuils_classe = {c: bloc_metriques(enr, a.plancher, c)["seuil_f1max"] for c in classes_presentes(frag)}
            print("seuils F1-max (fragments) :", seuils_classe)
    meta_attendu = construire_meta(a.plancher, a.coco, fusion, modeles, critere=a.critere,
                                   tolerance=tolerance, unite=a.unite, seuils_classe=seuils_classe, gsd=gsd)
    if seuils_classe:
        meta_attendu["seuils_classe_source"] = source_seuils
    params = {"unite": a.unite, "seuils_classe": seuils_classe, "tolerances": tolerances_m(tolerance or 0.0),
              "gsd": gsd}
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
        infere, tache = inferer(manquants, splits, fusion, a.plancher, tache, critere=a.critere,
                                params=params, masques_cache=a.masques_cache)
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
                     a.plancher, provenance_cache, critere=a.critere, tolerance=tolerance,
                     unite=a.unite, seuils_classe=seuils_classe, gsd=gsd)
    if seuils_classe:
        resume["seuils_classe_source"] = source_seuils
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
    for nom, m in resume["modeles"].items():
        for cl, fd in (m.get("fausses_par_distance") or {}).items():
            print(f"  {nom} / {cl} : fausses au seuil {fd['seuil']}, par distance médiane à l'axe : "
                  + ", ".join((f"au-delà de {BORNES_DISTANCE[-1]} m ou sans annotation" if h["d_max"] is None
                               else f"<= {h['d_max']} m") + f" {h['n']}" for h in fd["histogramme"]))
        for cl, se in (m.get("sensibilite") or {}).items():
            print(f"  {nom} / {cl} : fiabilité au seuil {se['seuil']} selon la tolérance : " + " | ".join(
                f"{t} m -> " + (", ".join(f"{c['categorie']} dès {c['seuil']} ({c['n']})" for c in v["fiabilite"]) or "—")
                for t, v in se["tolerances"].items()))

    # ------------------------------------------------------------- planches ---
    base_coco = os.path.basename(a.coco.rstrip("/\\"))
    iou_lib = "masque" if tache == "segmentation" else "bbox"
    titre = a.titre or (f"Évaluation {base_coco} — axe à {a.tolerance} m, unité {a.unite} (linéaires)"
                        if a.critere == "axe" else
                        f"Évaluation {base_coco} — couverture ≥ 0,5 (linéaires)"
                        if a.critere == "couverture" else f"Évaluation {base_coco} — IoU {iou_lib} ≥ 0,5")
    planche_principale(donnees, titre, a.plancher, os.path.join(a.out, "courbes_seuils_pr.png"))
    classes = classes_presentes(donnees)
    if len(classes) > 1:
        planche_classes(donnees, classes, a.plancher, os.path.join(a.out, "f1_par_classe.png"))
    planche_zones(donnees, tache, a.plancher, os.path.join(a.out, "zones_et_masques.png"))
    print("planches ->", a.out)


if __name__ == "__main__":
    main()
