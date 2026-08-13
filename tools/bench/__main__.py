"""CLI du banc d'essai.

    python -m tools.bench info      --data /data/test
    python -m tools.bench subset    --data /data/valid --n 400 --out /out/bench/subset_valid.json
    python -m tools.bench forward   --data /data/valid --subset ... --device gpu
    python -m tools.bench e0        --data /data/valid --subset ...
    python -m tools.bench sweep     --data /data/valid --subset ... --axes configs/bench/axes.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench import metrics as M                       # noqa: E402
from tools.bench.cache import Cache, charger_session, cle_cache, forward  # noqa: E402
from tools.bench.data import Corpus, selectionner           # noqa: E402
from tools.bench.decode import Params, SliceOut, run as decoder  # noqa: E402

MODELE_DEFAUT = os.environ.get(
    "BENCH_MODEL",
    "/plugin/data/models/lineaires_seg_v2_1/weights/best.onnx")
OUT_DEFAUT = os.environ.get("BENCH_OUT", "/out/bench")


def _charger_image(p: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"))


def _ids(corpus: Corpus, subset: Optional[str]) -> List[int]:
    if not subset:
        return sorted(corpus.images)
    d = json.loads(Path(subset).read_text(encoding="utf-8"))
    return d["image_ids"]


# --------------------------------------------------------------------------------------

def cmd_info(a) -> None:
    c = Corpus(Path(a.data))
    print(json.dumps(c.stats(), indent=2, ensure_ascii=False))


def cmd_subset(a) -> None:
    c = Corpus(Path(a.data))
    sel = selectionner(c, a.n, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(sel["resume"], indent=2, ensure_ascii=False))
    print(f"\n{len(sel['image_ids'])} images -> {a.out}"
          f"  (recensement {sel['n_recensement']}, negatifs {sel['n_negatifs']})")


def cmd_forward(a) -> None:
    import onnxruntime as ort
    corpus = Corpus(Path(a.data))
    ids = _ids(corpus, a.subset)
    session, input_name, shape, meta, provider = charger_session(a.model, a.device)
    hw = (int(shape[3]), int(shape[2]))
    cle = cle_cache(a.model, ort.__version__, provider, "plugin_v1", hw, a.tta, a.floor)
    cache = Cache(Path(a.out) / "cache", cle, meta={
        "modele": a.model, "ort": ort.__version__, "provider": provider,
        "prepro": "plugin_v1", "input_hw": list(hw), "tta": a.tta, "plancher": a.floor,
        "split": Path(a.data).name,
    })
    print(f"cache={cle} provider={provider} tta={a.tta} plancher={a.floor} n={len(ids)}")

    t0, n_k, n_fait = time.time(), 0, 0
    for i, iid in enumerate(ids):
        img = corpus.images[iid]
        unite = Path(img["file_name"]).stem
        if cache.existe(unite) and not a.force:
            continue
        b, l, m = forward(session, input_name, _charger_image(corpus.chemin(img)), hw, a.tta)
        n_k += cache.ecrire(unite, b, l, m, a.floor)
        n_fait += 1
        if n_fait % 50 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{len(ids)}  {dt/n_fait:.3f} s/img  K moyen {n_k/n_fait:.0f}",
                  flush=True)
    dt = time.time() - t0
    print(f"OK {n_fait} passes en {dt:.1f} s"
          + (f" ({dt/n_fait:.3f} s/img, K moyen {n_k/n_fait:.0f})" if n_fait else ""))


# --------------------------------------------------------------------------------------
# E0 — plafond de rappel
# --------------------------------------------------------------------------------------

def cmd_e0(a) -> None:
    """Le rappel class-agnostic quand le seuil tend vers 0, sur les 200 requêtes.

    Sépare « le modèle ne propose JAMAIS la structure » (échec de représentation :
    ré-entraîner, aucun réglage n'aidera) de « il la propose à 0,12 » (échec de
    calibration : un seuil suffit). C'est la seule expérience dont le résultat peut
    invalider toute la suite de la matrice.
    """
    import onnxruntime as ort
    corpus = Corpus(Path(a.data))
    ids = _ids(corpus, a.subset)
    session, input_name, shape, meta, provider = charger_session(a.model, a.device)
    hw = (int(shape[3]), int(shape[2]))
    cle = cle_cache(a.model, ort.__version__, provider, "plugin_v1", hw, "id", a.floor)
    cache = Cache(Path(a.out) / "cache", cle)

    seuils = [s for s in (0.30, 0.20, 0.10, 0.05, 0.02) if s >= a.floor - 1e-9]
    regles = ["max", "objectness"]
    # Échantillon à pas constant : les ids COCO sont groupés par zone, prendre les N
    # premiers ne donnerait qu'une seule zone.
    tous = [i for i in ids if corpus.anns.get(i) and cache.existe(Path(
        corpus.images[i]["file_name"]).stem)]
    annotes = tous[::max(1, len(tous) // a.limite)][:a.limite]
    print(f"E0 sur {len(annotes)} images annotees, plancher de cache {a.floor}, "
          f"seuils {seuils}")

    global _CTX_E0
    _CTX_E0 = dict(corpus=corpus, cache=cache, seuils=seuils, regles=regles, tau=a.tau)
    acc = {(r, s): [] for r in regles for s in seuils}
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init_e0,
                             initargs=(a.data, str(Path(a.out) / "cache"), cle,
                                       seuils, regles, a.tau)) as ex:
        for res in ex.map(_traiter_e0, annotes, chunksize=4):
            if res is None:
                continue
            for k, v in res.items():
                acc[k].append(v)

    print(f"\nE0 — plafond de rappel  ({len(acc[(regles[0], seuils[0])])} images, "
          f"tau={a.tau} m)\n")
    print(f"{'regle':<12}{'seuil':>7}{'completude':>12}{'correction':>12}{'F1_len':>9}")
    res = {}
    for r in regles:
        for s in seuils:
            g = M.agreger_ccq(acc[(r, s)], a.tau)
            res[f"{r}@{s}"] = g
            print(f"{r:<12}{s:>7.2f}{g['completude']:>12.3f}"
                  f"{g['correction']:>12.3f}{g['f1_len']:>9.3f}")
    out = Path(a.out) / "e0_plafond_rappel.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_images": len(annotes), "tau_m": a.tau,
                               "plancher_cache": a.floor, "resultats": res},
                              indent=2), encoding="utf-8")
    print(f"\n-> {out}")


_CTX_E0: dict = {}


def _init_e0(data, cache_dir, cle, seuils, regles, tau):
    _CTX_E0.update(corpus=Corpus(Path(data)), cache=Cache(Path(cache_dir), cle),
                   seuils=seuils, regles=regles, tau=tau)


def _traiter_e0(iid: int):
    c = _CTX_E0
    corpus, cache = c["corpus"], c["cache"]
    img = corpus.images[iid]
    unite = Path(img["file_name"]).stem
    if not cache.existe(unite):
        return None
    d = cache.lire_brut(unite)
    h, w = img["height"], img["width"]
    gt = M.masque_coco(corpus.anns.get(iid, []), h, w)
    cote = M.CoteGT(gt)
    sig = 1.0 / (1.0 + np.exp(-d["logits"].astype(np.float64)))
    if sig.size == 0:
        return None
    scores = {"max": sig.max(axis=1),
              "objectness": 1.0 - np.prod(1.0 - sig[:, :5], axis=1)}
    # UNE montée en résolution par requête, réutilisée par tous les seuils et règles.
    bins = [cv2.resize(1.0 / (1.0 + np.exp(-d["masks"][q].astype(np.float32))),
                       (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5
            for q in range(d["masks"].shape[0])]
    out = {}
    for r in c["regles"]:
        for s in c["seuils"]:
            pred = np.zeros((h, w), bool)
            for q in np.nonzero(scores[r] >= s)[0]:
                pred |= bins[q]
            out[(r, s)] = M.ccq_prepare(pred, cote, c["tau"])
    return out


# --------------------------------------------------------------------------------------
# Évaluation d'une config (niveau A)
# --------------------------------------------------------------------------------------

def verifier_seuils_vs_plancher(grille: Sequence[tuple], plancher: float) -> None:
    """Refuse un seuil de confiance SOUS le plancher du cache.

    Le cache ne conserve que les requêtes dont le score dépasse le plancher. Une config
    demandant un seuil inférieur ne voit donc rien de plus, et rend un résultat
    RIGOUREUSEMENT identique à celui du plancher — sans aucun signal d'erreur.
    Constaté en vrai : à plancher 0,15, les configs 0,15 / 0,12 / 0,10 de l'ancien modèle
    donnaient les mêmes 0,5458 / 0,517 / 0,578 / 26,3, ce qui faisait conclure à tort à un
    optimum intérieur alors que le balayage était clippé. Mieux vaut échouer bruyamment.
    """
    # Le minimum EFFECTIF, seuils par classe compris : une surcharge de classe sous le
    # plancher serait clippée aussi silencieusement qu'un seuil global.
    from tools.bench.decode import _seuil_min
    fautives = {n: _seuil_min(p) for n, p in grille if _seuil_min(p) < plancher - 1e-9}
    if fautives:
        raise SystemExit(
            f"seuil(s) sous le plancher de cache {plancher} : {fautives}\n"
            f"Ces configs rendraient un resultat identique a celui du plancher, sans le "
            f"signaler. Soit les retirer, soit relancer `forward` avec --floor plus bas "
            f"(ce qui invalide le cache : le plancher fait partie de la cle).")


def fenetres(h: int, w: int, p: Params) -> List[list]:
    """Fenêtres SAHI du plugin, avec déduplication optionnelle."""
    from pipeline.cv.sahi_lite import get_slice_bboxes
    bb = get_slice_bboxes(h, w, p.sahi_slice, p.sahi_slice, p.sahi_overlap, p.sahi_overlap)
    if not p.sahi_dedup:
        return bb
    vus, out = set(), []
    for b in bb:
        t = tuple(b)
        if t not in vus:
            vus.add(t)
            out.append(b)
    return out


def slices_niveau_a(sl: SliceOut, w: int, h: int, p: Params) -> List[SliceOut]:
    """Reconstruit la liste de tuiles SAHI vue par le plugin, depuis UNE entrée de cache.

    Au niveau A toutes les fenêtres retombent sur [0,0,W,H] : le forward est identique,
    donc on rejoue la même sortie cachée autant de fois que le plugin la calcule. C'est
    exact (ONNX est déterministe à entrée identique) et ça évite 4x le coût GPU.
    """
    bboxes = fenetres(h, w, p)
    plein = [b for b in bboxes if b == [0, 0, w, h]]
    if len(plein) != len(bboxes):
        raise RuntimeError(
            f"niveau A : {len(bboxes)-len(plein)} fenetre(s) ne couvrent pas l'image "
            f"entiere ({bboxes[:3]}...) — un vrai forward par fenetre est necessaire")
    return [SliceOut(0, 0, w, h, sl.boxes, sl.logits, sl.masks, sl.qidx, sl.logits_full)
            for _ in bboxes]


_CTX: dict = {}


def _init_worker(data: str, cache_dir: str, cle: str, params_ser: list, tau: float) -> None:
    # Chaque worker est déjà un process : laisser cv2/BLAS ouvrir leurs propres pools de
    # threads ne fait que du changement de contexte contre les autres workers.
    cv2.setNumThreads(1)
    _CTX.update(corpus=Corpus(Path(data)), cache=Cache(Path(cache_dir), cle),
                params=[Params(**d) for d in params_ser], tau=tau)


def _traiter_image(iid: int) -> Optional[tuple]:
    """Une image, TOUTES les configs.

    C'est l'inversion qui rend la campagne possible : le squelette et la carte de
    distance de la GT ne dépendent pas de la config, donc ils se calculent une fois
    par image au lieu d'une fois par (image, config) — 25x le budget sinon.
    """
    corpus, cache, params, tau = _CTX["corpus"], _CTX["cache"], _CTX["params"], _CTX["tau"]
    img = corpus.images[iid]
    unite = Path(img["file_name"]).stem
    if not cache.existe(unite):
        return None
    h, w = img["height"], img["width"]
    sl = cache.lire(unite, 0, 0, w, h)
    anns = corpus.anns.get(iid, [])
    gt_masque = M.masque_coco(anns, h, w)
    cote_gt = M.CoteGT(gt_masque) if anns else None

    sorties = []
    for p in params:
        dets = decoder(slices_niveau_a(sl, w, h, p), w, h, w, h, p)
        pred = M.masque_detections(dets, h, w)
        if anns:
            c = M.ccq_prepare(pred, cote_gt, tau)
            c["pixel"] = M.metriques_pixel(pred, gt_masque)
        else:
            c = None
        sorties.append((c, len(dets)))
    return iid, bool(anns), len(anns), sorties


def agreger(resultats: list, params: Sequence[Params], tau_m: float,
            details: Optional[list] = None) -> List[dict]:
    n_cfg = len(params)
    sorties = []
    for k in range(n_cfg):
        parts, pix, neg, detail = [], [], [], []
        n_pred = n_gt = 0
        aire_km2 = 0.0
        n_img = 0
        for iid, positif, n_ann, s in resultats:
            c, nd = s[k]
            n_pred += nd
            n_gt += n_ann
            n_img += 1
            aire_km2 += (648 * M.GSD_M) ** 2 / 1e6
            if positif:
                parts.append(c)
                pix.append(c["pixel"])
                detail.append((iid, c["len_gt_m"], c["len_pred_m"],
                               c["len_tp_gt_m"], c["len_tp_pred_m"], nd))
            else:
                neg.append(nd)
                detail.append((iid, 0.0, 0.0, 0.0, 0.0, nd))
        g = M.agreger_ccq(parts, tau_m)
        tp = sum(x["tp"] for x in pix); fp = sum(x["fp"] for x in pix)
        fn = sum(x["fn"] for x in pix)
        g.update({
            "n_images": n_img, "n_pred": n_pred, "n_gt": n_gt, "aire_km2": aire_km2,
            "polygones_par_km2": n_pred / aire_km2 if aire_km2 else float("nan"),
            "pixel_precision": tp / (tp + fp) if tp + fp else float("nan"),
            "pixel_recall": tp / (tp + fn) if tp + fn else float("nan"),
            "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
            "fragmentation": M.agreger_frag(parts),
            "n_negatifs": len(neg),
            "fp_par_km2_negatifs": (sum(neg) / (len(neg) * (648 * M.GSD_M) ** 2 / 1e6)
                                    if neg else float("nan")),
        })
        sorties.append(g)
        if details is not None:
            details.append(np.asarray(detail, dtype=np.float64))
    return sorties


def cmd_sweep(a) -> None:
    import onnxruntime as ort
    import yaml
    corpus = Corpus(Path(a.data))
    ids = _ids(corpus, a.subset)
    session_meta = json.loads((Path(a.out) / "cache" / a.cle / "meta.json").read_text()) \
        if a.cle else None
    cle = a.cle or cle_cache(a.model, ort.__version__, a.provider, "plugin_v1",
                             (648, 648), "id", a.floor)
    cache = Cache(Path(a.out) / "cache", cle)

    configs = yaml.safe_load(Path(a.axes).read_text(encoding="utf-8"))
    base = Params(**(configs.get("base") or {}))
    grille: List[tuple] = [("base", base)]
    for nom, surcharge in (configs.get("configs") or {}).items():
        grille.append((nom, replace(base, **surcharge)))
    if configs.get("grille"):
        axes = configs["grille"]
        noms = sorted(axes)
        for combo in itertools.product(*(axes[k] for k in noms)):
            surcharge = dict(zip(noms, combo))
            etiquette = "_".join(f"{k}={v}" for k, v in surcharge.items())
            grille.append((etiquette, replace(base, **surcharge)))

    # Le nom du run porte le split : sans ça, la même grille d'axes rejouée sur valid
    # puis sur test se croirait déjà faite et sauterait silencieusement le second run.
    nom_run = f"{Path(a.axes).stem}__{Path(a.data).name}"
    dossier = Path(a.out) / "runs" / nom_run
    dossier.mkdir(parents=True, exist_ok=True)
    manifest = Path(a.out) / "manifest.jsonl"
    faits = set()
    if manifest.exists() and not a.force:
        for ligne in manifest.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(ligne)
                if d.get("_run") == nom_run:
                    faits.add(d["_config"])
            except Exception:
                pass

    grille = [(n, p) for n, p in grille if n not in faits]
    if not grille:
        print("rien a faire (tout est deja dans le manifest)")
        return
    noms_cfg = [n for n, _ in grille]
    params = [p for _, p in grille]
    print(f"{len(params)} configs x {len(ids)} images, {a.jobs} process")

    t0 = time.time()
    resultats_bruts = []
    ctx = (a.data, str(Path(a.out) / "cache"), cle, [p.cle() for p in params], a.tau)
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init_worker,
                             initargs=ctx) as ex:
        for i, r in enumerate(ex.map(_traiter_image, ids, chunksize=8)):
            if r is not None:
                resultats_bruts.append(r)
            if (i + 1) % 200 == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{len(ids)} images  {dt:.0f}s  "
                      f"(reste ~{dt/(i+1)*(len(ids)-i-1):.0f}s)", flush=True)

    details: list = []
    agreges = agreger(resultats_bruts, params, a.tau, details)
    resultats, detail_par_config = {}, {}
    secondes = round(time.time() - t0, 1)
    for nom, r, d in zip(noms_cfg, agreges, details):
        r.update({"_run": nom_run, "_config": nom,
                  "_params": dict(zip(noms_cfg, params))[nom].cle(),
                  "_secondes": secondes})
        resultats[nom] = r
        detail_par_config[nom] = d
        with open(manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{nom:<40} F1_len={r['f1_len']:.4f}  comp={r['completude']:.3f}"
              f"  corr={r['correction']:.3f}  poly/km2={r['polygones_par_km2']:.1f}"
              f"  frag={r['fragmentation']:.2f}", flush=True)

    if detail_par_config:
        np.savez_compressed(dossier / "detail_par_image.npz", **detail_par_config)
    out = dossier / "resultats.json"
    anciens = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    anciens.update(resultats)
    out.write_text(json.dumps(anciens, indent=2, ensure_ascii=False), encoding="utf-8")
    meilleurs = sorted(anciens.values(), key=lambda r: -r["f1_len"])[:5]
    print(f"\nTOP 5 par F1_len :")
    for r in meilleurs:
        print(f"  {r['f1_len']:.4f}  {r['_config']}")
    print(f"-> {out}")


# --------------------------------------------------------------------------------------

def _f1_len(a: np.ndarray, idx: np.ndarray) -> float:
    lg, lp = a[idx, 1].sum(), a[idx, 2].sum()
    tg, tp = a[idx, 3].sum(), a[idx, 4].sum()
    if not lg or not lp:
        return float("nan")
    c, r = tg / lg, tp / lp
    return 2 * c * r / (c + r) if (c + r) else 0.0


def cmd_bootstrap(a) -> None:
    """Bootstrap APPARIÉ par tuile de l'écart de F1_len contre la config de référence.

    Apparié = on rééchantillonne les tuiles UNE fois par itération et on recalcule les
    deux configs sur le même tirage. Les deux métriques étant très corrélées d'une config
    à l'autre, ça resserre l'intervalle d'un facteur ~3 par rapport à un bootstrap
    indépendant, gratuitement.
    """
    d = np.load(Path(a.out) / "runs" / a.run / "detail_par_image.npz")
    if a.ref not in d.files:
        raise SystemExit(f"config de reference {a.ref!r} absente ({len(d.files)} configs)")
    base = d[a.ref]
    n = base.shape[0]
    rng = np.random.default_rng(12345)
    tirages = rng.integers(0, n, size=(a.n_boot, n))
    idx_id = np.arange(n)

    f1_base = _f1_len(base, idx_id)
    lignes = []
    for cfg in d.files:
        arr = d[cfg]
        if arr.shape[0] != n or not np.array_equal(arr[:, 0], base[:, 0]):
            print(f"  (saut {cfg} : images differentes)")
            continue
        delta = np.array([_f1_len(arr, t) - _f1_len(base, t) for t in tirages])
        lo, hi = np.percentile(delta, [2.5, 97.5])
        d_obs = _f1_len(arr, idx_id) - f1_base
        signif = "oui" if (lo > 0 or hi < 0) else "-"
        lignes.append((d_obs, lo, hi, signif, cfg))

    lignes.sort(key=lambda t: -t[0])
    print(f"\nBootstrap apparie ({a.n_boot} tirages, {n} tuiles) — reference {a.ref!r} "
          f"F1_len={f1_base:.4f}\n")
    print(f"{'dF1_len':>9} {'IC95 bas':>10} {'IC95 haut':>10} {'signif':>7}  config")
    for d_obs, lo, hi, s, cfg in lignes:
        print(f"{d_obs:>+9.4f} {lo:>+10.4f} {hi:>+10.4f} {s:>7}  {cfg}")
    out = Path(a.out) / "runs" / a.run / "bootstrap.json"
    out.write_text(json.dumps(
        {"reference": a.ref, "f1_reference": f1_base, "n_tuiles": n, "n_boot": a.n_boot,
         "resultats": [{"config": c, "delta": dd, "ic95": [lo, hi], "significatif": s == "oui"}
                       for dd, lo, hi, s, c in lignes]}, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


def _detail_tuiles(synthese: dict, cfg: str, exclure: Sequence[str] = ()) -> Dict[str, tuple]:
    """{nom de tuile: (len_gt, len_pred, len_tp_gt, len_tp_pred)} pour une config.

    S'appuie sur la ventilation par tuile du niveau B, qui est exactement additive en
    longueur (cf. metrics.carte_longueur). C'est ce qui permet d'apparier deux modèles
    tuile par tuile alors qu'ils ne partagent NI la taille de fenêtre NI la taxonomie.
    """
    if cfg not in synthese:
        raise SystemExit(f"config {cfg!r} absente ({sorted(synthese)[:8]}…)")
    out: Dict[str, tuple] = {}
    for mos_id, v in synthese[cfg]["par_mosaique"].items():
        if any(z and z in mos_id for z in exclure):
            continue
        for nom, t in zip(v["tuiles"], v["par_tuile"]):
            out[f"{mos_id}/{nom}"] = (t["len_gt_m"], t["len_pred_m"],
                                      t["len_tp_gt_m"], t["len_tp_pred_m"])
    return out


def _f1_depuis(arr: np.ndarray, idx: np.ndarray) -> float:
    lg, lp = arr[idx, 0].sum(), arr[idx, 1].sum()
    tg, tp = arr[idx, 2].sum(), arr[idx, 3].sum()
    if not lg or not lp:
        return float("nan")
    c, r = tg / lg, tp / lp
    return 2 * c * r / (c + r) if (c + r) else 0.0


def cmd_comparer(a) -> None:
    """Bootstrap APPARIÉ par tuile entre deux modèles.

    Les deux modèles n'ont ni la même taxonomie ni la même géométrie de découpe : seule
    la longueur de linéaire retrouvée, ventilée sur une grille de tuiles commune, permet
    de les apparier. Rééchantillonner les tuiles une fois par itération et recalculer les
    deux modèles sur le même tirage resserre l'intervalle d'un facteur ~3 par rapport à
    un bootstrap indépendant.
    """
    exclure = [z for z in (a.hors_agregat or "").split(",") if z]
    sa = json.loads(Path(a.a).read_text(encoding="utf-8"))
    sb = json.loads(Path(a.b).read_text(encoding="utf-8"))
    da = _detail_tuiles(sa, a.cfg_a, exclure)
    db = _detail_tuiles(sb, a.cfg_b, exclure)

    communes = sorted(set(da) & set(db))
    if not communes:
        raise SystemExit("aucune tuile commune aux deux mesures")
    seules_a, seules_b = set(da) - set(db), set(db) - set(da)
    if seules_a or seules_b:
        print(f"[!] tuiles non partagees ignorees : {len(seules_a)} cote A, "
              f"{len(seules_b)} cote B")

    A = np.array([da[t] for t in communes], float)
    B = np.array([db[t] for t in communes], float)
    n = len(communes)
    idx = np.arange(n)
    f1_a, f1_b = _f1_depuis(A, idx), _f1_depuis(B, idx)

    rng = np.random.default_rng(20260729)
    tirages = rng.integers(0, n, size=(a.n_boot, n))
    delta = np.array([_f1_depuis(B, t) - _f1_depuis(A, t) for t in tirages])
    lo, hi = np.percentile(delta, [2.5, 97.5])
    signif = lo > 0 or hi < 0

    nom_a = sa[a.cfg_a].get("modele", "A")
    nom_b = sb[a.cfg_b].get("modele", "B")
    km2 = sum(v["aire_km2"] for k, v in sa[a.cfg_a]["par_mosaique"].items()
              if not any(z and z in k for z in exclure))
    print(f"\nComparatif apparie par tuile — {n} tuiles, {km2:.2f} km2, "
          f"{a.n_boot} tirages\n")
    print(f"  A  {nom_a} / {a.cfg_a}")
    print(f"     F1 longueur = {f1_a:.4f}")
    print(f"  B  {nom_b} / {a.cfg_b}")
    print(f"     F1 longueur = {f1_b:.4f}")
    print(f"\n  ecart B - A = {f1_b - f1_a:+.4f}   IC95 [{lo:+.4f} ; {hi:+.4f}]   "
          f"{'SIGNIFICATIF' if signif else 'non significatif (IC contient zero)'}")

    out = Path(a.out) / "comparatif_modeles.json"
    charge = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    charge[f"{nom_a}:{a.cfg_a}__vs__{nom_b}:{a.cfg_b}"] = {
        "modele_a": nom_a, "config_a": a.cfg_a, "f1_a": f1_a,
        "modele_b": nom_b, "config_b": a.cfg_b, "f1_b": f1_b,
        "delta": f1_b - f1_a, "ic95": [lo, hi], "significatif": bool(signif),
        "n_tuiles": n, "aire_km2": km2, "n_boot": a.n_boot,
        "mosaiques_exclues": exclure,
    }
    out.write_text(json.dumps(charge, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


# --------------------------------------------------------------------------------------
# Niveau B — mosaïques géoréférencées
# --------------------------------------------------------------------------------------

def noms_classes_modele(model_path: str) -> List[str]:
    """Noms de classes DU MODÈLE chargé, dans son propre ordre.

    À ne surtout pas prendre dans le COCO de test : celui-ci porte la taxonomie du
    corpus v2 (5 classes), alors qu'un autre modèle peut en avoir 3 dans un ordre
    différent. Les nommer avec la mauvaise table donnerait des couches correctement
    géométriques mais faussement étiquetées.
    """
    d = Path(model_path).parent.parent          # weights/best.onnx -> racine du modèle
    sidecar = Path(model_path).with_suffix(".json")
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("class_names"):
            return [str(x) for x in meta["class_names"]]
    txt = d / "classes.txt"
    if txt.exists():
        return [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    raise RuntimeError(f"noms de classes introuvables pour {model_path}")


def _geo_postprocess(dets: List[dict], mos, noms: Sequence[str], p: Params) -> List[dict]:
    """Passe les détections par le VRAI post-traitement géo du plugin."""
    from shapely.geometry import Polygon
    from pipeline.cv.postprocessing import postprocess_geo_detections

    par_classe: Dict[str, List[dict]] = {}
    for d in dets:
        xy = np.asarray(d["polygon"], np.float64).reshape(-1, 2)
        gx, gy = mos.geo(xy[:, 0] * mos.w, xy[:, 1] * mos.h)
        if len(gx) < 3:
            continue
        g = Polygon(np.stack([gx, gy], axis=1))
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        nom = noms[d["class_id"]] if d["class_id"] < len(noms) \
            else f"classe_{d['class_id']}"
        par_classe.setdefault(nom, []).append(
            {"geometry": g, "confidence": d["confidence"], "class_id": d["class_id"]})

    sortie = postprocess_geo_detections(
        par_classe, merge_buffer_m=p.geo_merge_buffer_m, min_area_m2=p.geo_min_area_m2,
        do_merge=p.geo_merge, do_remove_overlaps=p.geo_remove_overlaps,
        overlap_strategy=p.geo_overlap_strategy)
    # Le post-traitement fusionne des géométries : `class_id` peut disparaître des dicts
    # de sortie. On le rétablit depuis le nom de couche, qui lui est conservé — c'est ce
    # qui permet ensuite la ventilation par classe canonique.
    index = {n: i for i, n in enumerate(noms)}
    out: List[dict] = []
    for nom, lst in sortie.items():
        for d in lst:
            d.setdefault("class_id", index.get(nom, -1))
            d["class_name"] = nom
            out.append(d)
    return out


def _rasteriser_geo(dets_geo: List[dict], mos) -> np.ndarray:
    m = np.zeros((mos.h, mos.w), np.uint8)
    for d in dets_geo:
        g = d.get("geometry")
        if g is None or g.is_empty:
            continue
        for part in (g.geoms if g.geom_type.startswith("Multi") else [g]):
            if part.geom_type != "Polygon":
                continue
            xs, ys = np.asarray(part.exterior.coords).T
            px = np.round((xs - mos.xmin) / M.GSD_M).astype(np.int32)
            py = np.round((mos.ymax - ys) / M.GSD_M).astype(np.int32)
            cv2.fillPoly(m, [np.stack([px, py], axis=1)], 1)
            for anneau in part.interiors:
                xs, ys = np.asarray(anneau.coords).T
                px = np.round((xs - mos.xmin) / M.GSD_M).astype(np.int32)
                py = np.round((mos.ymax - ys) / M.GSD_M).astype(np.int32)
                cv2.fillPoly(m, [np.stack([px, py], axis=1)], 0)
    return m.astype(bool)


def cmd_niveaub(a) -> None:
    import onnxruntime as ort
    import yaml
    from tools.bench.data import TUILE_PX, composantes, parse_tuile
    from tools.bench.mosaic import (
        CANONIQUES, Mosaique, canonique_pour, choisir, gt_lignes,
    )

    corpus = Corpus(Path(a.data))
    noms_modele = noms_classes_modele(a.model)
    canon = canonique_pour(a.model)
    print(f"modele    : {Path(a.model).parent.parent.name}")
    print(f"classes   : {noms_modele}")
    print(f"canonique : { {noms_modele[i]: c for i, c in sorted(canon.items()) } }\n")
    tuiles = [t for t in (parse_tuile(i["file_name"]) for i in corpus.images.values()) if t]
    mosaiques = choisir(composantes(tuiles, min_tuiles=a.min_tuiles), par_zone=a.par_zone,
                        max_tuiles=a.max_tuiles)
    print(f"{len(mosaiques)} mosaiques :")
    for m in mosaiques:
        md = m.meta()
        print(f"  {m.id:<44} {md['n_tuiles']:>3} tuiles  {md['taille_px']}  "
              f"{md['aire_km2']:.2f} km2  remplissage {md['taux_remplissage']:.0%}")

    configs = yaml.safe_load(Path(a.axes).read_text(encoding="utf-8"))
    base = Params(**(configs.get("base") or {}))
    grille = [("base", base)] + [(n, replace(base, **s))
                                 for n, s in (configs.get("configs") or {}).items()]
    verifier_seuils_vs_plancher(grille, a.floor)

    session, input_name, shape, meta, provider = charger_session(a.model, a.device)
    hw = (int(shape[3]), int(shape[2]))
    cle = cle_cache(a.model, ort.__version__, provider, "plugin_v1", hw, "id", a.floor)
    cache = Cache(Path(a.out) / "cache_b", cle, meta={
        "modele": a.model, "ort": ort.__version__, "provider": provider,
        "prepro": "plugin_v1", "input_hw": list(hw), "tta": "id", "plancher": a.floor,
        "split": Path(a.data).name + "_mosaiques"})

    racine = Path(a.out) / "mosaiques"
    racine.mkdir(parents=True, exist_ok=True)

    # Reprise au niveau CONFIG. L'accumulation d'instances peut saturer la mémoire sur une
    # grande mosaïque à seuil bas (mesuré : l'ancien modèle, fenêtre 1032 px, dépasse
    # 7,4 Go à confiance 0,10) ; sans reprise, un OOM en fin de grille perdrait tout le
    # travail des configs déjà calculées.
    partiel = Path(a.out) / f"niveau_b_{Path(a.axes).stem}__{Path(a.model).parent.parent.name}.json"
    deja: Dict[str, dict] = {}
    if partiel.exists() and not a.force:
        anciennes = json.loads(partiel.read_text(encoding="utf-8"))
        deja = {n: b.get("par_mosaique", {}) for n, b in anciennes.items()}
        faites = [n for n, m in deja.items() if len(m) == len(mosaiques)]
        if faites:
            print(f"reprise : {len(faites)} config(s) deja completes, ignorees\n")
    resultats: Dict[str, dict] = {n: dict(m) for n, m in deja.items()}
    grille = [(n, p) for n, p in grille
              if len(deja.get(n, {})) < len(mosaiques)]
    if not grille:
        print("rien a recalculer")

    for mos in mosaiques:
        t0 = time.time()
        dm = racine / mos.id
        dm.mkdir(exist_ok=True)
        canvas, valide = mos.construire(Path(a.data))
        (dm / "mosaique.pgw").write_text(mos.pgw(), encoding="utf-8")
        (dm / "meta.json").write_text(json.dumps(mos.meta(), indent=2), encoding="utf-8")
        gpkg = Path(a.gpkg) / f"{mos.zone}_entites_l93_v2.gpkg"
        skel_gt, longueurs = gt_lignes(mos, gpkg)
        skel_gt &= valide
        cote_gt = M.CoteGT.depuis_squelette(skel_gt)
        # GT par classe canonique : identique pour les deux modèles, donc comparable.
        cotes_canon = {}
        for cl in CANONIQUES:
            sk, _ = gt_lignes(mos, gpkg, canonique=cl)
            sk &= valide
            if sk.any():
                cotes_canon[cl] = M.CoteGT.depuis_squelette(sk)
        # Emprises des tuiles en pixels de mosaïque : unités du bootstrap apparié.
        emprises = []
        for t in mos.tuiles:
            x, y = mos.px(t)
            emprises.append((y, y + TUILE_PX, x, x + TUILE_PX))
        print(f"\n{mos.id} : GT {cote_gt.longueur/1000:.2f} km de lineaire, "
              f"{cote_gt.n_composantes} segments, "
              f"canonique { {c: round(v.longueur/1000, 2) for c, v in cotes_canon.items()} } km",
              flush=True)

        # Forward sur l'union des fenêtres demandées par toutes les configs.
        besoins = set()
        for _, p in grille:
            for x0, y0, x1, y1 in fenetres(mos.h, mos.w, p):
                besoins.add((x0, y0, x1, y1))
        from PIL import Image as _I
        for i, (x0, y0, x1, y1) in enumerate(sorted(besoins)):
            unite = f"{mos.id}/{x0}_{y0}_{x1}_{y1}"
            if cache.existe(unite):
                continue
            b, l, mk = forward(session, input_name, canvas[y0:y1, x0:x1], hw, "id")
            cache.ecrire(unite, b, l, mk, a.floor)
            if (i + 1) % 25 == 0:
                print(f"    forward {i+1}/{len(besoins)}", flush=True)
        print(f"  {len(besoins)} fenetres, {time.time()-t0:.0f}s", flush=True)

        for nom, p in grille:
            if mos.id in resultats.get(nom, {}):
                continue
            bb = fenetres(mos.h, mos.w, p)
            slices = [cache.lire(f"{mos.id}/{x0}_{y0}_{x1}_{y1}", x0, y0, x1 - x0, y1 - y0)
                      for x0, y0, x1, y1 in bb]
            dets = decoder(slices, mos.w, mos.h, hw[0], hw[1], p)
            dets_geo = _geo_postprocess(dets, mos, noms_modele, p)
            pred = _rasteriser_geo(dets_geo, mos) & valide
            c = M.ccq_prepare(pred, cote_gt, a.tau)
            c.update({"n_pred": len(dets_geo), "n_pred_avant_geo": len(dets),
                      "aire_km2": float(valide.sum()) * M.GSD_M ** 2 / 1e6,
                      "n_fenetres": len(bb)})
            c["polygones_par_km2"] = c["n_pred"] / max(c["aire_km2"], 1e-9)

            # Ventilation par classe canonique : prédictions du modèle regroupées dans
            # l'espace commun aux deux taxonomies, contre la GT de la même classe.
            c["par_classe"] = {}
            for cl, cote in cotes_canon.items():
                ids = [i for i, v in canon.items() if v == cl]
                sous = [d for d in dets_geo if d.get("class_id") in ids]
                pc = _rasteriser_geo(sous, mos) & valide
                g = M.ccq_prepare(pc, cote, a.tau)
                g["n_pred"] = len(sous)
                c["par_classe"][cl] = g

            # Ventilation par tuile : additive en longueur, donc c'est elle qui rend
            # possible un intervalle de confiance apparié (3 mosaïques -> ~86 tuiles).
            c["par_tuile"] = M.ccq_decompose(pred, cote_gt, emprises, a.tau)
            c["tuiles"] = [t.nom for t in mos.tuiles]
            resultats.setdefault(nom, {})[mos.id] = c
            print(f"    {nom:<28} F1_len={c['f1_len']:.4f} comp={c['completude']:.3f} "
                  f"corr={c['correction']:.3f} poly/km2={c['polygones_par_km2']:.1f} "
                  f"({len(bb)} fen.)", flush=True)
            # Sauvegarde incrémentale : une config coûteuse ne doit pas être perdue si la
            # suivante fait sauter la mémoire.
            partiel.write_text(json.dumps(
                {n: {"par_mosaique": m} for n, m in resultats.items()},
                indent=2, ensure_ascii=False), encoding="utf-8")

    # Agrégat des mosaïques LOYALES par défaut : Haye et Rambouillet sont dans l'emprise
    # d'entraînement de l'ancien modèle (134/134 et 210/211 tuiles test), donc les y
    # inclure mélangerait mémorisation et généralisation dans un même chiffre.
    exclues = set((a.hors_agregat or "").split(",")) - {""}
    synthese = {}
    for nom, par_mos in resultats.items():
        loyales = [v for k, v in par_mos.items()
                   if not any(z and z in k for z in exclues)]
        g = M.agreger_ccq(loyales, a.tau)
        g["n_pred"] = sum(v["n_pred"] for v in loyales)
        g["aire_km2"] = sum(v["aire_km2"] for v in loyales)
        g["polygones_par_km2"] = g["n_pred"] / max(g["aire_km2"], 1e-9)
        g["fragmentation"] = M.agreger_frag(loyales)
        g["mosaiques_agregees"] = [k for k in par_mos
                                   if not any(z and z in k for z in exclues)]
        g["mosaiques_exclues"] = [k for k in par_mos
                                  if any(z and z in k for z in exclues)]
        g["par_classe"] = {
            cl: M.agreger_ccq([v["par_classe"][cl] for v in loyales
                               if cl in v.get("par_classe", {})], a.tau)
            for cl in CANONIQUES
            if any(cl in v.get("par_classe", {}) for v in loyales)
        }
        g["modele"] = Path(a.model).parent.parent.name
        g["par_mosaique"] = par_mos
        synthese[nom] = g
    # Un fichier par (grille d'axes, modèle) : sinon deux passes successives s'écrasent
    # silencieusement et le rapport ne montre plus que la dernière.
    out = Path(a.out) / f"niveau_b_{Path(a.axes).stem}__{Path(a.model).parent.parent.name}.json"
    out.write_text(json.dumps(synthese, indent=2, ensure_ascii=False), encoding="utf-8")
    ref = synthese[list(synthese)[0]]
    print(f"\n=== SYNTHESE NIVEAU B — agregat sur {len(ref['mosaiques_agregees'])} mosaique(s) ===")
    if ref["mosaiques_exclues"]:
        print(f"  (hors agregat, publiees a part : {', '.join(ref['mosaiques_exclues'])})")
    for nom, g in sorted(synthese.items(), key=lambda kv: -kv[1]["f1_len"]):
        pc = "  ".join(f"{cl[:4]}={v['f1_len']:.3f}" for cl, v in g["par_classe"].items())
        print(f"  {g['f1_len']:.4f}  comp={g['completude']:.3f} corr={g['correction']:.3f} "
              f"poly/km2={g['polygones_par_km2']:6.1f}  {pc}  {nom}")
    print(f"-> {out}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="tools.bench")
    ap.add_argument("--out", default=OUT_DEFAUT)
    ap.add_argument("--model", default=MODELE_DEFAUT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info"); p.add_argument("--data", required=True); p.set_defaults(f=cmd_info)

    p = sub.add_parser("subset")
    p.add_argument("--data", required=True); p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", default="bench-v1"); p.add_argument("--out", dest="out", required=True)
    p.set_defaults(f=cmd_subset)

    p = sub.add_parser("forward")
    p.add_argument("--data", required=True); p.add_argument("--subset")
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--tta", default="id"); p.add_argument("--floor", type=float, default=0.05)
    p.add_argument("--force", action="store_true")
    p.set_defaults(f=cmd_forward)

    p = sub.add_parser("e0")
    p.add_argument("--data", required=True); p.add_argument("--subset")
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--floor", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--limite", type=int, default=300)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.set_defaults(f=cmd_e0)

    p = sub.add_parser("sweep")
    p.add_argument("--data", required=True); p.add_argument("--subset")
    p.add_argument("--axes", required=True); p.add_argument("--cle")
    p.add_argument("--provider", default="CUDAExecutionProvider")
    p.add_argument("--floor", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.set_defaults(f=cmd_sweep)

    p = sub.add_parser("bootstrap")
    p.add_argument("--run", required=True)
    p.add_argument("--ref", default="base")
    p.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    p.set_defaults(f=cmd_bootstrap)

    p = sub.add_parser("comparer")
    p.add_argument("--a", required=True, help="niveau_b_*.json du modele A")
    p.add_argument("--b", required=True, help="niveau_b_*.json du modele B")
    p.add_argument("--cfg-a", required=True, dest="cfg_a")
    p.add_argument("--cfg-b", required=True, dest="cfg_b")
    p.add_argument("--hors-agregat", dest="hors_agregat", default="")
    p.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    p.set_defaults(f=cmd_comparer)

    p = sub.add_parser("niveaub")
    p.add_argument("--data", required=True)
    p.add_argument("--gpkg", required=True)
    p.add_argument("--axes", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--floor", type=float, default=0.15)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--par-zone", type=int, default=1, dest="par_zone")
    p.add_argument("--min-tuiles", type=int, default=16, dest="min_tuiles")
    p.add_argument("--max-tuiles", type=int, default=49, dest="max_tuiles")
    p.add_argument("--hors-agregat", dest="hors_agregat", default="",
                   help="fragments de nom de mosaique a MESURER mais exclure de l'agregat "
                        "(ex. les zones d'entrainement d'un modele), separes par des virgules")
    p.add_argument("--force", action="store_true",
                   help="recalculer les configs deja presentes dans le json de sortie")
    p.set_defaults(f=cmd_niveaub)

    a = ap.parse_args(argv)
    a.f(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
