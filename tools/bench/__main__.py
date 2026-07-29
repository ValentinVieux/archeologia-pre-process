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


# --------------------------------------------------------------------------------------
# Niveau B — mosaïques géoréférencées
# --------------------------------------------------------------------------------------

def _geo_postprocess(dets: List[dict], mos, corpus: Corpus, p: Params) -> List[dict]:
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
        nom = corpus.noms_classes[d["class_id"]] if d["class_id"] < len(corpus.noms_classes) \
            else f"classe_{d['class_id']}"
        par_classe.setdefault(nom, []).append({"geometry": g, "confidence": d["confidence"]})

    sortie = postprocess_geo_detections(
        par_classe, merge_buffer_m=p.geo_merge_buffer_m, min_area_m2=p.geo_min_area_m2,
        do_merge=p.geo_merge, do_remove_overlaps=p.geo_remove_overlaps,
        overlap_strategy=p.geo_overlap_strategy)
    return [d for lst in sortie.values() for d in lst]


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
    from pipeline.cv.sahi_lite import get_slice_bboxes
    from tools.bench.data import composantes, parse_tuile
    from tools.bench.mosaic import Mosaique, choisir, gt_lignes

    corpus = Corpus(Path(a.data))
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

    session, input_name, shape, meta, provider = charger_session(a.model, a.device)
    hw = (int(shape[3]), int(shape[2]))
    cle = cle_cache(a.model, ort.__version__, provider, "plugin_v1", hw, "id", a.floor)
    cache = Cache(Path(a.out) / "cache_b", cle, meta={
        "modele": a.model, "ort": ort.__version__, "provider": provider,
        "prepro": "plugin_v1", "input_hw": list(hw), "tta": "id", "plancher": a.floor,
        "split": Path(a.data).name + "_mosaiques"})

    racine = Path(a.out) / "mosaiques"
    racine.mkdir(parents=True, exist_ok=True)
    resultats: Dict[str, dict] = {}

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
        print(f"\n{mos.id} : GT {cote_gt.longueur/1000:.2f} km de lineaire, "
              f"{cote_gt.n_composantes} segments", flush=True)

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
            bb = fenetres(mos.h, mos.w, p)
            slices = [cache.lire(f"{mos.id}/{x0}_{y0}_{x1}_{y1}", x0, y0, x1 - x0, y1 - y0)
                      for x0, y0, x1, y1 in bb]
            dets = decoder(slices, mos.w, mos.h, hw[0], hw[1], p)
            dets_geo = _geo_postprocess(dets, mos, corpus, p)
            pred = _rasteriser_geo(dets_geo, mos) & valide
            c = M.ccq_prepare(pred, cote_gt, a.tau)
            c.update({"n_pred": len(dets_geo), "n_pred_avant_geo": len(dets),
                      "aire_km2": float(valide.sum()) * M.GSD_M ** 2 / 1e6,
                      "n_fenetres": len(bb)})
            c["polygones_par_km2"] = c["n_pred"] / max(c["aire_km2"], 1e-9)
            resultats.setdefault(nom, {})[mos.id] = c
            print(f"    {nom:<28} F1_len={c['f1_len']:.4f} comp={c['completude']:.3f} "
                  f"corr={c['correction']:.3f} poly/km2={c['polygones_par_km2']:.1f} "
                  f"({len(bb)} fen.)", flush=True)

    synthese = {}
    for nom, par_mos in resultats.items():
        g = M.agreger_ccq(list(par_mos.values()), a.tau)
        g["n_pred"] = sum(v["n_pred"] for v in par_mos.values())
        g["aire_km2"] = sum(v["aire_km2"] for v in par_mos.values())
        g["polygones_par_km2"] = g["n_pred"] / max(g["aire_km2"], 1e-9)
        g["fragmentation"] = M.agreger_frag(list(par_mos.values()))
        g["par_mosaique"] = par_mos
        synthese[nom] = g
    out = Path(a.out) / "niveau_b.json"
    out.write_text(json.dumps(synthese, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SYNTHESE NIVEAU B ===")
    for nom, g in sorted(synthese.items(), key=lambda kv: -kv[1]["f1_len"]):
        print(f"  {g['f1_len']:.4f}  comp={g['completude']:.3f} corr={g['correction']:.3f} "
              f"poly/km2={g['polygones_par_km2']:6.1f}  frag={g['fragmentation']:.2f}  {nom}")
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
    p.set_defaults(f=cmd_niveaub)

    a = ap.parse_args(argv)
    a.f(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
