# -*- coding: utf-8 -*-
"""Test post-processing 01 — réattribution de classe par POLARITÉ du relief.

Constat mesuré (anatomie 2026-08-13) : 58 % de la longueur GT des chemins creux est
couverte par une prédiction, mais 17 % seulement par une prédiction `chemin_creux` —
ils sont détectés sous un autre nom, surtout `parcellaire`. Or la distinction est
physique : un chemin creux est EN CREUX (sombre dans le Local Dominance), un
parcellaire/talus EN RELIEF (clair) — la polarité déjà exploitée par le recalage.

Règle (sans jamais regarder la GT) : pour chaque polygone `parcellaire` ou `talus`,
signature LD = médiane(anneau de contexte) − médiane(intérieur). Si elle dépasse un
seuil delta (le polygone est nettement plus sombre que son voisinage), la classe
devient `chemin_creux`. La GT ne sert qu'à MESURER avant/après.

Sorties (D:\\pipeline_results\\bench\\postpro\\01_reattribution_chemin_creux\\) :
  metriques_avant_apres.json    CCQ par classe avant/après, par delta balayé
  <mosaique>/comparatif.gpkg    couches avant_*, apres_*, verite_terrain_*
                                (attributs : confiance, delta_ld, reattribue)
  <mosaique>/fond_LD.png/.pgw/.prj (lien copié pour QGIS)
  LISEZMOI.md

    python -m tools.bench.postpro_reattribution [--deltas 4,6,8,12] [--delta-gpkg 8]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench import metrics as M                                    # noqa: E402
from tools.bench.cache import Cache, cle_cache                          # noqa: E402
from tools.bench.data import Corpus, composantes, parse_tuile           # noqa: E402
from tools.bench.decode import Params, run as decoder                   # noqa: E402
from tools.bench.mosaic import COUCHES_CANONIQUES, choisir, gt_lignes   # noqa: E402
from tools.bench.__main__ import (                                      # noqa: E402
    _geo_postprocess, _rasteriser_geo, fenetres, noms_classes_modele,
)

MODELE = Path(r"C:\Users\valen\AppData\Roaming\QGIS\QGIS3\profiles\default\python"
              r"\plugins\archeologia-pipeline\data\models\lineaires_seg_v2_1"
              r"\weights\best.onnx")
BENCH = Path(r"D:\pipeline_results\bench")
DATA = Path(r"D:\bench\data\test")
GPKG_GT = Path(r"D:\bench\vecteurs")
SORTIE = BENCH / "postpro" / "01_reattribution_chemin_creux"
TAU = 5.0

# Config de PRODUCTION actuelle : seuils par classe déployés le 2026-08-13
# (model_card.yaml : talus_fosse 0,30, chemin_creux 0,15, défaut 0,25).
PROD = Params(
    confidence=0.25, confidence_par_classe={3: 0.30, 4: 0.15},
    class_offset=0, n_classes=5, mask_cutoff=0.5,
    sahi_slice=648, sahi_overlap=0.4, fusion="max", min_area_px=10.0,
    boxes_normalisees=True, sahi_dedup=True, prob_float16=True,
    geo_merge=True, geo_merge_buffer_m=0.5, geo_remove_overlaps=False,
    geo_min_area_m2=200.0)

# Classes SOURCES de la réattribution : celles dont la signature attendue est
# CLAIRE (relief). fosse et talus_fosse sont légitimement sombres — les toucher
# réattribuerait des vrais positifs ; chemin_creux est la cible.
SOURCES = {"parcellaire", "talus"}
CIBLE = "chemin_creux"


def signature_ld(ld: np.ndarray, geo_vers_px, geom, marge: int = 30) -> float:
    """médiane(anneau ~10 m) − médiane(intérieur), sur un recadrage local.
    Positif = intérieur plus sombre que son voisinage = en creux."""
    h, w = ld.shape
    parts = [p for p in (geom.geoms if geom.geom_type.startswith("Multi") else [geom])
             if p.geom_type == "Polygon"]
    if not parts:
        return 0.0
    pxs = [np.array([geo_vers_px(x, y) for x, y in p.exterior.coords], np.int32)
           for p in parts]
    x0 = max(0, min(int(p[:, 0].min()) for p in pxs) - marge)
    y0 = max(0, min(int(p[:, 1].min()) for p in pxs) - marge)
    x1 = min(w, max(int(p[:, 0].max()) for p in pxs) + marge)
    y1 = min(h, max(int(p[:, 1].max()) for p in pxs) + marge)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    crop = ld[y0:y1, x0:x1]
    masque = np.zeros(crop.shape, np.uint8)
    for p in pxs:
        cv2.fillPoly(masque, [p - [x0, y0]], 1)
    if masque.sum() < 8:
        return 0.0
    anneau = cv2.dilate(masque, np.ones((21, 21), np.uint8)) - masque   # ~5 m de contexte
    vi = crop[masque > 0]
    va = crop[anneau > 0]
    if not len(vi) or not len(va):
        return 0.0
    return float(np.median(va)) - float(np.median(vi))


def masque_valide(mos) -> np.ndarray:
    """Emprise des tuiles présentes, sans relire les images (trous du split)."""
    from tools.bench.data import TUILE_PX
    v = np.zeros((mos.h, mos.w), bool)
    for t in mos.tuiles:
        x, y = mos.px(t)
        v[y:y + TUILE_PX, x:x + TUILE_PX] = True
    return v


def ccq_par_classe(mos, valide, cotes, dets: List[dict]) -> Dict[str, dict]:
    """CCQ par classe canonique — même rasterisation que cmd_niveaub."""
    out = {}
    for canon, cote in cotes.items():
        sous = [d for d in dets
                if COUCHES_CANONIQUES.get(d["classe"], d["classe"]) == canon]
        pred = _rasteriser_geo(sous, mos) & valide
        out[canon] = M.ccq_prepare(pred, cote, TAU) if cote is not None else None
    return out


def main() -> int:
    import geopandas as gpd
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", default="4,6,8,12",
                    help="seuils de polarité balayés (niveaux de gris LD)")
    ap.add_argument("--delta-gpkg", type=float, default=8.0, dest="delta_gpkg",
                    help="delta exporté dans les GPKG de comparaison")
    a = ap.parse_args()
    deltas = [float(x) for x in a.deltas.split(",")]

    corpus = Corpus(DATA)
    noms = noms_classes_modele(str(MODELE))
    tuiles = [t for t in (parse_tuile(i["file_name"]) for i in corpus.images.values()) if t]
    mosaiques = choisir(composantes(tuiles, min_tuiles=12), par_zone=1, max_tuiles=42)

    # cache d'inférence : celui d'e9/e10 (CPU, plancher 0,05)
    cle = cle_cache(str(MODELE), ort.__version__, "CPUExecutionProvider",
                    "plugin_v1", (648, 648), "id", 0.05)
    cache = Cache(BENCH / "cache_b", cle)

    SORTIE.mkdir(parents=True, exist_ok=True)
    resultats = {str(d): {"par_mosaique": {}} for d in deltas}
    avant_glob: Dict[str, List[dict]] = {}

    for mos in mosaiques:
        vis = BENCH / "visuel" / mos.id
        ld = cv2.imread(str(vis / "fond_LD.png"), cv2.IMREAD_GRAYSCALE)
        assert ld is not None and ld.shape == (mos.h, mos.w), f"fond LD incoherent {mos.id}"

        def geo_vers_px(x, y, mos=mos):
            return (int((x - mos.xmin) / M.GSD_M), int((mos.ymax - y) / M.GSD_M))

        bb = fenetres(mos.h, mos.w, PROD)
        sl = [cache.lire(f"{mos.id}/{x0}_{y0}_{x1}_{y1}", x0, y0, x1 - x0, y1 - y0)
              for x0, y0, x1, y1 in bb]
        dets = _geo_postprocess(
            decoder(sl, mos.w, mos.h, 648, 648, PROD), mos, noms, PROD)
        for d in dets:
            d["classe"] = noms[d["class_id"]]
            d["delta_ld"] = (signature_ld(ld, geo_vers_px, d["geometry"])
                             if d["classe"] in SOURCES else None)
        zone_gpkg = GPKG_GT / f"{mos.zone}_entites_l93_v2.gpkg"
        valide = masque_valide(mos)
        cotes = {}
        for canon in ("parcellaire", "talus_fosse", "chemin_creux"):
            sk, _ = gt_lignes(mos, zone_gpkg, canonique=canon)
            sk &= valide
            cotes[canon] = M.CoteGT.depuis_squelette(sk) if sk.any() else None
        avant = ccq_par_classe(mos, valide, cotes, dets)
        avant_glob[mos.id] = dets

        for delta in deltas:
            apres_dets = []
            n_re = 0
            for d in dets:
                d2 = dict(d)
                if d["classe"] in SOURCES and d["delta_ld"] is not None \
                        and d["delta_ld"] >= delta:
                    d2["classe"] = CIBLE
                    n_re += 1
                apres_dets.append(d2)
            apres = ccq_par_classe(mos, valide, cotes, apres_dets)
            resultats[str(delta)]["par_mosaique"][mos.id] = {
                "n_reattribues": n_re, "n_total": len(dets),
                "avant": {k: (v and {kk: v[kk] for kk in
                              ("len_tp_gt_m", "len_gt_m", "len_tp_pred_m", "len_pred_m")})
                          for k, v in avant.items()},
                "apres": {k: (v and {kk: v[kk] for kk in
                              ("len_tp_gt_m", "len_gt_m", "len_tp_pred_m", "len_pred_m")})
                          for k, v in apres.items()},
            }
        print(f"{mos.id} : {len(dets)} dets, "
              f"{sum(1 for d in dets if d['delta_ld'] is not None)} candidates", flush=True)

    # ---- agrégats + synthèse ----
    def f1(parts):
        lg = sum(p["len_gt_m"] for p in parts)
        lp = sum(p["len_pred_m"] for p in parts)
        comp = sum(p["len_tp_gt_m"] for p in parts) / max(lg, 1e-9)
        corr = sum(p["len_tp_pred_m"] for p in parts) / max(lp, 1e-9)
        return {"completude": round(comp, 4), "correction": round(corr, 4),
                "f1_len": round(2 * comp * corr / max(comp + corr, 1e-9), 4)}

    synthese = {}
    for delta, r in resultats.items():
        s = {}
        for canon in ("parcellaire", "talus_fosse", "chemin_creux"):
            av = [m["avant"][canon] for m in r["par_mosaique"].values() if m["avant"][canon]]
            ap_ = [m["apres"][canon] for m in r["par_mosaique"].values() if m["apres"][canon]]
            s[canon] = {"avant": f1(av), "apres": f1(ap_)}
        s["n_reattribues"] = sum(m["n_reattribues"] for m in r["par_mosaique"].values())
        synthese[delta] = s

    (SORTIE / "metriques_avant_apres.json").write_text(json.dumps(
        {"config_avant": "production 2026-08-13 (seuils par classe)",
         "regle": f"{'/'.join(sorted(SOURCES))} -> {CIBLE} si mediane(anneau 10 m) - "
                  f"mediane(interieur) >= delta (LD 8 bits)",
         "synthese": synthese, "detail": resultats}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print("\n=== SYNTHESE (agrégat 5 mosaïques) ===")
    print(f"{'delta':>6} {'reattr.':>8} {'chemin avant':>13} {'chemin apres':>13} "
          f"{'parc avant':>11} {'parc apres':>11}")
    for delta, s in synthese.items():
        print(f"{delta:>6} {s['n_reattribues']:>8} "
              f"{s['chemin_creux']['avant']['f1_len']:>13.4f} "
              f"{s['chemin_creux']['apres']['f1_len']:>13.4f} "
              f"{s['parcellaire']['avant']['f1_len']:>11.4f} "
              f"{s['parcellaire']['apres']['f1_len']:>11.4f}")

    # ---- GPKG de comparaison au delta retenu ----
    for mos in mosaiques:
        dets = avant_glob[mos.id]
        dossier = SORTIE / mos.id
        dossier.mkdir(exist_ok=True)
        couches: Dict[str, list] = {}
        for d in dets:
            couches.setdefault(f"avant_{d['classe']}", []).append((d, False))
            cl = d["classe"]
            re = (cl in SOURCES and d["delta_ld"] is not None
                  and d["delta_ld"] >= a.delta_gpkg)
            couches.setdefault(f"apres_{CIBLE if re else cl}", []).append((d, re))
        gpkg = dossier / "comparatif.gpkg"
        premier = True
        for nom, items in sorted(couches.items()):
            gdf = gpd.GeoDataFrame(
                {"confiance": [d.get("confidence") for d, _ in items],
                 "delta_ld": [d.get("delta_ld") for d, _ in items],
                 "reattribue": [bool(r) for _, r in items]},
                geometry=[d["geometry"] for d, _ in items], crs="EPSG:2154")
            gdf.to_file(gpkg, layer=nom, driver="GPKG", mode="w" if premier else "a")
            premier = False
        # GT + fond pour QGIS
        vis = BENCH / "visuel" / mos.id
        src = vis / "comparatif.gpkg"
        import pyogrio
        for nom, _ in pyogrio.list_layers(src):
            if nom.startswith("verite_terrain_"):
                gpd.read_file(src, layer=nom).to_file(gpkg, layer=nom, driver="GPKG", mode="a")
        for f in ("fond_LD.png", "fond_LD.pgw", "fond_LD.prj"):
            if (vis / f).exists() and not (dossier / f).exists():
                shutil.copy2(vis / f, dossier / f)
    print(f"\n-> {SORTIE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
