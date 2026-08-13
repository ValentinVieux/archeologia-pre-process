# -*- coding: utf-8 -*-
"""Test post-processing 02 — hystérésis à deux seuils sur les instances.

Constat mesuré : les scores sont écrasés par la loss IA-BCE et la masse des
instances vit juste autour du seuil (confiance médiane des FP : 0,297). Une
instance sous le seuil de production n'est pas forcément du bruit : si elle
PROLONGE une détection sûre de la même classe, c'est probablement la suite de la
même structure — le mécanisme qui maintient la continuité des vaisseaux en
imagerie médicale (seuillage par hystérésis).

Règle (aveugle à la GT) :
  - sûres      : instances au seuil de PRODUCTION (0,25 ; talus_fosse 0,30 ;
                 chemin_creux 0,15) ;
  - candidates : instances entre `bas` et le seuil de production ;
  - propagation : une candidate est ACCEPTÉE si sa géométrie est à moins de D
                 mètres d'une géométrie déjà acceptée de la même classe
                 (BFS — une candidate acceptée peut en tirer une autre).
Le post-traitement géo de production (fusion 0,5 m, aire min 200 m²) s'applique
ensuite au lot accepté, comme en production.

Balayage : bas ∈ {0,15 ; 0,10} × D ∈ {1 m ; 5 m} + un CONTRÔLE par `bas`
(décodage à `bas`, on ne garde QUE les sûres) qui isole l'effet « pollution des
cartes argmax par les instances basses » de l'effet « candidates ajoutées ».

Sorties (D:\\pipeline_results\\bench\\postpro\\02_hysteresis\\) :
  metriques_avant_apres.json     global + par classe, par config, par mosaïque,
                                 + décomposition par tuile (bootstrap apparié)
  <mosaique>/comparatif.gpkg     avant_*, apres_* (config retenue), verite_terrain_*
                                 (attributs : confiance, ajoute_hysteresis)
  LISEZMOI.md (écrit après analyse)

    python -m tools.bench.postpro_hysteresis [--gpkg-config bas0.10_D5]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench import metrics as M                                    # noqa: E402
from tools.bench.cache import Cache, cle_cache                          # noqa: E402
from tools.bench.data import Corpus, TUILE_PX, composantes, parse_tuile  # noqa: E402
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
SORTIE = BENCH / "postpro" / "02_hysteresis"
TAU = 5.0

# Seuils de PRODUCTION (model_card 2026-08-13) par id de classe.
HAUT = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.30, 4: 0.15}
BAS_LISTE = (0.15, 0.10)
D_LISTE = (1.0, 5.0)

GEO = dict(geo_merge=True, geo_merge_buffer_m=0.5, geo_remove_overlaps=False,
           geo_min_area_m2=200.0)
COMMUN = dict(class_offset=0, n_classes=5, mask_cutoff=0.5, sahi_slice=648,
              sahi_overlap=0.4, fusion="max", min_area_px=10.0,
              boxes_normalisees=True, sahi_dedup=True, prob_float16=True)


def en_geo(det: dict, mos):
    """Polygone shapely EPSG:2154 depuis le polygone pixel du décodeur."""
    from shapely.geometry import Polygon
    xy = np.asarray(det["polygon"], np.float64).reshape(-1, 2)
    xs = mos.xmin + xy[:, 0] * M.GSD_M
    ys = mos.ymax - xy[:, 1] * M.GSD_M
    if len(xs) < 3:
        return None
    p = Polygon(zip(xs, ys))
    return p if not p.is_empty else None


def hysteresis(dets: List[dict], mos, bas: float, dist: float) -> List[dict]:
    """Sélection : sûres + candidates connectées (BFS, même classe, <= dist m)."""
    from shapely.strtree import STRtree
    sures, cands = [], []
    for d in dets:
        seuil_h = HAUT.get(d["class_id"], 0.25)
        if d["confidence"] >= seuil_h:
            sures.append(d)
        elif d["confidence"] >= bas:
            cands.append(d)
    if not cands:
        return sures
    acceptes = list(sures)
    for cid in set(d["class_id"] for d in cands):
        base = [d for d in acceptes if d["class_id"] == cid]
        pool = [d for d in cands if d["class_id"] == cid]
        if not base or not pool:
            continue
        geos = {}
        for d in base + pool:
            g = d.get("_geo")
            if g is None:
                g = en_geo(d, mos)
                d["_geo"] = g
        pool = [d for d in pool if d["_geo"] is not None]
        actifs = [d for d in base if d["_geo"] is not None]
        restants = pool
        frontiere = actifs
        while frontiere and restants:
            arbre = STRtree([d["_geo"] for d in frontiere])
            pris, reste = [], []
            for d in restants:
                hits = arbre.query(d["_geo"].buffer(dist))
                ok = any(d["_geo"].distance(frontiere[int(i)]["_geo"]) <= dist
                         for i in hits)
                (pris if ok else reste).append(d)
            acceptes.extend(pris)
            frontiere = pris
            restants = reste
    return acceptes


def main() -> int:
    import geopandas as gpd
    import onnxruntime as ort
    import pyogrio

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg-config", default="bas0.10_D5", dest="gpkg_config")
    a = ap.parse_args()

    corpus = Corpus(DATA)
    noms = noms_classes_modele(str(MODELE))
    tuiles = [t for t in (parse_tuile(i["file_name"]) for i in corpus.images.values()) if t]
    mosaiques = choisir(composantes(tuiles, min_tuiles=12), par_zone=1, max_tuiles=42)
    cle = cle_cache(str(MODELE), ort.__version__, "CPUExecutionProvider",
                    "plugin_v1", (648, 648), "id", 0.05)
    cache = Cache(BENCH / "cache_b", cle)
    SORTIE.mkdir(parents=True, exist_ok=True)

    resultats: Dict[str, dict] = {}
    export_gpkg: Dict[str, dict] = {}

    for mos in mosaiques:
        # côté GT (une fois)
        valide = np.zeros((mos.h, mos.w), bool)
        for t in mos.tuiles:
            x, y = mos.px(t)
            valide[y:y + TUILE_PX, x:x + TUILE_PX] = True
        zone_gpkg = GPKG_GT / f"{mos.zone}_entites_l93_v2.gpkg"
        sk_tout, _ = gt_lignes(mos, zone_gpkg)
        sk_tout &= valide
        cote_tout = M.CoteGT.depuis_squelette(sk_tout)
        cotes = {}
        for canon in ("parcellaire", "talus_fosse", "chemin_creux"):
            sk, _ = gt_lignes(mos, zone_gpkg, canonique=canon)
            sk &= valide
            cotes[canon] = M.CoteGT.depuis_squelette(sk) if sk.any() else None
        emprises = [(y, y + TUILE_PX, x, x + TUILE_PX)
                    for x, y in (mos.px(t) for t in mos.tuiles)]

        bb = fenetres(mos.h, mos.w, Params(confidence=0.25, **COMMUN))
        sl = [cache.lire(f"{mos.id}/{x0}_{y0}_{x1}_{y1}", x0, y0, x1 - x0, y1 - y0)
              for x0, y0, x1, y1 in bb]

        def mesurer(nom_cfg: str, dets_geo: List[dict]) -> None:
            entree = resultats.setdefault(nom_cfg, {"par_mosaique": {}})
            pred_tout = _rasteriser_geo(dets_geo, mos) & valide
            g = M.ccq_prepare(pred_tout, cote_tout, TAU)
            g["par_tuile"] = M.ccq_decompose(pred_tout, cote_tout, emprises, TAU)
            g["n_pred"] = len(dets_geo)
            g["aire_km2"] = mos.meta()["aire_km2"]
            par_classe = {}
            for canon, cote in cotes.items():
                if cote is None:
                    par_classe[canon] = None
                    continue
                sous = [d for d in dets_geo if COUCHES_CANONIQUES.get(
                    noms[d["class_id"]], noms[d["class_id"]]) == canon]
                par_classe[canon] = M.ccq_prepare(
                    _rasteriser_geo(sous, mos) & valide, cote, TAU)
            g["par_classe"] = par_classe
            entree["par_mosaique"][mos.id] = g

        # AVANT : décodage strict aux seuils de production
        p_avant = Params(confidence=0.25, confidence_par_classe={3: 0.30, 4: 0.15},
                         **COMMUN, **GEO)
        brut_avant = decoder(sl, mos.w, mos.h, 648, 648, p_avant)
        avant_geo = _geo_postprocess(brut_avant, mos, noms, p_avant)
        mesurer("avant_production", avant_geo)

        for bas in BAS_LISTE:
            # décodage UNIQUE à `bas` (les cartes argmax voient les instances basses)
            p_bas = Params(confidence=bas, confidence_par_classe={4: min(bas, 0.15)},
                           **COMMUN)          # géo appliqué APRÈS sélection
            p_geo = replace(p_bas, **GEO)
            brut = decoder(sl, mos.w, mos.h, 648, 648, p_bas)

            # CONTRÔLE : mêmes cartes, on ne garde que les sûres
            sures = [d for d in brut
                     if d["confidence"] >= HAUT.get(d["class_id"], 0.25)]
            mesurer(f"controle_bas{bas:g}", _geo_postprocess(sures, mos, noms, p_geo))

            for dist in D_LISTE:
                nom_cfg = f"bas{bas:g}_D{dist:g}"
                garde = hysteresis(brut, mos, bas, dist)
                dets_geo = _geo_postprocess(
                    [{k: v for k, v in d.items() if k != "_geo"} for d in garde],
                    mos, noms, p_geo)
                mesurer(nom_cfg, dets_geo)
                if nom_cfg == a.gpkg_config:
                    ids_surs = {id(d) for d in garde
                                if d["confidence"] >= HAUT.get(d["class_id"], 0.25)}
                    export_gpkg[mos.id] = {
                        "avant": avant_geo,
                        "apres": dets_geo,
                        "brut_garde": garde,
                    }
        print(f"{mos.id} : avant={len(avant_geo)} dets", flush=True)

    # ---- agrégat + impression ----
    def agrege(par_mos):
        parts = list(par_mos.values())
        g = M.agreger_ccq(parts, TAU)
        g["n_pred"] = sum(p["n_pred"] for p in parts)
        g["polygones_par_km2"] = g["n_pred"] / max(sum(p["aire_km2"] for p in parts), 1e-9)
        for canon in ("parcellaire", "talus_fosse", "chemin_creux"):
            sous = [p["par_classe"][canon] for p in parts if p["par_classe"][canon]]
            g[f"f1_{canon}"] = M.agreger_ccq(sous, TAU)["f1_len"] if sous else None
        return g

    synthese = {n: agrege(r["par_mosaique"]) for n, r in resultats.items()}
    print("\n=== SYNTHESE (agrégat 5 mosaïques) ===")
    print(f"{'config':<20}{'F1':>8}{'comp':>7}{'corr':>7}{'poly/km2':>9}"
          f"{'parc':>8}{'talus_f':>9}{'chemin':>8}")
    for n, s in sorted(synthese.items(), key=lambda kv: -kv[1]["f1_len"]):
        print(f"{n:<20}{s['f1_len']:>8.4f}{s['completude']:>7.3f}{s['correction']:>7.3f}"
              f"{s['polygones_par_km2']:>9.1f}{s['f1_parcellaire']:>8.4f}"
              f"{s['f1_talus_fosse']:>9.4f}{s['f1_chemin_creux']:>8.4f}")

    sortie_json = {
        "regle": "sures = seuils production ; candidates >= bas ; acceptation par "
                 "proximite <= D m d'une acceptee de meme classe (BFS) ; geo de "
                 "production ensuite",
        "synthese": {n: {k: v for k, v in s.items() if not isinstance(v, list)}
                     for n, s in synthese.items()},
        "detail": {n: {"par_mosaique": {m: {k: v for k, v in d.items()
                                            if k != "par_classe"}
                                        for m, d in r["par_mosaique"].items()}}
                   for n, r in resultats.items()},
    }
    (SORTIE / "metriques_avant_apres.json").write_text(
        json.dumps(sortie_json, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    # ---- GPKG avant/après ----
    for mos in mosaiques:
        exp = export_gpkg.get(mos.id)
        if not exp:
            continue
        dossier = SORTIE / mos.id
        dossier.mkdir(exist_ok=True)
        gpkg = dossier / "comparatif.gpkg"
        premier = True
        for prefixe, dets_geo in (("avant", exp["avant"]), ("apres", exp["apres"])):
            couches: Dict[str, list] = {}
            for d in dets_geo:
                nom = noms[d["class_id"]]
                couches.setdefault(f"{prefixe}_{nom}", []).append(d)
            for nom_c, items in sorted(couches.items()):
                gdf = gpd.GeoDataFrame(
                    {"confiance": [d.get("confidence") for d in items],
                     "ajoute_hysteresis": [
                         bool(d.get("confidence", 1.0) <
                              HAUT.get(d.get("class_id", 0), 0.25))
                         for d in items]},
                    geometry=[d["geometry"] for d in items], crs="EPSG:2154")
                gdf.to_file(gpkg, layer=nom_c, driver="GPKG",
                            mode="w" if premier else "a")
                premier = False
        vis = BENCH / "visuel" / mos.id
        src = vis / "comparatif.gpkg"
        for nom_c, _ in pyogrio.list_layers(src):
            if nom_c.startswith("verite_terrain_"):
                gpd.read_file(src, layer=nom_c).to_file(
                    gpkg, layer=nom_c, driver="GPKG", mode="a")
        for f in ("fond_LD.png", "fond_LD.pgw", "fond_LD.prj"):
            if (vis / f).exists() and not (dossier / f).exists():
                shutil.copy2(vis / f, dossier / f)
    print(f"\n-> {SORTIE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
