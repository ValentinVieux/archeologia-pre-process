# -*- coding: utf-8 -*-
"""Test post-processing 03 — les trous sont-ils aux joints des fenêtres SAHI ?

Hypothèse mécanique : une structure coupée par un bord de fenêtre peut y perdre ses
instances (masque tronqué, confiance affaiblie) ; les trous de couverture se
concentreraient alors près des joints de la grille SAHI (fenêtres 648 px, pas 389 px
soit 194,5 m à recouvrement 0,4). Si c'est vrai, une fusion des probabilités entre
fenêtres refermerait ces trous ; sinon, la piste tombe.

Méthode (GT parcellaire et talus_fosse, config production) :
  1. trous = segments de GT à >5 m de toute prédiction de la classe, ENTRE deux
     segments couverts de la même ligne (mêmes définitions que l'anatomie) ;
  2. distance du milieu de chaque trou au bord de fenêtre le plus proche
     (lignes x = x0|x1, y = y0|y1 de toutes les fenêtres intérieures) ;
  3. distribution NULLE : la même distance pour TOUS les points de la GT
     (échantillonnés à 1 m) — si les trous ne sont pas plus proches des joints
     que la GT elle-même, il n'y a pas d'effet de grille.

Sorties (D:\\pipeline_results\\bench\\postpro\\03_trous_grille_sahi\\) :
  stats.json                     distributions trous vs GT, par classe
  trous.gpkg                     couches trous_<classe> (longueur_m,
                                 dist_bord_fenetre_m), grille_fenetres (lignes)
  LISEZMOI.md (après analyse)

    python -m tools.bench.postpro_trous_sahi
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench import metrics as M                                    # noqa: E402
from tools.bench.cache import Cache, cle_cache                          # noqa: E402
from tools.bench.data import Corpus, composantes, parse_tuile           # noqa: E402
from tools.bench.decode import Params, run as decoder                   # noqa: E402
from tools.bench.mosaic import choisir                                  # noqa: E402
from tools.bench.__main__ import (                                      # noqa: E402
    _geo_postprocess, fenetres, noms_classes_modele,
)

MODELE = Path(r"C:\Users\valen\AppData\Roaming\QGIS\QGIS3\profiles\default\python"
              r"\plugins\archeologia-pipeline\data\models\lineaires_seg_v2_1"
              r"\weights\best.onnx")
BENCH = Path(r"D:\pipeline_results\bench")
DATA = Path(r"D:\bench\data\test")
GPKG_GT = Path(r"D:\bench\vecteurs")
SORTIE = BENCH / "postpro" / "03_trous_grille_sahi"
PAS = 1.0

PROD = Params(confidence=0.25, confidence_par_classe={3: 0.30, 4: 0.15},
              class_offset=0, n_classes=5, mask_cutoff=0.5, sahi_slice=648,
              sahi_overlap=0.4, fusion="max", min_area_px=10.0,
              boxes_normalisees=True, sahi_dedup=True, prob_float16=True,
              geo_merge=True, geo_merge_buffer_m=0.5, geo_remove_overlaps=False,
              geo_min_area_m2=200.0)

# couches de zone -> classe canonique mesurée
VERS_CANON = {"parcellaire": "parcellaire", "voie": "parcellaire",
              "talus": "talus_fosse", "fosse": "talus_fosse",
              "talus_fosse": "talus_fosse"}
CANONS = ("parcellaire", "talus_fosse")


def bords_fenetres_geo(mos, p) -> Dict[str, np.ndarray]:
    """Coordonnées géo des bords INTÉRIEURS de fenêtres (les bords de mosaïque
    ne peuvent pas expliquer un trou : il n'y a rien de l'autre côté)."""
    bb = fenetres(mos.h, mos.w, p)
    xs, ys = set(), set()
    for x0, y0, x1, y1 in bb:
        for x in (x0, x1):
            if 0 < x < mos.w:
                xs.add(mos.xmin + x * M.GSD_M)
        for y in (y0, y1):
            if 0 < y < mos.h:
                ys.add(mos.ymax - y * M.GSD_M)
    return {"x": np.array(sorted(xs)), "y": np.array(sorted(ys))}


def dist_bord(px_geo, py_geo, bords) -> float:
    dx = np.abs(bords["x"] - px_geo).min() if len(bords["x"]) else np.inf
    dy = np.abs(bords["y"] - py_geo).min() if len(bords["y"]) else np.inf
    return float(min(dx, dy))


def main() -> int:
    import geopandas as gpd
    import onnxruntime as ort
    import pyogrio
    from shapely.geometry import LineString
    from shapely.ops import substring
    from shapely.strtree import STRtree

    corpus = Corpus(DATA)
    noms = noms_classes_modele(str(MODELE))
    tuiles = [t for t in (parse_tuile(i["file_name"]) for i in corpus.images.values()) if t]
    mosaiques = choisir(composantes(tuiles, min_tuiles=12), par_zone=1, max_tuiles=42)
    cle = cle_cache(str(MODELE), ort.__version__, "CPUExecutionProvider",
                    "plugin_v1", (648, 648), "id", 0.05)
    cache = Cache(BENCH / "cache_b", cle)
    SORTIE.mkdir(parents=True, exist_ok=True)

    trous_par_canon: Dict[str, List[dict]] = {c: [] for c in CANONS}
    nul_par_canon: Dict[str, List[float]] = {c: [] for c in CANONS}
    lignes_grille = []

    for mos in mosaiques:
        bords = bords_fenetres_geo(mos, PROD)
        for xg in bords["x"]:
            lignes_grille.append(LineString([(xg, mos.ymin), (xg, mos.ymax)]))
        for yg in bords["y"]:
            lignes_grille.append(LineString([(mos.xmin, yg), (mos.xmax, yg)]))

        bb = fenetres(mos.h, mos.w, PROD)
        sl = [cache.lire(f"{mos.id}/{x0}_{y0}_{x1}_{y1}", x0, y0, x1 - x0, y1 - y0)
              for x0, y0, x1, y1 in bb]
        dets = _geo_postprocess(decoder(sl, mos.w, mos.h, 648, 648, PROD),
                                mos, noms, PROD)
        arbres = {}
        for canon in CANONS:
            geoms = [d["geometry"] for d in dets
                     if VERS_CANON.get(noms[d["class_id"]]) == canon]
            arbres[canon] = STRtree(geoms) if geoms else None

        zg = GPKG_GT / f"{mos.zone}_entites_l93_v2.gpkg"
        emprise_ok = lambda g: (mos.xmin <= g.centroid.x <= mos.xmax
                                and mos.ymin <= g.centroid.y <= mos.ymax)
        for couche, canon in VERS_CANON.items():
            if canon not in CANONS:
                continue
            try:
                gt = gpd.read_file(zg, layer=couche, bbox=(mos.xmin, mos.ymin,
                                                           mos.xmax, mos.ymax))
            except Exception:
                continue
            arbre = arbres[canon]
            for geom in gt.geometry:
                if geom is None or geom.is_empty:
                    continue
                parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
                for part in parts:
                    n = max(2, int(part.length / PAS) + 1)
                    dists_l = np.linspace(0.0, part.length, n)
                    pts = [part.interpolate(x) for x in dists_l]
                    if arbre is not None:
                        d5 = np.array([arbre.query_nearest(p, return_distance=True)[1][0]
                                       for p in pts])
                    else:
                        d5 = np.full(len(pts), np.inf)
                    couv = d5 <= 5.0
                    # distribution nulle : tous les points de GT
                    for p in pts[::5]:                     # 1 point sur 5 suffit
                        nul_par_canon[canon].append(dist_bord(p.x, p.y, bords))
                    if not couv.any():
                        continue
                    i0 = int(np.argmax(couv))
                    i1 = len(couv) - int(np.argmax(couv[::-1])) - 1
                    j = i0
                    while j <= i1:
                        if not couv[j]:
                            k = j
                            while k <= i1 and not couv[k]:
                                k += 1
                            seg = substring(part, dists_l[j], dists_l[min(k, len(dists_l) - 1)])
                            mid = seg.interpolate(0.5, normalized=True)
                            trous_par_canon[canon].append({
                                "geometry": seg, "mosaique": mos.id,
                                "longueur_m": float(seg.length),
                                "dist_bord_fenetre_m": dist_bord(mid.x, mid.y, bords),
                            })
                            j = k
                        else:
                            j += 1
        print(f"{mos.id} : trous cumules "
              + ", ".join(f"{c}={len(trous_par_canon[c])}" for c in CANONS), flush=True)

    # ---- stats ----
    stats = {}
    for canon in CANONS:
        trous = trous_par_canon[canon]
        dt = np.array([t["dist_bord_fenetre_m"] for t in trous])
        # pondération par la longueur : un trou de 100 m compte plus qu'un de 6 m
        poids = np.array([t["longueur_m"] for t in trous])
        nul = np.array(nul_par_canon[canon])
        def pct(v, w=None):
            if not len(v):
                return None
            if w is None:
                return {"n": int(len(v)), "mediane_m": float(np.median(v)),
                        "part_<=10m": float((v <= 10).mean()),
                        "part_<=20m": float((v <= 20).mean())}
            return {"n": int(len(v)), "mediane_m": float(np.median(v)),
                    "part_<=10m": float(((v <= 10) * w).sum() / w.sum()),
                    "part_<=20m": float(((v <= 20) * w).sum() / w.sum())}
        stats[canon] = {
            "trous": pct(dt),
            "trous_pondere_longueur": pct(dt, poids),
            "gt_entiere_nulle": pct(nul),
        }
    (SORTIE / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print("\n=== DISTANCE AU BORD DE FENETRE LE PLUS PROCHE ===")
    for canon, s in stats.items():
        print(f"\n{canon} :")
        for nom_l, v in s.items():
            if v:
                print(f"   {nom_l:<26} n={v['n']:>6}  mediane={v['mediane_m']:>6.1f} m"
                      f"  <=10m: {v['part_<=10m']:.1%}  <=20m: {v['part_<=20m']:.1%}")

    # ---- GPKG ----
    gpkg = SORTIE / "trous.gpkg"
    premier = True
    for canon in CANONS:
        trous = trous_par_canon[canon]
        if not trous:
            continue
        gdf = gpd.GeoDataFrame(
            {k: [t[k] for t in trous] for k in
             ("mosaique", "longueur_m", "dist_bord_fenetre_m")},
            geometry=[t["geometry"] for t in trous], crs="EPSG:2154")
        gdf.to_file(gpkg, layer=f"trous_{canon}", driver="GPKG",
                    mode="w" if premier else "a")
        premier = False
    gpd.GeoDataFrame(geometry=lignes_grille, crs="EPSG:2154").to_file(
        gpkg, layer="grille_fenetres", driver="GPKG", mode="a")
    print(f"\n-> {SORTIE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
