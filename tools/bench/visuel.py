"""Export VISIONNABLE du comparatif : couches QGIS + extraits superposés.

Les chiffres ne remplacent pas l'œil. Ce module produit, par mosaïque :

  visuel/<mosaique>/
      fond_LD.png / .pgw / .prj      le Local Dominance, géoréférencé EPSG:2154
      comparatif.gpkg                 couches `verite_terrain`, `ancien_*`, `nouveau_*`
      extraits/*.jpg                  vignettes superposées, centrées là où les deux
                                      modèles DIVERGENT le plus (pas au hasard)

Les GPKG portent leur CRS, donc ils se superposent directement aux rasters LD que
l'utilisateur a déjà dans QGIS — les mosaïques n'en sont que des découpes.

    python -m tools.bench.visuel --data /data/test --gpkg /vec \\
        --modele-a <onnx ancien> --axes-a <yaml> --cfg-a s1032_conf0.20 \\
        --modele-b <onnx nouveau> --axes-b <yaml> --cfg-b base
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench import metrics as M                                    # noqa: E402
from tools.bench.cache import Cache, charger_session, cle_cache         # noqa: E402
from tools.bench.data import Corpus, TUILE_PX, composantes, parse_tuile  # noqa: E402
from tools.bench.decode import Params, run as decoder                   # noqa: E402
from tools.bench.mosaic import canonique_pour, choisir, gt_lignes       # noqa: E402
from tools.bench.__main__ import (                                      # noqa: E402
    _geo_postprocess, _rasteriser_geo, fenetres, noms_classes_modele,
)

# BGR (cv2). Choisies pour rester lisibles sur un fond gris de relief.
COUL = {"gt": (60, 230, 0), "a": (200, 0, 255), "b": (255, 190, 0)}
LEGENDE = [("verite terrain", "gt"), ("ancien modele", "a"), ("nouveau modele", "b")]


def wkt_l93() -> str:
    from pyproj import CRS
    return CRS.from_epsg(2154).to_wkt()


def charger_config(axes: Path, cfg: str) -> Params:
    import yaml
    d = yaml.safe_load(axes.read_text(encoding="utf-8"))
    base = Params(**(d.get("base") or {}))
    if cfg == "base":
        return base
    surcharge = (d.get("configs") or {}).get(cfg)
    if surcharge is None:
        raise SystemExit(f"config {cfg!r} absente de {axes.name} "
                         f"({sorted((d.get('configs') or {}))})")
    return replace(base, **surcharge)


def detections(mos, canvas, cache: Cache, p: Params, hw, noms) -> List[dict]:
    bb = fenetres(mos.h, mos.w, p)
    sl = [cache.lire(f"{mos.id}/{x0}_{y0}_{x1}_{y1}", x0, y0, x1 - x0, y1 - y0)
          for x0, y0, x1, y1 in bb]
    return _geo_postprocess(decoder(sl, mos.w, mos.h, hw[0], hw[1], p), mos, noms, p)


def ecrire_gpkg(chemin: Path, couches: Dict[str, list]) -> None:
    """Une couche par (modèle, classe) + la vérité terrain, toutes en EPSG:2154."""
    import geopandas as gpd
    premier = True
    for nom, geoms in couches.items():
        if not geoms:
            continue
        gdf = gpd.GeoDataFrame(
            {"confiance": [g.get("confidence") for g in geoms]},
            geometry=[g["geometry"] for g in geoms], crs="EPSG:2154")
        gdf.to_file(chemin, layer=nom, driver="GPKG",
                    mode="w" if premier else "a")
        premier = False


def tuiles_divergentes(ja: dict, jb: dict, cfg_a: str, cfg_b: str, mos_id: str,
                       n: int) -> List[int]:
    """Indices des tuiles où les deux modèles diffèrent le plus en longueur retrouvée.

    Montrer des extraits au hasard serait peu informatif : on cible là où le désaccord
    est maximal, c'est-à-dire là où l'utilisateur peut juger par lui-même qui a raison.
    """
    va = ja.get(cfg_a, {}).get("par_mosaique", {}).get(mos_id)
    vb = jb.get(cfg_b, {}).get("par_mosaique", {}).get(mos_id)
    if not (va and vb):
        return list(range(min(n, 4)))
    ecarts = [abs(b["len_tp_gt_m"] - a["len_tp_gt_m"])
              for a, b in zip(va["par_tuile"], vb["par_tuile"])]
    return sorted(range(len(ecarts)), key=lambda i: -ecarts[i])[:n]


def dessiner(canvas: np.ndarray, mos, skel_gt: np.ndarray,
             da: List[dict], db: List[dict]) -> np.ndarray:
    img = canvas.copy()
    # Vérité terrain : le squelette rasterisé, dilaté pour rester visible.
    gt = cv2.dilate(skel_gt.astype(np.uint8), np.ones((3, 3), np.uint8))
    img[gt > 0] = COUL["gt"]
    for dets, cle in ((da, "a"), (db, "b")):
        for d in dets:
            g = d.get("geometry")
            if g is None or g.is_empty:
                continue
            for part in (g.geoms if g.geom_type.startswith("Multi") else [g]):
                if part.geom_type != "Polygon":
                    continue
                for anneau in [part.exterior, *part.interiors]:
                    xs, ys = np.asarray(anneau.coords).T
                    px = np.round((xs - mos.xmin) / M.GSD_M).astype(np.int32)
                    py = np.round((mos.ymax - ys) / M.GSD_M).astype(np.int32)
                    cv2.polylines(img, [np.stack([px, py], axis=1)], True,
                                  COUL[cle], 2)
    return img


def legender(vue: np.ndarray, titre: str) -> np.ndarray:
    h = 26 + 20 * len(LEGENDE)
    bandeau = np.full((h, vue.shape[1], 3), 20, np.uint8)
    cv2.putText(bandeau, titre, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 235), 1, cv2.LINE_AA)
    for i, (txt, cle) in enumerate(LEGENDE):
        y = 26 + 20 * i + 8
        cv2.line(bandeau, (10, y), (34, y), COUL[cle], 3)
        cv2.putText(bandeau, txt, (42, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (215, 215, 215), 1, cv2.LINE_AA)
    return np.vstack([bandeau, vue])


def main() -> int:
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gpkg", required=True)
    ap.add_argument("--out", default=os.environ.get("BENCH_OUT", "/out/bench"))
    ap.add_argument("--modele-a", required=True, dest="modele_a")
    ap.add_argument("--axes-a", required=True, dest="axes_a")
    ap.add_argument("--cfg-a", required=True, dest="cfg_a")
    ap.add_argument("--modele-b", required=True, dest="modele_b")
    ap.add_argument("--axes-b", required=True, dest="axes_b")
    ap.add_argument("--cfg-b", required=True, dest="cfg_b")
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--floor", type=float, default=0.15)
    ap.add_argument("--min-tuiles", type=int, default=12, dest="min_tuiles")
    ap.add_argument("--max-tuiles", type=int, default=42, dest="max_tuiles")
    ap.add_argument("--par-zone", type=int, default=1, dest="par_zone")
    ap.add_argument("--extraits", type=int, default=3,
                   help="nombre de vignettes par mosaique")
    a = ap.parse_args()

    corpus = Corpus(Path(a.data))
    tu = [t for t in (parse_tuile(i["file_name"]) for i in corpus.images.values()) if t]
    mosaiques = choisir(composantes(tu, min_tuiles=a.min_tuiles),
                        par_zone=a.par_zone, max_tuiles=a.max_tuiles)

    specs = []
    for cle, mod, axes, cfg in (("a", a.modele_a, a.axes_a, a.cfg_a),
                                ("b", a.modele_b, a.axes_b, a.cfg_b)):
        # `gpu` était codé en dur, hérité de l'image Docker CUDA : hors de ce conteneur
        # le module ne démarrait pas. Le défaut `auto` retombe sur le CPU sans rien dire,
        # et la clé de cache porte déjà le provider — aucun risque de mélanger les deux.
        sess, iname, shape, _meta, prov = charger_session(mod, a.device)
        hw = (int(shape[3]), int(shape[2]))
        cache = Cache(Path(a.out) / "cache_b",
                      cle_cache(mod, ort.__version__, prov, "plugin_v1", hw, "id", a.floor))
        specs.append({"cle": cle, "nom": Path(mod).parent.parent.name,
                      "p": charger_config(Path(axes), cfg), "cfg": cfg,
                      "hw": hw, "cache": cache, "noms": noms_classes_modele(mod),
                      "canon": canonique_pour(mod)})
        print(f"  {cle} = {specs[-1]['nom']} / {cfg}  (entree {hw[0]} px)")

    ja, jb = {}, {}
    for spec, cible in ((specs[0], "ja"), (specs[1], "jb")):
        f = Path(a.out) / (f"niveau_b_{Path(a.axes_a if spec['cle']=='a' else a.axes_b).stem}"
                           f"__{spec['nom']}.json")
        if f.exists():
            (ja if cible == "ja" else jb).update(json.loads(f.read_text(encoding="utf-8")))

    racine = Path(a.out) / "visuel"
    racine.mkdir(parents=True, exist_ok=True)
    index = []

    for mos in mosaiques:
        print(f"\n{mos.id}", flush=True)
        dm = racine / mos.id
        (dm / "extraits").mkdir(parents=True, exist_ok=True)
        canvas, valide = mos.construire(Path(a.data))
        skel_gt, _ = gt_lignes(mos, Path(a.gpkg) / f"{mos.zone}_entites_l93_v2.gpkg")
        skel_gt &= valide

        # Fond LD georeference : PNG + world file + CRS.
        cv2.imwrite(str(dm / "fond_LD.png"), canvas)
        (dm / "fond_LD.pgw").write_text(mos.pgw(), encoding="utf-8")
        (dm / "fond_LD.prj").write_text(wkt_l93(), encoding="utf-8")

        couches: Dict[str, list] = {}
        dets_par_cle = {}
        for spec in specs:
            d = detections(mos, canvas, spec["cache"], spec["p"], spec["hw"], spec["noms"])
            dets_par_cle[spec["cle"]] = d
            etiq = "ancien" if spec["cle"] == "a" else "nouveau"
            for g in d:
                cl = spec["canon"].get(g.get("class_id"), "autre")
                couches.setdefault(f"{etiq}_{cl}", []).append(g)
            print(f"    {etiq:<8} {len(d)} polygones", flush=True)

        # Verite terrain en LIGNES : c'est exactement ce qui a servi a mesurer.
        import geopandas as gpd
        from shapely.geometry import box as sbox
        from tools.bench.mosaic import COUCHES, COUCHES_CANONIQUES
        couches_gt: Dict[str, list] = {}
        emprise = sbox(mos.xmin, mos.ymin, mos.xmax, mos.ymax)
        for couche in COUCHES.get(mos.zone, {}):
            try:
                g = gpd.read_file(Path(a.gpkg) / f"{mos.zone}_entites_l93_v2.gpkg",
                                  layer=couche)
            except Exception:
                continue
            if g.empty:
                continue
            g = g[g.intersects(emprise)].clip(emprise)
            if g.empty:
                continue
            cl = COUCHES_CANONIQUES.get(couche, "autre")
            couches_gt.setdefault(f"verite_terrain_{cl}", []).extend(
                {"geometry": x, "confidence": None} for x in g.geometry if x is not None)

        gpkg_out = dm / "comparatif.gpkg"
        if gpkg_out.exists():
            gpkg_out.unlink()
        ecrire_gpkg(gpkg_out, {**couches_gt, **couches})
        print(f"    -> {gpkg_out.name} ({len(couches_gt) + len(couches)} couches)",
              flush=True)

        # Vignettes, centrees sur les tuiles ou les deux modeles divergent le plus.
        vue = dessiner(canvas, mos, skel_gt, dets_par_cle["a"], dets_par_cle["b"])
        idx = tuiles_divergentes(ja, jb, specs[0]["cfg"], specs[1]["cfg"], mos.id,
                                 a.extraits)
        for rang, i in enumerate(idx):
            t = mos.tuiles[i]
            x, y = mos.px(t)
            cx, cy = x + TUILE_PX // 2, y + TUILE_PX // 2
            demi = TUILE_PX          # fenetre de 2 tuiles de cote
            x0, y0 = max(0, cx - demi), max(0, cy - demi)
            x1, y1 = min(mos.w, cx + demi), min(mos.h, cy + demi)
            crop = vue[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (640, int(640 * crop.shape[0] / crop.shape[1])))
            img = legender(crop, f"{mos.zone} — {t.nom}")
            p = dm / "extraits" / f"{rang:02d}_{t.nom.replace('.png','')}.jpg"
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 72])
            index.append({"mosaique": mos.id, "zone": mos.zone, "tuile": t.nom,
                          "fichier": str(p.relative_to(racine)).replace("\\", "/")})
        print(f"    -> {len(idx)} extrait(s)", flush=True)

    (racine / "index.json").write_text(
        json.dumps({"extraits": index,
                    "modele_a": specs[0]["nom"], "config_a": specs[0]["cfg"],
                    "modele_b": specs[1]["nom"], "config_b": specs[1]["cfg"]},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {racine}  ({len(index)} extraits, {len(mosaiques)} mosaiques)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
