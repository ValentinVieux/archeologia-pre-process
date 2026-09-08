"""Inférence d'un modèle RF-DETR (seg ou détection) sur un RASTER entier (LD…) par fenêtres
glissantes, sortie GPKG EPSG du raster. Rééchantillonne le raster au --gsd du modèle
(ex. enclos = 1 m sur un LD à 0,5 m), fusionne les recouvrements par NMS (IoU bbox ≥ 0,5).

Sorties (couches ajoutées au GPKG, jamais écrasées) : <couche> = polygones (masque si seg,
sinon boîte) et <couche>_bbox = boîtes ; champs classe / score / retenu (score ≥ --seuil) /
aire_m2 / diam_eq_m / tuile. Le --seuil vient du metriques_eval.json canonique du modèle.

ATTENTION : venv_adaf (GPU) obligatoire :
    D:/veille_irlande/venv_adaf/Scripts/python.exe tools/inferer_raster.py <raster> <poids.pth> <gpkg> --couche enclos_detections --classes enclos --gsd 1.0 --seuil 0.41
"""
import argparse
import math
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tools.inferer_corpus import masque_vers_poly_px  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raster")
    ap.add_argument("poids")
    ap.add_argument("gpkg", help="GPKG de sortie (couches ajoutées ; refus si la couche existe)")
    ap.add_argument("--couche", required=True)
    ap.add_argument("--classes", nargs="+", required=True, help="noms des classes du modèle, dans l'ordre")
    ap.add_argument("--gsd", type=float, default=1.0, help="taille de pixel attendue par le modèle (m)")
    ap.add_argument("--resolution", type=int, default=648)
    ap.add_argument("--tuile", type=int, default=648, help="côté de la fenêtre (px au gsd du modèle)")
    ap.add_argument("--recouvrement", type=int, default=96, help="recouvrement des fenêtres (px)")
    ap.add_argument("--seuil", type=float, required=True, help="seuil F1-max du metriques_eval (champ retenu)")
    ap.add_argument("--plancher", type=float, default=0.15, help="score minimal conservé")
    ap.add_argument("--limite", type=int, default=0, help="debug : n fenêtres max")
    a = ap.parse_args()

    import geopandas as gpd
    import pyogrio
    from PIL import Image
    from rfdetr import RFDETR
    from shapely.geometry import Polygon, box

    if os.path.exists(a.gpkg) and any(l == a.couche for l, _ in pyogrio.list_layers(a.gpkg)):
        sys.exit(f"refus : la couche {a.couche} existe déjà dans {a.gpkg}")
    modele = RFDETR.from_checkpoint(a.poids, resolution=a.resolution)
    seg = bool(getattr(modele.model_config, "segmentation_head", False))
    modele.optimize_for_inference()
    src = rasterio.open(a.raster)
    nodata = src.nodata if src.nodata is not None else 255
    pas = (a.tuile - a.recouvrement) * a.gsd
    cote = a.tuile * a.gsd
    ncol = math.ceil((src.bounds.right - src.bounds.left - cote) / pas) + 1
    nrow = math.ceil((src.bounds.top - src.bounds.bottom - cote) / pas) + 1
    print(f"{'seg' if seg else 'det'} résolution {a.resolution}, gsd {a.gsd} m (raster {src.res[0]} m), "
          f"fenêtres {a.tuile} px pas {pas:.0f} m : {nrow} x {ncol} = {nrow * ncol}", flush=True)
    dets, n_fen, decal = [], 0, None
    for r in range(nrow):
        for c in range(ncol):
            if a.limite and n_fen >= a.limite:
                break
            x0 = src.bounds.left + c * pas
            y1 = src.bounds.top - r * pas
            b = (x0, y1 - cote, x0 + cote, y1)
            arr = src.read(1, window=from_bounds(*b, transform=src.transform), out_shape=(a.tuile, a.tuile),
                           resampling=Resampling.average, boundless=True, fill_value=nodata).astype("float32")
            valide = arr != nodata
            if valide.mean() < 0.05:
                continue
            n_fen += 1
            arr[~valide] = np.median(arr[valide])
            im = Image.fromarray(np.stack([arr.astype("uint8")] * 3, axis=-1))
            d = modele.predict(im, threshold=a.plancher)
            if len(d) == 0:
                continue
            if seg and d.mask is None:
                sys.exit("ERREUR : modèle segmentation sans masques prédits.")
            if decal is None:
                decal = 1 if (int(d.class_id.min()) >= 1 and int(d.class_id.max()) >= len(a.classes)) else 0
            for i in range(len(d)):
                idx = int(d.class_id[i]) - decal
                if not (0 <= idx < len(a.classes)):
                    continue
                bx = [float(v) for v in d.xyxy[i]]
                poly = masque_vers_poly_px(d.mask[i].astype(bool)) if seg else None
                if poly is None:
                    poly = [[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]]
                g = Polygon([(x0 + px * a.gsd, y1 - py * a.gsd) for px, py in poly]).buffer(0)
                if g.is_empty:
                    continue
                if g.geom_type == "MultiPolygon":
                    g = max(g.geoms, key=lambda q: q.area)
                dets.append({"classe": a.classes[idx], "score": round(float(d.confidence[i]), 4),
                             "tuile": f"r{r:03d}_c{c:03d}", "geometry": g,
                             "bbox": box(x0 + bx[0] * a.gsd, y1 - bx[3] * a.gsd, x0 + bx[2] * a.gsd, y1 - bx[1] * a.gsd)})
            if n_fen % 50 == 0:
                print(f"  {n_fen} fenêtres, {len(dets)} détections brutes", flush=True)
    print(f"{n_fen} fenêtres avec données, {len(dets)} détections brutes (plancher {a.plancher})", flush=True)
    # NMS global sur les boîtes (recouvrements de fenêtres), gloutonne par score, par classe
    gardees = []
    if dets:
        gdf = gpd.GeoDataFrame(dets, geometry="bbox", crs=src.crs).sort_values("score", ascending=False)
        sidx = gdf.sindex
        mort = np.zeros(len(gdf), dtype=bool)
        pos = {ix: k for k, ix in enumerate(gdf.index)}
        for k, (ix, row) in enumerate(gdf.iterrows()):
            if mort[k]:
                continue
            gardees.append(row)
            for j in sidx.query(row["bbox"], predicate="intersects"):
                jj = gdf.index[j]
                if pos[jj] <= k or mort[pos[jj]] or gdf.at[jj, "classe"] != row["classe"]:
                    continue
                inter = row["bbox"].intersection(gdf.at[jj, "bbox"]).area
                if inter / (row["bbox"].area + gdf.at[jj, "bbox"].area - inter) >= 0.5:
                    mort[pos[jj]] = True
    if not gardees:
        print("aucune détection"); return
    out = gpd.GeoDataFrame(gardees, crs=src.crs)
    out["retenu"] = out["score"] >= a.seuil
    for nom, geom in ((a.couche, "geometry"), (f"{a.couche}_bbox", "bbox")):
        g = gpd.GeoDataFrame(out[["classe", "score", "retenu", "tuile"]].copy(), geometry=list(out[geom]), crs=src.crs)
        g["aire_m2"] = g.geometry.area.round(1)
        g["diam_eq_m"] = (2 * np.sqrt(g.geometry.area / np.pi)).round(1)
        g.to_file(a.gpkg, layer=nom, driver="GPKG")
    print(f"écrit {len(out)} détections après NMS -> {a.gpkg} [{a.couche}, {a.couche}_bbox] : "
          f"retenues (>= {a.seuil}) {int(out['retenu'].sum())} ; classes {out['classe'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
