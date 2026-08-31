"""Auto-labellisation des dépressions circulaires d'une zone par le modèle run_rf_detr_1.

Décision utilisateur 2026-08-21 : les zones ajoutées à l'entraînement SANS annotations
`circular_depression` sont labellisées par le modèle fours/charbonnières
(checkpoint_best_ema, détection bbox) au seuil F1-max mesuré 0,395 — prédictions
gardées telles quelles, sans revue humaine.

Parité d'inférence avec l'entraînement : le LD 0,5 m est découpé en tuiles 1 km
alignées sur la grille km Lambert-93 (2000x2000 px, comme les dalles LHD des jpg
d'entraînement) ; rfdetr redimensionne en 704 comme Roboflow l'avait fait.
Cellules traitées : celles contenant >= 1 entité du GPKG de sélection + anneau de
voisines (même règle que telecharger_dalles_ign.py). Tuiles > 50 %% NoData sautées.

Usage (venv_adaf OBLIGATOIRE — GPU) :
  D:\\veille_irlande\\venv_adaf\\Scripts\\python.exe tools\\auto_label_depressions.py
      <ld.tif> <selection.gpkg> <sortie.gpkg> --poids <ckpt.pth>
      [--couche <nom>] [--seuil 0.395] [--resolution 704] [--anneau 1]
      [--classe-id 4] [--nom-classe circular_depression]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np


def cellules(gpkg, couche, anneau):
    import geopandas as gpd
    g = gpd.read_file(gpkg, layer=couche)
    if g.crs is None or g.crs.to_epsg() != 2154:
        sys.exit(f"{gpkg} : CRS {g.crs} != EPSG:2154")
    avec = {(int(p.x // 1000), int(p.y // 1000)) for p in g.geometry.centroid}
    out = set(avec)
    for x, y in avec:
        for dx in range(-anneau, anneau + 1):
            for dy in range(-anneau, anneau + 1):
                out.add((x + dx, y + dy))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ld", help="raster LD 8 bits 0,5 m EPSG:2154 (mosaïque une passe)")
    ap.add_argument("selection", help="GPKG d'entités : cellules à traiter (+ anneau)")
    ap.add_argument("sortie", help="GPKG des auto-labels")
    ap.add_argument("--poids", required=True)
    ap.add_argument("--couche", default=None)
    ap.add_argument("--seuil", type=float, default=0.395)
    ap.add_argument("--resolution", type=int, default=704)
    ap.add_argument("--anneau", type=int, default=1)
    ap.add_argument("--classe-id", type=int, default=4, help="id COCO de circular_depression dans le modèle")
    ap.add_argument("--nom-classe", default="circular_depression")
    a = ap.parse_args()

    import geopandas as gpd
    import rasterio
    import rasterio.windows
    from PIL import Image
    from shapely.geometry import box
    from rfdetr import RFDETRLarge

    modele = RFDETRLarge(pretrain_weights=a.poids, resolution=a.resolution)
    modele.optimize_for_inference()

    cibles = cellules(a.selection, a.couche, a.anneau)
    lignes, sautees = [], 0
    with rasterio.open(a.ld) as src:
        nodata = src.nodata if src.nodata is not None else 255
        px = src.res[0]
        if abs(px - 0.5) > 0.01:
            sys.exit(f"LD à {px} m/px : la parité d'inférence exige 0,5 m")
        traitees = 0
        for x, y in sorted(cibles):
            x0, y1 = x * 1000.0, (y + 1) * 1000.0  # coin NW de la cellule
            fen = rasterio.windows.from_bounds(x0, y1 - 1000, x0 + 1000, y1, src.transform)
            fen = fen.round_offsets().round_lengths()
            if fen.col_off < 0 or fen.row_off < 0 or \
               fen.col_off + fen.width > src.width or fen.row_off + fen.height > src.height:
                sautees += 1
                continue
            bande = src.read(1, window=fen)
            if bande.shape != (2000, 2000) or (bande == nodata).mean() > 0.5:
                sautees += 1
                continue
            im = Image.fromarray(np.stack([bande] * 3, axis=-1))
            d = modele.predict(im, threshold=a.seuil)
            for j in range(len(d)):
                if int(d.class_id[j]) != a.classe_id:
                    continue
                bx0, by0, bx1, by1 = (float(v) for v in d.xyxy[j])
                lignes.append({
                    "tuile": f"{x:04d}_{y + 1:04d}",
                    "score": round(float(d.confidence[j]), 4),
                    "geometry": box(x0 + bx0 * px, y1 - by1 * px, x0 + bx1 * px, y1 - by0 * px),
                })
            traitees += 1
            if traitees % 25 == 0:
                print(f"  {traitees} tuiles traitées, {len(lignes)} détections", flush=True)

    if not lignes:
        sys.exit("0 détection — vérifier LD/seuil/classe-id avant d'accepter ce résultat")
    Path(a.sortie).parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(lignes, crs="EPSG:2154")
    gdf.to_file(a.sortie, layer=f"{a.nom_classe}_auto", driver="GPKG")
    q = np.percentile(gdf["score"], [10, 50, 90])
    print(f"{len(gdf)} détections {a.nom_classe} sur {traitees} tuiles "
          f"({sautees} sautées) ; scores p10/p50/p90 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")
    print(f"Sorties : {a.sortie}")


if __name__ == "__main__":
    main()
