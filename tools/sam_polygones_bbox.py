"""Polygones SAM 2.1 (modèle par défaut, sans décodeur affiné) à partir d'une couche de
BOUNDING BOXES : chaque boîte, élargie de --marge m de chaque côté, sert d'invite (box
prompt) sur une imagette du raster LD centrée sur l'objet.

Sortie : couche polygone dans <gpkg_sortie> (même nombre d'entités que l'entrée, attributs
d'origine conservés + methode / sam_score / aire_m2 / diam_eq_m / iou_bbox). Quand SAM ne
propose aucun masque plausible (contenant le centre, aire entre --aire-min-ratio et
--aire-max-ratio de la boîte élargie), la boîte d'origine est reprise telle quelle
(methode = bbox_repli) pour que rien ne disparaisse.

ATTENTION : nécessite le venv SAM (torch CUDA), PAS le .venv du repo :
    D:/veille_irlande/venv_sam/Scripts/python.exe tools/sam_polygones_bbox.py
        <gpkg_entree> <couche> <ld.tif|vrt> <gpkg_sortie> [--couche-sortie X] [--marge 20]
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
from scipy import ndimage
from shapely.geometry import Polygon, box, shape

MODELE = "facebook/sam2.1-hiera-large"


def nettoyer(geom, lissage_m=2.0):
    """Composante principale, trous remplis, lissage rond (même recette que le corpus Irlande)."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not parts:
            return None
        geom = max(parts, key=lambda g: g.area)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    doux = Polygon(geom.exterior).buffer(lissage_m, join_style=1).buffer(-lissage_m, join_style=1)
    if doux.geom_type == "MultiPolygon":
        doux = max(doux.geoms, key=lambda g: g.area)
    return doux.simplify(0.8)


def choisir_masque(masks, scores, fenetre_px, tfm, boite, ratio_min, ratio_max):
    """Meilleur masque plausible parmi les candidats SAM : (polygone, score, motif).

    Jugé sur la composante principale du masque après fermeture 3x3 x2 (mouchetage comblé, vrais
    trous respectés), jamais sur un pixel brut : dans les fonds plats le masque brut est moucheté
    (dizaines à centaines de vides connectés) et le test du pixel central rejetait des masques
    corrects (Blois 2026-09-08 : 5 replis « centre_absent », centre à 1 px du masque).
    Plausible = contient le centre de la boîte, aire du polygone nettoyé entre ratio_min et
    ratio_max de la boîte élargie.
    """
    px0, py0, px1, py1 = fenetre_px
    clip = np.zeros_like(masks[0], dtype=bool)
    clip[max(0, int(py0)):int(np.ceil(py1)), max(0, int(px0)):int(np.ceil(px1))] = True
    best, rejets = None, []
    for m, s in zip(masks, scores):
        m = m & clip
        # fermeture 3x3 x2 (~1 m) : comble le mouchetage des fonds plats (vides de 1-3 px connectés
        # en labyrinthe) sans remplir un vrai anneau (trou de dizaines de px) — mesuré Blois 2026-09-08
        m |= ndimage.binary_closing(m, structure=np.ones((3, 3), bool), iterations=2)
        formes = [shape(g) for g, v in rasterio.features.shapes(m.astype("uint8"), transform=tfm) if v == 1]
        if not formes:
            rejets.append("masque_vide"); continue
        comp = max(formes, key=lambda q: q.area)
        if not comp.contains(boite.centroid):  # trous restants respectés : un anneau ne contient pas son centre
            rejets.append("centre_absent"); continue
        poly = nettoyer(comp.intersection(boite))
        if poly is not None:
            poly = poly.intersection(boite)  # le lissage peut ressortir de la boîte
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        if poly is None or poly.is_empty:
            rejets.append("lissage_vide"); continue
        if not poly.contains(boite.centroid):
            rejets.append("centre_absent_apres_lissage"); continue
        ratio = poly.area / boite.area
        if ratio < ratio_min:
            rejets.append(f"aire_{100 * ratio:.0f}pct"); continue
        if ratio > ratio_max:
            rejets.append("aire_trop_grande"); continue
        if best is None or s > best[1]:
            best = (poly, float(s))
    if best is None:
        return None, 0.0, "masques_rejetes:" + "+".join(rejets)
    return best[0], best[1], "ok"


class Segmenteur:
    def __init__(self, raster):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam2Model.from_pretrained(MODELE).to(self.device).eval()
        self.processor = Sam2Processor.from_pretrained(MODELE)
        self.src = rasterio.open(raster)
        self.res = self.src.res[0]
        self.nodata = 255 if self.src.nodata is None else self.src.nodata

    def imagette(self, cx, cy, demi_px):
        from PIL import Image
        r0, c0 = self.src.index(cx, cy)
        win = rasterio.windows.Window(c0 - demi_px, r0 - demi_px, 2 * demi_px, 2 * demi_px)
        a = self.src.read(1, window=win, boundless=True, fill_value=self.nodata).astype("float32")
        a[a == self.nodata] = np.nan
        if np.isfinite(a).mean() < 0.5:
            return None, None
        a = np.where(np.isnan(a), np.nanmedian(a), a).astype("uint8")
        return Image.fromarray(np.stack([a] * 3, axis=-1)), self.src.window_transform(win)

    def segmenter(self, bbox, marge, ratio_min, ratio_max, fenetre_min_px):
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ex0, ey0, ex1, ey1 = x0 - marge, y0 - marge, x1 + marge, y1 + marge
        cote_px = max(ex1 - ex0, ey1 - ey0) / self.res
        demi_px = int(max(fenetre_min_px, cote_px) )  # la boîte élargie occupe au plus la moitié
        img, tfm = self.imagette(cx, cy, demi_px)
        if img is None:
            return None, 0.0, "imagette_sans_donnees"
        inv = ~tfm
        px0, py0 = inv * (ex0, ey1)  # coin NW en pixels
        px1, py1 = inv * (ex1, ey0)  # coin SE
        boite_px = [float(px0), float(py0), float(px1), float(py1)]
        pi = self.processor(images=img, input_boxes=[[boite_px]], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(pixel_values=pi["pixel_values"], input_boxes=pi["input_boxes"],
                             multimask_output=True)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), pi["original_sizes"])[0][0].numpy().astype(bool)
        scores = out.iou_scores.cpu().numpy().ravel()
        return choisir_masque(masks, scores, boite_px, tfm, box(ex0, ey0, ex1, ey1), ratio_min, ratio_max)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpkg_entree")
    ap.add_argument("couche")
    ap.add_argument("raster", help="LD (tif ou vrt), même CRS que le GPKG")
    ap.add_argument("gpkg_sortie")
    ap.add_argument("--couche-sortie", default=None, help="défaut : <couche>_sam")
    ap.add_argument("--marge", type=float, default=20.0, help="élargissement de la boîte, en m de chaque côté (défaut 20)")
    ap.add_argument("--aire-min-ratio", type=float, default=0.10, help="aire mini du masque / aire de la boîte élargie")
    ap.add_argument("--aire-max-ratio", type=float, default=1.00, help="aire maxi du masque / aire de la boîte élargie")
    ap.add_argument("--fenetre-min", type=int, default=128, help="demi-côté mini de l'imagette en px (défaut 128 -> 256 px)")
    ap.add_argument("--limite", type=int, default=0, help="ne traiter que les N premières boîtes (test)")
    a = ap.parse_args()

    import pyogrio
    gdf = pyogrio.read_dataframe(a.gpkg_entree, layer=a.couche, fid_as_index=True)
    gdf.insert(0, "fid_source", gdf.index.astype("int64"))  # lien vers la boîte d'origine (runs incrémentaux)
    gdf = gdf.reset_index(drop=True)
    if a.limite:
        gdf = gdf.head(a.limite).copy()
    seg = Segmenteur(a.raster)
    if gdf.crs is not None and seg.src.crs is not None and gdf.crs.to_epsg() != seg.src.crs.to_epsg():
        sys.exit(f"CRS discordants : GPKG {gdf.crs} vs raster {seg.src.crs}")
    print(f"{len(gdf)} boîtes, raster {seg.res} m/px, device {seg.device}, marge {a.marge} m", flush=True)

    geoms, meth, sscore, ious, motifs = [], [], [], [], []
    for i, (geom) in enumerate(gdf.geometry):
        bb = geom.bounds
        poly, s, motif = seg.segmenter(bb, a.marge, a.aire_min_ratio, a.aire_max_ratio, a.fenetre_min)
        if poly is None or poly.is_empty:
            poly, m = box(*bb), "bbox_repli"
        else:
            m = "sam"
        geoms.append(poly); meth.append(m); sscore.append(round(s, 3)); motifs.append(motif)
        b = box(*bb)
        ious.append(round(poly.intersection(b).area / poly.union(b).area, 3))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(gdf)} — sam {meth.count('sam')}, repli {meth.count('bbox_repli')}", flush=True)

    out = gdf.copy()
    out["geometry"] = gpd.GeoSeries(geoms, crs=gdf.crs)
    out["methode"] = meth
    out["motif"] = motifs  # « ok » ou la raison du repli (contrôlée par verif_sam_polygones_bbox)
    out["sam_score"] = sscore
    out["aire_m2"] = out.geometry.area.round(1)
    out["diam_eq_m"] = (2 * np.sqrt(out.geometry.area / np.pi)).round(1)
    out["iou_bbox"] = ious
    couche = a.couche_sortie or f"{a.couche}_sam"
    out.to_file(a.gpkg_sortie, layer=couche, driver="GPKG")
    n_sam = meth.count("sam")
    print(f"écrit {len(out)} polygones -> {a.gpkg_sortie} [{couche}] : sam {n_sam} ({100 * n_sam / len(out):.1f} %), "
          f"repli bbox {len(out) - n_sam} ; diam_eq médian {out['diam_eq_m'].median():.1f} m ; "
          f"iou_bbox médian {np.median(ious):.2f}", flush=True)


if __name__ == "__main__":
    main()
