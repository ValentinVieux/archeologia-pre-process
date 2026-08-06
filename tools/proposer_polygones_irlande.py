"""Propose un polygone d'emprise par point SMR recalé (corpus Irlande).

Pipeline hybride rodé sur Sligo (2026-08-06, ~0,80 IoU médian sur éval gelée,
99 % de l'oracle des deux briques) : cercle MULTI-ÉCHELLE façon méthode B
(rayon initial médian de classe x 0,5/1/1,6 ; profils radiaux, extremum_profil
de recaler_lignes, régularisation circulaire) -> boîte englobante -> SAM 2.1
fine-tuné (decoder_ft4b) -> post-traitement (composante principale, trous
remplis, lissage rond, simplification). Drapeau `a_verifier` quand cercle et
SAM divergent (accord < 0,4) ET que le contraste du cercle est faible (< 25).

ATTENTION : nécessite le venv SAM (torch CUDA), PAS le .venv du repo :
    D:\\veille_irlande\\venv_sam\\Scripts\\python.exe tools\\proposer_polygones_irlande.py
        <points.gpkg> <ld.tif> <sortie.gpkg> [--couche points_a_recaler]
        [--decodeur D:\\veille_irlande\\sam_ft\\decoder_ft4b.pt]

Les seuils vivent dans D:\\veille_irlande\\sam_ft\\{recalage_cercles_params,cascade_params}.yaml
(calibrés sur le train gelé de Sligo — recalibrer si un secteur diverge, jamais
à la main). Vérification jumelle : tools\\verif_polygones_irlande.py.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import yaml
from shapely.geometry import Point, Polygon, shape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.recaler_lignes import LecteurRaster, extremum_profil

DEMI = 128                    # imagette 256 px autour du point (1 m/px attendu)
AIRE_MIN, AIRE_MAX = 100, 12000
ECHELLES = (0.5, 1.0, 1.6)    # multi-échelle du rayon initial (petits + « plus gros qu'attendu »)
N_AZ, PAS_ECHANT = 72, 0.5
R0_DEFAUT = {"ringfort": 14.3, "enclosure": 12.2}  # médians train Sligo (m)


def nettoyer(geom, lissage_m=2.0):
    """Post-traitement acté : composante principale, trous remplis, lissage rond."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    doux = Polygon(geom.exterior).buffer(lissage_m, join_style=1).buffer(-lissage_m, join_style=1)
    if doux.geom_type == "MultiPolygon":
        doux = max(doux.geoms, key=lambda g: g.area)
    return doux.simplify(0.8)


def regulariser_circulaire(offsets, poids_derivee):
    o = np.asarray(offsets, dtype=float)
    n = len(o)
    w = np.where(np.isfinite(o), 1.0, 0.0)
    cible = np.where(np.isfinite(o), o, 0.0)
    A = np.diag(w)
    for i in range(n):
        j = (i + 1) % n
        A[i, i] += poids_derivee
        A[j, j] += poids_derivee
        A[i, j] -= poids_derivee
        A[j, i] -= poids_derivee
    if not np.isfinite(o).any():
        return np.zeros(n)
    return np.linalg.solve(A, w * cible)


class Proposeur:
    def __init__(self, ld_path, decodeur, params_cercle):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(self.device)
        self.model.load_state_dict(torch.load(decodeur, map_location=self.device), strict=False)
        self.model.eval()
        self.processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
        self.lecteur = LecteurRaster(ld_path)
        self.src = self.lecteur.src
        self.pc = params_cercle

    def imagette(self, x, y):
        from PIL import Image
        r0, c0 = self.src.index(x, y)
        win = rasterio.windows.Window(c0 - DEMI, r0 - DEMI, 2 * DEMI, 2 * DEMI)
        a = self.src.read(1, window=win, boundless=True, fill_value=255).astype("float32")
        a[a == 255] = np.nan
        if not np.isfinite(a).any():
            return None, None
        a = np.where(np.isnan(a), np.nanmedian(a), a).astype("uint8")
        return Image.fromarray(np.stack([a] * 3, axis=-1)), self.src.window_transform(win)

    def cercle(self, cx, cy, r0):
        fen = self.pc["fenetre_m"]
        az = np.linspace(0, 2 * np.pi, N_AZ, endpoint=False)
        cosa, sina = np.cos(az), np.sin(az)
        marge = r0 + fen + 10
        donnees, affine = self.lecteur.fenetre((cx - marge, cy - marge, cx + marge, cy + marge), 0)
        if donnees.size == 0:
            return None, 0.0
        ts = np.arange(-fen, fen + PAS_ECHANT, PAS_ECHANT)
        rr = r0 + ts[None, :] + np.zeros((N_AZ, 1))
        xs, ys = cx + rr * cosa[:, None], cy + rr * sina[:, None]
        prof = self.lecteur.echantillonner(donnees, affine,
                                           np.column_stack([xs.ravel(), ys.ravel()])).reshape(N_AZ, len(ts))
        meilleur = None
        for pol in ("clair", "sombre"):
            offs = np.full(N_AZ, np.nan)
            contrastes = []
            for i in range(N_AZ):
                off, c, _ = extremum_profil(prof[i], pol, self.pc["seuil_contraste"], 0.7,
                                            PAS_ECHANT, self.pc["poids_distance"])
                if off is not None:
                    offs[i] = off
                    contrastes.append(c)
            score = (len(contrastes) / N_AZ) * (np.median(contrastes) if contrastes else 0)
            if meilleur is None or score > meilleur[0]:
                meilleur = (score, offs)
        score, offs = meilleur
        rayons = np.clip(r0 + regulariser_circulaire(offs, self.pc["poids_derivee"]), 3.0, r0 + fen + 5)
        pts = np.column_stack([cx + rayons * cosa, cy + rayons * sina])
        return Polygon(pts).buffer(0).simplify(0.5), float(score)

    def cercle_ms(self, cx, cy, r0):
        best = None
        for e in ECHELLES:
            poly, sc = self.cercle(cx, cy, r0 * e)
            if poly is not None and (best is None or sc > best[1]):
                best = (poly, sc, e)
        return best if best else (None, 0.0, 1.0)

    def sam_boite(self, x, y, box_px):
        img, tfm = self.imagette(x, y)
        if img is None:
            return None, 0.0
        pi = self.processor(images=img, input_boxes=[[box_px]], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(pixel_values=pi["pixel_values"], input_boxes=pi["input_boxes"],
                             multimask_output=True)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), pi["original_sizes"])[0][0].numpy().astype(bool)
        scores = out.iou_scores.cpu().numpy().ravel()
        best = None
        for m, s in zip(masks, scores):
            aire = int(m.sum())
            if not (AIRE_MIN <= aire <= AIRE_MAX) or not m[DEMI, DEMI]:
                continue
            if best is None or s > best[1]:
                best = (m, float(s))
        if best is None:
            return None, 0.0
        formes = [shape(g) for g, v in rasterio.features.shapes(best[0].astype("uint8"), transform=tfm) if v == 1]
        return (nettoyer(max(formes, key=lambda q: q.area)) if formes else None), best[1]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    u = a.union(b).area
    return a.intersection(b).area / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("points", help="GPKG des points recalés par l'utilisateur")
    ap.add_argument("ld", help="raster LD du secteur (1 m/px, Rmin5/Rmax10)")
    ap.add_argument("sortie", help="GPKG de propositions à produire")
    ap.add_argument("--couche", default="points_a_recaler")
    ap.add_argument("--decodeur", default=r"D:\veille_irlande\sam_ft\decoder_ft4b.pt")
    ap.add_argument("--params-dir", default=r"D:\veille_irlande\sam_ft")
    a = ap.parse_args()

    pc = yaml.safe_load(open(Path(a.params_dir) / "recalage_cercles_params.yaml", encoding="utf-8"))
    cascade = yaml.safe_load(open(Path(a.params_dir) / "cascade_params.yaml", encoding="utf-8"))
    accord_min, contraste_min = cascade["accord_min"], cascade["contraste_min"]
    r0_classe = {**R0_DEFAUT, **{k: float(v) for k, v in pc.get("r0_classe", {}).items()}}

    prop = Proposeur(a.ld, a.decodeur, pc["params"])
    pts = gpd.read_file(a.points, layer=a.couche).to_crs(prop.src.crs)
    rows = []
    for k, pt in pts.iterrows():
        x, y = pt.geometry.x, pt.geometry.y
        cls = pt.get("classe") if pt.get("classe") in r0_classe else "ringfort"
        poly_c, sc_c, echelle = prop.cercle_ms(x, y, r0_classe[cls])
        poly_f, methode, sc_s = None, "echec", 0.0
        if poly_c is not None:
            b = poly_c.bounds
            x0, y0 = DEMI + (b[0] - x), DEMI + (y - b[3])
            x1, y1 = DEMI + (b[2] - x), DEMI + (y - b[1])
            if 0 <= x0 < x1 <= 2 * DEMI and 0 <= y0 < y1 <= 2 * DEMI:
                poly_f, sc_s = prop.sam_boite(x, y, [x0, y0, x1, y1])
                methode = "cercle_boite_sam"
            if poly_f is None:
                poly_f, methode = poly_c, "cercle_repli"
        accord = iou(poly_f, poly_c) if poly_c is not None else 0.0
        rows.append({"ENTITY_ID": pt.get("ENTITY_ID"), "classe": pt.get("classe"),
                     "methode": methode, "echelle": echelle,
                     "sc_cercle": round(sc_c, 1), "sc_sam": round(sc_s, 3), "accord": round(accord, 2),
                     "a_verifier": int(accord < accord_min and sc_c < contraste_min),
                     "verdict": "", "note": "",
                     "geometry": poly_f if poly_f is not None else Point(x, y).buffer(12)})
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(pts)}")

    g = gpd.GeoDataFrame(rows, crs=prop.src.crs).sort_values(["a_verifier", "accord"], ascending=[False, True])
    g.to_file(a.sortie, layer="propositions", driver="GPKG")
    n_flag = int(g.a_verifier.sum())
    print(f"{len(g)} propositions -> {a.sortie} | a_verifier : {n_flag} | "
          f"méthodes : {dict(g.methode.value_counts())} | échelles : {dict(g.echelle.value_counts())}")


if __name__ == "__main__":
    main()
