"""Repeint un dataset slice_zone avec de nouveaux canaux — splits/annotations INTACTS.

Produit, depuis un dataset 648 px existant (LD v1), DEUX nouveaux datasets aux pixels
recalculés depuis le MNT 1 m, sans jamais re-slicer (slice_zone re-tirerait la
sélection : dédup par hash de contenu, filtres de couverture — les splits gelés
seraient perdus). Les fenêtres viennent de `grille.origine` + row/col du
split_manifest (exact, pas les bounds arrondis) ; les `_annotations.coco.json` et le
manifeste sont copiés (manifeste : champ `dataset` + bloc `repeint` de provenance).

Variantes (décisions 2026-08-20, recettes de planche_indices.py) :
- csl  : RGB = R:cvat_combined (VAT general 50 % sur VAT flat) / G:slrm r10 ±0,5 m
         (standard ADAF) / B:LD relu du raster source v1 (byte-identique aux tuiles v1
         — vérifié par comparaison au PNG d'origine) ;
- crim : RGB = CRIM OrRd (pente rad 0-0,8 colorée + (O+ − O−) ±28 en luminosity 50
         puis overlay 50).

Convention pixels : 0-254, NoData 255 (MNT invalide pour csl R/G et crim ; le canal B
garde le NoData du LD v1). Marge de calcul 24 px (SVF r20 flat). Auto-vérification :
comptes par split ≡ manifeste, PNG 648x648 RGB, identité du canal B vs PNG v1.

Usage : .venv\\Scripts\\python.exe tools\\repeindre_dataset.py <dataset_v1> <dossier_datasets_sortie>
            --mnt <tif|motif_glob> [--mnt ...] [--ld <raster>]
Le nom des sorties dérive du nom source : ..._ld648_v1 -> ..._csl648_v1 + ..._crim648_v1.
Dépendance : rvt du plugin QGIS (repli sys.path comme generer_ld.py).
"""
import argparse
import glob
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
import yaml
from PIL import Image

CHEMIN_RVT_QGIS = (Path.home() / "AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/rvt-qgis")
try:
    import rvt.vis
except ImportError:
    sys.path.insert(0, str(CHEMIN_RVT_QGIS))
    import rvt.vis
from rvt.blend_func import (blend_images, gray_scale_to_color_ramp, normalize_image,
                            render_images)

MARGE = 24          # px — couvre SVF r20 (flat), plus grand rayon utilisé
NODATA = 255


def q8(v01):
    """float [0,1] -> uint8 0-254 (convention planche_indices)."""
    return np.clip(np.round(np.clip(v01, 0.0, 1.0) * 255), 0, 254)


def canaux_tuile(mnt):
    """MNT float32 (696x696, NaN=invalide) -> (cvat 2D, slrm 2D, crim (3,H,W)) float [0,1]."""
    res = 1.0
    pente = rvt.vis.slope_aspect(dem=mnt.copy(), resolution_x=res, resolution_y=res,
                                 output_units="degree", ve_factor=1)["slope"]
    hs35 = rvt.vis.hillshade(dem=mnt.copy(), resolution_x=res, resolution_y=res,
                             sun_azimuth=315, sun_elevation=35, ve_factor=1)
    hs15 = rvt.vis.hillshade(dem=mnt.copy(), resolution_x=res, resolution_y=res,
                             sun_azimuth=315, sun_elevation=15, ve_factor=1)
    d10 = rvt.vis.sky_view_factor(dem=mnt.copy(), resolution=res, compute_svf=True,
                                  compute_opns=True, svf_n_dir=16, svf_r_max=10,
                                  svf_noise=0, ve_factor=1)
    svf10, opns_pos = d10["svf"], d10["opns"]
    svf_flat = rvt.vis.sky_view_factor(dem=mnt.copy(), resolution=res, compute_svf=True,
                                       svf_n_dir=16, svf_r_max=20, svf_noise=3,
                                       ve_factor=1)["svf"]
    opns_neg = rvt.vis.sky_view_factor(dem=-mnt.copy(), resolution=res, compute_svf=False,
                                       compute_opns=True, svf_n_dir=16, svf_r_max=10,
                                       ve_factor=1)["opns"]
    slrm10 = rvt.vis.slrm(dem=mnt.copy(), radius_cell=10, ve_factor=1)

    def vat(svf_a, svf_lo, opns_lo, opns_hi, pente_hi, hs_a):
        rendu = normalize_image("hillshade", np.clip(hs_a, 0, 1), 0.0, 1.0, "value")
        pente_n = normalize_image("slope gradient", pente, 0.0, pente_hi, "value")
        opns_n = normalize_image("openness - positive", opns_pos, opns_lo, opns_hi, "value")
        svf_n = normalize_image("sky-view factor", svf_a, svf_lo, 1.0, "value")
        rendu = render_images(blend_images("luminosity", pente_n, rendu), rendu, 50)
        rendu = render_images(blend_images("overlay", opns_n, rendu), rendu, 50)
        rendu = render_images(blend_images("multiply", svf_n, rendu), rendu, 25)
        return rendu

    vat_gen = vat(svf10, 0.7, 68.0, 93.0, 50.0, hs35)
    vat_flat = vat(svf_flat, 0.9, 85.0, 93.0, 15.0, hs15)
    cvat = render_images(blend_images("normal", vat_gen, vat_flat), vat_flat, 50)

    slrm01 = (np.clip(slrm10, -0.5, 0.5) + 0.5) / 1.0

    posneg_n = normalize_image("Openness_Pos-Neg", opns_pos - opns_neg, -28.0, 28.0, "value")
    pente_rad_n = normalize_image("slope rad", np.radians(pente), 0.0, 0.8, "value")
    crim = gray_scale_to_color_ramp(pente_rad_n, "OrRd", 0, 1, output_8bit=False)
    crim = render_images(blend_images("luminosity", posneg_n, crim), crim, 50)
    crim = render_images(blend_images("overlay", posneg_n, crim), crim, 50)
    return cvat, slrm01, np.clip(crim, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("sortie_dir", type=Path)
    ap.add_argument("--mnt", action="append", required=True,
                    help="fichier .tif ou motif glob (répétable)")
    ap.add_argument("--ld", type=Path, default=None,
                    help="raster LD v1 (défaut : config.raster du manifeste)")
    a = ap.parse_args()

    man = yaml.safe_load((a.dataset / "split_manifest.yaml").read_text(encoding="utf-8"))
    nom_v1 = man["dataset"]
    if "ld648" not in nom_v1:
        sys.exit(f"nom de dataset inattendu (pas de 'ld648') : {nom_v1}")
    gr = man["grille"]
    ox, oy = gr["origine"]
    px = gr["tuile_px"]
    gsd = gr["gsd_m_px"][0]
    if abs(gsd - 1.0) > 1e-6:
        sys.exit(f"GSD {gsd} != 1 m : recettes calées pour le GSD enclos de 1 m")

    chemin_ld = a.ld or Path(man["config"]["raster"])
    src_ld = rasterio.open(chemin_ld)
    mnts = sorted({p for motif in a.mnt for p in glob.glob(motif)})
    if not mnts:
        sys.exit(f"aucun MNT ne correspond à : {a.mnt}")
    sources = []
    for p in mnts:
        ds = rasterio.open(p)
        sources.append((ds.bounds, ds))
    print(f"{nom_v1} : {len(man['tuiles'])} tuiles, {len(sources)} MNT, LD {chemin_ld}")

    def lire_mnt(bx):
        """Composite premier-valide des MNT intersectant les bounds bx, NaN=invalide."""
        h = w = px + 2 * MARGE
        out = np.full((h, w), np.nan, dtype="float32")
        for b, ds in sources:
            if b.left >= bx[2] or b.right <= bx[0] or b.bottom >= bx[3] or b.top <= bx[1]:
                continue
            fen = rasterio.windows.from_bounds(*bx, transform=ds.transform)
            fen = rasterio.windows.Window(round(fen.col_off), round(fen.row_off), w, h)
            bloc = ds.read(1, window=fen, boundless=True, fill_value=np.nan).astype("float32")
            bloc[~np.isfinite(bloc)] = np.nan
            if ds.nodata is not None:
                bloc[bloc == np.float32(ds.nodata)] = np.nan
            trou = np.isnan(out)
            out[trou] = bloc[trou]
        return out

    variantes = {"csl": nom_v1.replace("ld648", "csl648"),
                 "crim": nom_v1.replace("ld648", "crim648")}
    dossiers = {}
    for v, nom in variantes.items():
        d = a.sortie_dir / nom
        if d.exists():
            shutil.rmtree(d)
        for split in ("train", "valid", "test"):
            (d / split).mkdir(parents=True)
            shutil.copy2(a.dataset / split / "_annotations.coco.json",
                         d / split / "_annotations.coco.json")
        m2 = dict(man)
        m2["dataset"] = nom
        m2["repeint"] = {"source_dataset": nom_v1, "variante": v,
                         "canaux": ("R:cvat_combined G:slrm_r10_e0.5 B:ld_v1" if v == "csl"
                                    else "RGB:crim_orrd"),
                         "mnt": mnts, "ld": str(chemin_ld), "date": str(date.today())}
        (d / "split_manifest.yaml").write_text(
            yaml.safe_dump(m2, allow_unicode=True, sort_keys=False), encoding="utf-8")
        dossiers[v] = d

    ident_b = []   # taux d'identité canal B vs PNG v1
    for i, t in enumerate(man["tuiles"]):
        x0 = ox + t["col"] * px * gsd
        y1 = oy - t["row"] * px * gsd
        bx = (x0 - MARGE, y1 - px - MARGE, x0 + px + MARGE, y1 + MARGE)
        mnt = lire_mnt(bx)
        invalide = ~np.isfinite(mnt[MARGE:MARGE + px, MARGE:MARGE + px])
        if invalide.all():
            print(f"  ATTENTION {t['nom']} : MNT entièrement invalide")
            cvat = slrm01 = np.zeros((px, px), dtype="float32")
            crim = np.zeros((3, px, px), dtype="float32")
        else:
            cvat, slrm01, crim = canaux_tuile(mnt)
            cvat = cvat[MARGE:MARGE + px, MARGE:MARGE + px]
            slrm01 = slrm01[MARGE:MARGE + px, MARGE:MARGE + px]
            crim = crim[:, MARGE:MARGE + px, MARGE:MARGE + px]

        fen_ld = rasterio.windows.from_bounds(x0, y1 - px, x0 + px, y1,
                                              transform=src_ld.transform)
        fen_ld = rasterio.windows.Window(round(fen_ld.col_off), round(fen_ld.row_off), px, px)
        ld = src_ld.read(1, window=fen_ld, boundless=True, fill_value=NODATA)
        png_v1 = np.asarray(Image.open(a.dataset / t["split"] / t["nom"]))[:, :, 0]
        ident_b.append(float((ld == png_v1).mean()))

        r = np.where(np.isfinite(cvat) & ~invalide, q8(cvat), NODATA).astype("uint8")
        g = np.where(np.isfinite(slrm01) & ~invalide, q8(slrm01), NODATA).astype("uint8")
        Image.fromarray(np.stack([r, g, ld.astype("uint8")], axis=-1), mode="RGB").save(
            dossiers["csl"] / t["split"] / t["nom"])
        cr = np.where(np.isfinite(crim) & ~invalide[None, :, :], q8(crim), NODATA).astype("uint8")
        Image.fromarray(np.moveaxis(cr, 0, -1), mode="RGB").save(
            dossiers["crim"] / t["split"] / t["nom"])
        if (i + 1) % 20 == 0 or i + 1 == len(man["tuiles"]):
            print(f"  {i + 1}/{len(man['tuiles'])} tuiles")

    # auto-vérification
    erreurs = []
    ident_moy = float(np.mean(ident_b))
    if ident_moy < 0.999:
        erreurs.append(f"canal B (LD) ≠ PNG v1 : identité moyenne {ident_moy:.4%} < 99,9 % "
                       "(raster LD changé depuis le slicing v1 ?)")
    for v, d in dossiers.items():
        for split in ("train", "valid", "test"):
            attendu = [t["nom"] for t in man["tuiles"] if t["split"] == split]
            produits = sorted(p.name for p in (d / split).glob("*.png"))
            if sorted(attendu) != produits:
                erreurs.append(f"{v}/{split} : {len(produits)} PNG vs {len(attendu)} attendus")
        im = np.asarray(Image.open(d / man["tuiles"][0]["split"] / man["tuiles"][0]["nom"]))
        if im.shape != (px, px, 3):
            erreurs.append(f"{v} : forme PNG {im.shape} != ({px},{px},3)")
    if erreurs:
        sys.exit("NON CONFORME :\n  " + "\n  ".join(erreurs))
    print(f"CONFORME — {variantes['csl']} + {variantes['crim']} : "
          f"{len(man['tuiles'])} tuiles, canal B identique v1 à {ident_moy:.4%}")


if __name__ == "__main__":
    main()
