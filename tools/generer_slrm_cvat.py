"""Génère les indices SLRM et CVAT 8 bits d'un MNT, paramétrés PAR RÉSOLUTION.

- SLRM : paramètres EXACTS du standard ADAF irlandais (Čož et al. 2026, Table 1) —
  rayon de tendance 10 m (converti en pixels selon la résolution), étirement linéaire
  FIXE ±0,5 m. Sert de bras d'ablation face au LD (docs/rapport_test_adaf.html).
- CVAT : VAT général de Kokalj & Somrak 2019 (couches grises) — SVF [0,7965-1] multiply
  25 % + openness positive [68-93°] overlay 50 % + pente [0-50°] luminosity 50 % +
  ombrage 315°/35° — rendu mono-bande.

Sorties 8 bits convention maison : données 0-254, NoData 255 (≠ ADAF qui met 0 — ici
on veut des rasters lisibles en QGIS ET convertibles pour l'entraînement). Traitement
par tuiles 4096 px avec marge, auto-vérification CONFORME (taille/CRS, médianes en
gamme). Étirements FIXES : jamais par scène, comparabilité inter-zones (doctrine LD).

Usage : python tools\\generer_slrm_cvat.py <mnt.tif> <dossier_sortie> [--prefixe <nom>]
Produit <dossier>\\slrm_<prefixe>.tif et <dossier>\\cvat_<prefixe>.tif.
Dépendance : rvt-py (venv_adaf) ou rvt du plugin QGIS (repli comme generer_ld).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows

CHEMIN_RVT_QGIS = (Path.home() / "AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/rvt-qgis")
try:
    import rvt.vis
except ImportError:
    sys.path.insert(0, str(CHEMIN_RVT_QGIS))
    import rvt.vis
from rvt.blend_func import blend_images, render_images, normalize_image

TUILE = 4096
SLRM_RAYON_M = 10.0          # standard ADAF (Table 1)
SLRM_ETIREMENT = (-0.5, 0.5)  # metres, FIXE
NODATA_OUT = 255


def slrm_tuile(bloc, rayon_px):
    s = rvt.vis.slrm(dem=bloc, radius_cell=rayon_px, ve_factor=1)
    s = np.clip(s, *SLRM_ETIREMENT)
    b = ((s - SLRM_ETIREMENT[0]) / (SLRM_ETIREMENT[1] - SLRM_ETIREMENT[0]) * 255).round()
    return b


def cvat_tuile(bloc, res):
    hs = rvt.vis.hillshade(dem=bloc, resolution_x=res, resolution_y=res,
                           sun_azimuth=315, sun_elevation=35, ve_factor=1)
    pente = rvt.vis.slope_aspect(dem=bloc, resolution_x=res, resolution_y=res,
                                 output_units="degree", ve_factor=1)["slope"]
    svf_d = rvt.vis.sky_view_factor(dem=bloc, resolution=res, compute_svf=True,
                                    compute_opns=True, svf_n_dir=16, svf_r_max=10,
                                    svf_noise=0, ve_factor=1)
    svf, opns = svf_d["svf"], svf_d["opns"]
    # VAT general (Kokalj & Somrak 2019) — toutes couches grises, fond ombrage
    svf_n = normalize_image("sky-view factor", svf, 0.7965, 1.0, "value")
    opns_n = normalize_image("openness - positive", opns, 68.0, 93.0, "value")
    pente_n = normalize_image("slope gradient", pente, 0.0, 50.0, "value")
    hs_n = normalize_image("hillshade", hs, 0.0, 1.0, "value")
    # empilement du haut vers le bas : SVF multiply 25 % / opns overlay 50 % /
    # pente luminosity 50 % / ombrage (fond) — rendu de bas en haut
    rendu = hs_n
    rendu = render_images(blend_images("luminosity", pente_n, rendu), rendu, 50)
    rendu = render_images(blend_images("overlay", opns_n, rendu), rendu, 50)
    rendu = render_images(blend_images("multiply", svf_n, rendu), rendu, 25)
    return np.clip(rendu * 255.0, 0, 255).round()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mnt", type=Path)
    ap.add_argument("sortie_dir", type=Path)
    ap.add_argument("--prefixe", default=None, help="défaut : nom du MNT sans mnt_/_mosaique")
    a = ap.parse_args()

    src = rasterio.open(a.mnt)
    res = abs(src.res[0])
    rayon_px = round(SLRM_RAYON_M / res)
    if rayon_px < 10:
        # garde rvt : rayon SLRM dans [10, 50] px — on borne et on le dit (le rayon
        # métrique n'est alors plus 10 m ; cas des MNT >= 2 m)
        print(f"ATTENTION : rayon {rayon_px} px < 10 (garde rvt) -> borné à 10 px "
              f"= {10*res:g} m métriques (standard ADAF : 10 m)")
        rayon_px = 10
    prefixe = a.prefixe or a.mnt.stem.replace("mnt_", "").replace("_mosaique", "")
    a.sortie_dir.mkdir(parents=True, exist_ok=True)
    sorties = {"slrm": a.sortie_dir / f"slrm_{prefixe}.tif",
               "cvat": a.sortie_dir / f"cvat_{prefixe}.tif"}
    print(f"MNT {src.width}x{src.height} px à {res:g} m/px -> SLRM R{rayon_px} px "
          f"({rayon_px*res:g} m) + CVAT -> {a.sortie_dir}")

    profil = src.profile | {"dtype": "uint8", "nodata": NODATA_OUT, "count": 1,
                            "compress": "deflate", "tiled": True, "BIGTIFF": "IF_SAFER"}
    marge = max(rayon_px, 12) + 2
    dsts = {k: rasterio.open(p, "w", **profil) for k, p in sorties.items()}
    for r0 in range(0, src.height, TUILE):
        for c0 in range(0, src.width, TUILE):
            h, w = min(TUILE, src.height - r0), min(TUILE, src.width - c0)
            rl, cl = max(0, r0 - marge), max(0, c0 - marge)
            rh, ch = min(src.height, r0 + h + marge), min(src.width, c0 + w + marge)
            bloc = src.read(1, window=rasterio.windows.Window(cl, rl, ch - cl, rh - rl)).astype("float32")
            # NoData float64-max (dalles phase2) : devient inf au cast float32 et
            # echappe au test == — neutraliser TOUT non-fini AVANT les convolutions
            bloc[~np.isfinite(bloc)] = np.nan
            if src.nodata is not None:
                bloc[bloc == np.float32(src.nodata)] = np.nan
            invalide = ~np.isfinite(bloc)
            for cle, dst in dsts.items():
                if invalide.all():
                    tuile = np.full((h, w), NODATA_OUT, dtype="uint8")
                else:
                    b = slrm_tuile(bloc, rayon_px) if cle == "slrm" else cvat_tuile(bloc, res)
                    b = np.where(np.isfinite(b), b, NODATA_OUT)
                    b = np.clip(b, 0, 254)  # 255 réservé NoData
                    b[invalide] = NODATA_OUT
                    tuile = b[r0 - rl:r0 - rl + h, c0 - cl:c0 - cl + w].astype("uint8")
                dst.write(tuile, 1, window=rasterio.windows.Window(c0, r0, w, h))
        print(f"  lignes {min(r0 + TUILE, src.height)}/{src.height}")
    for dst in dsts.values():
        dst.close()

    # auto-vérification
    gammes = {"slrm": (100, 150), "cvat": (60, 250)}  # SLRM ~0 m -> ~127 ; CVAT clair sur le plat (~240)
    for cle, p in sorties.items():
        with rasterio.open(p) as v:
            assert (v.width, v.height) == (src.width, src.height), f"{cle} : taille != entrée"
            assert v.crs == src.crs, f"{cle} : CRS != entrée"
            ech = v.read(1, out_shape=(max(1, v.height // 8), max(1, v.width // 8)))
            valides = ech[ech != NODATA_OUT]
            med = float(np.median(valides)) if valides.size else float("nan")
            lo, hi = gammes[cle]
            assert valides.size and lo <= med <= hi, f"{cle} : médiane {med} hors gamme [{lo},{hi}]"
            print(f"CONFORME — {p} : médiane {med:.0f}, NoData {NODATA_OUT}, CRS {v.crs.to_epsg()}")


if __name__ == "__main__":
    main()
