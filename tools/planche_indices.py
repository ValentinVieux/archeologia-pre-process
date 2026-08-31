"""Planche d'indices de visualisation d'une dalle de MNT 1 m — choix des canaux multicanaux.

Calcule TOUS les indices RVT disponibles (défauts) + les variantes de paramètres des
canaux candidats à l'entraînement multicanal enclos, sur une emprise donnée d'un MNT
EPSG:2154 à 1 m/px. Sorties GeoTIFF 8 bits convention maison (données 0-254, NoData 255,
étirements FIXES — doctrine LD), lisibles telles quelles dans QGIS, + manifeste.md.

Recettes vérifiées sur l'install locale rvt-qgis (settings/blender_VAT.json,
rvt/blend.py) et la synthèse biblio 2026-08-20 :
- VAT general/flat = Kokalj & Somrak 2019 (les VRAIS étirements du JSON RVT, dont
  SVF 0,7-1 — le 0,7965 de generer_slrm_cvat.py est un écart non standard) ;
- CVAT (VAT combined) = VAT general opacité 50 SUR VAT flat (rvt_blender.py:165-197) ;
- e3MSTP = SLRM(±0,5) screen 25 / CRIM(OrRd) soft_light 70 / MSTP échelles
  DefaultValues (blend.py:1321) — ≠ MSTP « Guyot » (défauts de la FONCTION mstp) ;
- SLRM R10 ±0,5 m = standard ADAF ; LD Rmin5/Rmax10 étiré 0,5-1,8 = production maison.

Usage : .venv\\Scripts\\python.exe tools\\planche_indices.py <mnt> <sortie_dir>
            --emprise xmin ymin xmax ymax [--nom dalle_a] [--gpkg <vecteurs>] [--couche emprises]

Dépendance : rvt du plugin QGIS (repli sys.path comme generer_ld.py) — rvt.blend/default
inutilisables sans osgeo, les blends sont recomposés via rvt.blend_func.
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
from rvt.blend_func import (blend_images, gray_scale_to_color_ramp, normalize_image,
                            render_images)

NODATA_OUT = 255


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mnt", type=Path)
    ap.add_argument("sortie_dir", type=Path)
    ap.add_argument("--emprise", nargs=4, type=float, required=True,
                    metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    ap.add_argument("--nom", default=None, help="sous-dossier (défaut : dalle_<xmin>_<ymin>)")
    ap.add_argument("--gpkg", type=Path, default=None, help="vecteurs GT à clipper sur l'emprise")
    ap.add_argument("--couche", default=None, help="couche du gpkg (défaut : toutes)")
    a = ap.parse_args()

    src = rasterio.open(a.mnt)
    res = abs(src.res[0])
    if abs(res - 1.0) > 0.01:
        sys.exit(f"MNT à {res:g} m/px : la planche est paramétrée pour le GSD enclos de 1 m "
                 "(rayons px = mètres, étirements calés) — ré-échantillonner d'abord.")

    fen = rasterio.windows.from_bounds(*a.emprise, transform=src.transform)
    fen = rasterio.windows.Window(round(fen.col_off), round(fen.row_off),
                                  round(fen.width), round(fen.height))
    h, w = int(fen.height), int(fen.width)
    transform = src.window_transform(fen)
    nom = a.nom or f"dalle_{a.emprise[0]:.0f}_{a.emprise[1]:.0f}"
    dossier = a.sortie_dir / nom
    dossier.mkdir(parents=True, exist_ok=True)
    print(f"Dalle {nom} : {w}x{h} px à {res:g} m/px -> {dossier}")

    _cache: dict[int, np.ndarray] = {}

    def lire(marge: int) -> np.ndarray:
        """Bloc MNT float32 avec marge (px), non-finis et NoData -> NaN. Copie fraîche
        à chaque appel : les fonctions rvt MUTENT leur entrée."""
        if marge not in _cache:
            fm = rasterio.windows.Window(fen.col_off - marge, fen.row_off - marge,
                                         w + 2 * marge, h + 2 * marge)
            bloc = src.read(1, window=fm, boundless=True,
                            fill_value=np.nan).astype("float32")
            bloc[~np.isfinite(bloc)] = np.nan
            if src.nodata is not None:
                bloc[bloc == np.float32(src.nodata)] = np.nan
            _cache[marge] = bloc
        return _cache[marge].copy()

    def rogner(arr: np.ndarray, marge: int) -> np.ndarray:
        return arr[..., marge:marge + h, marge:marge + w] if marge else arr

    invalide = ~np.isfinite(lire(0))
    profil = {"driver": "GTiff", "width": w, "height": h, "crs": src.crs,
              "transform": transform, "compress": "deflate", "tiled": True}
    manifeste = []

    def ecrire_8bits(fichier, arr, lo, hi, note, inverser=False):
        """Étirement FIXE lo-hi -> 0-254 (inversé si demandé), NoData 255."""
        v = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        if inverser:
            v = 1.0 - v
        b = np.where(np.isfinite(v), np.clip(np.round(v * 255), 0, 254), NODATA_OUT)
        b[invalide] = NODATA_OUT
        with rasterio.open(dossier / fichier, "w", **profil, dtype="uint8",
                           count=1, nodata=NODATA_OUT) as dst:
            dst.write(b.astype("uint8"), 1)
        manifeste.append((fichier, note, f"étiré {lo:g}→{hi:g}" + (" inversé (1-x)" if inverser else "")))
        print(f"  {fichier}")

    def ecrire_rgb(fichier, arr, note):
        """RGB float [0,1] (3,H,W) -> uint8 0-254, NoData 255."""
        b = np.where(np.isfinite(arr), np.clip(np.round(arr * 255), 0, 254), NODATA_OUT)
        b[:, invalide] = NODATA_OUT
        with rasterio.open(dossier / fichier, "w", **profil, dtype="uint8",
                           count=3, nodata=NODATA_OUT) as dst:
            dst.write(b.astype("uint8"))
        manifeste.append((fichier, note, "RGB 0-1 → 0-254"))
        print(f"  {fichier}")

    # ---------- MNT de référence ----------
    mnt0 = lire(0)
    with rasterio.open(dossier / "mnt.tif", "w", **profil, dtype="float32",
                       count=1, nodata=-9999.0) as dst:
        dst.write(np.where(np.isfinite(mnt0), mnt0, -9999.0), 1)
    manifeste.append(("mnt.tif", "MNT source clippé (float32) — fond de contrôle", "brut"))

    # ---------- indices simples ----------
    m = 4
    pente = rvt.vis.slope_aspect(dem=lire(m), resolution_x=res, resolution_y=res,
                                 output_units="degree", ve_factor=1)["slope"]
    pente = rogner(pente, m)
    ecrire_8bits("slope_e0-15.tif", pente, 0, 15, "Pente — étirement flat (plateau) [VARIANTE canal]")
    ecrire_8bits("slope_e0-51.tif", pente, 0, 51, "Pente — étirement général RVT")

    hs = {}
    for az, el in [(315, 35), (22.5, 35), (90, 35), (315, 15)]:
        v = rvt.vis.hillshade(dem=lire(m), resolution_x=res, resolution_y=res,
                              sun_azimuth=az, sun_elevation=el, ve_factor=1)
        hs[(az, el)] = rogner(v, m)
    ecrire_8bits("hillshade_a315_h35.tif", hs[(315, 35)], 0, 1,
                 "Ombrage classique — biais d'azimut, jamais seul en entrée modèle")
    ecrire_rgb("multi_hillshade_rgb_a315_a22_a90.tif",
               np.stack([np.clip(hs[(315, 35)], 0, 1), np.clip(hs[(22.5, 35)], 0, 1),
                         np.clip(hs[(90, 35)], 0, 1)]),
               "Multi-ombrage RGB (3 azimuts du rendu RVT)")

    for r in (10, 20, 40):
        v = rogner(rvt.vis.slrm(dem=lire(r + 2), radius_cell=r, ve_factor=1), r + 2)
        ecrire_8bits(f"slrm_r{r}_e0.5.tif", v, -0.5, 0.5,
                     f"SLRM rayon {r} m ±0,5 m — {'standard ADAF' if r == 10 else 'défaut RVT/littérature' if r == 20 else '> rayon des grands enclos'} [CANAL n°1, variantes]")
        if r == 20:
            ecrire_8bits("slrm_r20_e2.tif", v, -2, 2,
                         "SLRM r20 ±2 m (étirement RVT) — montre l'écrasement des talus < 0,5 m")

    m = 12
    svf10 = rvt.vis.sky_view_factor(dem=lire(m), resolution=res, compute_svf=True,
                                    compute_opns=True, compute_asvf=True, svf_n_dir=16,
                                    svf_r_max=10, svf_noise=0, asvf_dir=315, asvf_level=1,
                                    ve_factor=1)
    svf_r10 = rogner(svf10["svf"], m)
    opns_pos10 = rogner(svf10["opns"], m)
    ecrire_8bits("svf_r10_e0.6375-1.tif", svf_r10, 0.6375, 1.0,
                 "SVF r10 16 dir — étirement défaut RVT [CANAL creux]")
    ecrire_8bits("svf_r10_e0.9-1.tif", svf_r10, 0.9, 1.0,
                 "SVF r10 — étirement flat resserré [VARIANTE]")
    ecrire_8bits("opns_pos_r10_e68-93.tif", opns_pos10, 68, 93,
                 "Openness positive r10 — étirement VAT general")
    ecrire_8bits("asvf_a315_e0.7-0.9.tif", rogner(svf10["asvf"], m), 0.7, 0.9,
                 "SVF anisotrope 315° — pour mémoire (jamais recommandé en entrée modèle)")

    m = 22
    svf_flat = rogner(rvt.vis.sky_view_factor(dem=lire(m), resolution=res, compute_svf=True,
                                              svf_n_dir=16, svf_r_max=20, svf_noise=3,
                                              ve_factor=1)["svf"], m)
    ecrire_8bits("svf_r20_n3_e0.9-1.tif", svf_flat, 0.9, 1.0,
                 "SVF r20 noise3 étiré 0,9-1 — paramétrage « flat » complet (celui du CVAT) [VARIANTE]")

    m = 12
    opns_neg10 = rogner(rvt.vis.sky_view_factor(dem=-lire(m), resolution=res,
                                                compute_svf=False, compute_opns=True,
                                                svf_n_dir=16, svf_r_max=10,
                                                ve_factor=1)["opns"], m)
    ecrire_8bits("opns_neg_r10_e60-95_inv.tif", opns_neg10, 60, 95,
                 "Openness négative r10 (creux sombres) — canal fossé [CANAL creux]", inverser=True)
    m = 22
    opns_neg20 = rogner(rvt.vis.sky_view_factor(dem=-lire(m), resolution=res,
                                                compute_svf=False, compute_opns=True,
                                                svf_n_dir=16, svf_r_max=20,
                                                ve_factor=1)["opns"], m)
    ecrire_8bits("opns_neg_r20_e75-95_inv.tif", opns_neg20, 75, 95,
                 "Openness négative r20 étirée flat [VARIANTE]", inverser=True)

    for rmin, rmax in [(5, 10), (10, 20)]:
        m = rmax + 2
        v = rogner(rvt.vis.local_dominance(dem=lire(m), min_rad=rmin, max_rad=rmax,
                                           rad_inc=1, angular_res=15, observer_height=1.7,
                                           ve_factor=1), m)
        ecrire_8bits(f"ld_r{rmin}-{rmax}_e0.5-1.8.tif", v, 0.5, 1.8,
                     f"Local Dominance anneau {rmin}-{rmax} m — "
                     f"{'PRODUCTION maison (acté Sligo 2026-08-05)' if rmin == 5 else 'défaut Hesse/RVT [VARIANTE]'}")

    m = 64
    v = rogner(rvt.vis.msrm(dem=lire(m), resolution=res, feature_min=0, feature_max=20,
                            scaling_factor=2, ve_factor=1), m)
    ecrire_8bits("msrm_f0-20_s2_e2.5.tif", v, -2.5, 2.5,
                 "MSRM défauts RVT ±2,5 m — pour mémoire (redondant SLRM)")

    m = 105
    bloc = lire(m)
    trous = ~np.isfinite(bloc)
    if trous.any() and not trous.all():
        # sky_illumination (pyramides + splines) propage les NaN à TOUTE l'image :
        # boucher les trous au plus proche voisin avant calcul (re-masqués à l'écriture)
        from scipy import ndimage
        bloc = bloc[tuple(ndimage.distance_transform_edt(trous, return_distances=False,
                                                         return_indices=True))]
    v = rvt.vis.sky_illumination(dem=bloc, resolution=res, sky_model="overcast",
                                 compute_shadow=False, max_fine_radius=100,
                                 num_directions=32, ve_factor=1)
    v = rogner(np.asarray(v, dtype="float32"), m)
    if np.isfinite(v).any():
        lo = float(np.nanpercentile(v, 0.25))
        hi = float(np.nanmax(v))
        ecrire_8bits("sky_illum_overcast_PERC.tif", v, lo, hi,
                     "Sky illumination overcast — ATTENTION étirement en PERCENTILES par dalle "
                     "(convention RVT) : NON comparable inter-dalles, contraire à la doctrine")
    else:
        print("  ATTENTION sky_illumination tout-NaN : non écrit")

    # ---------- MSTP : les deux jeux d'échelles ----------
    m = 1012
    v = rogner(rvt.vis.mstp(dem=lire(m), local_scale=(3, 21, 2), meso_scale=(23, 203, 18),
                            broad_scale=(223, 2023, 180), lightness=1.2, ve_factor=1), m)
    ecrire_rgb("mstp_guyot_rgb.tif", v,
               "MSTP échelles Guyot 2021 (3-21/23-203/223-2023 px, lightness 1,2) — "
               "le gagnant de la comparaison CNN sur monuments bretons [CANAL contexte]")
    m = 251
    mstp_dv = rogner(rvt.vis.mstp(dem=lire(m), local_scale=(1, 5, 1), meso_scale=(5, 50, 5),
                                  broad_scale=(50, 500, 50), lightness=0.9, ve_factor=1), m)
    ecrire_rgb("mstp_rvt_rgb.tif", mstp_dv,
               "MSTP échelles DefaultValues RVT (1-5/5-50/50-500 px) — celles de l'e3MSTP [VARIANTE]")

    # ---------- blends (recomposés via blend_func, rvt.blend inutilisable sans osgeo) ----------
    def vat(terrain: str) -> np.ndarray:
        """VAT 4 couches (blender_VAT.json) ; étirements general vs flat de
        default_terrains_settings.json + rvt_blender.py."""
        if terrain == "general":
            svf_n = normalize_image("sky-view factor", svf_r10, 0.7, 1.0, "value")
            opns_n = normalize_image("openness - positive", opns_pos10, 68.0, 93.0, "value")
            pente_n = normalize_image("slope gradient", pente, 0.0, 50.0, "value")
            hs_n = normalize_image("hillshade", np.clip(hs[(315, 35)], 0, 1), 0.0, 1.0, "value")
        else:  # flat
            svf_n = normalize_image("sky-view factor", svf_flat, 0.9, 1.0, "value")
            opns_n = normalize_image("openness - positive", opns_pos10, 85.0, 93.0, "value")
            pente_n = normalize_image("slope gradient", pente, 0.0, 15.0, "value")
            hs_n = normalize_image("hillshade", np.clip(hs[(315, 15)], 0, 1), 0.0, 1.0, "value")
        rendu = hs_n
        rendu = render_images(blend_images("luminosity", pente_n, rendu), rendu, 50)
        rendu = render_images(blend_images("overlay", opns_n, rendu), rendu, 50)
        rendu = render_images(blend_images("multiply", svf_n, rendu), rendu, 25)
        return rendu

    vat_gen, vat_flat = vat("general"), vat("flat")
    ecrire_8bits("vat_general.tif", vat_gen, 0, 1, "VAT terrain general (Kokalj & Somrak 2019)")
    ecrire_8bits("vat_flat.tif", vat_flat, 0, 1, "VAT terrain flat — la composante qui contraste sur plateau")
    cvat = render_images(blend_images("normal", vat_gen, vat_flat), vat_flat, 50)
    ecrire_8bits("cvat_combined.tif", cvat, 0, 1,
                 "CVAT = VAT combined (general 50 % sur flat) — le vrai CVAT de la littérature [CANDIDAT mono-image]")

    # CRIM (blend.py:1248) : pos-neg ±28 en overlay 50 puis luminosity 50, sur pente
    # radians 0-0,8 colorée OrRd
    posneg_n = normalize_image("Openness_Pos-Neg", opns_pos10 - opns_neg10, -28.0, 28.0, "value")
    pente_rad_n = normalize_image("slope rad", np.radians(pente), 0.0, 0.8, "value")
    crim = gray_scale_to_color_ramp(pente_rad_n, "OrRd", 0, 1, output_8bit=False)
    crim = render_images(blend_images("luminosity", posneg_n, crim), crim, 50)
    crim = render_images(blend_images("overlay", posneg_n, crim), crim, 50)
    ecrire_rgb("crim_orrd_rgb.tif", np.clip(crim, 0, 1), "CRIM (colored relief image map, OrRd)")

    # e3MSTP (blend.py:1321) : SLRM r20 ±0,5 screen 25 / CRIM soft_light 70 / MSTP RVT
    slrm20 = rogner(rvt.vis.slrm(dem=lire(22), radius_cell=20, ve_factor=1), 22)
    slrm_n = normalize_image("slrm", slrm20, -0.5, 0.5, "value")
    e3 = mstp_dv
    e3 = render_images(blend_images("soft_light", crim, e3), e3, 70)
    e3 = render_images(blend_images("screen", slrm_n, e3), e3, 25)
    ecrire_rgb("e3mstp_rgb.tif", np.clip(e3, 0, 1),
               "e3MSTP (équivalent RVT du e2MSTP gagnant chez Guyot 2021) [CANDIDAT mono-image]")
    # ponytail: e4MSTP non calculé (4e blend imbriqué, jamais cité gagnant) — l'ajouter
    # ici depuis les composants déjà en cache si la revue visuelle le réclame.

    # ---------- GT ----------
    if a.gpkg:
        import geopandas as gpd
        couches = [a.couche] if a.couche else __import__("fiona").listlayers(a.gpkg)
        for c in couches:
            gdf = gpd.read_file(a.gpkg, layer=c, bbox=tuple(a.emprise))
            if len(gdf):
                gdf.to_file(dossier / "enclos_gt.gpkg", layer=c, driver="GPKG")
                print(f"  enclos_gt.gpkg [{c}] : {len(gdf)} entités")
                manifeste.append((f"enclos_gt.gpkg [{c}]", f"{len(gdf)} entités GT clippées", "vecteur"))

    # ---------- manifeste + auto-vérification ----------
    lignes = [f"# Planche d'indices — {nom}", "",
              f"MNT : `{a.mnt}` — emprise EPSG:2154 : {a.emprise} — {w}x{h} px à {res:g} m/px.",
              "Convention 8 bits : données 0-254, NoData 255, étirements FIXES (sauf mention PERC).", "",
              "| Fichier | Contenu | Normalisation |", "|---|---|---|"]
    lignes += [f"| `{f}` | {n} | {e} |" for f, n, e in manifeste]
    (dossier / "manifeste.md").write_text("\n".join(lignes) + "\n", encoding="utf-8")

    erreurs = []
    for f, _, _ in manifeste:
        if not f.endswith(".tif"):
            continue
        with rasterio.open(dossier / f) as v:
            ech = v.read(out_shape=(v.count, max(1, h // 4), max(1, w // 4)))
            valides = ech[ech != NODATA_OUT] if v.dtypes[0] == "uint8" else ech[ech != v.nodata]
            if (v.width, v.height) != (w, h):
                erreurs.append(f"{f} : taille {v.width}x{v.height} != {w}x{h}")
            if v.crs != src.crs:
                erreurs.append(f"{f} : CRS {v.crs} != {src.crs}")
            if not valides.size:
                erreurs.append(f"{f} : aucune donnée valide")
            elif v.dtypes[0] == "uint8" and valides.std() < 1.0:
                # contraste faible = information pour la revue visuelle, pas un défaut
                # structurel (un étirement large sur terrain plat est plat par nature)
                print(f"  ATTENTION {f} : quasi constant (std {valides.std():.2f})")
    if erreurs:
        sys.exit("NON CONFORME :\n  " + "\n  ".join(erreurs))
    print(f"CONFORME — {len([1 for f, _, _ in manifeste if f.endswith('.tif')])} rasters "
          f"dans {dossier}, manifeste.md écrit")


if __name__ == "__main__":
    main()
