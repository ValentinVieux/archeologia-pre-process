"""Génère l'indice LD (Local Dominance) 8 bits d'un MNT, paramétré PAR RÉSOLUTION.

Règle actée (mesures Sligo 2026-08-05) : l'opérateur LD travaille en PIXELS mais
doit couvrir un anneau MÉTRIQUE constant de 5-10 m — celui de la chaîne française
(0,5 m/px -> Rmin10/Rmax20). Les rayons sont donc calculés depuis la résolution du
raster : rmin = round(5/res), rmax = round(10/res) — 1 m/px -> 5/10 ; 2 m/px -> 3/5
(validé par mesure, distribution alignée sur la référence française). Refus si
rmin < 2 px (résolution trop grossière pour l'anneau).

Rendu identique au pipeline du plugin : angular_res 15°, observateur 1,7 m, VE 1,
étirement 8 bits FIXE 0,5-1,8 (comparabilité inter-zones), NoData 255. Traitement
par tuiles de 4096 px avec marge rmax (les grosses mosaïques passent en RAM bornée).
Auto-vérification : taille/CRS identiques à l'entrée, médiane des valeurs valides
dans la gamme attendue (90-110), impression CONFORME.

Usage : .venv\\Scripts\\python.exe tools\\generer_ld.py <mnt.tif> <sortie_ld.tif>

Dépendance : bibliothèque rvt (celle du plugin rvt-qgis du profil QGIS, repli
`import rvt` si installée dans l'environnement).
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

TUILE = 4096
ANNEAU_M = (5.0, 10.0)          # anneau métrique constant (chaîne française)
ETIREMENT = (0.5, 1.8)          # bornes FIXES du 8 bits (jamais par scène)
NODATA_OUT = 255


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mnt", type=Path)
    ap.add_argument("sortie", type=Path)
    a = ap.parse_args()

    src = rasterio.open(a.mnt)
    res = abs(src.res[0])
    rmin, rmax = max(round(ANNEAU_M[0] / res), 1), max(round(ANNEAU_M[1] / res), 2)
    if rmin < 2:
        sys.exit(f"résolution {res} m/px trop grossière : rmin={rmin} px < 2 (anneau 5-10 m) — "
                 "générer le LD n'aurait pas de sens, sur-échantillonner le MNT d'abord")
    print(f"MNT {src.width}x{src.height} px à {res:g} m/px -> LD Rmin{rmin}/Rmax{rmax} px "
          f"(anneau {rmin*res:g}-{rmax*res:g} m)")

    profil = src.profile | {"dtype": "uint8", "nodata": NODATA_OUT, "count": 1,
                            "compress": "deflate", "tiled": True, "BIGTIFF": "IF_SAFER"}
    a.sortie.parent.mkdir(parents=True, exist_ok=True)
    marge = rmax + 1
    with rasterio.open(a.sortie, "w", **profil) as dst:
        for r0 in range(0, src.height, TUILE):
            for c0 in range(0, src.width, TUILE):
                h = min(TUILE, src.height - r0)
                w = min(TUILE, src.width - c0)
                rl, cl = max(0, r0 - marge), max(0, c0 - marge)
                rh = min(src.height, r0 + h + marge)
                ch = min(src.width, c0 + w + marge)
                bloc = src.read(1, window=rasterio.windows.Window(cl, rl, ch - cl, rh - rl)).astype("float32")
                # NoData float64-max : inf apres cast float32, echappe au == (bug vu
                # sur les SLRM galway_b_*) — neutraliser tout non-fini d'abord
                bloc[~np.isfinite(bloc)] = np.nan
                if src.nodata is not None:
                    bloc[bloc == np.float32(src.nodata)] = np.nan
                if not np.isfinite(bloc).any():
                    sortie = np.full((h, w), NODATA_OUT, dtype="uint8")
                else:
                    ld = rvt.vis.local_dominance(dem=bloc, min_rad=rmin, max_rad=rmax, rad_inc=1,
                                                 angular_res=15, observer_height=1.7, ve_factor=1)
                    ld = np.clip(ld, *ETIREMENT)
                    b = ((ld - ETIREMENT[0]) / (ETIREMENT[1] - ETIREMENT[0]) * 255).round()
                    b = np.where(np.isfinite(b), b, NODATA_OUT)
                    b = np.clip(b, 0, 254)  # 255 réservé NoData (collision connue du pipeline, évitée ici)
                    b[~np.isfinite(bloc)] = NODATA_OUT
                    sortie = b[r0 - rl:r0 - rl + h, c0 - cl:c0 - cl + w].astype("uint8")
                dst.write(sortie, 1, window=rasterio.windows.Window(c0, r0, w, h))
            print(f"  lignes {min(r0 + TUILE, src.height)}/{src.height}")

    # auto-vérification
    with rasterio.open(a.sortie) as v:
        assert (v.width, v.height) == (src.width, src.height), "taille sortie != entrée"
        assert v.crs == src.crs, "CRS sortie != entrée"
        ech = v.read(1, out_shape=(max(1, v.height // 8), max(1, v.width // 8)))
        valides = ech[ech != NODATA_OUT]
        med = float(np.median(valides)) if valides.size else float("nan")
        assert valides.size and 80 <= med <= 120, f"médiane LD {med} hors gamme attendue (80-120)"
    print(f"CONFORME — {a.sortie} : Rmin{rmin}/Rmax{rmax}, médiane {med:.0f}, NoData {NODATA_OUT}, CRS {src.crs.to_epsg()}")


if __name__ == "__main__":
    main()
