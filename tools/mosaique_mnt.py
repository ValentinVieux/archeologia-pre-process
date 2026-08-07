"""Mosaïque un dossier de dalles MNT téléchargées en UN GeoTIFF EPSG:2154 (passe unique).

Pourquoi : reprojeter les dalles une par une avec une emprise commune (batch QGIS,
veille Irlande 2026-08) donne N rasters portant chacun le canevas union de la zone
(99 % de NoData chacun) et des bandes sombres aux joints des indices RVT. La passe
unique gdalwarp -tap produit une seule mosaïque sur grille propre, sans joints.
Ne JAMAIS reprojeter dalle par dalle : toujours passer par ce script.

Usage : python tools\\mosaique_mnt.py <dossier_dalles> <sortie.tif> [--tr 1.0]

La sortie va dans un dossier ne contenant QUE la mosaïque (c'est lui qu'on pointe
en mode « MNT existant » du plugin). Le NoData doit être tagué dans les dalles
sources (gdalwarp le propage) ; sinon le taguer d'abord (gdal_edit -a_nodata).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def gdal_exe(nom: str) -> str:
    return shutil.which(nom) or rf"C:\OSGeo4W\bin\{nom}.exe"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dossier", type=Path, help="dossier des dalles téléchargées (.tif/.asc)")
    p.add_argument("sortie", type=Path, help="GeoTIFF mosaïque à produire")
    p.add_argument("--tr", type=float, default=1.0, help="résolution cible en m (défaut 1)")
    a = p.parse_args()

    dalles = sorted(a.dossier.glob("*.tif")) + sorted(a.dossier.glob("*.asc"))
    if not dalles:
        sys.exit(f"aucune dalle .tif/.asc dans {a.dossier}")
    # une dalle sans CRS ferait "reprojeter" gdalwarp sans transformation (mosaïque
    # étiquetée 2154 avec des coordonnées natives fausses, Kerry 2026-08-06) et un
    # CRS sans code EPSG (LOCAL_CS Galway 2026-08-07) fait échouer gdalwarp : refus net.
    import rasterio
    def _crs_ko(d):
        crs = rasterio.open(d).crs
        return crs is None or crs.to_epsg() is None
    sans_crs = [d.name for d in dalles if _crs_ko(d)]
    if sans_crs:
        sys.exit(f"{len(sans_crs)} dalle(s) sans CRS résolvable EPSG ({sans_crs[:3]}...) — "
                 "estampiller d'abord (telecharger_dalles_gsi le fait pour l'ITM) ; refus de mosaïquer")
    a.sortie.parent.mkdir(parents=True, exist_ok=True)

    print(f"{len(dalles)} dalles -> {a.sortie}")
    subprocess.run(
        [gdal_exe("gdalwarp"), "-overwrite", "-t_srs", "EPSG:2154",
         "-tr", str(a.tr), str(a.tr), "-tap", "-r", "bilinear",
         "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
         "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER",
         *map(str, dalles), str(a.sortie)],
        check=True,
    )

    # auto-vérification : grille propre, CRS cible, NoData propagé
    info = json.loads(subprocess.run(
        [gdal_exe("gdalinfo"), "-json", str(a.sortie)],
        check=True, capture_output=True, text=True).stdout)
    gt = info["geoTransform"]
    assert abs(gt[1] - a.tr) < 1e-9 and abs(gt[5] + a.tr) < 1e-9, f"résolution {gt[1]}"
    assert '"EPSG",2154' in info["coordinateSystem"]["wkt"].replace(" ", ""), "CRS != 2154"
    nodata = info["bands"][0].get("noDataValue")
    assert nodata is not None, "NoData absent de la sortie (taguer les sources ?)"
    print(f"CONFORME — {info['size'][0]}x{info['size'][1]} px, {a.tr} m/px, "
          f"EPSG:2154, NoData {nodata}")


if __name__ == "__main__":
    main()
