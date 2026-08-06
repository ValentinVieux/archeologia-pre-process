# Téléchargement de dalles LiDAR ouvertes irlandaises (GSI Open Topographic Data)
# depuis l'index local irlande_lidar_dalles_index.gpkg (champ DATA_URL des emprises).
# Ne conserve que le DTM de chaque dalle, converti en GeoTIFF prêt à glisser dans
# QGIS : <out>\<DATA_NAME>.tif. Les zips bruts sont gardés en cache dans <out>\_zips\
# (supprimables). Les fichiers de licence CC-BY sont recopiés à côté des tifs.
# Corpus Irlande : données et tracker sur D:\veille_irlande (cf. suivi_corpus.yaml) ;
# 4 connexions parallèles, reprise idempotente (dalle déjà convertie = sautée).
#
# Usage :
#   .venv\Scripts\python.exe tools\telecharger_dalles_gsi.py --liste [--couche gsi_dchg_dp_multires]
#   .venv\Scripts\python.exe tools\telecharger_dalles_gsi.py --noms 700_770 705_765 [--couche ...]
#                            [--out dossier] [--epsg2154] [--index <gpkg>]
import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.warp import Resampling, calculate_default_transform, reproject

sys.stdout.reconfigure(encoding="utf-8")
# le tool vit dans le repo ; les DONNÉES vivent sur D: (décision utilisateur 2026-08-05)
DONNEES = r"D:\veille_irlande"
INDEX = os.path.join(DONNEES, "irlande_lidar_dalles_index.gpkg")


def telecharger_zip(url, dest, prefixe=""):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        taille = int(r.headers.get("Content-Length", 0))
    if os.path.exists(dest) and os.path.getsize(dest) == taille and taille > 0:
        print(f"  {prefixe} : zip en cache ({taille/1e6:.0f} Mo)")
        return
    print(f"  {prefixe} : telechargement ({taille/1e6:.0f} Mo) ...")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            bloc = r.read(1 << 22)
            if not bloc:
                break
            f.write(bloc)
    if taille and os.path.getsize(tmp) != taille:
        raise IOError(f"taille inattendue : {os.path.getsize(tmp)} != {taille}")
    os.replace(tmp, dest)


def extraire_dtm_en_tif(zip_path, nom, out_tif, out_dir):
    """Extrait le DTM (grille ESRI .adf) du zip et l'écrit en GeoTIFF compressé."""
    with tempfile.TemporaryDirectory(dir=os.path.dirname(zip_path)) as tmpd:
        with zipfile.ZipFile(zip_path) as z:
            # 4 structures rencontrées : Boyne <n>/DTM/<grille ESRI>/ ; Sligo <n>/DTM/<n>.tif ;
            # OPW NASC <n>_DTM.tif à plat ; TII <n>/<n>.tif (un seul raster, pas de mention DTM).
            noms_zip = z.namelist()
            membres = [m for m in noms_zip if "dtm" in m.lower()]
            if not membres:
                membres = [m for m in noms_zip if "dsm" not in m.lower()]
                rasters = [m for m in membres
                           if m.lower().endswith((".tif", ".img", ".asc"))]
                if len(rasters) != 1:
                    raise RuntimeError(
                        f"{os.path.basename(zip_path)} : pas de membre 'DTM' et "
                        f"{len(rasters)} rasters candidats {rasters[:4]} — structure inconnue, à inspecter")
            z.extractall(tmpd, members=membres)
            # licences : recopiées une fois à côté des tifs
            for m in z.namelist():
                base = os.path.basename(m)
                if base and base.lower().endswith(".txt") and ("licen" in base.lower() or "copyright" in base.lower()):
                    cible = os.path.join(out_dir, base)
                    if not os.path.exists(cible):
                        tmp_lic = cible + "." + nom + ".tmp"  # écriture atomique (threads)
                        with z.open(m) as src, open(tmp_lic, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        try:
                            os.replace(tmp_lic, cible)
                        except OSError:
                            os.remove(tmp_lic)
        # le raster DTM : grille ESRI (dossier avec w001001.adf, ex. Boyne) ou
        # fichier direct .tif/.img/.asc (ex. Sligo)
        grille = None
        for dp, _dn, fn in os.walk(tmpd):
            bas = [f.lower() for f in fn]
            if "w001001.adf" in bas:
                grille = dp
                break
            for f in fn:
                if f.lower().endswith((".tif", ".img", ".asc")) and not f.lower().endswith(".aux.xml"):
                    grille = os.path.join(dp, f)
                    break
            if grille:
                break
        if grille is None:
            raise RuntimeError(f"raster DTM introuvable dans {os.path.basename(zip_path)}")
        with rasterio.open(grille) as src:
            print(f"    DTM {src.width}x{src.height} px, res {src.res[0]:g} m")
            rio_copy(src, out_tif, driver="GTiff",
                     compress="deflate", tiled=True, predictor=2)
        # le .prj des grilles GSI est un TM "unnamed" sans code EPSG : ce sont les
        # paramètres exacts de l'ITM (EPSG:2157) — on estampille pour QGIS.
        with rasterio.open(out_tif, "r+") as dst:
            if dst.crs and dst.crs.to_epsg() is None and "Transverse_Mercator" in dst.crs.to_wkt():
                dst.crs = rasterio.crs.CRS.from_epsg(2157)


def reprojeter_2154(src_tif, dst_tif):
    """Copie reprojetée en Lambert-93 (exigence du pipeline du plugin)."""
    with rasterio.open(src_tif) as src:
        transform, w, h = calculate_default_transform(
            src.crs, "EPSG:2154", src.width, src.height, *src.bounds, resolution=src.res)
        profil = src.profile | {"crs": "EPSG:2154", "transform": transform,
                                "width": w, "height": h, "driver": "GTiff",
                                "compress": "deflate", "tiled": True, "predictor": 2}
        with rasterio.open(dst_tif, "w", **profil) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1),
                      resampling=Resampling.bilinear)


def main():
    ap = argparse.ArgumentParser(
        description="Télécharge des dalles LiDAR GSI et n'en garde que le DTM en GeoTIFF.")
    ap.add_argument("--couche", default="gsi_dchg_dp_multires",
                    help="couche de l'index (défaut : gsi_dchg_dp_multires)")
    ap.add_argument("--noms", nargs="+", help="DATA_NAME des dalles à télécharger")
    ap.add_argument("--liste", action="store_true", help="lister les dalles de la couche et sortir")
    ap.add_argument("--out", default=None, help="dossier de sortie (défaut : dalles/<couche>/)")
    ap.add_argument("--epsg2154", action="store_true",
                    help="produire aussi une copie Lambert-93 dans <out>/epsg2154/ (pipeline plugin)")
    ap.add_argument("--index", default=INDEX)
    a = ap.parse_args()

    dal = gpd.read_file(a.index, layer=a.couche)
    if a.liste:
        for _, d in dal.sort_values("DATA_NAME").iterrows():
            print(f"{d.DATA_NAME:14} res {d.get('RESOLUTION', '?'):>6}  {d.get('DATECAPTUR', '')}  {d.DATA_URL}")
        print(f"\n{len(dal)} dalles dans la couche {a.couche}")
        return
    if not a.noms:
        ap.error("--noms requis (ou --liste pour voir les DATA_NAME)")

    out = a.out or os.path.join(DONNEES, "dalles", a.couche)
    zips = os.path.join(out, "_zips")
    os.makedirs(zips, exist_ok=True)
    connus = dict(zip(dal.DATA_NAME.astype(str), dal.DATA_URL))
    inconnus = [n for n in a.noms if n not in connus]
    if inconnus:
        sys.exit(f"DATA_NAME inconnus dans {a.couche} : {inconnus}")

    if a.epsg2154:
        os.makedirs(os.path.join(out, "epsg2154"), exist_ok=True)

    def traiter(n):
        out_tif = os.path.join(out, n + ".tif")
        if not os.path.exists(out_tif):
            zp = os.path.join(zips, n + ".zip")
            telecharger_zip(connus[n], zp, prefixe=n)
            extraire_dtm_en_tif(zp, n, out_tif, out)
            print(f"  {n} : ok -> {out_tif}")
        else:
            print(f"  {n} : deja converti")
        if a.epsg2154:
            tif_l93 = os.path.join(out, "epsg2154", n + ".tif")
            if not os.path.exists(tif_l93):
                reprojeter_2154(out_tif, tif_l93)
                print(f"  {n} : ok (L93)")
        return n

    # ponytail: 4 connexions fixes — politesse envers le serveur GSI ; passer a
    # un argument --workers si un jour un miroir plus costaud le justifie.
    echecs = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(traiter, n): n for n in a.noms}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                echecs.append((futs[f], e))
                print(f"  {futs[f]} : ECHEC — {e}")
    if echecs:
        sys.exit(f"{len(echecs)} dalle(s) en echec : {[n for n, _ in echecs]} — relancer la meme commande")

    print("\nTermine. GeoTIFF DTM prets pour QGIS dans :", out)
    print("(les zips bruts sont en cache dans", zips, "- supprimables)")


if __name__ == "__main__":
    main()
