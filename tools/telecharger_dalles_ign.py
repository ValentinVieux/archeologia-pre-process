"""Télécharge des dalles MNT LiDAR HD IGN (GeoTIFF 1 km, 0,5 m) via la grille WFS.

Méthode documentée (CLAUDE.md § Rasters externes) : la grille WFS
`IGNF_MNT-LIDAR-HD:dalle` de data.geopf.fr fournit pour chaque dalle l'URL WMS
GetMap GeoTIFF exacte (BBOX/2000x2000) — on télécharge CES URLs, jamais de
découpe maison. Sélection des dalles : les cellules 1 km contenant au moins une
entité du GPKG fourni, plus un anneau de N voisines (--anneau, défaut 1).

Reprise idempotente (dalle déjà téléchargée et lisible = sautée). Après chaque
téléchargement : ouverture rasterio, contrôle 2000x2000 + CRS ; si le CRS n'est
pas résolvable en EPSG:2154, estampillage `r+` (piège WKT custom mesuré).

Usage :
  .venv\\Scripts\\python.exe tools\\telecharger_dalles_ign.py <entites.gpkg> <dossier_sortie>
      [--couche <nom>] [--anneau 1] [--mt 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.crs import CRS

WFS = ("https://data.geopf.fr/wfs/ows?service=WFS&version=2.0.0&request=GetFeature"
       "&typeName=IGNF_MNT-LIDAR-HD:dalle&outputFormat=application/json&count=1000")


def cellules_cibles(gpkg: Path, couche: str | None, anneau: int) -> set[tuple[int, int]]:
    g = gpd.read_file(gpkg, layer=couche)
    if g.crs is None or g.crs.to_epsg() != 2154:
        sys.exit(f"{gpkg} : CRS {g.crs} != EPSG:2154")
    avec = {(int(p.x // 1000), int(p.y // 1000)) for p in g.geometry.centroid}
    cibles = set(avec)
    for x, y in avec:
        for dx in range(-anneau, anneau + 1):
            for dy in range(-anneau, anneau + 1):
                cibles.add((x + dx, y + dy))
    print(f"{len(g)} entités -> {len(avec)} dalles occupées, {len(cibles)} avec anneau {anneau}")
    return cibles


def dalles_wfs(cibles: set[tuple[int, int]]) -> list[dict]:
    xs = [c[0] for c in cibles]
    ys = [c[1] for c in cibles]
    bbox = f"{min(xs) * 1000},{(min(ys)) * 1000},{(max(xs) + 1) * 1000},{(max(ys) + 1) * 1000}"
    feats, start = [], 0
    while True:
        u = f"{WFS}&bbox={urllib.parse.quote(bbox + ',urn:ogc:def:crs:EPSG::2154')}&startIndex={start}"
        with urllib.request.urlopen(u, timeout=120) as r:
            page = json.load(r)["features"]
        feats += page
        if len(page) < 1000:
            break
        start += 1000
    retenues = []
    for f in feats:
        nw = f["properties"]["metadata"]
        nw = json.loads(nw)["coordonnees_NW"] if isinstance(nw, str) else nw["coordonnees_NW"]
        x, y = (int(v) for v in nw.split("-"))
        if (x, y - 1) in cibles:  # coin NW -> cellule couverte = (x, y-1)
            retenues.append({"nom": f["properties"]["name_download"], "url": f["properties"]["url"]})
    print(f"grille WFS : {len(feats)} dalles publiées dans l'emprise, {len(retenues)} retenues")
    return retenues


def telecharger(d: dict, dossier: Path) -> str | None:
    """-> None si OK, message d'erreur sinon."""
    cible = dossier / d["nom"]
    try:
        if cible.exists():
            with rasterio.open(cible):
                return None  # déjà là et lisible
    except Exception:
        cible.unlink()
    tmp = cible.with_suffix(".part")
    try:
        with urllib.request.urlopen(d["url"], timeout=300) as r, open(tmp, "wb") as f:
            f.write(r.read())
        with rasterio.open(tmp) as src:
            if src.width != 2000 or src.height != 2000:
                return f"{d['nom']} : {src.width}x{src.height} != 2000x2000"
            crs_ko = src.crs is None or src.crs.to_epsg() != 2154
        if crs_ko:
            with rasterio.open(tmp, "r+") as src:
                src.crs = CRS.from_epsg(2154)  # estampillage du vrai code (piège WKT custom)
        tmp.rename(cible)
        return None
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return f"{d['nom']} : {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gpkg", type=Path, help="GPKG d'entités EPSG:2154 (sélection des dalles)")
    ap.add_argument("sortie", type=Path)
    ap.add_argument("--couche", default=None)
    ap.add_argument("--anneau", type=int, default=1)
    ap.add_argument("--mt", type=int, default=4)
    a = ap.parse_args()
    a.sortie.mkdir(parents=True, exist_ok=True)

    dalles = dalles_wfs(cellules_cibles(a.gpkg, a.couche, a.anneau))
    if not dalles:
        sys.exit("aucune dalle à télécharger")
    echecs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.mt) as ex:
        for i, err in enumerate(ex.map(lambda d: telecharger(d, a.sortie), dalles)):
            if err:
                echecs.append(err)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(dalles)} dalles")
    ok = len(dalles) - len(echecs)
    print(f"{ok}/{len(dalles)} dalles OK dans {a.sortie}")
    if echecs:
        print("ÉCHECS :", *echecs[:10], sep="\n  ")
        sys.exit(f"{len(echecs)} échec(s) — relancer la même commande (reprise idempotente)")


if __name__ == "__main__":
    main()
