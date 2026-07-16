"""Génère tests/fixture_dataset/ : mini-livraison synthétique couvrant les cas limites.

Non commitée (gitignorée) : régénérable en quelques secondes via geopandas.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import Point, Polygon

FIXTURE = Path(__file__).with_name("fixture_dataset")

QGS_XML = """<?xml version="1.0" encoding="utf-8"?>
<qgis version="3.34">
  <layer-tree-group>
    <layer-tree-group name="Prospection Morvan">
      <layer-tree-layer name="Charbonnières validées" id="l1"/>
      <layer-tree-layer name="Fours à chaux" id="l2"/>
    </layer-tree-group>
  </layer-tree-group>
</qgis>
"""


def make() -> Path:
    if FIXTURE.exists():
        shutil.rmtree(FIXTURE)
    vec = FIXTURE / "vecteurs"
    vec.mkdir(parents=True)

    # GPKG 2 couches : points EPSG:2154 (dont une géométrie nulle) + polygone invalide (bowtie)
    pts = gpd.GeoDataFrame(
        {"type": ["charbonnière", "charbonnière", "four", "indéterminé"]},
        geometry=[Point(812000, 6712000), Point(812100, 6712050), Point(812200, 6712100), None],
        crs="EPSG:2154")
    pts.to_file(vec / "sites.gpkg", layer="charbonnieres", driver="GPKG")
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    gpd.GeoDataFrame({"nom": ["zone1"]}, geometry=[bowtie], crs="EPSG:2154") \
        .to_file(vec / "sites.gpkg", layer="zones", driver="GPKG")

    # Table attributaire GPKG sans géométrie ni champ string (couche non spatiale)
    pyogrio.write_dataframe(pd.DataFrame({"mesure_m": [1.5, 2.5], "annee": [1917, 1918]}),
                            vec / "tables.gpkg", layer="mesures")

    # Shapefile sans .prj ni .cpg, DBF en cp1252 (accents français)
    gpd.GeoDataFrame({"nature": ["fossé", "talus"]},
                     geometry=[Point(1, 1), Point(2, 2)], crs="EPSG:2154") \
        .to_file(vec / "no_prj.shp", encoding="cp1252")
    (vec / "no_prj.prj").unlink()
    (vec / "no_prj.cpg").unlink(missing_ok=True)

    # Shapefile UTF-8 sans .cpg (le cas Bretagne : utf-8 doit être tenté avant cp1252)
    gpd.GeoDataFrame({"type_site": ["mégalithe", "éperon barré"]},
                     geometry=[Point(3, 3), Point(4, 4)], crs="EPSG:2154") \
        .to_file(vec / "utf8_no_cpg.shp", encoding="utf-8")
    (vec / "utf8_no_cpg.cpg").unlink(missing_ok=True)

    # GeoJSON hors CRS de référence
    gpd.GeoDataFrame({"cat": ["tumulus"]}, geometry=[Point(2.3, 48.8)], crs="EPSG:4326") \
        .to_file(FIXTURE / "features.geojson", driver="GeoJSON")

    # Cas dégradés / divers
    (vec / "corrompu.shp").write_bytes(b"\x00" * 100)
    (vec / "corrompu.dbf").write_bytes(b"\x00" * 50)        # sidecar appariée au corrompu
    (FIXTURE / "orphelin.dbf").write_bytes(b"\x00" * 50)    # sidecar orpheline
    (FIXTURE / "dem.tif").write_bytes(b"II*\x00" + b"\x00" * 60)
    (FIXTURE / "notes.txt").write_text("compte-rendu de prospection", encoding="utf-8")
    (FIXTURE / "data.json").write_text('{"pas": "du geojson"}', encoding="utf-8")
    with zipfile.ZipFile(FIXTURE / "envoi.zip", "w"):
        pass
    with zipfile.ZipFile(FIXTURE / "projet.qgz", "w") as z:
        z.writestr("projet.qgs", QGS_XML)
    return FIXTURE


if __name__ == "__main__":
    print(make())
