"""Auto-test de l'API de tools/review_recalage (TestClient, sans navigateur)."""
import json
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from affine import Affine
from fastapi.testclient import TestClient
from shapely import wkt
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.review_recalage.__main__ import creer_app

tmp = Path(tempfile.mkdtemp(prefix="review_test_"))

# Raster 200x200 px à 0,5 m/px
raster = tmp / "ld.tif"
with rasterio.open(raster, "w", driver="GTiff", width=200, height=200, count=1,
                   dtype="uint8", crs="EPSG:2154",
                   transform=Affine(0.5, 0, 600000, 0, -0.5, 6700100)) as dst:
    dst.write(np.full((200, 200), 120, dtype="uint8"), 1)

# GPKG recalé minimal : 3 lignes (a_revoir, auto_ok x2)
lignes = [LineString([(600010, 6700050), (600040, 6700050)]),
          LineString([(600010, 6700030), (600040, 6700030)]),
          LineString([(600010, 6700070), (600040, 6700072)])]
gdf = gpd.GeoDataFrame({
    "id_recalage": ["parcellaire_0", "parcellaire_1", "parcellaire_2"],
    "statut_recalage": ["a_revoir", "auto_ok", "auto_ok"],
    "score": [12.0, 80.0, 65.0],
    "geom_origine": [l.wkt for l in lignes],
    "polarite_retenue": ["clair"] * 3, "pts_nets_pct": [30.0, 95.0, 90.0],
    "ambigus_pct": [40.0, 2.0, 5.0], "contraste": [40.0, 84.0, 72.0],
    "offset_median_m": [2.1, 1.2, 0.8], "offset_max_m": [5.0, 2.0, 1.5],
    "residu_m": [1.4, 0.2, 0.3]},
    geometry=[l.parallel_offset(1.0, "left") for l in lignes], crs="EPSG:2154")
gpkg = tmp / "jouet_entites_l93_recale.gpkg"
gdf.to_file(gpkg, layer="parcellaire", driver="GPKG")

decisions = tmp / "recalage_decisions_jouet.yaml"
client = TestClient(creer_app(gpkg, raster, decisions))

# Liste : périmètre par défaut = a_revoir + échantillon auto_ok
r = client.get("/api/lignes").json()
ids = {l["id"] for l in r["lignes"]}
assert "parcellaire_0" in ids and r["couches"] == ["parcellaire"]
assert r["zone"] == "jouet"
assert all(l["echantillon"] for l in r["lignes"] if l["statut"] == "auto_ok")
scores = [l["score"] for l in r["lignes"]]
assert scores == sorted(scores)  # pires d'abord

# Détail + crop
d = client.get("/api/ligne/parcellaire_0").json()
assert d["mesures"]["pts_nets_pct"] == 30.0
assert len(d["origine"][0]) == 2 and d["editee"] is None
crop = client.get("/api/crop/parcellaire_0")
assert crop.status_code == 200 and crop.headers["content-type"] == "image/png"
affine = json.loads(crop.headers["X-Affine"])
assert abs(affine[0] - 0.5) < 1e-9
assert client.get("/api/crop/inconnu").status_code == 404

# Décision : écriture IMMÉDIATE dans le YAML
r = client.post("/api/decision", json={"id": "parcellaire_0",
                                       "decision": "recale"}).json()
assert r["ok"] and yaml.safe_load(decisions.read_text(encoding="utf-8"))[
    "parcellaire_0"]["decision"] == "recale"

# Édition : géométrie monde -> WKT dans la décision
geom = [[[600012.0, 6700052.0], [600043.0, 6700051.0]]]
client.post("/api/decision", json={"id": "parcellaire_0", "decision": "editee",
                                   "geometrie": geom})
doc = yaml.safe_load(decisions.read_text(encoding="utf-8"))
g = wkt.loads(doc["parcellaire_0"]["geometrie_editee"])
assert g.geom_type == "LineString" and abs(g.length - 31.0) < 0.2
assert client.post("/api/decision", json={"id": "parcellaire_0",
                                          "decision": "bof"}).status_code == 422

# Reprise : nouvelle instance -> décisions rechargées
client2 = TestClient(creer_app(gpkg, raster, decisions))
d2 = client2.get("/api/ligne/parcellaire_0").json()
assert d2["decision"] == "editee" and d2["editee"] is not None
p = client2.get("/api/progression").json()
assert p["decidees_total"] == 1 and p["par_decision"]["editee"] == 1

# ---------------------------------------------------------------------------
# analyse_corrections sur ces décisions + décisions fabriquées
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyse_corrections import analyser, typologie_edition

client2.post("/api/decision", json={"id": "parcellaire_1",
                                    "decision": "original"})
client2.post("/api/decision", json={"id": "parcellaire_2", "decision": "exclue"})

# Voisines : géométrie ACTIVE (éditée), les exclues disparaissent du contexte
d1 = client2.get("/api/ligne/parcellaire_1").json()
voisines = {v["id"]: v for v in d1["voisines"]}
assert "parcellaire_0" in voisines and "parcellaire_2" not in voisines
assert voisines["parcellaire_0"]["couche"] == "parcellaire"
assert abs(voisines["parcellaire_0"]["parts"][0][0][0] - 600012.0) < 1e-6  # éditée
assert voisines["parcellaire_0"]["decision"] == "editee"
assert voisines["parcellaire_0"]["origine"][0][0] == [600010.0, 6700050.0]
assert voisines["parcellaire_0"]["statut"] == "a_revoir"
assert isinstance(voisines["parcellaire_0"]["echantillon"], bool)

rapport = analyser(decisions, gpkg)
parc = rapport["couches"]["parcellaire"]
assert rapport["decisions_total"] == 3
assert parc["decisions"] == {"editee": 1, "original": 1, "exclue": 1}
assert parc["distance_edition_mediane_m"] is not None
# parcellaire_1 gardée en original malgré 95 % de points nets -> comptée
typ, _ = typologie_edition(lignes[0].parallel_offset(3.0, "left"),
                           lignes[0].parallel_offset(1.0, "left"), lignes[0])
assert typ == "translation_residuelle", typ
typ2, _ = typologie_edition(lignes[0], lignes[0].parallel_offset(2.0, "left"),
                            lignes[0])
assert typ2 == "recalage_nuisible", typ2
print("api review + analyse corrections : OK")
