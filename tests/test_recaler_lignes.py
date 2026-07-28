"""Auto-test de tools/recaler_lignes.py — raster synthétique à structures connues."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from recaler_lignes import (LecteurRaster, densifier, extremum_profil,
                            noeuds_partages, recaler_ligne, regulariser)

# ---------------------------------------------------------------------------
# Fixture : raster 1200x800 px (0,5 m/px, origine (600000, 6700000)) avec une
# CRÊTE claire le long d'une sinusoïde connue et un CREUX sombre le long d'une
# autre ; fond à 100.
# ---------------------------------------------------------------------------
L, H = 1200, 800
T = Affine(0.5, 0, 600000, 0, -0.5, 6700000)
fond = np.full((H, L), 100.0)

def courbe_crete(x):   # y monde en fonction de x monde
    return 6699900.0 + 8.0 * np.sin(2 * np.pi * (x - 600050.0) / 200.0)

def courbe_creux(x):
    return 6699700.0 + 6.0 * np.sin(2 * np.pi * (x - 600080.0) / 260.0)

xs = np.arange(600050.0, 600520.0, 0.25)
from scipy.ndimage import distance_transform_edt

def marquer(courbe):
    masque = np.ones((H, L), dtype=bool)
    cols = ((xs - T.c) / T.a).astype(int)
    rows = ((courbe(xs) - T.f) / T.e).astype(int)
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < L)
    masque[rows[ok], cols[ok]] = False
    return distance_transform_edt(masque) * 0.5  # distance en mètres

d_crete = marquer(courbe_crete)
d_creux = marquer(courbe_creux)
img = fond + 80.0 * np.exp(-d_crete**2 / (2 * 2.0**2)) \
           - 70.0 * np.exp(-d_creux**2 / (2 * 2.0**2))
img = np.clip(img, 0, 255).astype("uint8")

tmp = Path(tempfile.mkdtemp(prefix="recalage_test_"))
raster_path = tmp / "ld.tif"
with rasterio.open(raster_path, "w", driver="GTiff", width=L, height=H, count=1,
                   dtype="uint8", crs="EPSG:2154", transform=T) as dst:
    dst.write(img, 1)
lecteur = LecteurRaster(raster_path)

verite_crete = LineString([(x, courbe_crete(x)) for x in np.arange(600050, 600500, 2.0)])
verite_creux = LineString([(x, courbe_creux(x)) for x in np.arange(600080, 600480, 2.0)])

PARAMS = {"polarite": "clair", "fenetre_m": 8.0, "pas_m": 2.0,
          "seuil_contraste": 10.0, "seuil_ambiguite": 0.7,
          "poids_derivee": 4.0, "pas_echant_m": 0.25}

# ---------------------------------------------------------------------------
# Unités
# ---------------------------------------------------------------------------
pts, normales, absc = densifier(LineString([(0, 0), (10, 0), (10, 10)]), 2.0)
assert abs(absc[-1] - 20.0) < 1e-6 and len(pts) == len(normales) == len(absc)
assert np.allclose(np.linalg.norm(normales, axis=1), 1.0)
assert np.allclose(pts[0], (0, 0)) and np.allclose(pts[-1], (10, 10))

profil = np.array([100, 100, 120, 180, 130, 100, 100], dtype=float)
off, contraste, ambigu = extremum_profil(profil, "clair", 10.0, 0.7, pas_echant_m=1.0)
assert abs(off - (3 - 3)) < 0.5  # extremum au centre (index 3 sur fenêtre ±3)
assert contraste > 50 and not ambigu
off2, _, _ = extremum_profil(255 - profil, "sombre", 10.0, 0.7, pas_echant_m=1.0)
assert abs(off2 - off) < 1e-6
plat = np.full(33, 100.0)
off3, c3, _ = extremum_profil(plat, "clair", 10.0, 0.7, pas_echant_m=0.5)
assert off3 is None and c3 < 1.0  # aucun signal

o = np.array([np.nan, 2.0, 2.2, np.nan, 2.4, 30.0, 2.6, np.nan, 3.0])
d = regulariser(o, poids_derivee=4.0, ancres={})
assert np.all(np.isfinite(d)) and abs(d[0] - d[1]) < 1.5
assert d[5] < 15.0  # l'aberration à 30 est fortement amortie
d2 = regulariser(o, poids_derivee=4.0, ancres={0: 0.0})
assert abs(d2[0]) < 0.01  # ancre respectée

# ---------------------------------------------------------------------------
# Recalage d'une ligne décalée de +6 m (crête claire)
# ---------------------------------------------------------------------------
decalee = LineString([(x, courbe_crete(x) + 6.0) for x in np.arange(600060, 600480, 40.0)])
recalee, mesures = recaler_ligne(decalee, lecteur, PARAMS)
dists = [recalee.interpolate(t, normalized=True).distance(verite_crete)
         for t in np.linspace(0.05, 0.95, 40)]
assert float(np.median(dists)) < 0.25, f"médiane {np.median(dists):.2f} m"
assert mesures["pts_nets_pct"] > 80 and 4.0 < mesures["offset_median_m"] < 8.0

# Ligne grossière (3 sommets) : la forme est ré-épousée, pas seulement translatée
grossiere = LineString([(600060, courbe_crete(600060) + 4), (600260, 6699900 + 4),
                        (600460, courbe_crete(600460) + 4)])
recalee_g, mesures_g = recaler_ligne(grossiere, lecteur, PARAMS)
dists_g = [recalee_g.interpolate(t, normalized=True).distance(verite_crete)
           for t in np.linspace(0.05, 0.95, 40)]
assert float(np.median(dists_g)) < 1.0, f"médiane grossière {np.median(dists_g):.2f} m"

# Polarité auto sur le creux sombre
creux_decale = LineString([(x, courbe_creux(x) - 5.0) for x in np.arange(600090, 600460, 30.0)])
params_auto = dict(PARAMS, polarite="auto")
recalee_c, mesures_c = recaler_ligne(creux_decale, lecteur, params_auto)
assert mesures_c["polarite_retenue"] == "sombre"
dists_c = [recalee_c.interpolate(t, normalized=True).distance(verite_creux)
           for t in np.linspace(0.05, 0.95, 40)]
assert float(np.median(dists_c)) < 0.5

# Ligne loin de tout signal : intacte
loin = LineString([(600100, 6699500), (600300, 6699500)])
recalee_l, mesures_l = recaler_ligne(loin, lecteur, PARAMS)
assert recalee_l.equals(loin) and mesures_l["pts_nets_pct"] == 0

# Déterminisme
r2, _ = recaler_ligne(decalee, lecteur, PARAMS)
assert r2.equals(recalee)

# ---------------------------------------------------------------------------
# Nœuds partagés
# ---------------------------------------------------------------------------
import geopandas as gpd
a = LineString([(600200, courbe_crete(600200) + 5), (600300, courbe_crete(600300) + 5)])
b = LineString([(600300, courbe_crete(600300) + 5), (600400, courbe_crete(600400) + 5)])
gdfs = {"c1": gpd.GeoDataFrame(geometry=[a], crs="EPSG:2154"),
        "c2": gpd.GeoDataFrame(geometry=[b], crs="EPSG:2154")}
noeuds = noeuds_partages(gdfs, tol=0.5)
partages = [n for n, membres in noeuds.items() if len(membres) >= 2]
assert len(partages) == 1, noeuds

# ---------------------------------------------------------------------------
# Intégration : pipeline complet sur mini GPKG 2 couches
# ---------------------------------------------------------------------------
import yaml
from shapely import wkt
from recaler_lignes import run_recalage

gpkg_path = tmp / "jouet_entites_l93.gpkg"
gpd.GeoDataFrame({"src": ["a", "b"]}, geometry=[decalee, loin],
                 crs="EPSG:2154").to_file(gpkg_path, layer="parcellaire", driver="GPKG")
gpd.GeoDataFrame({"src": ["c"]}, geometry=[creux_decale],
                 crs="EPSG:2154").to_file(gpkg_path, layer="talus_fosse", driver="GPKG")
cfg_path = tmp / "recalage_jouet.yaml"
cfg_path.write_text(yaml.safe_dump({
    "zone": "jouet", "raster_gsd_attendu": 0.5,
    "couches": {"parcellaire": {"polarite": "clair"},
                "talus_fosse": {"polarite": "auto"}},
    "lissage": {"poids_derivee": 4.0},
    "seuil_points_nets": 5, "seuil_ambiguite": 0.7}), encoding="utf-8")
out = tmp / "out"
run_recalage(cfg_path, gpkg_path, raster_path, out)

gpkg_out = out / "jouet_entites_l93_recale.gpkg"
parc = gpd.read_file(gpkg_out, layer="parcellaire")
tf = gpd.read_file(gpkg_out, layer="talus_fosse")
assert len(parc) == 2 and len(tf) == 1  # zéro ligne perdue/ajoutée
ligne_ok = parc[parc["src"] == "a"].iloc[0]
ligne_loin = parc[parc["src"] == "b"].iloc[0]
assert ligne_ok["statut_recalage"] == "auto_ok"
assert ligne_loin["statut_recalage"] == "sans_signal"
assert wkt.loads(ligne_loin["geom_origine"]).equals(ligne_loin.geometry)  # intacte
assert wkt.loads(ligne_ok["geom_origine"]).equals(decalee)  # origine conservée
assert ligne_ok.geometry.distance(verite_crete) < 1.0
assert tf.iloc[0]["polarite_retenue"] == "sombre"
assert {"score", "offset_median_m", "pts_nets_pct", "id_recalage"} <= set(parc.columns)

rapport = yaml.safe_load((out / "recalage_rapport.yaml").read_text(encoding="utf-8"))
assert rapport["couches"]["parcellaire"]["statuts"]["auto_ok"] == 1
assert rapport["couches"]["parcellaire"]["statuts"]["sans_signal"] == 1
assert rapport["parametres"]["couches"]["parcellaire"]["polarite"] == "clair"

run_recalage(cfg_path, gpkg_path, raster_path, out)  # re-run : écrase, ne plante pas
parc2 = gpd.read_file(gpkg_out, layer="parcellaire")
assert parc2[parc2["src"] == "a"].iloc[0].geometry.equals(ligne_ok.geometry)  # déterminisme
print("noyau + pipeline recalage : OK")
