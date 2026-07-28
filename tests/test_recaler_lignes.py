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

# Pénalité de distance : pic faible proche (60 à +1 m) vs fort lointain (80 à
# +6 m) — sans pénalité on capture le voisin fort, avec on reste sur le proche
double = np.full(17, 100.0)
double[9] = 160.0   # +1 m (pas 1 m, centre à l'index 8)
double[14] = 180.0  # +6 m
off_sans, _, _ = extremum_profil(double, "clair", 10.0, 0.2, pas_echant_m=1.0)
assert abs(off_sans - 6.0) < 0.6, off_sans
off_avec, c_avec, _ = extremum_profil(double, "clair", 10.0, 0.2,
                                      pas_echant_m=1.0, poids_distance=8.0)
assert abs(off_avec - 1.0) < 0.6, off_avec
assert c_avec > 50  # le contraste rapporté reste celui du pic brut

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
# Couloir partagé : deux lignes proches ne doivent pas fusionner sur le même
# signal — le profil est tronqué à mi-distance de la voisine
# ---------------------------------------------------------------------------
from shapely.strtree import STRtree
from recaler_lignes import bornes_laterales

L2, H2 = 600, 100
T2 = Affine(0.5, 0, 610000, 0, -0.5, 6710000)
ys = 6710000 - (np.arange(H2) + 0.5) * 0.5
d_fort = np.abs(ys - 6709975.0)[:, None] * np.ones((1, L2))   # crête forte
d_faible = np.abs(ys - 6709983.0)[:, None] * np.ones((1, L2))  # crête faible
img2 = np.clip(100 + 90 * np.exp(-d_fort**2 / 8) + 40 * np.exp(-d_faible**2 / 8),
               0, 255).astype("uint8")
raster2 = tmp / "couloir.tif"
with rasterio.open(raster2, "w", driver="GTiff", width=L2, height=H2, count=1,
                   dtype="uint8", crs="EPSG:2154", transform=T2) as dst:
    dst.write(img2, 1)
lecteur2 = LecteurRaster(raster2)

ligne_a = LineString([(610020, 6709975.5), (610280, 6709975.5)])  # sur la forte
ligne_b = LineString([(610020, 6709979.5), (610280, 6709979.5)])  # sienne = faible
verite_faible = LineString([(610010, 6709983.0), (610290, 6709983.0)])
parts_v = [ligne_a, ligne_b]
voisins_b = (STRtree(parts_v), parts_v, [0, 1], 1)
voisins_a = (STRtree(parts_v), parts_v, [0, 1], 0)

sans, _ = recaler_ligne(ligne_b, lecteur2, PARAMS)  # capture la crête forte
avec, _ = recaler_ligne(ligne_b, lecteur2, PARAMS, voisins=voisins_b)
rec_a, _ = recaler_ligne(ligne_a, lecteur2, PARAMS, voisins=voisins_a)
d_sans = [sans.interpolate(t, normalized=True).distance(verite_faible)
          for t in np.linspace(0.1, 0.9, 20)]
d_avec = [avec.interpolate(t, normalized=True).distance(verite_faible)
          for t in np.linspace(0.1, 0.9, 20)]
assert float(np.median(d_sans)) > 5.0, "le cas de bug ne se reproduit plus ?"
assert float(np.median(d_avec)) < 0.6, f"couloir inopérant : {np.median(d_avec):.2f} m"
assert rec_a.distance(avec) > 5.0  # les deux lignes restent séparées
d_a = [rec_a.interpolate(t, normalized=True).distance(
    LineString([(610010, 6709975.0), (610290, 6709975.0)]))
    for t in np.linspace(0.1, 0.9, 20)]
assert float(np.median(d_a)) < 0.5  # A reste sur sa crête forte

pts_b, nor_b, _ = densifier(ligne_b, 2.0)
b_neg, b_pos = bornes_laterales(pts_b, nor_b, 8.0, voisins_b)
assert np.all(b_pos >= 7.9) or np.all(b_neg >= 7.9)  # côté libre : fenêtre pleine
assert (np.median(b_neg) < 2.5) or (np.median(b_pos) < 2.5)  # côté voisin : mi-chemin

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

# seuils de statut surchargeables par zone (résidu ultra-strict -> a_revoir)
cfg_seuils = tmp / "recalage_seuils.yaml"
cfg_seuils.write_text(yaml.safe_dump({
    "zone": "jouet_seuils", "raster_gsd_attendu": 0.5,
    "couches": {"parcellaire": {"polarite": "clair"}},
    "seuils_statut": {"residu_max_m": 0.0001}}), encoding="utf-8")
out_seuils = tmp / "out_seuils"
run_recalage(cfg_seuils, gpkg_path, raster_path, out_seuils)
p_s = gpd.read_file(out_seuils / "jouet_seuils_entites_l93_recale.gpkg",
                    layer="parcellaire")
assert p_s[p_s["src"] == "a"].iloc[0]["statut_recalage"] == "a_revoir"

# ---------------------------------------------------------------------------
# Contrôleur verif_recalage : cas conforme puis cas volontairement cassé
# ---------------------------------------------------------------------------
import subprocess
verif = Path(__file__).resolve().parents[1] / "tools" / "verif_recalage.py"
args = [sys.executable, str(verif), str(cfg_path), str(gpkg_path)]
r = subprocess.run(args + [str(gpkg_out), str(raster_path)],
                   capture_output=True, text=True)
assert r.returncode == 0 and "CONFORME" in r.stdout, r.stdout + r.stderr

casse = tmp / "casse.gpkg"
parc_c = gpd.read_file(gpkg_out, layer="parcellaire")
masque_a = parc_c["src"] == "a"
parc_c.loc[masque_a, "geometry"] = parc_c[masque_a].geometry.translate(50, 0)
parc_c.to_file(casse, layer="parcellaire", driver="GPKG")
gpd.read_file(gpkg_out, layer="talus_fosse").to_file(casse, layer="talus_fosse",
                                                     driver="GPKG")
r2 = subprocess.run(args + [str(casse), str(raster_path)],
                    capture_output=True, text=True)
assert r2.returncode != 0 and "Hausdorff" in (r2.stdout + r2.stderr), r2.stdout

# ---------------------------------------------------------------------------
# Application des décisions + contrôleur d'application
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from appliquer_decisions import appliquer

edit_wkt = LineString([(x, courbe_crete(x) + 0.3)
                       for x in np.arange(600070, 600460, 40.0)]).wkt
dec_path = tmp / "decisions.yaml"
dec_path.write_text(yaml.safe_dump({
    "parcellaire_0": {"id": "parcellaire_0", "couche": "parcellaire",
                      "decision": "editee", "geometrie_editee": edit_wkt},
    "parcellaire_1": {"id": "parcellaire_1", "couche": "parcellaire",
                      "decision": "exclue"},
    "talus_fosse_0": {"id": "talus_fosse_0", "couche": "talus_fosse",
                      "decision": "recale"}}), encoding="utf-8")
gpkg_final, comptes = appliquer(gpkg_path, gpkg_out, dec_path,
                                tmp / "jouet_final.gpkg")
fin = gpd.read_file(gpkg_final, layer="parcellaire")
assert len(fin) == 1 and fin.iloc[0]["decision_humaine"] == "editee"
assert fin.iloc[0].geometry.equals(wkt.loads(edit_wkt))
assert wkt.loads(fin.iloc[0]["geom_origine"]).equals(decalee)  # origine gardée
verif_app = Path(__file__).resolve().parents[1] / "tools" / "verif_application.py"
r3 = subprocess.run([sys.executable, str(verif_app), str(gpkg_path),
                     str(gpkg_out), str(dec_path), str(gpkg_final)],
                    capture_output=True, text=True)
assert r3.returncode == 0 and "CONFORME" in r3.stdout, r3.stdout + r3.stderr
# cas cassé : géométrie éditée altérée dans le final -> détecté
fin2 = gpd.read_file(gpkg_final, layer="parcellaire")
fin2.geometry = fin2.geometry.translate(3, 0)
casse2 = tmp / "final_casse.gpkg"
fin2.to_file(casse2, layer="parcellaire", driver="GPKG")
gpd.read_file(gpkg_final, layer="talus_fosse").to_file(
    casse2, layer="talus_fosse", driver="GPKG")
r4 = subprocess.run([sys.executable, str(verif_app), str(gpkg_path),
                     str(gpkg_out), str(dec_path), str(casse2)],
                    capture_output=True, text=True)
assert r4.returncode != 0, "altération non détectée"

# --recale-depuis : les 'recale' prennent la géométrie de la version REVUE
ref_gpkg = tmp / "reference.gpkg"
for couche_r in ("parcellaire", "talus_fosse"):
    g_r = gpd.read_file(gpkg_out, layer=couche_r)
    if couche_r == "talus_fosse":
        g_r.geometry = g_r.geometry.translate(0.9, 0)  # version vue ≠ courante
    g_r.to_file(ref_gpkg, layer=couche_r, driver="GPKG")
gpkg_final2, _ = appliquer(gpkg_path, gpkg_out, dec_path,
                           tmp / "jouet_final2.gpkg", ref_gpkg)
tf_fin = gpd.read_file(gpkg_final2, layer="talus_fosse").iloc[0]
tf_ref = gpd.read_file(ref_gpkg, layer="talus_fosse").iloc[0]
assert tf_fin.geometry.equals(tf_ref.geometry), "référence non appliquée"
r5 = subprocess.run([sys.executable, str(verif_app), str(gpkg_path),
                     str(gpkg_out), str(dec_path), str(gpkg_final2),
                     str(ref_gpkg)], capture_output=True, text=True)
assert r5.returncode == 0 and "CONFORME" in r5.stdout, r5.stdout + r5.stderr

# defaut_original : les couches listées reviennent à l'origine si non décidées
dec_vide = tmp / "decisions_vides.yaml"
dec_vide.write_text("{}", encoding="utf-8")
gpkg_final3, _ = appliquer(gpkg_path, gpkg_out, dec_vide,
                           tmp / "jouet_final3.gpkg",
                           defaut_original={"parcellaire"})
p3 = gpd.read_file(gpkg_final3, layer="parcellaire")
assert (p3["decision_humaine"] == "auto_original").all()
for _, l in p3.iterrows():
    assert l.geometry.equals(wkt.loads(l["geom_origine"]))
t3 = gpd.read_file(gpkg_final3, layer="talus_fosse")
assert (t3["decision_humaine"] == "auto").all()  # couche non listée : recalé
r6 = subprocess.run([sys.executable, str(verif_app), str(gpkg_path),
                     str(gpkg_out), str(dec_vide), str(gpkg_final3)],
                    capture_output=True, text=True)
assert r6.returncode == 0 and "CONFORME" in r6.stdout, r6.stdout + r6.stderr
print("noyau + pipeline + contrôleurs recalage + application : OK")
