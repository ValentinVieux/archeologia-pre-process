"""Auto-test de tools/slice_zone.py — données jouets, style test_audit.py."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from affine import Affine

from slice_zone import affecter_splits, bloc_de, grille_tuiles

# --- grille : raster 2000x1600 px à 1 m/px, origine (500000, 6800000), tuiles 400 px
t = Affine(1.0, 0, 500000, 0, -1.0, 6800000)
tuiles = grille_tuiles(t, 2000, 1600, 400)
assert len(tuiles) == 5 * 4, f"5x4 tuiles pleines attendues, obtenu {len(tuiles)}"
assert tuiles[0]["bounds"] == (500000, 6799600, 500400, 6800000)
assert tuiles[-1]["bounds"] == (501600, 6798400, 502000, 6798800)
# raster non multiple de la tuile : les partielles sont écartées
assert len(grille_tuiles(t, 2100, 1650, 400)) == 5 * 4, "partielles non écartées"
# grille régulière : les origines X sont exactement les multiples de 400
xs = sorted({b["bounds"][0] for b in tuiles})
assert xs == [500000 + i * 400 for i in range(5)]
# la fenêtre raster correspond bien à la tuile (400 px, alignée)
f0 = tuiles[0]["fenetre"]
assert (f0.col_off, f0.row_off, f0.width, f0.height) == (0, 0, 400, 400)

# --- blocs : centre de tuile, blocs de 800 m
assert bloc_de((500000, 6799600, 500400, 6800000), 800) == (625, 8499)  # centre (500200, 6799800)
assert bloc_de((500400, 6799600, 500800, 6800000), 800) == (625, 8499)  # même bloc
assert bloc_de((500800, 6799600, 501200, 6800000), 800) == (626, 8499)  # bloc voisin

# --- split glouton équilibré par classe
annos = {
    (0, 0): Counter(parcellaire=100, rare=2), (1, 0): Counter(parcellaire=90),
    (2, 0): Counter(parcellaire=80, rare=2),  (0, 1): Counter(parcellaire=70),
    (1, 1): Counter(parcellaire=60, rare=2),  (2, 1): Counter(parcellaire=50),
    (3, 1): Counter(parcellaire=40, rare=2),  (3, 0): Counter(parcellaire=40),
    (4, 0): Counter(parcellaire=30),          (4, 1): Counter(parcellaire=30),
    (5, 0): Counter(),                        # bloc vide : jamais affecté
}
cibles = {"train": 70, "valid": 20, "test": 10}
aff = affecter_splits(annos, cibles, seed=42)
assert (5, 0) not in aff and len(aff) == 10
assert set(aff.values()) == {"train", "valid", "test"}
tot = Counter()
for b, s in aff.items():
    tot[s] += sum(annos[b].values())
part_train = tot["train"] / sum(tot.values())
assert 0.55 <= part_train <= 0.85, f"train {part_train:.0%} hors tolérance"
# la classe rare (8 annos, 4 blocs) doit exister dans train ET au moins un autre split
splits_rare = {aff[b] for b, c in annos.items() if c.get("rare")}
assert "train" in splits_rare and len(splits_rare) >= 2, f"rare mal réparti : {splits_rare}"
# déterminisme
assert aff == affecter_splits(annos, cibles, seed=42)

print("noyau géométrique : OK")

# ---------------------------------------------------------------------------
# Entités -> polygones COCO
# ---------------------------------------------------------------------------
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

from slice_zone import annotations_tuile, polygone_vers_coco, preparer_entites

# lignes -> buffer largeur totale 2 m ; points -> rayon
gl = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)])], crs="EPSG:2154")
(pl,) = preparer_entites(gl, buffer_m=2.0)
assert abs(pl.area - (10 * 2 + 3.14159)) < 0.5, pl.area  # rectangle + extrémités rondes
gp = gpd.GeoDataFrame(geometry=[Point(5, 5)], crs="EPSG:2154")
(pp,) = preparer_entites(gp, buffer_m=5.0)
assert abs(pp.area - 3.14159 * 25) < 1.0, pp.area
# MultiLineString explosée en autant de polygones, géométries nulles écartées
gm = gpd.GeoDataFrame(
    geometry=[MultiLineString([[(0, 0), (5, 0)], [(0, 10), (5, 10)]]), None],
    crs="EPSG:2154")
assert len(preparer_entites(gm, buffer_m=2.0)) == 2
# polygones : inchangés (buffer_m None)
gz = gpd.GeoDataFrame(geometry=[box(0, 0, 4, 4)], crs="EPSG:2154")
(pz,) = preparer_entites(gz, buffer_m=None)
assert pz.equals(box(0, 0, 4, 4))

# conversion pixels : tuile bounds (100, 100, 200, 200), 100 px -> 1 m/px
carre = Polygon([(110, 110), (120, 110), (120, 120), (110, 120)])
(ring,) = polygone_vers_coco(carre, (100, 100, 200, 200), 100)
xs, ys = ring[0::2], ring[1::2]
assert min(xs) == 10.0 and max(xs) == 20.0
assert min(ys) == 80.0 and max(ys) == 90.0, (min(ys), max(ys))  # y inversé

# clip à la tuile : un polygone débordant est tronqué, un disjoint disparaît
annos_t = annotations_tuile(
    {"a": [box(150, 150, 300, 300)], "b": [box(900, 900, 910, 910)]},
    (100, 100, 200, 200), 100)
assert [a["classe"] for a in annos_t] == ["a"]
assert annos_t[0]["aire_px"] == 2500.0  # 50x50 m visibles = 50x50 px
x, y, w, h = annos_t[0]["bbox_px"]
assert (x, y, w, h) == (50.0, 0.0, 50.0, 50.0), annos_t[0]["bbox_px"]
print("annotations : OK")
