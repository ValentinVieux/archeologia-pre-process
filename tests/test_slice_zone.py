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
