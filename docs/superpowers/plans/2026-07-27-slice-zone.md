# Plan d'implémentation — `tools/slice_zone.py`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implémenter le découpeur de tuiles d'entraînement à split spatial par blocs
spécifié dans `docs/superpowers/specs/2026-07-27-slice-zone-design.md`, puis produire le
premier dataset réel (`lineaires_haye_ld_648_v1`).

**Architecture:** Un script CLI unique `tools/slice_zone.py` (convention du repo :
outil déterministe autonome), fonctions pures testables (grille, blocs, split glouton,
conversion COCO) + un `main` d'orchestration. Test = script à asserts façon
`tests/test_audit.py` (pas de pytest dans ce repo), sur données jouets générées.

**Tech Stack:** Python 3.11 (`.venv`), rasterio (à ajouter), geopandas/shapely/pyogrio/
pyyaml (présents), PIL (présent via requirements existants).

## Global Constraints

- Jamais activer le venv : `.venv\Scripts\python.exe` directement (CLAUDE.md).
- Tuiles **648 px sans chevauchement**, tuiles partielles de bord écartées (spec).
- Le script **refuse un raster/GPKG sous `G:\`** (copie locale d'abord).
- Déterminisme : même config + même seed (défaut 42) → sorties identiques.
- Sorties non commitées (comme `audits\`) ; la CLI imprime `Sorties :` avec les chemins.
- Code/docstrings en français-technique, ids ASCII (conventions repo).
- Ne rien écrire sur G: dans ce plan (le dépôt Drive viendra après validation humaine).

---

### Task 1: Dépendance rasterio

**Files:**
- Modify: `requirements.txt` (ajout d'une ligne)

**Interfaces:**
- Produces: `import rasterio` fonctionnel dans le venv.

- [ ] **Step 1:** `.venv\Scripts\pip install rasterio` puis vérifier
  `.venv\Scripts\python.exe -c "import rasterio; print(rasterio.__version__)"`.
- [ ] **Step 2:** Ajouter `rasterio` à `requirements.txt` (ordre alphabétique du fichier).
- [ ] **Step 3:** Commit `deps : rasterio (decoupeur slice_zone)`.

---

### Task 2: Noyau géométrique — grille, blocs, split glouton (TDD)

**Files:**
- Create: `tools/slice_zone.py` (fonctions pures seulement)
- Create: `tests/test_slice_zone.py`

**Interfaces (Produces — utilisées par Task 3/4):**
- `grille_tuiles(transform, largeur_px, hauteur_px, tuile_px) -> list[dict]` — chaque
  dict : `{row, col, fenetre: rasterio.windows.Window, bounds: (minx,miny,maxx,maxy)}`,
  tuiles pleines uniquement, ordre (row, col).
- `bloc_de(bounds, bloc_m) -> tuple[int, int]` — id de bloc du **centre** de la tuile :
  `(floor(cx / bloc_m), floor(cy / bloc_m))` en coordonnées CRS.
- `affecter_splits(annos_par_bloc: dict[tuple, Counter], cibles: dict[str, float], seed) -> dict[tuple, str]`
  — glouton équilibré par classe : blocs triés par total d'annotations décroissant
  (départage : mélange seedé), chaque bloc affecté au split de plus grand déficit
  pondéré `Σ_c (1/total_c) * (part_cible_s * total_c - deja_alloue[s][c])` ;
  départage d'égalité : ordre train > valid > test. Les blocs sans annotation ne sont
  **pas** affectés (absents du dict retourné).

- [ ] **Step 1:** Écrire dans `tests/test_slice_zone.py` les tests du noyau (échouent) :

```python
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
# aucun chevauchement : bounds tous disjoints deux à deux (grille régulière)
xs = sorted({b["bounds"][0] for b in tuiles})
assert xs == [500000 + i * 400 for i in range(5)]

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
aff = affecter_splits(annos, {"train": 70, "valid": 20, "test": 10}, seed=42)
assert (5, 0) not in aff and len(aff) == 10
assert set(aff.values()) == {"train", "valid", "test"}
tot = Counter()
for b, s in aff.items():
    tot[s] += sum(annos[b].values())
part_train = tot["train"] / sum(tot.values())
assert 0.55 <= part_train <= 0.85, f"train {part_train:.0%} hors tolérance"
# la classe rare (8 annos, 4 blocs) doit exister dans au moins train ET un autre split
splits_rare = {aff[b] for b, c in annos.items() if c.get("rare")}
assert "train" in splits_rare and len(splits_rare) >= 2, f"rare mal réparti : {splits_rare}"
# déterminisme
assert aff == affecter_splits(annos, {"train": 70, "valid": 20, "test": 10}, seed=42)

print("noyau géométrique : OK")
```

- [ ] **Step 2:** Lancer `.venv\Scripts\python.exe tests\test_slice_zone.py` → échec
  attendu (`ModuleNotFoundError` puis `ImportError`).
- [ ] **Step 3:** Implémenter les trois fonctions dans `tools/slice_zone.py` (module
  avec docstring renvoyant à la spec ; `grille_tuiles` par double boucle
  `range(hauteur_px // tuile_px)` × `range(largeur_px // tuile_px)` et
  `rasterio.windows.Window(col*t, row*t, t, t)` + `rasterio.windows.bounds()` ;
  `affecter_splits` exactement selon la formule de l'interface).
- [ ] **Step 4:** Test vert.
- [ ] **Step 5:** Commit `slice_zone : noyau grille/blocs/split glouton (TDD)`.

---

### Task 3: Extraction des annotations — buffers, clip, polygones COCO (TDD)

**Files:**
- Modify: `tools/slice_zone.py`
- Modify: `tests/test_slice_zone.py` (append)

**Interfaces (Produces):**
- `preparer_entites(gdf, classe, buffer_m) -> list[shapely.Polygon]` — lignes/points
  bufferisés (`buffer_m` = largeur totale pour les lignes → `buffer(buffer_m / 2)`,
  rayon pour les points), polygones inchangés si `buffer_m` est None ; MultiX explosés ;
  géométries vides/invalides réparées (`make_valid`) ou écartées.
- `polygone_vers_coco(poly, bounds, tuile_px) -> list[list[float]]` — anneaux
  extérieurs seulement, coordonnées pixels (origine coin haut-gauche, y vers le bas),
  arrondi 2 décimales, anneaux < 3 points écartés.
- `annotations_tuile(polys_par_classe, bounds, tuile_px) -> list[dict]` — clip de
  chaque polygone à la tuile, dicts `{classe, segmentation, bbox_px, aire_px}`.

- [ ] **Step 1:** Ajouter les tests (échouent) :

```python
from shapely.geometry import LineString, Point, Polygon, box
import geopandas as gpd
from slice_zone import annotations_tuile, polygone_vers_coco, preparer_entites

# lignes -> buffer largeur totale 2 m ; points -> rayon
gl = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)])], crs="EPSG:2154")
(pl,) = preparer_entites(gl, "parcellaire", buffer_m=2.0)
assert abs(pl.area - (10 * 2 + 3.14159)) < 0.5  # rectangle + extrémités rondes
gp = gpd.GeoDataFrame(geometry=[Point(5, 5)], crs="EPSG:2154")
(pp,) = preparer_entites(gp, "tas", buffer_m=5.0)
assert abs(pp.area - 3.14159 * 25) < 1.0

# conversion pixels : tuile bounds (100, 100, 200, 200), 100 px -> 1 m/px
carre = Polygon([(110, 110), (120, 110), (120, 120), (110, 120)])
(ring,) = polygone_vers_coco(carre, (100, 100, 200, 200), 100)
xs, ys = ring[0::2], ring[1::2]
assert min(xs) == 10.0 and max(xs) == 20.0
assert min(ys) == 80.0 and max(ys) == 90.0  # y inversé (haut de tuile = maxy)

# clip à la tuile : un polygone débordant est tronqué, un disjoint disparaît
annos = annotations_tuile({"a": [box(150, 150, 300, 300)], "b": [box(900, 900, 910, 910)]},
                          (100, 100, 200, 200), 100)
assert [a["classe"] for a in annos] == ["a"]
assert annos[0]["aire_px"] == 2500.0  # 50x50 m visibles = 50x50 px
print("annotations : OK")
```

- [ ] **Step 2:** Test rouge, implémenter, test vert (mêmes commandes que Task 2).
- [ ] **Step 3:** Commit `slice_zone : entites -> polygones COCO (buffers, clip)`.

---

### Task 4: Pipeline complet + CLI — validité nodata, négatifs, sorties (TDD intégration)

**Files:**
- Modify: `tools/slice_zone.py`
- Modify: `tests/test_slice_zone.py` (append : fixture raster/GPKG jouets + intégration)

**Interfaces (Produces):**
- `charger_config(chemin) -> dict` — YAML validé (champs requis, split somme 100,
  `tuile_px > 0`, refus de chemins `G:` insensible à la casse pour raster/gpkg) ;
  défauts : `negatifs_pct: 10`, `min_couverture_valide: 0.5`,
  `min_visibilite_annotation: 0.5`, `bloc_m: 2000`, `assign_crs/nodata_supplementaire: null`.
- `run_slicing(cfg, out_dir, seed) -> dict` (stats) — orchestration complète, et CLI
  `main()` : `slice_zone.py <config.yaml> [--out D] [--seed N]`, imprime le récap et
  `Sorties : <dossier>` en dernière ligne.
- Sorties disque : `train|valid|test/*.png` (RGB, bande répliquée ×3, source Byte
  exigée — erreur claire sinon), `_annotations.coco.json` par split (categories dans
  l'ordre de la config, images {id, file_name, width, height}, annotations {id,
  image_id, category_id, segmentation, bbox [x,y,w,h], area, iscrowd: 0}),
  `split_manifest.yaml`, `controle_blocs.html`.
- Règles : nodata effectif = nodata déclaré ∪ `nodata_supplementaire` ∪ NaN ; nodata
  déclaré NaN sur bande entière → ignoré avec avertissement ; tuile gardée si
  couverture valide ≥ seuil ; annotation écartée si la fraction valide de sa bbox
  < `min_visibilite_annotation` ; négatifs tirés (Random(seed)) parmi les tuiles vides
  des blocs affectés, à hauteur de `negatifs_pct` % du nombre de tuiles annotées.

- [ ] **Step 1:** Fixture jouote dans le test : GeoTIFF Byte 2000×1600 px, 1 m/px,
  EPSG:2154, origine (500000, 6800000), damier de valeurs 1-255, bande **nodata=0 sur
  les 500 dernières colonnes** ; GPKG 3 couches — `lignes` (6 LineString dont une
  traversant 3 blocs de 800 m et une entièrement dans la zone nodata), `zones`
  (2 Polygon), `points` (3 Point) ; config YAML jouet écrite dans le dossier temp
  (couches : lignes→`parcellaire` buffer 2, zones→`chaussee`, points→`tas` buffer 5 ;
  `tuile_px: 400`, `bloc_m: 800`, `nodata_supplementaire: 0`, `negatifs_pct: 20`).
- [ ] **Step 2:** Tests d'intégration (échouent) :

```python
res = run_slicing(cfg, out, seed=42)
manif = yaml.safe_load((out / "split_manifest.yaml").read_text(encoding="utf-8"))
tuiles_m = manif["tuiles"]
# 1. chaque tuile appartient à exactement un split, cohérent avec son bloc
blocs_splits = {}
for tm in tuiles_m:
    blocs_splits.setdefault(tuple(tm["bloc"]), set()).add(tm["split"])
assert all(len(s) == 1 for s in blocs_splits.values()), "bloc à cheval sur 2 splits"
# 2. zéro partage de pixels : les bounds sont uniques et sur la grille de 400 m
assert len({tuple(tm["bounds"]) for tm in tuiles_m}) == len(tuiles_m)
# 3. COCO valide et images présentes sur disque
for split in ("train", "valid", "test"):
    cc = json.loads((out / split / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert {c["name"] for c in cc["categories"]} == {"parcellaire", "chaussee", "tas"}
    for im in cc["images"]:
        assert (out / split / im["file_name"]).exists()
        assert im["width"] == im["height"] == 400
    for an in cc["annotations"]:
        assert an["iscrowd"] == 0 and len(an["segmentation"][0]) >= 6
# 4. la ligne en zone nodata n'a produit aucune annotation ; les tuiles 100 % nodata absentes
noms = {tm["nom"] for tm in tuiles_m}
assert all(tm["bounds"][0] < 501500 for tm in tuiles_m), "tuile pleine zone nodata conservée"
# 5. négatifs : présents, ~20 % des tuiles annotées, dans des blocs affectés
n_annotees = sum(1 for tm in tuiles_m if tm["n_annotations"] > 0)
n_neg = sum(1 for tm in tuiles_m if tm["n_annotations"] == 0)
assert 0 < n_neg <= max(1, round(0.2 * n_annotees) + 1)
# 6. idempotence bit-à-bit du manifeste (hors date)
res2 = run_slicing(cfg, out2, seed=42)
m2 = yaml.safe_load((out2 / "split_manifest.yaml").read_text(encoding="utf-8"))
for m in (manif, m2): m.pop("genere_le", None)
assert manif == m2, "non déterministe"
# 7. garde-fou G:
try:
    charger_config(cfg_avec_raster_G); assert False, "chemin G: accepté"
except SystemExit: pass
print("ALL OK — pipeline slice_zone")
```

- [ ] **Step 3:** Implémenter `charger_config`, `run_slicing`, l'écriture des quatre
  sorties et `main()`. `controle_blocs.html` : SVG des blocs (rect par bloc, couleurs
  train `#7A8C55` / valid `#5E7F9E` / test `#C08A3E`, blocs non affectés gris),
  emprise en fond, tableau des comptes par classe/split — autonome, palette du doc
  fuite spatiale.
- [ ] **Step 4:** `.venv\Scripts\python.exe tests\test_slice_zone.py` → `ALL OK`.
- [ ] **Step 5:** Relancer aussi `tests\test_audit.py` (non-régression du repo).
- [ ] **Step 6:** Commit `slice_zone : pipeline complet + CLI (COCO, manifeste, carte de controle)`.

---

### Task 5: Documentation CLI + config Haye + découpe réelle

**Files:**
- Modify: `CLAUDE.md` (ligne de commande dans § Commands)
- Create: `configs/lineaires_haye_ld_648_v1.yaml`
- Sorties locales (non commitées) : `datasets/lineaires_haye_ld_648_v1/`

**Interfaces:**
- Consumes: la CLI de Task 4.

- [ ] **Step 1:** Ajouter à CLAUDE.md § Commands :
  `.venv\Scripts\python.exe tools\slice_zone.py <config.yaml> [--out <dossier>]  # découpe tuiles + split spatial`
- [ ] **Step 2:** Copier en local (robocopy vers le scratchpad) le dossier
  `raw\Haye_MNT_IGN\LD_A15_Rmin10_Rmax20_H1p7_V1\` (VRT + tif) et réutiliser le GPKG
  local déjà présent (copie conforme au dépôt du 27/07).
- [ ] **Step 3:** Écrire `configs/lineaires_haye_ld_648_v1.yaml` : couches
  `parcellaire/talus_fosse/rempart` (buffer 2 m chacune), `tuile_px: 648`,
  `bloc_m: 2000`, split 70/20/10, chemins locaux.
- [ ] **Step 4:** Lancer la découpe réelle, examiner le récap (comptes par classe/split,
  tuiles écartées) et `controle_blocs.html`.
- [ ] **Step 5:** Vérifications indépendantes sur le dataset réel (revue adversariale
  ultracode) : intégrité COCO, aucune paire de tuiles inter-splits se chevauchant,
  cohérence manifeste/disque, relecture du code.
- [ ] **Step 6:** Commit `slice_zone : config Haye lineaires + commande CLAUDE.md`
  (la sortie `datasets/` reste non commitée). Dépôt Drive : **après validation
  humaine** de la carte de contrôle.

## Self-review

- Couverture spec : interface YAML ✓ (Task 4), grille/partielles ✓ (T2), blocs par
  centre ✓ (T2), split glouton pondéré ✓ (T2), buffers/clip/COCO ✓ (T3), nodata +
  visibilité ✓ (T4), négatifs par bloc affecté ✓ (T4), refus G: ✓ (T4), manifeste +
  hashes + carte HTML ✓ (T4), récap stdout ✓ (T4), idempotence ✓ (T4), rasterio ✓
  (T1), run réel Haye ✓ (T5). `bande_tampon` : hors v1 (YAGNI, conforme spec).
- Types cohérents entre tasks (bounds tuple 4, bloc tuple 2, Counter par classe) ✓.
- Pas de placeholder ; les formules (déficit pondéré, buffer/2 lignes) sont dans les
  interfaces.
