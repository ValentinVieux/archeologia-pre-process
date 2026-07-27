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

# --- équilibre sur un parc réaliste : beaucoup de blocs, classes très déséquilibrées
# (régression : l'ancienne formule non normalisée mettait ~95 % des annos en train)
annos_reels = {}
for i in range(40):
    annos_reels[(i, 0)] = Counter(parcellaire=200 - 4 * i, __tuiles__=25 - (i % 7))
    if 30 <= i < 36:  # classe rare concentrée dans des PETITS blocs (traités tard)
        annos_reels[(i, 0)]["rare"] = 5
aff_r = affecter_splits(annos_reels, cibles, seed=42)
alloc = {s: Counter() for s in cibles}
for b, s in aff_r.items():
    alloc[s].update(annos_reels[b])
for classe, tol in (("parcellaire", 0.05), ("rare", 0.12), ("__tuiles__", 0.05)):
    total_c = sum(alloc[s][classe] for s in cibles)
    for s, cible_pct in cibles.items():
        part = alloc[s][classe] / total_c
        assert abs(part - cible_pct / 100) <= tol, \
            f"{classe}/{s} : {part:.1%} pour une cible de {cible_pct}%"

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

# enclos fermé (boucle bufferisée -> polygone à trou) : le trou ne doit PAS être rempli
boucle = MultiLineString([[(10, 10), (50, 10)], [(50, 10), (50, 50)],
                          [(50, 50), (10, 50)], [(10, 50), (10, 10)]])
(pe,) = preparer_entites(gpd.GeoDataFrame(geometry=[boucle], crs="EPSG:2154"), buffer_m=2.0)
assert len(pe.interiors) == 1
(a_enclos,) = annotations_tuile({"parcellaire": [pe]}, (0, 0, 324, 324), 648)
assert len(a_enclos["segmentation"]) >= 2, "trou non décomposé en morceaux sans trou"
aire_remplie = sum(Polygon(list(zip(r[0::2], r[1::2]))).area
                   for r in a_enclos["segmentation"])
assert abs(aire_remplie - a_enclos["aire_px"]) / a_enclos["aire_px"] < 0.05, \
    f"segmentation remplie {aire_remplie:.0f} px² vs aire réelle {a_enclos['aire_px']:.0f} px²"

# sliver sub-pixel : buffer qui effleure la tuile voisine -> aucune annotation
sl = preparer_entites(gpd.GeoDataFrame(
    geometry=[LineString([(324.95, 100), (324.95, 200)])], crs="EPSG:2154"), buffer_m=2.0)
assert annotations_tuile({"parcellaire": sl}, (0, 0, 324, 324), 648) == [], \
    "sliver sub-pixel conservé"

# GeometryCollection avec MultiPolygon imbriqué (sortie possible de make_valid)
from shapely.geometry import GeometryCollection, MultiPolygon
gc = GeometryCollection([MultiPolygon([box(0, 0, 2, 2), box(3, 0, 5, 2)]),
                         LineString([(0, 0), (1, 1)])])
assert len(preparer_entites(gpd.GeoDataFrame(geometry=[gc], crs="EPSG:2154"),
                            buffer_m=None)) == 2, "MultiPolygon imbriqué perdu"

# variantes de chemins Drive : toutes détectées
from slice_zone import chemin_sur_drive
for mauvais in (r"G:\Mon Drive\x.tif", "g:/mon drive/x.tif", r"\\?\G:\x.tif",
                "//?/G:/x.tif", r"\\localhost\G$\x.tif", "file:///G:/x.tif"):
    assert chemin_sur_drive(mauvais), f"non détecté : {mauvais}"
assert not chemin_sur_drive(r"C:\data\g_truc\x.tif")
assert not chemin_sur_drive(r"D:\gros.tif")
print("annotations : OK")

# ---------------------------------------------------------------------------
# Pipeline complet sur données jouets
# ---------------------------------------------------------------------------
import json
import shutil
import tempfile

import numpy as np
import rasterio
import yaml

from slice_zone import charger_config, run_slicing

tmp = Path(tempfile.mkdtemp(prefix="slice_zone_test_"))
try:
    # --- raster jouet : Byte 2000x1600 px, 1 m/px, EPSG:2154, damier, nodata=0 à droite
    raster = tmp / "indice.tif"
    donnees = ((np.indices((1600, 2000)).sum(axis=0) // 50) % 2 * 200 + 30).astype("uint8")
    donnees[:, 1500:] = 0  # zone sans dalle (fond implicite)
    with rasterio.open(
        raster, "w", driver="GTiff", width=2000, height=1600, count=1, dtype="uint8",
        crs="EPSG:2154", transform=Affine(1.0, 0, 500000, 0, -1.0, 6800000),
    ) as dst:
        dst.write(donnees, 1)

    # --- GPKG jouet : 3 couches réparties sur les 4 blocs valides de 800 m
    gpkg = tmp / "entites.gpkg"
    lignes = [
        LineString([(500050, 6799750), (501300, 6799750)]),   # traverse 2 blocs (row 0)
        LineString([(500100, 6799300), (500700, 6799100)]),
        LineString([(500900, 6798900), (501400, 6798700)]),
        LineString([(500100, 6798900), (500600, 6798650)]),
        LineString([(501650, 6799000), (501900, 6799200)]),   # entièrement en zone nodata
        LineString([(500250, 6799650), (500350, 6799650)]),
        # bbox majoritairement en nodata (visibilité < 0.5) mais entité partiellement
        # visible : sa tuile ne doit JAMAIS devenir un négatif
        LineString([(501440, 6799400), (501640, 6799400)]),
    ]
    gpd.GeoDataFrame(geometry=lignes, crs="EPSG:2154").to_file(gpkg, layer="lignes", driver="GPKG")
    gpd.GeoDataFrame(geometry=[box(500450, 6799450, 500550, 6799550),
                               box(501000, 6798500, 501150, 6798650)],
                     crs="EPSG:2154").to_file(gpkg, layer="zones", driver="GPKG")
    gpd.GeoDataFrame(geometry=[Point(500200, 6799000), Point(500900, 6799750)],
                     crs="EPSG:2154").to_file(gpkg, layer="points", driver="GPKG")
    # couche ignorée (classe non entraînée) : seule occupante de la tuile (2,3) —
    # elle ne doit produire AUCUNE annotation mais interdire cette tuile aux négatifs
    gpd.GeoDataFrame(geometry=[LineString([(501250, 6799000), (501350, 6799000)])],
                     crs="EPSG:2154").to_file(gpkg, layer="remparts_jouet", driver="GPKG")

    # --- config jouet
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "dataset": "jouet_v1", "zone": "test/zone_jouet",
        "raster": str(raster), "gpkg": str(gpkg),
        "couches": {"lignes": {"classe": "parcellaire", "buffer_m": 2.0},
                    "zones": {"classe": "chaussee"},
                    "points": {"classe": "tas", "buffer_m": 5.0},
                    "remparts_jouet": {"ignorer": True, "buffer_m": 2.0}},
        "tuile_px": 400, "bloc_m": 800,
        "split": {"train": 70, "valid": 20, "test": 10},
        "negatifs_pct": 50, "nodata_supplementaire": 0,
    }, allow_unicode=True), encoding="utf-8")

    cfg = charger_config(cfg_path)
    assert cfg["min_couverture_valide"] == 0.5 and cfg["negatifs_pct"] == 50  # défauts

    out = tmp / "out"
    res = run_slicing(cfg, out, seed=42)
    manif = yaml.safe_load((out / "split_manifest.yaml").read_text(encoding="utf-8"))
    tuiles_m = manif["tuiles"]

    # 1. chaque bloc est entièrement dans un seul split
    blocs_splits = {}
    for tm in tuiles_m:
        blocs_splits.setdefault(tuple(tm["bloc"]), set()).add(tm["split"])
    assert all(len(s) == 1 for s in blocs_splits.values()), "bloc à cheval sur 2 splits"
    # aucun split vide (l'amélioration locale ne doit jamais sacrifier test)
    assert {tm["split"] for tm in tuiles_m} == {"train", "valid", "test"}

    # 2. zéro partage de pixels : bounds uniques, alignés sur la grille de 400 m
    assert len({tuple(tm["bounds"]) for tm in tuiles_m}) == len(tuiles_m)
    assert all((tm["bounds"][0] - 500000) % 400 == 0 for tm in tuiles_m)

    # 3. COCO valides, images présentes, catégories conformes
    for split in ("train", "valid", "test"):
        cc = json.loads((out / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        assert {c["name"] for c in cc["categories"]} == {"parcellaire", "chaussee", "tas"}
        for im in cc["images"]:
            assert (out / split / im["file_name"]).exists()
            assert im["width"] == im["height"] == 400
        for an in cc["annotations"]:
            assert an["iscrowd"] == 0 and len(an["segmentation"][0]) >= 6

    # 4. tuiles 100 % nodata absentes (donc la ligne en zone nodata n'annote rien)
    assert all(tm["bounds"][0] < 501500 for tm in tuiles_m), "tuile pleine zone nodata conservée"

    # 5. négatifs : présents, bornés, dans des blocs affectés, et PURS — aucun négatif
    # ne contient d'entité, même une entité écartée par le filtre de visibilité
    n_annotees = sum(1 for tm in tuiles_m if tm["n_annotations"] > 0)
    n_neg = sum(1 for tm in tuiles_m if tm["n_annotations"] == 0)
    assert 0 < n_neg <= max(1, round(0.5 * n_annotees) + 1), (n_neg, n_annotees)
    blocs_affectes = set(blocs_splits)
    assert all(tuple(tm["bloc"]) in blocs_affectes for tm in tuiles_m)
    toutes_entites = []
    for nom_couche, spec_couche in cfg["couches"].items():
        toutes_entites += preparer_entites(
            gpd.read_file(gpkg, layer=nom_couche), spec_couche.get("buffer_m"))
    union_entites = gpd.GeoSeries(toutes_entites).union_all()
    for tm in tuiles_m:
        if tm["n_annotations"] == 0:
            assert not box(*tm["bounds"]).intersects(union_entites), \
                f"négatif {tm['nom']} contient une entité (filtrée, ignorée ou non)"
    # la couche ignorée n'apparaît nulle part : pas de 4e catégorie, et la tuile (2,3)
    # qu'elle occupe seule est absente du dataset (ni annotée, ni négative)
    assert not any(tm["nom"].endswith("_r0002_c0003.png") for tm in tuiles_m), \
        "tuile occupée par une entité ignorée exportée quand même"

    # 6. carte de contrôle et récap présents
    assert (out / "controle_blocs.html").exists()
    assert manif["comptes"]["train"]["parcellaire"] > 0

    # 7. idempotence (hors horodatage)
    out2 = tmp / "out2"
    run_slicing(cfg, out2, seed=42)
    m2 = yaml.safe_load((out2 / "split_manifest.yaml").read_text(encoding="utf-8"))
    for m in (manif, m2):
        m.pop("genere_le", None)
    assert manif == m2, "non déterministe"

    # 8. relance avec une autre config dans le MÊME dossier : purge complète,
    # le disque reflète exactement le dernier manifeste (zéro orphelin inter-splits)
    cfg_sans_neg = dict(cfg)
    cfg_sans_neg["negatifs_pct"] = 0
    run_slicing(cfg_sans_neg, out, seed=42)
    m3 = yaml.safe_load((out / "split_manifest.yaml").read_text(encoding="utf-8"))
    disque = {s: {p.name for p in (out / s).glob("*.png")}
              for s in ("train", "valid", "test")}
    attendu = {s: {tm["nom"] for tm in m3["tuiles"] if tm["split"] == s}
               for s in ("train", "valid", "test")}
    assert disque == attendu, "fichiers orphelins après relance (purge absente)"
    assert all(tm["n_annotations"] > 0 for tm in m3["tuiles"])  # negatifs_pct 0

    # 9. garde-fou G: refusé
    cfg_g = tmp / "config_g.yaml"
    cfg_g.write_text(cfg_path.read_text(encoding="utf-8").replace(
        str(raster).replace("\\", "\\\\"), "G:\\data\\indice.tif").replace(
        str(raster), "G:\\data\\indice.tif"), encoding="utf-8")
    try:
        charger_config(cfg_g)
        raise AssertionError("chemin G: accepté")
    except SystemExit:
        pass

    print(f"ALL OK — pipeline slice_zone ({len(tuiles_m)} tuiles, "
          f"{n_annotees} annotées, {n_neg} négatives, "
          f"splits {sorted(set(tm['split'] for tm in tuiles_m))})")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
