"""Reconstruit training/vecteurs/haye_entites_l93.gpkg de la zone 54_foret_de_haye.

Sources : raw/Haye_Shp_fourni_par_archéologue (shapefiles Lambert II étendu, EPSG:27572)
Sortie   : un GPKG EPSG:2154, une couche par entité de la taxonomie.

Cas particulier du parcellaire — la livraison contient deux versions :

    v1 (15/01/2011, finalisée)   5924 entités, aucune géométrie nulle
    v2 (2015, « EN COURS »)      5116 lignes dont 232 SANS géométrie

Malgré son nom, v2 n'est PAS une révision : sur les 5114 Id communs, aucune valeur
attributaire ne diffère de v1. C'est une copie de travail qui a perdu du tracé en cours
d'édition — 810 entités de v1 absentes, 232 géométries vidées (attributs intacts),
151 lignes raccourcies — soit 112,6 km sur 647 km (17,4 %). v2 n'apporte que 2 entités
inédites et 5 lignes rallongées.

Preuve que ces pertes sont accidentelles et non un tri archéologique :
- l'attribut `Longueur`, saisi à la digitalisation, est identique dans v1 et v2 y compris
  pour les raccourcies — v2 déclare 832 m là où sa géométrie n'en dessine plus que 97 ;
  la géométrie de v1 correspond à `Longueur` dans 100 % des cas ;
- la perte est géographique et non attributaire : stratifiée par cellules de 1 km, toute
  corrélation avec CHRONO / NETTETE / HABITAT s'annule (Mantel-Haenszel, p >= 0,28), et
  les motifs de rejet (VALLON, TEMPÊTE, DÉTRUIT) sont 4 à 13 fois moins fréquents chez
  les disparues que chez les conservées — l'inverse d'une curation.

D'où la fusion par `Id` (clé fiable : aucun Id réattribué entre v1 et v2, et 96,8 % des
Id communs encore géométrés ont un tracé strictement identique) :

    géométrie  = le tracé le plus long des deux (arbitrage utilisateur 2026-07-27)
    attributs  = v2 quand l'Id y existe, sinon v1 (choix sans effet : v2 == v1)
    + les entités présentes dans v2 seulement (ajouts v2 sans équivalent v1)

Une seule géométrie par Id : aucun doublon possible. Contrôlé en fin de script par une
recherche de paires quasi confondues.

Usage : .venv\\Scripts\\python.exe tools\\build_haye_gpkg.py [--out <dossier>]
Écrit en local ; le dépôt sur G: se fait ensuite par robocopy (jamais d'édition en place).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree, hausdorff_distance

RAW = Path(
    r"G:\Mon Drive\Archeologia\Archeologia_Shared\data\data_regions_v2"
    r"\grand_est\54_foret_de_haye\raw\Haye_Shp_fourni_par_archéologue"
)
CRS_SOURCE = 27572  # NTF (Paris) / Lambert zone II étendu — cf. .prj de la livraison
CRS_CIBLE = 2154  # RGF93 / Lambert-93
ENCODAGE_DBF = "ISO-8859-1"  # aucun octet 0x80-0x9F dans les champs texte des DBF

# stem du shapefile -> nom de couche (= id d'entité de la taxonomie)
COUCHES = {
    "Haye_lidar_chaussee": "chaussee",
    "Haye_lidar_depression": "circular_depression",
    "Haye_lidar_fossébutte": "talus_fosse",
    "Haye_lidar_rempart": "rempart",
    "Haye_lidar_tas": "tas",
    "Haye_lidar_tranchee_chasse": "tranchee_chasse",
}
PARCELLAIRE = {"v1": "Haye_lidar_parcellaire_v1", "v2": "Haye_lidar_parcellaire_v2"}


def lire(stem: str, staging: Path) -> gpd.GeoDataFrame:
    """Lit un shapefile de la livraison et le reprojette en Lambert-93.

    Les .prj sont encodés en latin-1 (« NTF_Lambert_II_étendu ») et pyogrio exige de
    l'UTF-8 : on copie en local sans le .prj et on force le CRS. Haye_lidar_tas n'a de
    toute façon pas de .prj.
    """
    prj = (RAW / stem).with_suffix(".prj")
    if prj.exists():
        wkt = prj.read_text(encoding="latin-1")
        assert "Lambert_II" in wkt, f"{stem}: CRS inattendu, EPSG:{CRS_SOURCE} présumé à tort\n{wkt}"
    for ext in (".shp", ".shx", ".dbf"):
        shutil.copy2((RAW / stem).with_suffix(ext), (staging / stem).with_suffix(ext))

    gdf = gpd.read_file((staging / stem).with_suffix(".shp"), encoding=ENCODAGE_DBF)
    gdf = gdf.set_crs(CRS_SOURCE, allow_override=True).to_crs(CRS_CIBLE)
    gdf["source_layer"] = stem
    return gdf


def fusionner_parcellaire(v1: gpd.GeoDataFrame, v2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Union v1 ∪ v2 par `Id` : géométrie la plus complète, attributs de v2."""
    v1 = v1.copy()
    v2 = v2.copy()
    for g in (v1, v2):
        g["Id"] = g["Id"].astype("Int64")
        assert not g["Id"].dropna().duplicated().any(), "Id non unique : jointure ambiguë"

    par_id_v1 = dict(zip(v1["Id"], v1.geometry))

    lignes, geoms, sources = [], [], []
    stats = {"identique": 0, "v2_recuperee": 0, "v1_plus_longue": 0, "v2_plus_longue": 0,
             "ajout_v2": 0, "v1_seule": 0}

    for _, r in v2.iterrows():  # v2 mène : c'est elle qui porte les Id les plus récents
        ident = r["Id"]
        g2 = r.geometry
        g1 = par_id_v1.get(ident) if pd.notna(ident) else None

        if g1 is None:  # ajout propre à v2 (pas d'équivalent v1)
            if g2 is None:
                continue  # ligne sans géométrie ni contrepartie : rien à écrire
            geom, src = g2, PARCELLAIRE["v2"]
            stats["ajout_v2"] += 1
        elif g2 is None:  # géométrie vidée par l'édition v2, récupérée dans v1
            geom, src = g1, PARCELLAIRE["v1"]
            stats["v2_recuperee"] += 1
        elif g1.equals(g2):
            geom, src = g2, PARCELLAIRE["v2"]
            stats["identique"] += 1
        elif g2.length > g1.length:  # arbitrage utilisateur : toujours le tracé le plus long
            geom, src = g2, PARCELLAIRE["v2"]
            stats["v2_plus_longue"] += 1
        else:
            geom, src = g1, PARCELLAIRE["v1"]
            stats["v1_plus_longue"] += 1

        lignes.append(r.drop(labels="geometry"))
        geoms.append(geom)
        sources.append(src)

    ids_v2 = set(v2["Id"].dropna())
    for _, r in v1[~v1["Id"].isin(ids_v2)].iterrows():  # entités disparues de v2
        lignes.append(r.drop(labels="geometry"))
        geoms.append(r.geometry)
        sources.append(PARCELLAIRE["v1"])
        stats["v1_seule"] += 1

    out = gpd.GeoDataFrame(pd.DataFrame(lignes).reset_index(drop=True),
                           geometry=geoms, crs=v1.crs)
    out["source_layer"] = sources
    # Le DataFrame est reconstruit ligne a ligne : pandas retombe en `object` sur les
    # colonnes numeriques comportant un vide. On restaure les types de la source, sans
    # quoi Id partirait en champ texte dans le GPKG.
    for col, dtype in v1.drop(columns="geometry").dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    # Le seul vrai risque de la boucle est de perdre des lignes en silence : toute entité
    # de v1 doit ressortir, plus les ajouts propres à v2.
    assert len(out) == len(v1) + stats["ajout_v2"], (
        f"{len(out)} entités en sortie pour {len(v1)} dans v1 + {stats['ajout_v2']} ajouts v2")
    assert set(v1["Id"].dropna()) <= set(out["Id"].dropna()), "des Id de v1 ont disparu"

    print("  fusion parcellaire :", ", ".join(f"{k}={v}" for k, v in stats.items()))
    return out


def controler_doublons(gdf: gpd.GeoDataFrame, seuil_m: float = 1.0) -> int:
    """Compte les paires d'entités quasi confondues (Hausdorff <= seuil).

    ponytail: seuil à 1 m et pas plus — la livraison contient de vrais talus parallèles
    distants de 4 m (Id 4931/4932, 5671/5672), déjà présents tels quels dans v1.
    """
    geoms = np.asarray(gdf.geometry)
    paires = STRtree(geoms).query(geoms, predicate="dwithin", distance=seuil_m)
    paires = paires[:, paires[0] < paires[1]]
    if not paires.shape[1]:
        return 0
    hd = hausdorff_distance(geoms[paires[0]], geoms[paires[1]])
    suspects = paires[:, hd <= seuil_m]
    for i, j in suspects.T[:10]:
        print(f"    ! doublon suspect : lignes {i} et {j} "
              f"(Id {gdf.iloc[i]['Id']} / {gdf.iloc[j]['Id']})")
    return int(suspects.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="build/54_foret_de_haye",
                    help="dossier de sortie local (défaut: build/54_foret_de_haye)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / "haye_entites_l93.gpkg"
    gpkg.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        couches = {nom: lire(stem, staging) for stem, nom in COUCHES.items()}
        v1 = lire(PARCELLAIRE["v1"], staging)
        v2 = lire(PARCELLAIRE["v2"], staging)
        couches["parcellaire"] = fusionner_parcellaire(v1, v2)

    total = 0
    for nom, gdf in sorted(couches.items()):
        nulles = int(gdf.geometry.isna().sum())
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        doublons = controler_doublons(gdf)
        assert doublons == 0, f"{nom}: {doublons} paires quasi confondues"
        assert gdf.crs.to_epsg() == CRS_CIBLE, f"{nom}: CRS {gdf.crs}"
        gdf.to_file(gpkg, layer=nom, driver="GPKG")
        total += len(gdf)
        print(f"  {nom:22s} n={len(gdf):5d}  géométries nulles écartées={nulles:4d}  "
              f"types={sorted(set(gdf.geometry.geom_type))}")

    print(f"\n{gpkg}  —  {len(couches)} couches, {total} entités, EPSG:{CRS_CIBLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
