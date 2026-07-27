# Spec — `tools/slice_zone.py` : découpeur de tuiles d'entraînement à split spatial

Statut : validé sur les décisions D1–D4 du 2026-07-27 (compte rendu :
https://claude.ai/code/artifact/86379b2c-aec1-4a68-8961-ad2e96fe337f — compagnon de
`docs/fuite_spatiale_train_test.html`, dont ce script implémente le remède).

## Objectif

Produire, pour une zone de `data_regions_v2`, un dataset de segmentation COCO
(train/valid/test) à partir d'un raster d'indice de visualisation et du GPKG d'entités,
**sans chevauchement de tuiles** et avec un **split spatial par blocs** équilibré par
classe — jamais de split aléatoire. Étage 1 de l'architecture d'entraînement v2 ;
consommé ensuite par `build_corpus.py` (hors périmètre ici).

## Décisions actées

- **Tuile 648×648 px, sans chevauchement** (640 invalide pour RF-DETR seg : divisibilité
  par 24 ; le chevauchement reste réservé au SAHI d'inférence du plugin).
- **Image d'entrée : mono-indice** (LD paramétrisation `A15_Rmin10_Rmax20_H1p7` pour
  Haye), répliqué sur 3 canaux à l'écriture.
- **Source de vérité locale** : le split vit dans `split_manifest.yaml`, versionné avec
  le dataset ; Roboflow n'est qu'un miroir de contrôle qualité.
- Données Haye : `raw/Haye_MNT_IGN/` (158 dalles LHD IGN 0,5 m/px, EPSG:2154 embarqué,
  un VRT par indice, emprise couvrant le GPKG). Même CRS que les vecteurs — aucune
  reprojection pour Haye ; la capacité de reprojection reste nécessaire pour les zones
  hétérogènes (ex. mosaïques d'archéologues en 27572, parfois sans CRS embarqué).

## Interface

```
.venv\Scripts\python.exe tools\slice_zone.py <dataset_config.yaml> [--out <dossier>] [--seed N]
```

Un dataset = un fichier de config YAML (versionnable, recopié dans le manifeste de
sortie). Champs :

```yaml
dataset: lineaires_haye_ld_648_v1        # nom du dataset (dossier de sortie)
zone: grand_est/54_foret_de_haye         # zone v2 (traçabilité)
raster: <chemin local du VRT/TIF indice> # copie locale — le script REFUSE un chemin G:\
assign_crs: null                         # ex. EPSG:27572 si le raster n'a pas de CRS
gpkg: <chemin local du GPKG>
couches:                                 # couche GPKG -> classe + rasterisation
  parcellaire:  {classe: parcellaire,  buffer_m: 2.0}   # lignes : buffer largeur totale
  talus_fosse:  {classe: talus_fosse,  buffer_m: 2.0}
  rempart:      {classe: rempart,      buffer_m: 2.0}
  # polygones : buffer_m absent ; points : buffer_m = rayon (ex. tas: 5.0)
tuile_px: 648
bloc_m: 2000                             # aligné sur les multiples de bloc_m en CRS raster
split: {train: 70, valid: 20, test: 10}
negatifs_pct: 10                         # % de tuiles vides ajoutées, tirées par bloc
min_couverture_valide: 0.5               # tuile écartée si < 50 % de pixels valides
min_visibilite_annotation: 0.5           # annotation écartée si > 50 % dans le nodata
nodata_supplementaire: null              # ex. 0 pour les mosaïques à fond implicite
```

`--seed` (défaut 42) ne sert qu'aux départages ; le pipeline est déterministe.

## Comportement

1. **Ouverture** : raster (VRT accepté) ; si `assign_crs`, l'assigner ; refuser un chemin
   sous `G:\` (règle « copie locale d'abord »). Vecteurs reprojetés vers le CRS du raster
   si besoin (jamais l'inverse).
2. **Grille** : tuiles 648 px jointives, origine = coin haut-gauche du raster ; les
   tuiles partielles de bord sont écartées (pas de padding : pas de bords noirs
   artificiels dans le dataset).
3. **Validité** : masque nodata = nodata déclaré ∪ `nodata_supplementaire` ; le nodata
   `nan` déclaré sur bande Byte (incohérence connue des chaînes LHD) est ignoré avec
   avertissement. Tuile gardée si couverture valide ≥ `min_couverture_valide`.
4. **Annotations** : par tuile, entités intersectantes → buffer selon la couche → clip →
   polygones COCO (pixels, catégories = classes de la config). Annotation dont l'emprise
   est majoritairement invalide (seuil `min_visibilite_annotation`) écartée.
5. **Blocs** : grille de `bloc_m` alignée sur les multiples de `bloc_m` ; une tuile
   appartient au bloc contenant son **centre**.
6. **Split par blocs, équilibré par classe** : blocs triés par richesse décroissante,
   affectation gloutonne au split le plus déficitaire par rapport aux cibles, pondérée
   par classe (les classes rares pèsent plus) ; départages par seed. Aucune tuile ne
   change jamais de split au sein d'un bloc.
7. **Négatifs** : parmi les tuiles valides sans annotation **des blocs affectés à un
   split** (les blocs entièrement vides ne sont pas affectés et sont ignorés),
   échantillon de `negatifs_pct` % du nombre de tuiles annotées ; un négatif porte le
   split de son bloc.
8. **Sorties** (`--out`, défaut `datasets\<dataset>\` local — jamais G: directement ;
   le dépôt Drive vers `training/datasets/<dataset>/` est l'affaire de la skill, via
   staging + robocopy) :
   - `train/ valid/ test/` : images PNG RGB 8 bits (indice répliqué ×3) +
     `_annotations.coco.json` par split ;
   - nommage image : `<zone_id>_r{row:04d}_c{col:04d}.png` — unique multi-zones ;
   - `split_manifest.yaml` : config résolue + seed + grille (origine, GSD) + par tuile
     `{nom, row, col, bloc, split, bounds, n_annotations, classes}` + comptes par
     classe/split + sha1 des listes de noms par split ;
   - `controle_blocs.html` : carte SVG autonome des blocs colorés train/valid/test
     (façon Fig. 4 du doc fuite spatiale) + tableau des comptes par classe/split —
     support du contrôle visuel humain avant tout upload.
9. **Récapitulatif stdout** : ligne `Sorties :` avec les chemins exacts (convention CLI
   du repo), comptes par classe/split, part de chaque classe par split, tuiles écartées.

## Garanties

- **Zéro fuite par pixels partagés** : tuiles jointives sans chevauchement, une tuile
  appartient à exactement un split.
- **Reproductible** : même config + même seed → sorties identiques (les hashes du
  manifeste en font foi).
- Limite résiduelle assumée : adjacence de tuiles de splits différents aux frontières de
  blocs (documentée dans le compte rendu). Une option `bande_tampon: true` (écarter les
  tuiles frontalières) pourra être ajoutée plus tard si un besoin dur apparaît — YAGNI
  pour la v1.

## Boucle de vérification des données (règle utilisateur du 2026-07-27, systématique)

Chaque production de dataset suit la boucle : **produire → vérifier → corriger →
reproduire → re-vérifier**, jusqu'à zéro défaut confirmé. La vérification porte sur les
**fichiers produits** (pas sur le code) et est menée par des contrôleurs indépendants :
intégrité COCO, anti-fuite (unicité des tuiles par split, cohérence manifeste/disque,
blocs mono-split, hashes), fidélité géométrique aux sources GPKG (IoU sur échantillon,
enclos non remplis), conformité des images au raster source, pureté des négatifs et
recomptes par classe/split. Aucune livraison (dépôt Drive, upload Roboflow) avant le
verdict « conforme ». La future skill `/prepare-zone-training` intègre cette boucle
comme étape obligatoire, en plus du contrôle visuel humain de `controle_blocs.html`.

## Tests — `tests/test_slice_zone.py` (TDD, même style que `tests/test_audit.py`)

Raster jouet généré (petit GeoTIFF synthétique avec zone nodata) + GPKG jouet (lignes,
polygones, points, dont une ligne traversant plusieurs blocs). Asserts :

- grille : positions/tailles des tuiles, tuiles partielles écartées, pas de recouvrement ;
- affectation par centre ; toutes les tuiles d'un bloc dans le même split ;
- proportions du split dans une tolérance raisonnable, classes rares présentes dans
  chaque split quand c'est géométriquement possible ;
- COCO : JSON chargeable, polygones dans l'emprise, catégories conformes ;
- nodata : tuile sous le seuil écartée, annotation majoritairement invalide écartée ;
- idempotence : deux runs → manifestes identiques (hors horodatage).

## Dépendances

`rasterio` ajouté à `requirements.txt` (roue binaire Windows, aucune dépendance système).
Le reste (geopandas, shapely, pyogrio, pyyaml) est déjà dans le venv.

## Hors périmètre (specs/étapes ultérieures)

`build_corpus.py` (fusion multi-zones, holdout LOZO), upload Roboflow à split imposé,
notebook v2, skills `.claude` (`/prepare-zone-training` orchestrera ce script : choix de
l'indice, validation de la config, contrôle de `controle_blocs.html`, dépôt Drive,
note manifest, index). Ancien `tif_slicer*` de `pipelines/utils_functions` : conservé
en l'état côté pipelines, non maintenu pour l'entraînement.
