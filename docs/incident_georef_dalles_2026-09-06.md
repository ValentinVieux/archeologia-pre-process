# Incident : géoréférencement des dalles LD « à marge » (2026-09-06)

## Symptôme

Dans QGIS (EPSG:2154), les annotations de Haut-Doubs se superposaient au LD livré par
les archéologues mais pas au LD régénéré depuis le MNT IGN ni à Google Maps.

## Cause

Les dalles LD reçues via Roboflow existent en deux formats : 2 000 px (1 000 m à 0,5 m,
correct) et **2 201 px** (1 100,5 m à 0,5 m : le kilomètre + 50,25 m de marge de chaque
côté, marge de calcul du LD). `tools/coco_a_gpkg.py` supposait `px = 1000 / width`
depuis le coin NW du nom LHD : les dalles de 2 201 px étaient comprimées de 10 % vers
ce coin (erreur nulle au centre, ±50 m aux bords) — et les annotations converties
depuis ces images avec elles. Le « gsd bâtard 0,4543 » des manifests v1 était le
symptôme (1000/2201). Zones touchées : 25_haut_doubs (246/264 dalles),
30_ales_garrigues_ne (473/509), 30_la_capelle_et_masmolene (77/77) ; toutes les
autres zones sont en 2 000 px.

## Preuves

- Corrélation de phase LD livré / LD régénéré à 9 positions dans une dalle : décalage
  qui change de signe entre nord et sud (±15-40 m), puis 0 ± 0,5 m après correction
  (origine NW − 50,25 m, 0,5 m/px).
- Sous les boîtes corrigées, le LD régénéré montre le même contenu que le LD livré
  sous les boîtes d'origine (corrélation médiane 1,00 sur 60 fours ; −0,03 sans correction).
- Rappel des fours par `fours_charb_seg_ld_v1_ep41` (Haut-Doubs, même code) :
  LD livré + annotations d'origine 94/82/64 % @0,05/0,15/0,43 ; LD régénéré +
  annotations corrigées 94/81/63 % ; LD régénéré + annotations comprimées 17/8/5 %.
  Le diagnostic « MNT sol strict gomme les fours (87 % vs 11 %) » du 2026-09-04 était
  un artefact du décalage.
- Les « joints fautifs 39/39 » de l'audit du 2026-09-01 sur ces trois zones = dalles
  voisines qui se recouvrent réellement de 50 m mais déclarées jointives.

## Conséquences

Datasets `ponctuelles_{25_haut_doubs,30_ales_garrigues_ne,30_la_capelle}_ld648_v2`
(LD régénéré juste + annotations comprimées), corpus `ponctuelles_648_v2` et run
`ponctuelles_det_ld_v1` : INVALIDES sur ces zones (labels décalés jusqu'à 50 m).
Les datasets v1 (LD livré + annotations comprimées ensemble) étaient cohérents en
interne : seg v1 n'en souffre pas directement, mais ses tuiles étaient à 0,4543 m
déclarés pour 0,5 m réels et contenaient les recouvrements de 50 m.

## Correctifs (2026-09-06/07)

| Quoi | Où |
|---|---|
| garde-fou à la racine : image = `width × 0,5 m` centrée sur le km, largeur incompatible = REFUS | `coco_a_gpkg.geo_dalle()`, `verif_coco_a_gpkg.py` aligné |
| correction des annotations existantes (par `tuile` + largeur réelle de l'image) | `tools/corriger_georef_dalles.py` |
| contrôle d'alignement de tout raster livré/régénéré (verdict ALIGNÉ / TRANSLATION / ÉCHELLE), OBLIGATOIRE avant découpe | `tools/verif_alignement_ld.py` ; CLAUDE.md § Entraînement ; skill prepare-zone-training étapes 1 et 5 |
| carte de contrôle sur un raster INDÉPENDANT des images livrées | skill prepare-zone-training étape 5 |
| `inferer_corpus.py` accepte un dataset brut (repli `dataset` = nom du dossier) | outil |

## Correspondance des fichiers (Haut-Doubs)

| Fichier | Nature | Géoréf |
|---|---|---|
| Drive `raw/MNT/haut_doubs_mnt.tif` | MNT IGN mosaïqué 0,5 m | juste |
| Drive `raw/MNT/haut_doubs_ld.tif` (= local `D:\entrainement_ponctuelles\haut_doubs\haut_doubs_ld.tif`, revue_zones `25_haut_doubs_ld_NOUVEAU.vrt`) | LD de référence | juste |
| Drive `raw/MNT/LD_livre_dalles/` | dalles livrées reconverties | corrigé |
| local `D:\entrainement_ponctuelles\zones\25_haut_doubs\rasters\` + `ld.vrt` (revue_zones `25_haut_doubs_ld_ANCIEN.vrt`) | dalles livrées telles que géoréférencées par l'ancien outil, base du dataset v1 | FAUX (conservé pour reproduire v1) |
| Drive `training/vecteurs/25_haut_doubs_entites_l93_v3.gpkg` (= local `zones/25_haut_doubs/annotations_coco_v3.gpkg`, revue_zones `25_haut_doubs_annotations_NOUVEAU.gpkg`) | annotations corrigées, bounding boxes, sans revue humaine | juste |
| local `zones/25_haut_doubs/annotations_coco{,_v2}.gpkg`, revue_zones `25_haut_doubs_annotations.gpkg` | annotations comprimées (v1 ellipses, v2 fusions de paires, revue bbox) | FAUX |

Alès et La Capelle : même correctif à appliquer (`corriger_georef_dalles.py`), puis
re-découpe v3 des trois zones, corpus v3, relance de la détection.
