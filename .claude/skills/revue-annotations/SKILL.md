---
name: revue-annotations
description: >
  Assister la revue humaine des annotations d'une zone (GPKG de revue ouverts dans
  QGIS, dossier revue_zones) : opérations sur les couches à la demande, propositions
  de polygones SAM depuis des bounding boxes, puis dépôt du GPKG revu sur le Drive.
  Utiliser quand l'utilisateur, en train de revoir des annotations dans QGIS, dit
  « renomme la couche X », « supprime les annotations de moins de N m », « nettoie
  les mardelles », « crée les polygones SAM sur les boîtes », « où seront rangés mes
  résultats », « c'est stabilisé, traite tout », ou après tout retour de validation
  d'un archéologue (docx + shapefiles de détections annotées).
argument-hint: <zone_id, ex. 25_besancon_chailluz> [<dossier de revue, défaut D:\entrainement_ponctuelles\revue_zones>]
entrees:
  - "revue_zones\\<zone>_annotations.gpkg (bounding boxes, attributs conservés) + <zone>_ld.vrt"
  - "éventuels GPKG de détections annotés par l'archéologue (champ validation oui/peut-être/non + OBS)"
sorties:
  - "couches revues dans le GPKG de revue (+ couche <classe>_sam), sauvegardes .bak à côté"
  - "training/vecteurs/<zone>_entites_l93_v3.gpkg (Drive) + note DÉCISION au manifest"
suivant: [prepare-zone-training, corpus-entrainement]
---

# Revue humaine des annotations

Interagir en français. Commandes des outils : CLAUDE.md § Commands. Journal des
incidents et de leurs parades : `soucis.md` à côté de cette skill (le lire avant
toute opération nouvelle, l'enrichir à chaque incident). L'utilisateur édite **en
direct dans QGIS** le GPKG sur lequel on travaille.

## Règles (non négociables)

1. **Relire avant, recompter après.** Avant toute écriture (ALTER, DELETE, append) :
   QGIS hors édition (`-wal` vide), comptes relus, jamais un chiffre d'un tour
   précédent. Après : recompte indépendant (.venv, pyogrio) annoncé, et dire de
   retirer/rajouter la couche dans QGIS.
2. **Rien d'irréversible.** `.bak` (`<nom>_avant_<operation>.gpkg.bak`, suffixe `_2`
   en cas de collision) avant suppression ou renommage. Suppression > 25 % d'une
   couche : annoncer le chiffre, attendre la confirmation.
3. **Témoin d'intégrité** avant tout run long : `tools\temoin_couche.py <gpkg> <couche>`
   (effectif + sha256 des géométries WKB, écrit `<gpkg>.<couche>.temoin.txt`) ;
   `--comparer` après ; différent = la couche a bougé, on relance.
4. **Sortie séparée.** Un run GPU écrit dans un GPKG à part, jamais dans le fichier
   ouvert par QGIS ; fusion après verdict CONFORME (`DROP TABLE` puis `ogr2ogr
   -update -append -nln <couche> <gpkg_revue> <gpkg_sep> <couche>`).
5. **Contrôler avant de livrer.** Un CONFORME ne vaut que pour ce que le vérificateur
   mesure : lire les distributions (méthode, motifs de repli, tailles, scores,
   remplissage) et les extrêmes ; toute anomalie est annoncée avec sa cause ou
   corrigée et relancée avant d'être annoncée.
6. **Contrôle d'alignement d'abord** pour toute zone dont le LD a été livré en images :
   `verif_alignement_ld.py` contre une référence indépendante (leçon 2026-09-06). Déjà
   fait si `raw/MNT/LISEZ-MOI.md` de la zone et la note DÉCISION du manifest le disent.
7. **Une version déposée ne se modifie pas en place** : la revue humaine d'un GPKG déjà
   déposé (ex. v3 = v2 corrigée, non revue) produit le numéro SUIVANT (v4) ; le dataset
   découpé depuis la version précédente reste tel quel tant que le corpus qui l'utilise
   n'est pas remplacé. La référence pour `decision_humaine` = la dernière version déposée
   (à défaut de `*_v2.gpkg` : `zones\<zone>\annotations_coco_v3.gpkg`, boîtes corrigées).
8. **Localiser le fichier réel** avant d'agir : une zone peut être dans `traité\` et encore
   en revue ; ses fichiers de travail (.bak, témoin, logs, GPKG `_sam`, `_dedup`) restent
   à côté du GPKG, où qu'il soit.

## Où sont les fichiers

- Revue : `revue_zones\<zone>_annotations.gpkg` + VRT (`_ANCIEN`/`_NOUVEAU` pour les
  LD régénérés). `<zone>_detections_*.gpkg` = sorties de modèle (`score IS NULL` =
  dessiné à la main ; dans les `_annotations`, proxy `uid IS NULL`). Mesurer les tailles
  sur `geom`, jamais sur `largeur_m/hauteur_m`.
- Zone close : fichiers déplacés dans `revue_zones\traité\` + `<zone>_DECISIONS.txt`
  (source de la note DÉCISION du manifest). **Regarder `traité\` avant de déclarer
  une zone manquante.**
- Référence v2 : `zones\<zone>\*_v2.gpkg` (ellipses, mêmes `uid`) — comparer bbox à
  bbox. `raw/` du Drive : jamais modifié.
- Destination : `data_regions_v2/<region>/<zone>/training/vecteurs/<zone>_entites_l93_v3.gpkg`
  (numéro = génération des datasets). Détections brutes hors data_regions_v2.

## Opérations

| Demande | Commande (GDAL `C:\OSGeo4W\bin\`) | Convention |
|---|---|---|
| renommer une couche | `ogrinfo <gpkg> -sql "ALTER TABLE a RENAME TO b"` | nom libre ≠ id taxonomie : mapping dans la config de re-slice, alias append-only proposé |
| supprimer par taille | `ogrinfo <gpkg> -dialect sqlite -sql "DELETE FROM c WHERE <diam> < N"` avec `<diam>` = `((ST_MaxX(geom)-ST_MinX(geom))+(ST_MaxY(geom)-ST_MinY(geom)))/2.0` | **diamètre = moyenne largeur/hauteur** ; annoncer aussi max et min ; `COUNT(*)` même condition avant ; dire combien étaient dessinées à la main |
| polygones SAM sur boîtes | `venv_sam tools\sam_polygones_bbox.py <gpkg> <couche> <ld.vrt> revue_zones\<zone>_<classe>_sam.gpkg --marge M` puis `tools\verif_sam_polygones_bbox.py` (MÊME marge) | `--marge` en **m par côté** ; SAM 2.1 par défaut, pas le décodeur Irlande ; marge 0 = choix acté 2026-09-06 (score médian 0,85 contre 0,61 à 20 m) ; test `--limite 40` une fois par zone et classe (vérifié sur les 40 boîtes extraites) ; en fond via Bash, `python -u` + `grep --line-buffered` ; relance complète si la couche bouge (`fid_source` permet un incrément à outiller) |
| dédoublonner | `tools\dedup_annotations.py <gpkg> <sortie_separee> --iou 0.5 --couches c` puis rapatriement `DROP TABLE c` + `ogr2ogr -update -append` de `c` ET `c_doublons` | obligatoire après toute correction géométrique (doublons inter-dalles) ; l'outil garde la PREMIÈRE de chaque paire — si l'utilisateur a retouché l'autre dans QGIS, le lui dire avant ; `c_doublons` devient `c_supprimees` au dépôt |
| test `--limite 40` | `ogr2ogr -limit 40 <tmp.gpkg> <gpkg> <couche>` (même ordre fid que le producteur) puis `verif_sam_polygones_bbox.py <tmp.gpkg> <couche> <sortie_test> --marge M` | le vérificateur compare à l'effectif de l'entrée : toujours extraire les 40 boîtes |

Les polygones SAM sont des **propositions à revoir** ; la cible d'entraînement
détection reste la couche de boîtes (config v3 : `<classe>_sam` mappé seulement
sur décision explicite). Marge < 10 m : prévenir (SAM déborde de 2 m en médiane),
exécuter quand même.

## Retour de validation d'un archéologue

Docx + shapefiles (`validation` oui / peut-être / non, `OBS`) : identifier la zone
par l'emprise EPSG:2154 (= dalles LHD du payload) et le contact du manifest ;
dumper OBS × validation (négatifs typés : batteries, entonnoirs, paléochenaux,
chablis) ; ranger dans `raw/docs/` de la zone, pas dans le repo.

## Dépôt (au « c'est bon » de l'utilisateur)

1. Staging local `<zone>_entites_l93_v3.gpkg`, EPSG:2154 : couches revues avec
   `decision_humaine` ∈ {inchangé, modifié, ajouté} (par `uid` contre la v2, bbox
   contre bbox ; ajouts `revue:<fid>`) et `geom_origine` ; supprimés dans
   `<classe>_supprimees` (jamais dans une couche d'entraînement) ; `<classe>_sam`
   telle quelle ; couches non traitées recopiées de la v2 ; couches IGNORER reportées
   en `ignorer: true` dans la config v3. Script de staging + son `verif_` à écrire au
   premier dépôt et à promouvoir dans `tools/` (schéma : `decision_humaine` texte,
   `geom_origine` WKT texte, `motif` texte dans `_supprimees` ∈ {doublon, supprime_qgis} ;
   couches non traitées comparées quand même par uid — l'utilisateur a pu les toucher).
2. `Get-PSDrive G` (DriveFS meurt en session : relancer `launch.bat`) ; `robocopy
   <staging> <training\vecteurs> /E /MT:16` (exit 1 = succès) ; sha256 et effectifs
   relus sur G:.
3. Note `DÉCISION <date>:` au manifest (copie locale → redépôt) + `build_v2_index.py`.
   « Dépose » s'arrête là ; `/prepare-zone-training` (`--split-depuis`) est proposé,
   pas lancé. `.bak`, logs et GPKG `_sam` séparés restent locaux.
