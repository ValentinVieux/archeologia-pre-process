---
name: prepare-zone-training
description: >
  Préparer une zone de data_regions_v2 pour l'entraînement segmentation : GPKG
  d'entités depuis la livraison auditée, découpe en tuiles 648 px avec split
  spatial par blocs, boucle de vérification, carte de contrôle, dépôt Drive.
  Utiliser quand l'utilisateur dit « prépare la zone X », « découpe X pour
  l'entraînement », « on passe à la zone suivante », « crée le dataset de X »,
  « slice cette zone », « prepare zone for training ».
argument-hint: <zone, ex. ile_de_france/78_rambouillet>
---

# Préparer une zone pour l'entraînement (GPKG → découpe → vérification → dépôt)

Interagir en français. Les commandes exactes sont dans CLAUDE.md § Commands ; les
règles dans § Entraînement, § Taxonomie et § Stockage Drive font autorité. La
mémoire persistante documente les pièges déjà rencontrés (données et plateforme).

## Étape 1 — Résoudre la zone et ses prérequis
- Zone depuis l'argument, sinon demander. Vérifier sur le Drive : manifest.yaml,
  livraison vecteur en raw/, statut d'audit (alias tracés dans taxonomy/aliases.yaml).
- **Audit manquant → s'arrêter et dérouler `/audit-dataset` d'abord** (0 inconnu requis).
- Raster : un indice de visualisation à 0,5 m/px (LD paramétrisation actée) avec VRT,
  produit par le pipeline v2.0 (run sur D:\pipeline_results, dépôt raw\<Zone>_MNT_IGN,
  ou run v1 gelé en lecture seule). Absent → le signaler (TODO manifest), c'est un
  préalable côté utilisateur. Toujours travailler sur COPIE LOCALE (les outils
  refusent G:) ; vérifier le nom réel du VRT (souvent `tif\index*.vrt`).

## Étape 2 — GPKG d'entités
- Écrire `configs/vecteurs_<zone>.yaml` : valeurs brutes → entity_id STRICTEMENT
  selon aliases.yaml ; couches de gestion/prospection exclues ; labels talus/fossé
  indistincts → `talus_fosse`, distincts → `talus`/`fosse`.
- Faire valider par l'utilisateur tout cas non couvert par un alias existant.
- `build_zone_gpkg` puis **`verif_zone_gpkg`** (boucle : corriger la config et
  reconstruire jusqu'à CONFORME). Déposer sur
  `training/vecteurs/<zone>_entites_l93.gpkg` (staging + robocopy).

## Étape 3 — Config de découpe
- Écrire `configs/lineaires_<zone>_ld_648_v1.yaml` (gabarit : zones existantes) :
  classes ENTRAÎNÉES validées par l'utilisateur ; le linéaire non entraîné passe en
  `ignorer: true` (jamais silencieusement retiré) ; buffer par zone validé par
  l'utilisateur (largeur totale ; historique : 4,8 m Haye/Fontainebleau, 5 m
  Rambouillet) ; `nodata_supplementaire: 0` pour les mosaïques à fond implicite.

## Étape 4 — Découpe + boucle de vérification
- `slice_zone` (seed 42, sortie locale). Puis **`verif_dataset`** : toute divergence
  → corriger → régénérer INTÉGRALEMENT → re-vérifier. Ne jamais rafistoler la sortie
  à la main.

## Étape 5 — Carte de contrôle (validation humaine obligatoire)
- Publier `controle_blocs.html` (artifact) et attendre la validation de l'utilisateur
  avant tout dépôt. Signaler les classes rares mal réparties (granularité des blocs).

## Étape 6 — Dépôt et comptabilité
- Miroir vers `training/datasets/<dataset>/` (robocopy /MIR, comparer comptes + SHA1
  du split_manifest). Note `DÉCISION:` au manifest de zone (comptes par classe,
  splits, reproductibilité), `TODO:` pour l'upload. Régénérer index.html.

## Étape 7 — Récapitulatif
- Tableau : tuiles/splits/classes/négatifs, chemins, commits proposés (config +
  éventuels outils). Proposer d'enchaîner sur `/upload-roboflow`.

## Garde-fous
- Boucle de vérification NON NÉGOCIABLE : aucune livraison avant verdict conforme.
- Zéro écriture sur G: hors staging+robocopy ; data_regions v1 = lecture seule.
- Sorties locales régénérables non commitées ; les configs YAML, si.
- Relancer la skill sur une zone déjà préparée : zéro question, re-vérification puis
  arrêt (idempotence via manifestes et hashes).
