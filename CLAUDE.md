# CLAUDE.md

Préprocessing des données d'entraînement des modèles CV RF-DETR (données sur Roboflow,
modèles consommés par le plugin QGIS `archeologia-pipeline`). Des archéologues de toute
la France envoient des datasets vecteur hétérogènes ; ce repo les audite et uniformise
les nommages vers une taxonomie maîtresse.

Deux couches : un outil Python **déterministe** (audit.json + rapport HTML statique par
livraison) et une couche sémantique pilotée par Claude (skill `/audit-dataset`) qui mappe
les noms bruts sur la taxonomie, avec validation humaine de chaque décision.

## Commands (Windows — ne jamais activer le venv, appeler son python directement)

```
.venv\Scripts\python.exe -m audit "<dataset-path>" [--no-open]   # audit d'une livraison
.venv\Scripts\python.exe tests\test_audit.py                     # auto-test complet
.venv\Scripts\python.exe tools\dispatch_roboflow.py <attr.json> <zips> <staging>  # cf. /dispatch-roboflow
.venv\Scripts\python.exe tools\build_v2_index.py "<racine data_regions_v2>"       # régénère index.html
.venv\Scripts\python.exe tools\build_haye_gpkg.py [--out <dossier>]  # reconstruit le GPKG de 54_foret_de_haye
.venv\Scripts\python.exe tools\slice_zone.py <config.yaml> [--out <dossier>] [--seed N]  # tuiles 648px + split spatial (cf. configs\)
.venv\Scripts\python.exe tools\build_zone_gpkg.py configs\vecteurs_<zone>.yaml <source> [--out <dossier>]  # GPKG entités depuis livraison auditée
.venv\Scripts\python.exe tools\verif_zone_gpkg.py configs\vecteurs_<zone>.yaml <source> <gpkg>  # boucle de vérification du GPKG
.venv\Scripts\python.exe tools\verif_dataset.py <dataset> [--gpkg <chemin>]  # boucle de vérification d'un dataset découpé
.venv\Scripts\python.exe tools\upload_roboflow_split.py <dataset> --workspace <id> --projet <id> [--test] [--dry-run]  # upload split IMPOSÉ (clé : env ROBOFLOW_API_KEY)
.venv\Scripts\python.exe tools\recaler_lignes.py configs\recalage_<zone>.yaml <gpkg> <raster> [--out <dossier>]  # recalage des lignes sur le relief (méthode B)
.venv\Scripts\python.exe tools\verif_recalage.py configs\recalage_<zone>.yaml <gpkg_source> <gpkg_recale> <raster>  # boucle de vérification du recalage
.venv\Scripts\python.exe -m tools.review_recalage <gpkg_recale> <raster> [--port 5175]  # app locale de revue/édition (décisions YAML, jamais le GPKG)
.venv\Scripts\python.exe tools\analyse_corrections.py <decisions.yaml> <gpkg_recale>  # typologie des corrections + suggestions de paramètres
.venv\Scripts\python.exe tools\appliquer_decisions.py <gpkg_source> <gpkg_recale> <decisions.yaml> [--out <gpkg_final>]  # GPKG final (décisions humaines appliquées)
.venv\Scripts\python.exe tools\verif_application.py <gpkg_source> <gpkg_recale> <decisions.yaml> <gpkg_final>  # boucle de vérification de l'application
.venv\Scripts\python.exe tools\coco_a_gpkg.py <payload> <sortie> --classes <c1> ... [--rasters]  # GPKG EPSG:2154 depuis un payload COCO dispatché (zones sans vecteurs source, ex. 57_fenetrange) + tuiles LD GeoTIFF ; uid=split:annotation_id, dalles LHD = coin NW (vérifié WFS IGN)
.venv\Scripts\python.exe tools\verif_coco_a_gpkg.py <payload> <sortie>  # boucle de vérification (recompte + regéoréférencement indépendants)
.venv\Scripts\python.exe tools\telecharger_dalles_ign.py <entites.gpkg> <sortie> [--anneau 1] [--mt 4]  # dalles MNT LiDAR HD IGN (GeoTIFF 1 km 0,5 m) via la grille WFS : cellules occupées + anneau, reprise idempotente, CRS estampillé
D:\veille_irlande\venv_adaf\Scripts\python.exe tools\auto_label_depressions.py <ld.tif> <selection.gpkg> <sortie.gpkg> --poids <ckpt>  # auto-labels circular_depression par run_rf_detr_1 @0,395 (tuiles 1 km alignées grille, parité d'inférence) ; 0 détection = garde-fou, diagnostiquer au plancher avant d'accepter
.venv\Scripts\python.exe tools\build_gpkg_fours_charbonnieres.py <zone> <sortie.gpkg> [--source|--payload|--auto-labels]  # GPKG des zones spéciales du corpus fours/charbonnières (chailluz r=5 m, blois rayons réels, rambouillet COCO+GPKG+ignorer)
.venv\Scripts\python.exe tools\mosaique_mnt.py <dossier_dalles> <sortie.tif> [--tr 1.0]  # MNT téléchargés (étrangers ou IGN) -> UNE mosaïque EPSG:2154 en une passe ; JAMAIS de reprojection dalle par dalle (canevas union + joints RVT)
.venv\Scripts\python.exe tools\telecharger_dalles_gsi.py --noms <DATA_NAME...> [--couche <couche>] [--epsg2154]  # dalles LiDAR ouvertes irlandaises (DTM GeoTIFF) depuis l'index D:\veille_irlande ; --liste pour l'inventaire ; cf. suivi_corpus.yaml (corpus Irlande)
.venv\Scripts\python.exe tools\generer_ld.py <mnt.tif> <sortie_ld.tif>  # LD 8 bits, rayons AUTO selon résolution (anneau 5-10 m constant), étirement fixe, auto-vérifié CONFORME
.venv\Scripts\python.exe tools\planche_indices.py <mnt> <sortie> --emprise xmin ymin xmax ymax [--nom d] [--gpkg gt.gpkg]  # planche de TOUS les indices RVT + variantes (choix des canaux multicanaux) — MNT 1 m exigé, recettes VAT/CVAT/e3MSTP vérifiées sur l'install rvt-qgis
D:\veille_irlande\venv_sam\Scripts\python.exe tools\proposer_polygones_irlande.py <points.gpkg> <ld.tif> <sortie.gpkg>  # propositions hybrides cercle+SAM (corpus Irlande, cf. /corpus-irlande) — venv_sam OBLIGATOIRE (torch)
.venv\Scripts\python.exe tools\verif_polygones_irlande.py <points.gpkg> <propositions.gpkg>  # boucle de vérification des propositions (CONFORME requis avant revue humaine)
.venv\Scripts\python.exe tools\build_corpus.py configs\corpus_lineaires_v2.yaml <dossier_datasets> [--out <dossier>]  # corpus d'entraînement multi-zones (classes canoniques) — ATTENTION : --out est le dossier DU corpus (rmtree !), défaut corpus\<nom> ; ne JAMAIS passer --out corpus
.venv\Scripts\python.exe tools\repeindre_dataset.py <dataset_ld_v1> datasets --mnt <tif|glob> [--ld <raster>]  # datasets multicanaux (csl+crim) aux pixels recalculés, splits/COCO INTACTS — jamais re-slicer pour changer de canaux
D:\veille_irlande\venv_adaf\Scripts\python.exe tools\courbes_eval.py --coco <parent valid+test|split> --modele "nom=best.pth@res" [--modele ...] --out <dossier> [--tache detection|segmentation] [--adopter-cache]  # éval outillée DÉTECTION+SEG (remplace seuil_f1_detection.py) : metriques_eval.json CANONIQUE (source des seuils du model_card) + planches P/R/F1/PR + cache appariements.json à empreinte (venv_adaf — GPU ; cache = re-rendu sans ré-inférence ; autocontrôle de chargement)
.venv\Scripts\python.exe tools\tableau_modeles.py "<racine model-training>" [--out <html>]  # dashboard HTML famille/version/classe depuis les metriques_eval.json (sparklines d'évolution, runs sans mesure listés) ; régénérer après tout dépôt d'évaluation
.venv\Scripts\python.exe tools\points_a_recaler.py <ld.tif> <sortie.gpkg> [--smr <points.gpkg>] [--couches ...]  # couche de recalage humain : points sur données valides, garde-fou bord 20 m, tri par contraste
.venv\Scripts\python.exe tools\fermer_lignes_emprises.py <lignes.gpkg> <sortie.gpkg> --couches <c1> <c2> ...  # lignes -> emprises pleines (chaînage+morpho, banc perforation IoU 0,998) ; 3 étages : auto / à vérifier / arbitrage humain
.venv\Scripts\python.exe tools\verif_corpus.py configs\corpus_lineaires_v2.yaml <dossier_datasets> <corpus>  # boucle de vérification du corpus
# --- banc d'essai d'inférence (tools\bench\, sorties D:\pipeline_results\bench) ---
docker build -t archeologia-bench:cpu --build-arg BASE=python:3.11-slim-bookworm --build-arg ORT_PKG=onnxruntime==1.24.1 tools\bench
docker build -f tools\bench\Dockerfile.gpu -t archeologia-bench:gpu tools\bench   # CUDA par wheels pip
# montages : /data (splits stagés en local), /plugin (plugin QGIS, ro), /harness (ce repo, ro), /out (D:\pipeline_results)
#   python3 -m tools.bench forward --data /data/valid --device cpu --floor 0.05   # cache des sorties brutes
#   python3 -m tools.bench e0      --data /data/valid                            # plafond de rappel
#   python3 -m tools.bench sweep   --data /data/valid --axes /harness/configs/bench/e2_un_axe.yaml --cle <cle>
#   python3 -m tools.bench bootstrap --run e2_un_axe                             # IC95 apparié par tuile
#   python3 -m tools.bench niveaub --data /data/test --gpkg /vec --axes .../e_niveaub.yaml
#   python3 -m tools.bench.report                                                # rapport HTML
.venv\Scripts\python.exe tools\verif_bench.py --out D:\pipeline_results\bench     # contrôleur indépendant
<venv_onnx>\python.exe tests\test_parity_bench.py    # PORTE : decode.py doit reproduire le plugin à l'identique
# setup initial : py -3.11 -m venv .venv ; .venv\Scripts\pip install -r requirements.txt
```

Sortie par défaut : `audits\<nom-normalisé>\` (stable, écrasée à chaque run — les audits
sont régénérables, ils ne sont pas commités). Normalisation = `normalize()` de scan.py :
accents supprimés (NFKD), minuscules, tout run de caractères hors `[a-z0-9]` fusionné en
un seul `_`, `_` de bord retirés, repli `dataset` si vide. Ex. « Prospection Rambouillet »
→ `audits\prospection_rambouillet\`. La CLI imprime les chemins exacts sur stdout
(ligne `Sorties :`). Cette section est la source de vérité de la CLI : la skill
`audit-dataset` lit la commande ici.

## Taxonomie — taxonomy/entities.yaml + taxonomy/aliases.yaml

Schémas documentés en tête de chaque fichier. Règles non négociables :

- `id` : snake_case ASCII, regex `^[a-z][a-z0-9_]*$` (contrat `docs/model_contract.md` du
  plugin). Les accents comptent partout ailleurs : `charbonniere` (id) ≠ `charbonnière`
  (label/classe Roboflow, conservée verbatim).
- **Ne JAMAIS renommer un id `canonical`** sans vérifier le plugin
  (`archeologia-pipeline/src/config/config_manager.py` → `_ENTITY_ID_RENAMES`) et les
  classes Roboflow liées (`roboflow_classes`).
- Claude ne crée des entités qu'en `status: candidate`. La promotion en `canonical` et le
  remplissage de `plugin_entity_id`/`roboflow_classes` sont des décisions humaines.
- `aliases.yaml` est **append-only** : jamais modifier ni supprimer une entrée existante.
- Après toute édition : valider (regex des ids, unicité des ids, chaque `entity_id`
  d'alias existe dans entities.yaml, `morphology` ∈ {circulaire, lineaire, zone},
  `status` ∈ {canonical, candidate}).

## Workflow

audit (Python) → classification interactive des inconnus (l'utilisateur valide chaque
mapping) → append aliases + entités candidates → relancer l'audit. Le tout via
`/audit-dataset <chemin>`. Historique des décisions = provenance dans aliases.yaml +
git log ; pas de fichier de log séparé.

## Entraînement (datasets v2, modèles RF-DETR segmentation)

- Chaîne par zone : audit (`/audit-dataset`) → GPKG entités (`build_zone_gpkg` +
  `configs/vecteurs_<zone>.yaml`) → découpe (`slice_zone` + `configs/lineaires_<zone>_*.yaml`)
  → **boucle de vérification** (`verif_zone_gpkg`, `verif_dataset`) → carte de contrôle
  validée par l'utilisateur → dépôt Drive → lot de TEST Roboflow (10 img) → inspection
  humaine des masques → upload complet vérifié. Skills : `/prepare-zone-training`,
  `/upload-roboflow`.
- **Jamais de split aléatoire** (cf. docs/fuite_spatiale_train_test.html) : split spatial
  par blocs de 2 km, tracé dans `split_manifest.yaml`, imposé à l'upload et jamais re-tiré.
  Tuiles **648 px sans chevauchement** (RF-DETR seg : résolution divisible par 24 ;
  pin `rfdetr>=1.8.3,<2.0` ; résolution d'entraînement = résolution d'export ONNX).
- **Boucle de vérification systématique** (règle utilisateur 2026-07-27) : produire →
  vérifier les FICHIERS produits par contrôleur indépendant → corriger → REproduire →
  re-vérifier. Aucune livraison (Drive/Roboflow) avant verdict conforme.
- **Éval outillée standard** (règle utilisateur 2026-08-17, révisée 2026-08-31) : TOUT
  nouveau modèle (détection bbox OU segmentation) passe par `tools/courbes_eval.py`,
  superposé au modèle précédent/baseline sur la MÊME éval gelée (appariement IoU
  masque/bbox ≥ 0,5, balayage de seuil — jamais de seuil fixe). Sortie canonique
  `metriques_eval.json` = SOURCE UNIQUE des seuils du model_card
  (`thresholds.confidence_default` + `confidence_per_class` = F1-max mesurés, champ
  `seuils_provenance`). Rangement : Drive `runs/training/<run>/evaluation/`
  (metriques_eval.json + planches + appariements.json) ; plugin
  `data/models/<modele>/entrainement/evaluation/` (éval du modèle déployé) et
  `entrainement/comparaison_<vs>/` (superpositions). `evaluation_results.json` racine
  = legacy notebook (appariement incompatible, seuil fixe 0,3) : documentaire, JAMAIS
  réécrit ni source de seuils. Après tout dépôt d'évaluation : régénérer le dashboard
  (`tools/tableau_modeles.py` → index.html racine model-training). **La traçabilité
  du run vit dans `entrainement/`** : `metrics.csv` (+ historiques si reprises +
  NOTE-metriques.md), `hparams.yaml`, tfevents, `visualizations/` — le contrat reste
  à la racine. Cf. skill `/installer-modele-plugin` pour la checklist complète
  (ONNX, parité binarisée, sidecar class_offset, entité catalogue).
- **Rangement model-training** (décision 2026-08-31) : UNE famille par thème à la
  racine du Drive `model-training/` (nom snake_case ASCII sans accents ni tirets —
  vérifier l'existant AVANT de créer une famille ou un run ; c'est ainsi que
  `fours_charbonnieres` et `fours-à-chaux_charbonnieres` ont coexisté) ; les
  générations obsolètes descendent dans `legacy_<annee>/` (les 3 familles détection
  2025 sont dans `legacy_2025/`, cf. son LISEZ-MOI — le run_rf_detr_1 du plugin en
  vient). Un lancement raté (dossier de run horodaté sans poids ni metrics, + son
  miroir runs/inference) se supprime IMMÉDIATEMENT, avant de relancer.
- **Chantier enclos** (2026-08) : entité plugin `enclos_circulaire` (le derived_target
  `enclos` du modèle formes_lineaires est COMMENTÉ, réactivable). Modèles :
  `enclos_ie_seg_ld_v1` et `enclos_fr_seg_ld_v1` (renommés le 2026-08-20, ex *_seg_v1 —
  corpus irlandais 1 089 emprises, test mAP@50 0,682, seuil 0,375 ; sélections UI QGIS à
  re-choisir). **Pas de Roboflow pour les enclos** : corpus COCO sur le
  Drive uniquement (`model-training/enclos/`, corpus enclos_ie_648_v1 / enclos_fr_648_v1 /
  enclos_frie_graduel_648_v1). **Éval française GELÉE** : `eval_fr_gelee_v1.yaml`
  (listes d'images nominatives) — jamais re-tirée, extensions de GT en train seulement.
  GSD enclos = 1 m (LD Rmin5/Rmax10) ; ringfort/enclosure fusionnés à l'entraînement.
- **Chantier multicanal enclos** (2026-08-20) : 4 corpus déposés sur le Drive —
  enclos_{ie,fr}_648_{csl,crim}_v1, mêmes tuiles/splits/COCO que les v1 (datasets
  repeints par `repeindre_dataset.py`, canal B byte-identique v1, éval gelée intacte).
  CSL = R:cvat_combined (vrai VAT combined) / G:slrm r10 ±0,5 (standard ADAF) / B:LD ;
  CRIM = RGB crim_orrd. 4 runs Colab à lancer (IE puis transfert FR par variante,
  hyperparamètres = runs v1) : cf. `docs/google_collab/NOTES-runs-multicanal-enclos.md` ;
  comparatif final 6 modèles par courbes_eval sur l'éval FR gelée, 3 passes (une par
  représentation). Le « cvat » de generer_slrm_cvat.py est un VAT general mal nommé
  (SVF 0,7965 non standard) — le vrai CVAT est dans planche_indices/repeindre_dataset.
- **Notebook d'entraînement canonique** (règle 2026-08-20) :
  `G:\Mon Drive\Colab Notebooks\rfdetr_unified_pipeline_v2.ipynb` — c'est LUI qu'on
  paramètre à chaque entraînement (copie locale docs/google_collab/ → édition cellule 2
  → redépôt G:). Guide complet section par section (défauts rfdetr 1.8.3, checkpoints,
  pièges de reprise, sources) : `docs/google_collab/GUIDE-parametrage-rfdetr_unified_pipeline_v2.md`.
- **Un script scratchpad réutilisé deux sessions de suite doit être promu dans
  `tools/`** (leçon : points_a_recaler perdu dans une purge de scratchpad et réécrit
  deux fois avant promotion).
- Talus/fossés : 3 entités — `talus` et `fosse` (sources qui distinguent), `talus_fosse`
  (labels indistincts, ex. fossébutte Haye). Classes plateforme suffixées `<entite>_<site>` ;
  buffer de lignes standard pour tout NOUVEAU dataset : **7 m de largeur totale**
  (décision 2026-07-28 — absorbe les offsets de digitalisation ; datasets existants :
  Haye/Fontainebleau 4,8 m, Rambouillet/Saint-Germain/Blois 5 m, à régénérer au
  recalage) ; couches linéaires non entraînées en `ignorer:` (bloquent les négatifs
  sans annoter).
- **Recalage des vecteurs** (2026-07-28, spec `docs/superpowers/specs/2026-07-28-recalage-vecteurs-design.md`,
  skill `/recalage-zone`) : lignes recalées sur le LD (méthode B — profils perpendiculaires,
  pénalité de distance, couloir partagé entre voisines) avec revue humaine exhaustive dans
  `tools/review_recalage` et **calibration de l'algo par les corrections humaines**
  (géométries éditées = vérité terrain, non-régression sur les acceptées). Polarités :
  talus/parcellaire clair, fossé/chemin creux sombre, `talus_fosse` fusionné = sombre imposé.
  Seuils a_revoir surchargeables par zone (`seuils_statut:`, justifiés par mesure). GPKG
  final v2 déposé avec `geom_origine` + `decision_humaine` ; **pas de re-upload Roboflow
  avant que toutes les zones soient recalées** (remplacement groupé, re-slice 7 m d'abord).
- Roboflow : la **source de vérité est locale** (split_manifest + upload_manifest) ; la
  plateforme est un miroir de contrôle qualité. Ses pièges connus (dédup corbeille,
  batch_name→file Annotate, annotation null = VOC vide, champ `labels` toujours vide,
  search plafonné à 250 à l'ordre instable, compteur de classes en cache) sont parés dans
  `upload_roboflow_split.py` — lire la mémoire persistante avant d'y toucher.

## Rasters externes (GSI, IGN WMS, .asc…) — pièges mesurés

Cinq variantes de livraison cassées rencontrées en une semaine (2026-08) ; règles :

- **Estampiller/harmoniser AVANT tout VRT/mosaïque** : CRS absent (GSI Kerry),
  LOCAL_CS (GSI phase2), WKT custom étiqueté "EPSG:2154" (WMS IGN) → estampiller le
  vrai code EPSG (rasterio `r+`). Dtype discordant (WMS IGN Float32 vs Drive Float64)
  → convertir (`gdal_translate -ot`).
- **gdalbuildvrt SAUTE les dalles incompatibles EN SILENCE** (projection, dtype) et
  sort quand même un VRT « valide » : toujours `grep -c Skipping` sur ses logs, et
  **vérifier les fenêtres produites en PLEINE résolution** (compter les pixels
  valides) — un verdict CONFORME global peut mentir (médiane échantillonnée au 1/8).
- GDAL (OSGeo4W) ne lit PAS les chemins git-bash `/g/...` : toujours `G:/...`.
- **Slashs avant dans tout code/config généré** (le `\v` de `D:\veille_irlande`
  devient une tabulation verticale — déjà mordu trois fois) ; valider le YAML/JSON
  généré AVANT écriture (octets de contrôle compris).
- Archives : zips GSI à 4 structures + .7z (py7zr) ; détecter par magic bytes, pas
  par extension. WMS IGN : GeoTIFF exact par dalle via la grille WFS
  `IGNF_MNT-LIDAR-HD:dalle` (data.geopf.fr).
- Montages fragiles : **GoogleDriveFS meurt en cours de session** (3 fois en un
  jour, y compris en plein gdalwarp) — vérifier `Get-PSDrive` avant toute opération
  G:/D:, relancer via `Start-Process GoogleDriveFS.exe`, attendre le remontage ;
  robocopy exit 1 = succès (piège connu).

## Stockage Drive — data_regions_v2

Racine : `G:\Mon Drive\Archeologia\Archeologia_Shared\data\data_regions_v2`.
L'ancien `data_regions` (même parent) est **GELÉ** : lecture seule, ne jamais y écrire.
Rapport d'audit fondateur : https://claude.ai/code/artifact/c1ccfd2e-37b0-4bfc-bf8e-7c853b3bfb03

- Par zone `<region>/<dept>_<site>[_<annee>]` (snake_case ASCII) : `manifest.yaml` +
  `raw/` (**sources** — données livrées par l'archéologue ET données de base produites par
  nous : vecteurs labels + `raw/docs/` (rapports/thèses/DRAC) + `raw/MNT/` = MNT de base
  **et** indices de visualisation (SVF/HS/LD/VAT…) à plat) + `training/vecteurs/` (GPKG
  nettoyés, EPSG:2154) + `training/roboflow/` (payloads d'upload + `upload_manifest.yaml`).
  `raw/` n'est plus « immuable » au sens strict (peut contenir des rasters régénérés) mais la
  règle « ne pas éditer en place sur G:, redéposer » tient. Convention : **le MNT de base et
  ses indices vont TOUJOURS dans `raw/MNT/`** (même endroit dans chaque zone). Les dalles
  LiDAR IGN source (`.laz`, ~100-140 Go/zone, re-téléchargeables) ne sont **pas** en cloud
  (pointeur manifest) ; les prédictions/détections du modèle restent **hors** `data_regions_v2`
  (pipeline_results local — risque de contamination si mêlées au raw). `transformed/` renommé
  `training/` le 2026-07-22.
- Tags Roboflow par image : `<zone_id>` + `<region>`. Sous-classes : `<entite>_<site>`
  (ex. charbonniere_vosges). L'export Roboflow ne préserve PAS les tags : les
  `upload_manifest.yaml` locaux font foi.
- `index.html` à la racine v2 = tableau de bord ; le régénérer (`tools/build_v2_index.py`)
  après toute évolution du dossier.
- **Notes typées** dans les `manifest.yaml` (alimentent la section « À traiter » de l'index) :
  `ARBITRAGE:` (bloquant, décision humaine), `TODO:` (action à mener), `ATTENTION:` (alerte
  non bloquante), `DÉCISION <date>:` (actée, informative), sans préfixe = info. Toujours
  écrire les nouvelles notes avec ces préfixes.
- Interdits absolus : supprimer/déplacer sans décision humaine explicite ; éditer un
  fichier en place sur G: (copie locale puis re-dépôt) ; écrire dans data_regions v1.
- Écritures massives vers G: : toujours staging local puis `robocopy /E /MT:16`.
- Workflows : `/dispatch-roboflow` (ranger des exports), `/audit-dataset` (auditer
  une livraison vecteur).
- Arbitrages pendants : aucun. Dept 44 (digitalisation jamais livrée) : relance envoyée par
  mail le 2026-07-16, en attente de re-dépôt. Vosges saônoises : résolu 2026-07-16 —
  Bourgogne-Franche-Comté confirmée par l'utilisateur (Mélisey 70270, Haute-Saône).
- Décisions actées 2026-07-27 (54_foret_de_haye, fusion parcellaire v1/v2) : quand deux
  versions d'une même entité coexistent, **garder toujours le tracé le plus long** ; sortir
  de l'emprise de la version la plus récente n'est pas disqualifiant.
- Décision actée 2026-07-27 : **split talus/fosse** — deux entités distinctes `talus` et
  `fosse` pour toute source qui les distingue (Fontainebleau) ; `talus_fosse` reste réservé
  aux sources fusionnées indissociables (Haye `fossébutte`) et au contrat plugin.
- Résolus le 2026-07-16 (base Notion + disque local) : `data_aude` = prénom du contact de
  la zone Blois (dépt 41 confirmé, cf. manifest sur le Drive) ; bataille de la Marne =
  Hauts-de-France confirmé (Ermenonville, Oise — l'hypothèse 77→IdF de l'audit était
  fausse) ; données Verdun retrouvées dans D:\data_regions\Verdun et restaurées dans v2
  (shapefiles complets + rasters 2013 + couche_regroupee.gpkg).
- Les données personnelles (noms/contacts des archéologues, issues de Notion) vivent dans
  les manifest.yaml du Drive privé — ne JAMAIS les committer dans ce repo public.
- Décisions actées 2026-07-16 : `verdun_3_classes` classe `abri`→`cratere` à la
  recréation ; base de recréation formes linéaires = `formes_lineaires_rmin10`
  (paramétrisation LD actuelle), `rmin20` = archive.

## Langue

Données, labels (`label_fr`) et interaction utilisateur : français. Code, commentaires et
identifiants : le code est en français-technique assumé (docstrings FR), les ids d'entités
sont du français translittéré ASCII (`charbonniere`, `chemin_creux`) — convention de tout
l'écosystème plugin/Roboflow.
