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
.venv\Scripts\python.exe tools\upload_roboflow_split.py <dataset> --workspace <id> --projet <id> [--test] [--dry-run]  # upload split IMPOSÉ (clé : env ROBOFLOW_API_KEY)
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
