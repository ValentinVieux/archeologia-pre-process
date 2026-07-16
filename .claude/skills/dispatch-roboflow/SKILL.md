---
name: dispatch-roboflow
description: >
  Ranger des exports Roboflow (zips COCO) dans data_regions_v2 sur le Drive :
  attribution des images à leur zone, split des COCO, tags zone+région,
  manifestes, régénération de l'index. Utiliser quand l'utilisateur dit
  « dispatch les datasets », « range les exports roboflow », « j'ai téléchargé
  des datasets roboflow », « nouveaux exports à ranger », ou fournit un dossier
  de zips d'exports Roboflow.
argument-hint: <dossier-des-zips>
---

# Dispatch d'exports Roboflow vers data_regions_v2

Interagir en français. Racine cible :
`G:\Mon Drive\Archeologia\Archeologia_Shared\data\data_regions_v2` (voir CLAUDE.md
§ Stockage Drive pour les règles — notamment : ne JAMAIS écrire dans `data_regions`
v1, ne rien supprimer, raw/ immuable).

## Étape 1 — Inspecter les zips (sans extraire)
Pour chaque zip : splits, classes (COCO `categories`), comptes d'annotations par
classe, motifs de noms de fichiers (`LHD_FXX_XXXX_YYYY` = dalle IGN Lambert-93 en km
→ géo-attribuable ; `tile_XXXX_YYYY` générique → inattribuable par le nom).
Présenter le tableau à l'utilisateur avec les indications qu'il donne.

## Étape 2 — Attribution zone par image
- Datasets mono-zone (ex. verdun_*) : zone forcée, la confirmer avec l'utilisateur.
- Noms LHD : matcher les coordonnées contre l'index des dalles par zone (inventaire
  du Drive) puis contre `EXTENDED_BBOX` de `tools/dispatch_roboflow.py`.
- TOUTE nouvelle règle de résolution (ambiguïté entre zones, extension de bbox,
  nouvelle zone) doit être validée par l'utilisateur (AskUserQuestion), puis
  documentée dans le script ET dans les notes du manifest.yaml de la zone.
- Le reste part en quarantaine `_a_trier/<dataset>/` — jamais de rangement deviné.

## Étape 3 — Dispatch vers un staging local
`.venv\Scripts\python.exe tools\dispatch_roboflow.py <attribution.json> <zips> <staging>`
avec un staging local (ex. `D:\data_regions\_staging_...`) — jamais d'écriture
directe fichier par fichier sur G:. Le script écrit les COCO refiltrés par
zone/split et les `upload_manifest.yaml` (fichier ↔ tags zone+région ↔ méthode
d'attribution).

## Étape 4 — Manifestes de zones
Pour toute zone nouvelle : créer `<region>/<zone_id>/manifest.yaml` (zone_id,
region, departement, source à compléter, roboflow.tags = [zone_id, region], notes).
Pour une zone existante : ajouter les notes utiles (décisions, méthodes
d'attribution) sans réécrire l'existant.
Les notes suivent la convention de préfixes de CLAUDE.md § Stockage Drive :
`ARBITRAGE:` / `TODO:` / `ATTENTION:` / `DÉCISION <date>:` / sans préfixe = info —
elles alimentent la section « À traiter » de l'index.

## Étape 5 — Pousser vers le Drive
`robocopy <staging> <racine v2> /E /MT:16 /R:3 /W:10` (arrière-plan). Archiver les
zips d'origine intacts dans `_archives_roboflow/` s'ils n'y sont pas déjà.
Vérifier ensuite : comptes fichiers/tailles staging vs Drive (écart attendu :
les desktop.ini générés par Drive for Desktop).

## Étape 6 — Régénérer l'index
`.venv\Scripts\python.exe tools\build_v2_index.py "<racine v2>"` puis ouvrir
`index.html`. L'index doit TOUJOURS refléter l'état courant du dossier.

## Étape 7 — Récapituler
Tableau : dataset | zone | images | annotations | méthode d'attribution ;
quarantaine ; décisions prises. Proposer (sans exécuter d'office) le commit des
évolutions d'outillage/skill et la mise à jour de la mémoire de session.

## Garde-fous
- Zéro suppression, zéro modification des zips sources et de data_regions (v1).
- Les résolutions d'ambiguïté sont des décisions utilisateur, jamais des
  suppositions silencieuses.
- Idempotence : le dispatch saute les fichiers déjà présents (le script vérifie
  l'existence) ; relancer ne duplique rien.
- Les tags vivent dans les upload_manifest.yaml locaux (l'export Roboflow ne les
  préserve pas) — ils seront appliqués côté plateforme via l'API lors des sessions
  d'upload/correction.
