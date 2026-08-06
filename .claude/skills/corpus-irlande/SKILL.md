---
name: corpus-irlande
description: >
  Avancer le corpus d'annotations polygones ringforts/enclosures sur LiDAR ouvert
  irlandais : téléchargement des dalles du secteur, mosaïque vérifiée, LD + recalage
  humain des points SMR, propositions hybrides cercle+SAM, consolidation versionnée
  et dépôt Drive. Utiliser quand l'utilisateur dit « on avance le corpus irlandais »,
  « on lance le secteur X », « télécharge kerry », « propose les polygones de X »,
  « où en est le corpus », « next Irish sector », ou livre une couche de points recalés.
argument-hint: <secteur, ex. kerry | cork_03 (cf. suivi_corpus.yaml)>
---

# Corpus Irlande (téléchargement → mosaïque → LD + recalage → propositions → dépôt versionné)

Interagir en français. Commandes exactes : CLAUDE.md § Commands. Données, modèles et
tracker : `D:\veille_irlande` (état par secteur dans `suivi_corpus.yaml` — le lire
AVANT toute action, l'écrire après chaque étape). La mémoire persistante documente
le pipeline et ses pièges. UN secteur en revue humaine à la fois ; le téléchargement
du suivant peut tourner pendant. Vérifier le cumul par classe avant d'ouvrir un
nouveau secteur (objectif dans le tracker).

## Étape 1 — Télécharger le secteur
- Dalles du secteur = liste `dalles:` du tracker. **Vérifier l'espace libre D: avant.**
- `telecharger_dalles_gsi` (reprise idempotente, 4 connexions). Purge des zips après
  conversion OK. Zips GSI : structures hétérogènes (arbo ET géoréférencement — Kerry
  livre des GeoTIFF sans CRS, estampillés ITM d'office) ; garde-fous intégrés aux outils.

## Étape 2 — Mosaïque unique + vérification
- `mosaique_mnt` vers `dalles\mnt_<secteur>_mosaique\` — **verdict CONFORME obligatoire**
  avant de donner la main. **JAMAIS de reprojection dalle par dalle, jamais de mosaïque
  géante** : les secteurs épars sont déjà découpés en sous-groupes ≤ 60 dalles au tracker.

## Étape 3 — LD automatique + recalage des points [Vous pour le recalage seul]
- LD généré par `generer_ld` (mosaïque → LD 8 bits) : **rayons dérivés de la
  résolution automatiquement** (anneau métrique constant 5-10 m ; refus si trop
  grossier), étirement fixe 0,5-1,8, auto-vérification **CONFORME obligatoire**.
- Préparer la couche des points SMR du secteur non encore annotés (exclure invisibles
  déjà jugés et polygones existants), triée par score de contraste local sur le LD.
- [Vous] recalage : déplacer chaque point sur sa structure ; **supprimer = invisible**.

## Étape 4 — Propositions hybrides + vérification
- `proposer_polygones_irlande` (⚠ python du venv_sam, pas .venv) sur la couche rendue :
  cercle multi-échelle → boîte → SAM ft4b → post-traitement (trous remplis, lissé).
- **Boucle : `verif_polygones_irlande` jusqu'à CONFORME** avant de livrer à la revue.
- [Vous] revue triée par `a_verifier` puis `accord` croissant ; corrections libres.

## Étape 5 — Consolidation versionnée + dépôt Drive
- Fusionner vérité terrain humaine + propositions acceptées avec champ **`source`**
  (`humain` / `pipeline_vN`) → `<secteur>_annotations_vN.gpkg`. **Jamais d'écrasement :
  chaque extension = nouveau fichier _vN.**
- Dépôt `data_regions_v2\irlande\ie_<secteur>\` (staging local puis robocopy, hash
  vérifié, note DÉCISION au manifest, double attribution CC-BY SMR+GSI) puis
  `build_v2_index`. Mettre à jour cumuls et statuts du tracker.

## Garde-fous
- **Évals gelées 100 % humaines et jamais re-tirées** (`sam_ft/split_irlande.yaml`) :
  une annotation `source: pipeline_*` n'entre JAMAIS dans une éval.
- Tout recalibrage (rayons, seuils cascade, décodeur) se mesure sur les évals gelées
  avant adoption — jamais de réglage à l'œil.
- Points à moins de 2×Rmax du bord de mosaïque : exclus, jamais annotés coupés.
- Licences : double attribution CC-BY-4.0 (NMS/SMR + GSI) propagée dans chaque GPKG.
- Sorties locales et dalles non commitées ; outils, skill et paramètres YAML, si.
- Idempotence : chaque étape relançable — `suivi_corpus.yaml` et les fichiers _vN
  disent ce qui est fait ; relancer un secteur soldé = re-vérification puis arrêt.
