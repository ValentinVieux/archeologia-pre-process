---
name: upload-roboflow
description: >
  Uploader un dataset découpé (slice_zone) vers Roboflow avec split IMPOSÉ :
  lot de test d'abord, validation humaine des masques, puis upload complet
  vérifié par l'API. Utiliser quand l'utilisateur dit « upload le dataset »,
  « envoie sur roboflow », « lance le lot de test », « upload le reste des
  images », « push to roboflow ».
argument-hint: <dossier local du dataset>
---

# Upload Roboflow à split imposé (test → validation humaine → complet)

Interagir en français. Commande exacte dans CLAUDE.md § Commands ; règles § Entraînement.
Les pièges plateforme connus et leurs parades vivent dans l'outil et la mémoire
persistante — les relire avant toute modification de `upload_roboflow_split.py`.

## Étape 1 — Résoudre les entrées
- Dataset local (vérifié conforme par `verif_dataset` — sinon repasser par
  `/prepare-zone-training`). Workspace/projet : défaut `work-o3omo` /
  `archeologia-lineaires-seg-v2` (dossier ArcheologIA) pour le modèle linéaire ;
  confirmer avec l'utilisateur pour tout autre modèle. Suffixe de classes = site
  (`--suffixe-classes <site>`, convention `<entite>_<site>` de CLAUDE.md).
- Clé API : env `ROBOFLOW_API_KEY` (jamais en clair dans le repo public).

## Étape 2 — Lot de TEST
- `--test` (10 images représentatives : classes rares, tuiles riches, négatifs).
  La vérification intégrée doit rendre CONFORME (total zone par tag, images par
  classe, échantillon par image). Échec → corriger, relancer (reprise idempotente
  via upload_manifest_test.yaml).

## Étape 3 — Validation humaine (STOP obligatoire)
- L'utilisateur inspecte les masques sur la plateforme (superposition, splits,
  noms de classes). Rappels UI : la barre de recherche du Dataset est SÉMANTIQUE
  (filtrer par Tags/Classes, pas par nom) ; le batch « Pip Package Upload » de
  l'onglet Annotate est un double affichage cosmétique. Toute rectification
  (buffer, classes…) → retour à `/prepare-zone-training`, purge du lot par ids,
  nouveau test sur tuiles vierges (`--eviter`, dédup corbeille).

## Étape 4 — Upload complet
- Même commande sans `--test` (hérite du lot de test, ne le renvoie pas). Longue
  durée : arrière-plan, reprise idempotente. La vérification finale tolère le
  retard d'ingestion (reprises espacées) ; « absente du dataset » massif juste
  après l'envoi = ingestion en cours, re-vérifier avant de conclure.

## Étape 5 — Comptabilité
- `TODO:` → `DÉCISION:` au manifest de zone (comptes vérifiés) ; déposer
  upload_manifest[_test].yaml dans le dossier dataset du Drive ; régénérer index.

## Garde-fous
- Ne JAMAIS lancer d'entraînement sur un état en divergence.
- Jamais de `--batch` (détourne vers la file Annotate) ; jamais re-uploader un
  contenu supprimé sans parade fantôme (l'outil la gère) ; ne pas supprimer de
  batch depuis l'UI (peut supprimer les images).
- Les manifestes locaux font foi (l'export Roboflow ne préserve ni tags ni splits).
- Relance de la skill = re-vérification seule si tout est déjà tracé (idempotence).
