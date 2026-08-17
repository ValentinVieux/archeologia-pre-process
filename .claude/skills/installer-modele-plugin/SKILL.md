---
name: installer-modele-plugin
description: >
  Installer un modèle entraîné dans le plugin QGIS archeologia-pipeline :
  dossier data/models complet (contrat + métriques + courbes), export ONNX,
  porte de parité binarisée, sidecar, entité catalogue, validation. Utiliser
  quand l'utilisateur dit « installe le modèle dans le plugin », « ajoute X au
  plugin », « package et déploie », « exporte en ONNX ».
argument-hint: <id du modèle, ex. enclos_fr_seg_v2>
---

# Installation d'un modèle dans le plugin (checklist complète)

Interagir en français. Le plugin VIVANT est le checkout du profil QGIS
(`%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\archeologia-pipeline`,
branche git propre à vérifier) — `data/models/**` y est gitignoré (déploiement),
mais `data/entities_catalog.json` est SUIVI : le signaler pour le prochain commit.
Gabarit de référence : un modèle déjà conforme (ex. `cratere_circulaire_2` +
compléments 2026-08).

## Le dossier data/models/<id>/ DOIT contenir
- Contrat (À LA RACINE) : `args.yaml`, `classes.txt`, `config.json`,
  `model_card.yaml`, `training_params.json`, `evaluation_results.json` ;
- **Traçabilité entraînement (dans `entrainement/`)** : `metrics.csv`
  (+ historiques numérotés si reprises, avec NOTE-metriques.md disant quel CSV
  = le checkpoint déployé), `hparams.yaml`, tfevents, `visualizations/`
  (training_curves + dashboards du run Drive), et les **courbes standard**
  `comparaison_<vs>/` (tools/courbes_eval.py — planches + appariements.json ;
  convention utilisateur 2026-08-17 : dans entrainement/). Ne JAMAIS écraser
  l'`evaluation_results.json` de référence d'un modèle existant ;
- `weights/` : `best.pth`, `best.onnx`, `best.json`.

## Étapes
1. Copier `package/` du run Drive → `data/models/<id>/` + compléments ci-dessus.
2. Cohérence des métadonnées : `classes.txt` = ids d'ENTITÉ du catalogue ;
   `thresholds.confidence_default` = seuil F1-max MESURÉ (balayage, jamais 0,3
   par défaut) ; version/description/known_limitations remplis ; bloc RVT/MNT
   = GSD et rayons LD du corpus d'entraînement.
3. Export ONNX : `dev/runner_onnx/.venv_onnx` du PROFIL, avec
   `PYTHONIOENCODING=utf-8` (sinon crash charmap en validation) :
   `export_to_onnx.py --model …best.pth --output …best.onnx --type rfdetr
   --imgsz <res> --opset 17`.
4. **Porte de parité** : le validateur peut ÉCHOUER sur l'atol des logits de
   masque (max_diff ~0,06 en bf16) — FAUX POSITIF CONNU si les détections
   (comptes, scores, classes) sont identiques. Contrôle indépendant obligatoire :
   masques BINARISÉS PT vs ONNX (`model.export()` avant forward !), exiger
   IoU 1,0 / identité stricte. Échec réel → ne pas installer.
5. Sidecar `best.json` : `class_offset` correct (rfdetr ≥1.8 → 0 ; vieux exports
   → 1 ; un offset faux SUPPRIME silencieusement des classes), `resolution`,
   `class_names` = classes.txt.
6. Entité(s) au catalogue `data/entities_catalog.json` (id snake_case, label,
   description, morphology, display_order) ; retirer/ne pas laisser d'entité
   orpheline sans modèle (elle s'affiche « Aucun modèle disponible » dans le
   wizard).
7. `scripts/validate_models_metadata.py data/models/<id>` → 1/1 OK exigé.
8. Recharger le plugin dans QGIS (plugin reloader) ; rappeler le commit du
   catalogue à l'utilisateur.

## Garde-fous
- Résolution d'entraînement = résolution d'export = résolution d'inférence.
- Ancien mécanisme remplacé : COMMENTER (model_card/args), jamais supprimer.
- Tout chiffre affiché dans model_card provient d'une mesure tracée
  (evaluation_results.json ou courbes) — pas de valeur héritée d'un autre modèle.
