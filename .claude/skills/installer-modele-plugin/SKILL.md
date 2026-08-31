---
name: installer-modele-plugin
description: >
  Installer un modèle entraîné dans le plugin QGIS archeologia-pipeline :
  dossier data/models complet (contrat + métriques + courbes), export ONNX,
  porte de parité binarisée, sidecar, entité catalogue, validation. Utiliser
  quand l'utilisateur dit « installe le modèle dans le plugin », « ajoute X au
  plugin », « package et déploie », « exporte en ONNX ».
argument-hint: <id du modèle, ex. enclos_fr_seg_v2>
entrees:
  - "runs/training/<run>/package/ + evaluation/metriques_eval.json CONFORME (verif_courbes_eval)"
sorties:
  - "data/models/<id>/ complet (contrat + entrainement/evaluation/ + weights ONNX+sidecar)"
  - "entities_catalog.json à jour (commit) + dashboards régénérés"
suivant: []
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
  (training_curves + dashboards du run Drive), **`evaluation/`**
  (metriques_eval.json + planches + appariements.json du modèle DÉPLOYÉ,
  copiés depuis `runs/training/<run>/evaluation/` du Drive) et les
  superpositions `comparaison_<vs>/` (tools/courbes_eval.py ; convention
  2026-08-17 : toujours dans entrainement/). `evaluation_results.json` racine
  = legacy notebook, documentaire, ne JAMAIS l'écraser ni s'en servir comme
  source de seuils ;
- `weights/` : `best.pth`, `best.onnx`, `best.json`.

## Étapes
1. Copier `package/` du run Drive → `data/models/<id>/` + compléments ci-dessus.
2. Cohérence des métadonnées : lancer `tools/verif_courbes_eval.py` sur
   `entrainement/evaluation/` — CONFORME AVANT de recopier les seuils ;
   `classes.txt` = ids d'ENTITÉ du catalogue ;
   `thresholds.confidence_default` + `confidence_per_class` = valeurs de
   `entrainement/evaluation/metriques_eval.json` (jamais 0,3 par défaut),
   champ `thresholds.seuils_provenance` renseigné (chemin + date) ; vérifier
   clés `par_classe` de metriques_eval.json == classes.txt (piège du mapping
   croisé enclos ie/fr) ; version/description/known_limitations remplis ;
   bloc RVT/MNT = GSD et rayons LD du corpus d'entraînement.
3. Export ONNX : `dev/runner_onnx/.venv_onnx` du PROFIL, avec
   `PYTHONIOENCODING=utf-8` (sinon crash charmap en validation) :
   `export_to_onnx.py --model …best.pth --output …best.onnx --type rfdetr
   --imgsz <res> --opset 17`.
4. **Porte de parité** (BLOQUANTE depuis 2026-08-31) : le verdict de
   `validate_onnx_export` fait échouer l'export ; l'ancien faux positif atol
   bf16 des logits de masque est absorbé par la porte elle-même (sorties
   spatiales jugées sur la DÉCISION : masques binarisés identiques / argmax
   identique — le contrôle manuel « IoU 1,0 binarisé » est désormais intégré).
   Échec réel → ne pas installer.
5. Sidecar `best.json` : `class_offset` correct (rfdetr ≥1.8 → 0 ; vieux exports
   → 1 ; un offset faux SUPPRIME silencieusement des classes), `resolution`,
   `class_names` = classes.txt.
6. Entité(s) au catalogue `data/entities_catalog.json` (id snake_case, label,
   description, morphology, display_order) ; retirer/ne pas laisser d'entité
   orpheline sans modèle (elle s'affiche « Aucun modèle disponible » dans le
   wizard).
7. `scripts/validate_models_metadata.py data/models/<id>` → 1/1 OK exigé
   (validateur v2 : valeurs d'inference_choices, seuils adossés à
   metriques_eval.json, entités ⊆ catalogue, derived_targets ↔ clustering).
8. Recharger le plugin dans QGIS (plugin reloader) ; rappeler le commit du
   catalogue à l'utilisateur.
9. **Régénérer le dashboard** : `tools/tableau_modeles.py` sur la racine
   model-training `--registre <data_regions_v2>\modeles.yaml` (règle CLAUDE.md
   « après tout dépôt d'évaluation ») + resynchroniser le `package/` du run
   Drive avec le model_card final (la seule copie correcte ne doit pas être
   uniquement le data/models gitignoré du laptop).

## Garde-fous
- Résolution d'entraînement = résolution d'export = résolution d'inférence.
- Ancien mécanisme remplacé : COMMENTER (model_card/args), jamais supprimer.
- Tout chiffre affiché dans model_card provient d'une mesure tracée
  (`entrainement/evaluation/metriques_eval.json`) — pas de valeur héritée d'un
  autre modèle, pas de chiffre lu sur un PNG.
