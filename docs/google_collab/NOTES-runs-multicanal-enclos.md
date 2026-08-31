# Runs multicanal enclos — 4 configurations de la cellule 2

**Notebook canonique : `G:\Mon Drive\Colab Notebooks\rfdetr_unified_pipeline_v2.ipynb`**
(règle 2026-08-20 : c'est LUI qu'on paramètre à chaque entraînement — copie locale
dans docs/google_collab/, édition, redépôt sur G: ; jamais d'édition en place).
Guide de paramétrage complet : `GUIDE-parametrage-rfdetr_unified_pipeline_v2.md`.
État : cellule 2 actuellement paramétrée pour le RUN 1 (fait le 2026-08-20).

Décisions 2026-08-20 : 2 représentations testées contre la baseline LD —
CSL (R:cvat_combined / G:slrm_r10±0,5 / B:ld_v1) et CRIM (RGB crim_orrd).
Mêmes tuiles/splits/annotations que les corpus v1 (datasets repeints,
éval FR gelée inchangée). Architecture et hyperparamètres = runs v1 :
RFDETRSegLarge, RESOLUTION 648, NUM_QUERIES 200, batch 8 × accum 2,
weight_decay 1e-4, cosine, seed 42, AUG_AERIAL, bf16-mixed, early stopping
patience 15, rfdetr==1.8.3. Seuls changent : corpus, TARGET_NAME, FINETUNE_FROM,
et pour les runs FR les LR/époques du transfert (comme v1).

Racine Drive : `/content/drive/MyDrive/Archeologia/Archeologia_Shared/model-training/enclos`

## Run 1 — IE CSL (`enclos_ie_seg_csl_v1`)
```
ENTITY_REPERTORY_NAME = '/enclos_ie'
TARGET_NAME  = 'enclos_ie_seg_csl_v1'
CORPUS_DRIVE_DIR = MODEL_TRAINING_PATH + '/enclos_ie/dataset/enclos_ie_648_csl_v1'
FINETUNE_FROM = ''            # départ poids pré-entraînés RF-DETR
NUM_EPOCHS = 100 ; LEARNING_RATE = 1.5e-4 ; LR_ENCODER = 1e-5 ; WARMUP_EPOCHS = 1.0
```

## Run 2 — FR CSL transfert (`enclos_fr_seg_csl_v1`) — APRÈS run 1
```
ENTITY_REPERTORY_NAME = '/enclos_fr'
TARGET_NAME  = 'enclos_fr_seg_csl_v1'
CORPUS_DRIVE_DIR = MODEL_TRAINING_PATH + '/enclos_fr/dataset/enclos_fr_648_csl_v1'
FINETUNE_FROM = MODEL_TRAINING_PATH + '/enclos_ie/runs/training/enclos_ie_seg_csl_v1/checkpoints/checkpoint_best_total.pth'
NUM_EPOCHS = 80 ; LEARNING_RATE = 1.5e-5 ; LR_ENCODER = 1e-6 ; WARMUP_EPOCHS = 0.5
```
(`checkpoint_best_total` = le protocole RÉEL du transfert FR v1, lu dans la cellule 2
du notebook — pas best_ema comme supposé initialement.)

## Run 3 — IE CRIM (`enclos_ie_seg_crim_v1`)
```
ENTITY_REPERTORY_NAME = '/enclos_ie'
TARGET_NAME  = 'enclos_ie_seg_crim_v1'
CORPUS_DRIVE_DIR = MODEL_TRAINING_PATH + '/enclos_ie/dataset/enclos_ie_648_crim_v1'
FINETUNE_FROM = ''
NUM_EPOCHS = 100 ; LEARNING_RATE = 1.5e-4 ; LR_ENCODER = 1e-5 ; WARMUP_EPOCHS = 1.0
```

## Run 4 — FR CRIM transfert (`enclos_fr_seg_crim_v1`) — APRÈS run 3
```
ENTITY_REPERTORY_NAME = '/enclos_fr'
TARGET_NAME  = 'enclos_fr_seg_crim_v1'
CORPUS_DRIVE_DIR = MODEL_TRAINING_PATH + '/enclos_fr/dataset/enclos_fr_648_crim_v1'
FINETUNE_FROM = MODEL_TRAINING_PATH + '/enclos_ie/runs/training/enclos_ie_seg_crim_v1/checkpoints/checkpoint_best_total.pth'
NUM_EPOCHS = 80 ; LEARNING_RATE = 1.5e-5 ; LR_ENCODER = 1e-6 ; WARMUP_EPOCHS = 0.5
```

## Garde-fous
- Vérifier avant chaque run que la cellule de vérification corpus passe
  (comptes + ordre des classes vs corpus_manifest.yaml).
- Piège connu : la cellule 24 `RESUME_RUN_PATH` contient un placeholder périmé —
  ne pas l'exécuter sans la re-renseigner.
- Mapping de classes au packaging : reproduire le mapping des runs v1
  (IE corpus `enclos`, FR corpus `enclos_circulaire`) pour que le comparatif
  `courbes_eval` reste homogène (`--fusion` sinon).
- Comparatif final (6 modèles) : 3 passes de `tools/courbes_eval.py` — une par
  représentation (LD / CSL / CRIM), mêmes tuiles + GT de l'éval FR gelée —
  puis planche de synthèse. Modèles LD renommés : enclos_ie_seg_ld_v1,
  enclos_fr_seg_ld_v1.
