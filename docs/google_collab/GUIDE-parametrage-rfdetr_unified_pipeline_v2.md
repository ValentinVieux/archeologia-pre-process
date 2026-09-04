# Guide de paramétrage — `rfdetr_unified_pipeline_v2.ipynb`

**Notebook canonique : `G:\Mon Drive\Colab Notebooks\rfdetr_unified_pipeline_v2.ipynb`**
(règle 2026-08-20). À chaque entraînement : copier G: → `docs/google_collab/` (copie de
travail versionnée), éditer, **vérifier** (`compile()` de la cellule 2), redéposer sur
G:. Jamais d'édition en place sur G:.

Rédigé le 2026-08-20 depuis : le code source rfdetr au tag 1.8.3 (config.py, detr.py,
trainer.py, callbacks/), la doc officielle versionnée, le papier LW-DETR, et la
littérature du finetuning sur petits datasets (sources en fin de chaque section, liste
complète en annexe). Ce guide suit l'ordre des 16 sections du notebook.

---

## 0. Les trois types de run — quoi toucher

Toute la configuration d'un run vit dans la **cellule 2**. Le reste du notebook se
déroule sans édition (sauf reprise, §9).

| Type | FINETUNE_FROM | LR / LR_ENCODER | EPOCHS | WARMUP | Exemple |
|---|---|---|---|---|---|
| **A — nouveau modèle** (départ poids pré-entraînés Roboflow) | `""` | 1,5e-4 / 1e-5 | 100 | 1.0 | runs IE (v1, csl, crim) |
| **B — transfert** (départ d'un run maison) | `.../runs/training/<run_source>/checkpoints/checkpoint_best_total.pth` | **1,5e-5 / 1e-6** (÷10) | 80 | 0.5 | runs FR (transfert IE→FR) |
| **C — reprise** (run interrompu) | inchangé | inchangés | inchangés | — | §9 du notebook, cellules 24-28 |

Le ÷10 du transfert est l'heuristique convergente CS231n / MMDetection / ULMFiT
(*discriminative fine-tuning*) : les poids sources sont déjà proches d'un bon optimum,
un LR fort les détruirait ([CS231n](https://cs231n.github.io/transfer-learning/),
[MMDetection finetune](https://mmdetection.readthedocs.io/en/main/user_guides/finetune.html),
[ULMFiT arXiv:1801.06146](https://arxiv.org/abs/1801.06146)).
`checkpoint_best_total` = « the final model selected for inference » (doc rfdetr) — et
c'est le protocole RÉEL du transfert FR v1 (lu dans la cellule 2 du run v1).

Configurations actées des 4 runs multicanaux : `NOTES-runs-multicanal-enclos.md`.

---

## 1. Cellule 2 — bloc par bloc

### TÂCHE / CHEMINS
- `TASK` : `"instance_segmentation"` pour tous les modèles enclos/linéaires actuels.
- `MODEL_TRAINING_PATH` : famille (`.../model-training/enclos`) ; `ENTITY_REPERTORY_NAME` :
  répertoire d'entité (`/enclos_ie` ou `/enclos_fr`) ; `TARGET_NAME` : **nom du run ET
  du futur dossier modèle plugin** (ex. `enclos_ie_seg_csl_v1`). Si le run existe déjà
  sur le Drive, le notebook suffixe d'un timestamp — nettoyer avant plutôt que de
  laisser faire.

### FINETUNE_FROM (mécanisme `pretrain_weights`)
`RFDETRSegLarge(pretrain_weights=...)` initialise les poids puis repart d'un
entraînement **neuf** : optimiseur, scheduler et numéro d'époque remis à zéro
(≠ `resume`, §9). `num_classes` est inféré des poids puis la tête est réalignée sur le
dataset — pas besoin de le déclarer. Les `.ckpt` Lightning sont acceptés (clés
normalisées). Un `PretrainWeightsCompatibilityWarning` signale un champ cassant
(encoder, dec_layers, patch_size…) ; `resolution` n'en fait pas partie (les embeddings
positionnels sont interpolés bicubiquement).
Source : detr.py @ 1.8.3, [doc advanced](https://rfdetr.roboflow.com/latest/learn/train/advanced/).

### MODÈLE
- `MODEL_VARIANT = "large"` (RFDETRSegLarge) : DINOv2 windowed small + décodeur DETR
  5 couches + tête seg « inspired by MaskDINO » ; héritages LW-DETR : two_stage,
  bbox_reparam, IA-BCE, group_detr 13.
- `RESOLUTION = 648` : famille Seg = patch **12** × num_windows **2** → **multiple de
  24 obligatoire** (le forward lève ValueError sinon ; 648 = 24×27). La résolution
  native Seg-Large est 504 — 648 est valide, le positional encoding est re-synchronisé
  automatiquement (648/12 = 54) : **ne jamais toucher `positional_encoding_size` à la
  main**. Règle contrat : résolution d'entraînement = résolution d'export ONNX =
  résolution d'inférence plugin.
- `NUM_QUERIES = 200` : défaut du checkpoint pré-entraîné — le changer provoque un
  chargement partiel des poids. 200 = plafond de détections/image (cf. diagnostic
  cratères Verdun : c'est LA limite dure en zones denses ; sans objet pour les enclos,
  max ~8/tuile).
Sources : config.py @ 1.8.3, CHANGELOG 1.7.0,
[blog seg preview](https://blog.roboflow.com/rf-detr-segmentation-preview/).

### HYPERPARAMÈTRES — valeurs, défauts rfdetr, justification

| Paramètre | Nous (A / B) | Défaut 1.8.3 | Justification sourcée |
|---|---|---|---|
| NUM_EPOCHS | 100 / 80 | 100 | Petits datasets : l'early stopping fait le vrai travail ; DETR-like finetuning converge en 10-30 époques sur ~550 img ([DeepWiki DETR finetune](https://deepwiki.com/facebookresearch/detr/5.3-fine-tuning-guide)) |
| BATCH_SIZE × GRAD_ACCUM | 8×2 | 4×4 | Règle upstream : batch effectif = **16**. Équivalence exacte de l'accumulation (architecture LayerNorm, pas de BatchNorm) ; A100 : 16×1 possible, T4 : 4×4 |
| LEARNING_RATE | 1,5e-4 / 1,5e-5 | 1e-4 | Régime DETR (transformer 1e-4, [Carion 2020](https://arxiv.org/abs/2005.12872)) ; ÷10 au transfert (§0) |
| LR_ENCODER | 1e-5 / 1e-6 | **1,5e-4** | Écart assumé au défaut : le backbone DINOv2 auto-supervisé est ce qui rend RF-DETR « highly fine-tunable » — LR fort sur 164-549 img = catastrophic forgetting. Facteur 10-15 conforme DETR (backbone 1e-5 vs tête 1e-4). LW-DETR fait l'inverse mais son backbone n'est pas DINOv2 : la règle transférable est « LR ∝ distance entre pré-entraînement de la couche et la tâche » |
| WEIGHT_DECAY | 1e-4 | 1e-4 | AdamW standard DETR/LW-DETR (biais/norm/pos_embed exclus par param_groups) |
| LR_SCHEDULER | cosine | step | Cosine ([SGDR](https://arxiv.org/abs/1608.03983)) > paliers pour la généralisation ([Bag of Tricks](https://arxiv.org/abs/1812.01187)) ; LR_DROP devient inerte (ne sert qu'à step) |
| WARMUP_EPOCHS | 1.0 / 0.5 | 0.0 | Warmup court : ne couvre que la phase de gradients erratiques de la tête neuve ([Goyal 2017](https://arxiv.org/abs/1706.02677)) ; MMDetection finetune = 500 itérations seulement. En transfert (poids déjà bons) : encore plus court |
| USE_EMA | True | True | Évaluer/déployer les poids EMA : généralisation et calibration meilleures ([Mean Teacher](https://arxiv.org/abs/1703.01780), [arXiv:2411.18704](https://arxiv.org/abs/2411.18704)). **Ne pas monter le decay à 0,9997** (LW-DETR/COCO) : fenêtre ~3 300 steps > nos runs entiers — le défaut rfdetr **0,993** (fenêtre ~140 steps, warmup tau=100) est calibré pour nos volumes |
| EARLY_STOPPING | True, patience 15, min_delta 1e-4, use_ema True | False (10/0.001) | Époques courtes (~35 pas IE, ~10 FR) : patience élargie ; suivi sur la métrique EMA (celle qu'on déploie) |
| FREEZE_AT | 0 | — | Pas de gel dur : le LR encodeur quasi nul en tient lieu (discriminative finetuning, plus souple qu'un freeze) |
| GRADIENT_CHECKPOINTING | True | False | −30-40 % de VRAM contre ~+20-30 % de temps ([Chen 2016](https://arxiv.org/abs/1604.06174)) — nécessaire à 648 px batch 8 sur A100 40 Go |
| PRECISION | bf16-mixed | amp auto | bf16 = plage dynamique de fp32, zéro loss scaling, zéro retuning ([Kalamkar 2019](https://arxiv.org/abs/1905.12322)) ; T4/V100 : repasser à 16-mixed |
| SEED | 42 | None | Comparabilité des bras — ne jamais changer entre bras d'une même expérience |
| AUG_CONFIG_NAME | AUG_AERIAL | HFlip seul | Flips H/V p=0.5 + Rotate 90° p=0.5 + brightness/contrast p=0.4. En vue nadir il n'y a pas de « haut » : le groupe D4 est gratuit et c'est l'augmentation au meilleur rendement mesuré en archéo-DL — **+0,11 mAP@50** chez [Guyot 2021](https://journal.caa-international.org/articles/10.5334/jcaa.64) (150 img, Bretagne, avec MOINS que ça). ⚠️ Nuance multicanal : LD/SLRM/SVF/O± sont isotropes, mais le **CVAT embarque un ombrage 315°** — une rotation crée des vues « éclairées du sud ». Assumé (l'augmentation s'applique à l'identique dans tous les bras, la comparaison reste équitable) ; à réexaminer si le bras CSL sous-performe étrangement |
| CONFIDENCE_THRESHOLD | 0.3 | — | Provisoire : écrasé en section 11 par le seuil F1-max mesuré par courbes_eval (règle projet) |
| EVAL_TOOLS_REPO / EVAL_TOOLS_REF | dépôt training-models / dev | — | Section 11 : dépôt d'outils cloné dans Colab. ÉPINGLER un commit POUSSÉ pour un run à reproduire (Colab ne voit pas le dépôt local) |
| EVAL_CHECKPOINT | checkpoint_best_ema.pth | — | Le checkpoint LIVRÉ (= package/weights/best.pth) — l'éval mesure ce qui part, rien d'autre |
| EVAL_BASELINES / EVAL_FUSIONS | [] / [] | — | Modèles à superposer sur la même éval (`nom=poids@res`) ; fusions de classes à l'éval. Jamais mélanger un cache Colab et un cache local |
| EVAL_PLANCHER | 0.05 | — | Plancher du balayage de seuil (standard des évals déposées) |

### Blocs contractuels (INFERENCE / MNT / RVT / UI / MODEL_CARD_META)
Pilotent le packaging (cellule 43) → model_card.yaml/args.yaml du plugin :
- `INFERENCE` : imgsz = RESOLUTION, SAHI = RESOLUTION overlap 0.2 (toute divergence
  doit être documentée dans `inference_choices` — le packaging l'exige) ;
- `RVT` : **documente la représentation d'entrée du modèle**. LD pour les modèles v1 ;
  pour les multicanaux : `type: CSL` (R:cvat_combined G:slrm_r10±0,5 B:ld) ou
  `type: CRIM` — et `known_limitations` doit porter « composition des canaux absente
  du plugin » tant que le chantier aval n'est pas fait ;
- `MODEL_CARD_META` : id = TARGET_NAME, display_name unique et parlant, status `beta`
  jusqu'à validation.

### CORPUS
`CORPUS_DRIVE_DIR` = le corpus construit et vérifié par pre-process-data
(`build_corpus` + `verif_corpus`, JAMAIS un dataset non vérifié) ;
`CORPUS_LOCAL_DIR` = copie disque local Colab — l'entraînement ne lit **jamais** le
montage Drive (I/O catastrophiques).

---

## 2. Sections 1-2 (cellules 4-10) — environnement

- HF_TOKEN depuis les secrets Colab ; montage Drive.
- **`rfdetr==1.8.3` épinglé strict** : 1.9.0 supprime `lr_drop`/`lr_min_factor` de
  TrainConfig (le config.json sauvé ne serait plus rejouable) ; la cellule garde-fou
  vérifie ≥1.8.3 <2.0. À savoir : **1.8.3 corrige `cls_loss_coef` seg 5.0→1.0** — des
  runs 1.8.2 et 1.8.3 ne sont pas comparables sur la loss de classification (CHANGELOG).
  Depuis 1.8.0, tout kwarg inconnu de `train()` lève une erreur (plus de faute de
  frappe silencieuse).

## 3. Section 3 (cellule 12) — dataset

Copie Drive→local puis **boucle de vérification embarquée** : comptes par split ≡
corpus_manifest.yaml, ordre des classes ≡ manifest. Si l'assert claque : la copie est
incomplète (GoogleDriveFS) → relancer la cellule (la copie reprend), ne PAS contourner.

## 4. Sections 4-7 (cellules 14-20) — classes, modèle, config

- Cellule 14 : mapping auto des catégories COCO (IGNORE_* vides pour les corpus
  canoniques). Vérifier l'affichage : 1 classe attendue (`enclos` IE / 
  `enclos_circulaire` FR — le renommage éventuel se fait au packaging, reproduire le
  mapping des runs v1 pour que courbes_eval reste homogène).
- Cellule 18 : création du modèle. **`resolution=RESOLUTION` DOIT être au
  constructeur** — sans lui le modèle se construit à 504 (mesuré) pendant que le print
  affiche 648. Vérifier la ligne « RF-DETR Seg LARGE @ 648px | queries=200 ».
- Cellule 20 : config.json initial (rejouabilité du run).

## 5. Section 8 (cellule 22) — entraînement

- Patches seg (idempotents) : retry sur annotations malformées ; vérification
  d'intégrité des images ; purge `output/` et `lightning_logs/` du run précédent.
- `resolution` n'est PAS un kwarg de `train()` (champ de ModelConfig) — le notebook le
  sait déjà.
- Sorties pendant le run : surveiller `val/ema_segm_mAP_50_95` (la métrique de
  l'early stopping avec use_ema) et la loss — un plateau precoce à ~0 de mAP signale
  un problème de masques/classes, pas de patience.

## 6. Les checkpoints (rfdetr 1.8.3) — sémantique exacte

Tout est écrit **à la racine du dossier de run** (piège connu — pas de version_0/) :

| Fichier | Contenu | Usage |
|---|---|---|
| `last.ckpt` | Lightning COMPLET (poids+optimiseur+scheduler+epoch), écrasé chaque époque | **reprise** uniquement |
| `checkpoint_<N>.ckpt` | archives complètes toutes les CHECKPOINT_INTERVAL époques | secours |
| `checkpoint_best_regular.pth` | léger, meilleurs poids bruts (val/segm_mAP_50_95) | — |
| `checkpoint_best_ema.pth` | léger, meilleurs poids EMA (val/ema_segm_mAP_50_95) | forcer l'EMA |
| `checkpoint_best_total.pth` | écrit en fin de fit : le GAGNANT (brut vs EMA) | **inférence, packaging, FINETUNE_FROM des transferts** |
| `last_ema.pth` | derniers poids EMA | — |

## 7. Section 9 (cellules 24-28) — reprise : les 3 pièges 1.8.3

1. **Reprendre UNIQUEMENT depuis `last.ckpt`** (`train(resume=...)`) : un `.pth` léger
   remet les callbacks best/early-stopping à zéro (restauration d'état = 1.9.2+).
2. **`metrics.csv` est TRONQUÉ à la reprise** (correctif 1.9.2 non rétroporté) —
   sauvegarder `metrics.csv` AVANT toute reprise (convention projet : historiques +
   NOTE-metriques.md dans entrainement/).
3. `RESUME_RUN_PATH` (cellule 24) traîne souvent un placeholder périmé — le
   re-renseigner ; exécuter la **sonde 9bis** avant d'armer ; la cellule 28 restaure la
   patience depuis config.json (une édition en cellule 2 serait écrasée).
   Issues amont : [#1028](https://github.com/roboflow/rf-detr/issues/1028), #460, #998.

## 8. Sections 10-15 (cellules 30-40) — chargement, éval, visualisation

- Cellule 30 : auto-détection du checkpoint (ordre : best_total > best_ema >
  best_regular) par le helper `_ckpt_le_plus_recent` (2026-08-31) : à nom égal,
  le fichier le plus RÉCENT gagne (racine vs output/ vs checkpoints/) — la copie
  de `checkpoints/` faite par le finally d'entraînement est PÉRIMÉE après une
  reprise (piège mesuré sur fours : ép. 22 vs 41), le helper l'écarte en le disant.
- Cellules 31-32 (depuis le 2026-09-03) : **éval outillée** — clone du dépôt d'outils
  (`EVAL_TOOLS_REPO` @ `EVAL_TOOLS_REF`), `tools/courbes_eval.py` sur valid + test avec
  le checkpoint livré, baselines superposées, puis `tools/verif_courbes_eval.py`
  (CONFORME exigé, sinon arrêt). Sorties dans `evaluation/` du run (metriques_eval.json
  canonique, planches, appariements.json, provenance_outils.txt). Les seuils F1-max
  écrasent `CONFIDENCE_THRESHOLD` et remplissent `UI` (confidence_per_class,
  seuils_provenance) pour le packaging. `evaluation_results.json` n'est plus produit
  (l'ancienne éval mesurait à seuil fixe 0,3 et tronquait les mAP). Cellule 34 :
  visualisation des logs d'entraînement seulement. Validation humaine des sorties de
  cellules = le protocole `/entrainement-modele`.
- Cellules 36-40 : visualisations des prédictions (test) — l'inspection humaine des
  masques fait partie de la boucle.

## 9. Sections 16 (cellules 43-44) — packaging et ONNX

- Cellule 43 : produit `package/` conforme à docs/model_contract.md (args.yaml,
  model_card.yaml, classes.txt, training_params.json…). Les divergences
  imgsz/SAHI/résolution non documentées sont auto-signalées « à compléter » — les
  compléter, pas les effacer. **Le poids packagé = `checkpoint_best_total.pth` en
  tête** (même ordre que l'éval, via `_ckpt_le_plus_recent`) — bug historique
  corrigé 2026-08-31 : l'ancien ordre prenait best_EMA alors que l'éval lisait
  best_total (deux modèles enclos livrés sur des poids non évalués par le
  notebook). Le sha256 du poids copié est tracé dans `config.json > weights`.
- Cellule 44 : export ONNX **après** copie de package/ vers data/models/. `export()`
  hérite de la résolution du modèle (entrée statique 648) — ne pas passer de `shape`.
  Opset 17. Puis checklist `/installer-modele-plugin` (porte de parité binarisée,
  sidecar class_offset, entité catalogue) — et pour les modèles multicanaux : **ne pas
  installer avant le chantier de composition des canaux à l'inférence**.
- Après tout run : l'éval outillée a tourné en section 11 (garde-fou : la cellule 43
  refuse de packager sans `UI["seuils_provenance"]`) ; sur le poste, re-verdict
  `tools/verif_courbes_eval.py` (sans GPU), dashboard `tools/tableau_modeles.py`,
  traçabilité dans `entrainement/`.

## 10. Reproductibilité (retouches 2026-08-31)

- **`params_run.yaml`** (cellule 20, écrit AVANT le fit) : archive automatique de
  toutes les MAJUSCULES de la cellule 2 — la cellule 2 est un slot mutable écrasé à
  chaque run, ce fichier fige le paramétrage de CE run et survit aux runs interrompus.
- **`config.json > provenance`** : rfdetr_version, precision, devices, finetune_from,
  base_weights(+sha256), aug, seed, date — tout ce que `training_config.json` (écrit
  seulement en fin de fit) ne dit pas.
- **Seed effectif** : `pl.seed_everything(SEED, workers=True)` + `seed=SEED` dans
  train_kwargs (avant : `training_config.json` disait `seed: null` pendant que
  `config.json` affirmait 42). Le régime multi_scale reste stochastique par batch —
  reproductibilité statistique, pas bit à bit.
- **Poids de base épinglés** : `BASE_WEIGHTS_DRIVE/_SHA256` (cellule 2) → copie Drive
  `model-training/_poids_base/` contrôlée par sha256 en cellule 18. Si le sha est vide,
  le run l'imprime pour l'épingler. Sans ça, un run TYPE A dépend d'une release GitHub
  Roboflow qui peut disparaître.
- Secrets Colab (`HF_TOKEN`, `ROBOFLOW_API_KEY`) : absents = warning, plus de crash
  (aucune famille active n'utilise Roboflow).

---

## Annexe — sources

**Amont RF-DETR/LW-DETR** : [code rfdetr tag 1.8.3](https://github.com/roboflow/rf-detr/tree/1.8.3)
(config.py, detr.py, trainer.py, callbacks/), [CHANGELOG](https://github.com/roboflow/rf-detr/blob/develop/CHANGELOG.md),
[doc training-parameters 1.8.3](https://rfdetr.roboflow.com/1.8.3/learn/train/training-parameters/),
[doc advanced](https://rfdetr.roboflow.com/latest/learn/train/advanced/),
[LW-DETR arXiv:2406.03459](https://arxiv.org/abs/2406.03459),
[blog RF-DETR](https://blog.roboflow.com/rf-detr/), [blog Seg](https://blog.roboflow.com/rf-detr-segmentation-preview/).
**Finetuning/HP** : [DETR arXiv:2005.12872](https://arxiv.org/abs/2005.12872),
[CS231n Transfer Learning](https://cs231n.github.io/transfer-learning/),
[MMDetection finetune](https://mmdetection.readthedocs.io/en/main/user_guides/finetune.html),
[ULMFiT arXiv:1801.06146](https://arxiv.org/abs/1801.06146),
[Goyal arXiv:1706.02677](https://arxiv.org/abs/1706.02677),
[SGDR arXiv:1608.03983](https://arxiv.org/abs/1608.03983),
[Bag of Tricks arXiv:1812.01187](https://arxiv.org/abs/1812.01187),
[Mean Teacher arXiv:1703.01780](https://arxiv.org/abs/1703.01780),
[EMA dynamics arXiv:2411.18704](https://arxiv.org/abs/2411.18704),
[Model soups arXiv:2203.05482](https://arxiv.org/abs/2203.05482),
[bfloat16 arXiv:1905.12322](https://arxiv.org/abs/1905.12322),
[gradient checkpointing arXiv:1604.06174](https://arxiv.org/abs/1604.06174).
**Archéo-DL** : [Guyot 2021 JCAA](https://journal.caa-international.org/articles/10.5334/jcaa.64)
(augmentation +0,11 mAP, Bretagne), [Somrak 2020 RS 12:2215](https://www.mdpi.com/2072-4292/12/14/2215)
(pseudo-RGB de visualisations ALS sur backbone pré-entraîné),
[Verschoof-van der Vaart 2019](https://journal.caa-international.org/articles/10.5334/jcaa.32),
[Fiorucci 2022 RS 14:1694](https://www.mdpi.com/2072-4292/14/7/1694),
[arXiv:2307.03512](https://arxiv.org/abs/2307.03512) (le transfert aide souvent, pas
systématiquement — d'où les bras témoins),
[Yu 2023 RS 15:827](https://www.mdpi.com/2072-4292/15/3/827) (augmentations télédétection).
