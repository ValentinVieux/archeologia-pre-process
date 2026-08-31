---
name: entrainement-modele
description: >
  Entraîner ou affiner un modèle RF-DETR-Seg (Colab) : paramétrage du notebook,
  validation des sorties de cellules par l'utilisateur, évaluation, courbes
  standard, seuil optimal, consignation. Utiliser quand l'utilisateur dit
  « on lance un entraînement », « prépare le notebook », « je te donne les
  sorties de cellules », « transfert learning depuis X », « run 2/3 de X ».
argument-hint: <cible, ex. enclos_fr_seg_v2 | corpus à utiliser>
---

# Entraînement d'un modèle (notebook Colab → éval → courbes → consignation)

Interagir en français. Le notebook vivant est `G:\Mon Drive\Colab Notebooks\
rfdetr_unified_pipeline_v2.ipynb` (copie du repo : docs/google_collab/) — le
paramétrer par édition JSON de la cellule 0, TOUJOURS vérifier les invariants
par relecture après écriture. Corpus : CONFORME (`verif_corpus`) et déposé sur
`model-training/<famille>/` AVANT tout run. L'utilisateur exécute Colab et
colle les sorties de cellules : les valider une à une.

## Cellule 0 — invariants à poser et relire
- `rfdetr==1.8.3` EXACT (1.9 change scheduler + éval segm) ; `RESOLUTION = 648`
  (multiple de 24) passée AU CONSTRUCTEUR ; variante Seg Large ; `NUM_QUERIES=200`
  (= checkpoint pré-entraîné, sinon chargement partiel).
- `AUG_CONFIG_NAME = "AUG_AERIAL"` (VFlip+rot90 — relief nadir).
- **Transfert** : `FINETUNE_FROM = <checkpoint .pth>`, LR ÷10 (1,5e-5 / encodeur
  1e-6), warmup 0,5. **Run frais** : `FINETUNE_FROM=""`, LR 1,5e-4 / 1e-5.
  **Reprise d'un run cassé** = sections 9/9bis/9ter (resume), PAS FINETUNE_FROM.
- Époques : raisonner en PAS (~500-1500 pas d'optimisation) ; batch 8×2=16.
- Contrat plugin : GSD + rayons LD du bloc RVT = ceux du corpus.

## Marqueurs de sortie à valider (l'utilisateur colle, Claude vérifie)
1. Bannière : cible, LR, epochs conformes au paramétrage.
2. Corpus copié : comptes = manifest ; classes = attendues.
3. Cellule 6 : « Fine-tuning depuis : … » si transfert (sinon STOP — chemin non
   résolu) ; tête « Using checkpoint class count » SANS « re-initialized » si
   même nombre de classes ; « résolution effective vérifiée : 648 px ».
4. Cellule 8 : « Augmentation : preset … » ; en transfert, la val d'époque 0-1
   doit être NON triviale (niveau zéro-shot) — 0,003 = tête perdue, STOP.
5. Bénins connus : warnings DINOv2/patch 12, `_kp_active_mask`, flat slice
   (inoffensif si queries=200), HFlip désactivé (keypoints).

## Après le run
- Best = post-hoc sur `ema_segm_mAP_50` depuis metrics.csv (PAS le best
  automatique) ; loss_ce qui monte sous IA-BCE = normal.
- **Éval outillée OBLIGATOIRE** : `tools/courbes_eval.py` (venv_adaf — détection
  bbox ET segmentation, tâche auto-détectée), le nouveau modèle SUPERPOSÉ au
  précédent/à la baseline sur la MÊME éval gelée. Sorties déposées dans
  `runs/training/<run>/evaluation/` du Drive (staging + robocopy) :
  `metriques_eval.json` (CANONIQUE — seuils F1-max global + par classe, AP@50,
  par zone, provenance) + planches + `appariements.json` (cache à empreinte).
  Les seuils du model_card (`confidence_default` + `confidence_per_class`)
  proviennent UNIQUEMENT de metriques_eval.json — plus jamais lus sur les PNG.
- **Dashboard** : régénérer `tools/tableau_modeles.py` → `index.html` racine
  model-training après le dépôt.
- Consigner (tracker de campagne + mémoire) : chiffres, époque best, artefacts.
- Installation plugin : skill `/installer-modele-plugin`.

## Garde-fous
- Éval gelée : listes d'images figées (yaml), JAMAIS re-tirée ; extensions de
  GT → train uniquement. Toute décision (GSD, stratégie, poids de départ) se
  mesure sur elle.
- Un seul facteur change par run comparatif ; verdict = mesure, jamais l'œil.
- CSVLogger écrase metrics.csv à la reprise : sauvegardes numérotées (9ter).
- Lancement raté = supprimer TOUT DE SUITE le dossier horodaté résiduel
  (`runs/training/<cible>_<horodatage>` ET son miroir `runs/inference/`) avant de
  relancer. Une famille = un thème à la racine de model-training (snake_case
  ASCII, vérifier l'existant avant d'en créer une — règle CLAUDE.md 2026-08-31).
