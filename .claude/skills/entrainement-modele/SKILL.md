---
name: entrainement-modele
description: >
  Entraîner ou affiner un modèle RF-DETR-Seg (Colab) : paramétrage du notebook,
  validation des sorties de cellules par l'utilisateur, évaluation, courbes
  standard, seuil optimal, consignation. Utiliser quand l'utilisateur dit
  « on lance un entraînement », « prépare le notebook », « je te donne les
  sorties de cellules », « transfert learning depuis X », « run 2/3 de X ».
argument-hint: <cible, ex. enclos_fr_seg_v2 | corpus à utiliser>
entrees:
  - "corpus CONFORME (verif_corpus) déposé sur model-training/<famille>/ (cf. /corpus-entrainement)"
  - "notebook canonique G:\\Mon Drive\\Colab Notebooks\\rfdetr_unified_pipeline_v2.ipynb"
sorties:
  - "runs/training/<run>/ : checkpoints, metrics.csv, config.json (provenance), params_run.yaml"
  - "runs/training/<run>/evaluation/ : metriques_eval.json (CONFORME verif_courbes_eval) + planches"
suivant: [installer-modele-plugin]
---

# Entraînement d'un modèle (notebook Colab → éval → courbes → consignation)

Interagir en français. Le notebook vivant est `G:\Mon Drive\Colab Notebooks\
rfdetr_unified_pipeline_v2.ipynb` (copie du repo : docs/google_collab/) — le
paramétrer par édition JSON de la CELLULE 2, TOUJOURS vérifier les invariants
par relecture après écriture. Corpus : CONFORME (`verif_corpus`) et déposé sur
`model-training/<famille>/` AVANT tout run. L'utilisateur exécute Colab et
colle les sorties de cellules : les valider une à une.

## Cellule 2 — invariants à poser et relire
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
- **Éval outillée OBLIGATOIRE** : `tools/courbes_eval.py` — depuis le 2026-09-03
  elle tourne DANS COLAB, section 11 du notebook canonique (clone du dépôt
  d'outils `EVAL_TOOLS_REPO` @ `EVAL_TOOLS_REF` : ÉPINGLER un commit poussé —
  un outil modifié localement mais non poussé n'est pas vu par Colab), sur le
  checkpoint LIVRÉ `EVAL_CHECKPOINT`, valid + test du corpus, baselines à
  superposer dans `EVAL_BASELINES` (mêmes tuiles, même machine : jamais mélanger
  un cache Colab et un cache local), fusions dans `EVAL_FUSIONS` ; puis
  `verif_courbes_eval.py` dans la même cellule, verdict CONFORME exigé, puis la
  cellule imprime l'**étude du seuil** (bloc `etude_seuil`). Sur le poste (venv_adaf)
  seulement pour une éval a posteriori ou une superposition hors Colab. Sorties dans
  `runs/training/<run>/evaluation/` du Drive :
  `metriques_eval.json` (CANONIQUE — seuils F1-max global + par classe, bloc
  `etude_seuil`, AP@50, par zone, provenance) + planches + `appariements.json`
  (cache à empreinte).
- **Choix des seuils — cellule 11bis (règle utilisateur 2026-09-09)** : le seuil de
  production se fixe SOUS le F1-max (qui pèse un oubli comme un faux positif), par
  ÉTUDE de la courbe à chaque modèle, dans la fenêtre [bas du plateau F1 ≥ 95 % ;
  F1-max], global ET par classe. Lire le tableau imprimé (F2-max, plateaux, R_max,
  FP/image, précision marginale des ajouts, `par_zone_classe`) : plateau étroit ou
  FP/image lourds → près du F1-max ; plateau large et FP/image faibles → jusqu'au
  F2-max ; classe rare/inutilisable → ne pas descendre ; précision marginale < ~0,15
  → bruit. Renseigner `SEUIL_RETENU`, `SEUILS_PAR_CLASSE_RETENUS`,
  `JUSTIFICATION_SEUILS` (obligatoire) — la cellule refuse hors fenêtre / sans
  justification et écrit `CONFIDENCE_THRESHOLD` + `UI` (`seuils_provenance` = mesure
  + choix). Proposer le choix à l'utilisateur avec les chiffres, jamais appliquer
  `seuil_propose` à l'aveugle. Les seuils du model_card (`confidence_default` +
  `confidence_per_class`) proviennent UNIQUEMENT de cette étude sur
  metriques_eval.json — plus jamais lus sur les PNG, plus jamais « = F1-max » par défaut.
  La même cellule calcule ensuite la **fiabilité affichée** (catégories douteux / possible /
  probable / très probable PAR CLASSE, part de vrais objets mesurée au banc, garantie
  35 / 60 / 85 %, fusion des catégories < 30 détections) → `UI["fiabilite"]` →
  `thresholds.fiabilite` du model_card. Lire les effectifs par catégorie imprimés ; ne
  surcharger (`FIABILITE_RETENUE`) qu'avec `JUSTIFICATION_FIABILITE`.
  Puis, sur le poste, **`tools/verif_courbes_eval.py <dossier_eval>`** (.venv,
  sans GPU) : re-verdict CONFORME AVANT toute lecture de seuils ou dépôt (le
  cache n'a pas à être recalculé). Depuis le 2026-09-03
  courbes_eval publie aussi `par_zone_classe` (zone × classe : n_gt, R/P au seuil
  global, R au seuil de classe, R_max au plancher) et, depuis le 2026-09-09, le bloc
  `etude_seuil` ; une éval antérieure se complète SANS GPU par
  `tools/completer_metriques_eval.py <dossier_eval>` puis re-verif.
- **Dashboard** : régénérer `tools/tableau_modeles.py` → `index.html` racine
  model-training après le dépôt, avec `--plugin <data/models ou zip publié>` pour
  la colonne « installé » ; les fiches par modèle (données par classe, zones,
  zone × classe, run, apprentissage) lisent `manifests/corpus/<corpus>.yaml` du
  dépôt (jointure par CORPUS_DRIVE_DIR de params_run.yaml) : un run sans
  params_run.yaml ni manifeste de corpus affiche « non tracé » — le notebook
  doit donc écrire params_run.yaml et le corpus avoir son manifeste versionné.
- **Section 11 non exécutée dans Colab** (cas 2026-09-08) : éval sur le poste, en
  chaîne (venv_adaf, un seul job CUDA) : (1) éval canonique `runs/training/<run>/
  evaluation/` = nouveau modèle + modèle DÉPLOYÉ de la même famille superposés (même
  tâche) ; (2) concurrents d'une autre tâche (seg vs det) dans `comparaison_<vs>/`
  séparé ; (3) planche 3 modèles depuis les caches (`planche_principale` /
  `planche_classes` de courbes_eval) ; `verif_courbes_eval` après chaque ; puis dépôt
  Drive et dashboard. Chaque `.pth` évalué a son sidecar `best.json` (`class_names`,
  `class_offset` — un vieux export Roboflow prédit parfois la catégorie 0 « entites »).
  Comparer sur les **zones non vues** par chaque modèle (fuite Roboflow).
- « multi-scale training ... scales: [832] » à RESOLUTION 672 = régime upstream rfdetr
  (train redimensionné à la plus grande échelle étendue, val/inférence à RESOLUTION) :
  normal, ne pas arrêter un run pour ça (GUIDE § multi-scale).
- Le `package/` du run doit sortir du notebook (cellule 43) ; s'il manque, l'assembler
  depuis le run + l'évaluation (contrat racine, `entrainement/`, `weights/`) et le
  valider avec `scripts/validate_models_metadata.py` du plugin — outil à écrire au
  prochain cas (fait à la main le 2026-09-08).
- Consigner (tracker de campagne + mémoire) : chiffres, époque best, artefacts.
- **Le local du chantier vit dans `D:\brouillons\<chantier>\`** (CLAUDE.md § Brouillons
  de chantier), avec son `CHANTIER.md` tenu à jour : poids rapatriés, évaluations,
  diagnostics. Rien d'original là-dedans — tout run/éval déposé sur
  `model-training/<famille>/runs/training/<run>/` AVANT de le considérer acquis.
  Clôture (contrôle + proposition de suppression) : `/installer-modele-plugin` § 10.
- **Ce que le plugin lira dans cette éval** (profil des scores, fiches, rapport) :
  `critere`, `etude_seuil.bandes` + `tableau`, `par_zone_classe[zone][classe].{bandes,n_gt}`
  — une éval de couverture (linéaires) doit donc aussi porter `par_zone_classe`, et l'éval
  canonique reste la référence du point d'équilibre P/R (CLAUDE.md § « Ce que le plugin
  lit dans une évaluation »).
- Installation plugin : skill `/archeologia:installer-modele-plugin`, puis fiche de chaque
  classe et cible dérivée : `/archeologia:fiche-classe-plugin` (obligatoire — le modèle
  n'est pas livré tant que le validateur du plugin signale « pas de bloc 'fiche' »).
  Skills partagés du plugin Claude Code `archeologia` (marketplace `../claude-plugins`).

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
