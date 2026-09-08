# Rétrospective 2026-09-06 → 08 — revue ponctuelles, incident géoréf, modèle 2 classes

## 1. Ce qui a été fait

| Étape | Résultat | Outils / fichiers |
|---|---|---|
| Recherche « mardelle » dans data_regions_v2 | 0 occurrence hors Fenêtrange ; retour de validation FDPE (Georges-Leroy, mars 2026) identifié et exploité (OBS = négatifs typés) | — |
| Revue Chailluz | couche `mardelles` (nettoyage < 40 m ponctuel, 1 012 boîtes revues), polygones SAM 2.1 (marge 0), charbonnières filtrées sur trace LD (200) ; classe `depression_circulaire_grande` (candidate + alias) ; zone dans `revue_zones\traité` | `sam_polygones_bbox.py` + `verif_sam_polygones_bbox.py` (fid_source, motif de repli) |
| Skill `/revue-annotations` | écrite et testée (agent sans skill → 15 manques → 2 tests avec skill → 11 manques) ; section « Soucis rencontrés » évolutive | `.claude/skills/revue-annotations` |
| **Incident géoréférencement** | dalles LD de 2 201 px (km + 50 m) géoréférencées comme 1 000 m : compression 10 %, ±50 m aux bords, annotations comprimées ; haut_doubs 246/264, ales 473/509, la_capelle 77/77 | `docs/incident_georef_dalles_2026-09-06.md` |
| Correction et preuve | `geo_dalle()` (refus des largeurs incompatibles), `corriger_georef_dalles.py`, `verif_alignement_ld.py` (ALIGNÉ / TRANSLATION / ÉCHELLE) ; seg v1 retrouve 94 % des fours sur LD régénéré + annotations corrigées (contre 17 % comprimées, 94 % sur LD livré) : thèse « MNT sol strict » réfutée | — |
| Dépôt Drive des 3 zones | `raw/MNT/{mnt, ld, LD_livre_dalles/ corrigées + vrt, LISEZ-MOI}`, `training/vecteurs/<zone>_entites_l93_v3.gpkg`, manifests notés, index régénéré | robocopy, build_v2_index |
| Blois | modèle enclos_fr sur raster entier (`inferer_raster.py`), 519 dépressions annotées + SAM ; replis injustifiés détectés par l'utilisateur → producteur (test d'aire sur masque découpé) et vérificateur (motifs, quasi-rectangles, taux) renforcés | règle « contrôler avant de livrer » |
| Corpus 2 classes | revue auto des 8 zones (tailles, doublons, hors raster) → `dedup_annotations.py` (doublons INTER-DALLES apparus après correction : −433 fours, −345 charbonnières) ; 8 configs v3 (splits hérités, Chailluz tuiles annotées seulement) ; 8 datasets CONFORMES ; corpus `ponctuelles_2cl_648_v3` CONFORME (5 099 tuiles) déposé + archivé | `configs/`, `manifests/` |
| Run `ponctuelles_2cl_det_ld_v1` | notebook paramétré (patch cellule 2, sorties purgées, sonde 9bis générique) ; 47 époques, best EMA ép. 34 (mAP@50:95 0,459) ; 5 h 25 Colab | GUIDE : régime multi-scale [832] documenté |
| Évaluation | canonique (superposition run_rf_detr_1 @704) + comparaison seg v1 (masques) + planches 3 modèles : **F1 0,641** vs 0,572 (rf_detr_1) vs 0,590 (seg v1) ; sur zones non vues par rf_detr_1 : Blois 0,43 vs 0,02, Chailluz 0,36 vs 0,05 | bug offset de `courbes_eval` corrigé (sidecar prioritaire) |
| Installation plugin | dossier complet, ONNX `--no-simplify` 672, porte de parité ÉTENDUE à la détection (permutation des requêtes), parité de décision 120 tuiles 0 écart, validateur 1/1, registre, dashboard, package/ resynchronisé ; run_rf_detr_1 retiré du plugin (Drive intact), entité orpheline retirée | catalogue + export_to_onnx.py du plugin à committer |

## 2. Ce qui a mal tourné, et pourquoi

1. **Hypothèse non contrôlée** (`px = 1000 / width`) : aucun contrôle croisé avec une référence géoréférencée indépendante ; la carte de contrôle était dessinée sur l'image fautive elle-même, donc toujours « alignée ». Coût : corpus v2 et run det v1 invalides, un faux diagnostic (MNT), une semaine.
2. **Des vérificateurs CONFORMES sur des résultats faux** : `verif_sam_polygones_bbox` ne jugeait que la géométrie (replis injustifiés livrés) ; `courbes_eval` devinait l'offset de classes sur la première image (fours de run_rf_detr_1 effacés) ; la porte de parité ONNX refusait un export correct (permutation des requêtes). Cause commune : le contrôleur mesure autre chose que ce que l'utilisateur regarde.
3. **Effets de bord d'une correction** : les doublons inter-dalles n'existaient pas avant la correction du géoréférencement ; personne ne les cherchait.
4. **Données qui bougent sous un calcul** : couche éditée dans QGIS pendant un run SAM (988 boîtes au lieu de 884).
5. **Chaîne Colab incomplète** : section 11 et packaging non exécutés → évaluation et `data/models` refaits à la main sur le poste.
6. **Outillage de session** : antislashs halvés par l'outil Bash (trois fois), GoogleDriveFS mort (trois fois), `Select-String`/`rg` inutilisables sur G:, sorties cp1252.
7. **Deux jours de travail non commités** (configs, manifests, 8 outils, 2 skills, doc, taxonomie).

## 3. Propositions

### CLAUDE.md

- § Entraînement : ajouter la règle « **un CONFORME ne vaut que pour ce que le contrôleur mesure** : avant toute livraison, lire les distributions et les cas extrêmes du résultat, étendre le vérificateur à ce qui a manqué » (déjà en mémoire feedback, pas encore règle écrite).
- § Entraînement : « **après toute correction géométrique d'annotations, dédoublonner** (`dedup_annotations.py`) et recontrôler les tailles ».
- § Entraînement : « **comparer les modèles sur les zones non vues** par chacun ; un modèle entraîné sur les mêmes dalles (Roboflow, split aléatoire) n'est pas une baseline ».
- § Entraînement : sidecar `best.json` avec `class_offset` obligatoire à côté de tout `.pth` évalué ; `courbes_eval` le lit en priorité.
- § Pièges (nouveau paragraphe « Outil Bash / session ») : antislashs halvés dans les heredocs (écrire les scripts avec Write, jamais `\\` dans un heredoc) ; `PYTHONIOENCODING=utf-8` pour tout script qui imprime des accents ; `Get-PSDrive G` avant chaque accès Drive et relance `launch.bat` ; grep sur G: = inventorier puis grep par liste.
- § Commands : les 8 nouveaux outils sont déjà inscrits ; ajouter le régime multi-scale (« scales [832] = normal ») en une ligne près du notebook.
- § Workflow : **commit à chaque jalon vérifié** (configs + manifests + outils + skills), pas en fin de chantier.

### Skills

- `/prepare-zone-training` : étape « revue automatique des couches » (tailles, allongement, doublons, hors raster, nodata) avant découpe, outil à promouvoir depuis le script de session ; option « zone partiellement revue → `negatifs_pct: 0` » ; `verif_alignement_ld` déjà inscrit.
- `/revue-annotations` : déjà à jour (témoin d'intégrité, motifs, fid_source, traité\) ; ajouter le run SAM incrémental (`--fids-absents-de`) si les vagues d'annotation continuent.
- `/entrainement-modele` : (a) si la section 11 n'a pas tourné, procédure « éval sur le poste » (canonique + superposition du modèle déployé + comparaison seg séparée + planche 3 modèles) ; (b) sidecars et `class_offset` ; (c) note multi-scale ; (d) exiger le `package/` produit par le notebook OU par un outil `preparer_package_plugin.py` à écrire (assemble contrat + entrainement/ + weights depuis un run et son évaluation, valide avec le validateur du plugin) — cette session l'a fait à la main.
- `/installer-modele-plugin` : (a) porte de parité : critère de décision pour la détection (fait dans le plugin) ; (b) **parité de décision sur ≥ 100 tuiles de test** obligatoire, outil à promouvoir depuis le script de session (`verif_parite_onnx.py`) ; (c) mise à jour du registre `modeles.yaml` et retrait des entités orphelines ; (d) piège : le validateur appelé sur `data/models` traite le dossier comme un modèle.
- `/corpus-entrainement` : renvoyer à la revue automatique et au dédoublonnage ; exiger dans la recette une note par zone sur la nature des boîtes (réelles / taille fixe), puisque Rambouillet plafonne à 0,45 pour tous les modèles.
- Nouvelle skill `/corriger-georef-zone` (ou section de prepare-zone-training) : diagnostic (`verif_alignement_ld`) → correction (`corriger_georef_dalles`) → dédoublonnage → re-dépôt raw/MNT + vecteurs v3 + notes → re-découpe. Trois zones l'ont suivie à l'identique.

### Outils à promouvoir (règle des deux usages)

`revue_auto_annotations.py` (revue des 8 zones), `verif_parite_onnx.py` (120 tuiles), `rappel_par_zone.py` (rappel en L93 contre une GT corrigée), `parametrer_notebook.py` (patch de la cellule 2 avec relecture des invariants).

### Mémoire

Le fichier `chantier-fours-v2-detection.md` dépasse la taille utile : scinder en « incident géoréf » (clos), « corpus/run 2 classes » (clos) et « chantier dépressions » (ouvert).
