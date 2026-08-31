---
name: corpus-entrainement
description: >
  Assembler un corpus d'entraînement multi-zones (build_corpus + verif_corpus +
  dépôt Drive) depuis des datasets découpés CONFORMES : recette YAML commitée,
  boucle de vérification, manifeste enrichi, dépôt model-training. Utiliser quand
  l'utilisateur dit « assemble le corpus X », « construis le corpus », « ajoute la
  zone Y au corpus », « dépose le corpus sur le Drive », « corpus v2 ».
argument-hint: <nom du corpus, ex. fours_charbonnieres_648_v2>
entrees:
  - "configs/corpus_<nom>.yaml (existante ou à écrire sur gabarit)"
  - "datasets/<dataset>/ locaux CONFORMES (verif_dataset), split_manifest.yaml présents"
sorties:
  - "corpus/<nom>/ local (train/valid/test + corpus_manifest.yaml enrichi) CONFORME"
  - "model-training/<famille>/dataset/<nom>/ (Drive) + manifests/corpus/<nom>.yaml (commité)"
suivant: [entrainement-modele]
---

# Assembler un corpus d'entraînement (recette → construction → vérification → dépôt)

Interagir en français. Commandes exactes : CLAUDE.md § Commands. Cette skill
existe parce que le corpus `enclos_frie_graduel_648_v1` a été construit hors
procédure en août 2026 : manifeste faux (164 img déclarées / 219 réelles),
aucune recette — bras d'expérience irreproductible (audit 2026-08-31).

## Étape 1 — Résoudre la recette
- Config `configs/corpus_<nom>.yaml` : la charger, sinon l'écrire sur gabarit
  (`corpus_fours_charbonnieres_v1.yaml` = le bon exemple : en-tête décisions,
  `gsd_m:`, `rvt:`, `notes:` reprises au manifeste). Entrées `datasets:` = noms
  nus (dataset entier) OU mappings restreints `{nom, splits: [...], tuiles: [...]}`
  (sous-échantillonnage tracé — cas du corpus graduel).
- Vérifier CHAQUE dataset source : présent dans `datasets/` local, split_manifest
  présent, trace CONFORME (verif_dataset ou manifeste). Manquant → le régénérer
  (`/prepare-zone-training`) ou restaurer depuis `model-training/_datasets/`.
- Classes canoniques + `fusions:` explicites, validées par l'utilisateur.

## Étape 2 — Construction
- `build_corpus` (cf. CLAUDE.md). **GARDE-FOU rmtree : `--out` désigne le dossier
  DU corpus, DÉTRUIT puis reconstruit. Défaut `corpus\<nom>`. Ne JAMAIS passer
  `--out corpus`, ni un dossier contenant autre chose, ni le dossier des datasets.**

## Étape 3 — Boucle de vérification
- **`verif_corpus`** jusqu'à CONFORME (recomptes par classe/split/zone depuis les
  sources, sha1, coco_sha1 du manifeste). Divergence → corriger la recette →
  reconstruire INTÉGRALEMENT → re-vérifier. Jamais de rafistolage à la main.
- Si le corpus alimente une éval gelée (enclos FR) : vérifier que les listes
  d'images valid/test == `configs/eval_fr_gelee_v1.yaml`, intactes.

## Étape 4 — Dépôt + traçabilité
- Vérifier AVANT que la famille cible existe à la racine de model-training (règle
  anti « fours-à-chaux_charbonnieres » : une famille par thème, snake_case ASCII).
- Staging local → `model-training/<famille>/dataset/<nom>/` (robocopy /E, jamais
  /MIR ; exit 1 = succès) ; comparer comptes + sha1 échantillon par relecture G:.
- Copier `corpus_manifest.yaml` dans `manifests/corpus/<nom>.yaml` (versionné).

## Étape 5 — Récapitulatif
- Tableau comptes/splits/zones + notes du manifeste ; commit de la config + du
  manifeste versionné ; proposer `/entrainement-modele`.

## Garde-fous
- Splits HÉRITÉS des split_manifest, jamais re-tirés (règle anti-fuite — les
  manifests versionnés font foi).
- Un corpus déposé ne se modifie pas en place : v2 = nouveau nom, nouvelle recette.
- Idempotence : relancer sur un corpus déjà construit = re-vérification puis arrêt.
