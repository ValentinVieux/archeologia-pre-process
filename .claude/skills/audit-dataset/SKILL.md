---
name: audit-dataset
description: >
  Auditer un dossier de données vecteur archéologiques et classifier interactivement
  les noms d'entités inconnus dans la taxonomie maîtresse (taxonomy/entities.yaml,
  taxonomy/aliases.yaml). Utiliser quand l'utilisateur dit « audite ce dataset »,
  « auditer le dossier », « classifie les entités », « nouvelles données des
  archéologues », « audit this dataset », « classify unknown names », ou fournit le
  chemin d'une nouvelle livraison.
argument-hint: <chemin-du-dossier-dataset>
entrees:
  - "livraison vecteur brute (dossier local ou raw/ d'une zone data_regions_v2)"
sorties:
  - "audits/<nom-normalise>/ (audit.json + report.html, régénérables)"
  - "taxonomy/aliases.yaml (append-only) + entités candidates (commit proposé)"
suivant: [prepare-zone-training]
---

# Audit d'un dataset + classification des noms inconnus

Interagir avec l'utilisateur en français. Toute écriture dans `taxonomy/` suit les
règles de CLAUDE.md § Taxonomie (regex des ids, candidate-only, aliases append-only,
validation post-édition).

## Étape 1 — Résoudre les entrées
- Chemin du dataset : depuis l'argument ; sinon le demander. Vérifier que le dossier
  existe. `<dataset-name>` = nom du dossier normalisé comme `normalize()` de scan.py :
  accents supprimés, minuscules, runs non alphanumériques fusionnés en un seul `_`,
  `_` de bord retirés, repli `dataset` si vide.
- Lire en entier `taxonomy/entities.yaml` et `taxonomy/aliases.yaml`.

## Étape 2 — Lancer l'audit Python
Exécuter la commande de CLAUDE.md § Commands avec `--no-open` :
`.venv\Scripts\python.exe -m audit "<chemin>" --no-open`
En cas d'échec : s'arrêter et rapporter l'erreur. Ne JAMAIS parser les fichiers du
dataset soi-même en secours.

## Étape 3 — Collecter les inconnus
Lire audit.json au chemin exact imprimé par l'Étape 2 (ligne `Sorties :` du stdout —
ne pas recalculer le nom normalisé) → candidats dont `match` est `null`.
Re-vérification défensive (idempotence) : écarter tout candidat dont la forme
normalisée correspond déjà à un alias, un ignored, un id d'entité, un label_fr ou une
classe Roboflow. S'il ne reste rien : afficher le tableau récap (Étape 8) et s'arrêter.

## Étape 4 — Clusteriser les variantes
Grouper les inconnus dont les formes normalisées sont identiques ou trivialement
proches (pluriel en `s`, ponctuation). Un cluster = une décision, mais chaque chaîne
brute du cluster recevra sa propre ligne dans aliases.yaml.

## Étape 5 — Proposer les classifications (AskUserQuestion, lots de ≤ 4 questions)
Pour chaque cluster, choisir un meilleur candidat en comparant la forme normalisée aux
ids, label_fr, alias et classes Roboflow existants. Options (≤ 4) :
1. Meilleure entité candidate — libellé « id (label_fr) »
2. Alternative plausible s'il y en a une
3. « Nouvelle catégorie »
4. « Ignorer (pas une entité) »
Dans le texte de la question : les chaînes brutes, leurs sources (fichier / couche /
champ) et les comptes d'occurrences tirés d'audit.json. Une réponse libre = un id à
valider contre entities.yaml.

## Étape 6 — Nouvelle catégorie
Si « Nouvelle catégorie » : proposer dans une question de suivi
- `id` : français translittéré ASCII snake_case, regex `^[a-z][a-z0-9_]*$`
- `label_fr` : label français accentué
- `morphology` : circulaire | lineaire | zone — inférée des types géométriques
  d'audit.json si disponibles, sinon du nom
- `status: candidate` — TOUJOURS. Ne jamais créer `canonical`, ne jamais remplir
  `plugin_entity_id` ni `roboflow_classes` (promotions manuelles).
Laisser l'utilisateur accepter ou ajuster avant d'écrire quoi que ce soit.

## Étape 7 — Persister
- Ajouter les entités candidates à `taxonomy/entities.yaml` (append en fin de liste).
- Ajouter une entrée par chaîne brute à `taxonomy/aliases.yaml` : section `aliases:`
  ({raw, entity_id, source_dataset, context, decided}) ou `ignored:`
  ({raw, source_dataset, decided}). Ne jamais modifier les lignes existantes.
- Valider les deux fichiers (règles CLAUDE.md § Taxonomie). Si invalide : corriger sa
  propre édition avant de continuer.

## Étape 8 — Régénérer + récapituler
- Relancer la commande de l'Étape 2 (audit.json/report.html reflètent les mappings).
- Afficher un tableau : nom(s) brut(s) | décision (entity_id, « NOUVEAU (candidate) »
  ou « ignoré ») | occurrences ; puis les comptes (mappés / nouveaux candidats /
  ignorés / inconnus restants — doit être 0).
- Proposer (sans l'exécuter d'office) :
  `git add taxonomy && git commit -m "taxonomy: map <N> aliases from <dataset-name>"`
- Si la zone est destinée à l'entraînement : proposer d'enchaîner sur
  `/prepare-zone-training`.

## Garde-fous
- Chaque mapping est validé par l'utilisateur — ne jamais écrire un alias en silence,
  même pour un match quasi exact.
- Création d'entités : candidate uniquement. Aliases : append-only. Jamais de
  renommage ni de suppression.
- L'idempotence vit dans aliases.yaml (y compris `ignored:`) — relancer la skill sur
  le même dataset doit poser zéro question.
- Journal des décisions = provenance d'aliases.yaml + historique git. Ne pas créer de
  fichier de log séparé.
