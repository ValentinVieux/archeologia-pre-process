# CLAUDE.md

Préprocessing des données d'entraînement des modèles CV RF-DETR (données sur Roboflow,
modèles consommés par le plugin QGIS `archeologia-pipeline`). Des archéologues de toute
la France envoient des datasets vecteur hétérogènes ; ce repo les audite et uniformise
les nommages vers une taxonomie maîtresse.

Deux couches : un outil Python **déterministe** (audit.json + rapport HTML statique par
livraison) et une couche sémantique pilotée par Claude (skill `/audit-dataset`) qui mappe
les noms bruts sur la taxonomie, avec validation humaine de chaque décision.

## Commands (Windows — ne jamais activer le venv, appeler son python directement)

```
.venv\Scripts\python.exe -m audit "<dataset-path>" [--no-open]   # audit d'une livraison
.venv\Scripts\python.exe tests\test_audit.py                     # auto-test complet
# setup initial : py -3.11 -m venv .venv ; .venv\Scripts\pip install -r requirements.txt
```

Sortie par défaut : `audits\<nom-normalisé>\` (stable, écrasée à chaque run — les audits
sont régénérables, ils ne sont pas commités). Normalisation = `normalize()` de scan.py :
accents supprimés (NFKD), minuscules, tout run de caractères hors `[a-z0-9]` fusionné en
un seul `_`, `_` de bord retirés, repli `dataset` si vide. Ex. « Prospection Rambouillet »
→ `audits\prospection_rambouillet\`. La CLI imprime les chemins exacts sur stdout
(ligne `Sorties :`). Cette section est la source de vérité de la CLI : la skill
`audit-dataset` lit la commande ici.

## Taxonomie — taxonomy/entities.yaml + taxonomy/aliases.yaml

Schémas documentés en tête de chaque fichier. Règles non négociables :

- `id` : snake_case ASCII, regex `^[a-z][a-z0-9_]*$` (contrat `docs/model_contract.md` du
  plugin). Les accents comptent partout ailleurs : `charbonniere` (id) ≠ `charbonnière`
  (label/classe Roboflow, conservée verbatim).
- **Ne JAMAIS renommer un id `canonical`** sans vérifier le plugin
  (`archeologia-pipeline/src/config/config_manager.py` → `_ENTITY_ID_RENAMES`) et les
  classes Roboflow liées (`roboflow_classes`).
- Claude ne crée des entités qu'en `status: candidate`. La promotion en `canonical` et le
  remplissage de `plugin_entity_id`/`roboflow_classes` sont des décisions humaines.
- `aliases.yaml` est **append-only** : jamais modifier ni supprimer une entrée existante.
- Après toute édition : valider (regex des ids, unicité des ids, chaque `entity_id`
  d'alias existe dans entities.yaml, `morphology` ∈ {circulaire, lineaire, zone},
  `status` ∈ {canonical, candidate}).

## Workflow

audit (Python) → classification interactive des inconnus (l'utilisateur valide chaque
mapping) → append aliases + entités candidates → relancer l'audit. Le tout via
`/audit-dataset <chemin>`. Historique des décisions = provenance dans aliases.yaml +
git log ; pas de fichier de log séparé.

## Langue

Données, labels (`label_fr`) et interaction utilisateur : français. Code, commentaires et
identifiants : le code est en français-technique assumé (docstrings FR), les ids d'entités
sont du français translittéré ASCII (`charbonniere`, `chemin_creux`) — convention de tout
l'écosystème plugin/Roboflow.
