# training-models

Audit et uniformisation des livraisons de données vecteur archéologiques, construction
des corpus d'entraînement multi-zones et entraînement des modèles RF-DETR (notebook
Colab canonique `docs/google_collab/`), consommés par le plugin QGIS
`archeologia-pipeline`.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Usage

```
.venv\Scripts\python.exe -m audit "<chemin-du-dataset>"   # audit.json + report.html
.venv\Scripts\python.exe tests\test_audit.py              # auto-test
```

La classification des noms inconnus se fait dans Claude Code : `/audit-dataset <chemin>`.
Le rangement d'exports Roboflow dans le Drive : `/dispatch-roboflow <dossier-zips>`
(outils : `tools/dispatch_roboflow.py`, `tools/build_v2_index.py` → `index.html`).
Taxonomie maîtresse : `taxonomy/entities.yaml` + `taxonomy/aliases.yaml`.
Stockage structuré des zones : voir CLAUDE.md § Stockage Drive (`data_regions_v2`).
