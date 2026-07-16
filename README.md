# pre-process-data

Audit et uniformisation des livraisons de données vecteur archéologiques (amont de
l'entraînement des modèles RF-DETR sur Roboflow, consommés par le plugin QGIS
`archeologia-pipeline`).

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
Taxonomie maîtresse : `taxonomy/entities.yaml` + `taxonomy/aliases.yaml`.
