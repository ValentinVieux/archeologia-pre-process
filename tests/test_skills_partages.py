"""Les skills d'installation d'un modèle sont identiques ici et dans le plugin.

`installer-modele-plugin` et `fiche-classe-plugin` vivent à l'identique dans
`.claude/skills/` des deux dépôts. Divergence = un seul dépôt a été modifié :
`python <plugin>/dev/sync_skills_training.py --vers-training | --depuis-training`.
Comparaison insensible aux fins de ligne (autocrlf). Sauté (exit 0) quand le plugin
n'est pas sur le poste.

Lancement : .venv\\Scripts\\python.exe tests\\test_skills_partages.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = Path(os.environ.get("APPDATA", "")) / "QGIS/QGIS3/profiles/default/python/plugins/archeologia-pipeline"
PARTAGES = ("installer-modele-plugin", "fiche-classe-plugin")


def _normalise(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


def main() -> int:
    if not PLUGIN.is_dir():
        print(f"SAUTÉ : plugin absent ({PLUGIN})")
        return 0
    divergents = []
    for skill in PARTAGES:
        ici = ROOT / ".claude" / "skills" / skill
        la = PLUGIN / ".claude" / "skills" / skill
        noms = {p.name for p in ici.glob("*")} | {p.name for p in la.glob("*")} if ici.is_dir() and la.is_dir() else set()
        if not noms:
            divergents.append((skill, "dossier absent d'un côté"))
            continue
        for nom in sorted(noms):
            a, b = ici / nom, la / nom
            if not (a.is_file() and b.is_file()):
                divergents.append((skill, f"{nom} absent d'un côté"))
            elif _normalise(a) != _normalise(b):
                divergents.append((skill, f"{nom} différent"))
    assert not divergents, f"skills partagés divergents : {divergents}"
    print(f"OK : {len(PARTAGES)} skills partagés identiques plugin/training-models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
