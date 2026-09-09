"""Auto-test de tools/cloturer_chantier.py sur une arborescence jetable.

Lancement : .venv\\Scripts\\python.exe tests\\test_cloturer_chantier.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTIL = ROOT / "tools" / "cloturer_chantier.py"


def ecrire(chemin: Path, contenu: bytes) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_bytes(contenu)


def lancer(brouillon: Path, cible: Path, rapport: Path) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(OUTIL), str(brouillon), "--sans-defaut",
         "--cible", str(cible), "--json", str(rapport)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.returncode in (0, 2), f"code inattendu {proc.returncode}\n{proc.stderr}"
    return proc.returncode, json.loads(rapport.read_text(encoding="utf-8"))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        brouillon, cible = base / "brouillon", base / "drive"
        rapport = base / "rapport.json"

        # 1. Un fichier déposé (même nom, même taille) et un fichier orphelin.
        ecrire(brouillon / "zones" / "entites.gpkg", b"A" * 100)
        ecrire(cible / "region" / "zone" / "entites.gpkg", b"B" * 100)  # taille identique
        ecrire(brouillon / "diag" / "jetable.log", b"x" * 42)

        code, rap = lancer(brouillon, cible, rapport)
        assert code == 2, "un orphelin doit donner exit 2"
        assert rap["sans_jumeau_fichiers"] == 1, rap["sans_jumeau_fichiers"]
        assert rap["sans_jumeau_octets"] == 42, rap["sans_jumeau_octets"]
        assert rap["total_fichiers"] == 2, rap["total_fichiers"]
        assert rap["par_dossier"]["zones"]["n_sans_jumeau"] == 0
        assert rap["par_dossier"]["diag"]["exemples"] == ["jetable.log"]

        # 2. Le même nom mais une AUTRE taille ne compte pas comme jumeau.
        ecrire(cible / "region" / "jetable.log", b"y" * 41)
        code, rap = lancer(brouillon, cible, rapport)
        assert code == 2 and rap["sans_jumeau_fichiers"] == 1, "taille différente = pas un jumeau"

        # 3. cloture-ignore.txt : un motif glob justifie l'orphelin -> CLÔTURABLE.
        ecrire(brouillon / "cloture-ignore.txt",
               "# journaux, régénérables\ndiag/*.log  # sorties de diagnostic\n".encode("utf-8"))
        code, rap = lancer(brouillon, cible, rapport)
        assert code == 0, f"orphelin justifié doit donner exit 0 (reçu {code})"
        assert rap["sans_jumeau_fichiers"] == 0, rap["sans_jumeau_fichiers"]
        # 2 justifiés : l'orphelin couvert par le motif, et cloture-ignore.txt lui-même
        assert rap["justifies_fichiers"] == 2, rap["justifies_fichiers"]
        assert rap["motifs_ignores"] == ["diag/*.log"], rap["motifs_ignores"]
        # le fichier d'ignore lui-même ne compte jamais comme orphelin
        assert rap["par_dossier"]["."]["n_sans_jumeau"] == 0

        # 4. Un motif qui ne couvre pas l'orphelin ne le justifie pas.
        ecrire(brouillon / "cloture-ignore.txt", b"autre/*.log\n")
        code, rap = lancer(brouillon, cible, rapport)
        assert code == 2 and rap["sans_jumeau_fichiers"] == 1, "motif hors sujet = pas de justification"

        # 5. Tout déposé -> CLÔTURABLE même sans motif.
        ecrire(brouillon / "cloture-ignore.txt", b"")
        ecrire(cible / "region" / "logs" / "jetable.log", b"z" * 42)
        code, rap = lancer(brouillon, cible, rapport)
        assert code == 0, f"tout déposé doit donner exit 0 (reçu {code})"
        assert rap["sans_jumeau_fichiers"] == 0

        # 6. L'outil ne supprime rien.
        assert (brouillon / "diag" / "jetable.log").exists(), "le brouillon doit rester intact"

        # 7. Plusieurs brouillons en une passe : verdict = le pire, un rapport chacun.
        autre = base / "brouillon2"
        ecrire(autre / "orphelin.bin", b"q" * 77)
        proc = subprocess.run(
            [sys.executable, str(OUTIL), str(brouillon), str(autre), "--sans-defaut", "--cible", str(cible)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert proc.returncode == 2, f"un brouillon non clôturable doit donner 2 (reçu {proc.returncode})"
        r1 = json.loads((brouillon / "cloture.json").read_text(encoding="utf-8"))
        r2 = json.loads((autre / "cloture.json").read_text(encoding="utf-8"))
        assert r1["sans_jumeau_fichiers"] == 0 and r2["sans_jumeau_fichiers"] == 1, (r1, r2)

    print("test_cloturer_chantier : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
