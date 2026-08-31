"""Lanceur de TOUS les auto-tests maison : .venv\\Scripts\\python.exe tests\\run_all.py

Chaque tests/test_*.py est exécuté en sous-processus avec l'interpréteur courant
(style maison assert+main, pas de pytest). Deux listes explicites :

- SAUTES : tests qui exigent un autre interpréteur (lancés à part, cf. CLAUDE.md) ;
- INFORMATIFS : doublures de code du plugin archeologia-pipeline recopié ici — leur
  échec est AFFICHÉ mais n'invalide pas le run (la vérité vit côté plugin, où les
  vrais tests tournent ; ces copies sont candidates au déménagement).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

SAUTES = {
    "test_parity_bench.py": "PORTE de parité — exige le .venv_onnx du plugin (cf. CLAUDE.md, bloc bench)",
}
INFORMATIFS = {
    "test_reset_defauts_ui.py",
    "test_seuils_par_classe_plugin.py",
}


def main() -> int:
    resultats: list[tuple[str, str]] = []
    echec_bloquant = False
    for test in sorted(TESTS_DIR.glob("test_*.py")):
        if test.name in SAUTES:
            resultats.append((test.name, f"SAUTÉ ({SAUTES[test.name]})"))
            continue
        r = subprocess.run([sys.executable, str(test)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0:
            tag = "OK [informatif]" if test.name in INFORMATIFS else "OK"
            resultats.append((test.name, tag))
        else:
            queue = "\n".join((r.stdout + r.stderr).splitlines()[-6:])
            if test.name in INFORMATIFS:
                resultats.append((test.name, f"ÉCHEC [informatif — doublure plugin]\n{queue}"))
            else:
                resultats.append((test.name, f"ÉCHEC\n{queue}"))
                echec_bloquant = True

    largeur = max(len(n) for n, _ in resultats)
    for nom, statut in resultats:
        print(f"{nom:<{largeur}}  {statut}")
    n_ok = sum(1 for _, s in resultats if s.startswith("OK"))
    print(f"\n{n_ok}/{len(resultats)} OK — verdict : "
          + ("ÉCHEC" if echec_bloquant else "CONFORME"))
    return 1 if echec_bloquant else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
