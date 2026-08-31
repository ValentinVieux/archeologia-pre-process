"""La doc décrit la CLI : CLAUDE.md § Commands vs argparse réels (par AST).

Trois étages :
1. doc -> code STRICT : chaque --option documentée existe dans l'argparse de l'outil
   (typo ou flag supprimé = rouge) ;
2. code -> doc sur les exigées : chaque option required=True d'un outil référencé
   apparaît dans sa ligne de doc ;
3. couverture : chaque tools/*.py de premier niveau possédant un ArgumentParser est
   référencé au moins une fois dans le bloc (liste EXEMPTES explicite) — c'est cet
   étage qui empêche le retour des outils fantômes (purge_roboflow_zone, 2026-08).

AST pur, AUCUN import des outils : points_a_recaler exécute son parse_args au niveau
module, et plusieurs outils exigent les venvs GPU (venv_adaf/venv_sam).

Lancement : .venv\\Scripts\\python.exe tests\\test_doc_cli.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEMPTES: set[str] = set()  # outils volontairement hors doc (vide : tout doit y être)


def bloc_commands() -> list[str]:
    texte = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    m = re.search(r"## Commands.*?```\n(.*?)```", texte, re.DOTALL)
    assert m, "bloc ``` du § Commands introuvable dans CLAUDE.md"
    return m.group(1).splitlines()


def cible_de(ligne: str) -> Path | None:
    """Fichier python visé par une ligne de commande du bloc (None si hors périmètre)."""
    avant = ligne.split(" # ")[0]
    m = re.search(r"-m\s+(audit|tools\.[\w.]+)", avant)
    if m:
        mod = m.group(1).replace(".", "/")
        for cand in (ROOT / f"{mod}/__main__.py", ROOT / f"{mod}.py"):
            if cand.exists():
                return cand
        return None
    m = re.search(r"(tools[\\/][\w\\/]+\.py|tests[\\/]run_all\.py)", avant)
    if m:
        p = ROOT / m.group(1).replace("\\", "/")
        return p if p.exists() else None
    return None


def options_du_code(fichier: Path) -> tuple[set[str], set[str]]:
    """(toutes les options longues, options required=True) des add_argument du fichier."""
    arbre = ast.parse(fichier.read_text(encoding="utf-8"))
    toutes: set[str] = set()
    requises: set[str] = set()
    for node in ast.walk(arbre):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        noms = [a.value for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
                and a.value.startswith("-")]
        toutes.update(noms)
        for kw in node.keywords:
            if (kw.arg == "required" and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True):
                requises.update(n for n in noms if n.startswith("--"))
    return toutes, requises


def a_un_parser(fichier: Path) -> bool:
    try:
        arbre = ast.parse(fichier.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    return any(
        isinstance(n, ast.Call) and (
            (isinstance(n.func, ast.Name) and n.func.id == "ArgumentParser")
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "ArgumentParser")
        )
        for n in ast.walk(arbre)
    )


def main() -> None:
    lignes = bloc_commands()
    problemes: list[str] = []
    references: set[Path] = set()

    for ligne in lignes:
        l = ligne.strip()
        if not l or l.startswith("#") or l.startswith("docker"):
            continue
        cible = cible_de(l)
        if cible is None:
            continue
        references.add(cible.resolve())
        avant = l.split(" # ")[0]
        opts_doc = set(re.findall(r"--[a-zA-Z][\w-]*", avant))
        opts_code, opts_requises = options_du_code(cible)
        # 1) doc -> code strict
        for o in sorted(opts_doc - opts_code):
            problemes.append(f"{cible.name} : option documentée inexistante {o}")
        # 2) required -> doc
        for o in sorted(opts_requises - opts_doc):
            problemes.append(f"{cible.name} : option REQUISE non documentée {o}")

    # 3) couverture : tout tools/*.py de premier niveau avec parser doit être référencé
    for outil in sorted((ROOT / "tools").glob("*.py")):
        if outil.name in EXEMPTES or not a_un_parser(outil):
            continue
        if outil.resolve() not in references:
            problemes.append(
                f"{outil.name} : outil à ArgumentParser ABSENT du bloc Commands de CLAUDE.md"
            )

    assert not problemes, "doc <-> CLI divergents :\n  " + "\n  ".join(problemes)
    print(f"OK — doc<->CLI : {len(references)} outils référencés, "
          f"{sum(1 for _ in (ROOT / 'tools').glob('*.py'))} tools/*.py scannés")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
