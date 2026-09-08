"""Témoin d'intégrité d'une couche GPKG : effectif + sha256 des géométries WKB (promu le
2026-09-08 après trois sessions d'usage inline — règle des deux usages).

Sans --comparer : écrit `<gpkg>.<couche>.temoin.txt` et l'affiche.
Avec --comparer <temoin.txt> : recalcule et compare ; exit 1 si la couche a bougé
(effectif OU géométries différents — un déplacement de boîte ne change pas l'effectif).

    .venv/Scripts/python.exe tools/temoin_couche.py <gpkg> <couche> [--comparer <temoin.txt>]
"""
import argparse
import hashlib
import sys

import pyogrio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpkg")
    ap.add_argument("couche")
    ap.add_argument("--comparer", default=None, help="témoin antérieur à comparer")
    a = ap.parse_args()
    df = pyogrio.read_dataframe(a.gpkg, layer=a.couche)
    h = hashlib.sha256(b"".join(g.wkb for g in df.geometry)).hexdigest()[:16]
    ligne = f"n={len(df)} sha={h}"
    if a.comparer:
        avant = open(a.comparer, encoding="utf-8").read().strip()
        if avant == ligne:
            print(f"IDENTIQUE — {ligne}")
        else:
            print(f"DIFFÉRENT — avant : {avant} ; maintenant : {ligne} (la couche a bougé : relancer)")
            sys.exit(1)
    else:
        sortie = f"{a.gpkg}.{a.couche}.temoin.txt"
        open(sortie, "w", encoding="utf-8").write(ligne + "\n")
        print(f"{ligne} -> {sortie}")


if __name__ == "__main__":
    main()
