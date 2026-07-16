"""CLI : python -m audit <dataset_path> [-o OUTDIR] [--row-cap N] [--no-open]"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from audit.report import render_report
from audit.scan import build_audit, dump_audit, normalize

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(prog="audit", description="Audit brut d'une livraison de données archéologiques.")
    p.add_argument("dataset", help="Chemin du dossier de la livraison")
    p.add_argument("-o", "--out", help="Dossier de sortie (défaut : audits/<nom-du-dataset>/)")
    p.add_argument("--row-cap", type=int, default=50_000, help="Nb max d'entités lues par couche")
    p.add_argument("--no-open", action="store_true", help="Ne pas ouvrir le rapport dans le navigateur")
    args = p.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.is_dir():
        print(f"Erreur : {dataset} n'est pas un dossier existant.", file=sys.stderr)
        return 2

    outdir = Path(args.out).resolve() if args.out else ROOT / "audits" / (normalize(dataset.name) or "dataset")
    outdir.mkdir(parents=True, exist_ok=True)

    audit = build_audit(dataset, ROOT / "taxonomy", row_cap=args.row_cap)
    json_path = outdir / "audit.json"
    html_path = outdir / "report.html"
    json_path.write_text(dump_audit(audit), encoding="utf-8")
    html_path.write_text(render_report(audit), encoding="utf-8")

    unknown = sum(1 for c in audit["name_candidates"] if c["match"] is None)
    known = len(audit["name_candidates"]) - unknown
    print(f"Audit : {audit['dataset']['n_files']} fichiers, {len(audit['layers'])} couches vecteur, "
          f"{len(audit['errors'])} erreur(s), {len(audit['anomalies'])} anomalie(s)")
    print(f"Noms candidats : {known} connus/ignorés, {unknown} inconnus à classifier")
    print(f"Sorties : {json_path}\n          {html_path}")
    if not args.no_open:
        webbrowser.open(html_path.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
