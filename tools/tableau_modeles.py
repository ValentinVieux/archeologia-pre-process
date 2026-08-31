"""Dashboard HTML de l'évolution des métriques des modèles par famille et par classe.

Scanne une racine (typiquement G:\\...\\model-training) PAR CONVENTION —
`**/runs/training/<run>/evaluation/metriques_eval.json` (sortie canonique de
tools/courbes_eval.py), marche élaguée des dossiers lourds (dataset/corpus/
checkpoints/…, GoogleDriveFS oblige) — et génère UN index.html statique
(zéro JS, CSS inline) :
  - par famille de modèles (1er segment du chemin relatif) : tableau des évaluations
    (tâche, seuil F1-max, F1, P, R, AP50, n_gt, IoU médian, lien vers les planches) ;
  - par classe : sparklines SVG de l'évolution (F1, AP50, seuil) entre versions,
    ordonnées par date de génération ;
  - les runs `runs/training/<run>` SANS metriques_eval.json sont listés « sans
    mesure » (trace le reste-à-faire) ;
  - un fichier `PROVISOIRE.txt` à côté du metriques_eval.json = modèle voué au
    remplacement : étiqueté dans le tableau, EXCLU de l'évolution par classe ;
  - fichiers illisibles = avertissements dans la page, jamais un crash
    (GoogleDriveFS fragile).

Usage (.venv — AUCUN GPU) :
  .venv\\Scripts\\python.exe tools\\tableau_modeles.py "<racine model-training>" [--out <html>]
Défaut : écrit <racine>\\index.html. Régénérer après tout dépôt d'évaluation.
"""
import argparse
import html
import json
import os
import sys
from datetime import datetime
from pathlib import Path


LOURDS = {"dataset", "datasets", "corpus", "checkpoints", "visualizations", "weights"}


def iter_runs(racine):
    """Dossiers runs/training/<run> par marche élaguée (GoogleDriveFS : jamais de
    rglob complet — les corpus d'images et les checkpoints rendent le scan infini)."""
    for base, dirs, _ in os.walk(racine):
        base = Path(base)
        if base.name == "training" and base.parent.name == "runs":
            for d in sorted(dirs):
                yield base / d
            dirs[:] = []  # ne pas descendre dans les runs
        elif base.name == "runs":
            dirs[:] = [d for d in dirs if d == "training"]
        else:
            dirs[:] = [d for d in dirs if d.lower() not in LOURDS and not d.startswith(".")]


def collecter(racine):
    """(évals valides, runs sans mesure, avertissements) — par CONVENTION :
    l'éval canonique d'un run vit dans runs/training/<run>/evaluation/metriques_eval.json."""
    racine = Path(racine)
    evals, sans_mesure, avertissements = [], [], []
    try:
        runs = list(iter_runs(racine))
    except OSError as e:
        return [], [], [f"scan des runs : {e}"]
    for run in runs:
        p = run / "evaluation" / "metriques_eval.json"
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if str(data.get("schema", "")).startswith("metriques_eval/"):
                    evals.append((p, data))
                else:
                    avertissements.append(f"{p} : schéma inattendu {data.get('schema')!r}")
            elif any(run.iterdir()):  # résidus vides ignorés
                sans_mesure.append(run)
        except Exception as e:  # fichier corrompu/illisible : on continue
            avertissements.append(f"{p} : {e}")
    return evals, sans_mesure, avertissements


def famille_de(chemin, racine):
    rel = chemin.relative_to(racine)
    return rel.parts[0] if len(rel.parts) > 1 else "(racine)"


def sparkline(valeurs, largeur=120, hauteur=28):
    """Polyligne SVG inline, échelle fixe 0-1 (comparabilité entre familles)."""
    pts = [v for v in valeurs if v is not None]
    if not pts:
        return "—"
    if len(pts) == 1:
        xs = [largeur / 2]
    else:
        xs = [4 + i * (largeur - 8) / (len(pts) - 1) for i in range(len(pts))]
    ys = [hauteur - 4 - max(0.0, min(1.0, v)) * (hauteur - 8) for v in pts]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    titre = " → ".join(f"{v:.3f}" for v in pts)
    return (f'<svg width="{largeur}" height="{hauteur}" role="img"><title>{titre}</title>'
            f'<polyline points="{poly}" fill="none" stroke="#1f77b4" stroke-width="1.5"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2.5" fill="#c1272d"/></svg>'
            f' <span class="val">{pts[-1]:.3f}</span>')


def construire(racine):
    """La page est destinée à <racine>/index.html : liens relatifs à la RACINE
    (un --out de staging ne change pas les liens)."""
    racine = Path(racine)
    evals, sans_mesure, avertissements = collecter(racine)

    familles = {}
    for p, data in evals:
        fam = famille_de(p, racine)
        # PROVISOIRE.txt à côté de metriques_eval.json = modèle voué au remplacement
        # (data retravaillées, réentraînement prévu) : gardé dans le tableau avec
        # son étiquette, EXCLU de l'évolution par classe. Le marqueur meurt avec
        # le run quand il est supprimé.
        provisoire = (p.parent / "PROVISOIRE.txt").exists()
        for nom, m in data.get("modeles", {}).items():
            familles.setdefault(fam, []).append({
                "nom": nom, "date": data.get("genere_le", ""), "tache": data.get("tache", "?"),
                "global": m.get("global", {}), "par_classe": m.get("par_classe", {}),
                "dossier": p.parent, "provisoire": provisoire,
            })
    for lignes in familles.values():
        lignes.sort(key=lambda l: (l["date"], l["nom"]))

    e = html.escape
    parties = [
        "<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'>",
        "<title>Modèles — métriques d'évaluation</title><style>",
        "body{font-family:Segoe UI,sans-serif;margin:1.5em;background:#fafafa;color:#222}",
        "table{border-collapse:collapse;margin:.5em 0 1em}",
        "th,td{border:1px solid #ccc;padding:3px 8px;font-size:13px;text-align:left}",
        "th{background:#eee}details{margin-bottom:1.2em}summary{font-size:17px;",
        "font-weight:600;cursor:pointer}.val{font-size:11px;color:#555}",
        ".note{color:#777;font-size:12px}.warn{color:#a33;font-size:12px}",
        "h3{margin:.6em 0 .2em;font-size:14px}</style></head><body>",
        "<h1>Modèles — métriques d'évaluation</h1>",
        f"<p class='note'>Généré le {datetime.now().isoformat(timespec='seconds')} par "
        f"tools/tableau_modeles.py — {len(evals)} évaluation(s), "
        f"{sum(len(v) for v in familles.values())} ligne(s) modèle, "
        f"{len(sans_mesure)} run(s) sans mesure. Source : metriques_eval.json "
        "(seuils F1-max mesurés — jamais de seuil fixe). "
        "<a href='../data/data_regions_v2/index.html'>→ index data_regions_v2</a></p>",
    ]
    for msg in avertissements:
        parties.append(f"<p class='warn'>⚠ {e(str(msg))}</p>")

    for fam in sorted(familles):
        lignes = familles[fam]
        parties.append(f"<details open><summary>{e(fam)}</summary>")
        parties.append("<table><tr><th>modèle</th><th>date</th><th>tâche</th>"
                       "<th>seuil F1-max</th><th>F1</th><th>P</th><th>R</th><th>AP50</th>"
                       "<th>n_gt</th><th>IoU méd.</th><th>planches</th></tr>")
        for l in lignes:
            g = l["global"]
            lien = os.path.relpath(l["dossier"], racine).replace("\\", "/")

            def cell(v):
                return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

            etiquette = " <span class='warn'>[provisoire]</span>" if l["provisoire"] else ""
            parties.append(
                f"<tr><td>{e(l['nom'])}{etiquette}</td><td>{e(l['date'][:10])}</td>"
                f"<td>{e(l['tache'])}</td><td>{cell(g.get('seuil_f1max'))}</td>"
                f"<td>{cell(g.get('F1'))}</td><td>{cell(g.get('P'))}</td>"
                f"<td>{cell(g.get('R'))}</td><td>{cell(g.get('AP50'))}</td>"
                f"<td>{cell(g.get('n_gt'))}</td><td>{cell(g.get('iou_median'))}</td>"
                f"<td><a href='{e(lien)}/courbes_seuils_pr.png'>courbes</a></td></tr>")
        parties.append("</table>")

        # évolution par classe : dernière valeur par nom de modèle, ordre par date ;
        # les modèles PROVISOIRES (marqueur dans evaluation/) n'entrent pas dans la série
        derniers = {}
        for l in lignes:  # déjà triées par date — la plus récente écrase
            if l["provisoire"]:
                continue
            derniers[l["nom"]] = l
        serie = sorted(derniers.values(), key=lambda l: (l["date"], l["nom"]))
        classes = sorted({c for l in serie for c in l["par_classe"]})
        if classes:
            parties.append("<h3>Évolution par classe (versions : "
                           + " → ".join(e(l["nom"]) for l in serie) + ")</h3>")
            parties.append("<table><tr><th>classe</th><th>F1</th><th>AP50</th>"
                           "<th>seuil F1-max</th><th>n_gt (dernier)</th></tr>")
            for cl in classes:
                blocs = [l["par_classe"].get(cl) for l in serie]
                col = lambda k: [b.get(k) if b else None for b in blocs]
                ngt = next((b.get("n_gt") for b in reversed(blocs) if b), None)
                parties.append(f"<tr><td>{e(cl)}</td><td>{sparkline(col('F1'))}</td>"
                               f"<td>{sparkline(col('AP50'))}</td>"
                               f"<td>{sparkline(col('seuil_f1max'))}</td>"
                               f"<td>{'—' if ngt is None else ngt}</td></tr>")
            parties.append("</table>")
        parties.append("</details>")

    if sans_mesure:
        parties.append("<details open><summary>Runs sans mesure "
                       f"({len(sans_mesure)})</summary><ul>")
        for run in sans_mesure:
            parties.append(f"<li class='note'>{e(str(run.relative_to(racine)))}</li>")
        parties.append("</ul></details>")
    parties.append("</body></html>")
    return "\n".join(parties)


def maj_registre(racine, chemin_registre):
    """Met à jour modeles.yaml (registre zone<->modèle de data_regions_v2) en MERGE :
    les champs manuels des entrées existantes (statut, entites, notes, zones) sont
    conservés ; chaque run porteur d'un metriques_eval.json est ajouté/actualisé
    avec son bloc `evaluation` et ses zones mesurées. Remplace la mise à jour
    manuelle (registre figé au 2026-07-16 constaté à l'audit 2026-08-31)."""
    import yaml

    chemin = Path(chemin_registre)
    registre = {"schema_version": 1, "modeles": []}
    if chemin.exists():
        registre = yaml.safe_load(chemin.read_text(encoding="utf-8")) or registre
    entrees = {m.get("nom"): m for m in registre.get("modeles", []) if isinstance(m, dict)}

    evals, _, _ = collecter(Path(racine))
    for _, data in sorted(evals, key=lambda t: t[1].get("genere_le", "")):
        tache = ("détection" if data.get("tache") == "detection"
                 else "segmentation d'instances")
        for nom, m in (data.get("modeles") or {}).items():
            e = entrees.setdefault(nom, {"nom": nom, "architecture": "RF-DETR",
                                         "statut": "expérimental"})
            e.setdefault("architecture", "RF-DETR")
            e["tache"] = tache
            if not e.get("entites"):
                e["entites"] = ", ".join(sorted(m.get("par_classe") or {}))
            zones = sorted({z.split("/", 1)[-1] for z in (m.get("par_zone") or {})})
            if zones:
                e["zones"] = zones
            g = m.get("global") or {}
            e["evaluation"] = {"date": (data.get("genere_le") or "")[:10],
                               "f1": g.get("F1"), "ap50": g.get("AP50"),
                               "seuil": g.get("seuil_f1max")}
    registre["modeles"] = sorted(entrees.values(), key=lambda m: m.get("nom", ""))
    chemin.write_text(yaml.safe_dump(registre, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")
    yaml.safe_load(chemin.read_text(encoding="utf-8"))  # relecture de validation
    print(f"registre -> {chemin} ({len(registre['modeles'])} modèles)")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("racine", help="racine à scanner (ex. G:\\...\\model-training)")
    ap.add_argument("--out", default=None, help="HTML de sortie (défaut : <racine>\\index.html)")
    ap.add_argument("--registre", default=None,
                    help="modeles.yaml de data_regions_v2 à mettre à jour (merge)")
    a = ap.parse_args()
    sortie = a.out or os.path.join(a.racine, "index.html")
    page = construire(a.racine)
    with open(sortie, "w", encoding="utf-8") as f:
        f.write(page)
    assert "<html" in open(sortie, encoding="utf-8").read(200), "relecture invalide"
    print("dashboard ->", sortie)
    if a.registre:
        maj_registre(a.racine, a.registre)


if __name__ == "__main__":
    main()
