"""Dashboard HTML de l'évolution des métriques des modèles par famille et par classe.

Scanne une racine (typiquement G:\\...\\model-training) PAR CONVENTION —
`**/runs/training/<run>/evaluation/metriques_eval.json` (sortie canonique de
tools/courbes_eval.py), marche élaguée des dossiers lourds (dataset/corpus/
checkpoints/…, GoogleDriveFS oblige) — et génère UN index.html statique
(zéro JS, CSS inline, <details> natifs) :
  - par famille de modèles (1er segment du chemin relatif) : ligne des classes
    (fusions corpus + fusions d'éval), bloc « À surveiller » (zones fragiles,
    runs sans mesure, manifeste incomplet, provisoire, seuil déployé ≠ F1-max —
    « décroche » et « rare en train » restent des étiquettes de ligne dans les
    tables zone × classe, hors alertes, décision 2026-09-03), tableau des
    évaluations (tâche, seuil F1-max, F1, P, R, AP50, n_gt, IoU médian, lien vers
    les planches, badge plugin avec --plugin) ;
  - par classe : sparklines SVG de l'évolution (F1, AP50, seuil) entre versions,
    ordonnées par date de génération, n_gt et écart F1 dernière − précédente ;
  - UNE FICHE repliée par modèle (2026-09-03) : données par classe (annotations
    train/valid/test du manifeste de corpus vs n_gt mesuré), zones (+ zone × classe
    si `par_zone_classe` est présent), run (params_run.yaml sinon config.json,
    jamais deviné : « non tracé »), époques FUSIONNÉES (metrics.csv + historiques
    *avant_reprise* — metrics.csv seul = dernière reprise uniquement), courbe
    mAP50 EMA par époque, bloc plugin (model_card.yaml : badge, seuils déployés
    vs F1-max, lignée) ;
  - pied de page : matrice de couverture zones × familles + glossaire ;
  - les runs `runs/training/<run>` SANS metriques_eval.json sont COMPTÉS dans
    l'en-tête et, s'ils portent un checkpoint, signalés dans « À surveiller » ;
  - un fichier `PROVISOIRE.txt` à côté du metriques_eval.json = modèle voué au
    remplacement : étiqueté dans le tableau, EXCLU de l'évolution par classe ;
  - fichiers illisibles = avertissements dans la page, jamais un crash
    (GoogleDriveFS fragile).

Jointure run -> corpus : run = dossier runs/training/<nom du modèle> s'il existe,
sinon le run porteur de l'éval pour son PREMIER modèle ; corpus = basename de
params_run.yaml:CORPUS_DRIVE_DIR sinon config.json:dataset.dataset_dir ;
manifeste = <depot>/manifests/corpus/<corpus>.yaml. Sans manifeste : « données
non tracées ».

Usage (.venv — AUCUN GPU) :
  .venv\\Scripts\\python.exe tools\\tableau_modeles.py "<racine model-training>" [--out <html>]
      [--depot <repo>] [--plugin <data/models>] [--seuil-fragile 30] [--min-train 100]
Défaut : écrit <racine>\\index.html. Régénérer après tout dépôt d'évaluation.
"""
import argparse
import csv
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


LOURDS = {"dataset", "datasets", "corpus", "checkpoints", "visualizations", "weights"}
NT = "non tracé"
FIN = "\u202f"  # espace fine insécable (séparateur de milliers)
TEINTE = "42,120,214"  # bleu unique des cellules P/R/F1/R_max


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


# --- formats ---------------------------------------------------------------

def fmt_n(n):
    return "—" if n is None else f"{int(n):,}".replace(",", FIN)


def fmt_f(v, dec=3):
    return "—" if v is None else f"{float(v):.{dec}f}".replace(".", ",")


def fmt_pct(x):
    return "—" if x is None else f"{100 * x:.1f}".replace(".", ",") + FIN + "%"


def teinte(v):
    """Fond bleu unique, alpha proportionnel à la valeur (0 = blanc)."""
    if v is None:
        return ""
    return f' style="background:rgba({TEINTE},{0.62 * max(0.0, min(1.0, float(v))):.2f})"'


def td_t(v):
    return f"<td{teinte(v)}>{fmt_f(v)}</td>"


def cell(v):
    if v is None:
        return "—"
    return fmt_f(v) if isinstance(v, float) else fmt_n(v) if isinstance(v, int) else str(v)


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
    titre = " → ".join(fmt_f(v) for v in pts)
    return (f'<svg width="{largeur}" height="{hauteur}" role="img"><title>{titre}</title>'
            f'<polyline points="{poly}" fill="none" stroke="#1f77b4" stroke-width="1.5"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2.5" fill="#c1272d"/></svg>'
            f' <span class="val">{fmt_f(pts[-1])}</span>')


def courbe_epoques(vals, prevues, meilleure, reprises, largeur=360, hauteur=64):
    """mAP50 EMA par époque : abscisse = époques prévues, point sur la meilleure,
    traits verticaux fins aux reprises."""
    if not vals:
        return "—"
    n = max(prevues or 0, max(vals) + 1)
    x = lambda ep: 4 + ep * (largeur - 8) / max(n - 1, 1)
    y = lambda v: hauteur - 4 - max(0.0, min(1.0, v)) * (hauteur - 8)
    poly = " ".join(f"{x(ep):.1f},{y(v):.1f}" for ep, v in sorted(vals.items()))
    traits = "".join(f'<line x1="{x(r):.1f}" y1="0" x2="{x(r):.1f}" y2="{hauteur}" '
                     'stroke="#888" stroke-width="0.8" stroke-dasharray="2,2"/>' for r in reprises)
    point = (f'<circle cx="{x(meilleure):.1f}" cy="{y(vals[meilleure]):.1f}" r="3" fill="#c1272d"/>'
             if meilleure in vals else "")
    return (f'<svg width="{largeur}" height="{hauteur}" role="img" style="border-bottom:1px solid #ccc">'
            f'<title>mAP50 EMA par époque</title>{traits}'
            f'<polyline points="{poly}" fill="none" stroke="#1f77b4" stroke-width="1.5"/>{point}</svg>')


# --- dépôt (manifestes de corpus, taxonomie) --------------------------------

def lire_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def charger_depot(depot):
    """Manifestes de corpus (par nom de corpus) + labels de la taxonomie ; fichier
    illisible = avertissement, jamais un crash."""
    depot = Path(depot)
    manifestes, labels, avert = {}, {}, []
    for p in sorted((depot / "manifests" / "corpus").glob("*.yaml")):
        try:
            m = lire_yaml(p) or {}
            manifestes[m.get("corpus") or p.stem] = m
        except Exception as e:
            avert.append(f"{p} : {e}")
    try:
        tax = lire_yaml(depot / "taxonomy" / "entities.yaml") or {}
        labels = {e["id"]: e.get("label_fr") or "" for e in tax.get("entities", [])
                  if isinstance(e, dict) and "id" in e}
    except Exception as e:
        avert.append(f"taxonomy/entities.yaml : {e}")
    return manifestes, labels, avert


def donnees_corpus(manifeste, fusion):
    """Annotations du manifeste : par classe (noms de l'éval — `fusion` appliquée)
    et par zone (total train + train par classe)."""
    par_classe, par_zone = {}, {}
    for ds in (manifeste.get("datasets") or {}).values():
        z = par_zone.setdefault(ds.get("zone", "?"), {"train": 0, "classes": {}})
        for split, bloc in (ds.get("splits") or {}).items():
            for cl, n in ((bloc or {}).get("annotations") or {}).items():  # split null toléré
                c = fusion.get(cl, cl)
                pc = par_classe.setdefault(c, {"train": 0, "valid": 0, "test": 0})
                pc[split] = pc.get(split, 0) + n
                if split == "train":
                    z["train"] += n
                    z["classes"][c] = z["classes"].get(c, 0) + n
    return par_classe, par_zone


def region_dept(zone):
    """'ile_de_france/78_rambouillet' -> ('ile de france', '78') ; 'irlande/ie_sligo' -> (…, 'ie')."""
    reg, _, site = zone.partition("/")
    m = re.match(r"(\d+|ie)_", site)
    return reg.replace("_", " "), (m.group(1) if m else "?")


# --- run --------------------------------------------------------------------

def basename(chemin):
    return str(chemin).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] if chemin else None


def epoques_du_run(run):
    """Fusion metrics.csv + historiques *avant_reprise* (piège connu : metrics.csv
    seul ne contient que la dernière reprise) : une valeur val/ema_mAP_50 par
    époque, la plus récente gagne ; reprises = époques de départ des fichiers
    postérieurs au premier historique."""
    series = []
    for p in run.glob("metrics*.csv"):
        if p.name != "metrics.csv" and "avant_reprise" not in p.name:
            continue
        faites, vals = set(), {}
        try:
            with open(p, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    if not (r.get("epoch") or "").strip():
                        continue
                    ep = int(float(r["epoch"]))
                    faites.add(ep)
                    v = (r.get("val/ema_mAP_50") or "").strip()
                    if v:
                        vals[ep] = float(v)
        except Exception:
            continue
        if faites:  # historiques avant metrics.csv à départ égal
            series.append((min(faites), p.name == "metrics.csv", faites, vals))
    if not series:
        return None
    series.sort(key=lambda s: (s[0], s[1]))
    faites, vals = set(), {}
    for _, _, f, v in series:
        faites |= f
        vals.update(v)
    debut0 = series[0][0]
    return {"faites": len(faites), "validees": len(vals), "vals": vals,
            "meilleure": max(vals, key=vals.get) if vals else None,
            "reprises": sorted({s[0] for s in series[1:] if s[0] != debut0})}


def lire_run(run, avertissements):
    """params_run.yaml / config.json / metrics*.csv d'un run — absent = None, jamais deviné."""
    info = {"run": run, "params": None, "config": None, "corpus": None, "epoques": None}
    if run is None:
        return info
    for cle, nom, lecteur in (("params", "params_run.yaml", lire_yaml),
                              ("config", "config.json",
                               lambda p: json.loads(Path(p).read_text(encoding="utf-8")))):
        p = run / nom
        try:
            if p.exists():
                info[cle] = lecteur(p) or {}
        except Exception as e:
            avertissements.append(f"{p} : {e}")
    params, config = info["params"] or {}, info["config"] or {}
    src = params.get("CORPUS_DRIVE_DIR") or (config.get("dataset") or {}).get("dataset_dir")
    info["corpus"] = basename(src)
    try:
        info["epoques"] = epoques_du_run(run)
    except OSError as e:
        avertissements.append(f"{run} : metrics : {e}")
    return info


def champs_run(info):
    """Liste (libellé, valeur) du bloc « Run » : params_run.yaml sinon config.json."""
    params, config = info["params"], info["config"]
    if params is not None:
        rvt, mnt = params.get("RVT") or {}, params.get("MNT") or {}
        rp = rvt.get("params") or {}
        g = lambda k, d=params: d.get(k) if d.get(k) is not None else NT
        ft = params.get("FINETUNE_FROM")
        ft = f"{basename(Path(ft).parent)}/{basename(ft)}" if ft else NT
        sha = str(params.get("CORPUS_MANIFEST_SHA1") or "")[:8]
        corpus = basename(params.get("CORPUS_DRIVE_DIR")) or NT
        return [("source", "params_run.yaml"), ("variante", g("MODEL_VARIANT")),
                ("résolution", g("RESOLUTION")),
                ("RVT", f"{g('type', rvt)} · gsd {g('gsd_m', rp)} m · rmin {g('rmin_px', rp)} px · "
                        f"rmax {g('rmax_px', rp)} px"),
                ("MNT", f"{g('resolution', mnt)} m"),
                ("corpus", f"{corpus} (manifeste {sha})" if sha else corpus),
                ("transfert depuis", ft),
                ("poids de base", basename(params.get("BASE_WEIGHTS_DRIVE")) or NT),
                ("époques prévues", g("NUM_EPOCHS")), ("patience", g("EARLY_STOPPING_PATIENCE")),
                ("batch × accumulation", f"{g('BATCH_SIZE')} × {g('GRAD_ACCUM_STEPS')}"),
                ("learning rate", g("LEARNING_RATE")), ("lr encodeur", g("LR_ENCODER")),
                ("seed", g("SEED")), ("augmentation", g("AUG_CONFIG_NAME")),
                ("précision", g("PRECISION"))]
    if config is not None:
        mo, tr = config.get("model") or {}, config.get("training") or {}
        g = lambda k, d: d.get(k) if d.get(k) is not None else NT
        return [("source", "config.json"), ("variante", g("variant", mo)),
                ("résolution", g("resolution", mo)), ("classes", g("num_classes", mo)),
                ("dataset", basename((config.get("dataset") or {}).get("dataset_dir")) or NT),
                ("époques prévues", g("num_epochs", tr)),
                ("patience", g("early_stopping_patience", tr)),
                ("batch × accumulation", f"{g('batch_size', tr)} × {g('grad_accum_steps', tr)}"),
                ("learning rate", g("learning_rate", tr)), ("lr encodeur", g("lr_encoder", tr)),
                ("seed", g("seed", tr))]
    return []


def lignee(info, nom):
    params, prov = info["params"] or {}, (info["config"] or {}).get("provenance") or {}
    base = basename(params.get("BASE_WEIGHTS_DRIVE") or prov.get("base_weights"))
    ft = params.get("FINETUNE_FROM") or prov.get("finetune_from")
    ft = basename(Path(ft).parent) if ft else None
    if not base and not ft:
        return "non tracée"
    return " -> ".join(x for x in (base, ft, nom) if x)


def epoques_prevues(info):
    params, tr = info["params"] or {}, (info["config"] or {}).get("training") or {}
    return params.get("NUM_EPOCHS") or tr.get("num_epochs")


# --- plugin -----------------------------------------------------------------

def lire_carte(plugin, nom, avertissements):
    """model_card.yaml du modèle `nom` dans un dossier data/models OU dans un zip de
    plugin publié (archeologia.<version>.zip : entrée */data/models/<nom>/model_card.yaml)
    — le zip du dépôt QGIS est ce que voient réellement les utilisateurs."""
    plugin = Path(plugin)
    if plugin.suffix.lower() == ".zip":
        import io
        import zipfile
        import yaml
        suffixe = f"data/models/{nom}/model_card.yaml"
        try:
            with zipfile.ZipFile(plugin) as z:
                entree = next((n for n in z.namelist() if n.replace("\\", "/").endswith(suffixe)), None)
                if entree is None:
                    return None
                return yaml.safe_load(io.TextIOWrapper(z.open(entree), encoding="utf-8")) or {}
        except Exception as e:
            avertissements.append(f"{plugin} ({nom}) : {e}")
            return None  # carte illisible = pas de badge « installé » trompeur
    p = plugin / nom / "model_card.yaml"
    if not p.exists():
        return None
    try:
        return lire_yaml(p) or {}
    except Exception as e:
        avertissements.append(f"{p} : {e}")
        return {}


def badge_plugin(carte):
    if carte is None:
        return "<span class='badge off'>non installé</span>"
    return (f"<span class='badge on'>installé · {html.escape(str(carte.get('status') or '?'))} · "
            f"{html.escape(str(carte.get('version') or '?'))}</span>")


def comparer_seuils(carte, l):
    """[(classe|global, verdict)] : seuil déployé vs seuil F1-max mesuré."""
    th = carte.get("thresholds") or {}
    prov = th.get("seuils_provenance")

    def verdict(deploye, mesure):
        if deploye is None or mesure is None:
            return NT
        if abs(float(deploye) - float(mesure)) < 1e-6:
            return f"{fmt_f(deploye)} = F1-max"
        just = f"justifié : {prov}" if prov else "non justifié"
        return f"{fmt_f(deploye)} ≠ F1-max {fmt_f(mesure)} ({just})"

    lignes = [("global", verdict(th.get("confidence_default"), l["global"].get("seuil_f1max")))]
    for c, s in (th.get("confidence_per_class") or {}).items():
        lignes.append((c, verdict(s, (l["par_classe"].get(c) or {}).get("seuil_f1max"))))
    return lignes


# --- fiche ------------------------------------------------------------------

def fiche(l, info, manifeste, labels, carte, seuil_fragile, min_train, racine):
    """(html de la fiche repliée, alertes de la fiche)."""
    e = html.escape
    nom, alertes = l["nom"], []
    fusion = l["fusion"]
    pc_corpus, pz_corpus = donnees_corpus(manifeste, fusion) if manifeste else ({}, {})
    parts = [f"<details class='fiche'><summary>fiche — {e(nom)}</summary>"]
    lien = os.path.relpath(l["dossier"], racine).replace("\\", "/")
    parts.append(f"<p class='note'>évaluation : <a href='{e(lien)}/metriques_eval.json'>{e(lien)}</a>"
                 f" ({e(l['date'][:10])}) · run : {e(info['run'].name) if info['run'] else NT}"
                 f" · corpus : {e(info['corpus'] or NT)}"
                 + ("" if manifeste else " · <b>données non tracées</b> (aucun manifeste de corpus)")
                 + "</p>")

    # 1. données par classe
    parts.append("<h4>Données par classe</h4><table><tr><th>classe</th><th>train</th>"
                 "<th>part du train</th><th>valid</th><th>test</th><th>mesuré n_gt</th>"
                 "<th>seuil F1-max</th><th>F1</th></tr>")
    tot_train = sum(v["train"] for v in pc_corpus.values()) or None
    tot = {"train": 0, "valid": 0, "test": 0, "n_gt": 0}
    for c in sorted(set(l["par_classe"]) | set(pc_corpus)):
        m, d = l["par_classe"].get(c) or {}, pc_corpus.get(c)
        ngt = m.get("n_gt")
        lab = f" ({e(labels[c])})" if labels.get(c) else ""
        if d:
            attendu = d["valid"] + d["test"]
            ok = ("<span class='ok'>✓</span>" if ngt == attendu
                  else f"<span class='ko'>≠ {fmt_n(attendu)}</span>") if ngt is not None else ""
            cols = (f"<td>{fmt_n(d['train'])}</td><td>{fmt_pct(d['train'] / tot_train) if tot_train else '—'}</td>"
                    f"<td>{fmt_n(d['valid'])}</td><td>{fmt_n(d['test'])}</td>")
            for k in ("train", "valid", "test"):
                tot[k] += d[k]
        else:
            ok, cols = "", "<td>—</td><td>—</td><td>—</td><td>—</td>"
        tot["n_gt"] += ngt or 0
        parts.append(f"<tr><td>{e(c)}{lab}</td>{cols}<td>{fmt_n(ngt)} {ok}</td>"
                     f"<td>{fmt_f(m.get('seuil_f1max'))}</td>{td_t(m.get('F1'))}</tr>")
    parts.append(f"<tr class='total'><td>total</td><td>{fmt_n(tot['train']) if pc_corpus else '—'}</td>"
                 f"<td>{'100' + FIN + '%' if pc_corpus else '—'}</td>"
                 f"<td>{fmt_n(tot['valid']) if pc_corpus else '—'}</td>"
                 f"<td>{fmt_n(tot['test']) if pc_corpus else '—'}</td>"
                 f"<td>{fmt_n(tot['n_gt'])}</td>"
                 f"<td>{fmt_f(l['global'].get('seuil_f1max'))}</td>{td_t(l['global'].get('F1'))}</tr></table>")

    # 2. zones
    pz = l["par_zone"]
    tot_ngt = sum((v.get("n_gt") or 0) for v in pz.values()) or None
    tot_train_z = sum(v["train"] for v in pz_corpus.values()) or None
    zones = sorted(set(pz) | set(pz_corpus), key=lambda z: (region_dept(z)[0], z))
    parts.append("<h4>Zones</h4><table><tr><th>zone</th><th>région / dépt</th>"
                 "<th>annotations train</th><th>part du train</th><th>n_gt</th><th>part</th>"
                 "<th>rappel</th><th>précision</th></tr>")
    fragiles = {}
    for z in zones:
        m, d = pz.get(z) or {}, pz_corpus.get(z)
        reg, dep = region_dept(z)
        ngt = m.get("n_gt")
        etiq = ""
        if ngt is not None and ngt < seuil_fragile:
            etiq, fragiles[z] = " <span class='tag'>fragile</span>", ngt
        parts.append(f"<tr><td>{e(z)}{etiq}</td><td>{e(reg)} / {e(dep)}</td>"
                     f"<td>{fmt_n(d['train']) if d else '—'}</td>"
                     f"<td>{fmt_pct(d['train'] / tot_train_z) if d and tot_train_z else '—'}</td>"
                     f"<td>{fmt_n(ngt)}</td><td>{fmt_pct(ngt / tot_ngt) if ngt is not None and tot_ngt else '—'}</td>"
                     f"{td_t(m.get('R'))}{td_t(m.get('P'))}</tr>")
    parts.append("</table>")
    if fragiles:
        alertes.append(("fragile", fragiles))

    # rares en train par zone (manifeste, indépendant de par_zone_classe)
    rares = {}
    classes_corpus = [fusion.get(c, c) for c in (manifeste.get("classes") or [])] if manifeste else []
    for z, d in pz_corpus.items():
        for c in sorted(set(classes_corpus) | set(d["classes"])):
            n = d["classes"].get(c, 0)
            if n < min_train:
                rares[(z, c)] = n
    if rares:
        alertes.append(("rare", rares))

    pzc = l.get("par_zone_classe")
    if pzc:
        parts.append("<h4>Zone × classe</h4><table><tr><th>zone</th><th>classe</th><th>n_gt</th>"
                     "<th>R</th><th>R au seuil de la classe</th><th>P</th><th>R max</th><th>FP</th></tr>")
        decroche = {}
        for z in sorted(pzc, key=lambda z: (region_dept(z)[0], z)):
            for c in sorted(pzc[z]):
                d = pzc[z][c] or {}
                n = d.get("n_gt") or 0
                tags = []
                if n == 0:
                    tags.append("absente")
                else:
                    if n < seuil_fragile:
                        tags.append("fragile")
                    rmax = d.get("R_max")
                    if rmax is not None and rmax < 0.5 and n >= seuil_fragile:
                        tags.append("décroche")
                        decroche[(z, c)] = rmax
                if pz_corpus and (pz_corpus.get(z) or {}).get("classes", {}).get(c, 0) < min_train:
                    tags.append("rare en train")
                etiq = "".join(f" <span class='tag'>{t}</span>" for t in tags)
                if n == 0:
                    vals = "<td>—</td>" * 4
                else:
                    vals = td_t(d.get("R")) + td_t(d.get("R_seuil_classe")) + td_t(d.get("P")) + td_t(d.get("R_max"))
                parts.append(f"<tr><td>{e(z)}</td><td>{e(c)}{etiq}</td><td>{fmt_n(n)}</td>{vals}"
                             f"<td>{fmt_n(d.get('fp'))}</td></tr>")
        parts.append("</table>")
        if decroche:
            alertes.append(("decroche", decroche))
    else:
        parts.append("<p class='note'>détail zone × classe : relancer "
                     "tools/completer_metriques_eval.py sur cette évaluation</p>")

    # 3. run
    parts.append("<h4>Run</h4>")
    champs = champs_run(info)
    if info["run"] is None:
        parts.append(f"<p class='note'>run {NT} (aucun dossier runs/training/{e(nom)})</p>")
    elif not champs:
        parts.append(f"<p class='note'>paramètres {NT}s (ni params_run.yaml ni config.json)</p>")
    else:
        parts.append("<dl>" + "".join(f"<dt>{e(k)}</dt><dd>{e(str(v))}</dd>" for k, v in champs) + "</dl>")
    ep, prevues = info["epoques"], epoques_prevues(info)
    if ep:
        b = ep["meilleure"]
        parts.append(f"<p>époques : <b>{ep['faites']} faites ({ep['validees']} validées)</b> sur "
                     f"{prevues if prevues is not None else NT} prévues · meilleure époque "
                     f"{b if b is not None else '—'}"
                     + (f" (mAP50 EMA {fmt_f(ep['vals'][b])})" if b is not None else "")
                     + " · reprises : " + (", ".join(str(r) for r in ep["reprises"]) or "aucune") + "</p>")
    else:
        parts.append(f"<p class='note'>époques : {NT}es (aucun metrics.csv)</p>")

    # 4. apprentissage
    parts.append("<h4>Apprentissage</h4>")
    if ep:
        parts.append(courbe_epoques(ep["vals"], prevues, ep["meilleure"], ep["reprises"]))
        parts.append("<p class='note'>mAP50 EMA (val) par époque, abscisse = époques prévues, "
                     "point = meilleure époque, pointillés = reprises.</p>")
    else:
        parts.append(f"<p class='note'>{NT}</p>")

    # 5. plugin
    if carte is not None or l.get("plugin_actif"):
        parts.append("<h4>Plugin</h4>" + badge_plugin(carte))
        if carte is not None:
            seuils = comparer_seuils(carte, l)
            parts.append("<ul>" + "".join(f"<li>seuil {e(k)} : {e(v)}</li>" for k, v in seuils) + "</ul>")
            if any("≠" in v for _, v in seuils):
                alertes.append(("seuil", [f"{k} : {'non justifié' if 'non justifié' in v else 'justifié'}"
                                          for k, v in seuils if "≠" in v]))
        parts.append(f"<p>lignée : {e(lignee(info, nom))}</p>")
    parts.append("</details>")
    return "\n".join(parts), alertes


def ligne_classes(l, manifeste):
    e = html.escape
    sources = {}
    for src, dst in l["fusion"].items():
        sources.setdefault(dst, []).append(src)
    classes = [c + (f" (= {' + '.join(sources[c])} fusionné{'s' if len(sources[c]) > 1 else ''})"
                    if c in sources else "") for c in sorted(l["par_classe"])]
    txt = "classes : " + ", ".join(classes)
    fus = (manifeste or {}).get("fusions") or {}
    if fus:
        txt += " · fusions corpus : " + ", ".join(f"{s} -> {d}" for s, d in fus.items())
    return e(txt)


# --- page -------------------------------------------------------------------

def construire(racine, depot=None, plugin=None, seuil_fragile=30, min_train=100):
    """La page est destinée à <racine>/index.html : liens relatifs à la RACINE
    (un --out de staging ne change pas les liens)."""
    racine = Path(racine)
    evals, sans_mesure, avertissements = collecter(racine)
    manifestes, labels = {}, {}
    if depot:
        manifestes, labels, av = charger_depot(depot)
        avertissements += av
    runs_par_nom = {r.name: r for r in sans_mesure}
    runs_par_nom.update({p.parent.name: p.parent for p, _ in evals})

    familles = {}
    for p, data in evals:
        fam = famille_de(p, racine)
        # PROVISOIRE.txt à côté de metriques_eval.json = modèle voué au remplacement
        # (data retravaillées, réentraînement prévu) : gardé dans le tableau avec
        # son étiquette, EXCLU de l'évolution par classe. Le marqueur meurt avec
        # le run quand il est supprimé.
        provisoire = (p.parent / "PROVISOIRE.txt").exists()
        for i, (nom, m) in enumerate((data.get("modeles") or {}).items()):
            familles.setdefault(fam, []).append({
                "nom": nom, "date": data.get("genere_le", ""), "tache": data.get("tache", "?"),
                "global": m.get("global", {}), "par_classe": m.get("par_classe", {}),
                "par_zone": m.get("par_zone") or {}, "par_zone_classe": m.get("par_zone_classe"),
                "dossier": p.parent, "provisoire": provisoire,
                "fusion": data.get("fusion") or {}, "premier": i == 0,
                "plugin_actif": plugin is not None,
            })
    for lignes in familles.values():
        lignes.sort(key=lambda l: (l["date"], l["nom"]))

    e = html.escape
    parties = [
        "<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'>",
        "<title>Modèles — métriques d'évaluation</title><style>",
        "body{font-family:Segoe UI,sans-serif;margin:1.5em;background:#fafafa;color:#222}",
        "table{border-collapse:collapse;margin:.5em 0 1em}",
        "th,td{border:1px solid #ccc;padding:3px 8px;font-size:13px;text-align:left;",
        "font-variant-numeric:tabular-nums}",
        "th{background:#eee}details{margin-bottom:1.2em}summary{font-size:17px;",
        "font-weight:600;cursor:pointer}.val{font-size:11px;color:#555}",
        ".note{color:#777;font-size:12px}.warn{color:#a33;font-size:12px}",
        "h3{margin:.6em 0 .2em;font-size:14px}h4{margin:.8em 0 .2em;font-size:13px}",
        "details.fiche{margin:.3em 0 .3em 1em;padding:.2em .8em;border-left:3px solid #ddd}",
        "details.fiche summary{font-size:14px;font-weight:500}",
        ".ok{color:#2a7d2a;font-weight:700}.ko{color:#c1272d;font-weight:700}",
        ".tag{font-size:10px;padding:0 4px;border-radius:3px;background:#f3e4c0;color:#6b4c00}",
        ".badge{font-size:11px;padding:1px 6px;border-radius:3px}.badge.on{background:#d8ecd8;color:#1f5e1f}",
        ".badge.off{background:#eee;color:#666}",
        "dl{display:grid;grid-template-columns:max-content 1fr;gap:1px 12px;font-size:12px;margin:.3em 0}",
        "dt{color:#666}dd{margin:0}tr.total td{font-weight:600;background:#f4f4f4}",
        ".alertes{background:#fff7e6;border:1px solid #f0d9a8;padding:.3em .8em;margin:.4em 0;font-size:12px}",
        ".alertes ul{margin:.2em 0 .2em 1.2em}",
        "</style></head><body>",
        "<h1>Modèles — métriques d'évaluation</h1>",
        f"<p class='note'>Généré le {datetime.now().isoformat(timespec='seconds')} par "
        f"tools/tableau_modeles.py — {len(evals)} évaluation(s), "
        f"{sum(len(v) for v in familles.values())} ligne(s) modèle, "
        f"{len(sans_mesure)} run(s) sans mesure. Source : metriques_eval.json "
        "(seuils F1-max mesurés — jamais de seuil fixe). "
        + (f"Manifestes de corpus : {len(manifestes)} (dépôt {e(Path(depot).name)}). " if depot else "")
        + (f"Plugin : {e(Path(plugin).name)}. " if plugin else "")
        + "<a href='../data/data_regions_v2/index.html'>→ index data_regions_v2</a></p>",
    ]
    avert_idx = len(parties)  # les avertissements de lecture des fiches s'ajoutent ici

    infos_runs, cartes, zones_famille = {}, {}, {}
    for fam in sorted(familles):
        lignes = familles[fam]
        bloc = [f"<details open><summary>{e(fam)}</summary>"]

        # fiches : une par nom de modèle, l'éval la plus récente fait foi
        derniers_tous = {}
        for l in lignes:
            derniers_tous[l["nom"]] = l
        fiches, alertes = [], []
        entetes = []
        for nom, l in derniers_tous.items():
            run = runs_par_nom.get(nom) or (l["dossier"].parent if l["premier"] else None)
            if run not in infos_runs:
                infos_runs[run] = lire_run(run, avertissements)
            info = infos_runs[run]
            manifeste = manifestes.get(info["corpus"]) if info["corpus"] else None
            carte = None
            if plugin is not None:
                if nom not in cartes:
                    cartes[nom] = lire_carte(plugin, nom, avertissements)
                carte = cartes[nom]
            l["carte"] = carte
            h, al = fiche(l, info, manifeste, labels, carte, seuil_fragile, min_train, racine)
            fiches.append(h)
            alertes.append((nom, info, manifeste, al))
            entetes.append(ligne_classes(l, manifeste))
            zf = zones_famille.setdefault(fam, set())
            zf.update(l["par_zone"])
            if manifeste:
                zf.update(ds.get("zone", "?") for ds in (manifeste.get("datasets") or {}).values())
        for t in dict.fromkeys(entetes):
            bloc.append(f"<p class='note'>{t}</p>")

        # à surveiller
        surveiller = []
        fragiles, rares, incomplets, sans = {}, {}, {}, []
        for nom, info, manifeste, al in alertes:
            for genre, d in al:
                if genre == "fragile":
                    for z, n in d.items():
                        fragiles[z] = min(n, fragiles.get(z, n))
                elif genre == "rare":
                    rares.setdefault(info["corpus"], {}).update(d)
                elif genre == "decroche":
                    pass  # décision utilisateur 2026-09-03 : reste une étiquette de ligne, pas une alerte
                elif genre == "seuil":
                    surveiller.append(f"{e(nom)} — seuil déployé ≠ F1-max mesuré ({', '.join(e(x) for x in d)}), "
                                      "cf. fiche")
            if manifeste:
                manque = [k for k in ("genere_le", "gsd_m", "rvt") if not manifeste.get(k)]
                if manque:
                    incomplets[info["corpus"]] = manque
            if derniers_tous[nom]["provisoire"]:
                surveiller.append(f"{e(nom)} — modèle provisoire (voué au remplacement)")
        if fragiles:
            surveiller.insert(0, f"zones fragiles (n_gt &lt; {seuil_fragile}) : "
                              + ", ".join(f"{e(z)} ({fmt_n(n)})" for z, n in sorted(fragiles.items())))
        # classes rares en train : étiquette de ligne seulement (décision utilisateur 2026-09-03)
        for run in sans_mesure:
            if famille_de(run, racine) == fam and any(
                    (run / f).exists() for f in ("checkpoint_best_ema.pth", "checkpoint_best_total.pth")):
                sans.append(run.name)
        if sans:
            surveiller.append("runs entraînés sans mesure (checkpoint sans evaluation/metriques_eval.json) : "
                              + ", ".join(e(r) for r in sans))
        for corpus, manque in incomplets.items():
            surveiller.append(f"manifeste de corpus {e(corpus)} incomplet ({', '.join(manque)} manquant)")
        if surveiller:
            bloc.append("<div class='alertes'><h3>À surveiller</h3><ul>"
                        + "".join(f"<li>{s}</li>" for s in surveiller) + "</ul></div>")

        # tableau des évaluations
        bloc.append("<table><tr><th>modèle</th><th>date éval</th><th>tâche</th>"
                    "<th>seuil F1-max</th><th>F1</th><th>P</th><th>R</th><th>AP50</th>"
                    "<th>n_gt</th><th>IoU méd.</th><th>planches</th>"
                    + ("<th>plugin</th>" if plugin is not None else "") + "</tr>")
        for l in lignes:
            g = l["global"]
            lien = os.path.relpath(l["dossier"], racine).replace("\\", "/")
            etiquette = " <span class='warn'>[provisoire]</span>" if l["provisoire"] else ""
            bloc.append(
                f"<tr><td>{e(l['nom'])}{etiquette}</td><td>{e(l['date'][:10])}</td>"
                f"<td>{e(l['tache'])}</td><td>{cell(g.get('seuil_f1max'))}</td>"
                f"{td_t(g.get('F1'))}{td_t(g.get('P'))}{td_t(g.get('R'))}"
                f"<td>{cell(g.get('AP50'))}</td>"
                f"<td>{cell(g.get('n_gt'))}</td><td>{cell(g.get('iou_median'))}</td>"
                f"<td><a href='{e(lien)}/courbes_seuils_pr.png'>courbes</a></td>"
                + (f"<td>{badge_plugin(cartes.get(l['nom']))}</td>" if plugin is not None else "")
                + "</tr>")
        bloc.append("</table>")

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
            bloc.append("<h3>Évolution par classe (versions : "
                        + " → ".join(e(l["nom"]) for l in serie) + ")</h3>")
            bloc.append("<table><tr><th>classe</th><th>F1</th><th>AP50</th>"
                        "<th>seuil F1-max</th><th>n_gt (dernier)</th><th>écart F1</th></tr>")
            for cl in classes:
                blocs = [l["par_classe"].get(cl) for l in serie]
                col = lambda k: [b.get(k) if b else None for b in blocs]
                ngt = next((b.get("n_gt") for b in reversed(blocs) if b), None)
                f1s = [v for v in col("F1") if v is not None]
                ecart = f"{f1s[-1] - f1s[-2]:+.3f}".replace(".", ",") if len(f1s) >= 2 else "—"
                bloc.append(f"<tr><td>{e(cl)}</td><td>{sparkline(col('F1'))}</td>"
                            f"<td>{sparkline(col('AP50'))}</td>"
                            f"<td>{sparkline(col('seuil_f1max'))}</td>"
                            f"<td>{fmt_n(ngt)}</td><td>{ecart}</td></tr>")
            bloc.append("</table>")

        bloc.extend(fiches)
        bloc.append("</details>")
        parties.extend(bloc)

    # pied de page : couverture des zones + glossaire
    zones_manifestes = {ds.get("zone", "?") for m in manifestes.values()
                        for ds in (m.get("datasets") or {}).values()}
    fams = sorted(familles)
    if zones_manifestes and fams:
        parties.append("<h2>Couverture des zones</h2><p class='note'>zones citées par les manifestes de "
                       "corpus × familles (● = zone dans le corpus ou l'éval d'un modèle de la famille)</p>")
        parties.append("<table><tr><th>région</th><th>zone</th>" + "".join(f"<th>{e(f)}</th>" for f in fams) + "</tr>")
        for z in sorted(zones_manifestes, key=lambda z: (region_dept(z)[0], z)):
            parties.append(f"<tr><td>{e(region_dept(z)[0])}</td><td>{e(z)}</td>"
                           + "".join(f"<td>{'●' if z in zones_famille.get(f, ()) else ''}</td>" for f in fams)
                           + "</tr>")
        parties.append("</table>")
    parties.append("<h2>Méthode et glossaire</h2><ul class='note'>"
                   "<li><b>n_gt</b> : objets de vérité terrain mesurés — splits valid + test de l'éval "
                   "(jamais train).</li>"
                   "<li><b>seuil F1-max</b> : balayage de confiance 0,05–0,95 par pas de 0,005, seuil qui "
                   "maximise F1 ; source unique des seuils du model_card du plugin.</li>"
                   "<li><b>F1 / P / R</b> : au seuil F1-max ; P = précision, R = rappel ; appariement "
                   "IoU masque ou boîte ≥ 0,5.</li>"
                   "<li><b>AP50</b> : aire sous la courbe précision/rappel à IoU ≥ 0,5.</li>"
                   "<li><b>IoU médian</b> : IoU médian des appariements vrais positifs.</li>"
                   "<li><b>split spatial</b> : blocs de 2 km, 70/20/10 train/valid/test, seed 42, "
                   "tracé dans split_manifest.yaml et jamais re-tiré.</li>"
                   "<li><b>provisoire</b> : modèle voué au remplacement (PROVISOIRE.txt à côté de "
                   "metriques_eval.json), étiqueté au tableau, exclu de l'évolution par classe.</li>"
                   "<li><b>sans mesure</b> : run entraîné (checkpoint) sans evaluation/metriques_eval.json ; "
                   "seulement compté.</li>"
                   "<li><b>cellules teintées</b> : P, R, F1 et R max sur fond bleu unique d'intensité "
                   "proportionnelle à la valeur (blanc = 0, plein = 1).</li></ul>")

    for msg in avertissements:
        parties.insert(avert_idx, f"<p class='warn'>⚠ {e(str(msg))}</p>")
        avert_idx += 1
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("racine", help="racine à scanner (ex. G:\\...\\model-training)")
    ap.add_argument("--out", default=None, help="HTML de sortie (défaut : <racine>\\index.html)")
    ap.add_argument("--depot", default=str(Path(__file__).resolve().parents[1]),
                    help="racine du dépôt training-models (manifests/corpus, taxonomy) ; "
                         "défaut : le dépôt de ce script")
    ap.add_argument("--plugin", default=None,
                    help="dossier data/models d'un plugin (model_card.yaml) OU zip de plugin "
                         "publié (archeologia.<version>.zip) ; sans l'option, pas de colonne "
                         "ni de bloc plugin")
    ap.add_argument("--seuil-fragile", type=int, default=30,
                    help="n_gt en dessous duquel une zone (ou zone × classe) est « fragile »")
    ap.add_argument("--min-train", type=int, default=100,
                    help="annotations train par zone × classe en dessous desquelles la classe est « rare »")
    ap.add_argument("--registre", default=None,
                    help="modeles.yaml de data_regions_v2 à mettre à jour (merge)")
    a = ap.parse_args()
    sortie = a.out or os.path.join(a.racine, "index.html")
    page = construire(a.racine, depot=a.depot, plugin=a.plugin,
                      seuil_fragile=a.seuil_fragile, min_train=a.min_train)
    with open(sortie, "w", encoding="utf-8") as f:
        f.write(page)
    assert "<html" in open(sortie, encoding="utf-8").read(200), "relecture invalide"
    print("dashboard ->", sortie)
    if a.registre:
        maj_registre(a.racine, a.registre)


if __name__ == "__main__":
    main()
