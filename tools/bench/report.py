"""Génère le rapport HTML du banc depuis les JSON produits.

    python -m tools.bench.report --out D:\\pipeline_results\\bench
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Optional

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--bd:#e2e2e2;--acc:#0b6b3a;--bad:#a3301f;--z:#f7f7f7}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8e8e8;--mut:#9aa0a6;--bd:#2c3138;--acc:#4ade80;--bad:#f87171;--z:#1b1e24}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1rem;background:var(--bg);color:var(--fg);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .2rem} h2{font-size:1.2rem;margin:2.2rem 0 .6rem;
 padding-bottom:.3rem;border-bottom:2px solid var(--bd)} h3{font-size:1rem;margin:1.4rem 0 .4rem}
.sub{color:var(--mut);margin:0 0 1.5rem}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.8rem;margin:1rem 0}
.kpi div{border:1px solid var(--bd);border-radius:8px;padding:.8rem 1rem;background:var(--z)}
.kpi b{display:block;font-size:1.5rem;line-height:1.2} .kpi span{color:var(--mut);font-size:.82rem}
.tw{overflow-x:auto;margin:.6rem 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:520px}
th,td{padding:.4rem .55rem;text-align:right;border-bottom:1px solid var(--bd);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{background:var(--z);position:sticky;top:0;font-weight:600}
tbody tr:hover{background:var(--z)}
.pos{color:var(--acc);font-weight:600} .neg{color:var(--bad)}
.sig{font-weight:700}
code{background:var(--z);padding:.1rem .3rem;border-radius:3px;font-size:.9em}
.note{border-left:3px solid var(--acc);padding:.5rem .9rem;background:var(--z);margin:.8rem 0;
 border-radius:0 6px 6px 0}
.warn{border-left-color:var(--bad)}
footer{margin-top:3rem;color:var(--mut);font-size:.85rem;border-top:1px solid var(--bd);padding-top:1rem}
"""


def f(x, n=4, defaut="—"):
    try:
        if x is None or x != x:
            return defaut
        return f"{x:.{n}f}"
    except Exception:
        return defaut


def tableau(entetes: List[str], lignes: List[List[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in entetes)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in l) + "</tr>" for l in lignes)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def charger(racine: Path) -> dict:
    d: dict = {"runs": {}, "bootstrap": {}}
    for r in sorted((racine / "runs").glob("*/resultats.json")) if (racine / "runs").exists() else []:
        d["runs"][r.parent.name] = json.loads(r.read_text(encoding="utf-8"))
    for b in sorted((racine / "runs").glob("*/bootstrap.json")) if (racine / "runs").exists() else []:
        d["bootstrap"][b.parent.name] = json.loads(b.read_text(encoding="utf-8"))
    for nom, fichier in (("e0", "e0_plafond_rappel.json"), ("niveau_b", "niveau_b.json")):
        p = racine / fichier
        if p.exists():
            d[nom] = json.loads(p.read_text(encoding="utf-8"))
    p = Path(__file__).with_name("defauts.json")
    if p.exists():
        d["defauts"] = json.loads(p.read_text(encoding="utf-8"))
    return d


def section_e0(e0: dict) -> str:
    if not e0:
        return ""
    lignes = []
    for k, g in e0["resultats"].items():
        regle, seuil = k.split("@")
        lignes.append([f"<code>{regle}</code> @ {seuil}", f(g["completude"], 3),
                       f(g["correction"], 3), f(g["f1_len"], 3),
                       f(g["len_pred_m"] / 1000, 1)])
    return f"""<h2>1. Diagnostic — plafond de rappel</h2>
<p>La question qui commande tout le reste&nbsp;: la perte vient-elle de ce que le modèle
<em>ne propose jamais</em> la structure (échec de représentation — seul un ré-entraînement
aide), ou de ce qu'il la propose avec un score trop bas (échec de calibration — un seuil
suffit)&nbsp;? On fait tomber le seuil jusqu'au plancher du cache
({e0.get('plancher_cache')}) et on regarde la complétude atteignable.</p>
{tableau(["règle @ seuil", "complétude", "correction", "F1_len", "longueur prédite (km)"], lignes)}
<p class="sub">{e0['n_images']} images annotées, τ = {e0['tau_m']} m.
<code>objectness</code> = 1 − Π(1 − p_c) au lieu de max_c p_c&nbsp;: teste si le modèle
répartit son score entre classes ambiguës au point de ne jamais franchir le seuil.</p>"""


def section_run(nom: str, res: dict, boot: Optional[dict]) -> str:
    base = res.get("base")
    if not base:
        return ""
    deltas = {}
    if boot:
        deltas = {r["config"]: r for r in boot["resultats"]}
    lignes = []
    for cfg, g in sorted(res.items(), key=lambda kv: -(kv[1].get("f1_len") or 0)):
        b = deltas.get(cfg)
        d = b["delta"] if b else (g["f1_len"] - base["f1_len"])
        cls = "pos" if d > 0 else ("neg" if d < 0 else "")
        ic = f'{b["ic95"][0]:+.4f} … {b["ic95"][1]:+.4f}' if b else "—"
        sig = '<span class="sig">✓</span>' if b and b["significatif"] else ""
        nom_cfg = f"<code>{html.escape(cfg)}</code>" + (" (référence)" if cfg == "base" else "")
        lignes.append([nom_cfg, f(g["f1_len"]), f(g["completude"], 3), f(g["correction"], 3),
                       f'<span class="{cls}">{d:+.4f}</span>', ic, sig,
                       f(g.get("polygones_par_km2"), 1), f(g.get("fragmentation"), 2)])
    n = base.get("n_images")
    return f"""<h3>Run <code>{html.escape(nom)}</code> — {n} tuiles</h3>
{tableau(["config", "F1_len", "complétude", "correction", "ΔF1_len", "IC95 apparié",
          "signif.", "polygones/km²", "fragm."], lignes)}"""


def section_defauts(d: dict) -> str:
    if not d:
        return ""
    blocs = []
    for x in d["defauts"]:
        cls = "note warn" if x["gravite"] == "majeur" else "note"
        blocs.append(f"""<div class="{cls}">
<b>{html.escape(x['titre'])}</b> — <code>{html.escape(x['fichier'])}:{x['ligne']}</code>
· gravité <i>{x['gravite']}</i><br>
<code>{html.escape(x['code'])}</code>
<p style="margin:.5rem 0 .3rem">{html.escape(x['constat'])}</p>
<p style="margin:.3rem 0"><b>Mesure&nbsp;:</b> {html.escape(x['mesure'])}</p>
<p style="margin:.3rem 0 0"><b>Correctif&nbsp;:</b> {html.escape(x['correctif'])}</p></div>""")
    return ("<h2>Défauts mesurés dans la chaîne</h2><p>Chaque entrée porte le chiffre qui "
            "l'établit, pas une impression de lecture de code.</p>" + "".join(blocs))


def section_b(nb: dict) -> str:
    if not nb:
        return ""
    lignes = []
    base = nb.get("base", {})
    for cfg, g in sorted(nb.items(), key=lambda kv: -(kv[1].get("f1_len") or 0)):
        d = g["f1_len"] - (base.get("f1_len") or 0)
        cls = "pos" if d > 0 else ("neg" if d < 0 else "")
        lignes.append([f"<code>{html.escape(cfg)}</code>", f(g["f1_len"]),
                       f(g["completude"], 3), f(g["correction"], 3),
                       f'<span class="{cls}">{d:+.4f}</span>',
                       f(g.get("polygones_par_km2"), 1), f(g.get("fragmentation"), 2)])
    return f"""<h2>3. Niveau B — mosaïques géoréférencées</h2>
<p>Ce que le niveau tuile ne peut pas mesurer&nbsp;: fusion inter-fenêtres sur du contenu
réellement différent, et tout le post-traitement géographique. La vérité terrain vient
des GPKG v2 recalés — ce sont des <em>lignes</em>, donc rasterisées à 1&nbsp;px elles
sont déjà la ligne de centre, sans approximation par squelettisation.</p>
{tableau(["config", "F1_len", "complétude", "correction", "ΔF1_len",
          "polygones/km²", "fragm."], lignes)}"""


def construire(racine: Path, titre: str = "Banc d'inférence — lineaires_seg_v2_1") -> str:
    d = charger(racine)
    # Les runs ne portent PAS sur le même échantillon (balayage exploratoire sur 390
    # tuiles, départage sur 1558) : comparer leurs F1 entre eux n'aurait pas de sens.
    # Le résumé se lit donc sur le run le plus large, et uniquement sur lui.
    principal = max(d["runs"].items(),
                    key=lambda kv: max((g.get("n_images") or 0) for g in kv[1].values()),
                    default=(None, {}))
    base = principal[1].get("base")
    meilleur = None
    for cfg, g in principal[1].items():
        if meilleur is None or (g.get("f1_len") or 0) > (meilleur[1].get("f1_len") or 0):
            meilleur = (cfg, g)

    kpi = ""
    if base and meilleur:
        gain = meilleur[1]["f1_len"] - base["f1_len"]
        kpi = f"""<div class="kpi">
<div><span>F1 longueur — configuration actuelle du plugin</span><b>{f(base['f1_len'],3)}</b></div>
<div><span>F1 longueur — meilleure configuration</span><b>{f(meilleur[1]['f1_len'],3)}</b></div>
<div><span>gain absolu</span><b class="{'pos' if gain>0 else 'neg'}">{gain:+.3f}</b></div>
<div><span>polygones faux/km² — actuel → meilleur</span>
<b>{f(base.get('polygones_par_km2'),0)} → {f(meilleur[1].get('polygones_par_km2'),0)}</b></div>
</div>"""

    runs = "".join(section_run(n, r, d["bootstrap"].get(n)) for n, r in sorted(d["runs"].items()))
    return f"""<title>{html.escape(titre)}</title><style>{CSS}</style>
<main>
<h1>{html.escape(titre)}</h1>
<p class="sub">Mesure de la chaîne d'inférence RF-DETR-Seg telle qu'elle tourne dans le
plugin QGIS, sur des tuiles jamais vues à l'entraînement (blocs spatiaux disjoints de 2 km).</p>
{kpi}
<div class="note"><b>Métrique d'arbitrage — F1 longueur @ 5 m, toutes classes confondues.</b>
La vérité terrain est un buffer de 7 m <em>arbitraire</em> autour d'une ligne digitalisée :
un décalage latéral de 2,35 m suffit à faire tomber l'IoU de masque sous 0,5, et une
rotation de 7,5° aussi. Toute métrique fondée sur l'IoU note donc surtout l'erreur
résiduelle de digitalisation et la largeur fabriquée du buffer. La métrique en longueur
(complétude = mètres de linéaire retrouvés, correction = mètres tracés à bon escient) est
invariante à la largeur et à la fragmentation par la grille de tuilage.</div>
{section_e0(d.get('e0'))}
{section_defauts(d.get('defauts'))}
<h2>2. Niveau A — un axe à la fois</h2>
<p>Chaque configuration ne bouge qu'un paramètre depuis la configuration réelle du plugin.
La colonne IC95 est un bootstrap <em>apparié</em> par tuile : on rééchantillonne les
tuiles une fois par itération et on recalcule les deux configurations sur le même tirage.
Une coche signale un intervalle qui ne contient pas zéro.</p>
<div class="note"><b>Les deux runs ne portent pas sur le même échantillon</b> —
<code>e2_un_axe</code> explore 42 axes sur 390 tuiles (échantillon à pas constant, donc
proportionnel par zone), <code>e4_finalistes</code> tranche le seul axe qui bougeait sur
les 1558 tuiles. Les F1 ne se comparent qu'à l'intérieur d'un run.</div>
<div class="note"><b>Pourquoi tant d'axes sont exactement à zéro.</b> La métrique en
longueur est calculée sur l'<em>union</em> des masques, toutes classes confondues. Elle est
donc volontairement aveugle à la façon dont cette union est découpée en instances — ce qui
est précisément sa qualité (la fragmentation par la grille de tuilage ne la pollue pas),
mais aussi sa limite : le correctif des boîtes, la déduplication des fenêtres, le décodage
top-k ne la déplacent pas d'un millième. Leur valeur se lit dans la colonne
polygones/km² et dans le temps de calcul, pas ici.</div>
{runs}
{section_b(d.get('niveau_b'))}
<footer>Généré par <code>tools/bench</code>. Parité banc↔plugin vérifiée à l'identique
(polygones au flottant près) avant toute mesure.</footer>
</main>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\pipeline_results\bench")
    ap.add_argument("--fichier", default=None)
    a = ap.parse_args()
    racine = Path(a.out)
    cible = Path(a.fichier) if a.fichier else racine / "report.html"
    cible.write_text(construire(racine), encoding="utf-8")
    print(f"-> {cible}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
