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
/* Palette tirée du sujet : le gris bleuté du raster Local Dominance, le vert-lichen des
   massifs audités (Haye, Fontainebleau, Rambouillet), la terre cuite des régressions. */
:root{
  --paper:#f7f7f5; --ink:#191c1e; --mut:#5f6b66; --bd:#dedfd9; --zone:#eeefea;
  --moss:#3f6b46; --clay:#a04430; --ochre:#8a6a1f; --rule:#c9cdc4;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#131618; --ink:#e6e8e5; --mut:#98a49e; --bd:#2a2f31; --zone:#1a1e20;
  --moss:#7fb083; --clay:#e08a72; --ochre:#d0aa55; --rule:#333a3c;
}}
:root[data-theme="dark"]{
  --paper:#131618; --ink:#e6e8e5; --mut:#98a49e; --bd:#2a2f31; --zone:#1a1e20;
  --moss:#7fb083; --clay:#e08a72; --ochre:#d0aa55; --rule:#333a3c;
}
:root[data-theme="light"]{
  --paper:#f7f7f5; --ink:#191c1e; --mut:#5f6b66; --bd:#dedfd9; --zone:#eeefea;
  --moss:#3f6b46; --clay:#a04430; --ochre:#8a6a1f; --rule:#c9cdc4;
}
*{box-sizing:border-box}
body{margin:0;padding:2.5rem 1.2rem 4rem;background:var(--paper);color:var(--ink);
 font:16px/1.65 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 font-variant-numeric:tabular-nums}
main{max-width:1120px;margin:0 auto;display:flex;flex-direction:column;gap:.2rem}
h1,h2,h3{font-family:"Iowan Old Style",Georgia,"Times New Roman",serif;
 font-weight:600;text-wrap:balance;letter-spacing:-.01em}
h1{font-size:2.1rem;line-height:1.15;margin:0 0 .3rem}
h2{font-size:1.45rem;margin:2.8rem 0 .5rem;padding-bottom:.35rem;border-bottom:1px solid var(--rule)}
h3{font-size:1.1rem;margin:1.8rem 0 .3rem;color:var(--mut)}
p{max-width:68ch}
.sub{color:var(--mut);margin:0 0 1.8rem;max-width:68ch}
.eyebrow{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--moss);
 font-weight:600;margin:0 0 .5rem}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;
 margin:1.4rem 0 .6rem;background:var(--rule);border:1px solid var(--rule)}
.kpi div{padding:1rem 1.1rem;background:var(--paper)}
.kpi b{display:block;font-size:1.9rem;line-height:1.1;font-family:"Iowan Old Style",Georgia,serif}
.kpi span{color:var(--mut);font-size:.8rem;display:block;margin-top:.3rem}
.tw{overflow-x:auto;margin:.7rem 0 1.2rem}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:560px}
th,td{padding:.45rem .6rem;text-align:right;border-bottom:1px solid var(--bd);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal;
 font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;font-size:12.5px}
thead th{background:var(--zone);position:sticky;top:0;font-weight:600;font-size:12px;
 letter-spacing:.03em;text-transform:uppercase;color:var(--mut)}
tbody tr:hover{background:var(--zone)}
.pos{color:var(--moss);font-weight:600} .neg{color:var(--clay)}
.sig{font-weight:700;color:var(--moss)}
code{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;font-size:.88em;
 background:var(--zone);padding:.08rem .32rem;border-radius:2px}
.note{border-left:2px solid var(--moss);padding:.7rem 1rem;background:var(--zone);
 margin:1rem 0;max-width:78ch}
.note.warn{border-left-color:var(--clay)}
.note.latent{border-left-color:var(--ochre)}
.note p{margin:.45rem 0 0;max-width:none}
.reco{display:flex;flex-direction:column;gap:1px;background:var(--rule);
 border:1px solid var(--rule);margin:1.2rem 0}
.reco>div{background:var(--paper);padding:1rem 1.1rem}
.reco h4{margin:0 0 .3rem;font-size:1rem;font-family:"Iowan Old Style",Georgia,serif}
.reco .chg{font-family:ui-monospace,Consolas,monospace;font-size:.85rem;color:var(--moss)}
footer{margin-top:3.5rem;color:var(--mut);font-size:.85rem;border-top:1px solid var(--rule);
 padding-top:1rem;max-width:68ch}
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


def section_reco(runs: dict) -> str:
    """Ce qu'il faut changer, avec le chiffre qui le justifie."""
    t = runs.get("e4_finalistes__test", {})
    v = runs.get("e4_finalistes", {})
    if not t or "base" not in t:
        return ""
    b, c = t["base"], t.get("conf_0.25", {})
    return f"""<h2>Ce qu'il faut changer</h2>
<div class="reco">
<div><h4>Seuil de confiance <span class="chg">0,30 → 0,25</span></h4>
<p>Sur les 830 tuiles de test, jamais utilisées pour régler&nbsp;: F1 longueur
<b>{f(b['f1_len'],4)} → {f(c.get('f1_len'),4)}</b>, écart apparié
<b class="pos">+0,0359</b> IC95 [+0,0277&nbsp;; +0,0442]. La complétude passe de
{f(b['completude'],3)} à {f(c.get('completude'),3)} — soit
{f((c.get('completude',0)-b['completude'])*100,0)} points de longueur de structure
retrouvée en plus. Contrepartie assumée&nbsp;: {f(b.get('polygones_par_km2'),0)} →
{f(c.get('polygones_par_km2'),0)} polygones/km² à relire.</p></div>

<div><h4>Recouvrement SAHI <span class="chg">0,2 → 0,4</span></h4>
<p>Mesuré au niveau mosaïque uniquement — sur une tuile isolée toutes les fenêtres
retombent sur la même image, l'axe y est inerte. Sur 4 mosaïques (17&nbsp;km²)&nbsp;:
F1 longueur 0,6223 → 0,6401. À l'inverse supprimer le recouvrement fait tomber à 0,5937.
Coûte 1,67× plus de fenêtres. <i>Réserve&nbsp;: 4 mosaïques, pas d'intervalle de
confiance.</i></p></div>

<div><h4>Suppression des superpositions <span class="chg">activée → désactivée</span></h4>
<p>La stratégie <code>difference</code> est class-agnostic et rogne le polygone le moins
confiant sur l'union des plus confiants — ce que la docstring du plugin signale elle-même
comme fabriquant des artefacts sur des formes linéaires qui se croisent. Niveau B&nbsp;:
0,6223 → 0,6291, à nombre de polygones quasi inchangé.</p></div>

<div><h4>Les quatre correctifs de code <span class="chg">neutres sur cette métrique</span></h4>
<p>Écart apparié <b>+0,0000</b> [−0,0001&nbsp;; +0,0002]. C'est attendu&nbsp;: la métrique
en longueur est calculée sur l'union des masques, donc aveugle au découpage en instances.
Leur valeur est ailleurs — séparation correcte des instances (2× plus d'instances
distinctes à seuil bas), ×11 sur le temps de calcul à seuil 0,05, et cohérence CPU/GPU.
Ils sont à appliquer, mais pas pour le F1.</p></div>
</div>"""


def section_defauts(d: dict) -> str:
    if not d:
        return ""
    blocs = []
    for x in d["defauts"]:
        cls = {"majeur": "note warn", "latent": "note latent"}.get(x["gravite"], "note")
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
<p class="eyebrow">Banc d'essai — structures linéaires sur Local Dominance 0,5 m</p>
<h1>{html.escape(titre)}</h1>
<p class="sub">Mesure de la chaîne d'inférence RF-DETR-Seg telle qu'elle tourne réellement
dans le plugin QGIS, sur des tuiles jamais vues à l'entraînement (blocs spatiaux disjoints
de 2 km). Le banc reproduit <code>_run_rfdetr_seg_with_sahi</code> à l'identique — 132
détections sur 20 tuiles, polygones égaux au flottant près — avant toute mesure&nbsp;:
ce qui est chiffré ici est la production, pas une réimplémentation.</p>
{kpi}
{section_reco(d['runs'])}
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
