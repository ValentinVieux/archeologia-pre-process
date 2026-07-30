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
.galerie{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1px;
 background:var(--rule);border:1px solid var(--rule);margin:1.2rem 0}
.galerie figure{margin:0;background:var(--paper);padding:.5rem}
.galerie img{width:100%;height:auto;display:block;border-radius:2px}
.galerie figcaption{font-size:.78rem;color:var(--mut);margin-top:.35rem}
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
    p = racine / "e0_plafond_rappel.json"
    if p.exists():
        d["e0"] = json.loads(p.read_text(encoding="utf-8"))
    d["niveau_b"] = {}
    for p in sorted(racine.glob("niveau_b*.json")):
        cle = p.stem.replace("niveau_b_", "") or "niveau_b"
        for cfg, g in json.loads(p.read_text(encoding="utf-8")).items():
            d["niveau_b"][f"{cle}/{cfg}"] = g
    p = Path(__file__).with_name("defauts.json")
    if p.exists():
        d["defauts"] = json.loads(p.read_text(encoding="utf-8"))
    p = racine / "comparatif_modeles.json"
    if p.exists():
        d["comparatif"] = json.loads(p.read_text(encoding="utf-8"))
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
F1 longueur <b>0,6223 → 0,6402</b>. À l'inverse supprimer le recouvrement fait tomber à
0,5937&nbsp;: sur un raster large le recouvrement sert réellement.
Le ratio est une fraction de la tuile retranchée au pas — 0,4 = 259&nbsp;px (129,5&nbsp;m)
partagés, pas de 389&nbsp;px au lieu de 519. Coûte <b>1,53×</b> plus de fenêtres
(72 → 110 sur 4,41&nbsp;km²). <i>Réserve&nbsp;: 4 mosaïques, pas d'intervalle de
confiance.</i></p></div>

<div><h4>Suppression des superpositions <span class="chg">activée → désactivée</span></h4>
<p>La stratégie <code>difference</code> est class-agnostic et rogne le polygone le moins
confiant sur l'union des plus confiants — ce que la docstring du plugin signale elle-même
comme fabriquant des artefacts sur des formes linéaires qui se croisent. Niveau B&nbsp;:
0,6223 → 0,6291, à nombre de polygones quasi inchangé.</p></div>

<div><h4>Aire minimale <span class="chg">0 → 200 m²</span></h4>
<p>Arbitrage de charge de relecture, pas gain de métrique. Dans la configuration retenue&nbsp;:
0&nbsp;m² → F1 0,6494 pour 46,7 polygones/km²&nbsp;; 200&nbsp;m² → F1 0,6489 pour
<b>34,5</b>, soit <b>−26 % de polygones</b> pour −0,0005 de F1. 300&nbsp;m² décroche
(−0,004). 200&nbsp;m² ≈ 29 m de linéaire au buffer de 7 m&nbsp;: en dessous, un fragment
n'est plus interprétable seul.</p></div>

<div><h4>Fusion inter-tuiles <span class="chg">inchangée — hypothèse réfutée</span></h4>
<p>L'argument était solide&nbsp;: <code>np.maximum</code> est monotone, donc sur la bande de
recouvrement un pixel faiblement prédit dans une fenêtre l'emporte sur un fond confiant
dans l'autre, et les masques ne peuvent que grossir. Mesuré à recouvrement égal, avec le
seuil de binarisation rebalayé dans chaque règle&nbsp;: <b>max 0,6494 contre moyenne
0,6486</b> — égalité. Le contre-effet compense&nbsp;: une fenêtre qui ne voit qu'un bout
tronqué de structure émet un avis faible et légitime, que la moyenne pénalise.
Une première mesure donnait la moyenne gagnante&nbsp;; elle venait d'un défaut du banc
(comparaison d'une somme de votes au seuil sans normaliser), corrigé depuis.
<b>Aucun changement de comportement cœur sans gain mesuré.</b></p></div>

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


ZONES_TRAIN_ANCIEN = ("54_foret_de_haye", "78_rambouillet")


def section_comparatif(d: dict) -> str:
    """Comparatif ancien / nouveau modèle, ventilé par zone et par classe canonique."""
    runs = {k: v for k, v in d.get("niveau_b", {}).items() if "__" in k.split("/")[0]}
    if not runs:
        return ""
    # {modele: {config: bloc}}
    par_modele: Dict[str, Dict[str, dict]] = {}
    for cle, bloc in runs.items():
        grille, cfg = cle.split("/", 1)
        mod = bloc.get("modele") or grille.split("__")[-1]
        par_modele.setdefault(mod, {})[cfg] = bloc
    if len(par_modele) < 2:
        return ""

    ancien = next((m for m in par_modele if m.startswith("formes_lineaires")), None)
    nouveau = next((m for m in par_modele if m.startswith("lineaires_seg")), None)
    if not (ancien and nouveau):
        return ""

    def meilleur(mod: str) -> tuple:
        return max(par_modele[mod].items(), key=lambda kv: kv[1].get("f1_len") or 0)

    cfg_a, ba = meilleur(ancien)
    cfg_n, bn = meilleur(nouveau)

    # --- tableau par zone ---
    zones = sorted({z for b in (ba, bn) for z in b.get("par_mosaique", {})})
    lignes = []
    for z in zones:
        va, vn = ba["par_mosaique"].get(z), bn["par_mosaique"].get(z)
        if not (va and vn):
            continue
        train = any(t in z for t in ZONES_TRAIN_ANCIEN)
        etiq = (f"<code>{html.escape(z)}</code>"
                + (' <b style="color:var(--clay)">⚠ train de l\'ancien</b>' if train else ""))
        d_f1 = (vn["f1_len"] or 0) - (va["f1_len"] or 0)
        lignes.append([etiq, f(va["f1_len"]), f(vn["f1_len"]),
                       f'<span class="{"pos" if d_f1 > 0 else "neg"}">{d_f1:+.4f}</span>',
                       f(va["completude"], 3), f(vn["completude"], 3),
                       f(va["correction"], 3), f(vn["correction"], 3),
                       f(va.get("polygones_par_km2"), 0), f(vn.get("polygones_par_km2"), 0)])

    # --- tableau par classe canonique (agrégat loyal) ---
    lignes_cl = []
    for cl in sorted(set(ba.get("par_classe", {})) | set(bn.get("par_classe", {}))):
        ca, cn = ba.get("par_classe", {}).get(cl, {}), bn.get("par_classe", {}).get(cl, {})
        d_f1 = (cn.get("f1_len") or 0) - (ca.get("f1_len") or 0)
        lignes_cl.append([f"<code>{html.escape(cl)}</code>",
                          f(ca.get("f1_len")), f(cn.get("f1_len")),
                          f'<span class="{"pos" if d_f1 > 0 else "neg"}">{d_f1:+.4f}</span>',
                          f(ca.get("completude"), 3), f(cn.get("completude"), 3),
                          f(ca.get("len_gt_m", 0) / 1000, 2)])

    # --- bootstrap apparié ---
    # Prendre le bootstrap qui correspond AUX CONFIGS AFFICHÉES, pas le premier venu :
    # le fichier en accumule plusieurs (chacun à son meilleur, chacun tel que livré,
    # poids seuls…) et en afficher un autre que celui du tableau serait incohérent.
    boot = ""
    for _, b in (d.get("comparatif") or {}).items():
        if b.get("config_a") != cfg_a or b.get("config_b") != cfg_n:
            continue
        sig = ("<b class=\"sig\">significatif</b>" if b["significatif"]
               else "non significatif (l'intervalle contient zéro)")
        boot = (f"<p><b>Écart apparié par tuile</b> sur {b['n_tuiles']} tuiles "
                f"({b['aire_km2']:.2f} km² loyaux, {b['n_boot']} tirages) : "
                f"<b>{b['delta']:+.4f}</b> de F1 longueur, "
                f"IC95 [{b['ic95'][0]:+.4f} ; {b['ic95'][1]:+.4f}] — {sig}.</p>")
        break
    if not boot and (d.get("comparatif") or {}):
        boot = ('<p class="sub">Aucun bootstrap ne correspond au couple de configurations '
                'affiché ci-dessous — relancer <code>bench comparer</code> avec ces '
                'configs avant de citer un intervalle.</p>')

    # Décomposition : ce que le changement de POIDS apporte, ce que le RÉGLAGE apporte.
    # Sans elle on attribuerait au nouveau modèle un gain dont une part vient du réglage,
    # qui aurait pu bénéficier aussi à l'ancien.
    decompo = ""
    av = par_modele[nouveau].get("avant_reglage")
    ga = par_modele[nouveau].get("geo_ancien")
    if av and bn:
        lg = [["poids seuls — ancien à son meilleur → nouveau à sa config d'avant réglage",
               f(ba["f1_len"]), f(av["f1_len"]),
               f'<span class="{"pos" if av["f1_len"] > ba["f1_len"] else "neg"}">'
               f'{av["f1_len"] - ba["f1_len"]:+.4f}</span>'],
              ["réglage seul — nouveau avant réglage → nouveau réglé",
               f(av["f1_len"]), f(bn["f1_len"]),
               f'<span class="pos">{bn["f1_len"] - av["f1_len"]:+.4f}</span>'],
              ["<b>total</b> — ancien à son meilleur → nouveau réglé",
               f(ba["f1_len"]), f(bn["f1_len"]),
               f'<span class="pos"><b>{bn["f1_len"] - ba["f1_len"]:+.4f}</b></span>']]
        if ga:
            lg.insert(2, ["post-traitement seul — nouveau avec le post-traitement de "
                          "l'ancien → nouveau réglé",
                          f(ga["f1_len"]), f(bn["f1_len"]),
                          f'<span class="pos">{bn["f1_len"] - ga["f1_len"]:+.4f}</span>'])
        decompo = ("<h3>D'où vient le gain</h3>"
                   + tableau(["effet isolé", "de", "à", "Δ F1 longueur"], lg))

    return f"""<h2>Comparatif ancien / nouveau modèle</h2>
<div class="note warn"><b>La comparaison n'est loyale que sur trois zones.</b>
L'ancien modèle a été entraîné sur Rambouillet et Forêt de Haye uniquement, avec un split
<em>aléatoire par tuile</em> sur une grille à 40 % de recouvrement réel. Résultat mesuré :
<b>134/134</b> tuiles test de Haye et <b>210/211</b> de Rambouillet tombent dans son emprise
d'entraînement, soit ≈ 92 % de leurs pixels. Ce n'est pas de la fuite spatiale, c'est du
train — et à Haye la vérité terrain v2 est construite depuis les mêmes shapefiles qu'il a
appris. Ces deux zones sont donc mesurées mais <b>exclues de l'agrégat</b> : elles donnent
son plafond de mémorisation, pas sa performance.</div>
<p>Configurations retenues&nbsp;: ancien <code>{html.escape(cfg_a)}</code>, nouveau
<code>{html.escape(cfg_n)}</code> — chacun à son propre optimum mesuré, pour ne pas
comparer un réglage optimisé à un réglage qui ne l'a jamais été.</p>
<div class="note latent"><b>Deux limites de la mesure de l'ancien modèle, à connaître avant
de lire les chiffres.</b>
<p>Son seuil de confiance optimal tombe <b>sur la borne</b> du balayage (0,15) et non à
l'intérieur. Descendre plus bas exigerait un cache à plancher inférieur puis un décodage à
bien plus de requêtes retenues — ce qui sature les 7,4 Go disponibles. Son optimum réel est
peut-être plus bas&nbsp;; on ne peut pas l'affirmer.</p>
<p>L'axe d'échelle, lui, est complet (512 / 640 / 1032&nbsp;px) et il révèle que le
bénéfice de l'échelle native est surtout un bénéfice de <em>mémorisation</em>&nbsp;: sur ses
zones d'entraînement 512→1032 fait passer le F1 de 0,512 à 0,620, mais sur les zones
loyales seulement de 0,318 à 0,465, où 640 et 1032 sont quasi à égalité. On ne peut donc
pas déduire de l'inférence la taille de ses tuiles d'entraînement. Ce qui reste établi&nbsp;:
sa configuration livrée (<code>slice: 512</code>) laisse <b>0,15 de F1</b> sur la table.</p></div>
{boot}
{decompo}
<h3>Par zone</h3>
{tableau(["mosaïque", "F1 ancien", "F1 nouveau", "Δ", "compl. anc.", "compl. nouv.",
          "corr. anc.", "corr. nouv.", "poly/km² anc.", "poly/km² nouv."], lignes)}
<h3>Par classe, dans l'espace canonique à 3 classes</h3>
<p>L'ancien modèle ne connaît que 3 classes et ne peut structurellement pas produire
<code>talus</code> ni <code>fosse</code>&nbsp;: sans fusion vers un espace commun, les 728
annotations correspondantes du test lui seraient comptées en manques par construction.
<code>talus</code>, <code>fosse</code> et <code>talus_fosse</code> du nouveau sont donc
regroupés. Réserve&nbsp;: le sens de <code>talus_fosse</code> a changé entre les deux
générations — fourre-tout chez l'ancien, cas indissociables chez le nouveau — donc la
fusion avantage légèrement l'ancien sur cette ligne.</p>
{tableau(["classe canonique", "F1 ancien", "F1 nouveau", "Δ", "compl. anc.",
          "compl. nouv.", "GT (km)"], lignes_cl)}"""


def section_visuel(racine: Path, n_max: int = 8) -> str:
    """Extraits superposés, intégrés en base64 pour que le HTML reste autonome.

    Les vignettes sont centrées sur les tuiles où les deux modèles DIVERGENT le plus :
    montrer des extraits au hasard n'aiderait pas à juger, alors que le désaccord est
    précisément là où l'œil de l'archéologue tranche mieux que la métrique.
    """
    import base64
    idx = racine / "visuel" / "index.json"
    if not idx.exists():
        return ""
    meta = json.loads(idx.read_text(encoding="utf-8"))
    extraits = meta.get("extraits", [])
    if not extraits:
        return ""
    # Un extrait par zone d'abord, puis on complète — pour ne pas montrer 8 vues de Blois.
    par_zone: Dict[str, list] = {}
    for e in extraits:
        par_zone.setdefault(e["zone"], []).append(e)
    ordonnes: list = []
    rang = 0
    while len(ordonnes) < min(n_max, len(extraits)):
        ajoute = False
        for z in sorted(par_zone):
            if rang < len(par_zone[z]) and len(ordonnes) < n_max:
                ordonnes.append(par_zone[z][rang])
                ajoute = True
        if not ajoute:
            break
        rang += 1

    vignettes = []
    for e in ordonnes:
        p = racine / "visuel" / e["fichier"]
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode()
        vignettes.append(
            f'<figure><img src="data:image/jpeg;base64,{b64}" '
            f'alt="{html.escape(e["zone"])} {html.escape(e["tuile"])}" loading="lazy">'
            f'<figcaption>{html.escape(e["zone"])} · <code>{html.escape(e["tuile"])}</code>'
            f'</figcaption></figure>')
    if not vignettes:
        return ""
    return f"""<h2>Voir par soi-même</h2>
<p>Vignettes centrées sur les tuiles où les deux modèles <b>divergent le plus</b> en
longueur retrouvée — c'est là que l'œil tranche mieux que la métrique. En vert la vérité
terrain (les lignes des GPKG recalés), en magenta l'ancien modèle, en cyan le nouveau.</p>
<div class="galerie">{''.join(vignettes)}</div>
<div class="note"><b>Pour inspecter dans QGIS.</b> Chaque mosaïque a son
<code>comparatif.gpkg</code> dans <code>D:\\pipeline_results\\bench\\visuel\\</code>, avec
une couche par modèle et par classe plus la vérité terrain, en EPSG:2154 — elles se
superposent donc directement à vos rasters LD existants, les mosaïques n'en étant que des
découpes. Un <code>fond_LD.png</code> géoréférencé (<code>.pgw</code> + <code>.prj</code>)
est fourni si vous préférez ne rien avoir à charger d'autre.</div>"""


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
    # Le bandeau annonce ce qu'on LIVRE, sur le split de test jamais utilisé pour régler —
    # pas la meilleure cellule d'une grille sur le split de réglage. conf 0,22 a un F1 un
    # peu plus haut que 0,25 mais rend 1,6x plus de polygones à relire : ce n'est pas la
    # config retenue, elle n'a donc rien à faire en titre.
    principal = d["runs"].get("e4_finalistes__test") or max(
        d["runs"].items(),
        key=lambda kv: max((g.get("n_images") or 0) for g in kv[1].values()),
        default=(None, {}))[1]
    base = principal.get("base")
    meilleur = None
    for cfg in ("conf_0.25", "corrige_conf_0.25"):
        if cfg in principal:
            meilleur = (cfg, principal[cfg])
            break
    if meilleur is None:
        for cfg, g in principal.items():
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
{section_comparatif(d)}
{section_visuel(racine)}
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
