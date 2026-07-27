"""Découpeur de tuiles d'entraînement à split spatial par blocs.

Spec : docs/superpowers/specs/2026-07-27-slice-zone-design.md (décisions D1-D4 du
2026-07-27). Remède au split aléatoire documenté dans docs/fuite_spatiale_train_test.html :
tuiles jointives sans chevauchement, split par blocs géographiques équilibré par classe,
tracé dans split_manifest.yaml.

Usage :
    .venv\\Scripts\\python.exe tools\\slice_zone.py <dataset_config.yaml> [--out D] [--seed N]
"""
import argparse
import datetime
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from PIL import Image
from rasterio.crs import CRS
from rasterio.windows import Window, bounds as fenetre_bounds
from shapely import STRtree, make_valid
from shapely.geometry import LineString, Polygon
from shapely.geometry import box as boite
from shapely.ops import split as decouper

ORDRE_SPLITS = ("train", "valid", "test")  # ordre de départage des égalités


# ---------------------------------------------------------------------------
# Noyau géométrique
# ---------------------------------------------------------------------------

def grille_tuiles(transform, largeur_px, hauteur_px, tuile_px):
    """Grille de tuiles pleines, jointives, sans chevauchement.

    Les tuiles partielles de bord sont écartées (pas de padding : pas de bords
    noirs artificiels dans le dataset). Ordre (row, col).
    """
    tuiles = []
    for row in range(hauteur_px // tuile_px):
        for col in range(largeur_px // tuile_px):
            fenetre = Window(col * tuile_px, row * tuile_px, tuile_px, tuile_px)
            tuiles.append({
                "row": row,
                "col": col,
                "fenetre": fenetre,
                "bounds": fenetre_bounds(fenetre, transform),
            })
    return tuiles


def bloc_de(bounds, bloc_m):
    """Id du bloc contenant le CENTRE de la tuile (grille de bloc_m alignée sur 0)."""
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    return (math.floor(cx / bloc_m), math.floor(cy / bloc_m))


def affecter_splits(annos_par_bloc, cibles, seed):
    """Affectation gloutonne des blocs aux splits, équilibrée par classe.

    Blocs triés par richesse (total annotations) décroissante, départage par mélange
    seedé ; chaque bloc va au split de plus grand besoin RELATIF
    Σ_c (part_cible_s - deja_alloue[s][c]/total_c) / part_cible_s.
    La normalisation par la part cible est essentielle : en déficit absolu, train
    (cible 70 %) partirait avec un besoin triple de valid et accaparerait ~95 % des
    annotations avant de céder un bloc (observé sur les données réelles de Haye).
    Les classes rares pèsent autant que les abondantes (fractions, pas comptes).
    Les blocs sans annotation ne sont pas affectés (ils restent hors split ; les
    tuiles vides des blocs affectés servent de vivier de négatifs).
    """
    parts = {s: cibles[s] / sum(cibles.values()) for s in cibles}
    total_par_classe = Counter()
    for c in annos_par_bloc.values():
        total_par_classe.update(c)

    rng = random.Random(seed)
    blocs = [b for b, c in annos_par_bloc.items() if sum(c.values()) > 0]
    rng.shuffle(blocs)  # départage des égalités de richesse
    blocs.sort(key=lambda b: -sum(annos_par_bloc[b].values()))

    alloue = {s: Counter() for s in cibles}
    affectation = {}
    for b in blocs:
        besoins = {}
        for s in cibles:
            besoins[s] = sum(
                (parts[s] - alloue[s][c] / total_par_classe[c]) / parts[s]
                for c in total_par_classe
            )
        meilleur = max(sorted(besoins, key=lambda s: ORDRE_SPLITS.index(s)),
                       key=lambda s: besoins[s])
        affectation[b] = meilleur
        alloue[meilleur].update(annos_par_bloc[b])
    return affectation


# ---------------------------------------------------------------------------
# Entités -> polygones COCO
# ---------------------------------------------------------------------------

def preparer_entites(gdf, buffer_m):
    """Géométries d'une couche -> liste de polygones prêts à rasteriser.

    buffer_m : largeur TOTALE pour les lignes (buffer de buffer_m/2), rayon pour les
    points, None pour les polygones (inchangés). MultiX explosés, vides écartées,
    invalides réparées (make_valid).
    """
    def _polygones_recursifs(geom):
        # make_valid peut produire GEOMETRYCOLLECTION(MULTIPOLYGON(...), LINESTRING) :
        # une extraction non récursive perdrait silencieusement les polygones imbriqués
        if geom.geom_type == "Polygon":
            yield geom
        elif hasattr(geom, "geoms"):
            for g in geom.geoms:
                yield from _polygones_recursifs(g)

    polys = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if buffer_m is not None:
            rayon = buffer_m / 2 if "LineString" in geom.geom_type else buffer_m
            geom = geom.buffer(rayon)
        if not geom.is_valid:
            geom = make_valid(geom)
        polys.extend(_polygones_recursifs(geom))
    return polys


def _sans_trous(poly, max_coupes=16):
    """Décompose un polygone à trous en morceaux SANS trou (coupes verticales).

    Un enclos parcellaire fermé bufferisé est un anneau : n'émettre que son contour
    extérieur remplirait tout l'intérieur au moment de la rasterisation COCO
    (défaut confirmé en revue : 5,5x la surface réelle sur un enclos de 40 m).
    Chaque trou est éliminé en coupant le polygone par une verticale passant par le
    centroïde du trou ; les morceaux portent le même id d'instance (une segmentation
    COCO accepte plusieurs anneaux).
    """
    morceaux = [poly]
    for _ in range(max_coupes):
        avec_trou = next((m for m in morceaux if m.interiors), None)
        if avec_trou is None:
            return morceaux
        cx = avec_trou.interiors[0].centroid.x
        minx, miny, maxx, maxy = avec_trou.bounds
        lame = LineString([(cx, miny - 1), (cx, maxy + 1)])
        pieces = [g for g in decouper(avec_trou, lame).geoms if g.geom_type == "Polygon"]
        if not pieces or pieces == [avec_trou]:
            break  # cas dégénéré : abandon des trous restants ci-dessous
        morceaux = [m for m in morceaux if m is not avec_trou] + pieces
    return [Polygon(m.exterior) if m.interiors else m for m in morceaux]


def polygone_vers_coco(poly, bounds, tuile_px):
    """Anneau extérieur d'un polygone -> coordonnées pixels COCO [x1,y1,x2,y2,...].

    Origine au coin haut-gauche de la tuile, y vers le bas, arrondi 2 décimales.
    Le polygone reçu ne doit plus avoir de trous (cf. _sans_trous).
    """
    minx, _, _, maxy = bounds
    sx = tuile_px / (bounds[2] - bounds[0])
    sy = tuile_px / (bounds[3] - bounds[1])
    anneau = []
    for x, y in list(poly.exterior.coords)[:-1]:
        anneau.extend([round((x - minx) * sx, 2), round((maxy - y) * sy, 2)])
    return [anneau] if len(anneau) >= 6 else []


SEUIL_AIRE_PX = 2.0   # un morceau clippé plus petit est un artefact de bord de tuile
SEUIL_DIM_PX = 0.5    # idem pour un ruban plus fin qu'un demi-pixel (buffer affleurant)


def annotations_tuile(polys_par_classe, bounds, tuile_px):
    """Clip des polygones à la tuile -> annotations {classe, segmentation, bbox_px, aire_px}."""
    tuile_geo = boite(*bounds)
    sx = tuile_px / (bounds[2] - bounds[0])
    sy = tuile_px / (bounds[3] - bounds[1])
    annos = []
    for classe, polys in polys_par_classe.items():
        for p in polys:
            if not p.intersects(tuile_geo):
                continue
            clip = p.intersection(tuile_geo)
            morceaux = ([clip] if clip.geom_type == "Polygon"
                        else [g for g in getattr(clip, "geoms", ())
                              if g.geom_type == "Polygon"])
            for m in morceaux:
                mnx, mny, mxx, mxy = m.bounds
                aire_px = m.area * sx * sy
                if (aire_px < SEUIL_AIRE_PX
                        or min((mxx - mnx) * sx, (mxy - mny) * sy) < SEUIL_DIM_PX):
                    continue  # sliver sub-pixel : bruit de label, pas une instance
                segmentation = []
                for piece in _sans_trous(m):
                    segmentation.extend(polygone_vers_coco(piece, bounds, tuile_px))
                if not segmentation:
                    continue
                annos.append({
                    "classe": classe,
                    "segmentation": segmentation,
                    "bbox_px": [round((mnx - bounds[0]) * sx, 2),
                                round((bounds[3] - mxy) * sy, 2),
                                round((mxx - mnx) * sx, 2),
                                round((mxy - mny) * sy, 2)],
                    "aire_px": round(aire_px, 2),
                })
    return annos


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def chemin_sur_drive(chemin):
    """Détecte G: sous toutes ses formes (préfixes longs, UNC, URI file)."""
    c = str(chemin).strip().lower().replace("\\", "/")
    for prefixe in ("file:///", "file://", "//?/unc/", "//?/", "//./"):
        if c.startswith(prefixe):
            c = c[len(prefixe):]
    if c.startswith("g:"):
        return True
    # partage administratif \\hote\g$\...
    morceaux = [m for m in c.split("/") if m]
    return len(morceaux) >= 2 and morceaux[1] == "g$"


def _refuser_drive(chemin, nom):
    if chemin_sur_drive(chemin):
        sys.exit(f"{nom} est sur G: — staging local d'abord (règle Drive, cf. CLAUDE.md)")


CHAMPS_REQUIS = ("dataset", "zone", "raster", "gpkg", "couches", "tuile_px",
                 "bloc_m", "split")
DEFAUTS = {"negatifs_pct": 10, "min_couverture_valide": 0.5,
           "min_visibilite_annotation": 0.5, "assign_crs": None,
           "nodata_supplementaire": None}


def charger_config(chemin):
    """Charge et valide la config YAML d'un dataset (cf. spec)."""
    cfg = yaml.safe_load(Path(chemin).read_text(encoding="utf-8"))
    manquants = [c for c in CHAMPS_REQUIS if c not in cfg]
    if manquants:
        sys.exit(f"config invalide : champs manquants {manquants}")
    for cle, valeur in DEFAUTS.items():
        cfg.setdefault(cle, valeur)
    if set(cfg["split"]) != set(ORDRE_SPLITS) or sum(cfg["split"].values()) != 100 \
            or any(v <= 0 for v in cfg["split"].values()):
        sys.exit("config invalide : split doit définir train/valid/test > 0 et sommer à 100")
    if not (isinstance(cfg["tuile_px"], int) and cfg["tuile_px"] > 0):
        sys.exit("config invalide : tuile_px doit être un entier > 0")
    for cle in ("raster", "gpkg"):
        _refuser_drive(cfg[cle], cle)
    for nom, spec in cfg["couches"].items():
        if not isinstance(spec, dict) or "classe" not in spec:
            sys.exit(f"config invalide : couche {nom} sans champ 'classe'")
        spec.setdefault("buffer_m", None)
    return cfg


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _masque_valide(donnees, nodatas):
    """Masque booléen des pixels valides (True = valide)."""
    valide = np.ones(donnees.shape, dtype=bool)
    for nd in nodatas:
        valide &= donnees != nd
    if np.issubdtype(donnees.dtype, np.floating):
        valide &= ~np.isnan(donnees)
    return valide


def _visibilite_bbox(bbox_px, masque, tuile_px):
    """Fraction valide de l'emprise (bbox) d'une annotation dans la tuile."""
    x, y, w, h = bbox_px
    x0 = max(0, int(x)); y0 = max(0, int(y))
    x1 = min(tuile_px, int(math.ceil(x + w))); y1 = min(tuile_px, int(math.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(masque[y0:y1, x0:x1].mean())


def run_slicing(cfg, out_dir, seed=42):
    """Orchestration complète : grille -> validité -> annotations -> split -> sorties."""
    out_dir = Path(out_dir)
    _refuser_drive(out_dir, "--out")
    if out_dir.exists():
        # purge complète : sans elle, une relance laisserait les PNG de l'ancienne
        # affectation dans leur ancien dossier de split — la même image dans train
        # ET test sur disque, précisément la fuite que cet outil doit empêcher
        shutil.rmtree(out_dir)
    tuile_px = cfg["tuile_px"]
    zone_id = str(cfg["zone"]).replace("\\", "/").rstrip("/").split("/")[-1]

    with rasterio.open(cfg["raster"]) as src:
        crs_raster = src.crs
        if crs_raster is None:
            if not cfg["assign_crs"]:
                sys.exit("raster sans CRS : renseigner assign_crs dans la config")
            crs_raster = CRS.from_user_input(cfg["assign_crs"])
        if src.dtypes[0] != "uint8":
            sys.exit(f"raster {src.dtypes[0]} : un indice 8 bits (Byte) est attendu — "
                     "les MNT bruts ne sont pas des images d'entraînement")
        nodatas = set()
        if cfg["nodata_supplementaire"] is not None:
            nodatas.add(cfg["nodata_supplementaire"])
        # nodata déclaré NaN sur bande Byte (chaînes LHD) : rasterio expose None,
        # il n'y a donc rien à filtrer côté fichier — seul nodata_supplementaire agit
        if src.nodata is not None and not math.isnan(src.nodata):
            nodatas.add(src.nodata)
        if not nodatas:
            print("avertissement : aucun nodata effectif — toutes les tuiles seront "
                  "considérées valides (zones sans dalle comprises). Renseigner "
                  "nodata_supplementaire si le raster a un fond implicite (souvent 0).")

        # --- entités : buffers, reprojection éventuelle, index spatial global
        polys, classes_polys = [], []
        classes_ordre = []
        for nom_couche, spec in cfg["couches"].items():
            gdf = gpd.read_file(cfg["gpkg"], layer=nom_couche)
            if gdf.crs is not None and crs_raster is not None and gdf.crs != crs_raster:
                gdf = gdf.to_crs(crs_raster)
            prepares = preparer_entites(gdf, spec["buffer_m"])
            polys.extend(prepares)
            classes_polys.extend([spec["classe"]] * len(prepares))
            if spec["classe"] not in classes_ordre:
                classes_ordre.append(spec["classe"])
        index_spatial = STRtree(polys) if polys else None

        # --- passe 1 : validité + annotations par tuile (sans garder les pixels)
        tuiles = grille_tuiles(src.transform, src.width, src.height, tuile_px)
        gardees = []   # dicts tuile + annos + bloc
        for t in tuiles:
            donnees = src.read(1, window=t["fenetre"])
            masque = _masque_valide(donnees, nodatas)
            couverture = float(masque.mean())
            if couverture < cfg["min_couverture_valide"]:
                continue
            annos, entites_presentes = [], False
            if index_spatial is not None:
                candidats = index_spatial.query(boite(*t["bounds"]))
                par_classe = {}
                for i in sorted(candidats):
                    par_classe.setdefault(classes_polys[i], []).append(polys[i])
                # l'ordre des classes suit la config (déterminisme du COCO)
                par_classe = {c: par_classe[c] for c in classes_ordre if c in par_classe}
                annos_brutes = annotations_tuile(par_classe, t["bounds"], tuile_px)
                entites_presentes = bool(annos_brutes)
                annos = [a for a in annos_brutes
                         if _visibilite_bbox(a["bbox_px"], masque, tuile_px)
                         >= cfg["min_visibilite_annotation"]]
            gardees.append({**t, "annos": annos, "entites_presentes": entites_presentes,
                            "bloc": bloc_de(t["bounds"], cfg["bloc_m"])})

        # --- split par blocs
        annos_par_bloc = {}
        for t in gardees:
            annos_par_bloc.setdefault(t["bloc"], Counter()).update(
                a["classe"] for a in t["annos"])
        tuiles_par_bloc = Counter(t["bloc"] for t in gardees)
        for b, cpt in annos_par_bloc.items():
            if cpt:  # bloc annoté : équilibrer aussi le volume de tuiles, pas
                cpt["__tuiles__"] = tuiles_par_bloc[b]  # seulement les annotations
        affectation = affecter_splits(annos_par_bloc, cfg["split"], seed)

        annotees = [t for t in gardees if t["annos"] and t["bloc"] in affectation]
        # négatifs PURS uniquement : une tuile dont les annotations ont été écartées
        # par le filtre de visibilité contient quand même l'entité — l'exporter comme
        # fond serait un signal d'entraînement contradictoire
        vivier_neg = sorted((t for t in gardees
                             if not t["annos"] and not t["entites_presentes"]
                             and t["bloc"] in affectation),
                            key=lambda t: (t["row"], t["col"]))
        n_neg = min(round(cfg["negatifs_pct"] / 100 * len(annotees)), len(vivier_neg))
        negatives = random.Random(seed).sample(vivier_neg, n_neg) if n_neg else []
        selection = sorted(annotees + negatives, key=lambda t: (t["row"], t["col"]))
        for t in selection:
            t["split"] = affectation[t["bloc"]]
            t["nom"] = f"{zone_id}_r{t['row']:04d}_c{t['col']:04d}.png"

        # --- passe 2 : écriture des images + COCO par split
        categories = [{"id": i + 1, "name": c} for i, c in enumerate(classes_ordre)]
        cat_ids = {c["name"]: c["id"] for c in categories}
        cocos = {s: {"images": [], "annotations": [], "categories": categories}
                 for s in ORDRE_SPLITS}
        compteur_annos = {s: 0 for s in ORDRE_SPLITS}
        for s in ORDRE_SPLITS:
            (out_dir / s).mkdir(parents=True, exist_ok=True)
        for t in selection:
            donnees = src.read(1, window=t["fenetre"])
            Image.fromarray(np.stack([donnees] * 3, axis=-1), mode="RGB").save(
                out_dir / t["split"] / t["nom"], "PNG")
            coco = cocos[t["split"]]
            image_id = len(coco["images"]) + 1
            coco["images"].append({"id": image_id, "file_name": t["nom"],
                                   "width": tuile_px, "height": tuile_px})
            for a in t["annos"]:
                compteur_annos[t["split"]] += 1
                coco["annotations"].append({
                    "id": compteur_annos[t["split"]], "image_id": image_id,
                    "category_id": cat_ids[a["classe"]],
                    "segmentation": a["segmentation"], "bbox": a["bbox_px"],
                    "area": a["aire_px"], "iscrowd": 0,
                })
        for s in ORDRE_SPLITS:
            (out_dir / s / "_annotations.coco.json").write_text(
                json.dumps(cocos[s], ensure_ascii=False), encoding="utf-8")

        gsd = (abs(src.transform.a), abs(src.transform.e))
        grille_info = {"origine": [src.transform.c, src.transform.f],
                       "gsd_m_px": list(gsd), "tuile_px": tuile_px,
                       "crs": str(crs_raster)}

    # --- manifeste + carte de contrôle + récap
    comptes = {s: Counter() for s in ORDRE_SPLITS}
    for t in selection:
        comptes[t["split"]].update(a["classe"] for a in t["annos"])
    hashes = {}
    for s in ORDRE_SPLITS:
        noms = sorted(t["nom"] for t in selection if t["split"] == s)
        hashes[s] = hashlib.sha1("\n".join(noms).encode("utf-8")).hexdigest()
    manifeste = {
        "dataset": cfg["dataset"], "zone": cfg["zone"], "seed": seed,
        "genere_le": datetime.date.today().isoformat(),
        "config": {k: cfg[k] for k in sorted(cfg)},
        "grille": grille_info,
        "comptes": {s: dict(sorted(comptes[s].items())) for s in ORDRE_SPLITS},
        "hashes_sha1": hashes,
        "tuiles": [{"nom": t["nom"], "row": t["row"], "col": t["col"],
                    "bloc": list(t["bloc"]), "split": t["split"],
                    "bounds": [round(v, 3) for v in t["bounds"]],
                    "n_annotations": len(t["annos"]),
                    "classes": sorted({a["classe"] for a in t["annos"]})}
                   for t in selection],
    }
    (out_dir / "split_manifest.yaml").write_text(
        yaml.safe_dump(manifeste, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8")
    _ecrire_carte_controle(out_dir, cfg, affectation, annos_par_bloc, comptes, selection)

    par_split = {s: sum(1 for t in selection if t["split"] == s) for s in ORDRE_SPLITS}
    for s in ORDRE_SPLITS:
        if par_split[s] == 0:
            print(f"AVERTISSEMENT : le split {s} est VIDE (trop peu de blocs annotés "
                  "pour la taille de bloc demandée) — dataset inutilisable tel quel "
                  "pour l'évaluation ; réduire bloc_m ou revoir la zone.")
    stats = {"tuiles": len(selection), "annotees": len(annotees),
             "negatives": len(negatives), "ecartees_nodata": len(tuiles) - len(gardees),
             "vides_non_retenues": len(vivier_neg) - len(negatives),
             "par_split": par_split,
             "comptes": {s: dict(sorted(comptes[s].items())) for s in ORDRE_SPLITS}}
    return stats


# ---------------------------------------------------------------------------
# Carte de contrôle
# ---------------------------------------------------------------------------

COULEURS_SPLIT = {"train": "#7A8C55", "valid": "#5E7F9E", "test": "#C08A3E"}


def _ecrire_carte_controle(out_dir, cfg, affectation, annos_par_bloc, comptes, selection):
    """Carte SVG autonome des blocs (palette du doc fuite_spatiale_train_test.html)."""
    bloc_m = cfg["bloc_m"]
    blocs = sorted(annos_par_bloc)
    if not blocs:
        return
    bxs = [b[0] for b in blocs]
    bys = [b[1] for b in blocs]
    nx = max(bxs) - min(bxs) + 1
    ny = max(bys) - min(bys) + 1
    cote = max(12, min(48, 720 // max(nx, ny)))
    largeur, hauteur = nx * cote + 2, ny * cote + 2
    rects = []
    for b in blocs:
        x = (b[0] - min(bxs)) * cote + 1
        y = (max(bys) - b[1]) * cote + 1  # nord en haut
        couleur = COULEURS_SPLIT.get(affectation.get(b), "#B7B5AA")
        n = sum(v for k, v in annos_par_bloc[b].items() if k != "__tuiles__")
        rects.append(
            f'<rect x="{x}" y="{y}" width="{cote - 1}" height="{cote - 1}" '
            f'fill="{couleur}" opacity="0.75"><title>bloc {b} : '
            f'{affectation.get(b, "non affecté")}, {n} annotations</title></rect>')
    lignes_tbl = []
    classes = sorted({c for cpt in comptes.values() for c in cpt})
    for c in classes:
        tot = sum(comptes[s].get(c, 0) for s in ORDRE_SPLITS) or 1
        cellules = "".join(
            f"<td>{comptes[s].get(c, 0)} ({comptes[s].get(c, 0) / tot:.0%})</td>"
            for s in ORDRE_SPLITS)
        lignes_tbl.append(f"<tr><td>{c}</td>{cellules}</tr>")
    n_par_split = {s: sum(1 for t in selection if t["split"] == s) for s in ORDRE_SPLITS}
    html = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Contrôle des blocs — {cfg["dataset"]}</title>
<style>body{{font:14px/1.5 system-ui;margin:24px;color:#26251F;background:#F4F4F1}}
table{{border-collapse:collapse;margin-top:14px}}td,th{{border:1px solid #DDDCD4;padding:4px 10px}}
.leg span{{display:inline-block;width:12px;height:12px;margin:0 4px 0 12px}}</style></head><body>
<h1>{cfg["dataset"]} — split spatial par blocs de {bloc_m} m</h1>
<p class="leg">tuiles par split :
<span style="background:#7A8C55"></span>train {n_par_split["train"]}
<span style="background:#5E7F9E"></span>valid {n_par_split["valid"]}
<span style="background:#C08A3E"></span>test {n_par_split["test"]}
<span style="background:#B7B5AA"></span>bloc sans annotation (non affecté)</p>
<svg viewBox="0 0 {largeur} {hauteur}" width="{min(largeur * 2, 900)}">{"".join(rects)}</svg>
<table><tr><th>classe</th><th>train</th><th>valid</th><th>test</th></tr>
{"".join(lignes_tbl)}</table></body></html>"""
    (out_dir / "controle_blocs.html").write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("config", help="dataset_config.yaml (cf. spec)")
    parseur.add_argument("--out", default=None,
                         help="dossier de sortie (défaut : datasets\\<dataset>)")
    parseur.add_argument("--seed", type=int, default=42)
    args = parseur.parse_args()

    cfg = charger_config(args.config)
    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "datasets" / cfg["dataset"])
    stats = run_slicing(cfg, out_dir, seed=args.seed)

    print(f"\n{cfg['dataset']} — {stats['tuiles']} tuiles "
          f"({stats['annotees']} annotées + {stats['negatives']} négatives), "
          f"répartition {stats['par_split']}")
    totaux = Counter()
    for c in stats["comptes"].values():
        totaux.update(c)
    for classe in sorted(totaux):
        parts = "  ".join(
            f"{s} {stats['comptes'][s].get(classe, 0)} "
            f"({stats['comptes'][s].get(classe, 0) / totaux[classe]:.0%})"
            for s in ("train", "valid", "test"))
        print(f"  {classe:20s} {parts}")
    print(f"écartées : {stats['ecartees_nodata']} sous couverture valide, "
          f"{stats['vides_non_retenues']} vides non retenues")
    print(f"contrôle visuel : {out_dir / 'controle_blocs.html'}")
    print(f"Sorties : {out_dir}")


if __name__ == "__main__":
    main()
