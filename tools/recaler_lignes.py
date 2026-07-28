"""Recalage de vecteurs linéaires sur un raster de relief — méthode B de la spec
docs/superpowers/specs/2026-07-28-recalage-vecteurs-design.md.

Profils perpendiculaires à polarité imposée (talus/parcellaire = clair sur LD,
fossé/chemin creux = sombre), extremum sous-pixel, régularisation des offsets le
long de l'abscisse curviligne (moindres carrés pénalisés — snake 1D contraint au
déplacement normal), nœuds partagés recalés une seule fois (topologie préservée).
La géométrie D'ORIGINE de chaque ligne est conservée (colonne geom_origine) ; seule
la géométrie active recalée part ensuite en découpe/upload.

Usage :
    .venv\\Scripts\\python.exe tools\\recaler_lignes.py configs\\recalage_<zone>.yaml
        <gpkg> <raster> [--out <dossier>] [--couches a,b]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
import yaml
from scipy.ndimage import map_coordinates
from shapely.geometry import LineString, MultiLineString

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slice_zone import _refuser_drive

PARAMS_DEFAUT = {"polarite": "auto", "fenetre_m": 8.0, "pas_m": 2.0,
                 "seuil_contraste": 10.0, "seuil_ambiguite": 0.7,
                 "poids_derivee": 4.0, "pas_echant_m": 0.25}
POIDS_ANCRE = 1e6


class LecteurRaster:
    """Lecture fenêtrée d'un raster mono-bande avec échantillonnage bilinéaire."""

    def __init__(self, chemin):
        self.src = rasterio.open(chemin)
        if self.src.crs is None:
            sys.exit(f"{chemin} : raster sans CRS")

    def fenetre(self, bounds, marge_m):
        minx, miny, maxx, maxy = bounds
        fen = rasterio.windows.from_bounds(
            minx - marge_m, miny - marge_m, maxx + marge_m, maxy + marge_m,
            self.src.transform).round_offsets().round_lengths()
        try:
            fen = fen.intersection(
                rasterio.windows.Window(0, 0, self.src.width, self.src.height))
        except rasterio.errors.WindowError:  # hors emprise du raster
            return np.empty((0, 0)), self.src.transform
        donnees = self.src.read(1, window=fen).astype(float)
        return donnees, self.src.window_transform(fen)

    def echantillonner(self, donnees, affine, pts_xy):
        """Valeurs bilinéaires aux points monde (NaN hors fenêtre)."""
        cols = (pts_xy[:, 0] - affine.c) / affine.a
        rows = (pts_xy[:, 1] - affine.f) / affine.e
        return map_coordinates(donnees, [rows, cols], order=1, mode="constant",
                               cval=np.nan)


def densifier(ligne, pas_m):
    """Points réguliers le long de la ligne + normales unitaires + abscisses."""
    longueur = ligne.length
    n = max(int(round(longueur / pas_m)) + 1, 2)
    absc = np.linspace(0.0, longueur, n)
    pts = np.array([ligne.interpolate(s).coords[0] for s in absc])
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    normales = np.column_stack([-tang[:, 1], tang[:, 0]])
    return pts, normales, absc


def extremum_profil(profil, polarite, seuil_contraste, seuil_ambiguite,
                    pas_echant_m=0.25):
    """(offset_m | None, contraste, ambigu) — extremum sous-pixel d'un profil.

    L'offset est relatif au CENTRE du profil (position actuelle du point),
    positif vers l'extrémité du profil (sens de la normale).
    """
    p = np.asarray(profil, dtype=float)
    if np.isnan(p).all():
        return None, 0.0, False
    signe = 1.0 if polarite == "clair" else -1.0
    s = signe * p
    ref = np.nanmedian(s)
    idx = int(np.nanargmax(s))
    contraste = float(s[idx] - ref)
    if contraste < seuil_contraste:
        return None, contraste, False
    # sous-pixel : parabole sur 3 échantillons
    delta = 0.0
    if 0 < idx < len(s) - 1 and np.isfinite(s[idx - 1]) and np.isfinite(s[idx + 1]):
        denom = s[idx - 1] - 2 * s[idx] + s[idx + 1]
        if abs(denom) > 1e-9:
            delta = float(np.clip(0.5 * (s[idx - 1] - s[idx + 1]) / denom, -1, 1))
    centre = (len(s) - 1) / 2.0
    offset = (idx + delta - centre) * pas_echant_m
    # ambiguïté : second pic net à plus de 3 m de l'extremum
    exclu = int(round(3.0 / pas_echant_m))
    reste = s.copy()
    reste[max(0, idx - exclu):idx + exclu + 1] = -np.inf
    second = float(np.nanmax(reste)) if np.isfinite(reste).any() else -np.inf
    ambigu = (second - ref) >= seuil_ambiguite * contraste
    return offset, contraste, bool(ambigu)


def regulariser(offsets, poids_derivee, ancres):
    """Moindres carrés pénalisés (dérivée première) sur la série des offsets.

    offsets : array avec NaN aux points sans mesure (interpolés par la pénalité) ;
    ancres : {index: valeur} imposées à poids fort (extrémités recalées aux nœuds).
    """
    o = np.asarray(offsets, dtype=float)
    n = len(o)
    w = np.where(np.isfinite(o), 1.0, 0.0)
    cible = np.where(np.isfinite(o), o, 0.0)
    for i, v in ancres.items():
        w[i] = POIDS_ANCRE
        cible[i] = v
    # (W + λ DᵀD) d = W·cible, D = différences premières
    A = np.diag(w)
    for i in range(n - 1):
        A[i, i] += poids_derivee
        A[i + 1, i + 1] += poids_derivee
        A[i, i + 1] -= poids_derivee
        A[i + 1, i] -= poids_derivee
    return np.linalg.solve(A, w * cible)


def recaler_ligne(ligne, lecteur, params, ancres_noeuds=None):
    """Recale une ligne par profils perpendiculaires régularisés.

    ancres_noeuds : {0: (x,y), -1: (x,y)} positions imposées des extrémités
    (nœuds partagés déjà recalés). Retourne (ligne_recalee, mesures).
    """
    p = {**PARAMS_DEFAUT, **params}
    pts, normales, _ = densifier(ligne, p["pas_m"])
    n = len(pts)
    donnees, affine = lecteur.fenetre(ligne.bounds, p["fenetre_m"] + 5.0)
    if donnees.size == 0:  # hors emprise : géométrie intacte
        return LineString(ligne), {"polarite_retenue": p["polarite"],
                                   "pts_nets_pct": 0.0, "ambigus_pct": 0.0,
                                   "contraste": 0.0, "offset_median_m": 0.0,
                                   "offset_max_m": 0.0, "residu_m": 0.0,
                                   "recale": False}

    ts = np.arange(-p["fenetre_m"], p["fenetre_m"] + 1e-9, p["pas_echant_m"])
    echant = (pts[:, None, :] + normales[:, None, :] * ts[None, :, None]).reshape(-1, 2)
    profils = lecteur.echantillonner(donnees, affine, echant).reshape(n, len(ts))

    polarites = ([p["polarite"]] if p["polarite"] != "auto" else ["clair", "sombre"])
    meilleurs = None
    for pol in polarites:
        offsets = np.full(n, np.nan)
        contrastes = np.zeros(n)
        ambigus = np.zeros(n, dtype=bool)
        for i in range(n):
            off, c, amb = extremum_profil(profils[i], pol, p["seuil_contraste"],
                                          p["seuil_ambiguite"], p["pas_echant_m"])
            contrastes[i], ambigus[i] = c, amb
            if off is not None and not amb:
                offsets[i] = off
        nets = int(np.isfinite(offsets).sum())
        score = float(np.nansum(np.where(np.isfinite(offsets), contrastes, 0.0)))
        cand = {"pol": pol, "offsets": offsets, "contrastes": contrastes,
                "ambigus": ambigus, "nets": nets, "score": score}
        if meilleurs is None or cand["score"] > meilleurs["score"]:
            meilleurs = cand

    offsets = meilleurs["offsets"]
    nets = meilleurs["nets"]
    mesures = {
        "polarite_retenue": meilleurs["pol"],
        "pts_nets_pct": round(100.0 * nets / n, 1),
        "ambigus_pct": round(100.0 * float(meilleurs["ambigus"].mean()) or 0.0, 1),
        "contraste": round(float(np.mean(meilleurs["contrastes"][np.isfinite(offsets)]))
                           if nets else 0.0, 1),
    }
    seuil_nets = params.get("seuil_points_nets", 5)
    if nets < seuil_nets:
        mesures.update({"offset_median_m": 0.0, "offset_max_m": 0.0,
                        "residu_m": 0.0, "recale": False})
        return LineString(ligne), mesures  # sans signal : géométrie intacte

    ancres = {}
    ancres_noeuds = ancres_noeuds or {}
    for cle, pos in ancres_noeuds.items():
        i = 0 if cle == 0 else n - 1
        ancres[i] = float(np.dot(np.asarray(pos) - pts[i], normales[i]))
    d = regulariser(offsets, p["poids_derivee"], ancres)
    nouveaux = pts + d[:, None] * normales
    for cle, pos in ancres_noeuds.items():  # position exacte du nœud (2D)
        nouveaux[0 if cle == 0 else n - 1] = pos
    residu = float(np.nanmedian(np.abs(offsets - d))) if nets else 0.0
    mesures.update({
        "offset_median_m": round(float(np.nanmedian(np.abs(d))), 2),
        "offset_max_m": round(float(np.max(np.abs(d))), 2),
        "residu_m": round(residu, 2),
        "recale": True,
    })
    return LineString(nouveaux).simplify(0.25), mesures


def noeuds_partages(gdfs, tol=0.5):
    """Extrémités partagées entre lignes (toutes couches) -> groupes d'incidence.

    Retourne {noeud: {(couche, index, 'debut'|'fin'), ...}} où noeud = coordonnées
    arrondies à la tolérance.
    """
    noeuds = {}
    for couche, gdf in gdfs.items():
        for idx, geom in zip(gdf.index, gdf.geometry):
            if geom is None or geom.is_empty:
                continue
            g = max(geom.geoms, key=lambda x: x.length) \
                if geom.geom_type == "MultiLineString" else geom
            for extremite, pt in (("debut", g.coords[0]), ("fin", g.coords[-1])):
                cle = (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)
                noeuds.setdefault(cle, set()).add((couche, idx, extremite))
    return noeuds


SEUILS_STATUT = {"pts_nets_pct_min": 40.0, "ambigus_pct_max": 35.0,
                 "residu_max_m": 1.0}


def statut_ligne(mesures):
    if not mesures.get("recale"):
        return "sans_signal"
    if (mesures["pts_nets_pct"] < SEUILS_STATUT["pts_nets_pct_min"]
            or mesures["ambigus_pct"] > SEUILS_STATUT["ambigus_pct_max"]
            or mesures["residu_m"] > SEUILS_STATUT["residu_max_m"]):
        return "a_revoir"
    return "auto_ok"


def _recaler_geom(geom, lecteur, params, ancres_noeuds):
    """Recale une géométrie ligne (Multi ou simple) ; mesures agrégées par longueur."""
    if geom is None or geom.is_empty:
        return geom, {"polarite_retenue": params.get("polarite", "auto"),
                      "pts_nets_pct": 0.0, "ambigus_pct": 0.0, "contraste": 0.0,
                      "offset_median_m": 0.0, "offset_max_m": 0.0, "residu_m": 0.0,
                      "recale": False}
    if geom.geom_type == "LineString":
        return recaler_ligne(geom, lecteur, params, ancres_noeuds)
    # ponytail: multiparts recalées partie par partie, sans ancres de nœuds
    parties, mesures_p, poids = [], [], []
    for partie in geom.geoms:
        g, m = recaler_ligne(partie, lecteur, params)
        parties.append(g)
        mesures_p.append(m)
        poids.append(partie.length)
    poids = np.array(poids) / max(sum(poids), 1e-9)
    mesures = {"polarite_retenue": mesures_p[0]["polarite_retenue"],
               "offset_max_m": max(m["offset_max_m"] for m in mesures_p),
               "recale": any(m.get("recale") for m in mesures_p)}
    for cle in ("pts_nets_pct", "ambigus_pct", "contraste", "offset_median_m",
                "residu_m"):
        mesures[cle] = round(float(sum(w * m[cle] for w, m in zip(poids, mesures_p))), 2)
    return MultiLineString(parties), mesures


def run_recalage(config_path, gpkg_path, raster_path, out_dir):
    """Pipeline complet : nœuds d'abord, recalage par couche, GPKG + rapport."""
    for chemin, nom in ((gpkg_path, "gpkg"), (raster_path, "raster"),
                        (out_dir, "--out")):
        _refuser_drive(chemin, nom)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    for champ in ("zone", "couches"):
        if champ not in cfg:
            sys.exit(f"config : champ '{champ}' manquant")
    lecteur = LecteurRaster(raster_path)
    gsd = cfg.get("raster_gsd_attendu")
    if gsd and abs(abs(lecteur.src.transform.a) - gsd) > 0.1 * gsd:
        sys.exit(f"raster : GSD {lecteur.src.transform.a} ≠ attendu {gsd}")

    disponibles = {c[0] for c in pyogrio.list_layers(str(gpkg_path))}
    manquantes = set(cfg["couches"]) - disponibles
    if manquantes:
        sys.exit(f"couches absentes du GPKG : {sorted(manquantes)}")

    base = {**PARAMS_DEFAUT,
            "poids_derivee": cfg.get("lissage", {}).get("poids_derivee", 4.0),
            "seuil_ambiguite": cfg.get("seuil_ambiguite", 0.7),
            "seuil_points_nets": cfg.get("seuil_points_nets", 5)}
    params_couches = {c: {**base, **(sur or {})}
                      for c, sur in cfg["couches"].items()}

    gdfs = {}
    for couche in cfg["couches"]:
        gdf = gpd.read_file(gpkg_path, layer=couche)
        if str(gdf.crs).lower() != str(lecteur.src.crs).lower():
            sys.exit(f"{couche} : CRS {gdf.crs} ≠ raster {lecteur.src.crs}")
        gdfs[couche] = gdf

    # Nœuds d'abord : recalés une fois si polarité imposée et unique, sinon figés
    cibles_noeuds = {}  # (couche, idx, extremite) -> (x, y)
    for pos, membres in noeuds_partages(gdfs).items():
        if len(membres) < 2:
            continue
        pols = {params_couches[c]["polarite"] for c, _, _ in membres}
        if len(pols) == 1 and (pol := pols.pop()) != "auto":
            fen = max(params_couches[c]["fenetre_m"] for c, _, _ in membres)
            seuil = min(params_couches[c]["seuil_contraste"] for c, _, _ in membres)
            pos2 = recaler_noeud(pos, lecteur, pol, fen, seuil)
        else:
            pos2 = pos  # polarités mixtes/auto : nœud figé, topologie préservée
        for membre in membres:
            cibles_noeuds[membre] = pos2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg_out = out_dir / f"{cfg['zone']}_entites_l93_recale.gpkg"
    if gpkg_out.exists():
        gpkg_out.unlink()

    rapport = {"zone": cfg["zone"], "raster": str(raster_path),
               "gpkg_source": str(gpkg_path), "seuils_statut": dict(SEUILS_STATUT),
               "parametres": {"couches": {c: dict(p) for c, p in
                                          params_couches.items()}},
               "couches": {}}
    for couche, gdf in gdfs.items():
        params = params_couches[couche]
        geoms, lignes_mesures, statuts = [], [], []
        for i, geom in enumerate(gdf.geometry):
            ancres = {}
            if geom is not None and geom.geom_type == "LineString":
                for cle_ext, extremite in ((0, "debut"), (-1, "fin")):
                    cible = cibles_noeuds.get((couche, gdf.index[i], extremite))
                    if cible is not None:
                        ancres[cle_ext] = cible
            g, m = _recaler_geom(geom, lecteur, params, ancres or None)
            geoms.append(g)
            lignes_mesures.append(m)
            statuts.append(statut_ligne(m))
        sortie = gdf.copy()
        sortie["geom_origine"] = [g.wkt if g is not None else None
                                  for g in gdf.geometry]
        sortie["id_recalage"] = [f"{couche}_{i}" for i in range(len(gdf))]
        sortie["statut_recalage"] = statuts
        for cle in ("polarite_retenue", "pts_nets_pct", "ambigus_pct", "contraste",
                    "offset_median_m", "offset_max_m", "residu_m"):
            sortie[cle] = [m[cle] for m in lignes_mesures]
        sortie["score"] = [round(m["pts_nets_pct"] * m["contraste"] / 100, 1)
                           for m in lignes_mesures]
        sortie.geometry = geoms
        sortie.to_file(gpkg_out, layer=couche, driver="GPKG")

        offsets = [m["offset_median_m"] for m, s in zip(lignes_mesures, statuts)
                   if s != "sans_signal"]
        histo = Counter(f"{int(o)}-{int(o) + 1} m" for o in offsets)
        rapport["couches"][couche] = {
            "lignes": len(gdf), "statuts": dict(Counter(statuts)),
            "offset_median_m": round(float(np.median(offsets)), 2) if offsets else 0.0,
            "histogramme_offsets": dict(sorted(histo.items())),
        }
        print(f"{couche} : {len(gdf)} lignes, statuts {dict(Counter(statuts))}")

    rapport_path = out_dir / "recalage_rapport.yaml"
    rapport_path.write_text(yaml.safe_dump(rapport, allow_unicode=True,
                                           sort_keys=False), encoding="utf-8")
    print(f"Sorties :\n  {gpkg_out}\n  {rapport_path}")
    return gpkg_out, rapport_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("gpkg")
    ap.add_argument("raster")
    ap.add_argument("--out", default=None)
    ap.add_argument("--couches", default=None,
                    help="sous-ensemble de couches, séparées par des virgules")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.couches:
        garder = set(args.couches.split(","))
        cfg["couches"] = {c: v for c, v in cfg["couches"].items() if c in garder}
        chemin_cfg = Path(args.config).with_suffix(".resolue.yaml")
        chemin_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True),
                              encoding="utf-8")
        args.config = chemin_cfg
    out = Path(args.out) if args.out else Path("recalage") / cfg["zone"]
    run_recalage(args.config, args.gpkg, args.raster, out)


def recaler_noeud(pos, lecteur, polarite, fenetre_m, seuil_contraste=10.0):
    """Déplace un nœud vers l'extremum (à la polarité donnée) dans un disque."""
    donnees, affine = lecteur.fenetre((pos[0], pos[1], pos[0], pos[1]), fenetre_m + 2)
    if donnees.size == 0:
        return pos
    signe = 1.0 if polarite == "clair" else -1.0
    h, l = donnees.shape
    cols, rows = np.meshgrid(np.arange(l), np.arange(h))
    xs = affine.c + (cols + 0.5) * affine.a
    ys = affine.f + (rows + 0.5) * affine.e
    dist = np.hypot(xs - pos[0], ys - pos[1])
    s = signe * donnees
    s[dist > fenetre_m] = -np.inf
    if not np.isfinite(s).any():
        return pos
    ref = np.nanmedian(signe * donnees)
    i, j = np.unravel_index(int(np.argmax(s)), s.shape)
    if s[i, j] - ref < seuil_contraste:
        return pos
    return (float(xs[i, j]), float(ys[i, j]))


if __name__ == "__main__":
    main()
