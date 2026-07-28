"""Vérification indépendante d'un recalage (spec §2) — contrôleur de la boucle.

Usage : python verif_recalage.py <recalage.yaml> <gpkg_source> <gpkg_recale> <raster>
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import yaml
from shapely import wkt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recaler_lignes import PARAMS_DEFAUT, LecteurRaster, densifier

config, source, recale, raster = (Path(a) for a in sys.argv[1:5])
cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
lecteur = LecteurRaster(raster)


def _extremites(geom, tol=0.5):
    parties = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
    return {(round(p[0] / tol) * tol, round(p[1] / tol) * tol)
            for g in parties for p in (g.coords[0], g.coords[-1])}


def _signe_moyen(geom, polarite):
    """Valeur moyenne du raster le long de la ligne, signée par la polarité."""
    signe = 1.0 if polarite == "clair" else -1.0
    parties = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
    vals = []
    for partie in parties:
        pts, _, _ = densifier(partie, 2.0)
        donnees, affine = lecteur.fenetre(partie.bounds, 2.0)
        if donnees.size == 0:
            continue
        vals.append(lecteur.echantillonner(donnees, affine, pts))
    return signe * float(np.nanmean(np.concatenate(vals))) if vals else np.nan


couches_src = {n for n, _ in pyogrio.list_layers(str(source)) if n in cfg["couches"]}
couches_rec = {n for n, _ in pyogrio.list_layers(str(recale))}
assert couches_rec == set(cfg["couches"]) == couches_src, \
    f"couches divergentes : src {couches_src}, recalé {couches_rec}"

total = 0
gains = []  # contraste signé recalé - origine, lignes recalées seulement
for couche in sorted(couches_rec):
    src = gpd.read_file(source, layer=couche)
    rec = gpd.read_file(recale, layer=couche)
    assert len(rec) == len(src), f"{couche} : {len(rec)} lignes vs {len(src)} source"
    assert rec["id_recalage"].is_unique, f"{couche} : ids non uniques"
    assert rec["statut_recalage"].isin(
        ["auto_ok", "a_revoir", "sans_signal"]).all(), f"{couche} : statut inconnu"

    params = {**PARAMS_DEFAUT, **(cfg["couches"][couche] or {})}
    borne = params["fenetre_m"] + params["pas_m"]
    for (_, ligne), g_src in zip(rec.iterrows(), src.geometry):
        origine = wkt.loads(ligne["geom_origine"])
        assert origine.equals(g_src), \
            f"{couche}/{ligne['id_recalage']} : geom_origine ≠ source"
        g = ligne.geometry
        if ligne["statut_recalage"] == "sans_signal":
            assert g.equals(origine), \
                f"{couche}/{ligne['id_recalage']} : sans_signal modifiée"
            continue
        h = origine.hausdorff_distance(g)
        assert h <= borne + 0.5, \
            f"{couche}/{ligne['id_recalage']} : Hausdorff {h:.1f} m > {borne} m"
        ratio = g.length / max(origine.length, 1e-9)
        assert 0.7 <= ratio <= 1.4, \
            f"{couche}/{ligne['id_recalage']} : ratio longueur {ratio:.2f}"
        pol = ligne["polarite_retenue"]
        avant, apres = _signe_moyen(origine, pol), _signe_moyen(g, pol)
        if np.isfinite(avant) and np.isfinite(apres):
            gains.append(apres - avant)
    total += len(rec)

    # topologie : les extrémités partagées du source le restent dans le recalé
    src_geoms = [wkt.loads(w) for w in rec["geom_origine"]]
    comptes_src, comptes_rec = {}, {}
    for g_o, g_r, statut in zip(src_geoms, rec.geometry, rec["statut_recalage"]):
        for n in _extremites(g_o):
            comptes_src[n] = comptes_src.get(n, 0) + 1
        for n in _extremites(g_r):
            comptes_rec[n] = comptes_rec.get(n, 0) + 1
    partages_src = sum(1 for v in comptes_src.values() if v >= 2)
    partages_rec = sum(1 for v in comptes_rec.values() if v >= 2)
    assert partages_rec >= partages_src, \
        f"{couche} : nœuds partagés {partages_src} -> {partages_rec} (topologie cassée)"

assert gains, "aucune ligne recalée à contrôler"
gain_moyen = float(np.mean(gains))
assert gain_moyen >= 0, \
    f"contraste moyen dégradé par le recalage ({gain_moyen:+.1f}) — signal fui"

print(f"vérification recalage : CONFORME — {len(couches_rec)} couches, {total} lignes, "
      f"gain de contraste moyen {gain_moyen:+.1f}")
