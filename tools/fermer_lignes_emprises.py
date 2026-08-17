"""Fermeture automatique de labellisations LIGNES en emprises pleines.

Semantique : une labellisation = UNE emprise = union remplie de toutes ses cellules
(murs mitoyens absorbes — jonctions en T ecartees de l'anneau). Methodes en cascade :
polygonisation directe -> chainage par extremites (ponts droits, trace preserve) ->
fermeture morphologique a d croissant. Etages de sortie (banc de perforation sur les
anneaux fermes, IoU 0,998 a ponts <= 10 %) : part_ponts <= 10 % = auto ; 10-30 % =
a_verifier ; sinon / infermable / aire hors bornes = a_arbitrer (humain).

Usage : python toolsermer_lignes_emprises.py <lignes.gpkg> <sortie.gpkg>
        [--couches enclos enceinte] [--aire-min 50] [--aire-max 25000]
"""
import collections
import glob
import sys

import geopandas as gpd
from shapely import make_valid
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union

sys.stdout.reconfigure(encoding="utf-8")


def polygones_de(geom):
    """Extraction recursive des Polygon d'une geometrie quelconque."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(polygones_de(g))
        return out
    return []


def emprise_union(geoms):
    propres = [make_valid(g) for g in geoms]
    pleines = [make_valid(Polygon(p.exterior)) for p in polygones_de(unary_union(propres)) if p.exterior]
    return unary_union(pleines) if pleines else None


def fermer_chainage(geom, pont_max=60.0, eps_t=1.0):
    """Chainage par extremites : ponts droits entre brins, murs internes ecartes.

    Un brin dont une extremite touche un AUTRE brin (jonction en T, < eps_t) est un
    mur interne : exclu de l'anneau, il sera absorbe par l'union des cellules.
    Retourne l'emprise ou None.
    """
    from shapely.geometry import LineString
    from shapely.ops import linemerge

    u = unary_union(geom)
    try:
        fusion = linemerge(u)
    except ValueError:
        fusion = u
    brins = [l for l in (fusion.geoms if hasattr(fusion, "geoms") else [fusion])
             if l.geom_type == "LineString" and l.length > 0]
    if not brins:
        return None
    if len(brins) > 40:
        return None  # trace pathologique : le chainage O(n^3) exploserait, repli morpho
    anneau, murs = [], []
    for b in brins:
        p0, p1 = Point(b.coords[0]), Point(b.coords[-1])
        autres = [o for o in brins if o is not b]
        en_t = autres and (min(o.distance(p0) for o in autres) < eps_t
                           or min(o.distance(p1) for o in autres) < eps_t)
        (murs if en_t and len(brins) > 1 else anneau).append(b)
    if not anneau:
        anneau, murs = murs, []
    ponts = []
    travail = list(anneau)
    while len(travail) > 1:
        # paire d'extremites la plus proche entre brins differents
        meilleur = None
        for i in range(len(travail)):
            for j in range(i + 1, len(travail)):
                for ci in (travail[i].coords[0], travail[i].coords[-1]):
                    for cj in (travail[j].coords[0], travail[j].coords[-1]):
                        dist = Point(ci).distance(Point(cj))
                        if meilleur is None or dist < meilleur[0]:
                            meilleur = (dist, i, j, ci, cj)
        dist, i, j, ci, cj = meilleur
        if dist > pont_max:
            return None
        ponts.append(LineString([ci, cj]))
        fusionne = linemerge(unary_union([travail[i], travail[j], ponts[-1]]))
        nouveaux = [l for l in (fusionne.geoms if hasattr(fusionne, "geoms") else [fusionne])]
        avant = len(travail)
        travail = [t for k, t in enumerate(travail) if k not in (i, j)] + nouveaux
        if len(travail) >= avant:
            return None  # la fusion ne reduit pas (geometrie degeneree) : repli morpho
    dernier = travail[0]
    p0, p1 = Point(dernier.coords[0]), Point(dernier.coords[-1])
    if p0.distance(p1) > pont_max:
        return None
    ferme = unary_union([dernier, LineString([dernier.coords[0], dernier.coords[-1]])] + murs)
    faces = list(polygonize(make_valid(ferme)))
    if not faces:
        return None
    return emprise_union(faces)


from shapely.geometry import Point  # noqa: E402  (utilise par fermer_chainage)


def fermer(geom, rayons=(2, 5, 10, 15, 20)):
    """(emprise, methode, d). Directe -> chainage -> morpho."""
    faces = list(polygonize(unary_union(geom)))
    if faces:
        return emprise_union(faces), "directe", 0
    emp = fermer_chainage(geom)
    if emp is not None and not emp.is_empty:
        return emp, "chainage", 0
    for d in rayons:
        gonfle = unary_union(geom).buffer(d, join_style=1)
        interieurs = [Polygon(t) for p in polygones_de(gonfle) for t in p.interiors]
        if interieurs:
            # interieur(s) de l'anneau gonfle + re-degonflage du tout = emprise
            emp = unary_union(interieurs + polygones_de(gonfle)).buffer(-d, join_style=1)
            emp = emprise_union(polygones_de(emp))
            if emp is not None and not emp.is_empty:
                return emp, "morpho", d
    return None, "echec", None


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpkg", help="GPKG source contenant les couches de lignes")
    ap.add_argument("sortie", help="GPKG de sortie (3 couches par etage)")
    ap.add_argument("--couches", nargs="+", required=True)
    ap.add_argument("--aire-min", type=float, default=50)
    ap.add_argument("--aire-max", type=float, default=25000)
    a = ap.parse_args()

    autos, a_verifier, arbitrage = [], [], []
    multi_cellules = 0
    for couche in a.couches:
        g = gpd.read_file(a.gpkg, layer=couche).to_crs(2154)
        for i, r in g.iterrows():
            if r.geometry is None or r.geometry.is_empty:
                continue
            if len(list(polygonize(unary_union(r.geometry)))) > 1:
                multi_cellules += 1
            emp, methode, d = fermer(r.geometry)
            if emp is None or emp.is_empty:
                arbitrage.append({"couche": couche, "fid_source": i, "raison": "infermable",
                                  "longueur_m": round(r.geometry.length), "geometry": r.geometry})
                continue
            perim = sum(p.exterior.length for p in polygones_de(emp))
            part_ponts = 0.0 if methode == "directe" else max(
                0.0, 1.0 - min(1.0, r.geometry.length / perim))
            rec = {"couche": couche, "fid_source": i, "methode": methode,
                   "part_ponts": round(part_ponts, 3), "aire_m2": round(emp.area),
                   "geometry": emp}
            if not (a.aire_min <= emp.area <= a.aire_max):
                arbitrage.append({"couche": couche, "fid_source": i,
                                  "raison": f"aire {round(emp.area)} m2 hors bornes",
                                  "longueur_m": round(r.geometry.length), "geometry": r.geometry})
            elif part_ponts <= 0.10:
                autos.append(rec)
            elif part_ponts <= 0.30:
                a_verifier.append(rec)
            else:
                arbitrage.append({"couche": couche, "fid_source": i,
                                  "raison": f"ponts {part_ponts:.0%} du perimetre",
                                  "longueur_m": round(r.geometry.length), "geometry": r.geometry})
    gpd.GeoDataFrame(autos, crs=2154).to_file(a.sortie, layer="emprises_fermees_auto", driver="GPKG")
    gpd.GeoDataFrame(a_verifier, crs=2154).to_file(a.sortie, layer="fermees_a_verifier", driver="GPKG")
    gpd.GeoDataFrame(arbitrage, crs=2154).to_file(a.sortie, layer="a_arbitrer", driver="GPKG")
    print(f"AUTO (ponts <= 10 %) : {len(autos)} | A VERIFIER (10-30 %) : {len(a_verifier)} "
          f"| ARBITRAGE : {len(arbitrage)} | multi-cellules unifiees : {multi_cellules}")
    print(f"raisons arbitrage : {dict(collections.Counter(x['raison'].split(' ')[0] for x in arbitrage))}")
    print("->", a.sortie)


if __name__ == "__main__":
    main()
