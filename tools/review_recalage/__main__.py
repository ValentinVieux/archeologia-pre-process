"""App locale de revue/édition du recalage (spec §3) — FastAPI + page vanilla.

Usage : .venv\\Scripts\\python.exe -m tools.review_recalage <gpkg_recale> <raster>
            [--decisions <yaml>] [--port 5175]

L'app n'écrit JAMAIS le GPKG : chaque décision part immédiatement dans
recalage_decisions_<zone>.yaml (source de vérité, reprise gratuite).
"""
import argparse
import datetime
import io
import json
import random
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
import rasterio.windows
import yaml
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from shapely import wkt
from shapely.geometry import LineString, MultiLineString

TAILLE_ECHANTILLON = 100  # auto_ok tirés (seed fixe) en plus des a_revoir
GRAINE = 42
MARGE_CROP = 25.0  # m autour de la ligne (fenêtre 8 m + contexte d'édition)


def _parts(geom):
    """Coordonnées [[x,y],...] par partie, LineString ou MultiLineString."""
    parties = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
    return [[[float(x), float(y)] for x, y in p.coords] for p in parties]


def creer_app(gpkg_path, raster_path, decisions_path, zone=None):
    zone = zone or Path(gpkg_path).stem.replace("_entites_l93_recale", "")
    src = rasterio.open(raster_path)
    lignes = {}  # id_recalage -> dict
    for couche in (n for n, _ in pyogrio.list_layers(str(gpkg_path))):
        gdf = gpd.read_file(gpkg_path, layer=couche)
        if "id_recalage" not in gdf.columns:
            continue  # couche non recalée (polygones, points)
        for _, l in gdf.iterrows():
            lignes[l["id_recalage"]] = {
                "id": l["id_recalage"], "couche": couche,
                "statut": l["statut_recalage"], "score": float(l["score"]),
                "geom": l.geometry, "geom_origine": wkt.loads(l["geom_origine"]),
                "mesures": {c: (float(l[c]) if c != "polarite_retenue" else l[c])
                            for c in ("polarite_retenue", "pts_nets_pct",
                                      "ambigus_pct", "contraste", "offset_median_m",
                                      "offset_max_m", "residu_m")},
            }

    auto_ok = sorted(i for i, l in lignes.items() if l["statut"] == "auto_ok")
    echantillon = set(random.Random(GRAINE).sample(
        auto_ok, min(TAILLE_ECHANTILLON, len(auto_ok))))

    ids_tous = list(lignes)
    bornes = (np.array([lignes[i]["geom"].bounds for i in ids_tous])
              if ids_tous else np.zeros((0, 4)))

    decisions_path = Path(decisions_path)
    decisions = (yaml.safe_load(decisions_path.read_text(encoding="utf-8")) or {}
                 if decisions_path.exists() else {})

    def sauver():
        decisions_path.write_text(
            yaml.safe_dump(decisions, allow_unicode=True, sort_keys=True),
            encoding="utf-8")

    app = FastAPI()

    @app.get("/api/lignes")
    def api_lignes(statut: str = "", couche: str = "", perimetre: int = 1):
        res = []
        for i, l in sorted(lignes.items()):
            if statut and l["statut"] != statut:
                continue
            if couche and l["couche"] != couche:
                continue
            if perimetre and not (l["statut"] == "a_revoir" or i in echantillon):
                continue
            res.append({"id": i, "couche": l["couche"], "statut": l["statut"],
                        "score": l["score"], "echantillon": i in echantillon,
                        "decision": decisions.get(i, {}).get("decision"),
                        "offset_median_m": l["mesures"]["offset_median_m"]})
        res.sort(key=lambda x: x["score"])  # pires d'abord
        return {"lignes": res, "zone": zone,
                "couches": sorted({l["couche"] for l in lignes.values()})}

    def geom_active(id_l):
        """Géométrie qui partira réellement au training, décisions incluses."""
        d = decisions.get(id_l, {})
        if d.get("decision") == "exclue":
            return None
        if d.get("decision") == "editee":
            return wkt.loads(d["geometrie_editee"])
        if d.get("decision") == "original":
            return lignes[id_l]["geom_origine"]
        return lignes[id_l]["geom"]

    @app.get("/api/ligne/{id_ligne}")
    def api_ligne(id_ligne: str):
        l = lignes.get(id_ligne)
        if l is None:
            raise HTTPException(404, id_ligne)
        d = decisions.get(id_ligne, {})
        minx, miny, maxx, maxy = l["geom"].union(l["geom_origine"]).bounds
        m = MARGE_CROP
        masque = ((bornes[:, 0] <= maxx + m) & (bornes[:, 2] >= minx - m)
                  & (bornes[:, 1] <= maxy + m) & (bornes[:, 3] >= miny - m))
        voisines = []
        for vid, ok in zip(ids_tous, masque):
            if not ok or vid == id_ligne:
                continue
            g = geom_active(vid)
            if g is not None:
                voisines.append({"id": vid, "couche": lignes[vid]["couche"],
                                 "parts": _parts(g)})
        return {"id": id_ligne, "couche": l["couche"], "statut": l["statut"],
                "mesures": l["mesures"],
                "origine": _parts(l["geom_origine"]), "recale": _parts(l["geom"]),
                "editee": (_parts(wkt.loads(d["geometrie_editee"]))
                           if d.get("geometrie_editee") else None),
                "decision": d.get("decision"), "voisines": voisines}

    @app.get("/api/crop/{id_ligne}")
    def api_crop(id_ligne: str):
        l = lignes.get(id_ligne)
        if l is None:
            raise HTTPException(404, id_ligne)
        minx, miny, maxx, maxy = l["geom"].union(l["geom_origine"]).bounds
        marge = MARGE_CROP
        fen = rasterio.windows.from_bounds(
            minx - marge, miny - marge, maxx + marge, maxy + marge,
            src.transform).round_offsets().round_lengths()
        fen = fen.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        donnees = src.read(1, window=fen)
        aff = src.window_transform(fen)
        png = io.BytesIO()
        Image.fromarray(donnees).save(png, format="PNG")
        return Response(png.getvalue(), media_type="image/png", headers={
            "X-Affine": json.dumps([aff.a, aff.b, aff.c, aff.d, aff.e, aff.f]),
            "X-Gsd": str(abs(aff.a))})

    @app.post("/api/decision")
    def api_decision(corps: dict):
        id_ligne = corps["id"]
        l = lignes.get(id_ligne)
        if l is None:
            raise HTTPException(404, id_ligne)
        decision = corps["decision"]
        if decision not in ("recale", "original", "editee", "exclue"):
            raise HTTPException(422, decision)
        entree = {"id": id_ligne, "couche": l["couche"], "decision": decision,
                  "horodatage": datetime.datetime.now().isoformat(timespec="seconds")}
        if decision == "editee":
            parties = [LineString(p) for p in corps["geometrie"] if len(p) >= 2]
            if not parties:
                raise HTTPException(422, "géométrie éditée vide")
            geom = parties[0] if len(parties) == 1 else MultiLineString(parties)
            entree["geometrie_editee"] = geom.wkt
        decisions[id_ligne] = entree
        sauver()  # écriture immédiate : source de vérité
        return {"ok": True, "decidees": len(decisions)}

    @app.get("/api/progression")
    def api_progression():
        perimetre = {i for i, l in lignes.items()
                     if l["statut"] == "a_revoir" or i in echantillon}
        return {"perimetre": len(perimetre),
                "decidees_perimetre": len(perimetre & set(decisions)),
                "decidees_total": len(decisions),
                "par_decision": {d: sum(1 for v in decisions.values()
                                        if v["decision"] == d)
                                 for d in ("recale", "original", "editee", "exclue")}}

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static",
                               html=True), name="static")
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gpkg")
    ap.add_argument("raster")
    ap.add_argument("--decisions", default=None)
    ap.add_argument("--port", type=int, default=5175)
    args = ap.parse_args()
    zone = Path(args.gpkg).stem.replace("_entites_l93_recale", "")
    decisions = args.decisions or (Path(args.gpkg).parent
                                   / f"recalage_decisions_{zone}.yaml")
    app = creer_app(args.gpkg, args.raster, decisions)
    print(f"Décisions : {decisions}\nhttp://127.0.0.1:{args.port}")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
