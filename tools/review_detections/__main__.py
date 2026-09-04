"""App locale de revue des détections d'un modèle sur un corpus (par tuile).

Usage : .venv\\Scripts\\python.exe -m tools.review_detections <corpus> <detections.json>
            [--decisions <yaml>] [--port 5176]

<detections.json> vient de tools/inferer_corpus.py (coordonnées pixel par tuile).
L'app n'écrit JAMAIS le corpus ni le GPKG : chaque décision humaine
(valide/invalide/editee/reclassement, ajouts dessinés) part immédiatement dans le
YAML de décisions (source de vérité, reprise gratuite) — même contrat que
tools/review_recalage.
"""
import argparse
import datetime
import io
import json
import re
import sys
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

RE_TUILE = re.compile(r"_r(\d+)_c(\d+)\.png$")
TUILE_PX = 648
GRIS_MANQUANT = 128  # remplissage des voisines absentes de la mosaïque 3x3


def _gt_bbox(ann):
    """Annotation COCO -> bbox [x0, y0, x1, y1] (revue en mode boîtes)."""
    x, y, w, h = ann["bbox"]
    return [round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1)]


def _bbox_valide(b):
    return (isinstance(b, (list, tuple)) and len(b) == 4
            and all(isinstance(v, (int, float)) for v in b)
            and b[2] - b[0] >= 2 and b[3] - b[1] >= 2)


def creer_app(corpus_path, detections_path, decisions_path):
    corpus = Path(corpus_path)
    donnees = json.loads(Path(detections_path).read_text(encoding="utf-8"))
    meta, tuiles = donnees["_meta"], donnees["tuiles"]

    # GT + index de grille (dataset, row, col) -> clé tuile, depuis les COCO du corpus
    gt, grille, infos_tuile = {}, {}, {}
    for split in meta["splits"]:
        coco = json.loads((corpus / split / "_annotations.coco.json")
                          .read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        par_image = {}
        for a in coco["annotations"]:
            par_image.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            cle = f"{split}/{im['file_name']}"
            m = RE_TUILE.search(im["file_name"])
            row, col = int(m.group(1)), int(m.group(2))
            grille[(im["dataset"], row, col)] = cle
            infos_tuile[cle] = {"split": split, "zone": im.get("zone", ""),
                                "dataset": im["dataset"], "row": row, "col": col,
                                "fichier": corpus / split / im["file_name"]}
            gt[cle] = [{"classe": cats[a["category_id"]], "bbox_px": _gt_bbox(a)}
                       for a in par_image.get(im["id"], [])]

    classes = sorted(meta["seuils_f1max"])
    decisions_path = Path(decisions_path)
    persiste = (yaml.safe_load(decisions_path.read_text(encoding="utf-8")) or {}
                if decisions_path.exists() else {})
    decisions = persiste.get("detections", {})
    ajouts = persiste.get("ajouts", {})
    for e in decisions.values():  # décisions d'une version polygone de l'app -> bbox
        if "geometrie_px" in e:
            xs, ys = zip(*e.pop("geometrie_px"))
            e["bbox_px"] = [min(xs), min(ys), max(xs), max(ys)]
    for a in ajouts.values():
        if "poly_px" in a:
            xs, ys = zip(*a.pop("poly_px"))
            a["bbox_px"] = [min(xs), min(ys), max(xs), max(ys)]

    def sauver():
        decisions_path.write_text(
            yaml.safe_dump({"detections": decisions, "ajouts": ajouts},
                           allow_unicode=True, sort_keys=True),
            encoding="utf-8")

    def dets_de(cle):
        return tuiles.get(cle, {}).get("detections", [])

    def retenues_restantes(cle):
        return [d for d in dets_de(cle)
                if d["retenu"] and d["uid"] not in decisions]

    app = FastAPI()

    @app.middleware("http")
    async def sans_cache(request, call_next):
        # outil local : le cache navigateur ne fait que servir de vieux fronts
        rep = await call_next(request)
        rep.headers["Cache-Control"] = "no-store"
        return rep

    @app.get("/api/tuiles")
    def api_tuiles(split: str = "", zone: str = "", classe: str = ""):
        res = []
        for cle, t in tuiles.items():
            if split and t["split"] != split:
                continue
            if zone and t["zone"] != zone:
                continue
            dets = t["detections"]
            if classe and not any(d["classe"] == classe for d in dets):
                continue
            retenues = [d for d in dets if d["retenu"]]
            res.append({"cle": cle, "split": t["split"], "zone": t["zone"],
                        "n_det": len(dets), "n_retenues": len(retenues),
                        "n_ajouts": sum(1 for a in ajouts.values()
                                        if a["tuile"] == cle),
                        "restantes": len(retenues_restantes(cle)),
                        "score_max": max((d["score"] for d in retenues),
                                         default=0.0),
                        "non_gt": sum(1 for d in retenues if not d["gt_apparie"])})
        res.sort(key=lambda x: x["cle"])
        return {"tuiles": res, "classes": classes,
                "zones": sorted({t["zone"] for t in tuiles.values()}),
                "splits": meta["splits"], "seuils": meta["seuils_f1max"],
                "modele": meta["modele"]}

    def _voisines(cle):
        info = infos_tuile[cle]
        res = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                vcle = grille.get((info["dataset"], info["row"] + dr,
                                   info["col"] + dc))
                if vcle:
                    res.append({"dr": dr, "dc": dc, "cle": vcle,
                                "detections": [d for d in dets_de(vcle)
                                               if d["retenu"]],
                                "gt": gt.get(vcle, [])})
        return res

    @app.get("/api/tuile/{split}/{fichier}")
    def api_tuile(split: str, fichier: str):
        cle = f"{split}/{fichier}"
        if cle not in infos_tuile:
            raise HTTPException(404, cle)
        t = tuiles.get(cle, {})
        return {"cle": cle, "zone": t.get("zone", infos_tuile[cle]["zone"]),
                "split": split, "n_gt": len(gt.get(cle, [])),
                "detections": [{**d, "decision": decisions.get(d["uid"])}
                               for d in dets_de(cle)],
                "ajouts": {u: a for u, a in ajouts.items() if a["tuile"] == cle},
                "gt": gt.get(cle, []), "voisines": _voisines(cle)}

    @app.get("/api/image/{split}/{fichier}")
    def api_image(split: str, fichier: str, contexte: int = 1):
        cle = f"{split}/{fichier}"
        info = infos_tuile.get(cle)
        if info is None:
            raise HTTPException(404, cle)
        if not contexte:
            png = io.BytesIO()
            Image.open(info["fichier"]).convert("L").save(png, format="PNG")
            return Response(png.getvalue(), media_type="image/png",
                            headers={"X-Decalage": "[0,0]"})
        mos = Image.new("L", (3 * TUILE_PX, 3 * TUILE_PX), GRIS_MANQUANT)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                vcle = grille.get((info["dataset"], info["row"] + dr,
                                   info["col"] + dc))
                if vcle:
                    mos.paste(Image.open(infos_tuile[vcle]["fichier"]).convert("L"),
                              ((dc + 1) * TUILE_PX, (dr + 1) * TUILE_PX))
        png = io.BytesIO()
        mos.save(png, format="PNG")
        return Response(png.getvalue(), media_type="image/png",
                        headers={"X-Decalage": f"[{TUILE_PX},{TUILE_PX}]"})

    @app.post("/api/decision")
    def api_decision(corps: dict):
        uid = corps["uid"]  # "<split>:<fichier>:<i>" -> la tuile se déduit de l'uid
        split, fichier, _ = uid.rsplit(":", 2) if uid.count(":") >= 2 else ("", "", "")
        det = next((d for d in dets_de(f"{split}/{fichier}") if d["uid"] == uid), None)
        if det is None:
            raise HTTPException(404, uid)
        decision = corps["decision"]
        if decision not in ("valide", "invalide", "editee"):
            raise HTTPException(422, decision)
        classe = corps.get("classe", det["classe"])
        if classe not in classes:
            raise HTTPException(422, classe)
        entree = {"uid": uid, "decision": decision, "classe": classe,
                  "horodatage": datetime.datetime.now().isoformat(timespec="seconds")}
        if decision == "editee":
            b = corps.get("bbox_px")
            if not _bbox_valide(b):
                raise HTTPException(422, "bbox éditée invalide")
            entree["bbox_px"] = [round(float(v), 1) for v in b]
        decisions[uid] = entree
        sauver()
        return {"ok": True}

    @app.post("/api/ajout")
    def api_ajout(corps: dict):
        if "supprimer" in corps:
            if ajouts.pop(corps["supprimer"], None) is None:
                raise HTTPException(404, corps["supprimer"])
            sauver()
            return {"ok": True}
        cle, classe, b = corps["tuile"], corps["classe"], corps.get("bbox_px")
        if cle not in infos_tuile:
            raise HTTPException(404, cle)
        if classe not in classes or not _bbox_valide(b):
            raise HTTPException(422, "classe ou bbox invalide")
        uid = f"ajout:{max((int(u.split(':')[1]) for u in ajouts), default=0) + 1:04d}"
        ajouts[uid] = {"tuile": cle, "classe": classe,
                       "bbox_px": [round(float(v), 1) for v in b],
                       "horodatage": datetime.datetime.now().isoformat(timespec="seconds")}
        sauver()
        return {"ok": True, "uid": uid}

    @app.get("/api/progression")
    def api_progression():
        toutes = [d for cle in tuiles for d in dets_de(cle)]
        retenues = [d for d in toutes if d["retenu"]]
        return {"retenues": len(retenues),
                "decidees_retenues": sum(1 for d in retenues
                                         if d["uid"] in decisions),
                "decidees_total": len(decisions), "ajouts": len(ajouts),
                "tuiles_finies": sum(1 for cle in tuiles
                                     if not retenues_restantes(cle)),
                "tuiles_total": len(tuiles),
                "par_decision": {d: sum(1 for v in decisions.values()
                                        if v["decision"] == d)
                                 for d in ("valide", "invalide", "editee")}}

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static",
                               html=True), name="static")
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus")
    ap.add_argument("detections", help="detections.json de tools/inferer_corpus.py")
    ap.add_argument("--decisions", default=None)
    ap.add_argument("--port", type=int, default=5176)
    args = ap.parse_args()
    decisions = args.decisions or (Path(args.detections).parent
                                   / "detections_decisions.yaml")
    app = creer_app(args.corpus, args.detections, decisions)
    print(f"Décisions : {decisions}\nhttp://127.0.0.1:{args.port}")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
