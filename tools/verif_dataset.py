"""Boucle de vérification indépendante d'un dataset produit par slice_zone.

Règle utilisateur du 2026-07-27 : produire -> vérifier (sur les FICHIERS) ->
corriger -> reproduire. Ce contrôleur ne partage aucun état avec slice_zone : il
recalcule tout depuis le disque et le GPKG source. Invariants contrôlés :
manifeste ≡ disque, noms uniques, blocs mono-split recalculés (anti-fuite),
hashes des splits, COCO ≡ manifeste (catégories, comptes par classe et par image),
chaque classe présente dans les 3 splits, pureté des négatifs (aucune entité —
entraînée OU ignorée — dans leurs emprises).

La configuration vient du split_manifest.yaml (config résolue embarquée) ;
--gpkg permet de pointer la copie locale si le chemin enregistré a bougé.

Usage :
    .venv\\Scripts\\python.exe tools\\verif_dataset.py <dossier_dataset> [--gpkg <chemin>]
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import yaml
from shapely.geometry import box as boite

from slice_zone import ORDRE_SPLITS, preparer_entites


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", help="dossier du dataset (sortie de slice_zone)")
    p.add_argument("--gpkg", default=None,
                   help="chemin local du GPKG source (défaut : celui du manifeste)")
    args = p.parse_args()
    d = Path(args.dataset)

    m = yaml.safe_load((d / "split_manifest.yaml").read_text(encoding="utf-8"))
    cfg = m["config"]
    tuiles = m["tuiles"]
    gpkg = Path(args.gpkg or cfg["gpkg"])
    classes = []
    for spec in cfg["couches"].values():
        if not spec.get("ignorer") and spec["classe"] not in classes:
            classes.append(spec["classe"])

    disque = {s: {f.name for f in (d / s).glob("*.png")} for s in ORDRE_SPLITS}
    manif = {s: {t["nom"] for t in tuiles if t["split"] == s} for s in ORDRE_SPLITS}
    assert disque == manif, "disque != manifeste"
    noms = [t["nom"] for t in tuiles]
    assert len(noms) == len(set(noms)), "nom dupliqué"

    blocs = {}
    for t in tuiles:
        cx = (t["bounds"][0] + t["bounds"][2]) / 2
        cy = (t["bounds"][1] + t["bounds"][3]) / 2
        bloc = [math.floor(cx / cfg["bloc_m"]), math.floor(cy / cfg["bloc_m"])]
        assert bloc == t["bloc"], f"bloc recalculé divergent : {t['nom']}"
        blocs.setdefault(tuple(bloc), set()).add(t["split"])
    assert all(len(v) == 1 for v in blocs.values()), "bloc multi-split : FUITE"

    for s in ORDRE_SPLITS:
        h = hashlib.sha1("\n".join(sorted(manif[s])).encode("utf-8")).hexdigest()
        assert h == m["hashes_sha1"][s], f"hash {s} divergent"
        cc = json.loads((d / s / "_annotations.coco.json").read_text(encoding="utf-8"))
        assert {c["name"] for c in cc["categories"]} == set(classes), f"catégories {s}"
        assert {i["file_name"] for i in cc["images"]} == disque[s]
        cats = {c["id"]: c["name"] for c in cc["categories"]}
        comptes = Counter(cats[a["category_id"]] for a in cc["annotations"])
        assert dict(comptes) == m["comptes"][s], f"comptes {s}"
        n_par_img = Counter(a["image_id"] for a in cc["annotations"])
        par_nom = {i["id"]: i["file_name"] for i in cc["images"]}
        manif_n = {t["nom"]: t["n_annotations"] for t in tuiles if t["split"] == s}
        for iid, nom in par_nom.items():
            assert n_par_img.get(iid, 0) == manif_n[nom], f"n_annotations {nom}"
        for classe in classes:
            assert comptes.get(classe, 0) > 0, f"classe {classe} absente de {s}"

    entites = []
    for couche, spec in cfg["couches"].items():
        entites += preparer_entites(gpd.read_file(gpkg, layer=couche),
                                    spec.get("buffer_m"))
    union = gpd.GeoSeries(entites).union_all()
    negs = [t for t in tuiles if t["n_annotations"] == 0]
    touches = [t["nom"] for t in negs if boite(*t["bounds"]).intersects(union)]
    assert not touches, f"négatif(s) impurs : {touches[:5]}"

    n = {s: len(disque[s]) for s in ORDRE_SPLITS}
    tot = sum(n.values())
    print(f"vérification {m['dataset']} : CONFORME —",
          {s: f"{v} ({v / tot:.1%})" for s, v in n.items()},
          f"| {len(blocs)} blocs mono-split | {len(negs)} négatifs purs "
          "(couches ignorées comprises)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        sys.exit(f"vérification : DIVERGENCE — {exc}")
