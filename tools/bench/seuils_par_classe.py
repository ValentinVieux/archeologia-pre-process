"""Optimum du seuil de confiance CLASSE PAR CLASSE.

Le plugin n'applique pas de seuil par classe : quand plusieurs entités d'un même modèle
sont dans un run, `model_orchestrator` écrase les surcharges par leur MINIMUM
(l. 747), et le décodage ONNX ne connaît qu'un seuil global. Obtenir un seuil réellement
différent par classe exigerait un run séparé par entité, soit une passe d'inférence
complète chacun.

Ce script répond à la seule question qui décide si ça vaut ce prix : les classes ont-elles
des optima franchement différents, ou tombent-elles toutes au même endroit ?

    python -m tools.bench.seuils_par_classe --data /data/valid --cle <cle> [--subset ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench import metrics as M                        # noqa: E402
from tools.bench.cache import Cache                          # noqa: E402
from tools.bench.data import Corpus                          # noqa: E402
from tools.bench.decode import Params, run as decoder        # noqa: E402
from tools.bench.__main__ import slices_niveau_a             # noqa: E402

SEUILS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
# Aires en m² ; le décodage travaille en pixels et le GSD vaut 0,5 m -> 1 m² = 4 px².
AIRES_M2 = [0.0, 50.0, 100.0, 200.0, 300.0, 500.0, 800.0]
_CTX: dict = {}


def _init(data: str, cache_dir: str, cle: str, tau: float, axe: str, conf: float) -> None:
    cv2.setNumThreads(1)
    _CTX.update(corpus=Corpus(Path(data)), cache=Cache(Path(cache_dir), cle), tau=tau,
                axe=axe, conf=conf)


def _une_image(iid: int):
    corpus, cache, tau = _CTX["corpus"], _CTX["cache"], _CTX["tau"]
    img = corpus.images[iid]
    unite = Path(img["file_name"]).stem
    if not cache.existe(unite):
        return None
    anns = corpus.anns.get(iid, [])
    if not anns:
        return None
    h, w = img["height"], img["width"]
    sl = cache.lire(unite, 0, 0, w, h)

    # Côté GT par classe, calculé UNE fois : il ne dépend d'aucun seuil.
    cotes = {}
    for cid in range(len(corpus.noms_classes)):
        cat = corpus.classe_vers_cat[cid]
        gt = M.masque_coco(anns, h, w, classes=[cat])
        if gt.any():
            cotes[cid] = M.CoteGT(gt)
    if not cotes:
        return None

    valeurs = SEUILS if _CTX["axe"] == "confiance" else AIRES_M2
    out: Dict[tuple, dict] = {}
    for v in valeurs:
        if _CTX["axe"] == "confiance":
            p = Params(confidence=v, class_offset=0,
                       boxes_normalisees=True, sahi_dedup=True)
        else:
            # 1 m² = 4 px² à 0,5 m/px.
            p = Params(confidence=_CTX["conf"], class_offset=0, min_area_px=v * 4.0,
                       boxes_normalisees=True, sahi_dedup=True)
        dets = decoder(slices_niveau_a(sl, w, h, p), w, h, w, h, p)
        for cid, cote in cotes.items():
            pred = M.masque_detections(dets, h, w, classes=[cid])
            c = M.ccq_prepare(pred, cote, tau)
            c["n_pred"] = sum(1 for d in dets if d["class_id"] == cid)
            out[(cid, v)] = c
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--subset")
    ap.add_argument("--cle", required=True)
    ap.add_argument("--out", default=os.environ.get("BENCH_OUT", "/out/bench"))
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--axe", default="confiance", choices=["confiance", "aire"])
    ap.add_argument("--conf", type=float, default=0.25,
                    help="seuil fixe quand --axe aire")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()

    valeurs = SEUILS if a.axe == "confiance" else AIRES_M2
    corpus = Corpus(Path(a.data))
    ids = (json.loads(Path(a.subset).read_text(encoding="utf-8"))["image_ids"]
           if a.subset else sorted(corpus.images))
    acc: Dict[tuple, List[dict]] = {}
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init,
                             initargs=(a.data, str(Path(a.out) / "cache"), a.cle, a.tau,
                                       a.axe, a.conf)) as ex:
        for res in ex.map(_une_image, ids, chunksize=4):
            if res:
                for k, v in res.items():
                    acc.setdefault(k, []).append(v)

    unite = "" if a.axe == "confiance" else " m2"
    titre = ("Seuil de confiance optimal par classe" if a.axe == "confiance"
             else f"Aire minimale optimale par classe (a confiance {a.conf})")
    print(f"\n{titre} — {len(ids)} tuiles, tau={a.tau} m\n")
    fmt = "{:>8.2f}" if a.axe == "confiance" else "{:>8.0f}"
    entete = "classe".ljust(15) + "".join(fmt.format(v) for v in valeurs) + "   optimum"
    print(entete)
    print("-" * len(entete))
    resultats = {}
    for cid, nom in enumerate(corpus.noms_classes):
        ligne, npoly, best = [], [], (None, -1.0)
        for v in valeurs:
            parts = acc.get((cid, v), [])
            if not parts:
                ligne.append(float("nan"))
                npoly.append(float("nan"))
                continue
            g = M.agreger_ccq(parts, a.tau)
            ligne.append(g["f1_len"])
            npoly.append(sum(p.get("n_pred", 0) for p in parts))
            if g["f1_len"] > best[1]:
                best = (v, g["f1_len"])
        resultats[nom] = {"f1_par_valeur": dict(zip(map(str, valeurs), ligne)),
                          "polygones_par_valeur": dict(zip(map(str, valeurs), npoly)),
                          "optimum": best[0], "f1_optimum": best[1],
                          "n_images": len(acc.get((cid, valeurs[0]), []))}
        print(nom.ljust(15) + "".join(f"{x:>8.3f}" for x in ligne)
              + f"   {best[0]:.2f}{unite} ({best[1]:.3f})")
    if a.axe == "aire":
        print("\npolygones rendus (somme sur le sous-ensemble) :")
        for nom, r in resultats.items():
            print(nom.ljust(15) + "".join(f"{x:>8.0f}" for x in r["polygones_par_valeur"].values()))

    p = Path(a.out) / f"{'seuils' if a.axe == 'confiance' else 'aires'}_par_classe.json"
    p.write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
