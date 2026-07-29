"""PORTE DU BANC : tools/bench/decode.py aux défauts doit reproduire le plugin.

decode.py est un port paramétré de `_run_rfdetr_seg_with_sahi`. C'est le SEUL fork du
banc, donc le seul endroit où une dérive avec la production peut entrer. Si ce test
échoue, tout ce que mesure le banc décrit une réimplémentation, pas la chaîne réelle.

Les deux chemins tournent dans le même process, sur la même session ONNX, sans cache :
l'égalité attendue est EXACTE, pas approchée.

Lancement (interpréteur du plugin, qui a onnxruntime) :
  & "$PLUGIN\dev\runner_onnx\.venv_onnx\Scripts\python.exe" tests\test_parity_bench.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PLUGIN = Path(os.environ.get(
    "ARCHEO_PLUGIN",
    r"C:\Users\valen\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\archeologia-pipeline"))
DATA = Path(os.environ.get("BENCH_DATA", r"D:\bench\data\test"))
MODELE = Path(os.environ.get(
    "BENCH_MODEL", PLUGIN / "data" / "models" / "lineaires_seg_v2_1" / "weights" / "best.onnx"))

sys.path.insert(0, str(PLUGIN / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from pipeline.cv.computer_vision_onnx import (  # noqa: E402
    _load_onnx_model, _preprocess_image, _run_rfdetr_seg_with_sahi,
)
from pipeline.cv.sahi_lite import slice_image as sahi_slice  # noqa: E402
from tools.bench.data import Corpus  # noqa: E402
from tools.bench.decode import Params, SliceOut, run as decoder  # noqa: E402

CONF = float(os.environ.get("BENCH_CONF", "0.3"))
CLASS_OFFSET = 0
SLICE = 648
OVERLAP = 0.2

# Après application du patch, le plugin ne doit plus reproduire les DÉFAUTS mais les
# correctifs. On relance alors le même test avec, par exemple :
#     BENCH_PARAMS="boxes_normalisees=true,sahi_dedup=true"
# et l'égalité exacte doit tenir contre decode.py configuré de la même façon. C'est le
# test de non-régression du patch : il prouve que le plugin patché fait ce que le banc a
# mesuré, et pas autre chose.
def _params_env() -> dict:
    brut = os.environ.get("BENCH_PARAMS", "").strip()
    if not brut:
        return {}
    out: dict = {}
    for morceau in brut.split(","):
        k, _, v = morceau.partition("=")
        k, v = k.strip(), v.strip()
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif v.lower() in ("none", "null"):
            out[k] = None
        else:
            try:
                out[k] = int(v) if v.isdigit() else float(v)
            except ValueError:
                out[k] = v
    return out


PARAMS_SUP = _params_env()


def echantillon(n: int = 20):
    """Tuiles déterministes : >=2 par zone, >=3 négatives."""
    corpus = Corpus(DATA)
    par_zone, negatives = {}, []
    for iid in sorted(corpus.images):
        img = corpus.images[iid]
        z = corpus.zone(img)
        if not corpus.anns.get(iid):
            negatives.append(img)
        else:
            par_zone.setdefault(z, []).append(img)
    choix = []
    for z in sorted(par_zone):
        pas = max(1, len(par_zone[z]) // 4)
        choix.extend(par_zone[z][::pas][:4])
    choix.extend(negatives[::max(1, len(negatives) // 3)][:3])
    return corpus, choix[:n]


def compare(a: dict, b: dict, i: int) -> list[str]:
    err = []
    if a["class_id"] != b["class_id"]:
        err.append(f"det[{i}].class_id {a['class_id']} != {b['class_id']}")
    if abs(a["confidence"] - b["confidence"]) > 1e-6:
        err.append(f"det[{i}].confidence {a['confidence']!r} != {b['confidence']!r}")
    if len(a["polygon"]) != len(b["polygon"]):
        err.append(f"det[{i}].polygon longueur {len(a['polygon'])} != {len(b['polygon'])}")
    elif a["polygon"] != b["polygon"]:                     # égalité EXACTE attendue
        d = max(abs(x - y) for x, y in zip(a["polygon"], b["polygon"]))
        err.append(f"det[{i}].polygon differe (ecart max {d:.3e})")
    if a["bbox"] != b["bbox"]:
        err.append(f"det[{i}].bbox {a['bbox']} != {b['bbox']}")
    if abs(a["area"] - b["area"]) > 1e-9:
        err.append(f"det[{i}].area {a['area']} != {b['area']}")
    if sorted(map(len, a.get("polygon_holes", []))) != sorted(map(len, b.get("polygon_holes", []))):
        err.append(f"det[{i}].polygon_holes different")
    return err


def main() -> int:
    corpus, images = echantillon()
    session, input_name, shape, _ = _load_onnx_model(str(MODELE))
    mw, mh = int(shape[3]), int(shape[2])
    print(f"modele {MODELE.name}  entree {mw}x{mh}  provider {session.get_providers()[0]}")
    print(f"{len(images)} tuiles, conf={CONF}, class_offset={CLASS_OFFSET}\n")

    total_dets, echecs, lignes, ecarts16 = 0, [], [], []
    for img in images:
        chemin = corpus.chemin(img)
        pil = Image.open(chemin).convert("RGB")
        w, h = pil.size

        ref = _run_rfdetr_seg_with_sahi(
            pil, session, input_name, mw, mh, SLICE, SLICE, OVERLAP, CONF,
            class_offset=CLASS_OFFSET)

        # Le banc doit voir EXACTEMENT les mêmes tuiles que le plugin. Sur une image
        # 648x648 le slicer en renvoie 4 identiques (stride 519 < 648, la fenêtre
        # débordante est ramenée en arrière sur [0,0,648,648]) — et ces passes en double
        # ne sont pas neutres : elles refont la recherche de clé et fusionnent des
        # instances. Reproduire le découpage fait partie de la parité.
        sliced, oh, ow = sahi_slice(image=np.asarray(pil), slice_height=SLICE,
                                    slice_width=SLICE, overlap_height_ratio=OVERLAP,
                                    overlap_width_ratio=OVERLAP)
        if PARAMS_SUP.get("sahi_dedup"):
            # Le patch déduplique côté plugin : le banc doit voir la même liste.
            vus, uniques = set(), []
            for s in sliced:
                k = (tuple(s.starting_pixel), s.image.shape)
                if k not in vus:
                    vus.add(k)
                    uniques.append(s)
            sliced = uniques
        slices = []
        for s in sliced:
            sp = Image.fromarray(s.image)
            out = session.run(None, {input_name: _preprocess_image(sp, (mw, mh), "rfdetr")})
            slices.append(SliceOut(s.starting_pixel[0], s.starting_pixel[1],
                                   sp.size[0], sp.size[1],
                                   out[0][0].astype(np.float32),
                                   out[1][0].astype(np.float32),
                                   out[2][0].astype(np.float32)))
        n_slices = len(slices)
        obt = decoder(slices, ow, oh, mw, mh, Params(confidence=CONF, class_offset=CLASS_OFFSET, **PARAMS_SUP))

        total_dets += len(ref)
        err = []
        if len(ref) != len(obt):
            err.append(f"nombre de detections {len(ref)} (plugin) != {len(obt)} (banc)")
        else:
            for i, (a, b) in enumerate(zip(ref, obt)):
                err.extend(compare(a, b, i))

        # (b) chemin CACHE : un seul forward répliqué sur les N fenêtres identiques,
        #     masques stockés en float16. C'est ce qui économise 4x le GPU — il faut
        #     donc prouver que ça ne change pas la sortie.
        s0 = slices[0]
        rep = [SliceOut(0, 0, w, h, s0.boxes, s0.logits, s0.masks) for _ in slices]
        obt_rep = decoder(rep, w, h, mw, mh, Params(confidence=CONF, class_offset=CLASS_OFFSET, **PARAMS_SUP))
        m16 = s0.masks.astype(np.float16).astype(np.float32)
        rep16 = [SliceOut(0, 0, w, h, s0.boxes, s0.logits, m16) for _ in slices]
        obt_16 = decoder(rep16, w, h, mw, mh, Params(confidence=CONF, class_offset=CLASS_OFFSET, **PARAMS_SUP))
        if len(obt_rep) != len(ref):
            err.append(f"replication 1 forward : {len(obt_rep)} det != {len(ref)}")
        n_ecart16 = abs(len(obt_16) - len(ref))
        ecarts16.append(n_ecart16)

        etat = "OK  " if not err else "ECHEC"
        lignes.append(f"  {etat} {img['file_name']:<40} {len(ref):>3} det  "
                      f"{n_slices} tuiles SAHI  cache_fp16 {'=' if not n_ecart16 else '~'}")
        if err:
            echecs.append((img["file_name"], err[:5]))

    print("\n".join(lignes))
    print(f"\n{total_dets} detections comparees sur {len(images)} tuiles")
    print(f"cache fp16 : {sum(1 for e in ecarts16 if e == 0)}/{len(ecarts16)} tuiles "
          f"identiques, ecart total {sum(ecarts16)} detection(s)")
    if echecs:
        print(f"\n{len(echecs)} tuile(s) en echec :")
        for nom, err in echecs:
            print(f"  {nom}")
            for e in err:
                print(f"     - {e}")
        print("\nPARITE ROUGE — ne rien mesurer tant que ce test n'est pas vert.")
        return 1
    print("\nPARITE VERTE : decode.py aux defauts reproduit le plugin a l'identique.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
