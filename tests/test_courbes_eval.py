"""Auto-test sans GPU de l'éval outillée : .venv\\Scripts\\python.exe tests\\test_courbes_eval.py

Couvre les briques importables sans torch/matplotlib : prf, ap50, bloc_metriques,
resumer (schéma metriques_eval/1), empreinte de cache, et le dashboard
tableau_modeles sur des fixtures synthétiques. L'inférence GPU elle-même est
validée par la boucle de rétrofit (re-rendu --adopter-cache vs chiffres legacy).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import courbes_eval as ce  # noqa: E402
import tableau_modeles as tm  # noqa: E402

# 3 GT (a×2, b×1), 2 TP de classe a (conf 0,9 et 0,6), 1 FP de classe b (conf 0,3)
ENREGS = [
    {"split": "valid", "zone": "za", "n_gt": 2, "gt_classes": ["a", "b"],
     "matches": [[0.9, 0.8, "a"]], "fps": [[0.3, "b"]]},
    {"split": "test", "zone": "zb", "n_gt": 1, "gt_classes": ["a"],
     "matches": [[0.6, 0.7, "a"]], "fps": []},
]


def approx(x, y, tol=1e-9):
    return abs(x - y) <= tol


def main() -> None:
    # --- grille : bornes et pas uniques -----------------------------------
    g = ce.grille(0.05)
    assert approx(g[0], 0.05) and approx(g[-1], 0.95) and approx(g[1] - g[0], 0.005), g

    # --- prf : valeurs à la main, convention P=1.0 quand TP+FP=0 ----------
    assert ce.prf(ENREGS, 0.5) == (1.0, 2 / 3, 2 * (2 / 3) / (1 + 2 / 3))
    p, r, f = ce.prf(ENREGS, 0.2)
    assert approx(p, 2 / 3) and approx(r, 2 / 3) and approx(f, 2 / 3), (p, r, f)
    p, r, f = ce.prf(ENREGS, 0.95)  # aucun TP ni FP au-dessus
    assert p == 1.0 and r == 0.0 and f == 0.0, (p, r, f)
    p, r, f = ce.prf(ENREGS, 0.2, classe="b")  # 1 FP, 0 TP
    assert p == 0.0 and r == 0.0 and f == 0.0, (p, r, f)

    # --- ap50 : toutes-points par rang de confiance ------------------------
    ap, rr, pp = ce.ap50(ENREGS)
    # rangs : TP@0,9 -> (R=1/3, P=1) ; TP@0,6 -> (2/3, 1) ; FP@0,3 -> (2/3, 2/3)
    assert approx(ap, 2 / 3), ap
    assert approx(rr[-1], 2 / 3) and approx(pp[0], 1.0), (rr, pp)
    sans_fp = [dict(e, fps=[]) for e in ENREGS]  # cas trivial : AP = rappel max
    assert approx(ce.ap50(sans_fp)[0], 2 / 3)
    assert ce.ap50(ENREGS, classe="b")[0] == 0.0  # que des FP
    assert ce.ap50([], classe=None)[0] == 0.0  # vide

    # --- bloc_metriques : point F1-max ------------------------------------
    b = ce.bloc_metriques(ENREGS, 0.05)
    # F1 passe de 2/3 (FP@0,3 inclus) à 0,8 dès le premier seuil > 0,3 -> 0,305
    assert approx(b["seuil_f1max"], 0.305), b
    assert b["F1"] == 0.8 and b["P"] == 1.0 and approx(b["R"], 0.6667, 1e-4), b
    assert b["n_gt"] == 3 and b["iou_median"] == 0.75, b  # médiane de 0,8 et 0,7
    bb = ce.bloc_metriques(ENREGS, 0.05, classe="b")  # classe sans aucun TP
    assert bb["F1"] == 0.0 and bb["n_gt"] == 1 and bb["iou_median"] is None, bb

    # --- resumer : schéma canonique ---------------------------------------
    donnees = {"m1": {"decal": 0, "enregs": ENREGS}}
    meta_modeles = {"m1": {"poids": "D:/poids/best.pth", "resolution": 648,
                           "taille_octets": 42}}
    dataset = {"chemin": "D:/corpus", "splits": ["valid", "test"], "n_images": 2, "n_gt": 3}
    resume = ce.resumer(donnees, meta_modeles, "segmentation", dataset, {}, 0.05, "calculee")
    assert resume["schema"] == "metriques_eval/1"
    assert resume["iou"] == {"type": "masque", "seuil": 0.5}
    assert resume["p_sans_prediction"] == 1.0
    m1 = resume["modeles"]["m1"]
    assert m1["poids"] == "D:/poids/best.pth" and m1["class_offset"] == 0
    assert set(m1["par_classe"]) == {"a", "b"}, m1["par_classe"]  # toujours présent
    assert m1["par_zone"]["za"] == {"P": 1.0, "R": 0.5, "n_gt": 2}, m1["par_zone"]
    assert m1["par_zone"]["zb"]["R"] == 1.0
    resume_det = ce.resumer(donnees, meta_modeles, "detection", dataset, {}, 0.05, "calculee")
    assert resume_det["iou"]["type"] == "bbox"
    sans_zone = {"m1": {"decal": 1, "enregs": [dict(e, zone="") for e in ENREGS]}}
    r2 = ce.resumer(sans_zone, meta_modeles, "segmentation", dataset, {}, 0.05, "calculee")
    assert "par_zone" not in r2["modeles"]["m1"]  # par_zone ssi zones
    json.dumps(resume)  # sérialisable

    # --- empreinte de cache -----------------------------------------------
    modeles = {"m1": {"poids": "D:/inexistant/best.pth", "resolution": 648, "noms": None}}
    meta = ce.construire_meta(0.05, "D:/corpus", {}, modeles, tache="segmentation")
    assert meta["schema"] == "appariements/2"
    assert meta["modeles"]["m1"]["taille_octets"] is None  # poids illisible : toléré
    assert "\\" not in meta["coco"] and "\\" not in meta["modeles"]["m1"]["poids"]
    assert ce.meta_divergence(meta, dict(meta)) is None  # identique -> accepté
    autre = json.loads(json.dumps(meta))
    autre["modeles"]["m1"]["resolution"] = 504
    assert "résolution" in ce.meta_divergence(meta, autre)  # divergent -> refusé
    autre2 = json.loads(json.dumps(meta))
    autre2["modeles"]["m1"]["poids"] = autre2["modeles"]["m1"]["poids"].upper()
    assert ce.meta_divergence(meta, autre2) is None  # casse Windows ignorée
    autre3 = json.loads(json.dumps(meta))
    autre3["plancher"] = 0.1
    assert "plancher" in ce.meta_divergence(meta, autre3)

    # --- reprise par modèle (--reprendre-de) ------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        modeles2 = {"mA": {"poids": "D:/pA/best.pth", "resolution": 648, "noms": None},
                    "mB": {"poids": "D:/pB/best.pth", "resolution": 1032, "noms": None}}
        attendu = ce.construire_meta(0.05, "D:/corpus", {}, modeles2)
        src_a = tmp / "evalA"
        src_a.mkdir()
        meta_a = ce.construire_meta(0.05, "D:/corpus", {}, {"mA": modeles2["mA"]},
                                    tache="segmentation")
        src_a.joinpath("appariements.json").write_text(
            json.dumps({"_meta": meta_a, "mA": {"decal": 0, "enregs": ENREGS}}),
            encoding="utf-8")
        donnees2 = {}
        tache2 = ce.reprendre_modeles([str(src_a)], attendu, modeles2, donnees2, None)
        assert tache2 == "segmentation" and list(donnees2) == ["mA"], donnees2
        assert donnees2["mA"]["enregs"] == ENREGS  # appariements repris tels quels
        # source redondante (mA déjà couvert, mB absent) : tolérée, n'apporte rien
        tache2 = ce.reprendre_modeles([str(src_a)], attendu, modeles2, donnees2, tache2)
        assert list(donnees2) == ["mA"]
        # empreinte par-modèle divergente (résolution) sous le bon nom = refus
        meta_div = json.loads(json.dumps(meta_a))
        meta_div["modeles"]["mA"]["resolution"] = 504
        src_b = tmp / "evalB"
        src_b.mkdir()
        src_b.joinpath("appariements.json").write_text(
            json.dumps({"_meta": meta_div, "mA": {"decal": 0, "enregs": ENREGS}}),
            encoding="utf-8")
        try:
            ce.reprendre_modeles([str(src_b)], attendu, modeles2, {}, "segmentation")
            raise AssertionError("divergence par modèle non détectée")
        except SystemExit:
            pass
        # provenance de RUN différente (fusion) = refus
        meta_run = ce.construire_meta(0.05, "D:/corpus", {"x": "y"},
                                      {"mA": modeles2["mA"]}, tache="segmentation")
        src_c = tmp / "evalC"
        src_c.mkdir()
        src_c.joinpath("appariements.json").write_text(
            json.dumps({"_meta": meta_run, "mA": {"decal": 0, "enregs": ENREGS}}),
            encoding="utf-8")
        try:
            ce.reprendre_modeles([str(src_c)], attendu, modeles2, {}, None)
            raise AssertionError("divergence de run non détectée")
        except SystemExit:
            pass
        # cache legacy sans empreinte = refus explicite
        src_d = tmp / "evalD"
        src_d.mkdir()
        src_d.joinpath("appariements.json").write_text(
            json.dumps({"mA": {"decal": 0, "enregs": ENREGS}}), encoding="utf-8")
        try:
            ce.reprendre_modeles([str(src_d)], attendu, modeles2, {}, None)
            raise AssertionError("cache legacy accepté à tort")
        except SystemExit:
            pass
        # tâche incompatible = refus
        try:
            ce.reprendre_modeles([str(src_a)], attendu, modeles2, {}, "detection")
            raise AssertionError("tâche incompatible acceptée à tort")
        except SystemExit:
            pass

    # --- dashboard tableau_modeles ----------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for fam, run, nom, date, f1 in (
                ("enclos", "enclos_v1", "enclos_seg_v1", "2026-08-01T10:00:00", 0.61),
                ("enclos", "enclos_v2", "enclos_seg_v2", "2026-08-20T10:00:00", 0.71),
                ("fours", "fours_v1", "fours_seg_v1", "2026-08-24T10:00:00", 0.62)):
            d = tmp / fam / "runs" / "training" / run / "evaluation"
            d.mkdir(parents=True)
            bloc = dict(ce.bloc_metriques(ENREGS, 0.05), F1=f1)
            d.joinpath("metriques_eval.json").write_text(json.dumps({
                "schema": "metriques_eval/1", "genere_le": date, "tache": "segmentation",
                "modeles": {nom: {"global": bloc, "par_classe": {"a": bloc, "b": bloc}}},
            }), encoding="utf-8")
        # planches présentes pour enclos_v2 seulement -> bloc de courbes intégré
        (tmp / "enclos" / "runs" / "training" / "enclos_v2" / "evaluation"
         / "courbes_seuils_pr.png").write_bytes(b"png factice")
        # modèle PROVISOIRE : dans le tableau (étiqueté) mais HORS évolution
        prov = tmp / "fours" / "runs" / "training" / "fours_prov" / "evaluation"
        prov.mkdir(parents=True)
        bloc_p = dict(ce.bloc_metriques(ENREGS, 0.05), F1=0.5)
        prov.joinpath("metriques_eval.json").write_text(json.dumps({
            "schema": "metriques_eval/1", "genere_le": "2026-08-30T10:00:00",
            "tache": "segmentation",
            "modeles": {"fours_prov_x": {"global": bloc_p, "par_classe": {"a": bloc_p}}},
        }), encoding="utf-8")
        prov.joinpath("PROVISOIRE.txt").write_text("data retravaillées demain",
                                                   encoding="utf-8")

        vide = tmp / "verdun" / "runs" / "training" / "verdun_v1"
        vide.mkdir(parents=True)
        vide.joinpath("metrics.csv").write_text("epoch\n", encoding="utf-8")
        casse = tmp / "corrompu" / "runs" / "training" / "run_x" / "evaluation"
        casse.mkdir(parents=True)
        casse.joinpath("metriques_eval.json").write_text("{pas du json", encoding="utf-8")

        page = tm.construire(tmp)
        assert "enclos" in page and "fours" in page
        assert "enclos_seg_v1" in page and "enclos_seg_v2" in page
        assert "0.710" in page and "0.620" in page
        assert "enclos_seg_v1 → enclos_seg_v2" in page  # ordre des versions par date
        assert "<polyline" in page  # sparklines
        assert "verdun_v1" not in page, "liste des runs sans mesure supprimée (2026-09-02)"
        assert "sans mesure" in page.lower()  # le COMPTEUR reste dans l'en-tête
        # courbes intégrées : bloc repliable là où la planche existe, pas ailleurs
        assert "courbes — enclos_v2" in page and "loading='lazy'" in page
        assert "enclos_v2/evaluation/courbes_seuils_pr.png'" in page
        assert "courbes — enclos_v1" not in page  # pas de planche -> pas de bloc
        assert "fours_prov_x" in page and "[provisoire]" in page  # étiqueté au tableau
        assert "fours_prov_x → " not in page and "→ fours_prov_x" not in page, \
            "un modèle provisoire ne doit pas entrer dans l'évolution"
        assert "corrompu" in page and "⚠" in page  # fichier cassé = warning, pas crash

    print("OK — courbes_eval (prf/ap50/resumer/empreinte/reprendre-de) + tableau_modeles")


if __name__ == "__main__":
    main()
