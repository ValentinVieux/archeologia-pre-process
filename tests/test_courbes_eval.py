"""Auto-test sans GPU de l'éval outillée : .venv\\Scripts\\python.exe tests\\test_courbes_eval.py

Couvre les briques importables sans torch/matplotlib : prf, ap50, bloc_metriques,
par_zone_classe, resumer (schéma metriques_eval/1), empreinte de cache, le
complément d'une éval antérieure (completer_metriques_eval + verif_courbes_eval en
sous-processus) et le dashboard tableau_modeles sur des fixtures synthétiques.
L'inférence GPU elle-même est validée par la boucle de rétrofit (re-rendu
--adopter-cache vs chiffres legacy).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import courbes_eval as ce  # noqa: E402
import completer_metriques_eval as cm  # noqa: E402
import tableau_modeles as tm  # noqa: E402

# 3 GT (a×2, b×1), 2 TP de classe a (conf 0,9 et 0,6), 1 FP de classe b (conf 0,3)
ENREGS = [
    {"split": "valid", "zone": "za", "n_gt": 2, "gt_classes": ["a", "b"],
     "matches": [[0.9, 0.8, "a"]], "fps": [[0.3, "b"]]},
    {"split": "test", "zone": "zb", "n_gt": 1, "gt_classes": ["a"],
     "matches": [[0.6, 0.7, "a"]], "fps": []},
]

# 2 zones × 2 classes pour par_zone_classe (7 GT : za a×2 b×2, zb a×1 b×2)
ENREGS_ZC = [
    {"split": "valid", "zone": "za", "n_gt": 3, "gt_classes": ["a", "a", "b"],
     "matches": [[0.9, 0.8, "a"], [0.4, 0.6, "a"]], "fps": [[0.5, "b"], [0.2, "a"]]},
    {"split": "valid", "zone": "za", "n_gt": 1, "gt_classes": ["b"],
     "matches": [[0.3, 0.7, "b"]], "fps": []},
    {"split": "test", "zone": "zb", "n_gt": 1, "gt_classes": ["a"],
     "matches": [], "fps": [[0.7, "a"], [0.35, "b"]]},
    {"split": "test", "zone": "zb", "n_gt": 2, "gt_classes": ["b", "b"],
     "matches": [[0.6, 0.9, "b"]], "fps": []},
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

    # --- par_zone_classe : zone × classe, valeurs à la main (2026-09-03) ---
    # seuil global 0,45 ; seuils de classe a 0,35 / b 0,55 ; classe c absente partout
    zc = ce.par_zone_classe(ENREGS_ZC, 0.45, {"a": 0.35, "b": 0.55, "c": 0.5}, ["a", "b", "c"])
    assert list(zc) == ["za", "zb"] and list(zc["za"]) == ["a", "b", "c"], zc
    # za/a : matches 0,9 et 0,4, fp 0,2 -> tp 1 (0,9) fp 0 ; @0,35 tp 2 ; R_max 2/2
    assert zc["za"]["a"] == {"n_gt": 2, "tp": 1, "fp": 0, "R": 0.5, "P": 1.0,
                             "R_seuil_classe": 1.0, "fp_seuil_classe": 0, "R_max": 1.0}, zc
    # za/b : match 0,3, fp 0,5 -> tp 0 fp 1 (P 0,0) ; @0,55 fp 0 ; R_max 1/2
    assert zc["za"]["b"] == {"n_gt": 2, "tp": 0, "fp": 1, "R": 0.0, "P": 0.0,
                             "R_seuil_classe": 0.0, "fp_seuil_classe": 0, "R_max": 0.5}, zc
    # zb/a : aucun match, fp 0,7 -> R_max 0 ; fp_seuil_classe 1 (0,7 >= 0,35)
    assert zc["zb"]["a"] == {"n_gt": 1, "tp": 0, "fp": 1, "R": 0.0, "P": 0.0,
                             "R_seuil_classe": 0.0, "fp_seuil_classe": 1, "R_max": 0.0}, zc
    # zb/b : match 0,6, fp 0,35 -> tp 1 fp 0 ; @0,55 tp 1 fp 0 ; R_max 1/2
    assert zc["zb"]["b"] == {"n_gt": 2, "tp": 1, "fp": 0, "R": 0.5, "P": 1.0,
                             "R_seuil_classe": 0.5, "fp_seuil_classe": 0, "R_max": 0.5}, zc
    # classe sans GT ni prédiction : n_gt 0 -> R/P/R_max null (pas la convention P=1)
    assert zc["za"]["c"] == {"n_gt": 0, "tp": 0, "fp": 0, "R": None, "P": None,
                             "R_seuil_classe": None, "fp_seuil_classe": 0, "R_max": None}, zc
    assert sum(b["n_gt"] for b in zc["za"].values()) == 4  # = n_gt de la zone
    assert ce.par_zone_classe([dict(e, zone="") for e in ENREGS_ZC], 0.45, {}, []) == {}
    # resumer : bloc présent ssi zones, seuils = global 0,305 / classes a,b 0,05
    zc1 = m1["par_zone_classe"]
    assert list(zc1) == ["za", "zb"]
    assert zc1["za"]["a"] == {"n_gt": 1, "tp": 1, "fp": 0, "R": 1.0, "P": 1.0,
                              "R_seuil_classe": 1.0, "fp_seuil_classe": 0, "R_max": 1.0}, zc1
    assert zc1["za"]["b"] == {"n_gt": 1, "tp": 0, "fp": 0, "R": 0.0, "P": None,
                              "R_seuil_classe": 0.0, "fp_seuil_classe": 1, "R_max": 0.0}, zc1
    assert zc1["zb"]["b"] == {"n_gt": 0, "tp": 0, "fp": 0, "R": None, "P": None,
                              "R_seuil_classe": None, "fp_seuil_classe": 0, "R_max": None}, zc1
    assert "par_zone_classe" not in r2["modeles"]["m1"]

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

    # --- completer_metriques_eval + verif_courbes_eval (éval antérieure) ---
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tmp.joinpath("appariements.json").write_text(
            json.dumps({"_meta": meta, "m1": {"decal": 0, "enregs": ENREGS}}), encoding="utf-8")
        ancien = json.loads(json.dumps(resume))
        del ancien["modeles"]["m1"]["par_zone_classe"]  # éval d'avant le 2026-09-03
        tmp.joinpath("metriques_eval.json").write_text(
            json.dumps(ancien, ensure_ascii=False, indent=1), encoding="utf-8")
        verif = [sys.executable, str(ROOT / "tools" / "verif_courbes_eval.py"), str(tmp)]
        completer = [sys.executable, str(ROOT / "tools" / "completer_metriques_eval.py"), str(tmp)]

        def run(cmd):
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            return r.returncode, r.stdout + r.stderr

        rc, out = run(verif)  # bloc absent = avertissement, verdict CONFORME
        assert rc == 0 and "CONFORME" in out and "par_zone_classe absent" in out, out
        rc, out = run(completer + ["--out", str(tmp / "out")])
        assert rc == 0, out
        complet = json.loads((tmp / "out" / "metriques_eval.json").read_text(encoding="utf-8"))
        assert complet["modeles"]["m1"]["par_zone_classe"] == zc1, complet
        assert complet["complete_le"]["outil"] == "tools/completer_metriques_eval.py"
        assert complet["complete_le"]["par_zone_classe"][:4] == "2026"
        assert cm.sans_bloc(complet) == ancien  # tout le reste identique
        rc, out = run(completer)  # en place
        assert rc == 0, out
        rc, out = run(verif)
        assert rc == 0 and "CONFORME" in out and "absent" not in out \
            and "par_zone_classe (1 modèle(s))" in out, out
        rc, out = run(completer)  # déjà présent = refus
        assert rc != 0 and "déjà présent" in out, out
        rc, out = run(completer + ["--forcer"])
        assert rc == 0, out
        # fonction pure : refus / --forcer sans passer par le disque
        cache = {"m1": {"decal": 0, "enregs": ENREGS}}
        try:
            cm.completer(json.loads(json.dumps(resume)), cache)
            raise AssertionError("bloc déjà présent accepté sans --forcer")
        except SystemExit:
            pass
        assert cm.completer(json.loads(json.dumps(resume)), cache, forcer=True) == 1
        # bloc trafiqué -> NON CONFORME sur la bonne clé
        faux = json.loads(tmp.joinpath("metriques_eval.json").read_text(encoding="utf-8"))
        faux["modeles"]["m1"]["par_zone_classe"]["za"]["a"]["tp"] = 0
        tmp.joinpath("metriques_eval.json").write_text(json.dumps(faux), encoding="utf-8")
        rc, out = run(verif)
        assert rc == 1 and "NON CONFORME" in out and "par_zone_classe.za.a.tp" in out, out

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

        # --- fixtures FICHES (2026-09-03) : dépôt synthétique, runs tracés, plugin ---
        depot = tmp / "depot"
        (depot / "manifests" / "corpus").mkdir(parents=True)
        (depot / "taxonomy").mkdir()
        (depot / "manifests" / "corpus" / "corpus_x.yaml").write_text(
            "corpus: corpus_x\ngenere_le: '2026-08-31'\ngsd_m: 0.5\nrvt: LD\n"
            "classes: [a, b]\nfusions: {aa: a}\ndatasets:\n"
            "  ds_ramb:\n    zone: ile_de_france/78_rambouillet\n    splits:\n"
            "      train: {images: 10, annotations: {a: 120, b: 5}}\n"
            "      valid: {images: 2, annotations: {a: 1}}\n"
            "      test: {images: 1, annotations: {a: 1, b: 1}}\n"
            "  ds_haye:\n    zone: grand_est/54_foret_de_haye\n    splits:\n"
            "      train: {images: 10, annotations: {a: 50, b: 200}}\n"
            "      valid: {images: 1, annotations: {}}\n"
            "      test: {images: 1, annotations: {}}\n", encoding="utf-8")
        # manifeste INCOMPLET (ni genere_le, ni gsd_m, ni rvt) lié au run fours_v1 par config.json
        (depot / "manifests" / "corpus" / "corpus_y.yaml").write_text(
            "corpus: corpus_y\nclasses: [a]\nfusions: {}\ndatasets:\n"
            "  ds_blois:\n    zone: centre_val_de_loire/41_blois\n    splits:\n"
            "      train: {images: 5, annotations: {a: 300}}\n"
            "      valid: {images: 1, annotations: {a: 2}}\n"
            "      test: {images: 1, annotations: {a: 1}}\n", encoding="utf-8")
        # manifeste sans annotation train (split null, train vide) lié au run provisoire :
        # part du train = « — », jamais de division par zéro / None
        (depot / "manifests" / "corpus" / "corpus_z.yaml").write_text(
            "corpus: corpus_z\nclasses: [a]\nfusions: {}\ndatasets:\n"
            "  ds_z:\n    zone: occitanie/30_la_capelle_et_masmolene\n    splits:\n"
            "      train: null\n      valid: {images: 1}\n"
            "      test: {images: 1, annotations: {a: 1}}\n", encoding="utf-8")
        (tmp / "fours" / "runs" / "training" / "fours_prov" / "config.json").write_text(json.dumps({
            "dataset": {"dataset_dir": "/content/corpus/corpus_z"}}), encoding="utf-8")
        (depot / "taxonomy" / "entities.yaml").write_text(
            "entities:\n  - id: a\n    label_fr: Alpha\n  - id: b\n    label_fr: Bêta\n",
            encoding="utf-8")
        run2 = tmp / "enclos" / "runs" / "training" / "enclos_v2"
        run2.joinpath("params_run.yaml").write_text(
            "MODEL_VARIANT: large\nRESOLUTION: 648\nNUM_EPOCHS: 10\nEARLY_STOPPING_PATIENCE: 5\n"
            "BATCH_SIZE: 8\nGRAD_ACCUM_STEPS: 2\nLEARNING_RATE: 1.0e-05\nLR_ENCODER: 1.0e-06\n"
            "SEED: 42\nAUG_CONFIG_NAME: AUG_AERIAL\nPRECISION: bf16-mixed\n"
            "CORPUS_DRIVE_DIR: /content/drive/x/dataset/corpus_x\nCORPUS_MANIFEST_SHA1: abcdef0123456789\n"
            "FINETUNE_FROM: /content/drive/x/runs/training/enclos_v1/checkpoint_best_ema.pth\n"
            "BASE_WEIGHTS_DRIVE: /content/drive/x/_poids_base/rf-detr-seg-large.pth\n"
            "RVT: {type: LD, params: {gsd_m: 0.5, rmin_px: 10, rmax_px: 20}}\nMNT: {resolution: 0.5}\n",
            encoding="utf-8")
        # metrics.csv seul = dernière reprise (ép. 3-5) ; l'historique porte les ép. 0-3
        run2.joinpath("metrics_avant_reprise_1.csv").write_text(
            "epoch,step,val/ema_mAP_50\n0,10,0.1\n0,20,\n1,30,0.2\n2,40,0.3\n3,50,0.35\n",
            encoding="utf-8")
        run2.joinpath("metrics.csv").write_text(
            "epoch,step,val/ema_mAP_50\n3,50,0.4\n4,60,0.7\n5,70,0.6\n", encoding="utf-8")
        bloc_a = dict(ce.bloc_metriques(ENREGS, 0.05), F1=0.7, seuil_f1max=0.3, n_gt=2)
        bloc_b = dict(ce.bloc_metriques(ENREGS, 0.05), F1=0.5, seuil_f1max=0.35, n_gt=2)  # ≠ valid+test (1)
        glob2 = dict(ce.bloc_metriques(ENREGS, 0.05), F1=0.71, seuil_f1max=0.3)
        (run2 / "evaluation" / "metriques_eval.json").write_text(json.dumps({
            "schema": "metriques_eval/1", "genere_le": "2026-08-20T10:00:00", "tache": "segmentation",
            "fusion": {},
            "modeles": {"enclos_seg_v2": {
                "global": glob2, "par_classe": {"a": bloc_a, "b": bloc_b},
                "par_zone": {"ile_de_france/78_rambouillet": {"n_gt": 2, "P": 0.5, "R": 0.5},
                             "grand_est/54_foret_de_haye": {"n_gt": 1, "P": 1.0, "R": 0.0}},
                "par_zone_classe": {
                    "ile_de_france/78_rambouillet": {
                        "a": {"n_gt": 40, "tp": 10, "fp": 2, "R": 0.25, "P": 0.8,
                              "R_seuil_classe": 0.3, "fp_seuil_classe": 3, "R_max": 0.3},
                        "b": {"n_gt": 0, "tp": 0, "fp": 3, "R": None, "P": None,
                              "R_seuil_classe": None, "fp_seuil_classe": 3, "R_max": None}},
                    "grand_est/54_foret_de_haye": {
                        "a": {"n_gt": 5, "tp": 4, "fp": 1, "R": 0.8, "P": 0.8,
                              "R_seuil_classe": 0.8, "fp_seuil_classe": 1, "R_max": 0.9}}}}},
        }), encoding="utf-8")
        run1 = tmp / "enclos" / "runs" / "training" / "enclos_v1"
        run1.joinpath("config.json").write_text(json.dumps({
            "model": {"variant": "large", "resolution": 648, "num_classes": 2},
            "training": {"num_epochs": 20, "seed": 42},
            "dataset": {"dataset_dir": "/content/corpus/inconnu"}}), encoding="utf-8")
        # run entraîné (checkpoint) SANS mesure -> « À surveiller » de la famille enclos
        run3 = tmp / "enclos" / "runs" / "training" / "enclos_v3"
        run3.mkdir()
        run3.joinpath("checkpoint_best_ema.pth").write_bytes(b"")
        # fours_seg_v1 n'a pas de run à son nom : premier modèle de l'éval -> run porteur (fours_v1)
        (tmp / "fours" / "runs" / "training" / "fours_v1" / "config.json").write_text(json.dumps({
            "model": {"variant": "large"}, "training": {"num_epochs": 5},
            "dataset": {"dataset_dir": "/content/corpus/corpus_y"}}), encoding="utf-8")
        plugin = tmp / "plugin"
        (plugin / "enclos_seg_v2").mkdir(parents=True)
        (plugin / "enclos_seg_v2" / "model_card.yaml").write_text(
            "id: enclos_seg_v2\nstatus: beta\nversion: '2026-09'\nthresholds:\n"
            "  confidence_default: 0.3\n  confidence_per_class: {a: 0.3, b: 0.4}\n"
            "  seuils_provenance: 'metriques_eval.json 2026-08-20, b arrondi'\n", encoding="utf-8")

        page = tm.construire(tmp)  # sans --depot ni --plugin : comportement d'origine
        assert "enclos" in page and "fours" in page
        assert "enclos_seg_v1" in page and "enclos_seg_v2" in page
        assert "0,710" in page and "0,620" in page  # décimales à la virgule (2026-09-03)
        assert "enclos_seg_v1 → enclos_seg_v2" in page  # ordre des versions par date
        assert "<polyline" in page  # sparklines
        assert "verdun_v1" not in page, "liste des runs sans mesure supprimée (2026-09-02)"
        assert "sans mesure" in page.lower()  # le COMPTEUR reste dans l'en-tête
        # planches : lien du tableau seulement, plus de bloc intégré (2026-09-03)
        assert "enclos_v1/evaluation/courbes_seuils_pr.png'>courbes</a>" in page
        assert "courbes — " not in page and "<img" not in page
        assert "fours_prov_x" in page and "[provisoire]" in page  # étiqueté au tableau
        assert "fours_prov_x → " not in page and "→ fours_prov_x" not in page, \
            "un modèle provisoire ne doit pas entrer dans l'évolution"
        assert "corrompu" in page and "⚠" in page  # fichier cassé = warning, pas crash
        assert "<script" not in page  # zéro JS
        assert "installé" not in page and "<th>plugin</th>" not in page  # pas de --plugin
        assert page.count("fiche — ") == 4  # une fiche par modèle (v1, v2, fours, prov)
        assert "données non tracées" in page  # sans --depot : aucun manifeste
        assert "6 faites (6 validées)</b> sur 10 prévues · meilleure époque 4 (mAP50 EMA 0,700) " \
               "· reprises : 3" in page, "époques fusionnées metrics.csv + historique"

        page = tm.construire(tmp, depot=depot, plugin=plugin)
        assert "<script" not in page
        assert "a (Alpha)" in page  # label_fr de la taxonomie
        assert "2 <span class='ok'>✓</span>" in page  # a : n_gt 2 = valid 1 + test 1
        assert "<span class='ko'>≠ 1</span>" in page  # b : n_gt 2 ≠ valid 0 + test 1
        assert "fusions corpus : aa -&gt; a" in page  # ligne d'en-tête de famille
        assert "<span class='tag'>fragile</span>" in page  # zone n_gt 2 < 30
        assert "<span class='tag'>décroche</span>" in page  # rambouillet·a R_max 0,3, n_gt 40
        assert "<span class='tag'>rare en train</span>" in page  # haye·a : 50 < 100
        assert "<span class='tag'>absente</span>" in page  # rambouillet·b : n_gt 0
        assert "zones fragiles (n_gt &lt; 30) : grand_est/54_foret_de_haye (1), " \
               "ile_de_france/78_rambouillet (2)" in page
        assert "classes rares ou absentes en train" not in page  # étiquette de ligne seulement (2026-09-03)
        assert "zone × classe qui décroche" not in page
        assert "runs entraînés sans mesure (checkpoint sans evaluation/metriques_eval.json) : enclos_v3" in page
        assert "manifeste de corpus corpus_y incomplet (genere_le, gsd_m, rvt manquant)" in page
        assert "relancer tools/completer_metriques_eval.py" in page  # enclos_seg_v1 sans par_zone_classe
        assert "installé · beta · 2026-09" in page and "non installé" in page and "<th>plugin</th>" in page
        assert "0,300 = F1-max" in page
        assert "0,400 ≠ F1-max 0,350 (justifié : metriques_eval.json 2026-08-20, b arrondi)" in page
        assert "enclos_seg_v2 — seuil déployé ≠ F1-max mesuré (b : justifié), cf. fiche" in page
        assert "lignée : rf-detr-seg-large.pth -&gt; enclos_v1 -&gt; enclos_seg_v2" in page
        assert "<dt>corpus</dt><dd>corpus_x (manifeste abcdef01)</dd>" in page
        assert "<dt>transfert depuis</dt><dd>enclos_v1/checkpoint_best_ema.pth</dd>" in page
        assert "<dt>source</dt><dd>config.json</dd>" in page  # enclos_v1 : repli config.json
        assert "corpus : inconnu · <b>données non tracées</b>" in page  # manifeste absent
        assert "corpus : corpus_y" in page and "<dt>dataset</dt><dd>corpus_y</dd>" in page  # fours : run porteur
        assert "corpus : corpus_z" in page and "<td>0</td><td>—</td><td>0</td><td>1</td>" in page  # train vide
        assert "<th>écart F1</th>" in page and "<td>+0,090</td>" in page and "<td>-0,110</td>" in page
        assert "rgba(42,120,214,0.44)" in page  # F1 0,71 -> alpha 0,62×0,71 (cellule teintée)
        assert "Couverture des zones" in page and "●" in page
        assert page.count("<h2>") == 2  # couverture + glossaire
        assert page.count("<li><b>") == 9  # glossaire : 9 entrées
        assert "<td>ile_de_france/78_rambouillet</td><td>●</td><td></td>" in page  # enclos oui, fours non

    print("OK — courbes_eval (prf/ap50/par_zone_classe/resumer/empreinte/reprendre-de) "
          "+ completer_metriques_eval/verif + tableau_modeles")


if __name__ == "__main__":
    main()
