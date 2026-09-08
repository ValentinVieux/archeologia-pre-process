"""Auto-test de tools/deposer_revue_zone.py + verif_deposer_revue_zone.py (SANS GPU, GPKG synthétiques) :
.venv\\Scripts\\python.exe tests\\test_deposer_revue_zone.py

Cas reproduits (panels de vérification du 2026-09-08, 2 rondes) : fid de revue à trous (suppression
QGIS) -> fid_source de la couche SAM remappé sur les fid du livrable, polygone SAM sans boîte écarté ;
geom_origine MIXTE (WKT + NULL) ; référence : inchangé / modifié / supprimé (motif supprime_qgis) +
couche _doublons rapatriée (motif doublon) ; boîte à 0,006 m de la référence = inchangé ; doublon exact
dans la revue ; chemin uid (uid NULL, uid dupliqué refusé, paire à uid déplacée de 60 m = modifié) ;
référence SANS uid et revue AVEC uid ; entier à NULL préservé (couche revue et --copier) ; SAM avec
--marge ; couche RENOMMÉE pendant la revue (c=ref, refus sans mapping) ; GÉNÉRATION SUIVANTE (v3 rouvert
comme revue contre --reference v3, uid ref:<N> hérités) ; livrable trafiqué refusé ; refus d'écraser.
"""
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import pyogrio
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "tools" / "deposer_revue_zone.py"
VERIF = ROOT / "tools" / "verif_deposer_revue_zone.py"
CRS = "EPSG:2154"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def b(x, y, w=20.0, h=20.0):
    return box(x, y, x + w, y + h)


def gdf(geoms, **cols):
    return gpd.GeoDataFrame(cols, geometry=list(geoms), crs=CRS)


def run(script, *args):
    r = subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    return r.returncode, r.stdout + r.stderr


def sql(gpkg, *ordres):
    con = sqlite3.connect(gpkg)
    for o in ordres:
        con.execute(o)
    con.commit(); con.close()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="test_depot_"))
    try:
        revue, ref, autre = tmp / "revue.gpkg", tmp / "ref.gpkg", tmp / "autre.gpkg"
        # référence dep : A, B, C (sera supprimée), D (à 0,006 m d'une boîte de revue)
        gdf([b(600000, 6700000), b(600100, 6700000), b(600200, 6700000), b(600300, 6700000)]).to_file(ref, layer="dep", driver="GPKG")
        # revue dep : fids 1..7 puis suppression du fid 3 (trou) ; 1 = A, 2 = B déplacé de 3 m, 4-5 nouvelles, 6 = D à 0,006 m,
        # 7 = doublon exact de A ; annotation_id entier avec NULL sur les ajouts (schéma des zones à uid)
        gdf([b(600000, 6700000), b(600103, 6700000), b(600500, 6700500), b(600900, 6700000), b(600950, 6700000),
             b(600300.006, 6700000.004), b(600000, 6700000)], annotation_id=[10, 20, None, None, None, 60, 70]
            ).astype({"annotation_id": "Int64"}).to_file(revue, layer="dep", driver="GPKG")
        sql(revue, "DELETE FROM dep WHERE fid = 3")
        boites = pyogrio.read_dataframe(revue, layer="dep", fid_as_index=True)
        # SAM : un polygone par boîte (fid_source = fid de REVUE), celui du fid 5 déborde de 3 m, + un orphelin (fid_source 3)
        sam = gdf([g.buffer(3) if f == 5 else g.buffer(-2) for f, g in zip(boites.index, boites.geometry)] + [b(600200, 6700000).buffer(-2)],
                  fid_source=list(boites.index) + [3], methode=["sam"] * (len(boites) + 1))
        sam.to_file(revue, layer="dep_sam", driver="GPKG")
        gdf([b(600700, 6700700)], doublon_de=[1]).to_file(revue, layer="dep_doublons", driver="GPKG")
        # chemin uid : four (uid partagé, uid NULL des deux côtés à géométries différentes, u3 dupliqué)
        gdf([b(610000, 6710000), b(610100, 6710000), b(610200, 6710000)], uid=["u1", None, "u3"]).to_file(ref, layer="four", driver="GPKG")
        gdf([b(610000, 6710000), b(610700, 6710000), b(610200, 6710000), b(610200, 6710000)], uid=["u1", None, "u3", "u3"]).to_file(revue, layer="four", driver="GPKG")
        # référence SANS uid, revue AVEC uid (auto-labels) ; couche renommée pendant la revue (mard -> ref circ)
        gdf([b(630000, 6730000), b(630100, 6730000)]).to_file(ref, layer="pts", driver="GPKG")
        gdf([b(630000, 6730000), b(630103, 6730000)], uid=["train:1", "train:2"]).to_file(revue, layer="pts", driver="GPKG")
        gdf([b(640000, 6740000), b(640100, 6740000), b(640200, 6740000)]).to_file(ref, layer="circ", driver="GPKG")
        gdf([b(640000, 6740000), b(640103, 6740000)]).to_file(revue, layer="mard", driver="GPKG")
        # couche à recopier : Integer avec NULL
        gpd.GeoDataFrame({"Id": [1, None, 3]}, geometry=[b(620000, 6720000), b(620100, 6720000), b(620200, 6720000)], crs=CRS).astype({"Id": "Int64"}).to_file(autre, layer="lignes", driver="GPKG")

        def args(rev, out, mard="mard=circ", reference=ref):
            return [rev, out, "--couches", "dep", "four", "pts", mard, "--reference", reference, "--sam", "dep_sam", "--copier", f"{autre}:lignes"]

        # couche renommée sans mapping : refus
        code, log = run(PROD, *args(revue, tmp / "x.gpkg", mard="mard"))
        assert code != 0 and "absente de la référence" in log, log

        out = tmp / "v3.gpkg"
        code, log = run(PROD, *args(revue, out))
        assert code == 0, log
        v3 = pyogrio.read_dataframe(out, layer="dep", fid_as_index=True)
        assert list(v3.index) == [1, 2, 3, 4, 5, 6], list(v3.index)
        assert list(v3["decision_humaine"]) == ["inchangé", "modifié", "ajouté", "ajouté", "inchangé", "ajouté"], list(v3["decision_humaine"])
        assert list(v3["uid"]) == ["ref:1", "ref:2", "revue:4", "revue:5", "ref:4", "revue:7"], list(v3["uid"])
        assert str(pyogrio.read_info(out, layer="dep")["dtypes"][0]) == "int64", pyogrio.read_info(out, layer="dep")["dtypes"]
        sup = pyogrio.read_dataframe(out, layer="dep_supprimees")
        assert sorted(sup["motif"]) == ["doublon", "supprime_qgis"], list(sup["motif"])
        s3 = pyogrio.read_dataframe(out, layer="dep_sam")
        assert len(s3) == 6 and list(s3["fid_source"]) == [1, 2, 3, 4, 5, 6], list(s3["fid_source"])
        assert "ÉCARTÉS" in log and "[3]" in log, log
        f3 = pyogrio.read_dataframe(out, layer="four")
        assert list(f3["decision_humaine"]) == ["inchangé", "ajouté", "inchangé", "ajouté"], list(f3["decision_humaine"])
        assert list(f3["uid"]) == ["u1", "revue:2", "u3", "u3"], list(f3["uid"])
        p3 = pyogrio.read_dataframe(out, layer="pts")
        assert list(p3["decision_humaine"]) == ["inchangé", "modifié"] and list(p3["uid"]) == ["train:1", "train:2"], (list(p3["decision_humaine"]), list(p3["uid"]))
        m3 = pyogrio.read_dataframe(out, layer="mard")
        assert list(m3["decision_humaine"]) == ["inchangé", "modifié"] and len(pyogrio.read_dataframe(out, layer="mard_supprimees")) == 1
        assert str(pyogrio.read_info(out, layer="lignes")["dtypes"][0]) == "int64"
        assert "NON reprises" not in log, log

        code, log = run(VERIF, *args(revue, out), "--marge", "3")
        assert code == 1 and "four : uid non uniques" in log, log  # le uid dupliqué de la revue est refusé
        # même jeu sans le doublon de uid, u3 déplacé de 60 m (paire à uid = modifié quel que soit l'IoU) : CONFORME
        revue2 = tmp / "revue2.gpkg"
        shutil.copy(revue, revue2)
        f2 = pyogrio.read_dataframe(revue2, layer="four")
        f2 = f2.iloc[:3].copy(); f2.loc[2, "geometry"] = b(610260, 6710000)
        sql(revue2, *[f'DROP TRIGGER "{t}"' for (t,) in sqlite3.connect(revue2).execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='four'")],
            "DROP TABLE four", "DELETE FROM gpkg_contents WHERE table_name='four'", "DELETE FROM gpkg_geometry_columns WHERE table_name='four'",
            "DROP TABLE IF EXISTS rtree_four_geom")
        f2.to_file(revue2, layer="four", driver="GPKG")
        out2 = tmp / "v3b.gpkg"
        code, log = run(PROD, *args(revue2, out2))
        assert code == 0 and "four: 3 (inchangé 1, modifié 1, ajouté 1)" in log, log
        code, log = run(VERIF, *args(revue2, out2), "--marge", "3")
        assert code == 0 and "CONFORME" in log, log
        code, log = run(VERIF, *args(revue2, out2))  # sans --marge : le polygone qui déborde de 3 m est refusé
        assert code == 1 and "hors de leur boîte" in log, log

        # GÉNÉRATION SUIVANTE : v3b rouvert comme revue (rien touché) contre --reference v3b -> tout inchangé, CONFORME
        out3 = tmp / "v4.gpkg"
        a3 = [out2, out3, "--couches", "dep", "four", "pts", "mard", "--reference", out2, "--sam", "dep_sam"]
        code, log = run(PROD, *a3)
        assert code == 0 and "dep: 6 (inchangé 6, modifié 0, ajouté 0)" in log and "NON reprises" in log, log
        code, log = run(VERIF, *a3, "--marge", "3")
        assert code == 0 and "CONFORME" in log, log
        assert list(pyogrio.read_dataframe(out3, layer="dep")["uid"]) == ["ref:1", "ref:2", "revue:4", "revue:5", "ref:4", "revue:7"]

        # livrable trafiqué : un fid_source décalé -> NON CONFORME (triggers rtree retirés : la géométrie ne bouge pas)
        sql(out2, *[f'DROP TRIGGER "{t}"' for (t,) in sqlite3.connect(out2).execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='dep_sam'")],
            "UPDATE dep_sam SET fid_source = 6 WHERE fid = 1")
        code, log = run(VERIF, *args(revue2, out2), "--marge", "3")
        assert code == 1 and "mal remappés" in log, log
        assert run(PROD, *args(revue2, out2))[0] != 0  # refus d'écraser
        print("OK test_deposer_revue_zone")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
