"""Auto-test de tools/build_corpus.py + verif_corpus.py sur datasets fabriqués."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_corpus import construire

tmp = Path(tempfile.mkdtemp(prefix="corpus_test_"))
datasets = tmp / "datasets"


def fabrique_dataset(nom, zone, par_split):
    """par_split : {split: [(nom_image, [classes des annotations]), ...]}"""
    racine = datasets / nom
    for split, images in par_split.items():
        (racine / split).mkdir(parents=True, exist_ok=True)
        noms_classes = sorted({c for _, cl in images for c in cl})
        cats = [{"id": i, "name": c, "supercategory": "none"}
                for i, c in enumerate(noms_classes)]
        cid = {c: i for i, c in enumerate(noms_classes)}
        coco = {"images": [], "annotations": [], "categories": cats}
        for k, (nom_img, cl) in enumerate(images):
            (racine / split / nom_img).write_bytes(b"PNG" + nom_img.encode())
            coco["images"].append({"id": k, "file_name": nom_img,
                                   "width": 648, "height": 648})
            for c in cl:
                coco["annotations"].append(
                    {"id": len(coco["annotations"]), "image_id": k,
                     "category_id": cid[c], "bbox": [0, 0, 5, 5], "area": 25,
                     "segmentation": [[0, 0, 5, 0, 5, 5]], "iscrowd": 0})
        (racine / split / "_annotations.coco.json").write_text(
            json.dumps(coco), encoding="utf-8")
    (racine / "split_manifest.yaml").write_text(
        yaml.safe_dump({"zone": zone, "dataset": nom}), encoding="utf-8")


fabrique_dataset("ds_a", "zone_a", {
    "train": [("a_1.png", ["parcellaire", "voie"]), ("a_2.png", ["talus"])],
    "valid": [("a_3.png", ["voie"])],
    "test": [("a_4.png", [])]})
fabrique_dataset("ds_b", "zone_b", {
    "train": [("b_1.png", ["fosse", "chemin_creux"])],
    "valid": [("b_2.png", ["talus_fosse"])],
    "test": [("b_3.png", ["parcellaire"])]})

cfg_p = tmp / "corpus.yaml"
cfg_p.write_text(yaml.safe_dump({
    "corpus": "jouet", "classes": ["parcellaire", "talus", "fosse",
                                   "talus_fosse", "chemin_creux"],
    "fusions": {"voie": "parcellaire"},
    "datasets": ["ds_a", "ds_b"]}), encoding="utf-8")
corpus = construire(cfg_p, datasets, tmp / "corpus")

train = json.loads((corpus / "train" / "_annotations.coco.json")
                   .read_text(encoding="utf-8"))
assert len(train["images"]) == 3
assert [c["name"] for c in train["categories"]] == [
    "parcellaire", "talus", "fosse", "talus_fosse", "chemin_creux"]
noms = {c["id"]: c["name"] for c in train["categories"]}
par_classe = {}
for a in train["annotations"]:
    par_classe[noms[a["category_id"]]] = par_classe.get(noms[a["category_id"]], 0) + 1
assert par_classe == {"parcellaire": 2, "talus": 1, "fosse": 1,
                      "chemin_creux": 1}, par_classe  # voie fusionnée
zones = {im["zone"] for im in train["images"]}
assert zones == {"zone_a", "zone_b"}
valid = json.loads((corpus / "valid" / "_annotations.coco.json")
                   .read_text(encoding="utf-8"))
assert {noms[a["category_id"]] for a in valid["annotations"]} == {
    "parcellaire", "talus_fosse"}  # voie seule -> parcellaire
assert (corpus / "test" / "a_4.png").read_bytes() == b"PNGa_4.png"

verif = Path(__file__).resolve().parents[1] / "tools" / "verif_corpus.py"
r = subprocess.run([sys.executable, str(verif), str(cfg_p), str(datasets),
                    str(corpus)], capture_output=True, text=True)
assert r.returncode == 0 and "CONFORME" in r.stdout, r.stdout + r.stderr

# cas cassé : une annotation supprimée du corpus -> détecté
coco_t = json.loads((corpus / "train" / "_annotations.coco.json")
                    .read_text(encoding="utf-8"))
coco_t["annotations"] = coco_t["annotations"][:-1]
(corpus / "train" / "_annotations.coco.json").write_text(json.dumps(coco_t),
                                                         encoding="utf-8")
r2 = subprocess.run([sys.executable, str(verif), str(cfg_p), str(datasets),
                     str(corpus)], capture_output=True, text=True)
assert r2.returncode != 0, "altération non détectée"
print("build_corpus + contrôleur : OK")
