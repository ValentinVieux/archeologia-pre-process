"""Upload d'un dataset découpé (slice_zone) vers Roboflow avec split IMPOSÉ.

Le split de chaque image vient de split_manifest.yaml — Roboflow ne tire plus jamais
au sort (cf. docs/fuite_spatiale_train_test.html). La clé API vient de la variable
d'environnement ROBOFLOW_API_KEY (jamais en clair dans ce repo public).

Modes :
  --dry-run   montre le plan complet sans appeler l'API ;
  --test      petit lot représentatif (défaut 5/3/2 par split, incluant si possible
              chaque classe et un négatif) pour valider la chaîne sur la plateforme
              avant l'upload complet — demande utilisateur : ne pas brûler de crédits
              tant que d'éventuelles rectifications ne sont pas actées.

L'état d'avancement vit dans upload_manifest[_test].yaml dans le dossier du dataset :
relancer reprend où l'upload s'était arrêté (aucun re-envoi des images déjà tracées).

Usage :
    .venv\\Scripts\\python.exe tools\\upload_roboflow_split.py <dossier_dataset>
        --workspace <id> --projet <id> [--creer-projet] [--suffixe-classes <site>]
        [--test] [--dry-run] [--batch <nom>]
"""
import argparse
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

from slice_zone import chemin_sur_drive

SPLITS = ("train", "valid", "test")
TEST_PAR_SPLIT = {"train": 5, "valid": 3, "test": 2}


def charger_dataset(dossier):
    """split_manifest.yaml + COCO par split -> plan d'images avec annotations par image."""
    dossier = Path(dossier)
    manifeste = yaml.safe_load((dossier / "split_manifest.yaml").read_text(encoding="utf-8"))
    zone = str(manifeste["zone"]).replace("\\", "/").rstrip("/")
    region, zone_id = zone.split("/")[-2], zone.split("/")[-1]

    images = []
    for split in SPLITS:
        coco = json.loads((dossier / split / "_annotations.coco.json")
                          .read_text(encoding="utf-8"))
        categories = coco["categories"]
        annos_par_image = {}
        for a in coco["annotations"]:
            annos_par_image.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            images.append({
                "filename": im["file_name"], "split": split,
                "chemin": dossier / split / im["file_name"],
                "image_coco": im, "annotations": annos_par_image.get(im["id"], []),
                "categories": categories,
            })
    images.sort(key=lambda i: (SPLITS.index(i["split"]), i["filename"]))
    return manifeste, images, zone_id, region


def echantillon_test(images, quotas=TEST_PAR_SPLIT):
    """Lot de test représentatif et déterministe : par split, priorité à la diversité —
    une tuile de chaque classe non dominante, un négatif, complété par les premières
    tuiles annotées."""
    lot = []
    for split in SPLITS:
        du_split = [i for i in images if i["split"] == split]
        retenues, vus = [], set()

        def prendre(im):
            if im["filename"] not in vus and len(retenues) < quotas[split]:
                retenues.append(im)
                vus.add(im["filename"])

        cats = {c["id"]: c["name"] for c in (du_split[0]["categories"] if du_split else [])}
        classes = sorted({cats[a["category_id"]] for i in du_split for a in i["annotations"]})
        for classe in reversed(classes):  # classes rares d'abord (ordre alpha inversé ~ heuristique neutre)
            for im in du_split:
                if any(cats[a["category_id"]] == classe for a in im["annotations"]):
                    prendre(im)
                    break
        for im in du_split:  # un négatif si disponible
            if not im["annotations"]:
                prendre(im)
                break
        # complément par richesse décroissante : le lot de test doit montrer des tuiles
        # représentatives sur la plateforme, pas les bords clairsemés
        for im in sorted(du_split, key=lambda i: -len(i["annotations"])):
            prendre(im)
        lot.extend(retenues)
    return lot


def coco_mono_image(im, suffixe):
    """COCO ne contenant que cette image et ses annotations (matching sans ambiguïté)."""
    categories = [dict(c) for c in im["categories"]]
    if suffixe:
        for c in categories:
            c["name"] = f"{c['name']}_{suffixe}"
    return {"images": [im["image_coco"]], "annotations": im["annotations"],
            "categories": categories}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", help="dossier local du dataset (sortie de slice_zone)")
    p.add_argument("--workspace", required=True)
    p.add_argument("--projet", required=True)
    p.add_argument("--creer-projet", action="store_true",
                   help="crée le projet (instance-segmentation) s'il n'existe pas")
    p.add_argument("--suffixe-classes", default=None,
                   help="suffixe de site pour les classes Roboflow (ex. haye -> parcellaire_haye)")
    p.add_argument("--test", action="store_true", help="petit lot représentatif 5/3/2")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--batch", default=None)
    args = p.parse_args()

    if chemin_sur_drive(args.dataset):
        sys.exit("dataset sur G: — uploader depuis la copie locale (règle Drive)")
    dossier = Path(args.dataset)
    manifeste, images, zone_id, region = charger_dataset(dossier)
    dataset_nom = manifeste["dataset"]

    if args.test:
        images = echantillon_test(images)
    batch = args.batch or (f"test_{dataset_nom}" if args.test else dataset_nom)
    suivi_path = dossier / ("upload_manifest_test.yaml" if args.test
                            else "upload_manifest.yaml")
    suivi = (yaml.safe_load(suivi_path.read_text(encoding="utf-8"))
             if suivi_path.exists() else None) or {
        "dataset": dataset_nom, "zone": zone_id, "region": region,
        "workspace": args.workspace, "projet": args.projet,
        "suffixe_classes": args.suffixe_classes, "batch": batch,
        "tags": [zone_id, region], "images": []}
    deja = {i["filename"] for i in suivi["images"]}
    a_envoyer = [i for i in images if i["filename"] not in deja]

    par_split = {s: sum(1 for i in a_envoyer if i["split"] == s) for s in SPLITS}
    n_annos = sum(len(i["annotations"]) for i in a_envoyer)
    print(f"{dataset_nom} -> {args.workspace}/{args.projet} (batch {batch})")
    print(f"  à envoyer : {sum(par_split.values())} images {par_split}, "
          f"{n_annos} annotations, {len(deja)} déjà tracées (reprises sautées)")
    print(f"  tags : [{zone_id}, {region}] ; classes suffixées : "
          f"{args.suffixe_classes or 'non (canoniques)'}")
    if args.dry_run:
        for i in a_envoyer[:20]:
            print(f"    {i['split']:5s} {i['filename']} ({len(i['annotations'])} annos)")
        if len(a_envoyer) > 20:
            print(f"    ... et {len(a_envoyer) - 20} autres")
        print("dry-run : aucun appel API.")
        return

    cle = os.environ.get("ROBOFLOW_API_KEY")
    if not cle:
        sys.exit("ROBOFLOW_API_KEY absente de l'environnement.")
    import roboflow  # import tardif : lourd, inutile en dry-run
    rf = roboflow.Roboflow(api_key=cle)
    ws = rf.workspace(args.workspace)
    try:
        projet = ws.project(args.projet)
    except Exception:
        if not args.creer_projet:
            raise
        print(f"projet {args.projet} introuvable -> création (instance-segmentation)")
        projet = ws.create_project(args.projet, "instance-segmentation",
                                   "private", "structure")

    envoyees = 0
    try:
        for i in a_envoyer:
            annotation_path = None
            if i["annotations"]:
                tmp = tempfile.NamedTemporaryFile("w", suffix=".coco.json",
                                                  delete=False, encoding="utf-8")
                json.dump(coco_mono_image(i, args.suffixe_classes), tmp,
                          ensure_ascii=False)
                tmp.close()
                annotation_path = tmp.name
            projet.single_upload(
                image_path=str(i["chemin"]),
                annotation_path=annotation_path,
                split=i["split"],                 # le split est IMPOSÉ, jamais tiré
                batch_name=batch,
                tag_names=[zone_id, region],
                num_retry_uploads=3,
            )
            if annotation_path:
                os.unlink(annotation_path)
            suivi["images"].append({"filename": i["filename"], "split": i["split"],
                                    "annotations": len(i["annotations"])})
            envoyees += 1
            if envoyees % 20 == 0 or envoyees == len(a_envoyer):
                print(f"  {envoyees}/{len(a_envoyer)}")
    finally:
        suivi["derniere_mise_a_jour"] = datetime.datetime.now().isoformat(timespec="seconds")
        suivi_path.write_text(yaml.safe_dump(suivi, allow_unicode=True, sort_keys=False),
                              encoding="utf-8")
    print(f"terminé : {envoyees} envoyées. Suivi : {suivi_path}")
    print(f"Sorties : {suivi_path}")


if __name__ == "__main__":
    main()
