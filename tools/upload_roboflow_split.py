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
import time
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


def annotation_fichier(im, suffixe):
    """Fichier d'annotation à joindre : COCO pour une tuile annotée, VOC XML sans
    objet pour un négatif — le parseur refuse un COCO à zéro annotation, mais le VOC
    vide est accepté et vaut annotation NULL (image de fond assumée, vérifié le
    2026-07-27). Retourne (contenu, suffixe_de_fichier)."""
    if im["annotations"]:
        return (json.dumps(coco_mono_image(im, suffixe), ensure_ascii=False),
                ".coco.json")
    voc = (f"<annotation><filename>{im['filename']}</filename>"
           f"<size><width>{im['image_coco']['width']}</width>"
           f"<height>{im['image_coco']['height']}</height><depth>3</depth></size>"
           "<segmented>0</segmented></annotation>")
    return voc, ".xml"


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
    p.add_argument("--batch", default=None,
                   help="ATTENTION : un batch détourne les images annotées vers la file "
                        "Annotate au lieu du dataset — ne l'utiliser que pour une "
                        "relecture humaine voulue (constat plateforme du 2026-07-27)")
    p.add_argument("--sans-verification", action="store_true",
                   help="saute la vérification API post-upload (déconseillé)")
    p.add_argument("--eviter", default=None,
                   help="fichier texte (un nom d'image par ligne) à exclure — la "
                        "plateforme déduplique par CONTENU corbeille comprise : "
                        "re-uploader une image déjà envoyée puis supprimée ressuscite "
                        "son fantôme (constat du 2026-07-27)")
    args = p.parse_args()

    if chemin_sur_drive(args.dataset):
        sys.exit("dataset sur G: — uploader depuis la copie locale (règle Drive)")
    dossier = Path(args.dataset)
    manifeste, images, zone_id, region = charger_dataset(dossier)
    dataset_nom = manifeste["dataset"]

    if args.eviter:
        exclus = {l.strip() for l in Path(args.eviter).read_text(encoding="utf-8")
                  .splitlines() if l.strip()}
        images = [i for i in images if i["filename"] not in exclus]
        print(f"  {len(exclus)} nom(s) exclus via --eviter")
    if args.test:
        images = echantillon_test(images)
    batch = args.batch  # None par défaut : les images annotées vont DIRECTEMENT au dataset
    suivi_path = dossier / ("upload_manifest_test.yaml" if args.test
                            else "upload_manifest.yaml")
    suivi = (yaml.safe_load(suivi_path.read_text(encoding="utf-8"))
             if suivi_path.exists() else None) or {
        "dataset": dataset_nom, "zone": zone_id, "region": region,
        "workspace": args.workspace, "projet": args.projet,
        "suffixe_classes": args.suffixe_classes, "batch": batch,
        "tags": [zone_id, region], "images": []}
    if not args.test:
        # le run complet hérite du lot de test : ces images sont déjà sur la
        # plateforme, vérifiées conformes — ne pas les re-envoyer
        test_path = dossier / "upload_manifest_test.yaml"
        if test_path.exists():
            deja_test = yaml.safe_load(test_path.read_text(encoding="utf-8"))
            noms_suivi = {s["filename"] for s in suivi["images"]}
            herites = [s for s in deja_test.get("images", [])
                       if s["filename"] not in noms_suivi]
            suivi["images"].extend(herites)
            if herites:
                print(f"  {len(herites)} image(s) héritées du lot de test")
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

    envoyees, echecs = 0, []
    try:
        for i in a_envoyer:
            # les négatifs reçoivent AUSSI une annotation (VOC vide = null) : sans
            # annotation jointe, la plateforme les met en file Annotate « à
            # étiqueter » au lieu du dataset
            contenu, suffixe_fichier = annotation_fichier(i, args.suffixe_classes)
            tmp = tempfile.NamedTemporaryFile("w", suffix=suffixe_fichier,
                                              delete=False, encoding="utf-8")
            tmp.write(contenu)
            tmp.close()
            annotation_path = tmp.name
            # la réponse est CONTRÔLÉE : un échec de sauvegarde d'annotation laissait
            # l'image nue dans le dataset sans aucun signal (constat du 2026-07-27 :
            # 4 images sur 10 avaient perdu leurs polygones en silence)
            erreur = None
            for essai in range(3):
                try:
                    reponse = projet.single_upload(
                        image_path=str(i["chemin"]),
                        annotation_path=annotation_path,
                        split=i["split"],             # le split est IMPOSÉ, jamais tiré
                        batch_name=batch,
                        tag_names=[zone_id, region],
                        num_retry_uploads=3,
                    )
                    img = (reponse or {}).get("image", {}) or {}
                    annot = (reponse or {}).get("annotation") or {}
                    if bool(img.get("id") or img.get("success")) and \
                            bool(annot.get("success") or annot.get("id")):
                        erreur = None
                        break
                    if img.get("id"):
                        # image présente (ou fantôme dédupliqué ressuscité avec son
                        # ancien id) mais annotation non enregistrée : écrasement
                        # direct via l'API — parade vérifiée le 2026-07-27
                        contenu, suffixe_fichier = annotation_fichier(
                            i, args.suffixe_classes)
                        nom_fichier = i["filename"].rsplit(".", 1)[0] + suffixe_fichier
                        import requests
                        ra = requests.post(
                            f"https://api.roboflow.com/dataset/{args.projet}"
                            f"/annotate/{img['id']}?api_key={cle}"
                            f"&name={nom_fichier}&overwrite=true",
                            data=contenu.encode("utf-8"),
                            headers={"Content-Type": "text/plain"}, timeout=60)
                        if ra.ok and ra.json().get("success"):
                            erreur = None
                            break
                        erreur = f"annotate overwrite refusé : {ra.status_code} {ra.text[:120]}"
                    else:
                        erreur = f"réponse incomplète : {reponse}"
                except Exception as exc:  # noqa: BLE001 — on retente puis on rapporte
                    erreur = str(exc)
            if annotation_path:
                os.unlink(annotation_path)
            if erreur:
                echecs.append((i["filename"], erreur))
                print(f"  ÉCHEC {i['filename']} : {erreur[:120]}")
                continue
            suivi["images"].append({"filename": i["filename"], "split": i["split"],
                                    "annotations": len(i["annotations"])})
            envoyees += 1
            if envoyees % 20 == 0 or envoyees == len(a_envoyer):
                print(f"  {envoyees}/{len(a_envoyer)}")
    finally:
        suivi["derniere_mise_a_jour"] = datetime.datetime.now().isoformat(timespec="seconds")
        suivi_path.write_text(yaml.safe_dump(suivi, allow_unicode=True, sort_keys=False),
                              encoding="utf-8")
    print(f"terminé : {envoyees} envoyées, {len(echecs)} échec(s). Suivi : {suivi_path}")

    if not args.sans_verification:
        # toutes les images du suivi (pas seulement la passe courante : une relance
        # doit pouvoir re-vérifier un état déjà envoyé), avec reprises espacées —
        # l'ingestion des annotations côté plateforme est asynchrone
        attendu = {s["filename"]: (s["split"], s["annotations"])
                   for s in suivi["images"]}
        divergences = []
        for attente in (0, 30, 60, 120):
            if attente:
                print(f"  divergences restantes : {len(divergences)} — "
                      f"nouvel essai dans {attente} s (ingestion asynchrone)")
                time.sleep(attente)
            divergences = verifier_plateforme(cle, args.workspace, args.projet, attendu)
            if not divergences:
                break
        if divergences:
            for d in divergences:
                print(f"  DIVERGENCE : {d}")
            sys.exit(f"vérification post-upload : {len(divergences)} divergence(s) — "
                     "ne PAS lancer d'entraînement sur cet état.")
        print(f"vérification post-upload : {len(attendu)} images conformes sur la "
              "plateforme (dataset, split, annotations présentes).")
    if echecs:
        sys.exit(f"{len(echecs)} image(s) en échec — relancer la même commande "
                 "(reprise idempotente) puis investiguer si ça persiste.")
    print(f"Sorties : {suivi_path}")


def verifier_plateforme(cle, workspace, projet, attendu):
    """Boucle de vérification post-upload : l'état API doit refléter ce qu'on a envoyé.

    Vérifie pour chaque image envoyée : présence dans le DATASET (pas la file
    Annotate), split imposé respecté, annotations non vides quand on en a envoyé.
    """
    import requests

    base = f"https://api.roboflow.com/{workspace}/{projet}"
    sur_plateforme, offset = {}, 0
    while True:
        # le témoin fiable est le champ `annotations` ({count, classes}) — `labels`
        # est toujours vide, il avait produit de faux diagnostics le 2026-07-27
        r = requests.post(f"{base}/search?api_key={cle}", json={
            "limit": 200, "offset": offset, "in_dataset": True,
            "fields": ["name", "split", "annotations"]}, timeout=60)
        r.raise_for_status()
        resultats = r.json().get("results", [])
        for res in resultats:
            annos = res.get("annotations") or {}
            n = annos.get("count", 0) if isinstance(annos, dict) else 0
            sur_plateforme[res["name"]] = (res.get("split"), n)
        if len(resultats) < 200:
            break
        offset += 200
    divergences = []
    for nom, (split, n_annos) in sorted(attendu.items()):
        if nom not in sur_plateforme:
            divergences.append(f"{nom} : absente du dataset (file Annotate ?)")
            continue
        split_plat, n_plat = sur_plateforme[nom]
        if split_plat != split:
            divergences.append(f"{nom} : split {split_plat} au lieu de {split}")
        if n_plat != n_annos:
            divergences.append(f"{nom} : {n_annos} annotations envoyées, "
                               f"{n_plat} sur la plateforme")
    return divergences


if __name__ == "__main__":
    main()
