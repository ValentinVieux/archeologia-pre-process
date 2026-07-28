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


def annotation_fichier(im, suffixe, renommages=None):
    """Fichier d'annotation à joindre : COCO pour une tuile annotée, VOC XML sans
    objet pour un négatif — le parseur refuse un COCO à zéro annotation, mais le VOC
    vide est accepté et vaut annotation NULL (image de fond assumée, vérifié le
    2026-07-27). Retourne (contenu, suffixe_de_fichier)."""
    if im["annotations"]:
        coco = coco_mono_image(im, suffixe)
        for c in coco["categories"]:
            c["name"] = (renommages or {}).get(c["name"], c["name"])
        return json.dumps(coco, ensure_ascii=False), ".coco.json"
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
    p.add_argument("--renommer", action="append", default=[],
                   help="ancien=nouveau : renomme une classe PLATEFORME (après "
                        "suffixe), répétable — ex. talus_fosse_haye=fosse")
    p.add_argument("--paralleles", type=int, default=10,
                   help="uploads concurrents (patron num_workers du SDK : défaut 10, "
                        "recommandation Roboflow <= 25)")
    args = p.parse_args()
    renommages = dict(r.split("=", 1) for r in args.renommer)
    if args.paralleles > 25:
        print("avertissement : Roboflow recommande <= 25 uploads concurrents")

    if chemin_sur_drive(args.dataset):
        sys.exit("dataset sur G: — uploader depuis la copie locale (règle Drive)")
    dossier = Path(args.dataset)
    manifeste, images, zone_id, region = charger_dataset(dossier)
    toutes_images = list(images)  # avant filtres --eviter/--test (pour la vérification)
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

    def _envoyer_image(i):
        """Upload d'UNE image (thread-safe : aucun état partagé mutable ici).

        Les négatifs reçoivent AUSSI une annotation (VOC vide = null) : sans
        annotation jointe, la plateforme les met en file Annotate. La réponse est
        CONTRÔLÉE (des annotations se perdaient en silence, constat 2026-07-27) ;
        en secours, écrasement direct via /annotate/:id?overwrite=true — gère
        aussi les fantômes dédupliqués."""
        contenu, suffixe_fichier = annotation_fichier(i, args.suffixe_classes,
                                                      renommages)
        tmp = tempfile.NamedTemporaryFile("w", suffix=suffixe_fichier,
                                          delete=False, encoding="utf-8")
        tmp.write(contenu)
        tmp.close()
        annotation_path = tmp.name
        erreur = None
        try:
            for _ in range(3):
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
                        return i, None
                    if img.get("id"):
                        nom_fichier = i["filename"].rsplit(".", 1)[0] + suffixe_fichier
                        import requests
                        ra = requests.post(
                            f"https://api.roboflow.com/dataset/{args.projet}"
                            f"/annotate/{img['id']}?api_key={cle}"
                            f"&name={nom_fichier}&overwrite=true",
                            data=contenu.encode("utf-8"),
                            headers={"Content-Type": "text/plain"}, timeout=60)
                        if ra.ok and ra.json().get("success"):
                            return i, None
                        erreur = (f"annotate overwrite refusé : {ra.status_code} "
                                  f"{ra.text[:120]}")
                    else:
                        erreur = f"réponse incomplète : {reponse}"
                except Exception as exc:  # noqa: BLE001 — on retente puis on rapporte
                    erreur = str(exc)
            return i, erreur
        finally:
            os.unlink(annotation_path)

    # parallélisation sur le patron num_workers du SDK (upload_dataset : défaut 10,
    # reco <= 25) — le tracker n'est touché que sous verrou, flush périodique pour
    # que la reprise idempotente survive à une interruption
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    verrou = threading.Lock()
    envoyees, echecs = 0, []

    def _flush_suivi():
        suivi["derniere_mise_a_jour"] = datetime.datetime.now().isoformat(
            timespec="seconds")
        suivi_path.write_text(yaml.safe_dump(suivi, allow_unicode=True,
                                             sort_keys=False), encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.paralleles)) as pool:
            futurs = [pool.submit(_envoyer_image, i) for i in a_envoyer]
            for fut in as_completed(futurs):
                i, erreur = fut.result()
                with verrou:
                    if erreur:
                        echecs.append((i["filename"], erreur))
                        print(f"  ÉCHEC {i['filename']} : {erreur[:120]}")
                    else:
                        suivi["images"].append(
                            {"filename": i["filename"], "split": i["split"],
                             "annotations": len(i["annotations"])})
                        envoyees += 1
                        if envoyees % 50 == 0 or envoyees == len(a_envoyer):
                            print(f"  {envoyees}/{len(a_envoyer)}")
                            _flush_suivi()
    finally:
        _flush_suivi()
    print(f"terminé : {envoyees} envoyées, {len(echecs)} échec(s). Suivi : {suivi_path}")

    if not args.sans_verification:
        # toutes les images du suivi (pas seulement la passe courante : une relance
        # doit pouvoir re-vérifier un état déjà envoyé), avec reprises espacées —
        # l'ingestion des annotations côté plateforme est asynchrone
        attendu = {s["filename"]: (s["split"], s["annotations"])
                   for s in suivi["images"]}
        from collections import Counter
        cats_par_fichier = {}
        for i in toutes_images:
            noms_cats = {c["id"]: c["name"] for c in i["categories"]}
            cats_par_fichier[i["filename"]] = Counter(
                noms_cats[a["category_id"]] for a in i["annotations"])
        splits_attendus = Counter(s["split"] for s in suivi["images"])
        classes_attendues = Counter()
        images_par_classe = Counter()
        for s_im in suivi["images"]:
            cpt = cats_par_fichier.get(s_im["filename"], {})
            for classe, n in cpt.items():
                nom = (f"{classe}_{args.suffixe_classes}"
                       if args.suffixe_classes else classe)
                classes_attendues[renommages.get(nom, nom)] += n
            # ensemble des classes FINALES de l'image : un renommage qui
            # FUSIONNE deux classes (ex. voie->parcellaire) ne doit compter
            # l'image qu'une fois pour la classe cible
            finales = set()
            for classe in cpt:
                nom = (f"{classe}_{args.suffixe_classes}"
                       if args.suffixe_classes else classe)
                finales.add(renommages.get(nom, nom))
            for nom in finales:
                images_par_classe[nom] += 1
        divergences = []
        for attente in (0, 60, 120, 180):
            if attente:
                print(f"  divergences restantes : {len(divergences)} — "
                      f"nouvel essai dans {attente} s (ingestion asynchrone)")
                time.sleep(attente)
            divergences = verifier_plateforme(
                cle, args.workspace, args.projet, attendu,
                dict(splits_attendus), dict(classes_attendues),
                dict(images_par_classe), zone_tag=zone_id)
            if not divergences:
                break
        if divergences:
            for d in divergences:
                print(f"  DIVERGENCE : {d}")
            sys.exit(f"vérification post-upload : {len(divergences)} divergence(s) — "
                     "ne PAS lancer d'entraînement sur cet état.")
        print(f"vérification post-upload : agrégats plateforme conformes "
              f"({len(attendu)} images, splits {dict(splits_attendus)}, "
              f"classes {dict(classes_attendues)}) + échantillon par image OK.")
    if echecs:
        sys.exit(f"{len(echecs)} image(s) en échec — relancer la même commande "
                 "(reprise idempotente) puis investiguer si ça persiste.")
    print(f"Sorties : {suivi_path}")


def verifier_plateforme(cle, workspace, projet, attendu, splits_attendus,
                        classes_attendues, images_par_classe_attendues=None,
                        zone_tag=None):
    """Vérification post-upload : agrégats du projet + échantillon par image.

    L'endpoint search PLAFONNE à 250 résultats avec un ordre instable entre pages
    (constat 2026-07-27 : une pagination par offset « perd » des images) — il ne sert
    donc que d'échantillon par image. La vérité globale vient des agrégats du projet
    (GET /:workspace/:projet -> images, splits, classes), exacts et immédiats.
    Témoin par image : champ `annotations` ({count, classes}) — `labels` est toujours
    vide et produit de faux diagnostics.
    """
    import requests

    base = f"https://api.roboflow.com/{workspace}/{projet}"
    divergences = []
    images_par_classe_attendues = images_par_classe_attendues or {}

    # Un projet peut héberger PLUSIEURS zones (corpus multi-zones) : toutes les
    # requêtes sont scopées par le tag de zone. Sans tag, les agrégats projet
    # (images/splits) sont comparés — valable seulement en mono-zone.
    filtre_zone = {"tag": zone_tag} if zone_tag else {}
    if zone_tag:
        rt = requests.post(f"{base}/search?api_key={cle}", json={
            "limit": 1, "in_dataset": True, **filtre_zone,
            "fields": ["id"]}, timeout=60)
        rt.raise_for_status()
        total_zone = rt.json().get("total", 0)
        if total_zone != len(attendu):
            divergences.append(f"total images de la zone {zone_tag} : {total_zone} "
                               f"sur la plateforme, {len(attendu)} attendues")
    else:
        r = requests.get(f"{base}?api_key={cle}", timeout=60)
        r.raise_for_status()
        projet_meta = r.json().get("project", {})
        if projet_meta.get("images") != len(attendu):
            divergences.append(f"total images : {projet_meta.get('images')} sur la "
                               f"plateforme, {len(attendu)} attendues")
        splits_plat = projet_meta.get("splits") or {}
        for s, n in sorted(splits_attendus.items()):
            if splits_plat.get(s) != n:
                divergences.append(f"split {s} : {splits_plat.get(s)} images sur la "
                                   f"plateforme, {n} attendues")
    # nombre d'IMAGES par classe (exact et immédiat) — les classes étant suffixées
    # par site, elles sont propres à la zone ; le compteur d'ANNOTATIONS par classe
    # du projet est un cache parfois périmé (a affiché -5) : jamais bloquant
    for c, n_imgs in sorted(images_par_classe_attendues.items()):
        rc = requests.post(f"{base}/search?api_key={cle}", json={
            "limit": 1, "in_dataset": True, "class_name": c,
            "fields": ["id"]}, timeout=60)
        rc.raise_for_status()
        total_c = rc.json().get("total", 0)
        if total_c != n_imgs:
            divergences.append(f"classe {c} : {total_c} images sur la plateforme, "
                               f"{n_imgs} attendues")

    rs = requests.post(f"{base}/search?api_key={cle}", json={
        "limit": 250, "in_dataset": True, **filtre_zone,
        "fields": ["name", "split", "annotations"]}, timeout=60)
    rs.raise_for_status()
    for res in rs.json().get("results", []):
        nom = res.get("name")
        if nom not in attendu:
            divergences.append(f"{nom} : présente sur la plateforme mais inconnue "
                               "du suivi")
            continue
        split, n_annos = attendu[nom]
        annos = res.get("annotations") or {}
        n_plat = annos.get("count", 0) if isinstance(annos, dict) else 0
        if res.get("split") != split:
            divergences.append(f"{nom} : split {res.get('split')} au lieu de {split}")
        if n_plat != n_annos:
            divergences.append(f"{nom} : {n_annos} annotations envoyées, "
                               f"{n_plat} sur la plateforme")
    return divergences


if __name__ == "__main__":
    main()
