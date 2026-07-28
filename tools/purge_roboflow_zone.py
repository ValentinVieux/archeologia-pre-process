"""Purge les images d'une ZONE (par tag) dans un projet Roboflow — préalable au
remplacement d'un dataset régénéré. Clé : env ROBOFLOW_API_KEY.

La suppression envoie à la CORBEILLE : la dédup par contenu pourra ressusciter
les images identiques au re-upload (ghost), c'est paré côté uploader
(annotate?overwrite=true). Le search plafonne à 250 à l'ordre instable : on
boucle jusqu'à zéro résultat au lieu de paginer.

Usage : python purge_roboflow_zone.py <zone_tag> --workspace <ws> --projet <p>
            [--dry-run] [--oui]
"""
import argparse
import os
import sys
import time

import requests


def rechercher(base, cle, tag, limit=250):
    r = requests.post(f"{base}/search?api_key={cle}", json={
        "limit": limit, "tag": tag, "fields": ["id", "name"]}, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j.get("total", 0), j.get("results", [])


def supprimer(base, cle, image_id):
    """Supprime une image — variantes d'endpoint + retries sur erreurs 5xx.

    Les messages d'erreur ne doivent JAMAIS contenir l'URL brute (la clé API y
    figure) : on ne remonte que le code HTTP et l'id d'image.
    """
    for chemin in (f"{base}/images/{image_id}", f"{base}/{image_id}"):
        for tentative in range(4):
            try:
                r = requests.delete(f"{chemin}?api_key={cle}", timeout=60)
            except requests.RequestException:
                time.sleep(2 * (tentative + 1))
                continue
            if r.status_code in (200, 204):
                return True
            if r.status_code in (404, 405):
                break  # variante d'endpoint suivante
            if r.status_code >= 500:  # transitoire : backoff puis retry
                time.sleep(2 * (tentative + 1))
                continue
            print(f"HTTP {r.status_code} sur suppression de {image_id}")
            return False
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zone_tag")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--projet", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--oui", action="store_true",
                    help="confirme la purge (sinon dry-run forcé)")
    args = ap.parse_args()
    cle = os.environ.get("ROBOFLOW_API_KEY")
    if not cle:
        sys.exit("ROBOFLOW_API_KEY absent de l'environnement")
    base = f"https://api.roboflow.com/{args.workspace}/{args.projet}"

    total, _ = rechercher(base, cle, args.zone_tag, limit=1)
    print(f"zone '{args.zone_tag}' : {total} images sur la plateforme")
    if args.dry_run or not args.oui:
        print("dry-run (utiliser --oui pour purger)")
        return

    supprimees, echecs = 0, 0
    while True:
        total, lot = rechercher(base, cle, args.zone_tag)
        if not lot:
            break
        for res in lot:
            if supprimer(base, cle, res["id"]):
                supprimees += 1
            else:
                echecs += 1
                print(f"ECHEC suppression {res.get('name')} ({res['id']})")
            if supprimees % 100 == 0 and supprimees:
                print(f"  {supprimees} supprimées…")
            time.sleep(0.05)
        if echecs > 20:
            sys.exit(f"trop d'échecs ({echecs}) — arrêt, {supprimees} supprimées")
    reste, _ = rechercher(base, cle, args.zone_tag, limit=1)
    print(f"purge terminée : {supprimees} supprimées, {echecs} échecs, "
          f"reste sur la plateforme : {reste}")
    if reste:
        sys.exit("des images subsistent — relancer ou inspecter")


if __name__ == "__main__":
    main()
