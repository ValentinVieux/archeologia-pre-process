"""Les poids déployés sont-ils CEUX QU'ON A MESURÉS ?

Garde-fou contre le bug best_ema/best_total, qui a livré deux modèles sur des poids jamais
évalués (crateres_seg_ld_v1 le 2026-09-11, tranchees_seg_ld_v1 le 2026-09-10 — découvert le
2026-09-14) : la section d'évaluation du notebook mesure `checkpoint_best_ema.pth` tandis que le
packaging copie `checkpoint_best_total.pth` dans `package/weights/best.pth`. Rien ne comparait
les deux.

Chaque évaluation enregistre l'IDENTITÉ du checkpoint mesuré à deux endroits :
  - `metriques_eval.json` -> `modeles.<nom>.poids` : son chemin ;
  - `appariements.json`   -> `_meta.modeles.<nom>.taille_octets` : sa TAILLE EXACTE.

La taille est la preuve la plus solide, parce qu'elle survit à la disparition du fichier (les
brouillons `D:\\entrainement_*` sont purgés à la clôture) et qu'elle ne peut pas être fabriquée
par une homonymie de nom de fichier. Les trois checkpoints d'un run rfdetr ont des tailles
distinctes — best_ema 144 234 793, best_total 144 257 681, best_regular 144 273 909 sur le run
tranchées : la taille seule aurait suffi à attraper l'incident.

On contrôle donc, dans cet ordre : taille de `weights/best.pth` contre `taille_octets` mesurée,
puis — si les tailles concordent et que le fichier mesuré est retrouvable — son empreinte.

Seuls les dossiers `evaluation*` sont contrôlés, et dans chacun la SEULE entrée du modèle
déployé : un `comparaison_*` mesure d'autres poids par construction, et une entrée de baseline
dans une éval superposée n'a aucune raison de valoir les poids livrés.

    .venv\\Scripts\\python.exe tools\\verif_poids_evalues.py <data/models du plugin> [--strict]

Verdicts : CONFORME (empreinte identique) · CONFORME_TAILLE (taille identique, fichier mesuré
introuvable — présomption, pas preuve) · AUTO_REFERENCE (l'éval a mesuré le fichier livré
lui-même : vrai par construction, ne prouve rien sur un remplacement ultérieur) · SUSPECT (les
poids livrés ne sont PAS ceux mesurés) · INDÉTERMINABLE (l'éval n'enregistre pas de quoi
conclure). Sortie 2 si un SUSPECT ; avec --strict, 2 aussi sur tout ce qui n'est pas CONFORME.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PREFIXES_DRIVE = ("/content/drive/MyDrive/", "/content/drive/My Drive/")
RACINE_DRIVE = "G:/Mon Drive/"

CONFORME, CONFORME_TAILLE = "CONFORME", "CONFORME_TAILLE"
AUTO_REFERENCE, SUSPECT, INDETERMINABLE = "AUTO_REFERENCE", "SUSPECT", "INDÉTERMINABLE"
# du pire au meilleur : le verdict d'un modèle est le pire de ses évaluations
ORDRE = [SUSPECT, INDETERMINABLE, AUTO_REFERENCE, CONFORME_TAILLE, CONFORME]


def empreinte(chemin: Path, taille_bloc: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as flux:
        for bloc in iter(lambda: flux.read(taille_bloc), b""):
            h.update(bloc)
    return h.hexdigest()


def chemin_mesure(poids: str) -> Path | None:
    """Le fichier mesuré, s'il est atteignable depuis ce poste (chemin Colab -> G:)."""
    if not poids.strip():
        return None
    for prefixe in PREFIXES_DRIVE:
        if poids.startswith(prefixe):
            candidat = Path(RACINE_DRIVE + poids[len(prefixe):])
            return candidat if candidat.is_file() else None
    candidat = Path(poids)
    return candidat if candidat.is_file() else None


def entree_du_modele(donnees: dict, ident: str) -> tuple[str, dict] | None:
    """L'entrée de `modeles` qui décrit le modèle DÉPLOYÉ, jamais une baseline.

    On exige que la clé corresponde à l'id du modèle (exactement, ou comme préfixe : le notebook
    suffixe parfois l'époque, « ponctuelles_2cl_det_ld_v1_ep34 »). Une éval superposée contient
    aussi les baselines : les confondre rendrait un SUSPECT faux.
    """
    modeles = donnees.get("modeles") or {}
    if not modeles:
        return None
    for nom in modeles:
        if nom == ident:
            return nom, modeles[nom]
    candidats = [n for n in modeles if n.startswith(ident)]
    if len(candidats) == 1:
        return candidats[0], modeles[candidats[0]]
    if len(modeles) == 1:
        nom = next(iter(modeles))
        return nom, modeles[nom]
    return None


def taille_mesuree(dossier_eval: Path, nom_entree: str) -> int | None:
    """La taille du checkpoint mesuré, lue dans le cache d'appariements posé à côté."""
    cache = dossier_eval / "appariements.json"
    if not cache.is_file():
        return None
    try:
        meta = json.loads(cache.read_text(encoding="utf-8")).get("_meta") or {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    entree = (meta.get("modeles") or {}).get(nom_entree) or {}
    taille = entree.get("taille_octets")
    return int(taille) if isinstance(taille, (int, float)) else None


def controler_une_eval(best: Path, taille_livree: int, sha_livree: str | None,
                       fichier: Path, ident: str) -> tuple[str, str, str | None]:
    """(verdict, explication, sha de best.pth calculé si besoin) pour UNE évaluation."""
    ou = fichier.parent.name
    try:
        donnees = json.loads(fichier.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as erreur:
        return INDETERMINABLE, f"{ou}: metriques_eval.json illisible ({erreur})", sha_livree

    trouve = entree_du_modele(donnees, ident)
    if trouve is None:
        noms = ", ".join((donnees.get("modeles") or {})) or "aucune"
        return INDETERMINABLE, (f"{ou}: aucune entrée de `modeles` ne correspond à « {ident} » "
                                f"(présentes : {noms}) — rien à comparer"), sha_livree
    nom_entree, modele = trouve
    poids = str(modele.get("poids", "")).strip()
    if not poids:
        return INDETERMINABLE, f"{ou}: entrée « {nom_entree} » sans champ `poids`", sha_livree

    # 1. la taille — preuve disponible même si le fichier mesuré a disparu
    attendue = taille_mesuree(fichier.parent, nom_entree)
    if attendue is not None and attendue != taille_livree:
        return SUSPECT, (f"{ou}: MESURÉ {Path(poids).name} de {attendue} o != LIVRÉ best.pth de "
                         f"{taille_livree} o — les chiffres publiés ne décrivent pas ce modèle"), sha_livree

    # 2. l'empreinte, quand le fichier mesuré est encore là
    mesure = chemin_mesure(poids)
    if mesure is not None and mesure.resolve() == best.resolve():
        return AUTO_REFERENCE, (f"{ou}: l'éval a mesuré le fichier livré lui-même ({poids}) — vrai "
                                f"par construction, ne prouve rien sur un remplacement ultérieur"), sha_livree
    if mesure is not None:
        if mesure.stat().st_size != taille_livree:
            return SUSPECT, (f"{ou}: MESURÉ {mesure} de {mesure.stat().st_size} o != LIVRÉ best.pth "
                             f"de {taille_livree} o"), sha_livree
        sha_livree = sha_livree or empreinte(best)
        if empreinte(mesure) == sha_livree:
            return CONFORME, f"{ou}: {Path(poids).name} == weights/best.pth (empreinte)", sha_livree
        return SUSPECT, (f"{ou}: MESURÉ {Path(poids).name} et LIVRÉ best.pth ont la même taille mais "
                         f"des empreintes DIFFÉRENTES — source {mesure}"), sha_livree

    if attendue is not None:
        return CONFORME_TAILLE, (f"{ou}: {Path(poids).name} introuvable sur ce poste ; taille mesurée "
                                 f"{attendue} o == best.pth — présomption, pas preuve"), sha_livree
    return INDETERMINABLE, (f"{ou}: checkpoint mesuré introuvable ({poids}) et aucune "
                            f"`taille_octets` dans appariements.json — rien à comparer"), sha_livree


def controler(dossier: Path) -> tuple[str, list[str]]:
    best = dossier / "weights" / "best.pth"
    if not best.is_file():
        return INDETERMINABLE, ["weights/best.pth absent"]
    evaluations = sorted(dossier.glob("entrainement/evaluation*/metriques_eval.json"))
    if not evaluations:
        return INDETERMINABLE, ["aucune évaluation outillée (entrainement/evaluation*/)"]

    taille = best.stat().st_size
    sha: str | None = None
    verdicts, lignes = [], []
    for fichier in evaluations:
        verdict, ligne, sha = controler_une_eval(best, taille, sha, fichier, dossier.name)
        verdicts.append(verdict)
        lignes.append(ligne)
    return min(verdicts, key=ORDRE.index), lignes


def main() -> int:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("racine", type=Path,
                         help="dossier data/models du plugin, ou un dossier de modèle unique")
    parseur.add_argument("--strict", action="store_true",
                         help="sortir en erreur sur tout ce qui n'est pas CONFORME (recommandé "
                              "avant installation : INDÉTERMINABLE est ce que rendent les deux "
                              "bugs historiques quand le checkpoint mesuré est hors de portée)")
    args = parseur.parse_args()

    racine: Path = args.racine
    modeles = ([racine] if (racine / "weights").is_dir()
               else sorted(p for p in racine.iterdir() if p.is_dir() and (p / "weights").is_dir()))
    if not modeles:
        print(f"aucun modèle sous {racine}", file=sys.stderr)
        return 1

    marques = {CONFORME: "  ok  ", CONFORME_TAILLE: " ~ok~ ", AUTO_REFERENCE: " auto ",
               SUSPECT: " SUSP ", INDETERMINABLE: " ???? "}
    compte: dict[str, int] = {v: 0 for v in ORDRE}
    for modele in modeles:
        verdict, lignes = controler(modele)
        compte[verdict] += 1
        print(f"[{marques[verdict]}] {modele.name}")
        for ligne in lignes:
            print(f"           {ligne}")

    print(f"\n{len(modeles)} modèle(s) : "
          + ", ".join(f"{compte[v]} {v}" for v in reversed(ORDRE) if compte[v]))
    if compte[SUSPECT]:
        print("\nUn SUSPECT signifie que les chiffres publiés (model_card, registre, dashboard) ne "
              "décrivent PAS les poids installés. Remesurer les poids LIVRÉS avec "
              "tools/courbes_eval.py avant de rien conclure sur la qualité du modèle.")
        return 2
    if args.strict and (len(modeles) - compte[CONFORME]):
        print("\n--strict : seul CONFORME (empreinte vérifiée) est accepté.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
