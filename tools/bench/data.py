"""Chargement du corpus, géoréférencement des tuiles, sélection du sous-ensemble."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Origines L93 des grilles de tuilage, par zone (coin haut-gauche), depuis les
# split_manifest.yaml. GSD 0,5 m, tuile 648 px -> pas de 324 m.
ORIGINES = {
    "54_foret_de_haye":        (919000.0, 6856000.0),
    "77_fontainebleau":        (655000.0, 6825000.0),
    "78_rambouillet":          (603000.0, 6855000.0),
    "78_saint_germain_marly":  (625000.0, 6878000.0),
    "41_blois":                (565000.0, 6731000.0),
}
PAS_M = 324.0
TUILE_PX = 648
GSD_M = 0.5

# `78_rambouillet` contient `_r` : tout parseur qui coupe sur `_r` se trompe.
# Le regex est ancré sur le SUFFIXE, les \d{4} forcent le bon point de coupe.
_RE_TUILE = re.compile(r"^(?P<zone>.+)_r(?P<row>\d{4})_c(?P<col>\d{4})\.(png|jpg)$")


@dataclass(frozen=True)
class Tuile:
    nom: str
    zone: str
    row: int
    col: int

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        x0, y0 = ORIGINES[self.zone]
        xmin = x0 + self.col * PAS_M
        ymax = y0 - self.row * PAS_M
        return (xmin, ymax - PAS_M, xmin + PAS_M, ymax)


def parse_tuile(nom: str) -> Optional[Tuile]:
    m = _RE_TUILE.match(nom)
    if not m or m.group("zone") not in ORIGINES:
        return None
    return Tuile(nom, m.group("zone"), int(m.group("row")), int(m.group("col")))


class Corpus:
    """Un split COCO (test / valid) + index par image."""

    def __init__(self, dossier: Path):
        self.dir = Path(dossier)
        d = json.loads((self.dir / "_annotations.coco.json").read_text(encoding="utf-8"))
        self.images = {i["id"]: i for i in d["images"]}
        self.categories = {c["id"]: c["name"] for c in d["categories"]}
        self.anns: Dict[int, List[dict]] = defaultdict(list)
        for a in d["annotations"]:
            self.anns[a["image_id"]].append(a)
        # cat_id COCO (1..5) -> class_id modèle (0..4)
        self.cat_vers_classe = {c: i for i, c in enumerate(sorted(self.categories))}
        self.classe_vers_cat = {i: c for c, i in self.cat_vers_classe.items()}
        self.noms_classes = [self.categories[self.classe_vers_cat[i]]
                             for i in range(len(self.categories))]

    def __len__(self) -> int:
        return len(self.images)

    def zone(self, img: dict) -> str:
        return img.get("zone") or (parse_tuile(img["file_name"]).zone)

    def chemin(self, img: dict) -> Path:
        return self.dir / img["file_name"]

    def classes_presentes(self, iid: int) -> frozenset:
        return frozenset(a["category_id"] for a in self.anns.get(iid, []))

    def stats(self) -> dict:
        par_classe = defaultdict(int)
        img_par_classe = defaultdict(int)
        for iid in self.images:
            for c in self.classes_presentes(iid):
                img_par_classe[c] += 1
            for a in self.anns.get(iid, []):
                par_classe[a["category_id"]] += 1
        return {
            "n_images": len(self.images),
            "n_annotations": sum(par_classe.values()),
            "n_negatifs": sum(1 for i in self.images if not self.anns.get(i)),
            "annotations_par_classe": {self.categories[c]: n for c, n in sorted(par_classe.items())},
            "images_par_classe": {self.categories[c]: n for c, n in sorted(img_par_classe.items())},
        }


# --------------------------------------------------------------------------------------
# Sélection du sous-ensemble — déterministe, stratifiée, recensement des classes rares
# --------------------------------------------------------------------------------------

def _rang(seed: str, nom: str) -> int:
    return int(hashlib.sha1(f"{seed}:{nom}".encode()).hexdigest()[:16], 16)


def selectionner(corpus: Corpus, n_cible: int, seed: str = "bench-v1",
                 classes_rares: Sequence[str] = ("fosse", "talus", "chemin_creux"),
                 taux_negatifs: float = 0.10,
                 plancher_strate: int = 5) -> dict:
    """Sous-ensemble stratifié.

    Couche RECENSEMENT : toutes les images contenant une classe rare sont prises. Un
    tirage aléatoire naïf rendrait `fosse` (50 images sur 830) non mesurable.
    Couche ÉCHANTILLONNÉE : allocation proportionnelle au nombre d'annotations, plancher
    par strate non vide.
    Négatifs : exactement le taux du corpus (10 %), pour que la précision reste
    comparable d'un sous-ensemble à l'autre.

    Le tri par sha1(seed:nom) est stable si une strate change de taille, contrairement
    à random.sample — les sous-ensembles restent comparables d'une régénération à l'autre.
    """
    cats_rares = {c for c, n in corpus.categories.items() if n in classes_rares}

    recensement, positives, negatives = [], [], []
    for iid, img in corpus.images.items():
        cls = corpus.classes_presentes(iid)
        if not cls:
            negatives.append(iid)
        elif cls & cats_rares:
            recensement.append(iid)
        else:
            positives.append(iid)

    # Strates sur la couche échantillonnée : zone x signature de classes x tercile de densité
    def densite_tercile(iid: int, bornes: Dict[str, Tuple[float, float]]) -> int:
        n = len(corpus.anns.get(iid, []))
        z = corpus.zone(corpus.images[iid])
        b1, b2 = bornes[z]
        return 0 if n <= b1 else (1 if n <= b2 else 2)

    par_zone = defaultdict(list)
    for iid in positives:
        par_zone[corpus.zone(corpus.images[iid])].append(len(corpus.anns.get(iid, [])))
    bornes = {}
    for z, ns in par_zone.items():
        ns = sorted(ns)
        bornes[z] = (ns[len(ns) // 3] if ns else 0, ns[2 * len(ns) // 3] if ns else 0)

    strates: Dict[tuple, List[int]] = defaultdict(list)
    for iid in positives:
        img = corpus.images[iid]
        cle = (corpus.zone(img), tuple(sorted(corpus.classes_presentes(iid))),
               densite_tercile(iid, bornes))
        strates[cle].append(iid)

    budget_annote = max(0, n_cible - int(round(n_cible * taux_negatifs)))
    reste = budget_annote - len(recensement)

    choisis = list(recensement)
    pi = {iid: 1.0 for iid in recensement}

    if reste > 0 and strates:
        poids = {k: sum(len(corpus.anns.get(i, [])) for i in v) for k, v in strates.items()}
        total = sum(poids.values()) or 1
        alloc = {k: max(plancher_strate, int(round(reste * poids[k] / total)))
                 for k in strates}
        # Ramener au budget si les planchers font déborder
        while sum(min(alloc[k], len(strates[k])) for k in strates) > reste:
            k = max(strates, key=lambda k: min(alloc[k], len(strates[k])))
            if alloc[k] <= 1:
                break
            alloc[k] -= 1
        for k, ids in strates.items():
            k_s = min(alloc[k], len(ids))
            pris = sorted(ids, key=lambda i: _rang(seed, corpus.images[i]["file_name"]))[:k_s]
            choisis.extend(pris)
            for i in ids:
                pi[i] = k_s / len(ids)

    n_neg = int(round(len(choisis) * taux_negatifs / max(1e-9, 1 - taux_negatifs)))
    n_neg = min(n_neg, len(negatives))
    neg_pris = sorted(negatives, key=lambda i: _rang(seed, corpus.images[i]["file_name"]))[:n_neg]
    choisis.extend(neg_pris)
    for i in negatives:
        pi[i] = n_neg / len(negatives) if negatives else 0.0

    choisis = sorted(set(choisis))
    return {
        "seed": seed,
        "n_cible": n_cible,
        "image_ids": choisis,
        "fichiers": [corpus.images[i]["file_name"] for i in choisis],
        "pi": {corpus.images[i]["file_name"]: pi.get(i, 1.0) for i in choisis},
        "n_recensement": len(recensement),
        "n_negatifs": len(neg_pris),
        "resume": _resume(corpus, choisis),
    }


def _resume(corpus: Corpus, ids: Sequence[int]) -> dict:
    par_classe, par_zone = defaultdict(int), defaultdict(int)
    img_par_classe = defaultdict(int)
    for iid in ids:
        par_zone[corpus.zone(corpus.images[iid])] += 1
        for c in corpus.classes_presentes(iid):
            img_par_classe[c] += 1
        for a in corpus.anns.get(iid, []):
            par_classe[a["category_id"]] += 1
    return {
        "n_images": len(ids),
        "n_annotations": sum(par_classe.values()),
        "annotations_par_classe": {corpus.categories[c]: n for c, n in sorted(par_classe.items())},
        "images_par_classe": {corpus.categories[c]: n for c, n in sorted(img_par_classe.items())},
        "images_par_zone": dict(sorted(par_zone.items())),
    }


# --------------------------------------------------------------------------------------
# Composantes connexes de tuiles -> mosaïques (niveau B)
# --------------------------------------------------------------------------------------

def composantes(tuiles: Sequence[Tuile], min_tuiles: int = 4) -> List[List[Tuile]]:
    """Composantes 4-connexes sur la grille (row, col), par zone.

    Pas de tolérance de trou, pas de 8-connexité : une jonction en diagonale produirait
    une mosaïque dont la couture est un coin d'un pixel, pire que deux mosaïques.
    """
    par_zone = defaultdict(dict)
    for t in tuiles:
        par_zone[t.zone][(t.row, t.col)] = t

    out = []
    for zone, grille in par_zone.items():
        vus = set()
        for depart in grille:
            if depart in vus:
                continue
            pile, comp = [depart], []
            vus.add(depart)
            while pile:
                r, c = pile.pop()
                comp.append(grille[(r, c)])
                for v in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if v in grille and v not in vus:
                        vus.add(v)
                        pile.append(v)
            if len(comp) >= min_tuiles:
                out.append(sorted(comp, key=lambda t: (t.row, t.col)))
    return sorted(out, key=lambda c: -len(c))
