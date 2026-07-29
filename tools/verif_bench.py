"""Contrôleur INDÉPENDANT du banc d'essai.

Règle projet (CLAUDE.md) : produire -> vérifier les FICHIERS produits par un contrôleur
indépendant -> corriger -> reproduire. Ce script ne partage aucun état avec le
producteur : il relit tout depuis le disque et sort en code non nul au moindre défaut.

    python tools/verif_bench.py --out D:\\pipeline_results\\bench
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ECHECS: list[str] = []
CONTROLES = 0


def verifier(condition: bool, message: str) -> None:
    global CONTROLES
    CONTROLES += 1
    if not condition:
        ECHECS.append(message)


def verif_cache(racine: Path) -> None:
    dossier = racine / "cache"
    if not dossier.exists():
        ECHECS.append("cache/ absent")
        return
    for cle in sorted(p for p in dossier.iterdir() if p.is_dir()):
        meta_path = cle / "meta.json"
        verifier(meta_path.exists(), f"{cle.name}/meta.json absent")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for champ in ("modele", "ort", "provider", "prepro", "input_hw", "plancher"):
            verifier(champ in meta, f"{cle.name}/meta.json : champ {champ} manquant")
        npz = sorted(cle.glob("*.npz"))
        verifier(len(npz) > 0, f"{cle.name} : aucun .npz")
        if not npz:
            continue
        # Échantillon : 5 fichiers répartis, relus intégralement.
        pas = max(1, len(npz) // 5)
        for p in npz[::pas][:5]:
            try:
                d = np.load(p)
            except Exception as e:
                ECHECS.append(f"{cle.name}/{p.name} illisible : {e}")
                continue
            for k in ("boxes", "logits", "masks", "qidx", "logits_full"):
                verifier(k in d.files, f"{p.name} : tableau {k} manquant")
            if not {"boxes", "logits", "masks", "qidx"} <= set(d.files):
                continue
            k_n = len(d["qidx"])
            verifier(d["boxes"].shape == (k_n, 4), f"{p.name} : boxes {d['boxes'].shape} != ({k_n},4)")
            verifier(d["logits"].shape[0] == k_n, f"{p.name} : logits {d['logits'].shape[0]} != {k_n}")
            verifier(d["masks"].shape[0] == k_n, f"{p.name} : masks {d['masks'].shape[0]} != {k_n}")
            verifier(d["masks"].dtype == np.float16, f"{p.name} : masks dtype {d['masks'].dtype} != float16")
            verifier(np.isfinite(d["logits"]).all(), f"{p.name} : logits non finis")
            # Le plancher annoncé doit être respecté : aucune requête cachée sous le seuil.
            sc = 1.0 / (1.0 + np.exp(-d["logits"].astype(np.float64)))
            if k_n:
                verifier(float(sc.max(axis=1).min()) >= meta["plancher"] - 1e-6,
                         f"{p.name} : requete sous le plancher {meta['plancher']}")
            # Et rien au-dessus du plancher ne doit avoir été jeté.
            sf = 1.0 / (1.0 + np.exp(-d["logits_full"].astype(np.float64)))
            verifier(int((sf.max(axis=1) >= meta["plancher"]).sum()) == k_n,
                     f"{p.name} : {int((sf.max(axis=1) >= meta['plancher']).sum())} requetes "
                     f"au-dessus du plancher mais {k_n} cachees")
        print(f"  cache {cle.name} : {len(npz)} npz, plancher {meta['plancher']}, "
              f"{meta['provider']}, split {meta.get('split','?')}")


def verif_runs(racine: Path) -> None:
    manifest = racine / "manifest.jsonl"
    if not manifest.exists():
        print("  (pas encore de manifest.jsonl)")
        return
    lignes = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    verifier(len(lignes) > 0, "manifest.jsonl vide")
    doublons = [c for c, n in Counter((l.get("_run"), l["_config"]) for l in lignes).items() if n > 1]
    verifier(not doublons, f"manifest : configs en double {doublons[:3]}")

    for l in lignes:
        nom = f"{l.get('_run')}/{l['_config']}"
        # Cohérence interne des compteurs de longueur.
        verifier(l["len_gt_m"] >= 0 and l["len_pred_m"] >= 0, f"{nom} : longueur negative")
        for cle_c, cle_l in (("completude", "len_gt_m"), ("correction", "len_pred_m")):
            v = l.get(cle_c)
            if v is not None and v == v:
                verifier(-1e-9 <= v <= 1 + 1e-9, f"{nom} : {cle_c}={v} hors [0,1]")
        c, r, f1 = l.get("completude"), l.get("correction"), l.get("f1_len")
        if all(x == x for x in (c, r, f1)) and (c + r) > 0:
            verifier(abs(f1 - 2 * c * r / (c + r)) < 1e-6,
                     f"{nom} : F1_len={f1} incoherent avec comp={c} corr={r}")
        verifier(l["n_pred"] >= 0 and l["n_gt"] >= 0, f"{nom} : compteur negatif")
        if l.get("aire_km2"):
            attendu = l["n_pred"] / l["aire_km2"]
            verifier(abs(l["polygones_par_km2"] - attendu) < 1e-6,
                     f"{nom} : polygones_par_km2 incoherent")

    # Une seule config `base` par run, et elle doit exister.
    for run in {l.get("_run") for l in lignes}:
        du_run = [l for l in lignes if l.get("_run") == run]
        verifier(any(l["_config"] == "base" for l in du_run),
                 f"run {run} : pas de config 'base' (reference de comparaison absente)")
        dossier = racine / "runs" / str(run) / "resultats.json"
        verifier(dossier.exists(), f"run {run} : runs/{run}/resultats.json absent")
        if dossier.exists():
            res = json.loads(dossier.read_text(encoding="utf-8"))
            manquants = {l["_config"] for l in du_run} - set(res)
            verifier(not manquants, f"run {run} : configs du manifest absentes du json {sorted(manquants)[:3]}")
        print(f"  run {run} : {len(du_run)} configs")

    # Toutes les configs d'un run doivent porter sur le même nombre d'images,
    # sinon les F1 ne sont pas comparables entre elles.
    for run in {l.get("_run") for l in lignes}:
        n = {l["n_images"] for l in lignes if l.get("_run") == run}
        verifier(len(n) == 1, f"run {run} : nombre d'images variable entre configs {sorted(n)}")


def verif_detail(racine: Path) -> None:
    for npz in sorted((racine / "runs").glob("*/detail_par_image.npz")) if (racine / "runs").exists() else []:
        d = np.load(npz)
        for cfg in d.files:
            a = d[cfg]
            if a.size == 0:
                continue
            verifier(a.shape[1] == 6, f"{npz.parent.name}/{cfg} : {a.shape[1]} colonnes != 6")
            verifier((a[:, 3] <= a[:, 1] + 1e-6).all(),
                     f"{npz.parent.name}/{cfg} : longueur GT couverte > longueur GT")
            verifier((a[:, 4] <= a[:, 2] + 1e-6).all(),
                     f"{npz.parent.name}/{cfg} : longueur predite couverte > longueur predite")
        print(f"  detail {npz.parent.name} : {len(d.files)} configs, "
              f"{d[d.files[0]].shape[0]} images")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\pipeline_results\bench")
    a = ap.parse_args()
    racine = Path(a.out)
    print(f"Verification independante de {racine}\n")
    verif_cache(racine)
    verif_runs(racine)
    verif_detail(racine)

    print(f"\n{CONTROLES} controles")
    if ECHECS:
        print(f"\n{len(ECHECS)} ECHEC(S) :")
        for e in ECHECS[:30]:
            print(f"  - {e}")
        return 1
    print("VERDICT : conforme")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
