# -*- coding: utf-8 -*-
"""Génère l'index HTML auto-suffisant de data_regions_v2.

Usage :
    .venv\\Scripts\\python.exe tools\\build_v2_index.py <racine_data_regions_v2> [-o index.html]

Scanne la racine (manifest.yaml des zones, upload_manifest.yaml des datasets,
_annotations.coco.json, raw/, _a_trier/, _archives_roboflow/) et écrit
<racine>/index.html : page statique, CSS/JS inline, ouvrable en file://.
À relancer après chaque évolution du dossier.

Convention de notes typées dans manifest.yaml (voir CLAUDE.md § Stockage Drive) :
    ARBITRAGE: ...        -> bloquant, remonte en tête de page
    TODO: ...             -> action à mener
    ATTENTION: ...        -> alerte non bloquante
    DÉCISION <date>: ...  -> actée, informatif
    (sans préfixe)        -> info
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import unicodedata
from datetime import date
from pathlib import Path

import yaml

TEMPLATE = Path(__file__).with_name("v2_index_template.html")
IGNORE = {"desktop.ini", "thumbs.db", "index.html"}
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}

NOTE_RE = re.compile(
    r"^\s*(ARBITRAGE(?:\s+PENDANT)?|TODO|ATTENTION|D[ÉE]CISION[^:]*)\s*:\s*(.*)$",
    re.IGNORECASE | re.DOTALL)
NOTE_TYPE = {"arbitrage": "arbitrage", "arbitrage pendant": "arbitrage",
             "todo": "todo", "attention": "attention"}


def norm(s: str) -> str:
    """minuscules sans accents (haystack de recherche) — même esprit que audit.scan.normalize."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def parse_note(txt: str) -> dict:
    m = NOTE_RE.match(txt)
    if m:
        head = norm(m.group(1)).strip()
        typ = NOTE_TYPE.get(head, "decision" if head.startswith("decision") else None)
        if typ:
            return {"type": typ, "texte": m.group(2).strip()}
    first = txt.strip().split(":", 1)[0].split()[0] if txt.strip() else ""
    if first.isupper() and len(first) > 2 and first not in ("CRS", "EPSG", "MNT", "RVT"):
        print(f"  [note?] préfixe non reconnu « {first} » : {txt[:70]}...")
    return {"type": "info", "texte": txt.strip()}


def dir_stats(root: Path) -> dict:
    n, size, exts = 0, 0, collections.Counter()
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if f.lower() in IGNORE:
                continue
            n += 1
            exts[os.path.splitext(f)[1].lower() or "(sans ext)"] += 1
            try:
                size += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    top = exts.most_common(8)
    return {"n_files": n, "size": size, "exts": dict(top),
            "exts_autres": n - sum(c for _, c in top)}


def read_dataset(dsdir: Path) -> dict:
    """Un dossier transformed/roboflow/<dataset> ou _a_trier/<dataset>."""
    out = {"name": dsdir.name, "splits": {}, "classes": {}, "manifest": None}
    um = dsdir / "upload_manifest.yaml"
    if um.exists():
        m = yaml.safe_load(um.read_text(encoding="utf-8"))
        out["manifest"] = {k: m.get(k) for k in
                           ("source_export", "dispatched", "zone", "region", "tags",
                            "attribution_methods")}
    split_dirs = sorted((p for p in dsdir.iterdir() if p.is_dir()),
                        key=lambda p: (SPLIT_ORDER.get(p.name, 9), p.name))
    for split_dir in split_dirs:
        ann = split_dir / "_annotations.coco.json"
        n_img = sum(1 for f in split_dir.iterdir()
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rec = {"images": n_img, "annotations": 0}
        if ann.exists():
            coco = json.loads(ann.read_text(encoding="utf-8"))
            cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
            per = collections.Counter(cats[a["category_id"]]
                                      for a in coco.get("annotations", []))
            rec["annotations"] = sum(per.values())
            for cls, cnt in per.items():
                if cnt:
                    out["classes"][cls] = out["classes"].get(cls, 0) + cnt
        out["splits"][split_dir.name] = rec
    return out


FLAG_ORDER = {"arbitrage": 0, "todo": 1, "attention": 2, "source": 3, "raw": 4}


def zone_flags(zone: dict) -> list[str]:
    flags = set()
    for n in zone["notes"]:
        if n["type"] in ("arbitrage", "todo", "attention"):
            flags.add(n["type"])
    src = (zone["manifest"] or {}).get("source") or {}
    if not src.get("contact") or not src.get("licence"):
        flags.add("source")
    if zone["raw"] is None:
        flags.add("raw")
    return sorted(flags, key=lambda f: FLAG_ORDER.get(f, 9))


def build(root: Path) -> dict:
    data = {"generated": str(date.today()), "root": str(root),
            "zones": [], "a_trier": [], "archives": [], "totals": {},
            "actions": [], "actions_source": {"contact": [], "licence": []}}

    for region_dir in sorted(p for p in root.iterdir()
                             if p.is_dir() and not p.name.startswith("_")):
        for zone_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            zone = {"zone_id": zone_dir.name, "region": region_dir.name,
                    "manifest": None, "raw": None, "datasets": [], "vecteurs": None}
            mf = zone_dir / "manifest.yaml"
            raw_notes = []
            if mf.exists():
                zone["manifest"] = yaml.safe_load(mf.read_text(encoding="utf-8"))
                raw_notes = zone["manifest"].pop("notes", []) or []
            raw = zone_dir / "raw"
            if raw.is_dir():
                zone["raw"] = dir_stats(raw)
                zone["raw"]["subdirs"] = sorted(p.name for p in raw.iterdir() if p.is_dir())
            vec = zone_dir / "transformed" / "vecteurs"
            if vec.is_dir():
                zone["vecteurs"] = dir_stats(vec)
            rf = zone_dir / "transformed" / "roboflow"
            if rf.is_dir():
                for dsdir in sorted(p for p in rf.iterdir() if p.is_dir()):
                    zone["datasets"].append(read_dataset(dsdir))

            zone["notes"] = [parse_note(n) for n in raw_notes]
            zone["stats"] = {
                "images": sum(s["images"] for d in zone["datasets"]
                              for s in d["splits"].values()),
                "annotations": sum(s["annotations"] for d in zone["datasets"]
                                   for s in d["splits"].values()),
                "n_datasets": len(zone["datasets"]),
                "n_classes": len({c for d in zone["datasets"] for c in d["classes"]}),
            }
            zone["flags"] = zone_flags(zone)
            zone["search"] = norm(" ".join(
                [zone["zone_id"], zone["region"],
                 str((zone["manifest"] or {}).get("departement", ""))]
                + [d["name"] for d in zone["datasets"]]
                + [c for d in zone["datasets"] for c in d["classes"]]))
            data["zones"].append(zone)

            for n in zone["notes"]:
                if n["type"] in ("arbitrage", "todo", "attention"):
                    data["actions"].append({"zone_id": zone["zone_id"],
                                            "region": zone["region"],
                                            "type": n["type"], "texte": n["texte"]})
            src = (zone["manifest"] or {}).get("source") or {}
            if not src.get("contact"):
                data["actions_source"]["contact"].append(zone["zone_id"])
            if not src.get("licence"):
                data["actions_source"]["licence"].append(zone["zone_id"])

    data["actions"].sort(key=lambda a: (FLAG_ORDER.get(a["type"], 9), a["zone_id"]))

    atrier = root / "_a_trier"
    if atrier.is_dir():
        for dsdir in sorted(p for p in atrier.iterdir() if p.is_dir()):
            data["a_trier"].append(read_dataset(dsdir))

    modeles = root / "modeles.yaml"
    if modeles.exists():
        data["modeles"] = (yaml.safe_load(modeles.read_text(encoding="utf-8")) or {}).get("modeles", [])
    else:
        data["modeles"] = []

    arch = root / "_archives_roboflow"
    if arch.is_dir():
        for f in sorted(arch.iterdir()):
            if f.is_file() and f.suffix.lower() == ".zip":
                data["archives"].append({"name": f.name, "size": f.stat().st_size})

    tz = data["totals"]
    tz["zones"] = len(data["zones"])
    tz["regions"] = len({z["region"] for z in data["zones"]})
    tz["images"] = sum(z["stats"]["images"] for z in data["zones"])
    tz["annotations"] = sum(z["stats"]["annotations"] for z in data["zones"])
    tz["a_trier_images"] = sum(s["images"] for d in data["a_trier"]
                               for s in d["splits"].values())
    tz["raw_files"] = sum((z["raw"] or {}).get("n_files", 0) for z in data["zones"])
    tz["raw_size"] = sum((z["raw"] or {}).get("size", 0) for z in data["zones"])
    tz["actions"] = (len(data["actions"])
                     + (1 if data["actions_source"]["contact"] else 0)
                     + (1 if data["actions_source"]["licence"] else 0)
                     + (1 if tz["a_trier_images"] else 0))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("-o", "--out", help="défaut : <racine>/index.html")
    a = ap.parse_args()
    root = Path(a.root)
    data = build(root)
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__V2_JSON__", payload)
    out = Path(a.out) if a.out else root / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"index : {out}")
    print(f"  {data['totals']['zones']} zones, {data['totals']['images']:,} images, "
          f"{data['totals']['annotations']:,} annotations, "
          f"{data['totals']['actions']} action(s) en attente")


if __name__ == "__main__":
    main()
