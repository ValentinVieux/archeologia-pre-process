# -*- coding: utf-8 -*-
"""Génère l'index HTML auto-suffisant de data_regions_v2.

Usage :
    .venv\\Scripts\\python.exe tools\\build_v2_index.py <racine_data_regions_v2> [-o index.html]

Scanne la racine (manifest.yaml des zones, upload_manifest.yaml des datasets,
_annotations.coco.json, raw/, _a_trier/, _archives_roboflow/) et écrit
<racine>/index.html : page statique, CSS/JS inline, ouvrable en file://.
À relancer après chaque évolution du dossier.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from datetime import date
from pathlib import Path

import yaml

TEMPLATE = Path(__file__).with_name("v2_index_template.html")
IGNORE = {"desktop.ini", "thumbs.db", "index.html"}


def dir_stats(root: Path) -> dict:
    n, size, exts = 0, 0, collections.Counter()
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if f.lower() in IGNORE:
                continue
            n += 1
            ext = os.path.splitext(f)[1].lower() or "(sans ext)"
            exts[ext] += 1
            try:
                size += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return {"n_files": n, "size": size, "exts": dict(exts.most_common(8))}


def read_dataset(dsdir: Path) -> dict:
    """Un dossier transformed/roboflow/<dataset> ou _a_trier/<dataset>."""
    out = {"name": dsdir.name, "splits": {}, "classes": {}, "manifest": None}
    um = dsdir / "upload_manifest.yaml"
    if um.exists():
        m = yaml.safe_load(um.read_text(encoding="utf-8"))
        out["manifest"] = {k: m.get(k) for k in
                           ("source_export", "dispatched", "zone", "region", "tags",
                            "attribution_methods")}
    for split_dir in sorted(p for p in dsdir.iterdir() if p.is_dir()):
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
                if cls in cats.values() and per[cls]:
                    out["classes"][cls] = out["classes"].get(cls, 0) + cnt
        out["splits"][split_dir.name] = rec
    return out


def build(root: Path) -> dict:
    data = {"generated": str(date.today()), "root": str(root),
            "zones": [], "a_trier": [], "archives": [], "totals": {}}

    for region_dir in sorted(p for p in root.iterdir()
                             if p.is_dir() and not p.name.startswith("_")):
        for zone_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            zone = {"zone_id": zone_dir.name, "region": region_dir.name,
                    "manifest": None, "raw": None, "datasets": [], "vecteurs": None}
            mf = zone_dir / "manifest.yaml"
            if mf.exists():
                zone["manifest"] = yaml.safe_load(mf.read_text(encoding="utf-8"))
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
            data["zones"].append(zone)

    atrier = root / "_a_trier"
    if atrier.is_dir():
        for dsdir in sorted(p for p in atrier.iterdir() if p.is_dir()):
            data["a_trier"].append(read_dataset(dsdir))

    arch = root / "_archives_roboflow"
    if arch.is_dir():
        for f in sorted(arch.iterdir()):
            if f.is_file() and f.suffix.lower() == ".zip":
                data["archives"].append({"name": f.name, "size": f.stat().st_size})

    tz = data["totals"]
    tz["zones"] = len(data["zones"])
    tz["regions"] = len({z["region"] for z in data["zones"]})
    tz["images"] = sum(s["images"] for z in data["zones"] for d in z["datasets"]
                       for s in d["splits"].values())
    tz["annotations"] = sum(s["annotations"] for z in data["zones"] for d in z["datasets"]
                            for s in d["splits"].values())
    tz["a_trier_images"] = sum(s["images"] for d in data["a_trier"]
                               for s in d["splits"].values())
    tz["raw_files"] = sum((z["raw"] or {}).get("n_files", 0) for z in data["zones"])
    tz["raw_size"] = sum((z["raw"] or {}).get("size", 0) for z in data["zones"])
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
    print(f"  {data['totals']['zones']} zones, {data['totals']['images']:,} images annotées, "
          f"{data['totals']['annotations']:,} annotations, quarantaine {data['totals']['a_trier_images']:,}")


if __name__ == "__main__":
    main()
