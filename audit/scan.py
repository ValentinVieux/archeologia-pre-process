"""Audit déterministe d'une livraison de données archéologiques (dossier hétérogène).

Entrée publique : build_audit(dataset_path, taxonomy_dir, row_cap) -> dict (schéma v1).
Aucun appel réseau, aucun LLM, aucun fuzzy matching : la classification sémantique
des noms inconnus est le travail de Claude Code (skill /audit-dataset).
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pyogrio
import yaml

SCHEMA_VERSION = 1
VALUE_CAP = 100  # top values conservées par champ string
CANDIDATE_DISTINCT_CAP = 30  # ponytail: seuil catégoriel, à ajuster sur cas réels

VECTOR_EXTS = {".shp", ".gpkg", ".geojson", ".json", ".gml", ".kml", ".tab", ".dxf"}
RASTER_EXTS = {".tif", ".tiff", ".asc", ".vrt", ".jp2", ".ecw", ".img"}
QGIS_EXTS = {".qgz", ".qgs"}
ARCHIVE_EXTS = {".zip", ".7z", ".rar"}
DOCUMENT_EXTS = {".pdf", ".doc", ".docx", ".odt", ".ods", ".xls", ".xlsx", ".csv", ".txt", ".md"}
# sidecar -> extension du fichier principal attendu à côté (même nom de base)
SIDECAR_MAIN = {
    ".dbf": ".shp", ".shx": ".shp", ".prj": ".shp", ".cpg": ".shp", ".qpj": ".shp",
    ".qix": ".shp", ".sbn": ".shp", ".sbx": ".shp",
    ".gpkg-wal": ".gpkg", ".gpkg-shm": ".gpkg", ".gpkg-journal": ".gpkg",
    ".map": ".tab", ".id": ".tab", ".dat": ".tab", ".ind": ".tab",
}


def normalize(s: str) -> str:
    """'Chemins-Creux ' -> 'chemins_creux'. Même convention que le contrat du plugin."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ---------------------------------------------------------------- taxonomie

def load_taxonomy(taxonomy_dir: Path) -> dict:
    """Index de matching depuis entities.yaml + aliases.yaml (absents/vides tolérés)."""

    def _load(name: str) -> dict:
        p = Path(taxonomy_dir) / name
        try:
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            return {}

    entities = _load("entities.yaml").get("entities") or []
    al = _load("aliases.yaml")
    tax = {
        "exact_alias": {}, "norm_alias": {},
        "exact_ignored": set(), "norm_ignored": set(),
        "norm_entity": {}, "norm_label": {}, "norm_roboflow": {},
        "entities": entities,
    }
    for e in entities:
        eid = e.get("id", "")
        tax["norm_entity"][normalize(eid)] = eid
        if e.get("label_fr"):
            tax["norm_label"].setdefault(normalize(e["label_fr"]), eid)
        for rc in e.get("roboflow_classes") or []:
            tax["norm_roboflow"].setdefault(normalize(rc), eid)
    for a in al.get("aliases") or []:
        tax["exact_alias"].setdefault(a["raw"], a["entity_id"])
        tax["norm_alias"].setdefault(normalize(a["raw"]), a["entity_id"])
    for i in al.get("ignored") or []:
        tax["exact_ignored"].add(i["raw"])
        tax["norm_ignored"].add(normalize(i["raw"]))
    return tax


def _match(raw: str, norm: str, tax: dict) -> dict | None:
    if raw in tax["exact_alias"]:
        return {"status": "known", "entity_id": tax["exact_alias"][raw], "via": "exact", "matched_on": "alias"}
    if raw in tax["exact_ignored"]:
        return {"status": "ignored", "entity_id": None, "via": "exact", "matched_on": "ignored"}
    if norm in tax["norm_alias"]:
        return {"status": "known", "entity_id": tax["norm_alias"][norm], "via": "normalized", "matched_on": "alias"}
    if norm in tax["norm_ignored"]:
        return {"status": "ignored", "entity_id": None, "via": "normalized", "matched_on": "ignored"}
    for key, on in (("norm_entity", "entity_id"), ("norm_label", "label_fr"), ("norm_roboflow", "roboflow_class")):
        if norm in tax[key]:
            return {"status": "known", "entity_id": tax[key][norm], "via": "normalized", "matched_on": on}
    return None


# ---------------------------------------------------------------- inventaire

def _categorize(path: Path, siblings: set[str]) -> tuple[str, str | None]:
    """-> (categorie, anomalie|None). `siblings` = noms lowercase du même dossier."""
    name = path.name.lower()
    if path.is_dir():  # seul cas : .gdb
        return "vector", None
    if name.endswith(".shp.xml"):
        return ("sidecar", None) if name[:-4] in siblings else ("other", "lone_sidecar")
    ext = _ext(name)
    if ext in SIDECAR_MAIN:
        main = name[: -len(ext)] + SIDECAR_MAIN[ext]
        return ("sidecar", None) if main in siblings else ("other", "lone_sidecar")
    if ext in VECTOR_EXTS:
        return "vector", None
    if ext in RASTER_EXTS:
        return "raster", None
    if ext in QGIS_EXTS:
        return "qgis_project", None
    if ext in ARCHIVE_EXTS:
        return "archive", "archive_not_scanned"
    if ext in DOCUMENT_EXTS:
        return "document", None
    return "other", "unrecognized_extension"


def _ext(name: str) -> str:
    """Extension lowercase, en gérant les suffixes composés type .gpkg-wal."""
    for composed in SIDECAR_MAIN:
        if "-" in composed and name.endswith(composed):
            return composed
    i = name.rfind(".")
    return name[i:] if i >= 0 else ""


def _walk(root: Path):
    """Fichiers + dossiers .gdb, ordre déterministe, sans suivre les liens."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for d in list(dirnames):
            if d.lower().endswith(".gdb"):
                dirnames.remove(d)
                yield Path(dirpath) / d
        for f in sorted(filenames):
            yield Path(dirpath) / f


def _size(path: Path) -> int:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size


# ---------------------------------------------------------------- lecture vecteur

def _crs_record(raw, anomalies, rel, layer, spatial=True):
    from pyproj import CRS

    if raw in (None, ""):
        if not spatial:  # table attributaire (GPKG/GDB) : aucun CRS attendu
            return {"raw": None, "epsg": None, "missing": False}
        anomalies.append({"kind": "missing_crs", "file": rel, "layer": layer, "detail": "CRS absent (.prj manquant ?)"})
        return {"raw": None, "epsg": None, "missing": True}
    epsg = None
    try:
        epsg = CRS.from_user_input(raw).to_epsg()
    except Exception:
        pass
    if epsg and epsg != 2154:
        anomalies.append({"kind": "non_reference_crs", "file": rel, "layer": layer,
                          "detail": f"EPSG:{epsg} (référence : EPSG:2154 Lambert-93)"})
    return {"raw": str(raw)[:300], "epsg": epsg, "missing": False}


def _read_layer(path: Path, rel: str, layer_name: str, declared_geom, row_cap: int,
                shp_encoding: str | None, audit: dict) -> dict:
    import geopandas as gpd

    rec = {
        "file": rel, "layer": layer_name, "driver": None,
        "crs": {"raw": None, "epsg": None, "missing": True},
        "declared_geometry_type": declared_geom,
        "feature_count": None, "bbox": None, "encoding_used": shp_encoding,
        "sampled": None, "fields": [], "errors": [],
    }
    anomalies = audit["anomalies"]

    kw = {"encoding": shp_encoding} if shp_encoding else {}
    try:
        info = pyogrio.read_info(path, layer=layer_name, **kw)
    except Exception as exc:
        rec["errors"].append({"stage": "read_info", "error": f"{type(exc).__name__}: {exc}"})
        return rec
    rec["driver"] = info.get("driver")
    rec["crs"] = _crs_record(info.get("crs"), anomalies, rel, layer_name,
                             spatial=declared_geom is not None)
    n = int(info.get("features", -1))
    rec["feature_count"] = n if n >= 0 else None

    try:
        _, bounds = pyogrio.read_bounds(path, layer=layer_name)
        if bounds.size and not np.isnan(bounds).all():  # tout-NaN : nanmin warnerait
            bbox = [float(np.nanmin(bounds[0])), float(np.nanmin(bounds[1])),
                    float(np.nanmax(bounds[2])), float(np.nanmax(bounds[3]))]
            rec["bbox"] = None if any(np.isnan(bbox)) else [round(v, 2) for v in bbox]
    except Exception as exc:
        rec["errors"].append({"stage": "read_bounds", "error": f"{type(exc).__name__}: {exc}"})

    fields = [str(f) for f in (info.get("fields") if info.get("fields") is not None else [])]
    dtypes = [str(d) for d in (info.get("dtypes") if info.get("dtypes") is not None else [])]
    dtypes += ["object"] * (len(fields) - len(dtypes))  # prudence si dtypes absent
    string_cols = [f for f, d in zip(fields, dtypes) if d == "object"]

    df, first_err = None, None
    for enc in ([shp_encoding, "latin-1"] if shp_encoding else [None, "latin-1"]):
        try:
            # string_cols or fields[:1] : une table non spatiale sans champ string rendrait
            # sinon un DataFrame à 0 colonne (len 0) et un rows_read faux.
            df = pyogrio.read_dataframe(path, layer=layer_name, columns=string_cols or fields[:1],
                                        max_features=row_cap, **({"encoding": enc} if enc else {}))
            rec["encoding_used"] = enc
            break
        except Exception as exc:
            first_err = first_err or {"stage": "read_dataframe", "error": f"{type(exc).__name__}: {exc}"}
    if df is None:
        rec["errors"].append(first_err)
        return rec
    if shp_encoding:
        anomalies.append({"kind": "encoding_fallback", "file": rel, "layer": layer_name,
                          "detail": f"pas de .cpg, lu en {rec['encoding_used']}"})

    rows = len(df)
    if rec["feature_count"] is None and rows < row_cap:
        rec["feature_count"] = rows
    # feature_count encore None => driver sans comptage ET rows == row_cap : lecture tronquée
    partial = rec["feature_count"] is None or rec["feature_count"] > rows
    if rec["feature_count"] == 0:
        anomalies.append({"kind": "empty_layer", "file": rel, "layer": layer_name, "detail": "0 entité"})

    geom_stats = None
    if isinstance(df, gpd.GeoDataFrame) and df._geometry_column_name in df.columns:
        g = df.geometry
        types = {str(k): int(v) for k, v in g.geom_type.dropna().value_counts().items()}
        geom_stats = {
            "empty": int((g.is_empty | g.isna()).sum()),
            "invalid": int((~g.is_valid & g.notna() & ~g.is_empty).sum()),
            "types": dict(sorted(types.items())),
        }
        if len(types) > 1:
            anomalies.append({"kind": "mixed_geometry", "file": rel, "layer": layer_name,
                              "detail": "+".join(sorted(types)) + " dans l'échantillon"})
    rec["sampled"] = {"rows_read": rows, "is_partial": bool(partial), "geometry": geom_stats}

    for fname, dtype in zip(fields, dtypes):
        stats = None
        if fname in string_cols and fname in df.columns:
            s = df[fname].dropna().astype(str).str.strip()
            s = s[s != ""]
            vc = s.value_counts()
            stats = {
                "n_distinct_sampled": int(vc.size),
                "truncated": bool(vc.size > VALUE_CAP),
                "top_values": [[str(v), int(c)] for v, c in vc.head(VALUE_CAP).items()],
            }
        rec["fields"].append({"name": fname, "dtype": dtype, "string_stats": stats})
    return rec


def _read_vector_file(path: Path, rel: str, row_cap: int, audit: dict) -> bool:
    """Lit toutes les couches d'un fichier vecteur. False si un .json n'est pas du GeoJSON."""
    try:
        layers = pyogrio.list_layers(path)
    except Exception as exc:
        if path.suffix.lower() == ".json":
            return False  # JSON quelconque, reclassé 'other' sans erreur
        audit["errors"].append({"file": rel, "layer": None, "stage": "list_layers",
                                "error": f"{type(exc).__name__}: {exc}"})
        return True

    shp_encoding = None
    if path.suffix.lower() == ".shp" and not path.with_suffix(".cpg").exists():
        shp_encoding = "cp1252"  # cas français dominant pour les shapefiles sans .cpg

    for layer_name, declared_geom in sorted((str(l[0]), l[1]) for l in layers):
        rec = _read_layer(path, rel, layer_name, str(declared_geom) if declared_geom else None,
                          row_cap, shp_encoding, audit)
        audit["layers"].append(rec)
    return True


# ---------------------------------------------------------------- projets QGIS

def _read_qgis_project(path: Path, rel: str, audit: dict) -> list[str]:
    rec = {"file": rel, "layer_names": [], "error": None}
    try:
        if path.suffix.lower() == ".qgz":
            with zipfile.ZipFile(path) as z:
                inner = next(n for n in z.namelist() if n.lower().endswith(".qgs"))
                root = ElementTree.fromstring(z.read(inner))
        else:
            root = ElementTree.fromstring(path.read_bytes())
        names: list[str] = []
        for tag in ("layer-tree-group", "layer-tree-layer"):
            for el in root.iter(tag):
                if el.get("name"):
                    names.append(el.get("name"))
        for el in root.iter("layername"):
            if el.text and el.text.strip():
                names.append(el.text.strip())
        rec["layer_names"] = sorted(set(names))
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        audit["errors"].append({"file": rel, "layer": None, "stage": "qgis_project",
                                "error": rec["error"]})
    audit["qgis_projects"].append(rec)
    return rec["layer_names"]


# ---------------------------------------------------------------- candidats

class _Candidates:
    def __init__(self, tax: dict):
        self.tax = tax
        self.by_raw: dict[str, dict] = {}

    def add(self, raw: str, kind: str, file: str, layer=None, field=None, count: int = 1):
        raw = str(raw).strip()
        if not raw:
            return
        c = self.by_raw.setdefault(raw, {"normalized": normalize(raw), "sources": {}})
        key = (kind, file, layer, field)
        c["sources"][key] = c["sources"].get(key, 0) + count

    def emit(self) -> list[dict]:
        out = []
        for raw, c in self.by_raw.items():
            sources = [{"kind": k, "file": f, "layer": l, "field": fd, "count": n}
                       for (k, f, l, fd), n in sorted(c["sources"].items(),
                                                      key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "", kv[0][3] or ""))]
            total = sum(s["count"] for s in sources)
            out.append({"raw": raw, "normalized": c["normalized"], "sources": sources,
                        "total_count": total, "match": _match(raw, c["normalized"], self.tax)})
        out.sort(key=lambda c: (-c["total_count"], c["raw"]))
        return out


# ---------------------------------------------------------------- audit

def build_audit(dataset_path: Path, taxonomy_dir: Path, row_cap: int = 50_000) -> dict:
    dataset_path = Path(dataset_path)
    tax = load_taxonomy(Path(taxonomy_dir))
    audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),  # seul champ non déterministe
        "tool": "pre-process-data/audit",
        "dataset": {"name": dataset_path.name, "path": str(dataset_path), "n_files": 0, "total_size_bytes": 0},
        "settings": {"row_cap": row_cap, "value_cap": VALUE_CAP, "candidate_distinct_cap": CANDIDATE_DISTINCT_CAP},
        "files": [], "layers": [], "qgis_projects": [],
        "crs_census": {}, "geometry_census": {},
        "name_candidates": [], "anomalies": [], "errors": [],
    }
    cands = _Candidates(tax)

    entries = list(_walk(dataset_path))
    siblings_by_dir: dict[Path, set[str]] = {}
    for p in entries:
        siblings_by_dir.setdefault(p.parent, set()).add(p.name.lower())

    for p in entries:
        rel = p.relative_to(dataset_path).as_posix()
        category, anomaly = _categorize(p, siblings_by_dir[p.parent])
        try:
            size = _size(p)
        except OSError:
            size = 0
        n_layers_before = len(audit["layers"])

        if category == "vector":
            if not _read_vector_file(p, rel, row_cap, audit):
                category, anomaly = "other", "unrecognized_extension"  # .json non GeoJSON
            else:
                stem = p.name[: -len(_ext(p.name.lower()))] if _ext(p.name.lower()) else p.name
                cands.add(stem, "filename", rel)
                for rec in audit["layers"][n_layers_before:]:
                    cands.add(rec["layer"], "layername", rel, layer=rec["layer"])
                    for fld in rec["fields"]:
                        st = fld["string_stats"]
                        if st and 0 < st["n_distinct_sampled"] <= CANDIDATE_DISTINCT_CAP:
                            for value, count in st["top_values"]:
                                cands.add(value, "field_value", rel, layer=rec["layer"],
                                          field=fld["name"], count=count)
        elif category == "qgis_project":
            cands.add(p.stem, "filename", rel)
            for name in _read_qgis_project(p, rel, audit):
                cands.add(name, "qgis_project", rel)
        if anomaly:
            audit["anomalies"].append({"kind": anomaly, "file": rel, "layer": None, "detail": None})

        audit["files"].append({"path": rel, "category": category,
                               "extension": _ext(p.name.lower()) if not p.is_dir() else ".gdb",
                               "size_bytes": size})

    audit["files"].sort(key=lambda f: f["path"])
    audit["layers"].sort(key=lambda l: (l["file"], l["layer"]))
    audit["qgis_projects"].sort(key=lambda q: q["file"])
    audit["dataset"]["n_files"] = len(audit["files"])
    audit["dataset"]["total_size_bytes"] = sum(f["size_bytes"] for f in audit["files"])

    crs_census: Counter = Counter()
    geom_census: Counter = Counter()
    for rec in audit["layers"]:
        crs = rec["crs"]
        if crs["raw"] is None and not crs["missing"]:
            continue  # table non spatiale : hors census CRS
        crs_census[f"EPSG:{crs['epsg']}" if crs["epsg"] else ("missing" if crs["missing"] else "non_epsg")] += 1
        g = (rec["sampled"] or {}).get("geometry")
        for t in (g or {}).get("types", {}):
            geom_census[t] += 1
    audit["crs_census"] = dict(sorted(crs_census.items()))
    audit["geometry_census"] = dict(sorted(geom_census.items()))
    audit["name_candidates"] = cands.emit()
    audit["anomalies"].sort(key=lambda a: (a["kind"], a["file"], a["layer"] or ""))
    audit["errors"].sort(key=lambda e: (e["file"], e["stage"]))
    return audit


def dump_audit(audit: dict) -> str:
    return json.dumps(audit, ensure_ascii=False, indent=2)
