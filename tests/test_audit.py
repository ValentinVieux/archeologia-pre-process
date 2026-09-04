"""Auto-test de bout en bout, sans framework : .venv\\Scripts\\python.exe tests\\test_audit.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from audit.report import render_report  # noqa: E402
from audit.scan import build_audit, normalize  # noqa: E402
import make_fixture  # noqa: E402


def main() -> None:
    fixture = make_fixture.make()  # toujours régénérée : rapide et déterministe
    audit = build_audit(fixture, ROOT / "taxonomy")
    audit2 = build_audit(fixture, ROOT / "taxonomy")

    # Déterminisme : seul generated_at diffère
    a = {k: v for k, v in audit.items() if k != "generated_at"}
    b = {k: v for k, v in audit2.items() if k != "generated_at"}
    assert a == b, "audit non déterministe"

    # Le fichier corrompu est capturé en erreur, jamais en crash
    assert any("corrompu" in e["file"] for e in audit["errors"]), audit["errors"]

    layers = {(l["file"], l["layer"]) for l in audit["layers"]}
    assert ("vecteurs/sites.gpkg", "charbonnieres") in layers, layers
    assert ("vecteurs/sites.gpkg", "zones") in layers
    assert ("vecteurs/no_prj.shp", "no_prj") in layers
    assert ("features.geojson", "features") in layers

    # Matching taxonomie (id, label, classe Roboflow), et inconnus laissés inconnus
    cand = {c["raw"]: c for c in audit["name_candidates"]}
    assert cand["charbonnière"]["match"]["entity_id"] == "charbonniere"
    assert cand["tumulus"]["match"]["entity_id"] == "tumulus"
    # split talus/fosse (2026-07-28) : « fossé » distinctement labellisé -> fosse
    # (l'alias Fontainebleau prime sur la classe Roboflow historique de talus_fosse,
    # qui reste réservée aux labels indistincts type fossébutte)
    assert cand["fossé"]["match"]["entity_id"] == "fosse", cand["fossé"]
    # « indéterminé » a été ignoré lors de l'audit data_bretagne_1 (aliases.yaml ignored:)
    assert cand["indéterminé"]["match"]["status"] == "ignored", cand["indéterminé"]
    # 2026-09-03 : label_fr de `four` précisé en « Fours à chaux » (décision utilisateur) ->
    # le nom brut « Fours à chaux » est désormais reconnu par le label (avant : inconnu)
    assert cand["Fours à chaux"]["match"]["entity_id"] == "four", cand["Fours à chaux"]
    assert cand["Fours à chaux"]["match"]["matched_on"] == "label_fr", cand["Fours à chaux"]

    # DBF utf-8 sans .cpg : accents intacts (pas de mojibake cp1252) et matchés
    assert cand["mégalithe"]["match"]["entity_id"] == "megalithe", cand.keys()
    assert cand["éperon barré"]["match"]["entity_id"] == "eperon_barre"
    utf8_layer = next(l for l in audit["layers"] if l["layer"] == "utf8_no_cpg")
    assert utf8_layer["encoding_used"] == "utf-8", utf8_layer["encoding_used"]

    # DBF cp1252 sans .cpg : le fallback joue après l'échec utf-8
    noprj = next(l for l in audit["layers"] if l["layer"] == "no_prj")
    assert noprj["encoding_used"] == "cp1252", noprj["encoding_used"]

    kinds = {an["kind"] for an in audit["anomalies"]}
    for expected in ("missing_crs", "non_reference_crs", "archive_not_scanned",
                     "lone_sidecar", "encoding_fallback", "unrecognized_extension"):
        assert expected in kinds, (expected, kinds)

    zones = next(l for l in audit["layers"] if l["layer"] == "zones")
    assert zones["sampled"]["geometry"]["invalid"] == 1, zones["sampled"]
    pts = next(l for l in audit["layers"] if l["layer"] == "charbonnieres")
    assert pts["sampled"]["geometry"]["empty"] == 1, pts["sampled"]
    assert pts["crs"]["epsg"] == 2154
    assert pts["bbox"] is not None

    # Table non spatiale : lignes bien comptées, pas de fausse alerte CRS ni de partiel
    mesures = next(l for l in audit["layers"] if l["layer"] == "mesures")
    assert mesures["sampled"]["rows_read"] == 2, mesures["sampled"]
    assert mesures["sampled"]["is_partial"] is False
    assert mesures["sampled"]["geometry"] is None
    assert mesures["crs"] == {"raw": None, "epsg": None, "missing": False}
    assert not any(a["kind"] == "missing_crs" and "tables.gpkg" in a["file"]
                   for a in audit["anomalies"])

    qgz = next(q for q in audit["qgis_projects"] if q["file"] == "projet.qgz")
    assert "Charbonnières validées" in qgz["layer_names"]
    assert "Prospection Morvan" in qgz["layer_names"]

    files = {f["path"]: f for f in audit["files"]}
    assert files["vecteurs/corrompu.dbf"]["category"] == "sidecar"
    assert files["orphelin.dbf"]["category"] == "other"
    assert files["data.json"]["category"] == "other"      # JSON non-GeoJSON reclassé
    assert files["dem.tif"]["category"] == "raster"

    # normalize : convention snake_case ASCII du contrat plugin
    assert normalize("Chemins-Creux ") == "chemins_creux"
    assert normalize("Dépréssions  circulaires") == "depressions_circulaires"
    assert normalize("trou d'obus") == "trou_d_obus"

    html = render_report(audit)
    assert "__AUDIT_JSON__" not in html
    assert "charbonnière" in html

    print(f"ALL OK — {len(audit['files'])} fichiers, {len(audit['layers'])} couches, "
          f"{len(audit['name_candidates'])} candidats, {len(audit['anomalies'])} anomalies, "
          f"{len(audit['errors'])} erreurs")


if __name__ == "__main__":
    main()
