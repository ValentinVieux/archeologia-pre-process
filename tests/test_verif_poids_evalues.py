"""Le garde-fou best_ema/best_total attrape-t-il ce qu'il doit, et RIEN d'autre ?

Il est la porte 1bis BLOQUANTE de /installer-modele-plugin : une porte qui rend CONFORME sans
preuve est pire que pas de porte. On rejoue donc les deux incidents réels (crateres 2026-09-11,
tranchees 2026-09-14) et les cas limites trouvés en contre-expertise le 2026-09-14 : entrée
`modeles` absente, baseline confondue avec le modèle, éval auto-référencée, checkpoint disparu.

Lancement : .venv\\Scripts\\python.exe tests\\test_verif_poids_evalues.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.verif_poids_evalues import (  # noqa: E402
    AUTO_REFERENCE, CONFORME, CONFORME_TAILLE, INDETERMINABLE, SUSPECT, controler)

# Tailles RÉELLES des checkpoints du run tranchees_seg_ld_v1 : elles diffèrent, et c'est ce qui
# rend le contrôle par la taille suffisant pour attraper l'incident.
TAILLE_TOTAL, TAILLE_EMA = 144257681, 144234793


def fabriquer(racine: Path, ident: str, octets_livres: bytes, evals: list[dict]) -> Path:
    """Un dossier de modèle plausible : weights/best.pth + N dossiers evaluation*/."""
    dossier = racine / ident
    (dossier / "weights").mkdir(parents=True)
    (dossier / "weights" / "best.pth").write_bytes(octets_livres)
    for spec in evals:
        ev = dossier / "entrainement" / spec.get("dossier", "evaluation")
        ev.mkdir(parents=True, exist_ok=True)
        (ev / "metriques_eval.json").write_text(
            json.dumps({"schema": "metriques_eval/1", "modeles": spec["modeles"]}), encoding="utf-8")
        if "meta_modeles" in spec:
            (ev / "appariements.json").write_text(
                json.dumps({"_meta": {"modeles": spec["meta_modeles"]}}), encoding="utf-8")
    return dossier


def cas(nom: str, attendu: str, racine: Path, **kwargs) -> None:
    dossier = fabriquer(racine, nom, **kwargs)
    verdict, lignes = controler(dossier)
    assert verdict == attendu, f"{nom} : attendu {attendu}, obtenu {verdict}\n  " + "\n  ".join(lignes)
    print(f"  OK  {nom:38s} -> {verdict}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        r = Path(tmp)
        mesure = r / "_mesures"
        mesure.mkdir()
        # deux checkpoints de tailles distinctes, comme dans un vrai run
        (mesure / "checkpoint_best_total.pth").write_bytes(b"T" * 4096)
        (mesure / "checkpoint_best_ema.pth").write_bytes(b"E" * 2048)
        total, ema = (mesure / "checkpoint_best_total.pth"), (mesure / "checkpoint_best_ema.pth")

        print("--- l'incident réel : on livre best_total, on a mesuré best_ema ---")
        cas("incident_tranchees", SUSPECT, r,
            octets_livres=total.read_bytes(),
            evals=[{"modeles": {"incident_tranchees": {"poids": str(ema)}},
                    "meta_modeles": {"incident_tranchees": {"taille_octets": ema.stat().st_size}}}])

        print("--- le même incident, checkpoint mesuré DISPARU (brouillon purgé) ---")
        cas("incident_sans_fichier", SUSPECT, r,
            octets_livres=total.read_bytes(),
            evals=[{"modeles": {"incident_sans_fichier": {"poids": "D:/purge/checkpoint_best_ema.pth"}},
                    "meta_modeles": {"incident_sans_fichier": {"taille_octets": TAILLE_EMA}}}])

        print("--- le cas sain ---")
        cas("modele_sain", CONFORME, r,
            octets_livres=total.read_bytes(),
            evals=[{"modeles": {"modele_sain": {"poids": str(total)}},
                    "meta_modeles": {"modele_sain": {"taille_octets": total.stat().st_size}}}])

        print("--- cas limites trouvés en contre-expertise ---")
        # `modeles` vide : ne doit JAMAIS rendre CONFORME faute de boucle exécutée
        cas("modeles_vide", INDETERMINABLE, r,
            octets_livres=total.read_bytes(), evals=[{"modeles": {}}])

        # une éval superposée : seule l'entrée du modèle compte, pas la baseline
        cas("avec_baseline", CONFORME, r,
            octets_livres=total.read_bytes(),
            evals=[{"modeles": {"avec_baseline": {"poids": str(total)},
                                "vieille_baseline": {"poids": str(ema)}},
                    "meta_modeles": {"avec_baseline": {"taille_octets": total.stat().st_size},
                                     "vieille_baseline": {"taille_octets": ema.stat().st_size}}}])

        # aucune entrée ne correspond au modèle : on ne conclut pas
        cas("que_des_baselines", INDETERMINABLE, r,
            octets_livres=total.read_bytes(),
            evals=[{"modeles": {"un_autre": {"poids": str(total)}, "encore_un": {"poids": str(ema)}}}])

        # taille concordante mais fichier absent : présomption, pas preuve
        cas("fichier_absent", CONFORME_TAILLE, r,
            octets_livres=b"T" * 4096,
            evals=[{"modeles": {"fichier_absent": {"poids": "D:/purge/checkpoint_best_total.pth"}},
                    "meta_modeles": {"fichier_absent": {"taille_octets": 4096}}}])

        # ni fichier ni taille : rien à comparer
        cas("sans_preuve", INDETERMINABLE, r,
            octets_livres=total.read_bytes(),
            evals=[{"modeles": {"sans_preuve": {"poids": "D:/purge/checkpoint.pth"}}}])

        # champ poids vide
        cas("poids_vide", INDETERMINABLE, r,
            octets_livres=total.read_bytes(), evals=[{"modeles": {"poids_vide": {"poids": ""}}}])

        # le pire verdict l'emporte sur l'ensemble des évaluations du modèle
        cas("une_bonne_une_mauvaise", SUSPECT, r,
            octets_livres=total.read_bytes(),
            evals=[{"dossier": "evaluation",
                    "modeles": {"une_bonne_une_mauvaise": {"poids": str(total)}},
                    "meta_modeles": {"une_bonne_une_mauvaise": {"taille_octets": total.stat().st_size}}},
                   {"dossier": "evaluation_couverture",
                    "modeles": {"une_bonne_une_mauvaise": {"poids": str(ema)}},
                    "meta_modeles": {"une_bonne_une_mauvaise": {"taille_octets": ema.stat().st_size}}}])

        # éval auto-référencée : vraie par construction, ne prouve rien
        auto = fabriquer(r, "auto_reference", octets_livres=total.read_bytes(), evals=[])
        ev = auto / "entrainement" / "evaluation"
        ev.mkdir(parents=True)
        livre = auto / "weights" / "best.pth"
        (ev / "metriques_eval.json").write_text(
            json.dumps({"modeles": {"auto_reference": {"poids": str(livre)}}}), encoding="utf-8")
        (ev / "appariements.json").write_text(
            json.dumps({"_meta": {"modeles": {"auto_reference": {"taille_octets": livre.stat().st_size}}}}),
            encoding="utf-8")
        verdict, lignes = controler(auto)
        assert verdict == AUTO_REFERENCE, f"auto_reference : obtenu {verdict}"
        print(f"  OK  {'auto_reference':38s} -> {verdict}")

        # un comparaison_*/ ne doit PAS être contrôlé : c'est sa raison d'être de mesurer autre chose
        sain = r / "modele_sain"
        comp = sain / "entrainement" / "comparaison_ema_ep79"
        comp.mkdir(parents=True)
        (comp / "metriques_eval.json").write_text(
            json.dumps({"modeles": {"modele_sain": {"poids": str(ema)}}}), encoding="utf-8")
        (comp / "appariements.json").write_text(
            json.dumps({"_meta": {"modeles": {"modele_sain": {"taille_octets": ema.stat().st_size}}}}),
            encoding="utf-8")
        verdict, _ = controler(sain)
        assert verdict == CONFORME, f"comparaison_* pris en compte à tort : {verdict}"
        print(f"  OK  {'comparaison_ignoree':38s} -> {verdict}")

        # weights/best.pth absent
        nu = r / "sans_poids"
        (nu / "weights").mkdir(parents=True)
        shutil.rmtree(nu / "weights")
        (nu / "weights").mkdir(parents=True)
        verdict, _ = controler(nu)
        assert verdict == INDETERMINABLE, verdict
        print(f"  OK  {'sans_best_pth':38s} -> {verdict}")

    print("\nOK — le garde-fou attrape les deux incidents réels (fichier présent ET disparu) "
          "et ne rend CONFORME que sur preuve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
