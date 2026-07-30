"""Bouton « Valeurs par défaut du modèle » — logique testée sans Qt.

Le code UI du plugin exige QGIS (qgis.PyQt), indisponible ici : on ne peut pas cliquer
le bouton. Mais le bouton ne fait que trois choses, et ce sont elles qui peuvent casser :

  1. vider les surcharges pour que les cartes retombent sur les défauts du modèle ;
  2. ne PAS recréer ces surcharges pendant le rafraîchissement qui suit (les spinbox
     reprennent leurs valeurs, et chaque setValue déclenche valueChanged) ;
  3. rester inerte quand il n'y a rien à effacer.

On reproduit ici la mécanique exacte des deux fichiers concernés — le garde `_loading`
de EntityCard et la retombée `override or défaut` — avec des doublures, et on vérifie
ces trois points. Si quelqu'un retire le garde de la carte, le test 2 tombe.

    python tests\\test_reset_defauts_ui.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PLUGIN = Path(
    r"C:\Users\valen\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\archeologia-pipeline"
)
STEP = PLUGIN / "src" / "ui" / "steps" / "step_3_detection.py"
CARD = PLUGIN / "src" / "ui" / "widgets" / "entity_card.py"

DEFAUT_CONF, DEFAUT_AIRE = 0.25, 200.0


class SpinDouble:
    """Doublure de NoWheelDoubleSpinBox : setValue déclenche le signal, comme Qt."""

    def __init__(self, on_change):
        self.value_ = 0.0
        self._on_change = on_change

    def setValue(self, v):
        self.value_ = float(v)
        self._on_change()


class CarteDouble:
    """Reproduit EntityCard : garde `_loading` autour des setValue (entity_card.py:373)."""

    def __init__(self, eid, page):
        self.eid, self.page, self._loading = eid, page, False
        self.conf = SpinDouble(self._emit)
        self.aire = SpinDouble(self._emit)

    def _emit(self):
        if not self._loading:                      # entity_card.py:421
            self.page.on_thresholds_changed(self.eid, self.conf.value_, self.aire.value_)

    def update_state(self, conf_override, area_override):
        self._loading = True
        try:
            self.conf.setValue(conf_override if conf_override is not None else DEFAUT_CONF)
            self.aire.setValue(area_override if area_override is not None else DEFAUT_AIRE)
        finally:
            self._loading = False


class PageDouble:
    """Reproduit Step3DetectionPage pour les seules parties concernées."""

    def __init__(self, eids):
        self.entity_thresholds: dict = {}
        self.entity_cluster_params: dict = {}
        self.cartes = {e: CarteDouble(e, self) for e in eids}
        self.n_changed = 0
        self.reset_actif = False
        self.readonly = False
        self.advanced = True

    def on_thresholds_changed(self, eid, conf, aire):
        self.entity_thresholds[eid] = {"confidence_threshold": conf, "min_area_m2": aire}

    def refresh(self):
        for eid, c in self.cartes.items():
            ov = self.entity_thresholds.get(eid, {})
            c.update_state(ov.get("confidence_threshold"), ov.get("min_area_m2"))
        self.reset_actif = (not self.readonly
                            and bool(self.entity_thresholds or self.entity_cluster_params))

    def on_reset_defaults(self):
        if not (self.entity_thresholds or self.entity_cluster_params):
            return
        self.entity_thresholds.clear()
        self.entity_cluster_params.clear()
        self.refresh()
        self.n_changed += 1


STEP2 = PLUGIN / "src" / "ui" / "steps" / "step_2_indices.py"


def verifier_coherence() -> list[str]:
    """Les deux remises a defaut du plugin doivent se comporter pareil.

    L'utilisateur a demande explicitement cette coherence. Sans test, elle derive au
    premier changement d'un seul des deux ecrans.
    """
    err = []
    s3, s2 = STEP.read_text(encoding="utf-8"), STEP2.read_text(encoding="utf-8")
    for nom, src in (("step_3_detection", s3), ("step_2_indices", s2)):
        if "↺" not in src:
            err.append(f"{nom} : le bouton n'a plus le prefixe « ↺ » commun")
        if 'setObjectName("GhostButton")' not in src:
            err.append(f"{nom} : le bouton n'est plus en style GhostButton")
        if "PointingHandCursor" not in src:
            err.append(f"{nom} : pas de curseur main au survol")
        if "show_toast" not in src:
            err.append(f"{nom} : pas de confirmation par Toast")
    # La confirmation doit DIRE ce qui a change, pas afficher un message fige.
    for nom, src, jeton in (("step_3_detection", s3, "n_seuils"),
                            ("step_2_indices", s2, "modifies")):
        if jeton not in src:
            err.append(f"{nom} : la confirmation ne compte pas ce qui a ete efface "
                       f"(jeton {jeton!r} absent) — un message generique laisserait "
                       f"douter que l'action ait eu un effet")
    return err


def verifier_source() -> list[str]:
    """La doublure ne vaut que si le vrai code fait bien ce qu'elle imite."""
    err = []
    step, card = STEP.read_text(encoding="utf-8"), CARD.read_text(encoding="utf-8")
    if "_on_reset_defaults" not in step:
        err.append("step_3_detection.py : _on_reset_defaults absent")
    if "self._entity_thresholds.clear()" not in step:
        err.append("step_3_detection.py : le reset ne vide pas _entity_thresholds")
    if "self._entity_cluster_params.clear()" not in step:
        err.append("step_3_detection.py : le reset ne vide pas _entity_cluster_params")
    if "self._reset_btn.setEnabled" not in step:
        err.append("step_3_detection.py : l'etat actif du bouton n'est jamais calcule")
    if "not self._readonly" not in step:
        err.append("step_3_detection.py : le bouton n'est pas desactive en lecture seule")
    # Le garde de la carte est ce qui evite la boucle de signaux : il doit rester.
    bloc = re.search(r"if show_adv:\s*\n\s*self\._loading = True", card)
    if not bloc:
        err.append("entity_card.py : le garde _loading autour des setValue a disparu — "
                   "le reset recreerait aussitot les surcharges qu'il efface")
    return err


def main() -> int:
    err = verifier_source() + verifier_coherence()
    if err:
        print("Le code source ne correspond plus a ce que le test reproduit :")
        for e in err:
            print("  -", e)
        return 1
    print("source conforme a la doublure (5 points de contact verifies)")
    print("coherence des DEUX remises a defaut : prefixe, style, curseur, toast, "
          "message chiffre\n")

    eids = ["parcellaire", "talus", "fosse", "talus_fosse", "chemin_creux"]
    page = PageDouble(eids)
    echecs = []

    # 3. inerte quand il n'y a rien a effacer
    page.on_reset_defaults()
    if page.n_changed != 0:
        echecs.append("le bouton a emis `changed` alors qu'il n'y avait rien a effacer")
    print(f"OK   sans surcharge, le reset est inerte (changed emis : {page.n_changed})")

    # surcharges persistees, telles qu'on les a lues dans last_ui_config.json
    page.entity_thresholds = {
        "chemin_creux": {"confidence_threshold": 0.15, "min_area_m2": 0.0},
        "talus_fosse": {"confidence_threshold": 0.20, "min_area_m2": 0.0},
        "fosse": {"confidence_threshold": 0.20, "min_area_m2": 0.0},
        "parcellaire": {"confidence_threshold": 0.20, "min_area_m2": 0.0},
        "talus": {"confidence_threshold": 0.20, "min_area_m2": 0.0},
    }
    page.refresh()
    if not page.reset_actif:
        echecs.append("le bouton devrait etre actif quand des surcharges existent")
    mini = min(v["confidence_threshold"] for v in page.entity_thresholds.values())
    print(f"OK   avec surcharges, bouton actif ; le run tournerait au min = {mini}")

    # 1 + 2. le reset vide, et le rafraichissement ne recree rien
    page.on_reset_defaults()
    if page.entity_thresholds:
        echecs.append(f"surcharges recreees par le rafraichissement : {page.entity_thresholds}")
    for eid, c in page.cartes.items():
        if c.conf.value_ != DEFAUT_CONF or c.aire.value_ != DEFAUT_AIRE:
            echecs.append(f"{eid} : {c.conf.value_}/{c.aire.value_} au lieu des defauts")
    if page.reset_actif:
        echecs.append("le bouton reste actif apres un reset reussi")
    print(f"OK   apres reset : 0 surcharge, les 5 cartes a {DEFAUT_CONF}/{DEFAUT_AIRE:.0f} m2")
    print(f"OK   bouton redevenu inactif, `changed` emis {page.n_changed} fois")

    # lecture seule
    page.entity_thresholds = {"talus": {"confidence_threshold": 0.1, "min_area_m2": 0.0}}
    page.readonly = True
    page.refresh()
    if page.reset_actif:
        echecs.append("le bouton est actif en lecture seule (pendant un run)")
    print("OK   inactif en lecture seule, meme avec des surcharges")

    if echecs:
        print("\nECHEC :")
        for e in echecs:
            print("  -", e)
        return 1
    print("\nTOUT PASSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
