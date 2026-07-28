---
name: recalage-zone
description: >
  Recaler les vecteurs linéaires d'une zone sur le relief (méthode B) avec revue
  humaine dans l'app locale et calibration de l'algo par les corrections :
  config de zone, run + vérification, revue app, boucle d'amélioration
  quantifiée, application des décisions, dépôt Drive. Utiliser quand
  l'utilisateur dit « recale la zone X », « on recale X », « corrige les
  offsets de X », « lance la revue de recalage sur X ».
argument-hint: <zone, ex. ile_de_france/78_rambouillet>
---

# Recaler une zone (recalage → revue humaine → calibration → application → dépôt)

Interagir en français. Commandes exactes : CLAUDE.md § Commands. Spec de
référence : `docs/superpowers/specs/2026-07-28-recalage-vecteurs-design.md`
(sections « Retour de la session de revue 1/2 » = calibration Haye). UNE zone
à la fois ; pas de zone suivante tant que la revue en cours n'est pas soldée.

## Étape 1 — Prérequis et config

- GPKG d'entités vérifié (`training/vecteurs`, COPIE locale) + VRT d'indice LD
  0,5 m/px **local** (D: accepté, G: refusé par les outils). Manquant → le
  signaler, c'est un préalable.
- `configs/recalage_<zone>.yaml`. **Polarités LD actées** (calibrées sur 956
  décisions humaines à Haye) : talus/parcellaire = `clair`, fosse/chemin_creux
  = `sombre`, `talus_fosse` fusionné = **`sombre` imposé** (le tracé suit le
  fossé). `poids_distance: 2.0`. Les couches non entraînées ne sont PAS
  recalées (copiées verbatim à l'application). Nouveau type de couche →
  demander la polarité à l'utilisateur ou mesurer (jamais deviner).

## Étape 2 — Run + boucle de vérification

- `recaler_lignes` puis `verif_recalage` : verdict CONFORME obligatoire.
- Taux a_revoir attendu **10-15 %**. Nettement au-dessus → DIAGNOSTIQUER avant
  de donner la main : quel critère déclenche ? le gain de contraste par
  tranche du critère est-il plat (= bruit de terrain, pas un défaut) ?
  Seuils par zone via `seuils_statut:` en config — toujours justifiés par une
  MESURE, documentée en commentaire (ex. Fontainebleau : relief gréseux,
  résidu 2,0).

## Étape 3 — Revue humaine + calibration par les corrections

- `python -m tools.review_recalage <gpkg_recale> <vrt>` → l'utilisateur revoit
  (décisions YAML immédiates, reprise libre, l'app n'écrit JAMAIS le GPKG).
- Après la première salve (~100+ décisions) : `analyse_corrections` + analyses
  ciblées sur l'hypothèse dominante. **Les géométries éditées par l'humain
  sont la vérité terrain** : tout correctif d'algo suit le cycle
  TDD sur le jouet → re-run → quantifier `d(édité, recalé)` avant/après →
  contrôle de NON-RÉGRESSION sur les lignes acceptées → rejeter tout réglage
  qui sur-apprend (dérive des validées ; n de calibration petit).
- Itérer l'algo TÔT dans la revue puis FIGER les paramètres de la zone. Un
  re-run déplace les référents des décisions « recale » : mesurer et annoncer
  la dérive médiane. Chaque adoption = commit avec les chiffres.

## Étape 4 — Application et dépôt

- `appliquer_decisions` puis `verif_application` (CONFORME) — le GPKG final
  garde `geom_origine` et `decision_humaine` sur chaque ligne.
- Dépôt (staging + robocopy) : `training/vecteurs/<zone>_entites_l93_v2.gpkg`
  + `recalage_decisions_<zone>.yaml` (provenance) + rapport ; le GPKG v1 reste
  en place. Manifest : note `DÉCISION:` chiffrée + `TODO:` re-slice buffer 7 m.
  Régénérer l'index v2.

## Garde-fous

- **JAMAIS d'upload Roboflow zone par zone** : remplacement groupé une fois
  TOUTES les zones recalées (décision utilisateur 2026-07-28), re-slice 7 m
  d'abord.
- Vérifications de l'app en Edge headless (playwright, `channel="msedge"`) ;
  tester les flux de décision sur une INSTANCE JETABLE (`--decisions` vers un
  YAML temporaire), jamais sur le fichier de l'utilisateur.
- Boucle de vérification non négociable à chaque étape ; sorties locales non
  commitées ; configs et outils, si.
