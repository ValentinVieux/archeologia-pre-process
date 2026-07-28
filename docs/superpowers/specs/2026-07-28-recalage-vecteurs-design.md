# Spec — Recalage des vecteurs linéaires sur le relief + app de validation/édition

Statut : validé sur les décisions utilisateur du 2026-07-28 (méthode B, stack
FastAPI + page vanilla, revue ciblée, éditeur de tracés intégré). Zone pilote :
**54_foret_de_haye** (premier dataset uploadé) — itérations sur la logique à partir
des corrections manuelles faites dans l'app.

## Pourquoi (mesuré, pas supposé)

838 lignes mesurées sur Blois et Saint-Germain (15 profils perpendiculaires ±12 m
par ligne, re-tirage stable à ±0,5 m) : offset absolu médian 1,0-2,5 m selon la
classe, p90 jusqu'à 7,5 m, max 11,5 m ; **16 à 41 % des lignes ont leur signature
réelle hors du masque d'annotation actuel**. Polarité cohérente par classe
(talus/parcellaire = extremum CLAIR sur LD, fossé/chemin creux = SOMBRE). Signal
présent sur 97-100 % des profils (exception : parcellaire Blois sur blend RVT,
15 % muets — LD dédié à générer pour cette zone le moment venu). Scripts et JSON de
mesure : session scratchpad `verifs/mesure_offsets.py` (reproductibles).

## Vue d'ensemble

```
GPKG zone (training/vecteurs) + raster LD (VRT local)
        │
        ▼
tools/recaler_lignes.py  (+ configs/recalage_<zone>.yaml)     [méthode B]
        │   GPKG recalé (géométrie d'origine conservée en colonne)
        │   + statuts auto_ok / a_revoir / sans_signal + scores
        ▼
tools/verif_recalage.py                                        [boucle de vérification]
        ▼
tools/review_recalage/  (FastAPI + page vanilla)               [validation humaine]
        │   décisions -> recalage_decisions_<zone>.yaml (écrites immédiatement)
        │   corrections manuelles (éditeur de tracés) -> géométries éditées
        ▼
tools/analyse_corrections.py            [boucle d'amélioration de la logique]
        ▼
GPKG final validé -> re-slice buffer 7 m -> boucle habituelle -> re-upload (remplacement)
```

## 1. `tools/recaler_lignes.py` — méthode B (profils perpendiculaires régularisés)

Usage : `recaler_lignes.py configs\recalage_<zone>.yaml <gpkg> <raster> [--out D] [--couches a,b]`

Config par zone/couche (YAML) :
```yaml
raster_gsd_attendu: 0.5
couches:
  parcellaire:  {polarite: clair,  fenetre_m: 8.0, pas_m: 2.0, seuil_contraste: 10}
  talus_fosse:  {polarite: auto,   fenetre_m: 8.0, pas_m: 2.0, seuil_contraste: 10}
  # polarite: clair | sombre | auto (auto = choisit par ligne l'extremum dominant,
  # utile pour les couches mixtes type talus_fosse)
lissage: {poids_donnees: 1.0, poids_derivee: 4.0}   # moindres carrés pénalisés
seuil_points_nets: 5        # < 5 points nets -> sans_signal (original conservé)
seuil_ambiguite: 0.7        # second pic >= 70 % de l'amplitude à > 3 m -> point ambigu
```

Algorithme, par couche :
1. **Nœuds d'abord** : recenser les extrémités partagées (tolérance 0,5 m) ; chaque
   nœud est recalé UNE fois (extremum de polarité dans un disque ±fenetre_m) et le
   déplacement s'applique à toutes les lignes incidentes — topologie préservée.
2. **Profils** : ligne densifiée tous les `pas_m` ; à chaque point, profil du raster
   ±fenetre_m le long de la normale (map_coordinates bilinéaire, vectorisé) ;
   extremum à la polarité imposée, position raffinée au sous-pixel (parabole sur
   3 échantillons) ; point « net » si contraste >= seuil, « ambigu » si second pic
   au-delà du seuil d'ambiguïté.
3. **Régularisation** : la série des offsets nets le long de l'abscisse curviligne
   est ajustée par moindres carrés pénalisés (dérivée première — snake 1D contraint
   au déplacement normal) ; les points non nets/ambigus sont interpolés par la
   régularisation, jamais par leur extremum. Extrémités contraintes aux positions
   de l'étape 1.
4. **Score et statut par ligne** : offset médian/max appliqué, contraste moyen,
   % points nets, % ambigus, résidu post-régularisation →
   `auto_ok` | `a_revoir` (attendu ~10-15 %) | `sans_signal` (géométrie d'origine).
   Seuils initiaux à calibrer sur Haye puis ajustés via l'analyse des corrections.

Sorties (locales, puis dépôt Drive après validation) :
- `<zone>_entites_l93_recale.gpkg` : mêmes couches ; colonnes ajoutées :
  `geom_origine` (WKT), `statut_recalage`, `score`, `offset_median_m`,
  `offset_max_m`, `contraste`, `pts_nets_pct`, `ambigus_pct` ;
- `recalage_rapport.yaml` : stats par couche, histogramme des offsets, paramètres
  résolus, seed — reproductible.

Déterminisme : aucune composante aléatoire ; re-run = mêmes sorties.

## 2. `tools/verif_recalage.py` — boucle de vérification (contrôleur indépendant)

- Zéro ligne perdue/ajoutée ; ids stables ; couches identiques.
- Topologie : les nœuds partagés le restent (mêmes groupes d'incidence).
- Déplacements bornés : distance de Hausdorff(origine, recalé) <= fenetre_m + pas_m
  pour 100 % des lignes ; ratio de longueur dans [0,7 ; 1,4].
- `sans_signal` ⇒ géométrie strictement identique à l'origine.
- Statistique de sanité : le contraste moyen au droit du tracé recalé doit être
  >= à celui du tracé d'origine (le recalage doit rapprocher du signal, en moyenne).

## 3. App de validation — `tools/review_recalage/` (FastAPI + page vanilla)

Recette Historydex transposée : serveur local minimal, front vanilla JS/HTML/CSS
sans build, décisions écrites immédiatement dans un fichier local, reprise gratuite
(état recalculé depuis les fichiers), clavier au centre du flux.

- Lancement : `.venv\Scripts\python.exe -m tools.review_recalage <zone> [--port 5175]`
  → sert la page + l'API sur localhost. Entrées : GPKG recalé + raster VRT local +
  décisions existantes.
- **API** : `GET /api/lignes?statut=&couche=&tri=score` (liste + méta),
  `GET /api/crop/<id>` (PNG du raster LD autour de la ligne, marge = fenêtre + 20 %,
  + l'affine px↔L93 en en-tête JSON), `POST /api/decision` (statut + géométrie
  éditée éventuelle — ÉCRITURE IMMÉDIATE dans le YAML), `GET /api/progression`.
- **Vue** : crop LD en niveaux de gris ; superpositions : tracé d'origine (rouge),
  tracé recalé (vert), tracé édité (jaune) ; **bande de 7 m** rendue en continu
  (trait de largeur 14 px à 0,5 m/px, extrémités rondes = exactement la géométrie
  du buffer) en transparence ; panneau des métriques de la ligne ; sidebar filtres
  (statut/couche/zone) + tri « pires d'abord » + compteur X/Y.
- **Clavier** : Entrée = accepter (recalé, ou édité si édition en cours) + suivant ;
  `O` = garder l'original ; `X` = exclure du training ; flèches/j/k = navigation ;
  Ctrl+Z = undo d'édition ; molette = zoom, drag milieu = pan.
- **Éditeur de tracés** (exigence utilisateur — édition fluide) :
  - poignées sur chaque sommet, **drag** au pixel (position monde recalculée via
    l'affine) ;
  - **ajout de sommet** : double-clic sur un segment (insertion au point cliqué) ;
  - **suppression** : sommet sélectionné + Suppr ;
  - **translation de la ligne entière** : drag du corps de la ligne, ou
    Alt+flèches (nudge 1 px, ×10 avec Shift — le geste Historydex) ;
  - la bande de 7 m suit l'édition en direct ; undo/redo pile locale ;
  - toute édition passe le statut à `editee` et sauve la géométrie monde dans la
    décision (l'app ne modifie JAMAIS le GPKG directement).
- **Décisions** : `recalage_decisions_<zone>.yaml` — une entrée par ligne
  `{id, couche, decision: recale|original|editee|exclue, geometrie_editee (WKT,
  si editee), horodatage}` ; append/écrasement par id ; c'est la source de vérité,
  versionnable, consommée par l'étape 5.
- **Périmètre de revue** (décision utilisateur) : `a_revoir` exhaustif +
  échantillon aléatoire seedé de ~100 `auto_ok` (contrôle qualité) ; les filtres
  par défaut de l'app reflètent ce périmètre.

## 4. `tools/analyse_corrections.py` — la boucle d'amélioration de la logique

Compare, pour chaque ligne éditée/rejetée par l'utilisateur, la géométrie corrigée
à ce que la méthode B avait produit : distribution des écarts, typologie des échecs
(mauvaise polarité ? capture du voisin parallèle ? fenêtre trop courte ? lissage
trop fort ?), et propose des ajustements de paramètres par couche. C'est le contrat
« tes corrections améliorent la logique » : chaque session de revue sur Haye nourrit
le paramétrage avant de passer aux zones suivantes.

## 5. Application des décisions et cascade aval

`appliquer_decisions` (mode de recaler_lignes.py ou petit outil) : GPKG final =
recalé, avec les géométries éditées substituées, les `original` restaurées, les
`exclue` retirées (comptées) ; vérification (verif_recalage adapté) ; dépôt Drive
(`training/vecteurs/<zone>_entites_l93_v2.gpkg` + note manifest) ; puis re-slice
**buffer 7 m** → boucle de vérification habituelle → re-upload de remplacement
(purge par tag de zone puis upload complet — parades fantômes acquises).

**Conservation des lignes non recalées (exigence utilisateur 2026-07-28)** : la
géométrie D'ORIGINE de CHAQUE ligne est conservée à toutes les étapes — colonne
`geom_origine` (WKT) dans le GPKG recalé ET dans le GPKG final v2, y compris pour
les lignes éditées à la main (l'origine reste l'origine, jamais écrasée) ; le GPKG
v1 du Drive n'est par ailleurs jamais modifié. En revanche, découpe, datasets et
uploads Roboflow ne consomment QUE la géométrie active (recalée/validée) — les
tracés d'origine ne partent jamais à l'entraînement.

## Zone pilote et itération

1. Recalage de Haye (parcellaire 5926 + talus_fosse 88, LD local déjà présent).
2. Vérification + revue dans l'app (vos corrections).
3. `analyse_corrections` → ajustement des paramètres → éventuel re-run.
4. Une fois la logique stabilisée : les 4 autres zones en série, puis la cascade
   re-slice/re-upload globale.

## Retour de la session de revue 1 (2026-07-28, 135 décisions sur Haye)

Parcellaire 110/128 recalés acceptés tels quels ; talus_fosse 0/7. Deux
mécanismes d'échec, une cause commune (« le plus contrasté gagne ») :

1. **Capture de voisin** (10 parcellaire) : une structure plus forte dans la
   fenêtre ±8 m aspire l'extremum alors que la vraie structure (faible,
   contraste 16-28) est à < 1,5 m de l'annotation.
2. **Dipôle talus/fossé** : le fossébutte = pôle clair + pôle sombre séparés
   de ~2,5 m (100 % des profils) ; l'auto-polarité au contraste total choisit
   parfois le mauvais pôle (jusqu'à 7,65 m d'erreur).

Correctifs actés (calibrés sur les 22 géométries éditées par l'humain =
vérité terrain, avec contrôle de non-régression sur les 110 acceptées) :

- **`poids_distance` (défaut 2,0)** : l'extremum est choisi sur le profil
  pénalisé de `poids × |offset|` (la structure proche gagne sauf si la
  lointaine est bien plus forte) ; la position exacte reste celle du pic brut
  (hill-climb) ; le score de polarité auto est pénalisé pareil.
  Parcellaire : médiane d(édité, recalé) 2,45 → 1,92 m ; poids 3 rejeté
  (dérive > 1 m sur 33/110 lignes validées = sur-apprentissage).
- **talus_fosse Haye : polarité SOMBRE imposée** (le tracé suit le fossé,
  6/7 éditions) — médiane 2,40 → 1,02 m ; fenêtre restée à 8 m (10 m
  capturait des sombres plus lointains).
- **Seuils a_revoir recalibrés** : pts_nets < 30 % / ambigus > 40 % /
  résidu > 1,2 m → ~15 % à revoir (objectif atteint). Limite connue : les
  échecs résiduels (~2-4 m, signal faible où l'humain sait mieux — ex.
  annotation à 13 m du signal) sont indétectables par ces mesures ; couverts
  par la revue + l'échantillon d'auto_ok.

## Retour de la session de revue 2 (2026-07-28, 363 décisions cumulées)

Observation utilisateur confirmée : **les entités proches se recalaient l'une
sur l'autre** — 250 paires quasi-fusionnées (recalés < 1 m pour des origines
> 2,5 m), la pénalité de distance ne suffisant pas quand le voisin plus
contrasté partage la fenêtre. Correctifs :

- **Couloir partagé** (`bornes_laterales`) : la fenêtre de chaque profil est
  tronquée à mi-distance de la ligne voisine la plus proche de chaque côté
  (plancher 2 m) — les annotations se partagent l'espace.
- **Pic local intérieur** : un extremum n'est retenu que s'il est un maximum
  local encadré — un maximum contre une borne (couloir ou fenêtre) est le
  flanc tronqué d'une structure voisine, jamais une cible ; même règle pour
  le second pic du détecteur d'ambiguïté (sinon le flanc du voisin rendait
  tout ambigu).
- Bug exposé par la vérification : le signal conditionnant le recalage d'un
  nœud partagé est désormais testé par PARTIE de multi-ligne.

Bilan (42 éditions parcellaire = vérité terrain) : médiane d(édité, recalé)
1,95 → **1,19 m** ; quasi-fusions 250 → **45** ; a_revoir 887 → 648 (11 %) ;
gain de contraste +6,6. Talus_fosse stable (l'unique « régression » est une
ligne dont le tracé suit le talus — l'exception 1/7 à la convention sombre —
couverte par l'édition manuelle).

## Hors périmètre (plus tard)

Côté entraînement RF-DETR (poids de loss, métriques CCQ/centerline — proposés le
2026-07-28, différés) ; LD dédié pour le parcellaire de Blois ; recalage des
polygones (seules les LIGNES sont recalées en v1).

## Tests

- `tests/test_recaler_lignes.py` (style repo) : raster synthétique avec crête/creux
  sinueux connus + lignes décalées/grossières → le recalage retombe sur la vérité à
  < 0,5 px ; nœuds partagés préservés ; polarité auto ; ligne sans signal intacte ;
  déterminisme.
- App : vérification manuelle sur Haye (c'est son rôle) + endpoint /api/decision
  testé par un petit script (écriture immédiate, reprise).
