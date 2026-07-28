# Plan d'implémentation — Recalage des vecteurs + app de validation/édition

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implémenter la spec `docs/superpowers/specs/2026-07-28-recalage-vecteurs-design.md` :
outil de recalage (méthode B), contrôleur, app FastAPI d'édition/validation, analyse des
corrections — pilote sur Haye.

**Architecture:** `tools/recaler_lignes.py` (fonctions pures testables + CLI),
`tools/verif_recalage.py`, `tools/review_recalage/` (serveur.py + static/), 
`tools/analyse_corrections.py`. Tests façon repo (scripts à asserts).

**Tech Stack:** numpy/scipy (map_coordinates, solve dense par ligne), rasterio,
geopandas/shapely ; fastapi + uvicorn (à ajouter aux requirements) ; front vanilla.

## Global Constraints

- Commandes via `.venv\Scripts\python.exe` ; code FR-technique ; sorties locales non
  commitées ; configs YAML commitées ; jamais d'écriture G: hors staging+robocopy.
- Déterminisme total du recalage (aucun aléa) ; l'app n'écrit JAMAIS le GPKG (décisions
  YAML uniquement, écriture immédiate).
- Boucle de vérification avant toute livraison.

---

### Task 1: Noyau recalage (TDD) — profils, extremum sous-pixel, régularisation, nœuds

**Files:** Create `tools/recaler_lignes.py`, `tests/test_recaler_lignes.py`.

**Interfaces (Produces):**
- `densifier(ligne: LineString, pas_m) -> (pts Nx2, normales Nx2, absc N)` — points
  réguliers + normales unitaires (moyenne des segments adjacents), extrémités incluses.
- `profils(raster_win, affine_win, pts, normales, fenetre_m, pas_echant=0.5) -> (K,N)` —
  échantillonnage bilinéaire (map_coordinates) le long des normales.
- `extremum_profil(profil, polarite, seuil_contraste, seuil_ambiguite) ->
  (offset_m|None, contraste, ambigu: bool)` — argmax/argmin + parabole sous-pixel ;
  None si contraste < seuil.
- `regulariser(offsets: N avec NaN, poids_derivee, ancres: {0: v0, N-1: v1}) -> N` —
  moindres carrés pénalisés (système tridiagonal dense), NaN interpolés, ancres à
  poids fort (1e6).
- `recaler_ligne(ligne, lecteur_raster, params) -> (ligne_recalee, mesures dict)`.
- `noeuds_partages(gdfs: {couche: gdf}, tol=0.5) -> {noeud_id: {(couche,idx,extremite)}}`
  + `recaler_noeud(pos, lecteur, polarite, fenetre) -> pos2`.

- [x] **Step 1:** tests rouges : raster synthétique 1200x800 px (0,5 m/px, EPSG:2154) —
  crête gaussienne CLAIRE le long d'une sinusoïde connue + creux SOMBRE le long d'une
  autre ; lignes de test = vérité décalée de +6 m / bruitée / grossière (3 sommets) ;
  asserts : recalage à < 0,5 px de la vérité (médiane des distances), polarité auto
  correcte, ligne hors signal intacte (statut sans_signal), nœud partagé entre 2 lignes
  → déplacé identiquement, déterminisme (2 runs identiques).
- [x] **Step 2:** implémenter ; vert ; commit `recalage : noyau methode B (TDD)`.

### Task 2: Pipeline CLI + statuts + GPKG + rapport

**Files:** Modify `tools/recaler_lignes.py` ; Create `configs/recalage_haye.yaml`.

- [x] Étapes : charger config/gpkg/raster (refus G:) ; nœuds d'abord ; recaler chaque
  ligne (fenêtre par couche, polarité, ancres aux nœuds) ; statuts auto_ok/a_revoir/
  sans_signal selon seuils (pts_nets_pct < 40 % ou ambigus_pct > 35 % ou résidu élevé
  → a_revoir) ; GPKG sortie avec `geom_origine` WKT + mesures ; rapport YAML ; récap
  stdout `Sorties :`. Test intégration dans test_recaler_lignes (mini GPKG 2 couches).
- [x] Commit `recalage : pipeline CLI + statuts + rapport`.

### Task 3: `tools/verif_recalage.py` (boucle de vérification)

- [x] Contrôles de la spec §2 (comptes, topologie, Hausdorff borné, sans_signal
  intact, contraste moyen amélioré) — pilotés par le rapport + GPKG + sources ;
  test dans test_recaler_lignes (cas conforme + cas volontairement cassé).
- [x] Commit.

### Task 4: Run réel Haye + vérification

- [x] `configs/recalage_haye.yaml` (parcellaire clair, talus_fosse auto, fenêtre 8 m) ;
  run sur copie locale (GPKG build4 + VRT haye_ld) ; verif_recalage ; stats présentées
  (répartition des statuts, offsets appliqués) — PAS de dépôt Drive avant revue app.

### Task 5: App `tools/review_recalage/`

**Files:** Create `tools/review_recalage/__main__.py` (FastAPI), `static/index.html`,
`static/app.js`, `static/style.css` ; requirements += fastapi, uvicorn.

- [x] API : GET /api/lignes (méta + filtres + tri score), GET /api/crop/{couche}/{fid}
  (PNG niveaux de gris via rasterio window + affine json en header), POST /api/decision
  (écriture immédiate YAML), GET /api/progression. Décisions :
  `recalage_decisions_<zone>.yaml`.
- [x] Front : layout 3 zones (sidebar liste filtrée / canvas central / panneau
  métriques) ; superpositions origine(rouge)/recalé(vert)/édité(jaune) + bande 7 m
  (lineWidth=14 px, caps ronds, alpha) ; clavier (Entrée valider+suivant, O original,
  X exclure, flèches/j/k, Ctrl+Z, molette zoom, drag-milieu pan) ; ÉDITEUR : poignées
  drag sommets, double-clic insertion, Suppr suppression, drag corps = translation,
  Alt+flèches nudge (×10 Shift) ; conversions px↔monde via l'affine du crop.
- [x] Périmètre par défaut : statut=a_revoir + échantillon seedé 100 auto_ok (fourni
  par l'API, flag `echantillon: true`).
- [x] Test : script `tests/test_review_api.py` (TestClient FastAPI : lignes, crop,
  décision écrite immédiatement, reprise).
- [x] Commit `recalage : app de validation/edition (FastAPI + vanilla)`.

### Task 6: `tools/analyse_corrections.py`

- [x] Pour chaque décision `editee`/`original`/`exclue` : distance recalé↔édité,
  typologie simple (offset résiduel constant → fenêtre/lissage ; zigzag → voisin
  parallèle ; etc. heuristiques v1) ; sortie rapport YAML + suggestions de paramètres.
  Test minimal sur décisions fabriquées.
- [x] Commit.

## Self-review

Spec couverte : méthode B ✓(T1-2), nœuds/topologie ✓(T1), statuts/scores ✓(T2),
verif ✓(T3), pilote Haye ✓(T4), app+éditeur+décisions immédiates ✓(T5), périmètre
de revue ✓(T5), analyse corrections ✓(T6), cascade re-slice différée (post-revue) ✓.
