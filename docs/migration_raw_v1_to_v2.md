# Stratégie — migration du raw `data_regions` (v1) → `data_regions_v2/raw`

> Rédigé le 2026-07-22. Audit source : 3 agents d'exploration (v1, v2, outillage).
> Décisions utilisateur actées (voir §2). **Aucune exécution sans validation zone par zone.**

## 0. Le constat qui recadre la tâche

`data_regions` v1 **n'est pas majoritairement du raw**. Sur les 18 livraisons, la plupart ne
contiennent que des **sorties de pipeline** (RVT, détections du modèle, tuiles YOLO/jpg, tuiles
MNT IGN re-téléchargeables) — pas la livraison brute de l'archéologue. Le vrai raw est souvent
resté sur le Drive privé de l'archéologue (`lien_drive_source` des manifests), jamais rapatrié.

**Contrainte matérielle dure :** G: = **31 Go libres**, C: = **33 Go libres**, **D: non monté**.
- Copier tout v1 (≈272 Gio) ou tous les dérivés dans `training/` est **impossible** (débordement).
- Extraire chailluz (~30 Go) ne rentre pas sur C: → nécessite D: monté ou du ménage préalable.

Donc : `raw/` = **vraie source uniquement** (~9-10 Go à ajouter), le dérivé lourd régénérable
reste en v1 (pointeur), seul le dérivé léger/utile va dans `training/`.

## 1. Principes (non négociables)

- **v1 est GELÉ** : lecture seule, jamais d'écriture.
- **Jamais d'édition en place sur G:** : nettoyage en **staging local**, puis `robocopy /E /MT:16`.
- **`raw/` immuable** une fois déposé → on nettoie AVANT dépôt, on valide, puis on écrit une fois.
- **Rien n'est supprimé de v1** ; on migre par copie additive.
- **Validation obligatoire avant dépôt** : `.venv\Scripts\python.exe -m audit "<zone stagée>"` doit
  être vert (encodage, CRS, géométries, sidecars) sur chaque `raw/` avant robocopy.

## 2. Décisions actées (2026-07-22)

1. **Sorties pipeline → `training/`** (jamais `raw/`). *Nuancé par l'espace disque : seul le
   dérivé léger/utile est copié ; le lourd régénérable reste en v1 avec pointeur manifest.*
2. **Gros rasters SOURCES → `raw/`** : `MNT_Haye.tif` (3,45 Go), Saint-Germain `*.asc` (une seule
   version, ~5,5 Go). Dédoublonner la version redondante.
3. **Zones sans raw v1 → flaguer « raw manquant » + relancer** via `lien_drive_source`. Pas de faux raw.
4. **chailluz `Download.zip` (15,8 Go) → extraire + auditer** avant de trancher (après déblocage disque).

## 3. Règle de routage par classe de fichier

Appliquée à chaque livraison v1, fichier par fichier :

| Classe | Reconnaissance | Destination v2 |
|---|---|---|
| Vecteur source archéo | shp/gpkg de digitalisation/labels **hors** `output_pipeline_*`, `predictions_*`, `detections_*` | `raw/` |
| Raster source non-dérivé, non re-téléchargeable | `MNT_Haye.tif`, Saint-Germain `*.asc` | `raw/` |
| Docs (rapports DRAC, thèses, `.lyr` symbologie) | `*.pdf`, `.lyr` | `raw/docs/` |
| Prédiction du modèle | `predictions_*`, `detections_*.shp/.gpkg` | **EXCLU du raw** (contamination entraînement) |
| Sortie pipeline vecteur légère / vérité-terrain | `run_rf_detr_*/GT`, shp de détection utiles | `training/pipeline_v1/` (si léger) |
| Raster pipeline régénérable | RVT/LD/SVF/HS `.tif`, `.png`, tuiles `.jpg/.jgw`, `.vrt`, `_8bit` | **NON copié** → pointeur v1 |
| Tuiles MNT IGN LiDAR HD publiques | `LHD_FXX_*_MNT_*.tif` | **NON copié** (re-téléchargeable IGN) → pointeur |
| Junk | `desktop.ini`, `*.lock`, `*.sr.lock`, `.aux.xml` orphelin | jamais |

## 4. Nettoyage appliqué en staging (avant dépôt)

- **`.prj` manquants** (shp confirmés Lambert-93) : écrire le WKT standard EPSG:2154.
  Concernés : Fontainebleau `SHP/77_ONF_*_{parcellaire,limites}` (×2), Saint-Germain `SHP/78_ONF_*_{limites,parcellaire,sites_ONF}` (×3).
- **`patates.prj` = WGS84** (Rambouillet) alors que ses voisins `digit_*` sont L93 : **vérifier les
  coordonnées** ; si la géométrie est en L93 → corriger le `.prj` ; si vraiment WGS84 → laisser tel
  quel en raw + noter. *(sous-décision à trancher à l'inspection)*
- **Dédoublonnage** : Saint-Germain garder **une** version de MNT `.asc` (v1 vs v2 — inspecter, défaut
  v2) et écarter l'autre (~5,5 Go). Rambouillet `78_Foret-Rambouillet_SHP` : la copie sous
  `output_pipeline_*` est un snapshot dérivé → non copiée (racine = canonique, déjà en raw v2).
- **Alès** : exclure `shapefiles_labels/predictions_GARD_40e_data6.*` du raw (déjà acté).
- **Purge junk** : `desktop.ini`, `*.sr.lock` (verrous ArcGIS), `*.lock`.

## 5. Plan par zone

### Groupe A — vrai raw à compléter (le travail réel, ~9-10 Go)

| Zone v2 | Action | Ajout `raw/` |
|---|---|---|
| `grand_est/54_foret_de_haye` | ajouter `MNT_Haye.tif` (source) + `.lyr` symbologie ; indices dérivés = pointeur | ~3,45 Go |
| `ile_de_france/78_saint_germain_marly` | ajouter MNT `.asc` (1 version) + 3 `.prj` + 2 rapports PDF | ~5,5 Go |
| `ile_de_france/77_fontainebleau` | ajouter 10 PDF (thèse/rapports) `raw/docs/` + 2 `.prj` | ~30 Mo |
| `ile_de_france/78_rambouillet` | ajouter rapports PDF (`RAP11530` 269 Mo + `260521`) ; fix `patates.prj` | ~280 Mo |
| `bourgogne_franche_comte/25_haut_doubs` | raw vecteur déjà complet — vérif audit seule | — |
| `occitanie/30_ales_garrigues_ne` | raw complet — confirmer exclusion predictions | — |
| `bretagne/35_anomalie_lidar_bretagne` | complet (1 shp, aucun raster attendu) | — |
| `centre_val_de_loire/41_blois` | vecteurs SOLiDAR OK ; LD/MNT annoncés = absents de v1 → flag | — |

### Groupe B — raw absent de v1 (flaguer + relancer, ne rien fabriquer)

Zones existantes sans `raw/` (v1 = sorties pipeline uniquement) :
`70_vosges_saonoises`, `57_fenetrange`, `30_la_capelle_et_masmolene`.
→ Ajouter au `manifest.yaml` une note :
`ARBITRAGE: raw archéologue absent de v1 (v1 = sorties pipeline). À récupérer via lien_drive_source <lien>.`
(+ corriger : `30_la_capelle` n'a **pas** de `lien_drive_source` ; `78_saint_germain_marly` a le
`lien_drive_source` **dupliqué** de Rambouillet — à remplacer par le vrai.)

Livraisons v1 orphelines, dérivé-seul (pas de zone v2) — **sous-décision** : créer un stub de zone
(manifest + note « raw manquant ») ou juste consigner ?
- `data_dreux` → Centre-Val de Loire (28)
- `data_bataille_de_la_marne` → Hauts-de-France (Oise, Ermenonville — cf. manifest)
- `data_hautes_chaumes_forez` → Auvergne-Rhône-Alpes (Loire 42)
- `data_mont_de_la_croix` → BFC (recouvert par l'emprise de `25_haut_doubs` — noter sous haut_doubs)

### Groupe C — cassé / vide / opaque

- `55_verdun` : `raw/` déjà **complet** (restauré depuis D:) ; v1 n'ajoute rien d'exploitable. RAS.
- `data_Loire_Atlantique` (dept 44) : dossiers de données **vides** (6 PDF only) ; relance mail en cours. Noter, ne rien migrer.
- `data_foret_de_chailluz` : `Download.zip` 15,8 Go **jamais ouvert** → Phase 7 (après déblocage disque).
- `data_bretagne_1` : zip déjà extrait dans `35_anomalie_lidar_bretagne/raw`. RAS.

## 6. Exécution — phases

- **Phase 0 — Prérequis** : monter D: (ou libérer ≥35 Go sur C:) pour chailluz + staging ; confirmer
  headroom G: (31 Go libres > ~10 Go raw à ajouter → OK, marge ~21 Go).
- **Phase 1 — Staging + nettoyage** (Groupe A) : construire `<staging>/<region>/<zone>/raw/…` en
  miroir de v2, en appliquant §3 (routage) et §4 (nettoyage). Staging local (C: ou D:).
- **Phase 2 — Validation** : `python -m audit` sur chaque `raw/` stagé. Zéro erreur avant dépôt.
- **Phase 3 — Dépôt** : `robocopy <staging> "G:\…\data_regions_v2" /E /MT:16 /R:3 /W:10`. Vérifier
  comptes/tailles staging vs Drive (écart attendu = `desktop.ini`).
- **Phase 4 — Dérivé léger → training/** : détections/GT utiles dans `training/pipeline_v1/` ;
  raster/tuiles lourds = **pointeur** dans le manifest (pas de copie).
- **Phase 5 — Manifests** : notes flags Groupe B, décisions dédoublonnage, pointeurs v1, remplir `docs/`.
- **Phase 6 — Index** : `python tools\build_v2_index.py "G:\…\data_regions_v2"` ; commit repo
  (taxonomie/manifests si versionnés). *Le Drive n'est pas commité.*
- **Phase 7 — chailluz** (après Phase 0 débloquée) : extraire en staging, `python -m audit`, créer la
  zone Franche-Comté, migrer le raw validé.

## 7. Outillage réutilisé (rien de neuf à coder)

- Validation : `audit/scan.py` — `build_audit()`, `normalize()`, échelle encodage utf-8→cp1252→latin-1,
  `_crs_record()` (flag ≠2154), détection géométries empty/invalid/mixed, complétude sidecars.
- Dépôt Drive : pattern staging + `robocopy /E /MT:16` (cf. `CLAUDE.md` §Stockage, `dispatch-roboflow`).
- Index : `tools/build_v2_index.py` (schéma manifest attendu : `source`, `notes` typées, `raw/`, `training/`).
- **À écrire si besoin** (petit) : génération d'un `.prj` EPSG:2154 (copier depuis un `.prj` voisin) ;
  reprojection 27572→2154 `geopandas.to_crs(2154)` — **uniquement pour `training/`**, jamais raw.

## 8. Estimation de volume

| Poste | Volume | Destination |
|---|---|---|
| MNT_Haye.tif | 3,45 Go | `raw/` |
| Saint-Germain `.asc` (1 version) | ~5,5 Go | `raw/` |
| PDF/docs (toutes zones) | ~0,5 Go | `raw/docs/` |
| Vecteurs manquants | négligeable | `raw/` |
| **Total ajouté à G:** | **~9,5 Go** (marge OK sur 31 libres) | |
| chailluz extrait | ~30 Go | staging → `raw/` (Phase 7, hors G: tant que non trié) |
| Dérivé lourd régénérable | ~250 Go | **non copié** (pointeur v1) |

---

## 9. Exécution 2026-07-22 — la source réelle était `D:\`, pas le cloud v1

Rebondissement à l'inventaire de `D:\` (disque externe, 1,9 To, monté ce jour) : `D:\data_regions`
contenait les **vraies sources raw locales** (plus propres/complètes que le cloud v1), et
`D:\pipeline_results` ~900 Go de dérivé régénérable (gardé tel quel, hors cloud). Décision
utilisateur : mode **streaming** (pas de miroir), ajouter au cloud les sources uniques de D:,
supprimer les doublons.

**Ajouts déposés dans le cloud v2 `raw/` (depuis D:, robocopy vérifié octet/compte) :**

| Zone | Ajout | Vol | Vérif |
|---|---|---|---|
| `54_foret_de_haye` | `raw/Haye_MNT_INDICES/MNT_Haye.tif` (EPSG:27572) | 3,45 Go | 3448065638 o ✓ |
| `41_blois` | `raw/data_aude/MNT/` (11 dalles + .ovr, Sologne) | ~5 Go | 11 tif ✓ |
| `78_saint_germain_marly` | `raw/MNT/02_MNT_v2/` (.asc) + `raw/docs/` (2 rapports) | ~5,5 Go | 690 asc ✓ |
| `77_fontainebleau` | `raw/docs/` (thèse S.David, rapport 2017, autorisations DRAC) | ~25 Mo | ✓ |
| `78_rambouillet` | `raw/docs/RAP11530…pdf` | 269 Mo | ✓ |
| `44_loire_atlantique` **(zone créée)** | `raw/docs/` 4 rapports LiDAR 44-2024 | 12,5 Go | 4 pdf ✓ |
| `25_besancon_chailluz` **(zone créée)** | `raw/VECTO/bes_charb2015` (1712 charbonnières, 27572) | 48 Ko | ✓ |

**Non migré (volontaire) :** indices dérivés Haye + indices chailluz (SVF/HS/SLOPE/VAT, ~17 Go, pas
de MNT de base dans le zip) → pointeur ; dérivé lourd de `pipeline_results` (~900 Go) → reste sur D:.

**Manifests** : notes `DÉCISION 2026-07-22` sur chaque zone enrichie + 2 notes `ATTENTION` (shp ONF
Fontainebleau/Saint-Germain en **coordonnées géographiques sans .prj** — CRS **non fabriqué**, à
confirmer). Nouveaux manifests `44_loire_atlantique` (ARBITRAGE digitalisation dept 44 absente) et
`25_besancon_chailluz`.

**Doublons D: supprimés après vérif d'intégrité cloud (~26 Go récupérés)** : `_staging_data_regions_v2`
(26575/26582 fichiers identiques au cloud, 7 diffs bénins), `datasets_roboflow` (5 zips octet-identiques
à `_archives_roboflow`), `data_aude/francetransfert.zip` (contenu déjà en cloud).

**Reste côté data (hors code, à relancer)** : digitalisation vecteur **dept 44** ; MNT de base
**chailluz** ; CRS des shp **ONF** (Fontainebleau/Saint-Germain) à confirmer ; digitalisation **dept 41**
LD/MNT si le SOLiDAR n'est pas complet. `D:\_chailluz_extract` (17 Go) conservé pour le futur
`training/` (reprojection 27572→2154).
