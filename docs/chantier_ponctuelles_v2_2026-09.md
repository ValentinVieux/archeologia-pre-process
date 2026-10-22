# Chantier ponctuelles v2 — notes de chantier (2026-09-03 / 2026-09-04)

Archivé ici le 2026-09-09 à la clôture du brouillon `D:\entrainement_ponctuelles`
(CLAUDE.md § Brouillons de chantier). Ces deux notes vivaient à la racine du brouillon
sous `REPRISE-chantier-v2.md` et `REMISE-datasets-v2.md` ; elles sont la trace des
décisions et des pièges du chantier « correctifs marges + cible détection bbox ».

Compléments déposés sur le Drive :
`model-training/ponctuelles/_audits/audit_final_2026-09-04/` (scripts, JSON, et le
`gt_paires_residuelles.gpkg` de l'arbitrage encore PENDANT).

---

# Remise — datasets ponctuelles v2 (2026-09-04, ~06h)

Chantier « correctifs marges + cible détection bbox » TERMINÉ côté datasets.
Audit final indépendant : 2 vérificateurs (annotations + rasters), verdicts ci-dessous.
Rien n'est commité ; le corpus n'est PAS construit (attente de votre vérification).

## Les 8 zones

| zone | dataset | tuiles train/valid/test | verdicts |
|---|---|---|---|
| 57_fenetrange | ponctuelles_57_fenetrange_ld648_v2 | 726/220/117 | CONFORME, split v1 préservé (42 blocs, 0 div.) |
| 70_vosges_saonoises | ponctuelles_70_vosges_saonoises_ld648_v2 | 223/70/38 | CONFORME, split préservé (41, 0 div.) |
| 78_rambouillet | ponctuelles_78_rambouillet_ld648_v2 | 988/330/136 | CONFORME, split préservé (85, 0 div.) |
| 25_chailluz (EXCLU du corpus — décision 03/09) | ponctuelles_25_chailluz_ld648_v2 | 252/82/50 | CONFORME, split préservé (37, 0 div.) |
| 30_la_capelle | ponctuelles_30_la_capelle_ld648_v2 | 167/46/22 | CONFORME, split préservé (41 communs, 0 div., ±2 blocs de bord) |
| 25_haut_doubs | ponctuelles_25_haut_doubs_ld648_v2 | 575/161/100 | CONFORME, split préservé (93 communs, 0 div., ±1) |
| 30_ales_garrigues_ne | ponctuelles_30_ales_garrigues_ne_ld648_v2 | 1620/446/247 | CONFORME, split préservé (207 communs, 0 div., +13/-4) |
| 41_blois | fours_charb_41_blois_ld648_v1 (INCHANGÉ) | 211/62/33 | aucun correctif nécessaire (audité sain) |

Correctifs appliqués : GPKG `_v2` avec 34 composantes d'objets coupés aux joints km
fusionnées (fenêtrange 27, vosges 4, rambouillet 3) ; auto-labels Chailluz refaits en
mode recouvrement CONSERVATEUR (92 détections de base identiques v1 + 3 géométries
réparées + 8 objets de joint récupérés = 100) ; LD RÉGÉNÉRÉS depuis le MNT LiDAR HD
IGN pour haut_doubs (411 dalles), ales (1106), la_capelle (243) — mosaïque une passe
0,5 m + generer_ld (gsd bâtard 0,4543 -> 0,5 partout).

## Audit final — rasters LD régénérés : CONFORME

| raster | joints km fautifs | ratio médian v2 (v1) | joints internes 4096 | halos |
|---|---|---|---|---|
| haut_doubs_ld | 0/40 | 0,86 (10,74) | 0/20 | 0 |
| ales_ld | 0/40 | 0,835 (10,57) | 0/20 | 0 |
| la_capelle_ld | 0/40 | 0,80 (8,39) | 0/19 | 0 |

Géométrie 3/3 : gsd exactement 0,5, EPSG:2154, -tap, NoData 255. Le défaut de marge
v1 est ÉLIMINÉ. Détail : audit_final_2026-09-04\ld_audit_final.json.

## Audit final — annotations : corrections effectives, 1 trouvaille

- **35/35 paires v1 fusionnées et absentes des datasets** (aucun signal à leurs
  emplacements) ; bord_km 681 -> 620 (2,77 %), baisses par zone conformes au prévu.
- gsd 0,5 vérifié sur les 7 manifests ; zones re-grillées : plus AUCUNE coïncidence
  bord de tuile / ligne km.
- **Restants CONNUS et acceptés** : ~606 troncatures sèches (moitié jamais annotée —
  incorrigible sans revue humaine) ; ratios l/h>=2 inchangés (hors périmètre marges).
- **Trouvaille (pré-existante, pas une régression)** : 7 paires résiduelles à bord plat
  sur ligne km (2 vosges + 5 rambouillet, 14 annotations), invisibles au critère du
  diagnostic v1 (ellipses tangentes : distance géométrique 8-20 m, bbox à 0 m).
  À ARBITRER (fusion ou suppression du reliquat) :
  gt_paires_residuelles.gpkg dans audit_final_2026-09-04\ (+ détail au JSON).

## Outillage modifié (non commité)

slice_zone `--split-depuis` (reprise blocs->split d'un manifest, jamais re-tirer) ;
mosaique_mnt `--optfile` (limite argv Windows >300 dalles) + `--mono` (gdalwarp -multi
corrompt son tas sur gros lots, 3 crashs mesurés) ; generer_ld `--procs` (tuiles en
parallèle, sortie BYTE-IDENTIQUE vérifiée — Alès 24 h -> 5 h) ; auto_label_depressions
`--recouvrement` (4 passes décalées 500 m + fusion conservatrice) ; + les configs
`ponctuelles_*_ld648_v2` et le renommage repo/D:/Drive de l'autre session.

## Prochaines étapes (après VOTRE vérification)

1. Arbitrage des 7 paires résiduelles (QGIS : gt_paires_residuelles.gpkg).
2. Corpus ponctuelles v2 SANS chailluz ni blois-dépressions (recette 7 datasets :
   6 zones v2 + blois v1) via /corpus-entrainement.
3. Colab DÉTECTION : TASK=object_detection, RESOLUTION 672 (648 invalide en détection),
   NUM_QUERIES 300, run type A depuis rf-detr-large-2026.pth (à épingler _poids_base
   +sha256) ; corriger les garde-fous du notebook (assert %24 seg-only, warning %14
   périmé) ; éval courbes_eval vs fours_charb_seg_ld_v1_ep41 sur le MÊME split test.

---

# Reprise du chantier datasets ponctuelles v2 (pause utilisateur 2026-09-03 ~18h)

État figé proprement : tous les calculs arrêtés, dossier renommé
`D:\entrainement_ponctuelles` (ex entrainement_fours_charbonnieres), VRT réécrits
et vérifiés CONFORMES. Le repo (configs `ponctuelles_*_ld648_v2`, outils modifiés)
n'est PAS commité.

## Fait et vérifié
- GPKG corrigés : fenetrange/vosges/rambouillet `*_v2.gpkg` (34 composantes d'objets
  coupés aux joints km fusionnées, verdict CONFORME) ; chailluz_entites_v2.gpkg
  (auto-labels recouvrement conservateur : 92 base + 3 réparées + 8 récupérées = 100).
- 4 datasets découpés et CONFORMES sous l'ANCIEN nommage `fours_charb_*_ld648_v2`
  (fenetrange, vosges, rambouillet, chailluz) — À REDÉCOUPER sous les noms
  `ponctuelles_*` (configs de l'autre session), puis supprimer les anciens dossiers.
- La Capelle : mnt + LD régénérés CONFORMES (la_capelle/la_capelle_ld.tif).
- Téléchargements MNT complets : haut_doubs 411 dalles, ales 1106 dalles.
- Blois : aucun correctif, dataset v1 conservé tel quel.

## Reste à faire (dans l'ordre, SÉRIALISER les gros jobs — 15 Go de RAM)
1. Haut-Doubs :
   .venv\Scripts\python.exe tools\mosaique_mnt.py D:\entrainement_ponctuelles\haut_doubs\mnt_dalles D:\entrainement_ponctuelles\haut_doubs\haut_doubs_mnt.tif --tr 0.5
   .venv\Scripts\python.exe tools\generer_ld.py D:\entrainement_ponctuelles\haut_doubs\haut_doubs_mnt.tif D:\entrainement_ponctuelles\haut_doubs\haut_doubs_ld.tif
2. Alès : idem avec ales\ (1106 dalles — le plus long).
3. Découpes ×7 (configs ponctuelles_*_ld648_v2, split v1 imposé) :
   .venv\Scripts\python.exe tools\slice_zone.py configs\ponctuelles_<zone>_ld648_v2.yaml --split-depuis manifests\split\fours_charb_<zone>_ld648_v1.yaml
4. Par dataset : tools\verif_dataset.py (avec --gpkg du producteur) + comparer_splits.py
   (archivé dans audit_dalles_2026-09-01\) v1 vs v2 — exiger CONFORME + SPLIT PRÉSERVÉ.
5. Supprimer datasets\fours_charb_*_ld648_v2 (anciens noms).
6. AUDIT FINAL indépendant du dataset complet (demande utilisateur) : re-contrôle
   des signatures de joints (bords km, paires, ratios) sur les COCO v2, coutures des
   3 nouveaux LD, rapport — puis remise à l'utilisateur pour SA vérification.
7. Ensuite (phases suivantes, après validation humaine) : corpus v2 (« ponctuelles »)
   + notebook Colab en DÉTECTION (TASK=object_detection, RESOLUTION 672, NUM_QUERIES
   300, run type A depuis rf-detr-large-2026.pth — cf. mémoire chantier).

Détail des pièges déjà mordus : >300 chemins gdalwarp = --optfile (corrigé dans
mosaique_mnt.py) ; 2 gros jobs raster en parallèle = crash heap gdalwarp (sérialiser) ;
reprises IGN idempotentes (relancer la même commande).
