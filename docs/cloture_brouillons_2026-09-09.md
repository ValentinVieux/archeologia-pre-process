# Clôture des brouillons de chantier — 2026-09-09

Les deux dossiers de travail locaux du chantier ponctuelles/linéaires ont été supprimés
ce jour après contrôle (`tools/cloturer_chantier.py`, verdict **CLÔTURABLE** pour les
deux : tout fichier a un jumeau dans une destination pérenne, ou une justification
écrite ci-dessous).

| Brouillon | Fichiers | Volume | Justifiés (non déposés) |
|---|---|---|---|
| `D:\entrainement_ponctuelles` | 12 564 | 78,33 Go | 2 873 f / 30,60 Go |
| `D:\entrainement_lineaires_v3` | 18 | 0,74 Go | 7 f / 0,03 Go |

Dépôts effectués avant clôture (détail : commit `e882197` et notes `DÉCISION 2026-09-09`
des manifests de zone) : GPKG d'entités v3/v4 de 4 zones, `raw/MNT/LD_livre_dalles` de
57_fenetrange et 70_vosges_saonoises, package du modèle plugin retiré `run_rf_detr_1`,
6 sidecars `best.json`, 6 `<zone>_DECISIONS.txt`, audit final 2026-09-04.

Ce document conserve les `cloture-ignore.txt` des deux brouillons : c'est la seule trace
de ce qui a été sciemment détruit, et la justification de chaque ligne.

## `D:\entrainement_ponctuelles` — cloture-ignore.txt

```
# cloture-ignore.txt — D:\entrainement_ponctuelles (chantier ponctuelles, clôturé le 2026-09-09)
#
# Ce qui n'a DÉLIBÉRÉMENT aucune copie pérenne : un motif glob par ligne (chemin relatif
# au brouillon, slashs avant), justification OBLIGATOIRE en commentaire.
# Tout le reste doit être déposé (Drive data_regions_v2 / model-training, dépôt, plugin).
# Contrôle : .venv\Scripts\python.exe tools\cloturer_chantier.py D:\entrainement_ponctuelles

# --- données re-téléchargeables ------------------------------------------------------
*/mnt_dalles/*                          # 29,3 Go de dalles MNT LiDAR HD IGN (ales 1106, haut_doubs 411, la_capelle 243, chailluz 208) : re-téléchargeables par tools/telecharger_dalles_ign.py — règle CLAUDE.md « pas en cloud » ; les mosaïques et LD qui en dérivent SONT déposés (raw/MNT)
controle_zones/align_*/mnt_ign/*        # 10 dalles MNT IGN tirées pour le contrôle d'alignement de fenetrange/vosges : re-téléchargeables à l'identique
controle_zones/align_*/mnt_ign.vrt      # VRT de ces dalles

# --- produits recalculables ----------------------------------------------------------
blois/blois_mnt.tif                     # mosaïque locale des a.tif…k.tif du Drive (41_blois/raw/MNT) : mosaique_mnt.py la refait ; le LD qui en dérive EST déposé
*.aux.xml                               # statistiques GDAL, recalculées à la lecture
zones/*/ld.vrt                          # VRT à chemins locaux ; équivalent déposé = <zone>/raw/MNT/LD_livre_dalles.vrt (cf. commentaires des configs)
zones/*/liste_tifs.txt                  # inventaire du VRT ci-dessus
zones/78_rambouillet/ld/*.vrt           # idem, LD du run v1 gelé (déposé en raw/Rambouillet_MNT_IGN/)
zones/78_rambouillet/ld/liste_tifs.txt
zones/78_rambouillet/ld_complement/*    # dalles ré-extraites du payload Roboflow, lui déposé (training/roboflow) : re-extractibles

# --- journaux ------------------------------------------------------------------------
*.log
*_log.txt
*_err.txt
*.temoin.txt                            # témoins d'intégrité (effectif + sha256 WKB) d'un run donné : jetables une fois le run relu

# --- intermédiaires périmés, supersédés par un dépôt ---------------------------------
zones/*/annotations_coco.gpkg           # annotations v1 (géoréf FAUSSE pour haut_doubs/ales/la_capelle) ; état corrigé, revu et dédoublonné déposé en training/vecteurs (cf. commentaires des configs)
zones/*/annotations_coco_v2.gpkg        # idem, état v2
zones/*/annotations_v3_dedup.gpkg       # = les <zone>_entites_l93_v3/v4.gpkg déposés le 2026-09-09 (mêmes géométries, contrôlé)
zones/25_besancon_chailluz/*_v2.gpkg    # auto-labels v2 de Chailluz ; état revu déposé (25_besancon_chailluz_entites_l93_v3.gpkg)
zones/78_rambouillet/rambouillet_entites*.gpkg  # états v1/v2 pré-revue ; état revu + dédoublonné déposé (78_rambouillet_entites_l93_v3.gpkg)
zones/*/LISEZ-MOI-georef.md             # repris et complété dans raw/MNT/LISEZ-MOI.md des zones concernées, sur le Drive
controle_zones/*.gpkg                   # copies de contrôle des GPKG de zone (dédoublonnage, comparaisons) : régénérables depuis les fichiers déposés
revue_zones/**                          # GPKG de travail de la revue QGIS et leurs .bak ; le RÉSULTAT est déposé (training/vecteurs/<zone>_entites_l93_v3|v4.gpkg) et les décisions aussi (<zone>_DECISIONS.txt)
staging/**                              # copies de staging du dépôt du 2026-09-08 : le Drive fait foi
staging_drive/**                        # copies de staging du dépôt du 2026-09-07 : le Drive fait foi et est en avance (manifests, modeles.yaml)
audit_dalles_2026-09-01/*               # scripts et JSON de diagnostic du 2026-09-01 (joints km, splits) : l'audit FINAL du 2026-09-04 est déposé (model-training/ponctuelles/_audits/)
diag_four/**                            # diagnostics de contraste et inférences de mise au point : les conclusions sont dans docs/ et les manifests ; les détections restent hors data_regions_v2 (règle CLAUDE.md)
seuils/*                                # étude de seuils du 2026-08-21, remplacée par l'éval outillée (courbes_eval + metriques_eval.json déposés)
eval_epoque41/corpus/**                 # copie locale du corpus d'éval : le corpus est déposé (model-training/ponctuelles/ponctuelles_3_classes/corpus/)
rfdetr_unified_pipeline_v2*.ipynb       # instantanés datés du notebook canonique ; le Drive (G:\Mon Drive\Colab Notebooks) est en avance et le dépôt en garde une copie sans sorties
REMISE-datasets-v2.md                   # archivé dans le dépôt : docs/chantier_ponctuelles_v2_2026-09.md
REPRISE-chantier-v2.md                  # idem
rf-detr-large-2026.pth                  # poids de base, déposé en model-training/_poids_base/

# --- ajouts après la 1re passe de contrôle (2026-09-09) -------------------------------
controle_zones/LISEZ-MOI.txt            # guide de chargement QGIS de la session de contrôle du 2026-09-04, entièrement en chemins D: ; l'information (quel GPKG avec quel LD par zone) vit désormais dans les commentaires des configs et les manifests
zones/*/annotations_coco_v3.gpkg        # annotations corrigées du géoréférencement : VÉRIFIÉES identiques (effectifs + sha256 des WKB triés, par couche) aux <zone>_entites_l93_v3.gpkg déposés — même contenu, autre nom
```

## `D:\entrainement_lineaires_v3` — cloture-ignore.txt

```
# cloture-ignore.txt — D:\entrainement_lineaires_v3 (chantier lineaires_seg_v3_1, clôturé le 2026-09-09)
#
# Ce qui n'a DÉLIBÉRÉMENT aucune copie pérenne : un motif glob par ligne (chemin relatif
# au brouillon, slashs avant), justification OBLIGATOIRE en commentaire.
# Contrôle : .venv\Scripts\python.exe tools\cloturer_chantier.py D:\entrainement_lineaires_v3

evaluation_v3_1_vs_v2_1/*   # éval 2 modèles du 2026-09-01, SUPERSÉDÉE par evaluation_3modeles (v3_1 + v2_1 + formes_lineaires_1 sur la MÊME éval gelée) : celle-ci est déposée en runs/training/lineaires_seg_v3_1/evaluation/ (metriques_eval.json complété par par_zone_classe le 2026-09-03) et dans le plugin. Rien d'unique ici.
```
