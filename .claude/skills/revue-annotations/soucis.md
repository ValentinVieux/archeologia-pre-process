# Journal des soucis — revue d'annotations (à enrichir à chaque incident)

Chaque entrée : date, symptôme, cause, parade (outil ou règle). Les règles qui en
découlent sont dans SKILL.md ; ici, le détail pour ne pas rediagnostiquer.

| Date | Symptôme | Cause | Parade |
|---|---|---|---|
| 2026-09-06 | Un run SAM lancé sur 884 boîtes, la couche en compte 988 à la fin (817 dessinées à la main) | l'utilisateur édite pendant le calcul | témoin d'intégrité (effectif + sha256 WKB) avant/après ; sortie séparée ; relance après stabilisation |
| 2026-09-06 | 249 polygones SAM débordent de leur boîte (2 m en médiane, 13 m max) | SAM déborde de l'invite ; lissage à 2 m | producteur : découpe à la boîte, repli si le centre n'est pas couvert |
| 2026-09-07 (Blois) | Polygones SAM restés rectangulaires, livrés « CONFORMES » | test d'aire sur le masque brut (qui déborde) ; vérificateur qui ne jugeait que la géométrie | test sur le masque découpé ; champ `motif` par repli ; vérificateur : replis sans motif, quasi-rectangles, taux > 5 % refusés ; règle « contrôler avant de livrer » |
| 2026-09-06 | Annotations alignées sur le LD ANCIEN mais pas sur le NOUVEAU ni Google Maps | dalles de 2 201 px (km + 50 m) géoréférencées comme 1 000 m : compression 10 %, ±50 m aux bords, héritée par les annotations (`px = 1000/width`) | `verif_alignement_ld.py` (décalage qui change de signe N/S = ÉCHELLE), `corriger_georef_dalles.py`, `geo_dalle()` dans coco_a_gpkg ; doc `docs/incident_georef_dalles_2026-09-06.md` |
| 2026-09-07 | Doublons massifs après correction (fours Haut-Doubs 231, charb. Alès 341) | objets de la marge annotés dans deux dalles voisines, coïncidents une fois le géoréf juste | `dedup_annotations.py` après toute correction géométrique |
| 2026-09-06 | Rambouillet « sans GPKG de revue » | il était déjà dans `traité\` | regarder `traité\` d'abord |
| 2026-09-06 | pyogrio `local variable 'wkt' referenced before assignment` (Haye) | `.prj` en latin-1 | lire les `.dbf` copiés seuls, sans `.prj` |
| 2026-09-06 | `Select-String` explose la mémoire, `rg` > 20 s sur G: | gros JSON, Drive lent | inventorier (`Get-ChildItem`), grep par liste de fichiers < 5 Mo ; PDF via pypdf (les scans n'ont pas de texte) |
| 2026-09-06/07 | Scripts cassés trois fois par des `\\` | l'outil Bash halve les antislashs des heredocs ; `\t` dans un YAML entre guillemets = tabulation | écrire les scripts avec Write ; `bytes([92])` ; slashs avant partout |
| 2026-09-06 | Boîtes ajoutées par vagues, SAM relancé à chaque fois (6 min / 1 000 boîtes) | pas de lien polygone → boîte | `fid_source` dans la couche `_sam` ; run incrémental à outiller si les vagues continuent |
| 2026-09-07 | « élargie de 20 » compris 20 m par côté alors que l'utilisateur pensait 20 % | unité implicite | annoncer l'unité avant de lancer ; marge 0 acté |
