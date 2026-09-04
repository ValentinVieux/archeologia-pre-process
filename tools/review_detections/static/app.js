/* Revue de détections par tuile, en BOUNDING BOXES — vanilla, aucune dépendance.
   Coordonnées de travail = pixels de la TUILE CENTRALE (648 px) ; l'image servie
   peut être une mosaïque 3×3, décalée de X-Decalage (le dessin ajoute l'offset). */
"use strict";

const $ = (s) => document.querySelector(s);
const canvas = $("#canvas");
const ctx = canvas.getContext("2d");
const TUILE = 648;

const etat = {
  tuiles: [], classes: [], zones: [], splits: [], seuils: {}, modele: "",
  cle: null, detail: null, img: null, decalage: [0, 0],
  vue: { zoom: 1, px: 0, py: 0 },
  selUid: null, boite: null, baseBoite: null, dirty: false,
  undo: [], redo: [],
  montrerGT: true, montrerDet: true, sousSeuil: false, voisines: true,
  masquer: false, contexte: true, classeActive: null, ajout: null, historique: [],
};

/* couleurs par classe : stables, distinctes du blanc (GT) et du jaune (édité) */
const PALETTE = [[90, 162, 224], [224, 120, 70], [86, 214, 200],
                 [199, 123, 214], [159, 214, 90], [214, 90, 126]];
const couleurClasse = (classe, a) => {
  const rgb = PALETTE[Math.max(0, etat.classes.indexOf(classe)) % PALETTE.length];
  return `rgba(${rgb.join(",")},${a})`;
};

/* ---------- conversions (tuile centrale px <-> écran) ---------- */
const depuisEcran = ([sx, sy]) => [(sx - etat.vue.px) / etat.vue.zoom - etat.decalage[0],
                                   (sy - etat.vue.py) / etat.vue.zoom - etat.decalage[1]];
const normaliser = ([x0, y0, x1, y1]) => [Math.min(x0, x1), Math.min(y0, y1),
                                          Math.max(x0, x1), Math.max(y0, y1)];

/* ---------- accès aux entités de la tuile ---------- */
const detections = () => (etat.detail ? etat.detail.detections : []);
const ajouts = () => Object.entries((etat.detail && etat.detail.ajouts) || {})
  .map(([uid, a]) => ({ uid, ...a, ajout: true }));
const parUid = (uid) => detections().find((d) => d.uid === uid)
  || ajouts().find((a) => a.uid === uid);
const aTraiter = () => detections().filter((d) => d.retenu && !d.decision);

function boiteDe(item) {
  if (item.decision && item.decision.bbox_px) return item.decision.bbox_px;
  return item.bbox_px;
}

/* ---------- liste des tuiles ---------- */
async function chargerListe(ouvrirPremiere = true) {
  const q = new URLSearchParams({ split: $("#f-split").value,
                                  zone: $("#f-zone").value,
                                  classe: $("#f-classe").value });
  const r = await (await fetch("/api/tuiles?" + q)).json();
  Object.assign(etat, { tuiles: r.tuiles, classes: r.classes, zones: r.zones,
                        splits: r.splits, seuils: r.seuils, modele: r.modele });
  if (!etat.classeActive) etat.classeActive = r.classes[0];
  $("#titre-modele").textContent = r.modele;
  document.title = `Revue détections — ${r.modele}`;
  if ($("#f-zone").options.length === 1) {
    for (const z of r.zones)
      $("#f-zone").insertAdjacentHTML("beforeend", `<option value="${z}">${z}</option>`);
    for (const s of r.splits)
      $("#f-split").insertAdjacentHTML("beforeend", `<option value="${s}">${s}</option>`);
    for (const c of r.classes)
      $("#f-classe").insertAdjacentHTML("beforeend", `<option value="${c}">${c}</option>`);
    $("#lg-classes").innerHTML = r.classes.map((c) =>
      `<span class="lg"><span class="chip" style="background:${
        couleurClasse(c, 1)}"></span>${c} (seuil ${etat.seuils[c]})</span>`).join("");
  }
  rendreListe();
  await progression();
  if (!ouvrirPremiere) return;
  const vis = visibles();
  const premiere = vis.find((t) => t.restantes > 0) || vis[0];
  if (premiere) ouvrir(premiere.cle);
}

const visibles = () => {
  const mode = $("#f-etat").value;
  let l = etat.tuiles;
  if (mode === "a_traiter") l = l.filter((t) => t.restantes > 0);
  else if (mode === "finies") l = l.filter((t) => !t.restantes);
  l = [...l];
  if ($("#f-tri").value === "score") l.sort((a, b) => b.score_max - a.score_max);
  else if ($("#f-tri").value === "non_gt") l.sort((a, b) => b.non_gt - a.non_gt);
  return l;
};

function rendreListe() {
  $("#liste").innerHTML = visibles().map((t) => `
    <li data-cle="${t.cle}" class="${t.restantes ? "" : "finie"}
        ${t.cle === etat.cle ? "actif" : ""}">
      <span>${t.cle.replace(/^(train|valid|test)\//, "$1 · ").replace(".png", "")}</span>
      ${t.non_gt ? `<span class="badge b-nongt">${t.non_gt} hors GT</span>` : ""}
      <span class="compte">${t.n_retenues - t.restantes}/${t.n_retenues}</span>
    </li>`).join("");
  document.querySelectorAll("#liste li").forEach((li) =>
    li.addEventListener("click", () => ouvrir(li.dataset.cle)));
  const actif = $("#liste li.actif");
  if (actif) actif.scrollIntoView({ block: "nearest" });
}

async function progression() {
  const p = await (await fetch("/api/progression")).json();
  $("#progression").textContent =
    `${p.decidees_retenues}/${p.retenues} détections retenues — ` +
    `${p.tuiles_finies}/${p.tuiles_total} tuiles — ${p.ajouts} ajouts — ` +
    Object.entries(p.par_decision).filter(([, n]) => n)
      .map(([d, n]) => `${d} ${n}`).join(", ");
}

/* ---------- ouverture d'une tuile ---------- */
async function ouvrir(cle, depuisHistorique = false) {
  if (etat.dirty
      && !window.confirm("Édition non validée (Entrée pour la valider) — l'abandonner ?"))
    return;
  const precedent = etat.cle;
  const [detail, img] = await Promise.all([
    (await fetch("/api/tuile/" + cle)).json(),
    fetch(`/api/image/${cle}?contexte=${etat.contexte ? 1 : 0}`),
  ]);
  if (!depuisHistorique && precedent && precedent !== cle) {
    etat.historique.push(precedent);
    if (etat.historique.length > 50) etat.historique.shift();
  }
  etat.decalage = JSON.parse(img.headers.get("X-Decalage"));
  etat.img = await createImageBitmap(await img.blob());
  etat.detail = detail;
  etat.cle = cle;
  etat.dirty = false;
  etat.undo = []; etat.redo = []; etat.ajout = null;
  const premiere = aTraiter()[0];
  selectionner(premiere ? premiere.uid : null, false);
  cadrer();
  rendrePanneau();
  rendreListe();
  dessiner();
}

function cadrer() {
  /* cadre la TUILE CENTRALE (le contexte déborde autour) */
  const z = Math.min(canvas.clientWidth / TUILE, canvas.clientHeight / TUILE) * 0.92;
  etat.vue.zoom = z;
  etat.vue.px = (canvas.clientWidth - TUILE * z) / 2 - etat.decalage[0] * z;
  etat.vue.py = (canvas.clientHeight - TUILE * z) / 2 - etat.decalage[1] * z;
}

function selectionner(uid, rafraichir = true) {
  etat.selUid = uid;
  etat.undo = []; etat.redo = []; etat.dirty = false;
  const item = uid && parUid(uid);
  etat.baseBoite = item ? [...boiteDe(item)] : null;
  etat.boite = item ? [...etat.baseBoite] : null;
  if (rafraichir) { rendrePanneau(); dessiner(); }
}

function rendrePanneau() {
  const t = etat.tuiles.find((x) => x.cle === etat.cle) || {};
  $("#p-tuile").textContent = etat.cle || "";
  const d = etat.detail || {};
  $("#p-mesures").innerHTML = [
    ["zone", d.zone], ["split", d.split], ["GT sur la tuile", d.n_gt],
    ["détections", (d.detections || []).length + " (" + (t.n_retenues || 0) + " retenues)"],
    ["ajouts", Object.keys(d.ajouts || {}).length],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v ?? ""}</td></tr>`).join("");
  const item = etat.selUid && parUid(etat.selUid);
  $("#p-selection").innerHTML = !item ? "<i>aucune sélection</i>" : (item.ajout
    ? `<b>${item.uid}</b> — ajout ${item.classe}`
    : `<b>${item.uid}</b><br>${(item.decision && item.decision.classe) || item.classe}
       — score ${item.score}${item.retenu ? "" : " (sous le seuil)"}
       — ${item.gt_apparie ? "GT appariée (IoU " + item.iou_gt + ")" : "hors GT"}`);
  $("#p-decision").textContent = etat.ajout
    ? `mode ajout ${etat.classeActive} — glisser une boîte · Échap : annuler`
    : (etat.dirty ? "édition en cours — Entrée pour valider"
       : (item && item.decision ? "décision : " + item.decision.decision
          + (item.decision.classe !== item.classe ? " → " + item.decision.classe : "")
          : ""));
  $("#p-classe-active").innerHTML =
    `classe active (ajouts/reclassement) : <b>${etat.classeActive || ""}</b> — ` +
    etat.classes.map((c, i) => `${i + 1}=${c}`).join(" · ");
}

/* ---------- dessin ---------- */
function dessiner() {
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(canvas.clientWidth * dpr)
      || canvas.height !== Math.round(canvas.clientHeight * dpr)) {
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0d0e11";
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (!etat.img) return;
  const { zoom, px, py } = etat.vue;
  ctx.setTransform(dpr * zoom, 0, 0, dpr * zoom, dpr * px, dpr * py);
  ctx.imageSmoothingEnabled = zoom < 1.5;
  ctx.drawImage(etat.img, 0, 0);
  const [dx, dy] = etat.decalage;
  ctx.translate(dx, dy); // -> coordonnées tuile centrale
  if (dx || dy) { // liseré de la tuile centrale quand le contexte est affiché
    ctx.strokeStyle = "rgba(214,217,222,.35)";
    ctx.lineWidth = 1 / zoom;
    ctx.strokeRect(0, 0, TUILE, TUILE);
  }

  if (!etat.masquer && etat.detail) {
    if (etat.voisines && (dx || dy)) {
      for (const v of etat.detail.voisines || []) {
        const ox = v.dc * TUILE, oy = v.dr * TUILE;
        if (etat.montrerGT) for (const g of v.gt)
          rect(g.bbox_px, "rgba(255,255,255,.18)", 1 / zoom, ox, oy, [3 / zoom, 3 / zoom]);
        if (etat.montrerDet) for (const d of v.detections)
          rect(d.bbox_px, couleurClasse(d.classe, 0.35), 1.2 / zoom, ox, oy);
      }
    }
    if (etat.montrerGT) for (const g of etat.detail.gt) {
      rect(g.bbox_px, "rgba(255,255,255,.75)", 1.4 / zoom, 0, 0, [4 / zoom, 3 / zoom]);
      ctx.fillStyle = "rgba(255,255,255,.75)";
      ctx.font = `600 ${10 / zoom}px system-ui`;
      ctx.fillText("GT " + g.classe.slice(0, 5), g.bbox_px[0] + 2 / zoom,
                   g.bbox_px[3] - 3 / zoom);
    }

    for (const item of [...detections(), ...ajouts()]) {
      const sel = item.uid === etat.selUid;
      if (!etat.montrerDet && !sel) continue;
      if (!item.ajout && !item.retenu && !item.decision && !etat.sousSeuil && !sel)
        continue;
      const dec = item.decision && item.decision.decision;
      const classe = (item.decision && item.decision.classe) || item.classe;
      const b = sel ? etat.boite : boiteDe(item);
      const c = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
      if (sel) rect(b, "rgba(255,255,255,.9)", 4.5 / zoom);
      if (dec === "invalide") {
        rect(b, couleurClasse(classe, 0.22), 1.5 / zoom);
        croix(c, 6 / zoom, "rgba(224,85,85,.9)");
      } else if (dec === "editee" || (sel && etat.dirty)) {
        rect(b, "#e8c93e", 2 / zoom);
      } else if (item.ajout) {
        rect(b, couleurClasse(classe, 0.95), 2 / zoom);
        croix(c, 5 / zoom, couleurClasse(classe, 0.95), true);
      } else {
        const alpha = dec === "valide" ? 0.95 : (item.retenu ? 0.85 : 0.5);
        rect(b, couleurClasse(classe, alpha), (dec ? 2 : 2.2) / zoom,
             0, 0, item.retenu || dec ? null : [5 / zoom, 4 / zoom]);
        if (dec === "valide") coche(c, 6 / zoom);
      }
      if (!item.ajout && (sel || zoom > 0.8)) {
        ctx.fillStyle = sel ? "#ffffff" : couleurClasse(classe, 0.8);
        ctx.font = `${11 / zoom}px system-ui`;
        ctx.fillText(item.score.toFixed(2), b[0], b[1] - 3 / zoom);
      }
      if (sel && !etat.ajout) { // poignées coins + milieux de bords
        const t = 6 / zoom;
        ctx.fillStyle = "#e8c93e";
        for (const [u, v] of poignees(b))
          ctx.fillRect(u - t / 2, v - t / 2, t, t);
      }
    }
    if (etat.ajout && etat.ajout.b) { // aperçu de la boîte en cours
      rect(normaliser(etat.ajout.b), couleurClasse(etat.classeActive, 0.9), 2 / zoom);
    }
  }
  ctx.translate(-dx, -dy);
  $("#hud").textContent = `${etat.cle || ""} — zoom ${etat.vue.zoom.toFixed(1)}×`
    + (etat.masquer ? " — LD à nu" : "")
    + (etat.montrerDet ? "" : " — détections masquées")
    + (etat.sousSeuil ? " — sous-seuil affiché" : "");
}

function rect([x0, y0, x1, y1], style, largeur, ox = 0, oy = 0, tirets = null) {
  ctx.strokeStyle = style;
  ctx.lineWidth = largeur;
  ctx.setLineDash(tirets || []);
  ctx.strokeRect(x0 + ox, y0 + oy, x1 - x0, y1 - y0);
  ctx.setLineDash([]);
}

const poignees = ([x0, y0, x1, y1]) => {
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  return [[x0, y0], [mx, y0], [x1, y0], [x1, my],
          [x1, y1], [mx, y1], [x0, y1], [x0, my]];
};
const POIGNEE_AXES = [ // pour chaque poignée : bouge-t-elle x0/y0/x1/y1 ?
  [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0],
  [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]];

function croix([x, y], t, style, plus = false) {
  ctx.strokeStyle = style;
  ctx.lineWidth = t / 3;
  ctx.beginPath();
  if (plus) { ctx.moveTo(x - t, y); ctx.lineTo(x + t, y);
              ctx.moveTo(x, y - t); ctx.lineTo(x, y + t); }
  else { ctx.moveTo(x - t, y - t); ctx.lineTo(x + t, y + t);
         ctx.moveTo(x - t, y + t); ctx.lineTo(x + t, y - t); }
  ctx.stroke();
}
function coche([x, y], t) {
  ctx.strokeStyle = "rgba(79,195,107,.95)";
  ctx.lineWidth = t / 3;
  ctx.beginPath();
  ctx.moveTo(x - t, y); ctx.lineTo(x - t / 3, y + t * 0.7); ctx.lineTo(x + t, y - t * 0.6);
  ctx.stroke();
}

/* ---------- édition de la sélection ---------- */
function marquer() {
  etat.undo.push([...etat.boite]);
  etat.redo = [];
  if (etat.undo.length > 100) etat.undo.shift();
}
function muter() { etat.dirty = true; rendrePanneau(); dessiner(); }

function poigneeProche(pos) {
  if (!etat.boite) return null;
  const seuil = 9 / etat.vue.zoom;
  const pts = poignees(etat.boite);
  for (let i = 0; i < pts.length; i++)
    if (Math.hypot(pts[i][0] - pos[0], pts[i][1] - pos[1]) < seuil) return i;
  return null;
}

const dansBoite = (pos, [x0, y0, x1, y1], marge = 0) =>
  pos[0] >= x0 - marge && pos[0] <= x1 + marge
  && pos[1] >= y0 - marge && pos[1] <= y1 + marge;

function itemSous(pos) {
  /* la plus petite boîte contenant le point (bord toléré) */
  const marge = 4 / etat.vue.zoom;
  let meilleur = null;
  for (const item of [...detections(), ...ajouts()]) {
    if (!etat.montrerDet && item.uid !== etat.selUid) continue;
    if (!item.ajout && !item.retenu && !item.decision && !etat.sousSeuil) continue;
    const b = boiteDe(item);
    if (dansBoite(pos, b, marge)) {
      const aire = (b[2] - b[0]) * (b[3] - b[1]);
      if (!meilleur || aire < meilleur.aire) meilleur = { item, aire };
    }
  }
  return meilleur && meilleur.item;
}

/* ---------- souris ---------- */
let drag = null;
canvas.addEventListener("mousedown", (e) => {
  const pos = depuisEcran([e.offsetX, e.offsetY]);
  if (e.button === 1) {
    e.preventDefault();
    drag = { mode: "pan", x: e.offsetX, y: e.offsetY }; return;
  }
  if (e.button !== 0) return;
  if (etat.ajout) { etat.ajout.b = [pos[0], pos[1], pos[0], pos[1]];
                    drag = { mode: "ajout" }; return; }
  if (etat.boite) {
    const p = poigneeProche(pos);
    if (p !== null) { marquer(); drag = { mode: "poignee", p }; return; }
  }
  const item = itemSous(pos);
  if (item && item.uid !== etat.selUid) {
    if (etat.dirty && !window.confirm("Édition non validée — l'abandonner ?")) return;
    selectionner(item.uid);
    return;
  }
  if (item && etat.boite && dansBoite(pos, etat.boite)) {
    marquer(); drag = { mode: "corps", dernier: pos }; return;
  }
  drag = { mode: "pan", x: e.offsetX, y: e.offsetY };
});
canvas.addEventListener("mousemove", (e) => {
  if (!drag) return;
  if (drag.mode === "pan") {
    etat.vue.px += e.offsetX - drag.x; etat.vue.py += e.offsetY - drag.y;
    drag.x = e.offsetX; drag.y = e.offsetY; dessiner(); return;
  }
  const pos = depuisEcran([e.offsetX, e.offsetY]);
  if (drag.mode === "ajout") {
    etat.ajout.b[2] = pos[0]; etat.ajout.b[3] = pos[1];
    dessiner();
  } else if (drag.mode === "poignee") {
    const axes = POIGNEE_AXES[drag.p];
    if (axes[0]) etat.boite[0] = pos[0];
    if (axes[1]) etat.boite[1] = pos[1];
    if (axes[2]) etat.boite[2] = pos[0];
    if (axes[3]) etat.boite[3] = pos[1];
    muter();
  } else if (drag.mode === "corps") {
    const ddx = pos[0] - drag.dernier[0], ddy = pos[1] - drag.dernier[1];
    etat.boite[0] += ddx; etat.boite[2] += ddx;
    etat.boite[1] += ddy; etat.boite[3] += ddy;
    drag.dernier = pos; muter();
  }
});
window.addEventListener("mouseup", async () => {
  if (drag && drag.mode === "ajout") {
    const b = etat.ajout.b && normaliser(etat.ajout.b);
    etat.ajout = null;
    drag = null;
    if (b && b[2] - b[0] >= 3 && b[3] - b[1] >= 3) await posterAjout(b);
    else { rendrePanneau(); dessiner(); }
    return;
  }
  if (drag && (drag.mode === "poignee" || drag.mode === "corps")) {
    etat.boite = normaliser(etat.boite);
    if (JSON.stringify(etat.boite) === JSON.stringify(etat.undo.at(-1))) {
      etat.undo.pop();
      etat.dirty = etat.undo.length > 0
        && JSON.stringify(etat.boite) !== JSON.stringify(etat.baseBoite);
    }
    dessiner();
  }
  drag = null;
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (drag) return;
  const f = e.deltaY < 0 ? 1.2 : 1 / 1.2;
  const z2 = Math.min(40, Math.max(0.05, etat.vue.zoom * f));
  etat.vue.px = e.offsetX - (e.offsetX - etat.vue.px) * (z2 / etat.vue.zoom);
  etat.vue.py = e.offsetY - (e.offsetY - etat.vue.py) * (z2 / etat.vue.zoom);
  etat.vue.zoom = z2;
  dessiner();
}, { passive: false });

/* ---------- décisions ---------- */
function majCompteTuile() {
  const t = etat.tuiles.find((x) => x.cle === etat.cle);
  if (t) {
    t.restantes = aTraiter().length;
    t.n_ajouts = ajouts().length;
  }
}

async function posterDecision(uid, decision, classe = null, bbox = null) {
  const item = parUid(uid);
  const corps = { uid, decision,
                  classe: classe || (item.decision && item.decision.classe)
                          || item.classe };
  if (bbox) corps.bbox_px = bbox;
  const r = await fetch("/api/decision", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(corps),
  });
  if (!r.ok) { alert("décision refusée : " + (await r.text())); return false; }
  item.decision = { decision, classe: corps.classe };
  if (bbox) item.decision.bbox_px = bbox;
  majCompteTuile();
  return true;
}

async function posterAjout(b) {
  const bbox = b.map((v) => +v.toFixed(1));
  const r = await fetch("/api/ajout", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tuile: etat.cle, classe: etat.classeActive, bbox_px: bbox }),
  });
  const rep = await r.json();
  if (r.ok) {
    etat.detail.ajouts[rep.uid] = { tuile: etat.cle, classe: etat.classeActive,
                                    bbox_px: bbox };
    majCompteTuile();
    selectionner(rep.uid, false);
  }
  await progression();
  rendreListe(); rendrePanneau(); dessiner();
}

async function supprimerAjout(uid) {
  await fetch("/api/ajout", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ supprimer: uid }),
  });
  delete etat.detail.ajouts[uid];
  majCompteTuile();
  selectionner(aTraiter()[0]?.uid || null, false);
  await progression();
  rendreListe(); rendrePanneau(); dessiner();
}

async function decider(decision) {
  const uid = etat.selUid;
  if (!uid) { suivantTuile(1, true); return; }
  const item = parUid(uid);
  if (item.ajout) { // un ajout invalidé = supprimé ; son édition = re-création
    if (decision === "invalide") { await supprimerAjout(uid); return; }
    if (etat.dirty) {
      const bbox = normaliser(etat.boite).map((v) => +v.toFixed(1));
      await fetch("/api/ajout", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ supprimer: uid }) });
      delete etat.detail.ajouts[uid];
      etat.dirty = false;
      await posterAjout(bbox); // re-création avec la boîte éditée
      return;
    }
    selectionner(aTraiter()[0]?.uid || null);
    return;
  }
  const bbox = etat.dirty ? normaliser(etat.boite).map((v) => +v.toFixed(1)) : null;
  const dec = decision === "valide" && bbox ? "editee" : decision;
  if (!(await posterDecision(uid, dec, null, bbox))) return;
  etat.dirty = false;
  await progression();
  const suite = aTraiter()[0];
  if (suite) selectionner(suite.uid);
  else { rendreListe(); suivantTuile(1, true); }
}

async function validerTuile() {
  if (etat.selUid && etat.dirty) await decider("valide");
  for (const d of aTraiter()) await posterDecision(d.uid, "valide");
  await progression();
  rendreListe();
  suivantTuile(1, true);
}

async function reclasser(classe) {
  const item = etat.selUid && parUid(etat.selUid);
  if (!item) { etat.classeActive = classe; rendrePanneau(); return; }
  if (item.ajout) { // pas d'endpoint de reclassement d'ajout : suppr + re-création
    const bbox = item.bbox_px;
    await fetch("/api/ajout", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ supprimer: item.uid }) });
    delete etat.detail.ajouts[item.uid];
    const r = await (await fetch("/api/ajout", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tuile: etat.cle, classe, bbox_px: bbox }) })).json();
    etat.detail.ajouts[r.uid] = { tuile: etat.cle, classe, bbox_px: bbox };
    selectionner(r.uid);
    return;
  }
  const dec = (item.decision && item.decision.decision) || "valide";
  await posterDecision(item.uid, dec, classe,
                       item.decision && item.decision.bbox_px);
  rendrePanneau(); dessiner();
}

function suivantTuile(sens, nonFinie = false) {
  const liste = visibles();
  if (!liste.length) { rendreListe(); return; }
  let i = liste.findIndex((t) => t.cle === etat.cle);
  if (i < 0) i = 0;
  for (let k = 1; k <= liste.length; k++) {
    const cand = liste[(i + sens * k + liste.length * k) % liste.length];
    if (!nonFinie || cand.restantes > 0) { ouvrir(cand.cle); return; }
  }
  rendreListe();
}

/* ---------- clavier ---------- */
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || !etat.detail) return;
  if (e.key === " ") {
    e.preventDefault();
    if (!e.repeat) { etat.masquer = true; dessiner(); }
    return;
  }
  const fleches = { ArrowLeft: [-1, 0], ArrowRight: [1, 0],
                    ArrowUp: [0, -1], ArrowDown: [0, 1] };
  if (e.altKey && fleches[e.key] && etat.boite) {
    e.preventDefault();
    const n = e.shiftKey ? 10 : 1;
    marquer();
    etat.boite[0] += fleches[e.key][0] * n; etat.boite[2] += fleches[e.key][0] * n;
    etat.boite[1] += fleches[e.key][1] * n; etat.boite[3] += fleches[e.key][1] * n;
    muter(); return;
  }
  if (e.ctrlKey && e.key.toLowerCase() === "z") {
    e.preventDefault();
    if (etat.undo.length) {
      etat.redo.push([...etat.boite]);
      etat.boite = etat.undo.pop();
      etat.dirty = JSON.stringify(etat.boite) !== JSON.stringify(etat.baseBoite);
      rendrePanneau(); dessiner();
    }
    return;
  }
  if (e.ctrlKey && e.key.toLowerCase() === "y") {
    if (etat.redo.length) { etat.undo.push([...etat.boite]);
      etat.boite = etat.redo.pop(); muter(); }
    return;
  }
  const num = parseInt(e.key, 10);
  if (num >= 1 && num <= etat.classes.length) { reclasser(etat.classes[num - 1]); return; }
  switch (e.key) {
    case "Enter":
      if (e.shiftKey) validerTuile();
      else decider("valide");
      break;
    case "x": case "X": decider("invalide"); break;
    case "Delete": case "Backspace": {
      const item = etat.selUid && parUid(etat.selUid);
      if (item && item.ajout) supprimerAjout(item.uid);
      else if (item) decider("invalide");
      break;
    }
    case "Tab": {
      e.preventDefault();
      const liste = aTraiter();
      if (!liste.length) break;
      const i = liste.findIndex((d) => d.uid === etat.selUid);
      selectionner(liste[(i + 1) % liste.length].uid);
      break;
    }
    case "a": case "A":
      etat.ajout = { b: null };
      etat.selUid = null; etat.boite = null;
      rendrePanneau(); dessiner(); break;
    case "ArrowRight": case "j": suivantTuile(1); break;
    case "ArrowLeft": case "k": suivantTuile(-1); break;
    case "r": case "R":
      if (!etat.boite) break;
      marquer(); etat.boite = [...etat.baseBoite]; etat.dirty = false;
      rendrePanneau(); dessiner(); break;
    case "g": case "G": etat.montrerGT = !etat.montrerGT; dessiner(); break;
    case "d": case "D": etat.montrerDet = !etat.montrerDet; dessiner(); break;
    case "s": case "S": etat.sousSeuil = !etat.sousSeuil; dessiner(); break;
    case "v": case "V": etat.voisines = !etat.voisines; dessiner(); break;
    case "t": case "T": etat.masquer = !etat.masquer; dessiner(); break;
    case "c": case "C":
      etat.contexte = !etat.contexte;
      ouvrir(etat.cle, true); break;
    case "Escape": {
      if (etat.ajout) { etat.ajout = null; rendrePanneau(); dessiner(); break; }
      if (etat.dirty) {
        etat.boite = [...etat.baseBoite]; etat.dirty = false;
        rendrePanneau(); dessiner(); break;
      }
      if (etat.selUid) { selectionner(null); break; }
      const prec = etat.historique.pop();
      if (prec) ouvrir(prec, true);
      break;
    }
  }
});
window.addEventListener("keyup", (e) => {
  if (e.key === " " && etat.masquer) { etat.masquer = false; dessiner(); }
});

/* ---------- init ---------- */
for (const id of ["#f-split", "#f-zone", "#f-classe"])
  $(id).addEventListener("change", () => chargerListe());
for (const id of ["#f-etat", "#f-tri"])
  $(id).addEventListener("change", () => rendreListe());
new ResizeObserver(() => dessiner()).observe(canvas);
chargerListe();
