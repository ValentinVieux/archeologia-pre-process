/* Revue/édition du recalage — vanilla, aucune dépendance.
   Coordonnées de travail = pixels image du crop ; conversions monde <-> px via
   l'affine renvoyée par /api/crop (en-tête X-Affine). */
"use strict";

const $ = (s) => document.querySelector(s);
const canvas = $("#canvas");
const ctx = canvas.getContext("2d");

const etat = {
  lignes: [], couches: [], idx: -1,
  detail: null, img: null, affine: null, gsd: 0.5,
  vue: { zoom: 1, px: 0, py: 0 },
  parts: null, base: null, editee: false, dirty: false, sel: null,
  undo: [], redo: [], bande: true, masquer: false, voisines: true,
  corrVoisines: true, dessin: null, historique: [],
};

/* une couleur stable par couche (l'active reste verte/jaune, l'origine rouge) */
const PALETTE = [[90, 162, 224], [199, 123, 214], [224, 165, 63],
                 [86, 214, 200], [159, 214, 90], [214, 90, 126]];
const couleurCouche = (couche, a) => {
  const rgb = PALETTE[Math.max(0, etat.couches.indexOf(couche)) % PALETTE.length];
  return `rgba(${rgb.join(",")},${a})`;
};

/* ---------- conversions ---------- */
const versPx = ([x, y]) => [(x - etat.affine[2]) / etat.affine[0],
                            (y - etat.affine[5]) / etat.affine[4]];
const versMonde = ([u, v]) => [etat.affine[0] * u + etat.affine[2],
                               etat.affine[4] * v + etat.affine[5]];
const versEcran = ([u, v]) => [u * etat.vue.zoom + etat.vue.px,
                               v * etat.vue.zoom + etat.vue.py];
const depuisEcran = ([sx, sy]) => [(sx - etat.vue.px) / etat.vue.zoom,
                                   (sy - etat.vue.py) / etat.vue.zoom];
const clone = (parts) => parts.map((p) => p.map((c) => [...c]));

/* ---------- liste ---------- */
async function chargerListe() {
  const statut = $("#f-statut").value;
  const couche = $("#f-couche").value;
  const q = new URLSearchParams({ statut, couche, perimetre: statut ? 0 : 1 });
  const r = await (await fetch("/api/lignes?" + q)).json();
  etat.lignes = r.lignes;
  etat.couches = r.couches;
  $("#titre-zone").textContent = r.zone;
  document.title = `Revue recalage — ${r.zone}`;
  if (!$("#f-couche").options.length || $("#f-couche").options.length === 1) {
    for (const c of r.couches) {
      $("#f-couche").insertAdjacentHTML("beforeend",
        `<option value="${c}">${c}</option>`);
    }
    $("#lg-couches").innerHTML = r.couches.map((c) =>
      `<span class="lg"><span class="chip" style="background:${
        couleurCouche(c, 1)}"></span>${c}</span>`).join("");
  }
  rendreListe();
  await progression();
  const premiere = visibles().findIndex((l) => !l.decision);
  if (etat.lignes.length) ouvrir(premiere >= 0 ? visibles()[premiere].id
                                               : etat.lignes[0].id);
}

const cleId = (id) => {
  const m = id.match(/^(.*)_(\d+)$/);
  return m ? [m[1], +m[2]] : [id, 0];
};

const visibles = () => {
  const mode = $("#f-etat").value;
  let l = etat.lignes;
  if (mode === "a_traiter") l = l.filter((x) => !x.decision);
  else if (mode === "decidees") l = l.filter((x) => x.decision);
  l = [...l];
  if ($("#f-tri").value === "score") l.sort((a, b) => a.score - b.score);
  else l.sort((a, b) => { // ordre stable des entités (numérique, pas lexical)
    const [ca, na] = cleId(a.id), [cb, nb] = cleId(b.id);
    return ca < cb ? -1 : ca > cb ? 1 : na - nb;
  });
  return l;
};

function rendreListe() {
  const courant = etat.detail && etat.detail.id;
  $("#liste").innerHTML = visibles().map((l) => `
    <li data-id="${l.id}" class="${l.decision ? "decidee" : ""}
        ${l.id === courant ? "actif" : ""}">
      <span>${l.id}${l.echantillon ? " ◦" : ""}</span>
      <span class="badge ${l.decision ? "d-" + l.decision : "s-" + l.statut}">
        ${l.decision || l.statut}</span>
      <span class="score">${l.score.toFixed(0)}</span>
    </li>`).join("");
  document.querySelectorAll("#liste li").forEach((li) =>
    li.addEventListener("click", () => ouvrir(li.dataset.id)));
  const actif = $("#liste li.actif");
  if (actif) actif.scrollIntoView({ block: "nearest" });
}

async function progression() {
  const p = await (await fetch("/api/progression")).json();
  $("#progression").textContent =
    `${p.decidees_perimetre}/${p.perimetre} du périmètre — ` +
    Object.entries(p.par_decision).filter(([, n]) => n)
      .map(([d, n]) => `${d} ${n}`).join(", ");
}

/* ---------- ouverture d'une ligne ---------- */
async function ouvrir(id, depuisHistorique = false) {
  if (etat.dirty && etat.detail && id !== etat.detail.id
      && !window.confirm(`Édition non validée sur ${etat.detail.id} `
                         + "(Entrée pour la valider) — l'abandonner ?")) return;
  const precedent = etat.detail && etat.detail.id;
  const [detail, crop] = await Promise.all([
    (await fetch("/api/ligne/" + id)).json(),
    fetch("/api/crop/" + id),
  ]);
  if (!depuisHistorique && precedent && precedent !== id) {
    etat.historique.push(precedent); // Échap = revenir en arrière
    if (etat.historique.length > 50) etat.historique.shift();
  }
  etat.affine = JSON.parse(crop.headers.get("X-Affine"));
  etat.gsd = parseFloat(crop.headers.get("X-Gsd"));
  etat.img = await createImageBitmap(await crop.blob());
  etat.detail = detail;
  etat.base = detail.recale.map((p) => p.map(versPx));
  etat.parts = detail.editee ? detail.editee.map((p) => p.map(versPx))
                             : clone(etat.base);
  etat.chargees = clone(etat.parts); // état à l'ouverture (cible d'Échap)
  etat.editee = !!detail.editee;
  etat.dirty = false;
  etat.sel = null; etat.undo = []; etat.redo = [];
  cadrer();
  rendrePanneau();
  rendreListe();
  dessiner();
}

function cadrer() {
  const z = Math.min(canvas.clientWidth / etat.img.width,
                     canvas.clientHeight / etat.img.height) * 0.95;
  etat.vue.zoom = z;
  etat.vue.px = (canvas.clientWidth - etat.img.width * z) / 2;
  etat.vue.py = (canvas.clientHeight - etat.img.height * z) / 2;
}

/* ---------- mode dessin : re-tracer l'annotation de zéro ---------- */
function commencerDessin() {
  etat.dessin = { pts: [], curseur: null };
  etat.sel = null;
  rendrePanneau();
  dessiner();
}

function terminerDessin() {
  const pts = (etat.dessin ? etat.dessin.pts : []).filter((p, i, arr) =>
    !i || Math.hypot(p[0] - arr[i - 1][0], p[1] - arr[i - 1][1]) > 0.5);
  etat.dessin = null;
  if (pts.length >= 2) { marquer(); etat.parts = [pts]; muter(); }
  else { rendrePanneau(); dessiner(); }
}

function annulerDessin() {
  etat.dessin = null;
  rendrePanneau();
  dessiner();
}

function rendrePanneau() {
  const d = etat.detail;
  $("#p-id").textContent = d.id;
  const m = d.mesures;
  $("#p-mesures").innerHTML = [
    ["statut", d.statut], ["polarité", m.polarite_retenue],
    ["pts nets", m.pts_nets_pct + " %"], ["ambigus", m.ambigus_pct + " %"],
    ["contraste", m.contraste], ["offset médian", m.offset_median_m + " m"],
    ["offset max", m.offset_max_m + " m"], ["résidu", m.residu_m + " m"],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  $("#p-decision").textContent = etat.dessin
    ? "mode dessin — clic : sommet · Entrée : terminer · Échap : annuler"
    : (etat.dirty ? "édition en cours — Entrée pour valider"
                  : (d.decision ? "décision : " + d.decision : ""));
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

  if (!etat.masquer) {
    ctx.save(); // les superpositions restent dans l'emprise du crop
    ctx.beginPath();
    ctx.rect(0, 0, etat.img.width, etat.img.height);
    ctx.clip();
    const active = etat.parts;
    if (etat.voisines) { // contexte : les autres lignes du crop, par couche
      for (const v of etat.detail.voisines || []) {
        if (etat.corrVoisines && v.origine) // leur origine, en tirets
          tracer(v.origine.map((p) => p.map(versPx)),
                 couleurCouche(v.couche, 0.45), 1.2 / zoom, false,
                 [6 / zoom, 5 / zoom]);
        tracer(v.parts.map((p) => p.map(versPx)),
               couleurCouche(v.couche, 0.8), 1.5 / zoom);
      }
    }
    if (etat.dessin) { // re-tracé de zéro : l'existant en retrait, dessin jaune
      tracer(etat.detail.origine.map((p) => p.map(versPx)),
             "rgba(224,85,85,.5)", 1.5 / zoom);
      tracer(active, "rgba(79,195,107,.35)", 1.5 / zoom);
      const pts = etat.dessin.pts;
      const apercu = (etat.dessin.curseur && pts.length)
        ? [...pts, etat.dessin.curseur] : pts;
      if (apercu.length >= 2) {
        if (etat.bande) tracer([apercu], "rgba(232,201,62,.28)",
                               7 / etat.gsd, true);
        tracer([apercu], "#e8c93e", 2 / zoom);
      }
      const t = 7 / zoom;
      ctx.fillStyle = "#e8c93e";
      pts.forEach(([u, v]) => ctx.fillRect(u - t / 2, v - t / 2, t, t));
    } else {
      if (etat.bande) { // bande de 7 m : trait continu, bouts ronds = buffer
        tracer(active, `rgba(${etat.editee ? "232,201,62" : "79,195,107"},.28)`,
               7 / etat.gsd, true);
      }
      tracer(etat.detail.origine.map((p) => p.map(versPx)), "#e05555", 2 / zoom);
      if (!etat.editee) tracer(active, "#4fc36b", 2 / zoom);
      else { tracer(etat.base, "rgba(79,195,107,.5)", 1.5 / zoom);
             tracer(active, "#e8c93e", 2 / zoom); }

      const taille = 7 / zoom; // poignées
      active.forEach((part, ip) => part.forEach(([u, v], iv) => {
        const s = etat.sel && etat.sel.p === ip && etat.sel.v === iv;
        ctx.fillStyle = s ? "#ffffff" : (etat.editee ? "#e8c93e" : "#4fc36b");
        ctx.fillRect(u - taille / 2, v - taille / 2, taille, taille);
      }));
    }
    ctx.restore();
  }
  $("#hud").textContent =
    `${etat.detail.id} — zoom ${(zoom).toFixed(1)}× — ${etat.gsd} m/px` +
    (etat.masquer ? " — LD à nu" : "");
}

function tracer(parts, style, largeur, ronde = false, tirets = null) {
  ctx.strokeStyle = style;
  ctx.lineWidth = largeur;
  ctx.lineCap = ronde ? "round" : "butt";
  ctx.lineJoin = "round";
  ctx.setLineDash(tirets || []);
  for (const part of parts) {
    ctx.beginPath();
    part.forEach(([u, v], i) => (i ? ctx.lineTo(u, v) : ctx.moveTo(u, v)));
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

/* ---------- édition ---------- */
function marquer() { // à appeler AVANT toute mutation de etat.parts
  etat.undo.push(clone(etat.parts));
  etat.redo = [];
  if (etat.undo.length > 100) etat.undo.shift();
}

function muter() {
  etat.editee = true;
  etat.dirty = true;
  rendrePanneau();
  dessiner();
}

function sommetProche(pos) {
  const seuil = 9 / etat.vue.zoom;
  for (let ip = 0; ip < etat.parts.length; ip++) {
    const part = etat.parts[ip];
    for (let iv = 0; iv < part.length; iv++) {
      if (Math.hypot(part[iv][0] - pos[0], part[iv][1] - pos[1]) < seuil)
        return { p: ip, v: iv };
    }
  }
  return null;
}

function segmentProche(pos) {
  // saisie sur toute la bande de 7 m (3,5 m de part et d'autre de l'axe),
  // jamais moins de 10 px écran : cliquer l'annotation ne doit pas panner
  const seuil = Math.max(10 / etat.vue.zoom, 3.5 / etat.gsd);
  let meilleur = null;
  etat.parts.forEach((part, ip) => {
    for (let i = 0; i < part.length - 1; i++) {
      const [ax, ay] = part[i], [bx, by] = part[i + 1];
      const l2 = (bx - ax) ** 2 + (by - ay) ** 2;
      const t = l2 ? Math.max(0, Math.min(1,
        ((pos[0] - ax) * (bx - ax) + (pos[1] - ay) * (by - ay)) / l2)) : 0;
      const qx = ax + t * (bx - ax), qy = ay + t * (by - ay);
      const d = Math.hypot(pos[0] - qx, pos[1] - qy);
      if (d < seuil && (!meilleur || d < meilleur.d))
        meilleur = { p: ip, i, t, q: [qx, qy], d };
    }
  });
  return meilleur;
}

/* ---------- souris ---------- */
let drag = null; // {mode: 'pan'|'sommet'|'ligne', ...}
canvas.addEventListener("mousedown", (e) => {
  const pos = depuisEcran([e.offsetX, e.offsetY]);
  if (e.button === 1) { // preventDefault : sinon l'autoscroll du navigateur
    e.preventDefault();  // émet des wheel pendant le drag (= dézoom parasite)
    drag = { mode: "pan", x: e.offsetX, y: e.offsetY }; return; }
  if (e.button !== 0) return;
  if (etat.dessin) { etat.dessin.pts.push(pos); dessiner(); return; }
  const s = sommetProche(pos);
  if (s) { etat.sel = s; marquer(); drag = { mode: "sommet" }; dessiner(); return; }
  const seg = segmentProche(pos);
  if (seg) { marquer(); drag = { mode: "ligne", p: seg.p, dernier: pos }; return; }
  drag = { mode: "pan", x: e.offsetX, y: e.offsetY };
});
canvas.addEventListener("mousemove", (e) => {
  if (etat.dessin && !drag) { // aperçu élastique du prochain segment
    etat.dessin.curseur = depuisEcran([e.offsetX, e.offsetY]);
    dessiner();
    return;
  }
  if (!drag) return;
  if (drag.mode === "pan") {
    etat.vue.px += e.offsetX - drag.x; etat.vue.py += e.offsetY - drag.y;
    drag.x = e.offsetX; drag.y = e.offsetY; dessiner(); return;
  }
  const pos = depuisEcran([e.offsetX, e.offsetY]);
  if (drag.mode === "sommet") {
    etat.parts[etat.sel.p][etat.sel.v] = pos; muter();
  } else if (drag.mode === "ligne") {
    const dx = pos[0] - drag.dernier[0], dy = pos[1] - drag.dernier[1];
    for (const c of etat.parts[drag.p]) { c[0] += dx; c[1] += dy; }
    drag.dernier = pos; muter();
  }
});
window.addEventListener("mouseup", () => {
  if (drag && drag.mode !== "pan"
      && JSON.stringify(etat.parts) === JSON.stringify(etat.undo.at(-1)))
    etat.undo.pop(); // clic sans déplacement : pas d'entrée d'undo
  drag = null;
});
function voisineProche(pos) {
  const seuil = 8 / etat.vue.zoom;
  for (const v of (etat.voisines && etat.detail.voisines) || []) {
    for (const part of v.parts) {
      const px = part.map(versPx);
      for (let i = 0; i < px.length - 1; i++) {
        const [ax, ay] = px[i], [bx, by] = px[i + 1];
        const l2 = (bx - ax) ** 2 + (by - ay) ** 2;
        const t = l2 ? Math.max(0, Math.min(1,
          ((pos[0] - ax) * (bx - ax) + (pos[1] - ay) * (by - ay)) / l2)) : 0;
        if (Math.hypot(pos[0] - (ax + t * (bx - ax)),
                       pos[1] - (ay + t * (by - ay))) < seuil) return v.id;
      }
    }
  }
  return null;
}

canvas.addEventListener("dblclick", (e) => {
  if (etat.dessin) { terminerDessin(); return; }
  const pos = depuisEcran([e.offsetX, e.offsetY]);
  const seg = segmentProche(pos);
  if (seg) { // sur la ligne active : insertion de sommet
    marquer();
    etat.parts[seg.p].splice(seg.i + 1, 0, seg.q);
    etat.sel = { p: seg.p, v: seg.i + 1 };
    muter();
    return;
  }
  const id = voisineProche(pos); // sur une voisine : on la révise
  if (id) ouvrir(id);
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (drag) return; // pas de zoom pendant un drag (pan molette compris)
  const f = e.deltaY < 0 ? 1.2 : 1 / 1.2;
  const z2 = Math.min(40, Math.max(0.05, etat.vue.zoom * f));
  etat.vue.px = e.offsetX - (e.offsetX - etat.vue.px) * (z2 / etat.vue.zoom);
  etat.vue.py = e.offsetY - (e.offsetY - etat.vue.py) * (z2 / etat.vue.zoom);
  etat.vue.zoom = z2;
  dessiner();
}, { passive: false });

/* ---------- décisions ---------- */
async function poster(id, decision, geometrie = null) {
  const corps = { id, decision };
  if (geometrie) corps.geometrie = geometrie;
  await fetch("/api/decision", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(corps),
  });
  const l = etat.lignes.find((x) => x.id === id);
  if (l) l.decision = decision;
}

/* voisines du crop encore à traiter (périmètre de revue, sans décision) */
const voisinesAValider = () => (etat.detail.voisines || []).filter((v) =>
  !v.decision && (v.statut === "a_revoir" || v.echantillon));

async function decider(decision) {
  await poster(etat.detail.id, decision,
               decision === "editee" ? etat.parts.map((p) => p.map(versMonde))
                                     : null);
  etat.detail.decision = decision;
  etat.dirty = false; // l'édition est actée : plus rien à perdre en naviguant
  await progression();
  const suite = voisinesAValider(); // on reste dans l'image tant qu'il en reste
  if (suite.length) ouvrir(suite[0].id);
  else suivant(1, true);
}

/* Shift+Entrée : valide toutes les entités restantes de l'image, puis suivante */
async function validerImage() {
  if (!etat.detail.decision)
    await poster(etat.detail.id, etat.dirty ? "editee" : "recale",
                 etat.dirty ? etat.parts.map((p) => p.map(versMonde)) : null);
  for (const v of voisinesAValider()) {
    await poster(v.id, "recale");
    v.decision = "recale";
  }
  etat.dirty = false;
  await progression();
  suivant(1, true);
}

function suivant(sens, nonDecidee = false) {
  const liste = visibles();
  if (!liste.length) { rendreListe(); return; }
  let i = liste.findIndex((l) => l.id === etat.detail.id);
  if (i < 0) i = 0;
  for (let k = 1; k <= liste.length; k++) {
    const cand = liste[(i + sens * k + liste.length * k) % liste.length];
    if (!nonDecidee || !cand.decision) { ouvrir(cand.id); return; }
  }
  rendreListe();
}

/* ---------- clavier ---------- */
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || !etat.detail) return;
  if (e.key === " ") { // maintenu = LD à nu, relâché = tracés de retour
    e.preventDefault();
    if (!e.repeat) { etat.masquer = true; dessiner(); }
    return;
  }
  if (etat.dessin) { // en mode dessin, le clavier sert au tracé
    if (e.key === "Enter") { e.preventDefault(); terminerDessin(); }
    else if (e.key === "Escape") annulerDessin();
    else if (e.key === "Delete" || e.key === "Backspace") {
      etat.dessin.pts.pop(); dessiner();
    } else if ("bB".includes(e.key)) { etat.bande = !etat.bande; dessiner(); }
    else if ("vV".includes(e.key)) { etat.voisines = !etat.voisines; dessiner(); }
    else if ("cC".includes(e.key)) { etat.corrVoisines = !etat.corrVoisines; dessiner(); }
    else if ("tT".includes(e.key)) { etat.masquer = !etat.masquer; dessiner(); }
    else if ("dD".includes(e.key)) annulerDessin();
    return;
  }
  const fleches = { ArrowLeft: [-1, 0], ArrowRight: [1, 0],
                    ArrowUp: [0, -1], ArrowDown: [0, 1] };
  if (e.altKey && fleches[e.key]) { // nudge de la ligne entière (geste Historydex)
    e.preventDefault();
    const n = e.shiftKey ? 10 : 1;
    marquer();
    for (const part of etat.parts)
      for (const c of part) { c[0] += fleches[e.key][0] * n; c[1] += fleches[e.key][1] * n; }
    muter(); return;
  }
  if (e.ctrlKey && e.key.toLowerCase() === "z") {
    e.preventDefault();
    if (etat.undo.length) {
      etat.redo.push(clone(etat.parts));
      etat.parts = etat.undo.pop();
      etat.editee = JSON.stringify(etat.parts) !== JSON.stringify(etat.base);
      etat.dirty = etat.editee;
      rendrePanneau(); dessiner();
    }
    return;
  }
  if (e.ctrlKey && e.key.toLowerCase() === "y") {
    if (etat.redo.length) { etat.undo.push(clone(etat.parts));
      etat.parts = etat.redo.pop(); muter(); }
    return;
  }
  switch (e.key) {
    case "Enter":
      if (e.shiftKey) validerImage();
      else decider(etat.editee ? "editee" : "recale");
      break;
    case "o": case "O": decider("original"); break;
    case "x": case "X": decider("exclue"); break;
    case "ArrowRight": case "j": suivant(1); break;
    case "ArrowLeft": case "k": suivant(-1); break;
    case "Delete": case "Backspace": {
      if (!etat.sel) break;
      const part = etat.parts[etat.sel.p];
      if (part.length <= 2) break; // une ligne garde au moins 2 sommets
      marquer(); part.splice(etat.sel.v, 1);
      etat.sel = null; muter(); break;
    }
    case "r": case "R":
      marquer(); etat.parts = clone(etat.base); etat.editee = false;
      etat.dirty = false; // reset volontaire : rien à protéger
      rendrePanneau(); dessiner(); break;
    case "b": case "B": etat.bande = !etat.bande; dessiner(); break;
    case "t": case "T": etat.masquer = !etat.masquer; dessiner(); break;
    case "v": case "V": etat.voisines = !etat.voisines; dessiner(); break;
    case "c": case "C": etat.corrVoisines = !etat.corrVoisines; dessiner(); break;
    case "d": case "D": commencerDessin(); break;
    case "Escape": {
      if (etat.dirty) { // édition en cours : Échap l'annule (Ctrl+Z la rend)
        marquer();
        etat.parts = clone(etat.chargees);
        etat.editee = JSON.stringify(etat.parts) !== JSON.stringify(etat.base);
        etat.dirty = false;
        etat.sel = null;
        rendrePanneau();
        dessiner();
        break;
      }
      let prec = etat.historique.pop(); // sinon : retour à l'entité précédente
      while (prec === etat.detail.id) prec = etat.historique.pop();
      if (prec) ouvrir(prec, true);
      break;
    }
  }
});
window.addEventListener("keyup", (e) => {
  if (e.key === " " && etat.masquer) { etat.masquer = false; dessiner(); }
});

/* ---------- init ---------- */
for (const id of ["#f-statut", "#f-couche"])
  $(id).addEventListener("change", chargerListe);
for (const id of ["#f-etat", "#f-tri"])
  $(id).addEventListener("change", () => { rendreListe(); });
new ResizeObserver(() => dessiner()).observe(canvas);
chargerListe();
