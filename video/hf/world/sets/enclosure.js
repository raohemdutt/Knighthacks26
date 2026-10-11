// The Atlas enclosure, dimensionally true (world/ENCLOSURE.md; 1 unit = 100 mm). Every part is its own Group so it
// can move: explode, x-ray, highlight, drop. buildEnclosure() is the hero; buildEnclosureLite() is the shell-only copy
// used for the friends' boxes and the kitchen device. Pure functions of state and t.
import * as THREE from "three";
import {
  SURFACE_3D, SHADE, LIGHT, INK, ACCENT, MUTED, ENV_INTENSITY, DESK_Y, hash, clamp01, heartbeat,
  mm, mmBox, boxMesh, softTexture, FONT, flat, matte, RoundedBoxGeometry,
} from "./common.js";

const KRAFT = 0xddd2c2;     // cardboard: a warm matte kraft-grey close to the surface colour
const PRINT = 0xe3e9f1;     // printed parts (lid, front, rail, clip): the soft white surface
const FABRIC = 0x4f5663;    // the speaker's fabric front
const CAM_BODY = 0x4a515c;  // the webcam housing
const BOARD = 0x23382f;     // the Jetson carrier board, green-black
const SINK = 0x2c3139;      // anodised heat sink
const METAL = 0xbcc3cc;     // port shells

// ---- Textures (built once) ----
let _tex = null;
function textures() {
  if (_tex) return _tex;
  const kraft = softTexture((ctx, w, h) => {
    ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 90; i++) { // the faintest flute streaks and fibres: matte, not wood
      const x = hash(i, 71) * w, a = 0.006 + hash(i, 72) * 0.014;
      ctx.fillStyle = `rgba(150,130,100,${a})`; ctx.fillRect(x, 0, 2 + hash(i, 73) * 4, h);
    }
    for (let i = 0; i < 700; i++) { ctx.fillStyle = `rgba(140,120,95,${0.02 + hash(i, 75) * 0.04})`; ctx.fillRect(hash(i, 76) * w, hash(i, 77) * h, 2, 1); }
  }, 256, 256);
  kraft.wrapS = kraft.wrapT = THREE.RepeatWrapping;
  const fabric = softTexture((ctx, w, h) => {
    ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "rgba(0,0,0,0.10)";
    for (let y = 0; y < h; y += 4) ctx.fillRect(0, y, w, 1.5);
    for (let x = 0; x < w; x += 4) ctx.fillRect(x, 0, 1.5, h);
  }, 128, 128);
  fabric.wrapS = fabric.wrapT = THREE.RepeatWrapping; fabric.repeat.set(10, 6);
  // The raised "Atlas" logo, printed white: the word and a soft bevel shadow drawn into one map.
  const logo = softTexture((ctx, w, h) => {
    ctx.clearRect(0, 0, w, h);
    ctx.font = `800 ${h * 0.78}px ${FONT}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(112,122,138,0.95)"; ctx.fillText("Atlas", w / 2 + 9, h / 2 + 12); // the raised edge's shadow
    ctx.fillStyle = "rgba(255,255,255,1)"; ctx.fillText("Atlas", w / 2 - 3, h / 2 - 4);
    ctx.fillStyle = "rgba(205,212,222,1)"; ctx.fillText("Atlas", w / 2 + 2, h / 2 + 3);   // the face, a touch below white
    ctx.fillStyle = "#ffffff"; ctx.fillText("Atlas", w / 2, h / 2);
  }, 1024, 320);
  _tex = { kraft, fabric, logo };
  return _tex;
}

// ---- Geometry (built once, shared by the hero and the copies) ----
const roundRect = (path, x0, y0, x1, y1, r) => {
  r = Math.min(r, (x1 - x0) / 2, (y1 - y0) / 2);
  path.moveTo(x0 + r, y0); path.lineTo(x1 - r, y0); path.quadraticCurveTo(x1, y0, x1, y0 + r);
  path.lineTo(x1, y1 - r); path.quadraticCurveTo(x1, y1, x1 - r, y1); path.lineTo(x0 + r, y1);
  path.quadraticCurveTo(x0, y1, x0, y1 - r); path.lineTo(x0, y0 + r); path.quadraticCurveTo(x0, y0, x0 + r, y0);
  return path;
};
const roundRectHole = (x0, y0, x1, y1, r) => {
  // Holes wind the other way: build the same rounded rect clockwise.
  const p = new THREE.Path(); r = Math.min(r, (x1 - x0) / 2, (y1 - y0) / 2);
  p.moveTo(x0 + r, y0); p.quadraticCurveTo(x0, y0, x0, y0 + r); p.lineTo(x0, y1 - r); p.quadraticCurveTo(x0, y1, x0 + r, y1);
  p.lineTo(x1 - r, y1); p.quadraticCurveTo(x1, y1, x1, y1 - r); p.lineTo(x1, y0 + r); p.quadraticCurveTo(x1, y0, x1 - r, y0); p.lineTo(x0 + r, y0);
  return p;
};
const BEVEL = 0.004; // 0.4 mm soft edge on printed parts

let _geo = null;
function geometry() {
  if (_geo) return _geo;
  // Front panel, 125 x 79.5 x 4 mm (read as X x Z x thickness), honeycomb through-holes, a solid badge for the logo.
  // Shape space = world (x, y); extruded along +z from the panel's back face.
  const U = (X) => (X - 62.5) / 100, V = (Z) => DESK_Y + Z / 100;
  const fz0 = 0.5, fz1 = 80;
  const front = roundRect(new THREE.Shape(), U(0) + BEVEL, V(fz0) + BEVEL, U(125) - BEVEL, V(fz1) - BEVEL, 0.03);
  const badge = { x0: 33, x1: 92, z0: 30, z1: 51 };
  const Rh = 3.5, cell = 4.35, dx = Math.sqrt(3) * cell, dy = 1.5 * cell;
  let holes = 0;
  for (let row = 0; ; row++) {
    const zc = 7.5 + row * dy; if (zc > fz1 - 7) break;
    for (let col = -1; ; col++) {
      const xc = 7.5 + col * dx + (row % 2 ? dx / 2 : 0); if (xc > 125 - 6.5) break;
      if (xc < 6.5) continue;
      if (xc > badge.x0 - 4 && xc < badge.x1 + 4 && zc > badge.z0 - 4 && zc < badge.z1 + 4) continue;
      const p = new THREE.Path();
      for (let k = 6; k >= 0; k--) { const a = (k / 6) * Math.PI * 2 + Math.PI / 6; const X = xc + Math.cos(a) * Rh, Z = zc + Math.sin(a) * Rh; k === 6 ? p.moveTo(U(X), V(Z)) : p.lineTo(U(X), V(Z)); }
      front.holes.push(p); holes++;
    }
  }
  const frontGeo = new THREE.ExtrudeGeometry(front, { depth: 0.04 - 2 * BEVEL, bevelEnabled: true, bevelThickness: BEVEL, bevelSize: BEVEL * 0.6, bevelSegments: 2, curveSegments: 4 });
  frontGeo.translate(0, 0, (99 - 0.5) / 100 + BEVEL); // back face at Y = 0.5 (z = 0.985), front face proud of the box
  frontGeo.computeVertexNormals();

  // Top lid, 130.5 x 206.5 x 3, overhanging; socket opening 39.5 x 48.5 with a 22 x 22 cable gap on the device's right,
  // and a vent grille over the Jetson. Shape space (u, v) = (x, -z); rotated flat and lifted to Z = 80.
  const LU = (X) => (X - 62.5) / 100, LV = (Y) => (Y - 99) / 100;
  const lid = roundRect(new THREE.Shape(), LU(-2.75) + BEVEL, LV(-4.25) + BEVEL, LU(127.75) - BEVEL, LV(202.25) - BEVEL, 0.06);
  const sock = { x0: 42.75, x1: 82.25, y0: 7.9, y1: 56.4 };
  { // socket + cable gap as one L-shaped hole (clockwise)
    const p = new THREE.Path(), gy0 = 21.15, gy1 = 43.15, gx0 = 20.75;
    const pts = [[sock.x0, sock.y0], [sock.x0, gy0], [gx0, gy0], [gx0, gy1], [sock.x0, gy1], [sock.x0, sock.y1], [sock.x1, sock.y1], [sock.x1, sock.y0]];
    // Shape v = (Y - 99)/100 flips with -z; keep the winding opposite to the outline.
    pts.slice().reverse().forEach(([X, Y], i) => (i ? p.lineTo(LU(X), LV(Y)) : p.moveTo(LU(X), LV(Y))));
    p.lineTo(LU(sock.x1), LV(sock.y0));
    lid.holes.push(p);
  }
  const vent = { x0: 24, x1: 101, y0: 90, y1: 154, n: 9 };
  for (let k = 0; k < vent.n; k++) {
    const yc = vent.y0 + (k + 0.5) * ((vent.y1 - vent.y0) / vent.n);
    lid.holes.push(roundRectHole(LU(vent.x0), LV(yc - 1.9), LU(vent.x1), LV(yc + 1.9), 0.019));
  }
  const lidGeo = new THREE.ExtrudeGeometry(lid, { depth: 0.03 - 2 * BEVEL, bevelEnabled: true, bevelThickness: BEVEL, bevelSize: BEVEL * 0.6, bevelSegments: 2, curveSegments: 5 });
  lidGeo.rotateX(-Math.PI / 2); lidGeo.translate(0, DESK_Y + 0.80 + BEVEL, 0); lidGeo.computeVertexNormals();

  // Fan frame: a 40 x 40 square with a round throat.
  const ff = roundRect(new THREE.Shape(), -0.2, -0.2, 0.2, 0.2, 0.03);
  const throat = new THREE.Path(); throat.absarc(0, 0, 0.18, 0, Math.PI * 2, true); ff.holes.push(throat);
  const fanFrameGeo = new THREE.ExtrudeGeometry(ff, { depth: 0.07, bevelEnabled: false, curveSegments: 24 });
  fanFrameGeo.rotateX(-Math.PI / 2);

  _geo = { frontGeo, lidGeo, fanFrameGeo, holes, vent, sock, badge };
  return _geo;
}

// ---- Highlight rims: an inverted hull (back faces, accent, not tone mapped) a hair larger than the mesh's box ----
const RIM = 0.008;
function rimFor(mesh, mat) {
  mesh.geometry.computeBoundingBox();
  const b = mesh.geometry.boundingBox, s = b.getSize(new THREE.Vector3()), c = b.getCenter(new THREE.Vector3());
  const r = Math.min(0.03, Math.min(s.x, s.y, s.z) / 2);
  const hull = new THREE.Mesh(new RoundedBoxGeometry(s.x + 2 * RIM, s.y + 2 * RIM, s.z + 2 * RIM, 2, r + RIM * 0.5), mat);
  hull.position.copy(c); hull.renderOrder = 4; hull.castShadow = false; hull.receiveShadow = false;
  mesh.add(hull); return hull;
}

/**
 * The hero enclosure. Returns { group, parts, P, at(name, e), update(state, t) }.
 * group sits at the origin (world frame = ENCLOSURE.md frame); the scene adds it.
 */
export function buildEnclosure() {
  const T = textures(), G = geometry();
  const group = new THREE.Group(); group.name = "atlas";
  const parts = {};
  // A part: its own Group, pivot at its rest centre (so drop can tumble it), materials it owns (tint), glassy or not.
  function part(name, centerMm, opts = {}) {
    const g = new THREE.Group(); g.name = name; const c = mm(...centerMm); g.position.copy(c); group.add(g);
    const p = { name, g, rest: c.clone(), mats: [], rimMat: null, rims: [], glassy: !!opts.glassy, glassMeshes: [], edges: [], offset: new THREE.Vector3(...(opts.explode || [0, 0, 0])), dropLag: opts.lag || 0 };
    parts[name] = p; return p;
  }
  const own = (p, m) => { m.userData.base = m.color.clone(); p.mats.push(m); return m; };
  const put = (p, mesh, glass = false) => {
    mesh.position.sub(p.rest); p.g.add(mesh);
    if (glass) p.glassMeshes.push(mesh);
    return mesh;
  };
  const rim = (p, mesh) => { if (!p.rimMat) p.rimMat = flat({ color: ACCENT, side: THREE.BackSide, transparent: true, opacity: 0, depthWrite: false }); p.rims.push(rimFor(mesh, p.rimMat)); };
  const kraftMat = (p) => own(p, matte({ color: KRAFT, map: T.kraft, roughness: 0.95, transparent: true }));
  const printMat = (p, extra = {}) => own(p, matte({ color: PRINT, roughness: 0.7, ...extra }));

  // ---- Cardboard shell (1.3 mm panels) ----
  const bottom = part("bottom", [62.5, 99, 0.65], { lag: 0.0 });
  put(bottom, boxMesh([125, 198, 1.3], [0, 0, 0], kraftMat(bottom)));
  const wallR = part("wallR", [0.65, 99, 40.65], { explode: [-0.78, 0, 0], glassy: true, lag: 0.08 }); // device right = screen-left
  put(wallR, boxMesh([1.3, 198, 78.7], [0, 0, 1.3], kraftMat(wallR)), true);
  const wallL = part("wallL", [124.35, 99, 40.65], { explode: [0.78, 0, 0], glassy: true, lag: 0.12 });
  put(wallL, boxMesh([1.3, 198, 78.7], [123.7, 0, 1.3], kraftMat(wallL)), true);
  const back = part("back", [62.5, 197.35, 40.65], { explode: [0, 0, -0.9], glassy: true, lag: 0.04 });
  put(back, boxMesh([122.4, 1.3, 78.7], [1.3, 196.7, 1.3], kraftMat(back)), true);
  // The Ethernet jack on the back wall's outer face, behind the Jetson's port.
  const jackMat = own(back, matte({ color: INK, roughness: 0.5 }));
  put(back, boxMesh([17, 3, 14.5], [60.5, 198, 9.5], jackMat, 1.2));
  put(back, boxMesh([12, 1, 8.5], [63, 200.6, 12], own(back, matte({ color: 0x15181d, roughness: 0.9 }))));

  // ---- Printed parts ----
  const rail = part("rail", [62.5, -2.5, 1.25], { explode: [0, -0.02, 0.9], lag: 0.02 });
  put(rail, boxMesh([125, 6, 2.5], [0, -5.5, 0], printMat(rail), 1));
  const front = part("front", [62.5, -1.5, 40.25], { explode: [0, 0, 1.05], glassy: true, lag: 0.55 });
  const frontMat = printMat(front, { transparent: true, side: THREE.DoubleSide });
  const panel = new THREE.Mesh(G.frontGeo, frontMat); panel.castShadow = true; panel.receiveShadow = true; put(front, panel, true);
  // The raised "Atlas" logo on the badge: a printed-white word with its bevel shadow.
  const logoMat = own(front, matte({ color: 0xffffff, map: T.logo, transparent: true, alphaTest: 0.04, roughness: 0.55 }));
  const logo = new THREE.Mesh(new THREE.PlaneGeometry(0.56, 0.175), logoMat);
  logo.position.set(0, DESK_Y + 0.405, 1.025 + 0.0012); put(front, logo);
  // Thin accent frame around the panel (the front's highlight rim: a hull would show through the honeycomb).
  front.rimMat = flat({ color: ACCENT, transparent: true, opacity: 0, depthWrite: false });
  {
    const w = 1.25 + 2 * RIM, h = 0.795 + 2 * RIM, t = RIM * 1.4, zf = 1.026, yc = DESK_Y + 0.4025;
    for (const [sx, sy, px, py] of [[w, t, 0, h / 2], [w, t, 0, -h / 2], [t, h, w / 2, 0], [t, h, -w / 2, 0]]) {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, 0.002), front.rimMat); bar.position.set(px, yc + py, zf); bar.renderOrder = 4; put(front, bar); front.rims.push(bar);
    }
  }

  // Lid with vent grille and socket; the socket walls hang below it, shortened so they stop above the speaker.
  const lid = part("lid", [62.5, 99, 81.5], { explode: [0, 0.85, 0], glassy: true, lag: 0.7 });
  const lidMat = printMat(lid, { transparent: true });
  const lidMesh = new THREE.Mesh(G.lidGeo, lidMat); lidMesh.castShadow = true; lidMesh.receiveShadow = true; put(lid, lidMesh, true);
  rim(lid, lidMesh);
  { const s = G.sock, z0 = 73.5, hz = 80 - z0, t = 1.2;
    for (const [size, min] of [[[s.x1 - s.x0 + 2 * t, t, hz], [s.x0 - t, s.y0 - t, z0]], [[s.x1 - s.x0 + 2 * t, t, hz], [s.x0 - t, s.y1, z0]], [[t, s.y1 - s.y0, hz], [s.x0 - t, s.y0, z0]], [[t, s.y1 - s.y0, hz], [s.x1, s.y0, z0]]])
      put(lid, boxMesh(size, min, lidMat), true);
  }
  // The heartbeat seen from outside: a warm glow under the vent slots, over the fan.
  const ventGlow = new THREE.Mesh(new THREE.PlaneGeometry((G.vent.x1 - G.vent.x0) / 100, (G.vent.y1 - G.vent.y0) / 100), flat({ color: ACCENT, transparent: true, opacity: 0, depthWrite: false }));
  ventGlow.rotation.x = -Math.PI / 2; ventGlow.position.copy(mm((G.vent.x0 + G.vent.x1) / 2, (G.vent.y0 + G.vent.y1) / 2, 79.2)); put(lid, ventGlow);

  // ---- The speaker (onn, 121 x 65 x 70): soft dark body, fabric front, a faint driver ring ----
  const speaker = part("speaker", [62.5, 33, 37.9], { explode: [0, -0.14, 0.58], lag: 0.3 });
  const spkBody = own(speaker, matte({ color: 0x5a616d, roughness: 0.8 }));
  const spk = put(speaker, boxMesh([121, 65, 70], [2, 0.5, 2.9], spkBody, 7, 3)); rim(speaker, spk);
  const fabricMat = own(speaker, matte({ color: FABRIC, map: T.fabric, roughness: 1 }));
  const grille = new THREE.Mesh(new THREE.PlaneGeometry(1.13, 0.62), fabricMat); grille.position.copy(mm(62.5, 0.4, 37.9)); put(speaker, grille);
  const ringMat = own(speaker, matte({ color: 0x434955, roughness: 0.9 }));
  const driver = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.215, 48), ringMat); driver.position.copy(mm(62.5, 0.3, 37.9)); put(speaker, driver);

  // ---- The Jetson Orin Nano dev kit: carrier board, module, finned heat sink, fan, ports on the back edge ----
  const jetson = part("jetson", [62.5, 121.45, 20.3], { explode: [0, 0.46, -0.36], lag: 0.4 });
  const boardMat = own(jetson, matte({ color: BOARD, roughness: 0.6, envMapIntensity: 0.5 }));
  const darkMat = own(jetson, matte({ color: 0x1b1f25, roughness: 0.55 }));
  const sinkMat = own(jetson, matte({ color: SINK, roughness: 0.5, metalness: 0.35, envMapIntensity: 0.6 }));
  const metalMat = own(jetson, matte({ color: METAL, roughness: 0.35, metalness: 0.6, envMapIntensity: 0.8 }));
  const board = put(jetson, boxMesh([100, 79, 1.6], [12.5, 80, 8], boardMat, 0.6)); rim(jetson, board);
  for (const [X, Y] of [[16.5, 84], [108.5, 84], [16.5, 155], [108.5, 155]]) {
    const so = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.051, 10), metalMat); so.position.copy(mm(X, Y, 2.9 + 2.55)); put(jetson, so);
  }
  put(jetson, boxMesh([69.6, 45, 4], [27.7, 95, 9.6], darkMat, 0.6));             // the module
  put(jetson, boxMesh([51, 5, 8.5], [37, 81, 9.6], darkMat, 0.4));                 // 40-pin header
  for (let k = 0; k < 4; k++) put(jetson, boxMesh([7 + hash(k, 41) * 6, 7, 1.2], [16 + k * 22, 145, 9.6], darkMat, 0.3)); // small chips
  // Ports along the back edge (facing the back wall): 4 USB-A, Ethernet, DisplayPort, the barrel jack.
  put(jetson, boxMesh([14.5, 17, 15.5], [16, 150, 9.6], metalMat, 0.6));
  put(jetson, boxMesh([14.5, 17, 15.5], [33, 150, 9.6], metalMat, 0.6));
  put(jetson, boxMesh([16, 21, 13.5], [61, 146, 9.6], metalMat, 0.6));            // Ethernet (RJ45)
  put(jetson, boxMesh([11, 1, 7.5], [63.5, 166.6, 12.5], darkMat));
  put(jetson, boxMesh([17, 9, 6], [82, 158, 9.6], metalMat, 0.5));                 // DisplayPort
  put(jetson, boxMesh([9, 14, 11], [101, 153, 9.6], darkMat, 0.8));                // power
  // Heat sink (base + 16 fins, instanced) and the fan on top. They lift a little more than the board when exploding.
  const sinkG = new THREE.Group(); jetson.g.add(sinkG);
  const putS = (mesh) => { mesh.position.sub(jetson.rest); sinkG.add(mesh); return mesh; };
  const sinkBase = putS(boxMesh([63, 55, 4], [31, 90, 13.6], sinkMat, 0.6)); rim(jetson, sinkBase);
  const FINS = 16, finBox = mmBox([1.3, 55, 13], [31, 90, 17.6]);
  const fins = new THREE.InstancedMesh(new THREE.BoxGeometry(finBox.size.x, finBox.size.y, finBox.size.z), sinkMat, FINS);
  const fm = new THREE.Matrix4();
  for (let k = 0; k < FINS; k++) { const c = mm(31.65 + k * (61.7 / (FINS - 1)), 117.5, 24.1).sub(jetson.rest); fm.makeTranslation(c.x, c.y, c.z); fins.setMatrixAt(k, fm); }
  fins.castShadow = true; fins.receiveShadow = true; sinkG.add(fins);
  const fanG = new THREE.Group(); sinkG.add(fanG);
  const fanC = mm(62.5, 114.5, 30.6).sub(jetson.rest); fanG.position.copy(fanC);
  const fanMat = own(jetson, matte({ color: 0x30353d, roughness: 0.6 }));
  const fanFrame = new THREE.Mesh(G.fanFrameGeo, fanMat); fanFrame.castShadow = true; fanG.add(fanFrame); rim(jetson, fanFrame);
  const rotor = new THREE.Group(); rotor.position.y = 0.035; fanG.add(rotor);
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.05, 24), fanMat); rotor.add(hub);
  const bladeGeo = new THREE.BoxGeometry(0.1, 0.006, 0.05);
  for (let k = 0; k < 7; k++) { const b = new THREE.Mesh(bladeGeo, fanMat); const a = (k / 7) * Math.PI * 2; b.position.set(Math.cos(a) * 0.12, 0, Math.sin(a) * 0.12); b.rotation.y = -a; b.rotation.x = 0.35; rotor.add(b); }
  const ledMat = flat({ color: SHADE });
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), ledMat); led.position.copy(mm(104, 84, 10.2)).sub(jetson.rest); jetson.g.add(led);

  // ---- Webcam: the clip in the lid socket, the housing on top of the lid (lens forward), a short neck between ----
  const clip = part("clip", [62.5, 32.15, 77.05], { explode: [0, 1.2, 0], lag: 0.8 });
  put(clip, boxMesh([38, 47, 7.1], [43.5, 8.65, 73.5], printMat(clip), 1.2)); // shortened: stops above the speaker (CAD overlap)
  const webcam = part("webcam", [62.5, 32.9, 103.6], { explode: [0, 1.6, 0], lag: 0.9 });
  const camMat = own(webcam, matte({ color: CAM_BODY, roughness: 0.6 }));
  const housing = put(webcam, boxMesh([96.5, 30.5, 35.6], [14.25, 17.65, 88.6], camMat, 12, 4)); rim(webcam, housing);
  put(webcam, boxMesh([24, 14, 5.8], [50.5, 25.9, 82.9], camMat, 1.5));        // neck down to the socket
  const bezelMat = own(webcam, matte({ color: 0x1e2228, roughness: 0.25, envMapIntensity: 0.9 }));
  put(webcam, boxMesh([62, 2, 22], [31.5, 16.9, 95.4], bezelMat, 6));
  const eyeG = new THREE.Group(); eyeG.position.copy(mm(62.5, 16.7, 106.4)).sub(webcam.rest); webcam.g.add(eyeG);
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.095, 0.1, 0.02, 40), own(webcam, matte({ color: 0x2a2f37, roughness: 0.4, metalness: 0.3 }))); barrel.rotation.x = Math.PI / 2; eyeG.add(barrel);
  const lensMat = new THREE.MeshStandardMaterial({ color: INK, roughness: 0.12, metalness: 0.05, envMapIntensity: 1.2, emissive: ACCENT, emissiveIntensity: 0 });
  const lens = new THREE.Mesh(new THREE.SphereGeometry(0.068, 32, 16), lensMat); lens.scale.set(1, 1, 0.4); lens.position.z = 0.012; eyeG.add(lens);
  const glint = new THREE.Mesh(new THREE.CircleGeometry(0.012, 16), flat({ color: LIGHT })); glint.position.set(-0.022, 0.024, 0.04); eyeG.add(glint);
  const halo = new THREE.Mesh(new THREE.RingGeometry(0.104, 0.122, 48), flat({ color: ACCENT, transparent: true, opacity: 0, depthWrite: false })); halo.position.z = 0.012; eyeG.add(halo);
  const camLed = new THREE.Mesh(new THREE.CircleGeometry(0.009, 12), flat({ color: SHADE })); camLed.position.set(0.21, 0.0, 0.012); eyeG.add(camLed);

  // ---- Glass edges for x-ray: thin outlines that stay when the shell goes clear ----
  const edgeMat = new THREE.LineBasicMaterial({ color: MUTED, transparent: true, opacity: 0, toneMapped: false, depthWrite: false });
  for (const p of Object.values(parts)) for (const m of p.glassMeshes) {
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry, 30), edgeMat); e.position.copy(m.position); e.quaternion.copy(m.quaternion); e.renderOrder = 6; p.g.add(e); p.edges.push(e);
  }

  // ---- Anchors (assembled), and exploded positions on demand ----
  const LIFT = 0.3; // the whole diagram floats up a little as it explodes
  const P = {
    box: [0, -0.45, 0],
    webcam: mm(62.5, 17.6, 106.4).toArray(),   // the lens
    jetson: jetson.rest.toArray(),
    speaker: mm(62.5, 0.5, 37.9).toArray(),     // the fabric front
    lid: mm(62.5, 99, 83).toArray(),            // the lid's top face, centre
    front: mm(62.5, -3.5, 40.25).toArray(),     // the honeycomb's face
    ethernet: mm(68.5, 201.5, 16.75).toArray(), // the jack on the back wall
  };
  const anchorPart = { webcam: "webcam", jetson: "jetson", speaker: "speaker", lid: "lid", front: "front", ethernet: "back", clip: "clip", rail: "rail", wallL: "wallL", wallR: "wallR", back: "back", bottom: "bottom" };
  /** World position of an anchor (or part) at explode e (default: assembled). Ignores spin. */
  function at(name, e = 0) {
    const pn = anchorPart[name]; const base = P[name] ? new THREE.Vector3(...P[name]) : parts[pn].rest.clone();
    if (pn) base.addScaledVector(parts[pn].offset, e);
    if (name !== "box") base.y += LIFT * e;
    return base.toArray();
  }

  const accentC = new THREE.Color(ACCENT), shadeC = new THREE.Color(SHADE);
  const HL = { webcam: ["webcam", "clip"], jetson: ["jetson"], speaker: ["speaker"], lid: ["lid"], front: ["front", "rail"] };
  const hlOf = {}; for (const k in HL) for (const pn of HL[k]) hlOf[pn] = k;

  function update(state, t) {
    const ex = clamp01(Math.max(state.explode || 0, state.open || 0));
    const xr = clamp01(state.xray || 0), d = state.drop || 0, beat = heartbeat(t), pulse = state.pulse || 0;
    const hl = state.hl || {}, lay = state.layer || [];
    const level = {
      webcam: Math.max(hl.webcam || 0, lay[0] || 0), jetson: Math.max(hl.jetson || 0, lay[1] || 0, lay[2] || 0, lay[3] || 0),
      speaker: hl.speaker || 0, lid: hl.lid || 0, front: hl.front || 0,
    };
    group.rotation.y = (state.spin || 0) * Math.PI * 2;
    group.position.y = LIFT * ex;
    let k = 0;
    for (const p of Object.values(parts)) {
      // Drop: parts fall in (1 -> 0) in order of their lag; a small negative value squashes for a settle.
      const lag = p.dropLag * 0.6, u = d <= 0 ? 0 : clamp01((d - lag) / (1 - lag));
      const fall = u * u * (1.6 + p.dropLag * 1.4), squash = d < 0 ? d * 0.08 * p.dropLag : 0;
      p.g.position.copy(p.rest).addScaledVector(p.offset, ex); p.g.position.y += fall + squash;
      p.g.rotation.set(u * (hash(k, 22) - 0.5) * 0.5, u * (hash(k, 23) - 0.5) * 0.4, u * (hash(k, 21) - 0.5) * 0.5);
      k++;
      // Highlight: tint 35 % toward the accent, a small emissive, the accent rim.
      const lit = clamp01(level[hlOf[p.name]] || 0);
      for (const m of p.mats) { m.color.copy(m.userData.base).lerp(accentC, lit * 0.35); m.emissiveIntensity = lit * 0.22; }
      if (p.rimMat) { p.rimMat.opacity = lit * 0.95; for (const r of p.rims) r.visible = lit > 0.002; }
      // X-ray: walls, back, lid and front go glassy; the outlines stay.
      if (p.glassy) {
        const g = p.name === "front" ? xr * 0.82 : xr;
        for (const m of p.mats) { if (m === logoMat) continue; m.opacity = 1 - g * 0.88; m.depthWrite = g < 0.05; }
        for (const m of p.glassMeshes) m.castShadow = g < 0.3;
        for (const e of p.edges) e.visible = xr > 0.002;
      }
    }
    logoMat.opacity = 1 - xr * 0.6;
    edgeMat.opacity = xr * 0.75;
    // The Jetson's heat sink and fan lift a little more in the exploded diagram; the fan always turns.
    sinkG.position.y = ex * 0.16; fanG.position.y = fanC.y + ex * 0.1;
    rotor.rotation.y = -t * 9.0;
    // Pulse: the Jetson's LED, a breath of warmth on the board, and the glow through the lid's vent.
    const pb = pulse * beat;
    ledMat.color.copy(shadeC).lerp(accentC, clamp01(pulse * 2.5) * (0.55 + 0.45 * beat));
    for (const m of jetson.mats) m.emissiveIntensity += pb * 0.25;
    ventGlow.material.opacity = clamp01(pb * 0.75); ventGlow.visible = pb > 0.002;
    // Eyes: the lens warms up with a thin halo; the camera's LED turns accent.
    const eyes = clamp01(state.eyes || 0);
    lensMat.emissiveIntensity = eyes * (0.9 + 0.1 * Math.sin(t * 1.3));
    halo.material.opacity = eyes * 0.8; halo.visible = eyes > 0.002;
    camLed.material.color.copy(shadeC).lerp(accentC, eyes);
  }

  return { group, parts, P, at, update, LIFT, holes: G.holes };
}

/** A shell-only copy (no interior, no explode): the friends' boxes and the kitchen device. update(glow, t, eyes). */
export function buildEnclosureLite() {
  const T = textures(), G = geometry();
  const group = new THREE.Group(); group.name = "atlas-lite";
  const mats = [];
  const own = (m) => { m.userData.base = m.color.clone(); mats.push(m); return m; };
  const kraft = own(matte({ color: KRAFT, map: T.kraft, roughness: 0.95 })), print = own(matte({ color: PRINT, roughness: 0.7 }));
  const cam = own(matte({ color: CAM_BODY, roughness: 0.6 }));
  group.add(boxMesh([125, 198, 1.3], [0, 0, 0], kraft), boxMesh([1.3, 198, 78.7], [0, 0, 1.3], kraft), boxMesh([1.3, 198, 78.7], [123.7, 0, 1.3], kraft), boxMesh([122.4, 1.3, 78.7], [1.3, 196.7, 1.3], kraft));
  group.add(boxMesh([121, 65, 70], [2, 0.5, 2.9], own(matte({ color: FABRIC, roughness: 1 })), 7)); // what the honeycomb shows
  group.add(boxMesh([125, 6, 2.5], [0, -5.5, 0], print, 1));
  const fp = new THREE.Mesh(G.frontGeo, print); fp.castShadow = true; group.add(fp);
  const lg = new THREE.Mesh(new THREE.PlaneGeometry(0.56, 0.175), own(matte({ color: 0xffffff, map: T.logo, transparent: true, alphaTest: 0.04 }))); lg.position.set(0, DESK_Y + 0.405, 1.0262); group.add(lg);
  const lid = new THREE.Mesh(G.lidGeo, print); lid.castShadow = true; lid.receiveShadow = true; group.add(lid);
  group.add(boxMesh([96.5, 30.5, 35.6], [14.25, 17.65, 88.6], cam, 12, 3), boxMesh([24, 14, 5.8], [50.5, 25.9, 82.9], cam, 1.5));
  const lensMat = new THREE.MeshStandardMaterial({ color: INK, roughness: 0.12, envMapIntensity: 1.2, emissive: ACCENT, emissiveIntensity: 0 });
  const lens = new THREE.Mesh(new THREE.SphereGeometry(0.068, 24, 12), lensMat); lens.scale.set(1, 1, 0.4); lens.position.copy(mm(62.5, 16.9, 106.4)); group.add(lens);
  const glow = new THREE.Mesh(new THREE.PlaneGeometry((G.vent.x1 - G.vent.x0) / 100, (G.vent.y1 - G.vent.y0) / 100), flat({ color: ACCENT, transparent: true, opacity: 0, depthWrite: false }));
  glow.rotation.x = -Math.PI / 2; glow.position.copy(mm((G.vent.x0 + G.vent.x1) / 2, (G.vent.y0 + G.vent.y1) / 2, 79.2)); group.add(glow);
  const accentC = new THREE.Color(ACCENT);
  function update(v, t, eyes = 0) {
    const g = clamp01(v), beat = heartbeat(t);
    for (const m of mats) { m.color.copy(m.userData.base).lerp(accentC, g * 0.28); m.emissiveIntensity = g * (0.14 + 0.1 * beat); }
    glow.material.opacity = clamp01(g * (0.45 + 0.5 * beat)); glow.visible = g > 0.002;
    lensMat.emissiveIntensity = clamp01(Math.max(eyes, g)) * 0.9;
  }
  return { group, update };
}
