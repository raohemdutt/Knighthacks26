// The kitchen set, far to the left (x = KX) so the camera travels there. A counter with cabinets, a fridge edge, three
// bins (blue recycling, green compost, grey trash) with label plates, a small camera on the wall aimed at the bins, a
// second Atlas on the counter, the person standing by the bins with a can. state.kitchen rises it in, state.canThrow
// throws the can into the grey bin, state.binCount lights the counter device. Units: 1 = 100 mm.
import * as THREE from "three";
import { SHADE, INK, FLOOR_Y, DESK_Y, hash, clamp01, lerp, softTexture, FONT, flat, matte, tint, fadingSet, RoundedBoxGeometry } from "./common.js";
import { makeHuman, LOOKS } from "./human.js";
import { buildAtlasLite as buildEnclosureLite } from "./device.js";

export const KX = -42;
const WOOD = 0xc9a373, WALL = 0xe9e4da; // the website room's oak and warm wall
const BINS = [
  { name: "recycle", text: "RECYCLE", color: 0x2f7ed8, ink: "#4f6f99", x: -12.4 },
  { name: "compost", text: "COMPOST", color: 0x3f9b5a, ink: "#557a54", x: -8.6 },
  { name: "trash", text: "TRASH", color: 0x6c747e, ink: "#5b636e", x: -4.8 },
];

function rbox(w, h, d, mat, r = 0.12, seg = 3) {
  const m = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, seg, Math.min(r, Math.min(w, h, d) / 2 - 1e-3)), mat);
  m.castShadow = true; m.receiveShadow = true; return m;
}
function plateTexture(text, ink) {
  return softTexture((ctx, w, h) => {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#f1f4f8"; ctx.beginPath(); ctx.roundRect(6, 6, w - 12, h - 12, 34); ctx.fill();
    ctx.fillStyle = ink; ctx.beginPath(); ctx.arc(66, h / 2, 22, 0, Math.PI * 2); ctx.fill();
    ctx.font = `700 64px ${FONT}`; ctx.textBaseline = "middle"; ctx.fillText(text, 112, h / 2 + 3);
  }, 512, 160);
}

export function buildKitchen(scene, humanGltf) {
  const root = new THREE.Group(); root.name = "kitchen"; root.position.x = KX; scene.add(root);
  const set = fadingSet(root);
  const F = FLOOR_Y, TOP = 0.5; // counter top (90 cm)
  const white = matte({ roughness: 0.8 }), wood = matte({ color: WOOD, roughness: 0.75 }), wall = matte({ color: WALL, roughness: 0.95 });
  const ink = matte({ color: 0x3a404a, roughness: 0.5 }), steel = matte({ color: 0xc6ccd4, roughness: 0.4, metalness: 0.4 });
  const at = (m, x, y, z) => { m.position.set(x, y, z); return m; };

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(56, 44), new THREE.ShadowMaterial({ opacity: 0.2, color: SHADE, depthWrite: false }));
  floor.rotation.x = -Math.PI / 2; floor.position.set(0, F, 4); floor.receiveShadow = true; floor.renderOrder = -5; set.add(floor);
  const backWall = at(rbox(40, 26, 0.4, wall, 0.1, 2), 1.5, F + 13, -4.8); backWall.castShadow = false; set.add(backWall);

  // Counter: base cabinets with doors and handles, the top, upper cabinets, a backsplash line.
  const counter = new THREE.Group();
  counter.add(at(rbox(14, TOP - 0.4 - F - 0.5, 6.0, white, 0.1, 2), 5, (F + 0.5 + TOP - 0.4) / 2, -1.5));
  counter.add(at(rbox(13.6, 0.5, 5.6, ink, 0.05, 1), 5, F + 0.25, -1.7)); // toe kick
  for (let k = 0; k < 4; k++) {
    const x = -0.25 + k * 3.5 + 1.75;
    counter.add(at(rbox(3.35, TOP - 0.55 - F - 0.7, 0.12, white, 0.06, 2), x, (F + 0.6 + TOP - 0.5) / 2, 1.56));
    counter.add(at(rbox(0.16, 1.4, 0.2, ink, 0.07, 2), x + (k % 2 ? -1.3 : 1.3), TOP - 2.2, 1.72));
  }
  counter.add(at(rbox(14.4, 0.4, 6.4, wood, 0.08, 2), 5, TOP - 0.2, -1.5));
  counter.add(at(rbox(14, 4.8, 3.4, white, 0.1, 2), 5, 8.2, -2.9));
  for (let k = 0; k < 4; k++) counter.add(at(rbox(0.16, 1.2, 0.2, ink, 0.07, 2), -0.25 + k * 3.5 + 1.75 + (k % 2 ? -1.3 : 1.3), 6.4, -1.12));
  // A bowl of fruit and a cutting board on the counter.
  const board = at(rbox(3.0, 0.16, 2.0, tint(0xd8c6ae, { roughness: 0.8 }), 0.06, 2), 9.0, TOP + 0.08, -2.4); board.rotation.y = 0.15; counter.add(board);
  const bowl = new THREE.Mesh(new THREE.SphereGeometry(1.0, 24, 12, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), matte({ roughness: 0.7, side: THREE.DoubleSide })); bowl.position.set(-0.4, TOP + 0.9, -2.6); bowl.castShadow = true; counter.add(bowl);
  for (let k = 0; k < 4; k++) { const f = new THREE.Mesh(new THREE.SphereGeometry(0.38, 14, 10), tint([0xd6a68f, 0xc9b98a, 0xa9bb92, 0xd6a68f][k], { roughness: 0.6 })); f.position.set(-0.4 + (hash(k, 81) - 0.5) * 0.9, TOP + 0.75 + hash(k, 82) * 0.25, -2.6 + (hash(k, 83) - 0.5) * 0.9); f.castShadow = true; counter.add(f); }
  set.add(counter);

  // The fridge edge on the right.
  const fridge = new THREE.Group(); fridge.position.set(15.8, F, -1.0);
  fridge.add(at(rbox(7.0, 18.5, 7.2, white, 0.3, 3), 0, 9.25, 0));
  fridge.add(at(rbox(6.9, 0.08, 0.1, matte({ color: SHADE, roughness: 0.9 }), 0.03, 1), 0, 12.4, 3.6));
  for (const y of [14.6, 8.6]) fridge.add(at(rbox(0.22, 2.6, 0.3, steel, 0.1, 2), -2.9, y, 3.75));
  set.add(fridge);

  // The bins in a row: open tops, label plates on the front.
  const BW = 3.2, BH = 6.2, BD = 3.6, T = 0.14, BIN_Z = 0.6;
  const bins = BINS.map((b) => {
    const g = new THREE.Group(); g.position.set(b.x, F, BIN_Z);
    const m = tint(b.color, { roughness: 0.75 }), inner = tint(new THREE.Color(b.color).multiplyScalar(0.72).getHex(), { roughness: 0.9 });
    g.add(at(rbox(BW, 0.3, BD, m, 0.1, 2), 0, 0.15, 0));
    g.add(at(rbox(BW, BH, T, m, 0.06, 2), 0, BH / 2, BD / 2 - T / 2), at(rbox(BW, BH, T, m, 0.06, 2), 0, BH / 2, -BD / 2 + T / 2));
    g.add(at(rbox(T, BH, BD, m, 0.06, 2), BW / 2 - T / 2, BH / 2, 0), at(rbox(T, BH, BD, m, 0.06, 2), -BW / 2 + T / 2, BH / 2, 0));
    const floorIn = new THREE.Mesh(new THREE.PlaneGeometry(BW - 2 * T, BD - 2 * T), inner); floorIn.rotation.x = -Math.PI / 2; floorIn.position.y = 0.31; g.add(floorIn);
    // A soft rolled rim.
    for (const [w, d, x, z] of [[BW + 0.1, 0.3, 0, BD / 2], [BW + 0.1, 0.3, 0, -BD / 2], [0.3, BD + 0.1, BW / 2, 0], [0.3, BD + 0.1, -BW / 2, 0]]) g.add(at(rbox(w, 0.24, d, m, 0.11, 2), x, BH, z));
    const plate = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 0.75), flat({ map: plateTexture(b.text, b.ink), transparent: true })); plate.position.set(0, BH * 0.62, BD / 2 + 0.012); g.add(plate);
    set.add(g); return g;
  });

  // The bin camera high on the wall above the bins, aimed down at them (and at whoever stands in front).
  const wcam = new THREE.Group(); wcam.position.set(-8.6, 13.6, -4.55);
  wcam.add(at(rbox(1.4, 1.4, 0.2, white, 0.08, 2), 0, 0, 0));
  const arm = at(rbox(0.3, 0.3, 1.5, white, 0.1, 2), 0, 0, 0.8); wcam.add(arm);
  const head = new THREE.Group(); head.position.set(0, -0.15, 1.6); head.rotation.order = "YXZ"; head.rotation.set(1.08, 0.37, 0); wcam.add(head);
  head.add(at(rbox(1.0, 0.8, 1.5, white, 0.3, 3), 0, 0, 0.3));
  const wl = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.26, 0.12, 20), matte({ color: INK, roughness: 0.15, envMapIntensity: 1.2 })); wl.rotation.x = Math.PI / 2; wl.position.set(0, 0, 1.08); head.add(wl);
  const wled = new THREE.Mesh(new THREE.CircleGeometry(0.06, 12), flat({ color: 0x456cb3 })); wled.position.set(0.32, 0.22, 1.06); head.add(wled);
  set.add(wcam);

  // The second Atlas on the counter.
  const atlas = buildEnclosureLite();
  const atlasHolder = new THREE.Group(); atlasHolder.position.set(3.4, TOP - DESK_Y, -1.9); atlasHolder.rotation.y = -0.22; atlasHolder.add(atlas.group);
  atlas.group.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  set.add(atlasHolder);

  // The person (sets/human.js), standing by the bins with a can; Quaternius' "Interact" clip is the reach to drop it.
  const person = makeHuman(humanGltf, LOOKS.hemdutt);
  const pHolder = new THREE.Group(); pHolder.position.set(-1.6, F, 6.2); pHolder.rotation.y = 0.5 + Math.PI; pHolder.add(person.root); set.add(pHolder);
  const REACH = 0.9; // seconds into "Interact" where the hand is furthest out: the can leaves the hand here
  const can = new THREE.Group();
  const canBody = new THREE.Mesh(new THREE.CylinderGeometry(0.33, 0.33, 1.2, 22), steel); canBody.castShadow = true; can.add(canBody);
  const band = new THREE.Mesh(new THREE.CylinderGeometry(0.335, 0.335, 0.6, 22), tint(0xb98478, { roughness: 0.5 })); can.add(band);
  const lidTop = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.31, 0.08, 22), steel); lidTop.position.y = 0.63; can.add(lidTop);
  set.add(can);
  const binTarget = new THREE.Vector3(BINS[0].x + 0.1, F + 1.0, BIN_Z); // inside the blue recycling bin (film v4.7: he turns to the blue one), kitchen-local
  // Release point = the hand at the reach (kitchen-local), computed once.
  person.sample([["Interact", REACH, 1]]); root.updateMatrixWorld(true); const rel = root.worldToLocal(person.handPos("R").clone()).clone(); person.sample([["Idle_Loop", 0, 1]]);
  const hv = new THREE.Vector3();

  const P = {
    kitchen: [KX, 0, 0], counterAtlas: [KX + 3.4, TOP + 0.45, -1.9], binCam: [KX - 8.4, 13.0, -2.0],
    bins: BINS.map((b) => [KX + b.x, F + BH, BIN_Z]), person: [KX - 1.6, F + 16.6, 6.2],
  };
  function update(state, t) {
    const v = state.kitchen || 0; set.update(v);
    if (!root.visible) return;
    const c = clamp01(state.canThrow || 0), swing = clamp01(c / 0.18), sw = swing * swing * (3 - 2 * swing);
    const w = clamp01(c / 0.05), it = Math.min(swing, 1) * REACH + clamp01((c - 0.18) / 0.82) * (person.duration("Interact") - REACH);
    person.sample(w < 1 ? [["Idle_Loop", (t * 0.8) % person.duration("Idle_Loop"), 1 - w], ["Interact", it, w]] : [["Interact", it, 1]]);
    person.look(0, lerp(0.35, 0.5, c));
    person.express({ strain: 0.2 }, t);
    if (c < 0.18) { root.worldToLocal(person.handPos("R", hv)); can.position.copy(hv); can.position.y -= 0.35; can.rotation.set(0, 0, 0); }
    else {
      const u = (c - 0.18) / 0.82;
      can.position.lerpVectors(rel, binTarget, u); can.position.y += Math.sin(u * Math.PI) * 2.4 * (1 - u * 0.3);
      can.rotation.set(u * 5.2, 0, u * 1.4);
    }
    can.visible = can.visible && v > 0.001;
    atlas.update(state.binCount || 0, t, state.binCount || 0);
  }
  return { root, update, P, person, bins };
}
