// The dorm room set. The device stays at the origin: here it stands on a wall shelf at the person's eye height, to the
// right of their desk chair, its camera aimed at them. The person (sets/human.js, the Quaternius mannequin) faces the
// desk (-x); the bed is behind them (+x). state.room rises/fades the set in, state.person shows the person,
// state.personLook turns their head toward the device. The scene files give the person a schedule of acts
// (room.person.act): sitting up on the bed, standing, walking, sitting down at the desk. Units: 1 = 100 mm.
import * as THREE from "three";
import { SHADE, INK, ACCENT, MUTED, FLOOR_Y, hash, clamp01, lerp, softTexture, FONT, flat, matte, tint, between, fadingSet, RoundedBoxGeometry } from "./common.js";
import { makeHuman, LOOKS } from "./human.js";

const WOOD = 0xe0d6c8, WALL = 0xe1e6ed, FABRIC = 0xa9b4c5, BOOKS = [0x8f9cb0, 0xc9b7a2, 0x9fb09c, 0xd2c2b0, 0x566176, 0xb4a6b8, 0xa7b8c4, 0xd9cbb9];

function rbox(w, h, d, mat, r = 0.12, seg = 3) {
  const m = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, seg, Math.min(r, Math.min(w, h, d) / 2 - 1e-3)), mat);
  m.castShadow = true; m.receiveShadow = true; return m;
}
function screenTexture() {
  return softTexture((ctx, w, h) => {
    ctx.fillStyle = "#eef1f5"; ctx.fillRect(0, 0, w, h);
    const rr = (x, y, ww, hh, r, c) => { ctx.fillStyle = c; ctx.beginPath(); ctx.roundRect(x, y, ww, hh, r); ctx.fill(); };
    rr(28, 28, 180, h - 56, 22, "#e2e7ee");
    for (let i = 0; i < 5; i++) rr(52, 70 + i * 54, 130, 18, 9, i === 1 ? "#456cb3" : "#c8d0db");
    ctx.fillStyle = "#1b2433"; ctx.font = `700 40px ${FONT}`; ctx.fillText("Today", 250, 86);
    for (let d = 0; d < 5; d++) {
      const x = 250 + d * 150; ctx.fillStyle = "#566176"; ctx.font = `600 20px ${FONT}`; ctx.fillText(["Mon", "Tue", "Wed", "Thu", "Fri"][d], x, 130);
      for (let k = 0; k < 3; k++) { if (hash(d, k + 50) < 0.3) continue; rr(x, 150 + k * 90 + hash(d, k) * 30, 130, 56 + hash(d, k + 9) * 30, 12, d === 2 && k === 1 ? "#f0c9b4" : "#dde3eb"); }
    }
  }, 1024, 576);
}
function rugTexture() {
  return softTexture((ctx, w, h) => {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#d3dae3"; ctx.beginPath(); ctx.roundRect(8, 8, w - 16, h - 16, 60); ctx.fill();
    ctx.strokeStyle = "#e9edf2"; ctx.lineWidth = 10; ctx.beginPath(); ctx.roundRect(46, 46, w - 92, h - 92, 40); ctx.stroke();
    ctx.strokeStyle = "#c3cbd6"; ctx.lineWidth = 4; ctx.beginPath(); ctx.roundRect(74, 74, w - 148, h - 148, 28); ctx.stroke();
  }, 512, 420);
}

export function buildRoom(scene, humanGltf) {
  const root = new THREE.Group(); root.name = "room"; scene.add(root);
  const set = fadingSet(root);
  // The device's shelf is at DESK_Y (-1.0). The room floor sits 1.1 m below it, so the device is at a seated person's
  // eye height; the desk top is a normal 75 cm.
  const F = -12.0, SHELF = -1.002, TOP = F + 7.5;
  const wood = matte({ color: WOOD, roughness: 0.75 }), white = matte({ roughness: 0.8 }), wall = matte({ color: WALL, roughness: 0.95 });
  const ink = matte({ color: 0x3a404a, roughness: 0.5 }), fabric = tint(FABRIC, { roughness: 1 }), soft = matte({ color: 0xd3dae3, roughness: 0.9 });
  const at = (m, x, y, z) => { m.position.set(x, y, z); return m; };

  // Floor shadows (the floor itself is the page colour, so only the shadow shows), and a rug under the chair.
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(70, 50), new THREE.ShadowMaterial({ opacity: 0.2, color: SHADE, depthWrite: false }));
  floor.rotation.x = -Math.PI / 2; floor.position.set(4, F, 10); floor.receiveShadow = true; floor.renderOrder = -5; set.add(floor);
  const rug = new THREE.Mesh(new THREE.PlaneGeometry(12, 10), flat({ map: rugTexture(), transparent: true, depthWrite: false }));
  rug.rotation.x = -Math.PI / 2; rug.position.set(3.5, F + 0.01, 8.5); rug.renderOrder = -4; set.add(rug);

  // The back wall behind the device, with a window over the desk.
  const backWall = at(rbox(46, 28, 0.4, wall, 0.1, 2), 4, F + 14, -1.45); backWall.castShadow = false; set.add(backWall);
  const win = new THREE.Group(); win.position.set(-8.5, F + 15.5, -1.2);
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(7.6, 7.6), flat({ color: 0xf6f8fb })); glass.position.z = -0.02; win.add(glass);
  for (const [w, h, x, y] of [[8.4, 0.4, 0, 4.0], [8.4, 0.4, 0, -4.0], [0.4, 8.4, 4.0, 0], [0.4, 8.4, -4.0, 0], [7.6, 0.18, 0, 0], [0.18, 7.6, 0, 0]]) win.add(at(rbox(w, h, 0.4, white, 0.08, 2), x, y, 0.1));
  win.add(at(rbox(9.2, 0.25, 0.9, white, 0.08, 2), 0, -4.3, 0.4));
  set.add(win);

  // The device's shelf: a wall shelf under the box, two brackets, a few books and a plant beside it.
  const shelf = new THREE.Group(); shelf.position.set(1.0, SHELF, 0.05);
  shelf.add(at(rbox(6.2, 0.26, 2.6, wood, 0.08, 2), 0, -0.13, 0));
  for (const x of [-2.4, 2.4]) { const b = at(rbox(0.22, 1.3, 1.6, white, 0.05, 2), x, -0.9, -0.4); shelf.add(b); }
  let bx = 1.35;
  for (let i = 0; i < 5; i++) {
    const w = 0.3 + hash(i, 61) * 0.22, h = 1.3 + hash(i, 62) * 0.6, d = 1.1 + hash(i, 63) * 0.3;
    const b = at(rbox(w, h, d, tint(BOOKS[i % BOOKS.length], { roughness: 0.85 }), 0.05, 2), bx + w / 2, h / 2, -0.4);
    if (i === 4) { b.rotation.z = -0.3; b.position.x += 0.3; b.position.y -= 0.1; }
    shelf.add(b); bx += w + 0.03;
  }
  const pot = at(new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.3, 0.62, 20), white), -2.3, 0.31, -0.2); pot.castShadow = true; shelf.add(pot);
  for (let k = 0; k < 5; k++) { const leaf = new THREE.Mesh(new THREE.SphereGeometry(0.36, 14, 10), tint(0x9fb09c, { roughness: 0.9 })); leaf.scale.set(0.6, 1.2, 0.6); leaf.position.set(-2.3 + (hash(k, 64) - 0.5) * 0.5, 0.95 + hash(k, 65) * 0.3, -0.2 + (hash(k, 66) - 0.5) * 0.4); leaf.rotation.z = (hash(k, 67) - 0.5) * 1.2; leaf.castShadow = true; shelf.add(leaf); }
  set.add(shelf);

  // The bed, behind the person: headboard on the back wall, long side facing the desk.
  const BED_X = 19.0, BED_TOP = F + 5.0;
  const bed = new THREE.Group(); bed.position.set(BED_X, F, 8.6);
  bed.add(at(rbox(10.2, 2.4, 20.0, wood, 0.15), 0, 1.2, 0));
  bed.add(at(rbox(9.9, 2.6, 19.6, white, 0.45, 4), 0, 3.7, 0));
  bed.add(at(rbox(10.0, 0.5, 12.0, fabric, 0.25, 3), 0.1, 5.05, 3.6));
  bed.add(at(rbox(6.5, 1.0, 3.0, soft, 0.45, 4), 0, 5.4, -7.6));
  bed.add(at(rbox(10.2, 7.0, 0.45, wood, 0.15), 0, 3.5, -9.8));
  set.add(bed);

  // The desk, along the left: the person faces it (-x).
  const desk = new THREE.Group();
  desk.add(at(rbox(6.4, 0.32, 13.0, wood, 0.1, 2), -6.8, TOP - 0.16, 8.5));
  for (const z of [2.3, 14.7]) desk.add(at(rbox(6.0, TOP - 0.32 - F, 0.36, white, 0.08, 2), -6.8, (TOP - 0.32 + F) / 2, z));
  set.add(desk);

  // Monitor (no webcam: Atlas is the only camera in this room), keyboard, mouse, a mug, a notebook, the lamp.
  const mon = new THREE.Group(); mon.position.set(-8.8, TOP, 8.5); mon.rotation.y = Math.PI / 2;
  mon.add(at(rbox(2.4, 0.12, 1.6, white, 0.05, 2), 0, 0.06, 0));
  mon.add(at(rbox(0.38, 2.4, 0.3, white, 0.1, 2), 0, 1.3, -0.15));
  mon.add(at(rbox(6.3, 3.75, 0.28, white, 0.1, 3), 0, 3.3, 0.02));
  const scr = new THREE.Mesh(new THREE.PlaneGeometry(5.95, 3.38), flat({ map: screenTexture() })); scr.position.set(0, 3.32, 0.165); mon.add(scr);
  set.add(mon);
  const kb = new THREE.Group(); kb.position.set(-5.4, TOP, 8.5); kb.rotation.y = Math.PI / 2;
  kb.add(at(rbox(4.5, 0.16, 1.5, white, 0.06, 2), 0, 0.08, 0));
  const keys = new THREE.InstancedMesh(new RoundedBoxGeometry(0.24, 0.08, 0.24, 1, 0.04), matte({ color: 0xedf1f6, roughness: 0.7 }), 64);
  const km = new THREE.Matrix4(); let ki = 0;
  for (let r = 0; r < 4; r++) for (let c = 0; c < 16; c++) { km.makeTranslation(-1.98 + c * 0.264, 0.2, -0.5 + r * 0.33); keys.setMatrixAt(ki++, km); }
  keys.castShadow = false; keys.receiveShadow = true; kb.add(keys);
  set.add(kb);
  const mouse = new THREE.Mesh(new THREE.SphereGeometry(0.4, 16, 12), white); mouse.scale.set(1.1, 0.35, 0.75); mouse.position.set(-5.2, TOP + 0.12, 11.4); mouse.castShadow = true; set.add(mouse);
  const mug = new THREE.Group(); mug.position.set(-6.0, TOP, 13.4);
  const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.38, 0.95, 24), tint(0xc9b7a2, { roughness: 0.6 })); cup.position.y = 0.48; cup.castShadow = true; mug.add(cup);
  const handle = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.06, 8, 16), cup.material); handle.position.set(0.44, 0.5, 0); mug.add(handle);
  set.add(mug);
  // His keys, behind the mug ("where are my keys?"): a ring and two keys lying on the desk.
  const keyset = new THREE.Group(); keyset.position.set(-7.2, TOP + 0.05, 12.1); keyset.rotation.y = 0.6;
  const steelK = matte({ color: 0xb8c0cb, roughness: 0.35, metalness: 0.6 });
  const ringK = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.04, 8, 20), steelK); ringK.rotation.x = Math.PI / 2; keyset.add(ringK);
  for (const [dx, a2, c] of [[0.42, 0.2, 0xb8c0cb], [0.36, -0.5, 0xc9a86a]]) {
    const k = new THREE.Group(); k.rotation.y = a2; k.position.set(Math.cos(a2) * 0.2, 0, -Math.sin(a2) * 0.2);
    const bow = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.05, 16), matte({ color: c, roughness: 0.35, metalness: 0.6 })); bow.position.x = dx * 0.6; k.add(bow);
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.04, 0.09), matte({ color: c, roughness: 0.35, metalness: 0.6 })); blade.position.x = dx * 0.6 + 0.38; k.add(blade);
    keyset.add(k);
  }
  keyset.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  set.add(keyset);
  const note = at(rbox(2.8, 0.14, 2.1, tint(0x8f9cb0, { roughness: 0.9 }), 0.05, 2), -6.4, TOP + 0.07, 4.4); note.rotation.y = 0.18; set.add(note);
  const lamp = new THREE.Group(); lamp.position.set(-8.6, TOP, 3.6); lamp.rotation.y = Math.PI / 2;
  const lbase = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.85, 0.2, 28), white); lbase.position.y = 0.1; lbase.castShadow = true; lamp.add(lbase);
  const j1 = new THREE.Vector3(0, 0.2, 0), j2 = new THREE.Vector3(0.3, 4.0, -0.4), j3 = new THREE.Vector3(-0.9, 4.7, 1.1);
  for (const [a, b] of [[j1, j2], [j2, j3]]) { const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, a.distanceTo(b), 10), white); arm.castShadow = true; lamp.add(between(arm, a, b)); }
  const joint = new THREE.Mesh(new THREE.SphereGeometry(0.17, 12, 10), white); joint.position.copy(j2); lamp.add(joint);
  const shade = new THREE.Mesh(new THREE.ConeGeometry(0.85, 1.2, 28, 1, true), matte({ roughness: 0.8, side: THREE.DoubleSide })); shade.position.copy(j3); shade.position.y -= 0.35; shade.rotation.z = 0.3; shade.castShadow = true; lamp.add(shade);
  const bulb = new THREE.Mesh(new THREE.CircleGeometry(0.62, 24), flat({ color: 0xfbf6ef })); bulb.rotation.x = Math.PI / 2; bulb.position.copy(shade.position); bulb.position.y -= 0.5; bulb.position.x -= 0.15; lamp.add(bulb);
  set.add(lamp);

  // The chair, pulled out from the desk, facing it; the person sits down into it.
  const CHAIR = [0.6, F, 8.5];
  const chair = new THREE.Group(); chair.position.set(...CHAIR); chair.rotation.y = Math.PI / 2; // backrest toward the bed (+x), seat facing the desk
  chair.add(at(rbox(4.4, 0.6, 4.2, fabric, 0.28, 3), 0, 4.3, 0));
  const back = at(rbox(4.2, 4.6, 0.55, fabric, 0.28, 3), 0, 7.6, 2.25); back.rotation.x = 0.1; chair.add(back);
  for (const sd of [-1, 1]) chair.add(at(rbox(0.25, 3.0, 0.3, ink, 0.1, 2), sd * 1.4, 5.9, 2.1));
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 3.4, 12), ink); post.position.y = 2.3; post.castShadow = true; chair.add(post);
  for (let k = 0; k < 5; k++) {
    const a = (k / 5) * Math.PI * 2 + 0.3, leg = rbox(2.3, 0.22, 0.32, ink, 0.1, 2);
    leg.position.set(Math.cos(a) * 1.1, 0.62, Math.sin(a) * 1.1); leg.rotation.y = -a; chair.add(leg);
    const wheel = new THREE.Mesh(new THREE.SphereGeometry(0.26, 12, 8), ink); wheel.position.set(Math.cos(a) * 2.15, 0.26, Math.sin(a) * 2.15); chair.add(wheel);
  }
  set.add(chair);

  // The person (sets/human.js). Default: seated at the desk, facing it. Scenes add acts (bed, stand, walk, sit).
  const person = makeHuman(humanGltf, LOOKS.hemdutt);
  for (const m of person.mats) { m.transparent = true; m.userData.baseOpacity = m.opacity; }
  root.add(person.root);
  const FACE_DESK = -Math.PI / 2;
  // Where the acts happen (room coordinates = world, the set does not move).
  const SPOTS = {
    seat: [CHAIR[0] - 1.7, F, CHAIR[2]],          // the root of the seated pose, forward on the seat (at +0.4 the legs went through the chair)
    standDesk: [CHAIR[0] - 2.0, F, CHAIR[2]],     // standing in front of the chair, back to it, facing the desk
    bedEdge: [BED_X - 4.3, F + 0.6, 7.5],          // sitting on the mattress's edge (it starts at BED_X - 5.1), feet on the floor
    bedLie: [BED_X - 0.2, F + 7.2, 17.8],          // lying on the back, head on the pillow (toward -z), feet at the foot end
    bedFront: [BED_X - 6.6, F, 7.5],               // standing up from the bed
    faceDesk: FACE_DESK,
  };
  const SEATED_ACT = { t0: -1e9, t1: 1e9, clip: "Sitting_Idle_Loop", loop: true, speed: 0.6, from: SPOTS.seat, yaw: [FACE_DESK] };
  person.act([SEATED_ACT]);

  const head = new THREE.Vector3(), toBox = new THREE.Vector3(), BOX = new THREE.Vector3(0, -0.3, 0);
  const P = { keys: [-7.2, TOP + 0.05, 12.1], desk: [-6.8, TOP, 8.5], monitor: [-8.6, TOP + 3.3, 8.5], keyboard: [-5.4, TOP + 0.2, 8.5], shelf: [1.0, SHELF, 0.05], bed: [BED_X, BED_TOP, 8.6], chair: CHAIR, person: [CHAIR[0] + 0.2, F + 12.3, CHAIR[2]] };
  function update(state, t) {
    set.update(state.room || 0);
    const pv = clamp01(Math.min(state.room || 0, state.person || 0));
    person.root.visible = pv > 0.001;
    if (!person.root.visible) return;
    for (const m of person.mats) { m.opacity = m.userData.baseOpacity * pv; m.depthWrite = pv > 0.6; }
    person.at(t);
    person.express({ smile: state.personSmile || 0, sleepy: state.personSleepy || 0, strain: state.personStrain || 0, talk: state.personTalk || 0 }, t);
    person.root.position.y -= (1 - pv) * 0.6;
    // The head turns from where the body faces toward the device on the shelf.
    const l = clamp01(state.personLook || 0), e = l * l * (3 - 2 * l);
    if (e > 0.001) {
      person.headPos(head); toBox.copy(BOX).sub(head);
      const want = Math.atan2(toBox.x, toBox.z), have = person.root.rotation.y;
      let yaw = want - have; yaw = ((yaw + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
      const pitch = Math.atan2(-toBox.y, Math.hypot(toBox.x, toBox.z));
      person.look(yaw * e, pitch * e * 0.8);
    }
  }
  return { root, update, P, person, SPOTS, F };
}
