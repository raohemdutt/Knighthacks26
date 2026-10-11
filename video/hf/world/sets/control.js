// The control room set, far to the right (x = CX) so the camera travels there: a long console with three monitors
// and an Atlas on it, a big wall display showing the Inspection page, a window onto the plant floor (pipes, heat
// exchangers HX-202 and HX-204, a SPOT robot walking its round), Xander seated at the console in a hi-vis vest, and
// the trainee standing beside. state.control rises it in, state.spotWalk walks SPOT along the floor (0..1),
// state.trainee brings the trainee in, state.xTalk moves Xander's mouth, state.consoleCount lights the console
// Atlas. Units: 1 = 100 mm. Pure function of state and t.
import * as THREE from "three";
import { SHADE, FLOOR_Y, DESK_Y, hash, clamp01, lerp, softTexture, FONT, flat, matte, tint, fadingSet, RoundedBoxGeometry } from "./common.js";
import { makeHuman, LOOKS } from "./human.js";
import { buildEnclosureLite } from "./enclosure.js";

export const CX = 46;
const F = FLOOR_Y, TOP = F + 7.5; // console top (75 cm)

function rbox(w, h, d, mat, r = 0.12, seg = 3) {
  const m = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, seg, Math.min(r, Math.min(w, h, d) / 2 - 1e-3)), mat);
  m.castShadow = true; m.receiveShadow = true; return m;
}

/** A soft-UI screen: a title, then rows of [label, value, accent?]. */
function screenTexture(title, rows, w = 1024, h = 576) {
  return softTexture((ctx) => {
    ctx.fillStyle = "#eef2f8"; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#456cb3"; ctx.beginPath(); ctx.arc(46, 46, 12, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#1b2433"; ctx.font = `700 40px ${FONT}`; ctx.textBaseline = "middle"; ctx.fillText(title, 72, 48);
    rows.forEach(([k, v, hot], i) => {
      const y = 120 + i * 104;
      ctx.fillStyle = "#e6ebf3"; ctx.beginPath(); ctx.roundRect(32, y, w - 64, 84, 24); ctx.fill();
      ctx.fillStyle = hot ? "#c0332b" : "#456cb3"; ctx.font = `700 34px ${FONT}`; ctx.fillText(k, 60, y + 42);
      ctx.fillStyle = "#1b2433"; ctx.font = `600 32px ${FONT}`; ctx.fillText(v, 300, y + 42);
    });
  }, w, h);
}

function floorPlanTexture(w = 1024, h = 576) {
  return softTexture((ctx) => {
    ctx.fillStyle = "#eef2f8"; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#d8dfeb";
    for (const [x, y, ww, hh] of [[60, 60, 360, 200], [460, 60, 500, 200], [60, 300, 900, 210]]) { ctx.beginPath(); ctx.roundRect(x, y, ww, hh, 18); ctx.fill(); }
    ctx.strokeStyle = "#c0332b"; ctx.lineWidth = 12; ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(140, 470); ctx.lineTo(140, 280); ctx.lineTo(440, 280); ctx.lineTo(440, 160); ctx.lineTo(820, 160); ctx.stroke();
    ctx.fillStyle = "#c0332b"; ctx.beginPath(); ctx.arc(820, 160, 26, 0, Math.PI * 2); ctx.fill();
    ctx.font = `700 34px ${FONT}`; ctx.fillText("HX-202", 760, 110);
  }, w, h);
}

function plateTexture(text) {
  return softTexture((ctx, w, h) => {
    ctx.clearRect(0, 0, w, h); ctx.fillStyle = "#f1f4f8"; ctx.beginPath(); ctx.roundRect(6, 6, w - 12, h - 12, 30); ctx.fill();
    ctx.fillStyle = "#1b2433"; ctx.font = `700 64px ${FONT}`; ctx.textBaseline = "middle"; ctx.textAlign = "center"; ctx.fillText(text, w / 2, h / 2 + 3);
  }, 512, 160);
}

export const CONTROL_SHOTS = {
  controlWide: { x: CX + 10, y: F + 15.5, z: 17, tx: CX - 1, ty: F + 8.5, tz: -6, fov: 40 },      // behind Xander: console, display, window
  consoleClose: { x: CX + 12.5, y: TOP + 3.2, z: 3.0, tx: CX + 5.5, ty: TOP + 1.2, tz: -2.4, fov: 32 }, // from the right: the console Atlas, Xander, the monitors
  plantView: { x: CX + 6, y: F + 20, z: 7, tx: CX + 4, ty: F + 4, tz: -20, fov: 42 },                // high over Xander, through the window: the plant floor and SPOT
};

export function buildControl(scene, humanGltf) {
  const root = new THREE.Group(); root.name = "control"; root.position.x = CX; scene.add(root);
  const set = fadingSet(root);
  const white = matte({ roughness: 0.8 }), wall = matte({ color: 0xe1e6ed, roughness: 0.95 }), ink = matte({ color: 0x3a404a, roughness: 0.5 });
  const steel = matte({ color: 0xc6ccd4, roughness: 0.4, metalness: 0.4 }), pipeM = tint(0x9aa7b8, { roughness: 0.5, metalness: 0.3 });
  const at = (m, x, y, z) => { m.position.set(x, y, z); return m; };

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 50), new THREE.ShadowMaterial({ opacity: 0.2, color: SHADE, depthWrite: false }));
  floor.rotation.x = -Math.PI / 2; floor.position.set(0, F, 0); floor.receiveShadow = true; floor.renderOrder = -5; set.add(floor);

  // The back wall with a wide window (a frame of four pieces) onto the plant floor behind it.
  const WZ = -6, WIN = { x0: -2, x1: 14, y0: F + 7, y1: F + 18 };
  const wallG = new THREE.Group();
  wallG.add(at(rbox(44, WIN.y0 - F, 0.4, wall, 0.08, 2), 1, (WIN.y0 + F) / 2, WZ));
  wallG.add(at(rbox(44, F + 26 - WIN.y1, 0.4, wall, 0.08, 2), 1, (WIN.y1 + F + 26) / 2, WZ));
  wallG.add(at(rbox(WIN.x0 + 21, WIN.y1 - WIN.y0, 0.4, wall, 0.08, 2), (WIN.x0 - 21) / 2, (WIN.y0 + WIN.y1) / 2, WZ));
  wallG.add(at(rbox(23 - WIN.x1, WIN.y1 - WIN.y0, 0.4, wall, 0.08, 2), (WIN.x1 + 23) / 2, (WIN.y0 + WIN.y1) / 2, WZ));
  wallG.children.forEach((m) => { m.castShadow = false; });
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(WIN.x1 - WIN.x0, WIN.y1 - WIN.y0), new THREE.MeshStandardMaterial({ color: 0xdfe8f4, transparent: true, opacity: 0.18, roughness: 0.1 }));
  glass.position.set((WIN.x0 + WIN.x1) / 2, (WIN.y0 + WIN.y1) / 2, WZ + 0.25); wallG.add(glass);
  set.add(wallG);

  // The big wall display left of the window: the Inspection page's "Your shift".
  const disp = new THREE.Group(); disp.position.set(-11.5, F + 13.5, WZ + 0.35);
  disp.add(at(rbox(16.4, 9.4, 0.3, ink, 0.2, 2), 0, 0, 0));
  const dScreen = new THREE.Mesh(new THREE.PlaneGeometry(15.8, 8.8), flat({ map: screenTexture("Inspection · Your shift", [["Do this first", "HX-202 · re-inspect · 88 h", true], ["Then", "HX-204 · send a technician"], ["Keep an eye", "GAUGE-209 · 48 h"], ["Missions", "96% say complete · most hide a failure"]]) }));
  dScreen.position.z = 0.17; disp.add(dScreen);
  set.add(disp);

  // The plant floor behind the window: a floor, pipes, heat exchangers with plates, a tank.
  const plant = new THREE.Group();
  const pfloor = new THREE.Mesh(new THREE.PlaneGeometry(40, 26), matte({ color: 0xd9dee6, roughness: 0.95 }));
  pfloor.rotation.x = -Math.PI / 2; pfloor.position.set(6, F + 0.01, -20); pfloor.receiveShadow = true; plant.add(pfloor);
  for (const [x, z, name] of [[1.5, -24, "HX-202"], [10.5, -24, "HX-204"]]) {
    const hx = new THREE.Mesh(new THREE.CylinderGeometry(1.6, 1.6, 7.5, 28), steel); hx.rotation.z = Math.PI / 2; hx.position.set(x, F + 2.6, z); hx.castShadow = true; plant.add(hx);
    for (const dx of [-2.6, 2.6]) plant.add(at(rbox(0.6, 2.2, 2.0, ink, 0.1, 2), x + dx, F + 1.1, z));
    const pl = new THREE.Mesh(new THREE.PlaneGeometry(3.0, 0.94), flat({ map: plateTexture(name), transparent: true })); pl.position.set(x, F + 4.9, z + 1.7); plant.add(pl);
  }
  for (const y of [F + 8.5, F + 10]) { const p = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 34, 14), pipeM); p.rotation.z = Math.PI / 2; p.position.set(6, y, -27); plant.add(p); }
  const tank = new THREE.Mesh(new THREE.CylinderGeometry(2.6, 2.6, 11, 28), white); tank.position.set(18, F + 5.5, -26); tank.castShadow = true; plant.add(tank);
  set.add(plant);

  // SPOT: a stylised quadruped (body, head, four legs) walking its round along the floor in front of the exchangers.
  const spot = new THREE.Group();
  const yellow = tint(0xe8c64a, { roughness: 0.55 }), dark = matte({ color: 0x2b2f36, roughness: 0.5 });
  spot.add(at(rbox(3.2, 0.9, 1.2, yellow, 0.25, 3), 0, 2.0, 0));
  spot.add(at(rbox(0.9, 0.6, 1.0, dark, 0.2, 2), 1.9, 2.15, 0));
  const legs = [];
  for (const [lx, lz] of [[1.2, 0.5], [1.2, -0.5], [-1.2, 0.5], [-1.2, -0.5]]) {
    const hip = new THREE.Group(); hip.position.set(lx, 1.75, lz);
    hip.add(at(rbox(0.3, 1.7, 0.3, dark, 0.12, 2), 0, -0.85, 0));
    spot.add(hip); legs.push(hip);
  }
  spot.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  spot.position.set(-4, F, -19.5);
  set.add(spot);

  // The console: a long desk, three monitors, a keyboard, the console Atlas, a chair.
  const desk = new THREE.Group();
  desk.add(at(rbox(18, 0.4, 4.2, white, 0.1, 2), 0, TOP - 0.2, -2.6));
  desk.add(at(rbox(17.6, TOP - F - 0.4, 3.6, white, 0.1, 2), 0, (TOP - 0.4 + F) / 2, -2.9));
  const monTex = [screenTexture("Your shift", [["1", "HX-202 · re-inspect", true], ["2", "HX-204 · a technician"], ["3", "GAUGE-209 · 48 h"]]), floorPlanTexture(),
    screenTexture("Shift Run", [["Your plan", "★★☆"], ["Atlas", "★★★"], ["Why", "far exchangers first"]])];
  [-5.4, 0, 5.4].forEach((x, k) => {
    const m = new THREE.Group(); m.position.set(x, TOP, -3.8); m.rotation.y = -x * 0.03;
    m.add(at(rbox(0.5, 1.6, 0.5, ink, 0.1, 2), 0, 0.8, 0));
    m.add(at(rbox(5.0, 2.9, 0.22, ink, 0.12, 2), 0, 3.0, 0));
    const s = new THREE.Mesh(new THREE.PlaneGeometry(4.7, 2.64), flat({ map: monTex[k] })); s.position.set(0, 3.0, 0.12); m.add(s);
    desk.add(m);
  });
  desk.add(at(rbox(3.4, 0.14, 1.1, matte({ color: 0xf4f6f9, roughness: 0.7 }), 0.05, 2), -1.2, TOP + 0.07, -1.6));
  const mug = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.38, 0.95, 18), tint(0xd9774a, { roughness: 0.6 })); mug.position.set(2.6, TOP + 0.48, -1.5); mug.castShadow = true; desk.add(mug);
  set.add(desk);
  const atlas = buildEnclosureLite();
  const atlasHolder = new THREE.Group(); atlasHolder.position.set(7.6, TOP - DESK_Y, -1.9); atlasHolder.rotation.y = -0.5; atlasHolder.add(atlas.group);
  atlas.group.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  set.add(atlasHolder);
  const chair = new THREE.Group(); chair.position.set(0.2, F, 2.0);
  chair.add(at(rbox(3.4, 0.5, 3.2, ink, 0.2, 2), 0, 3.9, 0));
  chair.add(at(rbox(3.2, 4.2, 0.4, ink, 0.2, 2), 0, 6.3, 1.6));
  chair.add(at(rbox(0.4, 3.6, 0.4, steel, 0.15, 2), 0, 1.9, 0));
  set.add(chair);

  // Xander, seated at the console facing the window (-z), and the trainee standing to the right.
  const xander = makeHuman(humanGltf, LOOKS.xander);
  const xHolder = new THREE.Group(); xHolder.position.set(0.6, F, 2.0); xHolder.rotation.y = Math.PI; xHolder.add(xander.root); root.add(xHolder);
  const trainee = makeHuman(humanGltf, LOOKS.trainee);
  const tHolder = new THREE.Group(); tHolder.position.set(6.4, F, -0.3); tHolder.rotation.y = Math.PI - 0.9; tHolder.add(trainee.root); root.add(tHolder);
  const people = [...xander.mats, ...trainee.mats];
  for (const m of people) { m.transparent = true; m.userData.baseOpacity = m.opacity; }

  const P = { control: [CX, 0, 0], consoleAtlas: [CX + 7.6, TOP + 0.5, -1.9], controlDisplay: [CX - 11.5, F + 13.5, WZ], spot: [CX - 4, F + 2, -19.5] };
  function update(state, t) {
    const v = state.control || 0; set.update(v);
    const pv = clamp01(v * 1.4);
    xHolder.visible = pv > 0.001;
    tHolder.visible = pv > 0.001 && (state.trainee || 0) > 0.001;
    if (!root.visible) return;
    for (const m of xander.mats) { m.opacity = m.userData.baseOpacity * pv; m.depthWrite = pv > 0.6; }
    for (const m of trainee.mats) { const tv = pv * clamp01(state.trainee || 0); m.opacity = m.userData.baseOpacity * tv; m.depthWrite = tv > 0.6; }
    xander.sample([["Sitting_Idle_Loop", (t * 0.6) % xander.duration("Sitting_Idle_Loop"), 1]]);
    xander.look(0.25 * Math.sin(t * 0.3), 0.15);
    xander.express({ talk: state.xTalk || 0, smile: 0.2 }, t);
    trainee.sample([["Idle_Loop", (t * 0.8) % trainee.duration("Idle_Loop"), 1]]);
    trainee.look(-0.3, 0.3);
    trainee.express({ smile: 0.3 }, t);
    // SPOT walks its round in front of the exchangers; its legs swing while it moves.
    const w = clamp01(state.spotWalk || 0);
    spot.position.x = lerp(-6, 14, w);
    const moving = w > 0.001 && w < 0.999 ? 1 : 0;
    legs.forEach((hip, k) => { hip.rotation.z = moving * 0.42 * Math.sin(t * 9 + (k % 2 ? Math.PI : 0) + (k > 1 ? Math.PI : 0)); });
    atlas.update(state.consoleCount || 0, t, state.consoleCount || 0);
  }
  return { root, update, P, xander, trainee, spot };
}
