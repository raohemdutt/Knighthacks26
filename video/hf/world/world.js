// The Atlas world: one Three.js scene the whole video lives in. See WORLD.md for the contract.
// Everything that moves is a number on world.cam / world.state / world.flows[id] / world.labels[id];
// render(t) is a pure function of those numbers and t. No clocks, no Math.random.
// Sets live in world/sets/: the enclosure (enclosure.js), the dorm room (room.js), the kitchen (kitchen.js), the person (human.js).
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { SURFACE_3D, SHADE, LIGHT, INK, ACCENT, MUTED, ENV_INTENSITY, DESK_Y, hash, clamp01, smooth, softTexture, flat, matte } from "./sets/common.js";
import { buildEnclosure, buildEnclosureLite } from "./sets/enclosure.js";
import { buildRoom } from "./sets/room.js";
import { buildKitchen, KX } from "./sets/kitchen.js";
import { buildControl, CX, CONTROL_SHOTS } from "./sets/control.js";
import { loadHuman } from "./sets/human.js";
import UCF from "../assets/ucf-route.json" with { type: "json" };

export { hash };
const W = 1920, H = 1080;
// Image-based light at a whisper, and filmic tone mapping with the exposure that keeps a lit top face at #eef2f8.
const ENV_SCENE = 0.6, EXPOSURE = 1.0;

// Legacy geometry constants (the old four-slab box). Kept so older scene files still load; do not build on them.
const SLAB = { w: 1.25, h: 0.2, d: 1.98, r: 0.03, pitch: 0.2, boxY: -0.6, open: 0.5 };

function shadowTexture() {
  return softTexture((ctx, w, h) => {
    const g = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    g.addColorStop(0, "rgba(183,192,205,0.7)"); g.addColorStop(0.35, "rgba(183,192,205,0.42)"); g.addColorStop(0.7, "rgba(183,192,205,0.1)"); g.addColorStop(1, "rgba(183,192,205,0)");
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
  });
}

function labelTexture(text, size) {
  const pad = 36, fs = size, w = 1024, h = 256;
  return softTexture((ctx, cw, ch) => {
    ctx.font = `600 ${fs}px Quicksand, "Segoe UI", system-ui, sans-serif`;
    const tw = Math.min(cw - 2 * pad, ctx.measureText(text).width + 2 * pad);
    const x = (cw - tw) / 2, y = (ch - fs * 1.9) / 2, hh = fs * 1.9, r = 28;
    ctx.shadowColor = "rgba(183,192,205,0.9)"; ctx.shadowBlur = 24; ctx.shadowOffsetX = 10; ctx.shadowOffsetY = 10;
    ctx.fillStyle = "#eef2f8"; ctx.beginPath(); ctx.roundRect(x, y, tw, hh, r); ctx.fill();
    ctx.shadowColor = "rgba(255,255,255,0.95)"; ctx.shadowOffsetX = -8; ctx.shadowOffsetY = -8; ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.fillStyle = "#1b2433"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, cw / 2, ch / 2 + 2);
  }, w, h);
}

// The small name plate on the front of each lab pod: "Trainer A" / "Trainer B".
function plateTexture(name) {
  return softTexture((ctx, w, h) => {
    ctx.clearRect(0, 0, w, h);
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#1b2433"; ctx.font = `700 78px Quicksand, "Segoe UI", system-ui, sans-serif`; ctx.fillText(name, 24, 92);
    ctx.fillStyle = "#566176"; ctx.font = `600 32px Quicksand, "Segoe UI", system-ui, sans-serif`; ctx.fillText("AMD STRIX HALO · RADEON 8060S", 26, 140);
    ctx.fillStyle = "#c9d3e3"; ctx.fillRect(24, 106, 440, 3);
  }, 640, 160);
}

// A tidy campus: a grid of blocks, paths between them, two avenues, a quad with a fountain in the middle.
function campusTexture() {
  return softTexture((ctx, w, h) => {
    ctx.clearRect(0, 0, w, h);
    const cell = 128, gap = 24, n = 8;
    const line = (x1, y1, x2, y2) => { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); };
    const rr = (x, y, ww, hh, r) => { ctx.beginPath(); ctx.roundRect(x, y, ww, hh, r); };
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (let k = 1; k < n; k++) {
      const p = k * cell, main = k === 4;
      ctx.strokeStyle = main ? "rgba(111,120,134,0.34)" : "rgba(111,120,134,0.22)"; ctx.lineWidth = main ? 16 : 7;
      line(p, 36, p, h - 36); line(36, p, w - 36, p);
    }
    const q0 = 3 * cell + gap * 0.5, qs = 2 * cell - gap;
    ctx.fillStyle = "rgba(255,255,255,0.55)"; rr(q0, q0, qs, qs, 46); ctx.fill();
    ctx.strokeStyle = "rgba(183,192,205,0.9)"; ctx.lineWidth = 5; rr(q0 + 14, q0 + 14, qs - 28, qs - 28, 36); ctx.stroke();
    ctx.strokeStyle = "rgba(183,192,205,0.7)"; ctx.lineWidth = 5; line(q0 + 30, q0 + 30, q0 + qs - 30, q0 + qs - 30); line(q0 + qs - 30, q0 + 30, q0 + 30, q0 + qs - 30);
    ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.beginPath(); ctx.arc(w / 2, h / 2, 30, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(183,192,205,0.95)"; ctx.lineWidth = 6; ctx.beginPath(); ctx.arc(w / 2, h / 2, 30, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "rgba(183,192,205,0.8)"; ctx.beginPath(); ctx.arc(w / 2, h / 2, 9, 0, Math.PI * 2); ctx.fill();
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const inQuad = i >= 3 && i <= 4 && j >= 3 && j <= 4; if (inQuad) continue;
      const x0 = i * cell + gap, y0 = j * cell + gap, size = cell - 2 * gap, k = i * 8 + j;
      const kind = hash(k, 13);
      if (kind < 0.18) continue;
      const parts = kind < 0.55 ? [[0, 0, 1, 1]] : kind < 0.8 ? [[0, 0, 1, 0.42], [0, 0.58, 0.55, 0.42]] : [[0, 0, 0.44, 1], [0.56, 0, 0.44, 0.62]];
      for (const [px, py, pw, ph] of parts) {
        const x = x0 + px * size, y = y0 + py * size, ww = pw * size, hh = ph * size;
        ctx.fillStyle = "rgba(183,192,205,0.42)"; rr(x, y, ww, hh, 12); ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x + 8, y + hh - 10); ctx.lineTo(x + 8, y + 8); ctx.lineTo(x + ww - 10, y + 8); ctx.stroke();
      }
    }
    ctx.fillStyle = "rgba(183,192,205,0.55)";
    for (let k = 0; k < 20; k++) { const p = 56 + k * 48; if (Math.abs(p - w / 2) < 150) continue; ctx.beginPath(); ctx.arc(p, 4 * cell - 24, 6, 0, Math.PI * 2); ctx.arc(4 * cell + 24, p, 6, 0, Math.PI * 2); ctx.fill(); }
  }, 1024, 1024);
}

// The privacy shell: a barely-there glass with a fresnel rim and one top-left glint. Not tone mapped (it is a veil).
function shellMaterial(side = THREE.FrontSide, gain = 1) {
  return new THREE.ShaderMaterial({
    uniforms: { uOpacity: { value: 0 }, uGain: { value: gain }, uCore: { value: new THREE.Color(LIGHT) }, uRim: { value: new THREE.Color(SHADE) } },
    vertexShader: `varying vec3 vN; varying vec3 vV;
      void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform float uOpacity; uniform float uGain; uniform vec3 uCore; uniform vec3 uRim; varying vec3 vN; varying vec3 vV;
      void main(){
        vec3 n = normalize(vN), v = normalize(vV);
        float f = pow(1.0 - abs(dot(n, v)), 2.6);
        vec3 l = normalize(vec3(-0.55, 0.75, 0.6));
        float spec = pow(max(dot(reflect(-l, n), v), 0.0), 80.0);
        vec3 c = mix(uCore, uRim, f * 0.92) + spec * 0.5;
        float a = (0.05 + 0.9 * f + spec * 0.45) * uOpacity * uGain;
        gl_FragColor = vec4(c, a);
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false, side,
  });
}

// The membrane wraps the enclosure's footprint: an ellipsoid (radii x, y, z), centred on the box.
const SHELL = { rx: 1.3, ry: 1.0, rz: 1.75, y: -0.4 };

export async function createWorld(canvas) {
  // Canvas-drawn labels, plates and screens use Quicksand (world/soft.css): load it before anything is drawn.
  try { await Promise.all(["600 40px Quicksand", "700 40px Quicksand"].map((f) => document.fonts.load(f))); } catch (e) { /* falls back to Segoe UI */ }
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(W, H, false); renderer.setPixelRatio(1);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.VSMShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = EXPOSURE;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, W / H, 0.05, 220);

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture; scene.environmentIntensity = ENV_SCENE;
  pmrem.dispose();

  // Light: one soft sky, one key from the top left (the same light the CSS shadows pretend), a weak fill from the right.
  // The key's shadow frustum follows the action: tight around the box in the studio, wide for the sets, the lab and the
  // friends, and it slides to the kitchen. A pure function of state (see draw).
  scene.add(new THREE.HemisphereLight(LIGHT, SHADE, 1.4));
  const KEY_DIR = new THREE.Vector3(-10, 15, 11).normalize();
  const key = new THREE.DirectionalLight(LIGHT, 0.7); key.castShadow = true; scene.add(key); scene.add(key.target);
  key.shadow.mapSize.set(2048, 2048); key.shadow.camera.near = 0.5; key.shadow.camera.far = 90;
  key.shadow.radius = 7; key.shadow.blurSamples = 16; key.shadow.bias = -0.0003; key.shadow.normalBias = 0.004;
  const fill = new THREE.DirectionalLight(LIGHT, 0.5); fill.position.set(7, 4, -6); scene.add(fill);
  let shadowKey = "";
  function aimShadow(cx, cz, ext) {
    const k = `${cx.toFixed(2)}|${cz.toFixed(2)}|${ext.toFixed(2)}`; if (k === shadowKey) return; shadowKey = k;
    key.target.position.set(cx, DESK_Y, cz); key.position.copy(key.target.position).addScaledVector(KEY_DIR, 40);
    const c = key.shadow.camera; c.left = -ext; c.right = ext; c.top = ext * 0.85; c.bottom = -ext * 0.85; c.updateProjectionMatrix();
    key.shadow.normalBias = 0.002 + ext * 0.0007; key.shadow.radius = ext < 6 ? 5 : 7;
  }

  // Desk: shadow receiver (invisible except for the shadow) + the soft contact shadow under the box.
  const desk = new THREE.Mesh(new THREE.PlaneGeometry(90, 90), new THREE.ShadowMaterial({ opacity: 0.22, color: SHADE, depthWrite: false }));
  desk.rotation.x = -Math.PI / 2; desk.position.y = DESK_Y; desk.receiveShadow = true; desk.renderOrder = -5; scene.add(desk);
  const contact = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 3.3), flat({ map: shadowTexture(), transparent: true, depthWrite: false }));
  contact.rotation.x = -Math.PI / 2; contact.position.set(0.06, DESK_Y + 0.003, 0.04); contact.renderOrder = -4; scene.add(contact);

  // The Atlas enclosure (world/sets/enclosure.js).
  const atlas = buildEnclosure(); scene.add(atlas.group);

  // The membrane: the privacy shell around the enclosure. Front rim + a fainter back rim read as thickness.
  const shellGeo = new THREE.SphereGeometry(1, 80, 52);
  const membrane = new THREE.Mesh(shellGeo, shellMaterial(THREE.FrontSide, 1)); membrane.renderOrder = 10;
  const membraneBack = new THREE.Mesh(shellGeo, shellMaterial(THREE.BackSide, 0.45)); membraneBack.renderOrder = 9;
  membrane.add(membraneBack); membrane.position.y = SHELL.y; scene.add(membrane);

  // The time ring on the desk, and its beads (scaled to the real box: radius 3).
  const RING_R = 3.0;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(RING_R, 0.028, 12, 200), new THREE.MeshStandardMaterial({ color: MUTED, roughness: 0.7, envMapIntensity: 0.4, transparent: true, opacity: 0 }));
  ring.rotation.x = Math.PI / 2; ring.position.y = DESK_Y + 0.028; ring.renderOrder = -2; scene.add(ring);
  const BEADS = 48, BEAD = 0.17;
  const beads = new THREE.InstancedMesh(new RoundedBoxGeometry(BEAD, BEAD, BEAD, 3, 0.05), matte({ roughness: 0.8 }), BEADS);
  beads.castShadow = true; beads.instanceMatrix.setUsage(THREE.DynamicDrawUsage); scene.add(beads);

  // The lab, far to the right: two compact AMD Strix Halo pods, "Trainer A" and "Trainer B", and the loss ribbon.
  const lab = new THREE.Group(); lab.position.set(14, 0, -2); scene.add(lab);
  const POD = { w: 1.6, h: 1.2, d: 1.6, r: 0.17 }, podY = DESK_Y + POD.h / 2;
  const podGeo = new RoundedBoxGeometry(POD.w, POD.h, POD.d, 7, POD.r);
  const plateGeo = new THREE.PlaneGeometry(1.0, 0.25);
  const wellGeo = new THREE.BoxGeometry(1.12, 0.46, 0.02), wellMat = new THREE.MeshStandardMaterial({ color: 0x59616e, roughness: 0.95, envMapIntensity: 0.2 });
  const finGeo = new THREE.BoxGeometry(1.12, 0.032, 0.03), finMat = matte({ roughness: 0.75 });
  const FINS = 7, fins = new THREE.InstancedMesh(finGeo, finMat, 2 * FINS); let fi = 0; const fm = new THREE.Matrix4();
  const ledGeo = new THREE.SphereGeometry(0.028, 16, 12);
  const pods = [-1.15, 1.15].map((x, i) => {
    const p = new THREE.Mesh(podGeo, matte({ roughness: 0.8 })); p.position.set(x, podY, 0); p.castShadow = true; p.receiveShadow = true; lab.add(p);
    const well = new THREE.Mesh(wellGeo, wellMat); well.position.set(0, 0.16, POD.d / 2 + 0.004); p.add(well);
    for (let k = 0; k < FINS; k++) { fm.makeTranslation(x, podY + 0.16 + (k - (FINS - 1) / 2) * 0.064, POD.d / 2 + 0.03); fins.setMatrixAt(fi++, fm); }
    const plate = new THREE.Mesh(plateGeo, flat({ map: plateTexture(i === 0 ? "Trainer A" : "Trainer B"), transparent: true, depthWrite: false })); plate.position.set(0.02, -0.3, POD.d / 2 + 0.006); p.add(plate);
    const led = new THREE.Mesh(ledGeo, flat({ color: SHADE })); led.position.set(-0.6, -0.42, POD.d / 2 + 0.01); p.add(led); p.userData.led = led;
    const band = new THREE.Mesh(new RoundedBoxGeometry(POD.w + 0.05, 0.05, POD.d + 0.05, 2, 0.02), flat({ color: ACCENT, transparent: true, opacity: 0, depthWrite: false })); band.position.y = -POD.h / 2 + 0.2; band.renderOrder = 5; p.add(band); p.userData.ring = band;
    return p;
  });
  fins.castShadow = false; lab.add(fins);
  const curvePts = []; for (let i = 0; i <= 60; i++) { const u = i / 60; curvePts.push(new THREE.Vector3(-2.5 + u * 5.0, 1.05 + Math.exp(-u * 3.2) * 1.5 + Math.sin(u * 18) * 0.08 * (1 - u), 0.4)); }
  const CURVE_SEG = 120, CURVE_RAD = 8;
  const curve = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(curvePts), CURVE_SEG, 0.05, CURVE_RAD, false), new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.5, toneMapped: false, envMapIntensity: 0.2, transparent: true, opacity: 0 }));
  curve.castShadow = true; lab.add(curve);

  // Friends' boxes around the desk: shell-only copies of the enclosure, each in its own privacy shell.
  const friends = new THREE.Group(); scene.add(friends);
  const FRIEND_R = 5.5; // scaled with the real box (was 9 for the old 2.4-wide box)
  const friendBoxes = [0, 1, 2].map((i) => {
    const a = (i / 3) * Math.PI * 2 + 0.6, g = new THREE.Group(); g.position.set(Math.cos(a) * FRIEND_R, 0, Math.sin(a) * FRIEND_R);
    g.rotation.y = Math.atan2(-Math.cos(a), -Math.sin(a)); // the honeycomb front faces the centre
    const copy = buildEnclosureLite(); g.add(copy.group); g.userData.copy = copy;
    const shell = new THREE.Mesh(shellGeo, shellMaterial(THREE.FrontSide, 1)); shell.renderOrder = 10;
    shell.scale.set(SHELL.rx, SHELL.ry, SHELL.rz); shell.position.y = SHELL.y; g.add(shell); g.userData.shell = shell; friends.add(g); return g;
  });
  const linkGeo = new THREE.BufferGeometry(); const linkMat = new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0, toneMapped: false });
  const links = new THREE.LineSegments(linkGeo, linkMat); scene.add(links);

  // The campus map under everything: the real UCF campus (OpenStreetMap) in the website's map palette, baked by
// video/scripts/campus_bake.py; the route is Day Run's walk on the real footpaths (dorm, class, library, the Union).
const campusTex = await new THREE.TextureLoader().loadAsync("assets/ucf-campus.png");
campusTex.colorSpace = THREE.SRGBColorSpace; campusTex.anisotropy = 8;
const campus = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), flat({ map: campusTex, transparent: true, opacity: 0, depthWrite: false }));
campus.rotation.x = -Math.PI / 2; campus.position.y = DESK_Y + 0.01; campus.renderOrder = -3; scene.add(campus);
const routePts = UCF.route.map(([u, v]) => new THREE.Vector3((u - 0.5) * 30, DESK_Y + 0.06, (v - 0.5) * 30));
const routeCurve = new THREE.CatmullRomCurve3(routePts, false, "catmullrom", 0.1);
const route = new THREE.Mesh(new THREE.TubeGeometry(routeCurve, 400, 0.07, 8, false), new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.5, toneMapped: false, emissive: ACCENT, emissiveIntensity: 0.25, transparent: true }));
const routeCasing = new THREE.Mesh(new THREE.TubeGeometry(routeCurve, 400, 0.12, 8, false), flat({ color: 0xffffff, transparent: true, depthWrite: false }));
routeCasing.renderOrder = 1; route.renderOrder = 2; route.add(routeCasing); // the site's route: a white casing under the blue line
scene.add(route);

  // The Ethernet cable: from the jack on the back wall it droops to the desk and trails away. Turns with the box.
  const cableRig = new THREE.Group(); scene.add(cableRig);
  const cableSlide = new THREE.Group(); cableRig.add(cableSlide);
  const E = atlas.P.ethernet; // [x, y, z] on the back wall's outer face
  const cablePts = [[E[0], E[1], E[2] - 0.1], [E[0] - 0.02, E[1] - 0.01, E[2] - 0.3], [E[0] - 0.1, DESK_Y + 0.05, E[2] - 0.62], [E[0] - 0.35, DESK_Y + 0.022, E[2] - 1.0], [E[0] - 0.9, DESK_Y + 0.022, E[2] - 1.6]].map((p) => new THREE.Vector3(...p));
  const cableMat = new THREE.MeshStandardMaterial({ color: INK, roughness: 0.5, envMapIntensity: 0.6, transparent: true });
  const cable = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(cablePts, false, "catmullrom", 0.5), 48, 0.022, 10, false), cableMat);
  cable.castShadow = true;
  const plugMat = new THREE.MeshStandardMaterial({ color: INK, roughness: 0.45, envMapIntensity: 0.6, transparent: true });
  const plug = new THREE.Mesh(new RoundedBoxGeometry(0.13, 0.1, 0.18, 2, 0.02), plugMat);
  plug.position.set(E[0], E[1], E[2] - 0.075); plug.castShadow = true;
  cableSlide.add(cable, plug);

  // The sets.
  const humanGltf = await loadHuman(); // the person: Quaternius' Universal Animation Library mannequin (CC0)
  const room = buildRoom(scene, humanGltf);
  const kitchen = buildKitchen(scene, humanGltf);
  const control = buildControl(scene, humanGltf); // the plant control room (film v4.7, beat 16), x = CX

  // Studio three-quarter of the enclosure (no room): from the front-right, a touch high.
  const HOME = { x: 3.0, y: 1.1, z: 4.6, tx: 0, ty: -0.42, tz: 0.05, fov: 30 };
  const SHOTS = {
    home: HOME,
    front: { x: 0, y: -0.2, z: 5.4, tx: 0, ty: -0.42, tz: 0, fov: 28 },
    hero: { x: -3.2, y: -0.3, z: 4.4, tx: 0, ty: -0.3, tz: 0, fov: 30 },                        // low left three-quarter
    exploded: { x: 5.4, y: 2.2, z: 7.0, tx: 0, ty: 0.05, tz: 0.2, fov: 36 },                    // the whole exploded diagram
    xray: { x: 2.7, y: 1.7, z: 3.9, tx: 0, ty: -0.55, tz: -0.05, fov: 30 },                     // a high three-quarter into the box
    deskClose: { x: 2.4, y: 0.35, z: 4.4, tx: -0.2, ty: -0.45, tz: 0, fov: 30 },                // the device on its shelf (studio or room)
    roomWide: { x: 30, y: 7.0, z: 42, tx: 3.5, ty: -5.5, tz: 6.5, fov: 40 },                    // the dorm: bed, chair, desk, the shelf
    overShoulder: { x: 15.0, y: 8.5, z: 25.0, tx: -3.5, ty: -4.0, tz: 5.0, fov: 35 },            // behind the seated person: desk ahead, Atlas on their right
    kitchenWide: { x: KX - 5, y: 8.5, z: 44, tx: KX + 2.5, ty: 0.5, tz: 0, fov: 40 },
    binCam: { x: KX - 8.4, y: 13.0, z: -2.0, tx: KX - 6.2, ty: 1.4, tz: 3.6, fov: 76 },        // from the wall camera
    counterAtlas: { x: KX + 6.0, y: 1.9, z: 3.0, tx: KX + 3.3, ty: 0.9, tz: -1.9, fov: 30 },
    ...CONTROL_SHOTS,
  };
  const cam = { ...HOME };
  const state = {
    // The enclosure.
    explode: 0, xray: 0, hl: { webcam: 0, jetson: 0, speaker: 0, lid: 0, front: 0 }, spin: 0, drop: 0, eyes: 0, pulse: 0, cable: 0,
    // Legacy aliases (older scene files): open = explode, layer[0] = hl.webcam, layer[1..3] = hl.jetson.
    open: 0, layer: [0, 0, 0, 0],
    // Axes.
    membrane: 0, ring: 0, beads: 0, bars: 0, lab: 0, labGlow: 0, labCurve: 0, friends: 0, friendsLink: 0, campus: 0, route: 0, blur: 0,
    // Sets.
    room: 0, person: 0, personLook: 0, personSmile: 0, personSleepy: 0, personStrain: 0, personTalk: 0, kitchen: 0, canThrow: 0, binCount: 0,
    control: 0, spotWalk: 0, trainee: 0, xTalk: 0, consoleCount: 0,
    light: 1, // the room's light level: 1 = normal; the morning starts dark (about 0.22) and the lights come up
  };
  const P = {
    ...atlas.P, lab: [14, 0.8, -2], friend: friendBoxes.map((g) => [g.position.x, -0.4, g.position.z]),
    ring: (a) => [Math.cos(a) * RING_R, -0.9, Math.sin(a) * RING_R],
    ...room.P, ...kitchen.P, ...control.P,
  };
  /** Legacy: world-space height of old layer i, mapped onto the new parts (0 = webcam, 1..3 = the Jetson). */
  const layerY = (i) => (i === 0 ? atlas.P.webcam[1] : atlas.P.jetson[1] + (i - 2) * 0.1);

  // Particle flows: declared once, driven by flows[id].progress / density.
  const flows = {}, flowMeshes = {};
  const dot = new THREE.SphereGeometry(0.07, 10, 8);
  function flow(id, { from, to, count = 24, color = ACCENT, size = 1, lift = 2.2 }) {
    if (flows[id]) return flows[id];
    const mesh = new THREE.InstancedMesh(dot, new THREE.MeshStandardMaterial({ color, roughness: 0.45, toneMapped: false, envMapIntensity: 0.2, transparent: true, opacity: 0 }), count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); mesh.castShadow = false; mesh.renderOrder = 15; scene.add(mesh);
    const a = new THREE.Vector3(...from), b = new THREE.Vector3(...to);
    const mid = a.clone().lerp(b, 0.5); mid.y += lift;
    flowMeshes[id] = { mesh, curve: new THREE.QuadraticBezierCurve3(a, mid, b), count, size };
    flows[id] = { progress: 0, density: 1, spread: 0.3 };
    return flows[id];
  }

  // Labels: soft cards at 3D points, always facing the camera. scale 1 = 4 x 1 units (use ~0.25 near the box).
  const labels = {}, labelSprites = {};
  function label(id, { text, at, size = 56, scale = 1 }) {
    if (labels[id]) return labels[id];
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTexture(text, size), transparent: true, opacity: 0, depthWrite: false, depthTest: false, toneMapped: false }));
    sp.scale.set(4 * scale, 1 * scale, 1); sp.position.set(...at); sp.renderOrder = 20; scene.add(sp);
    labelSprites[id] = { sprite: sp, base: at.slice(), scale }; labels[id] = { opacity: 0, lift: 0 };
    return labels[id];
  }

  const accentC = new THREE.Color(ACCENT), shadeC = new THREE.Color(SHADE);
  const m4 = new THREE.Matrix4(), v3 = new THREE.Vector3(), q = new THREE.Quaternion(), s3 = new THREE.Vector3(), yAxis = new THREE.Vector3(0, 1, 0);

  function draw(t) {
    renderer.toneMappingExposure = EXPOSURE * Math.max(0.05, state.light ?? 1);
    camera.position.set(cam.x, cam.y, cam.z); camera.lookAt(cam.tx, cam.ty, cam.tz); camera.fov = cam.fov; camera.updateProjectionMatrix();
    // Shadows follow the action: tight in the studio, wide for the sets / lab / friends / campus, slid to the kitchen.
    const wide = clamp01(Math.max(state.lab, state.friends, state.campus, state.room, state.kitchen, state.control, state.ring * 0.6, state.explode * 0.25));
    const kx = KX * smooth(0.25, 0.75, state.kitchen) + CX * smooth(0.25, 0.75, state.control);
    aimShadow(kx * 1, 0, 3.2 + 22.8 * smooth(0, 1, wide));

    // The enclosure.
    atlas.update(state, t);
    const d = state.drop, ex = clamp01(Math.max(state.explode, state.open));
    contact.material.opacity = (1 - clamp01(d * 1.6)) * (1 - ex * 0.6);
    contact.scale.setScalar(1 + ex * 0.5);
    // The studio shadow plane gives way to the sets' own surfaces.
    desk.material.opacity = 0.22 * (1 - clamp01(state.room * 1.5)) * (1 - clamp01(state.kitchen * 1.5)) * (1 - clamp01(state.control * 1.5)); desk.visible = desk.material.opacity > 0.002;

    // The membrane drifts a little, like something alive.
    membrane.material.uniforms.uOpacity.value = state.membrane; membraneBack.material.uniforms.uOpacity.value = state.membrane;
    membrane.visible = state.membrane > 0.002;
    const br = 1 + 0.008 * Math.sin(t * 0.7 + 1);
    membrane.position.y = SHELL.y + 0.015 * Math.sin(t * 0.45); membrane.scale.set(SHELL.rx * br, SHELL.ry * br, SHELL.rz * br);

    // The ring and its beads: beads appear in order; `bars` stretches them upward into columns.
    ring.material.opacity = state.ring * 0.9; ring.visible = state.ring > 0.002;
    beads.visible = state.beads > 0.001;
    if (beads.visible) {
      for (let i = 0; i < BEADS; i++) {
        const a = (i / BEADS) * Math.PI * 2 - Math.PI / 2;
        const on = clamp01(state.beads - i), h = BEAD + state.bars * (0.25 + hash(i, 1) * 1.2);
        v3.set(Math.cos(a) * RING_R, DESK_Y + (h / 2) * on, Math.sin(a) * RING_R); q.setFromAxisAngle(yAxis, -a); s3.set(on, on * (h / BEAD), on);
        m4.compose(v3, q, s3); beads.setMatrixAt(i, m4);
      }
      beads.instanceMatrix.needsUpdate = true; beads.material.emissiveIntensity = state.bars * 0.3;
    }

    // The lab: the pods rise and settle in; `labGlow` warms them (watts); `labCurve` draws the loss ribbon.
    lab.visible = state.lab > 0.001; lab.scale.setScalar(0.6 + 0.4 * state.lab); lab.position.y = (1 - state.lab) * -1.5;
    pods.forEach((p, i) => {
      const g = clamp01(state.labGlow) * (0.92 + 0.08 * Math.sin(t * 3 + i));
      p.material.color.set(SURFACE_3D).lerp(accentC, g * 0.32); p.material.emissiveIntensity = g * 0.24;
      p.userData.led.material.color.copy(shadeC).lerp(accentC, clamp01(state.labGlow * 3));
      p.userData.ring.material.opacity = g * 0.85; p.userData.ring.visible = g > 0.002;
    });
    curve.material.opacity = state.labCurve > 0 ? 1 : 0; curve.visible = state.labCurve > 0.001;
    curve.geometry.setDrawRange(0, Math.floor(clamp01(state.labCurve) * CURVE_SEG) * CURVE_RAD * 6);

    // Friends and their links.
    friends.visible = state.friends > 0.001;
    if (friends.visible) friendBoxes.forEach((g) => { g.scale.setScalar(0.001 + state.friends); g.userData.shell.material.uniforms.uOpacity.value = state.friends * 0.85; g.userData.copy.update(state.friendsLink * 0.5, t); });
    if (state.friendsLink > 0) {
      const pts = []; const c = new THREE.Vector3(0, -0.4, 0);
      friendBoxes.forEach((g) => { const f = new THREE.Vector3(g.position.x, -0.4, g.position.z); const e = c.clone().lerp(f, state.friendsLink); pts.push(c.x, c.y, c.z, e.x, e.y, e.z); });
      linkGeo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3)); linkMat.opacity = 0.9; links.visible = true;
    } else links.visible = false;

    // Campus and route.
    campus.material.opacity = state.campus * 0.9; campus.visible = state.campus > 0.002;
    route.material.opacity = state.route > 0 ? 1 : 0; route.visible = state.route > 0.001; route.geometry.setDrawRange(0, Math.floor(state.route * 400 * 8 * 6)); routeCasing.geometry.setDrawRange(0, Math.floor(state.route * 400 * 8 * 6));

    // The cable turns with the box; `cable` pulls the plug back out of the jack and lets it drop; hidden while the
    // parts are in the air or the box is exploded.
    cableRig.rotation.y = (state.spin || 0) * Math.PI * 2;
    const cp = clamp01(state.cable);
    cableSlide.position.z = -cp * 0.55; cableSlide.position.y = -cp * 0.07; cableSlide.rotation.y = cp * 0.25;
    cableMat.opacity = plugMat.opacity = (1 - clamp01(d * 2)) * (1 - clamp01(ex * 3));
    cableRig.visible = cableMat.opacity > 0.01;

    // The sets.
    room.update(state, t);
    kitchen.update(state, t);
    control.update(state, t);

    // Particle flows.
    for (const id in flowMeshes) {
      const { mesh, curve: cv, count, size } = flowMeshes[id], f = flows[id];
      mesh.material.opacity = f.density; mesh.visible = f.density > 0.001 && f.progress > 0;
      if (!mesh.visible) continue;
      for (let i = 0; i < count; i++) {
        const u = clamp01(f.progress * (1 + f.spread) - hash(i, 5) * f.spread);
        cv.getPoint(u, v3); v3.x += (hash(i, 2) - 0.5) * 0.5; v3.y += (hash(i, 3) - 0.5) * 0.5; v3.z += (hash(i, 4) - 0.5) * 0.5;
        const sc = size * (u > 0 && u < 1 ? 1 : 0.001) * (0.7 + 0.6 * hash(i, 6));
        m4.compose(v3, q.identity(), s3.set(sc, sc, sc)); mesh.setMatrixAt(i, m4);
      }
      mesh.instanceMatrix.needsUpdate = true;
    }
    // Labels.
    for (const id in labelSprites) { const { sprite, base, scale } = labelSprites[id], l = labels[id]; sprite.material.opacity = l.opacity; sprite.position.set(base[0], base[1] + l.lift * scale, base[2]); sprite.visible = l.opacity > 0.001; }
    renderer.render(scene, camera);
  }

  // render(t) draws once per JavaScript task, after every tween of that tick has applied: the root timeline's clock
  // tween (the one that calls render) is the first child of the timeline, so it would otherwise draw before the slot
  // tweens of the same tick had updated cam/state. Still a pure function of (state, t); draw(t) is the synchronous form.
  let queued = false, lastT = 0;
  function render(t) { lastT = t; if (queued) return; queued = true; queueMicrotask(() => { queued = false; draw(lastT); }); }

  function leg(tl, at, to, dur = 1.6, ease = "power2.inOut") { tl.to(cam, { ...to, duration: dur, ease }, at); }
  /** Anchor (or part) position at explode e, e.g. world.at("jetson", 1) for a label on the exploded diagram. */
  const at = (name, e = 0) => atlas.at(name, e);

  const world = {
    THREE, scene, camera, renderer, cam, HOME, SHOTS, state, P, at, flows, flow, labels, label, render, draw, leg, hash,
    atlas, parts: atlas.parts, box: atlas.group, membrane, lab, pods, friends: friendBoxes, room, kitchen, KX, control, CX,
    SLAB, layerY, // legacy
    ucfStops: UCF.stops, // Day Run's stops on the real campus: [{ name, kind, u, v }], u east / v south in 0..1
  };
  draw(0);
  return world;
}
