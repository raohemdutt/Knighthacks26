// The Atlas world: one Three.js scene the whole video lives in. See WORLD.md for the contract.
// Everything that moves is a number on world.cam / world.state / world.flows[id] / world.labels[id];
// render(t) is a pure function of those numbers and t. No clocks, no Math.random.
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const SURFACE = 0xe4e8ee, SHADE = 0xb7c0cd, LIGHT = 0xffffff, INK = 0x2b313b, ACCENT = 0xd9774a, MUTED = 0x6f7886;
const W = 1920, H = 1080;
// Image-based light at a whisper, and filmic tone mapping with the exposure that keeps a lit top face at #e4e8ee.
// Measured on the hero shot: top face ≈ #eff0f1, lit front ≈ #dfe1e3, shaded side ≈ #cdd0d3 against the #e4e8ee page.
const ENV_INTENSITY = 0.35, ENV_SCENE = 0.6, EXPOSURE = 1.0;
// ACES pulls the cool cast out of a lit surface, so the 3D base colour sits a touch bluer than the CSS token.
const SURFACE_3D = 0xdfe6f0;

// The box's geometry, shared by the box, the friends' boxes and the slot files (world.SLAB).
const SLAB = { w: 2.4, h: 0.38, d: 1.6, r: 0.13, pitch: 0.4, boxY: -0.2, open: 0.55 };
const slabY = (i) => i * SLAB.pitch - 0.6; // rest y of layer i inside the box group
const DESK_Y = -1.0;

/** Deterministic 0..1 hash of two integers (same every frame, every render). */
export function hash(i, j = 0) {
  let n = (i * 374761393 + j * 668265263) | 0;
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const lerp = (a, b, t) => a + (b - a) * t;
/** A slow heartbeat, period 2 s: a strong bump and a softer echo. Pure function of t. */
const bump = (x, w) => Math.exp(-(x * x) / (w * w));
const heartbeat = (t) => { const p = (t % 2.0) / 2.0; return bump(p - 0.12, 0.07) + 0.55 * bump(p - 0.36, 0.09); };

function softTexture(draw, w = 512, h = 512) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  return tex;
}

function shadowTexture() {
  return softTexture((ctx, w, h) => {
    const g = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    g.addColorStop(0, "rgba(183,192,205,0.66)"); g.addColorStop(0.3, "rgba(183,192,205,0.40)"); g.addColorStop(0.65, "rgba(183,192,205,0.12)"); g.addColorStop(1, "rgba(183,192,205,0)");
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
  });
}

function labelTexture(text, size) {
  const pad = 36, fs = size, w = 1024, h = 256;
  return softTexture((ctx, cw, ch) => {
    ctx.font = `600 ${fs}px Inter, "Segoe UI", system-ui, sans-serif`;
    const tw = Math.min(cw - 2 * pad, ctx.measureText(text).width + 2 * pad);
    const x = (cw - tw) / 2, y = (ch - fs * 1.9) / 2, hh = fs * 1.9, r = 28;
    ctx.shadowColor = "rgba(183,192,205,0.9)"; ctx.shadowBlur = 24; ctx.shadowOffsetX = 10; ctx.shadowOffsetY = 10;
    ctx.fillStyle = "#e4e8ee"; ctx.beginPath(); ctx.roundRect(x, y, tw, hh, r); ctx.fill();
    ctx.shadowColor = "rgba(255,255,255,0.95)"; ctx.shadowOffsetX = -8; ctx.shadowOffsetY = -8; ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.fillStyle = "#2b313b"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, cw / 2, ch / 2 + 2);
  }, w, h);
}

// The small name plate on the front of each lab pod.
function plateTexture() {
  return softTexture((ctx, w, h) => {
    ctx.clearRect(0, 0, w, h);
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#2b313b"; ctx.font = `700 78px Inter, "Segoe UI", system-ui, sans-serif`; ctx.fillText("STRIX HALO", 24, 92);
    ctx.fillStyle = "#6f7886"; ctx.font = `600 32px Inter, "Segoe UI", system-ui, sans-serif`; ctx.fillText("AMD RYZEN AI MAX+ · RADEON 8060S", 26, 140);
    ctx.fillStyle = "#b7c0cd"; ctx.fillRect(24, 106, 440, 3);
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
    // Paths on the grid lines; the two central ones are avenues.
    for (let k = 1; k < n; k++) {
      const p = k * cell, main = k === 4;
      ctx.strokeStyle = main ? "rgba(111,120,134,0.34)" : "rgba(111,120,134,0.22)"; ctx.lineWidth = main ? 16 : 7;
      line(p, 36, p, h - 36); line(36, p, w - 36, p);
    }
    // The quad: the four centre cells, a lighter lawn with a ring path and a fountain.
    const q0 = 3 * cell + gap * 0.5, qs = 2 * cell - gap;
    ctx.fillStyle = "rgba(255,255,255,0.55)"; rr(q0, q0, qs, qs, 46); ctx.fill();
    ctx.strokeStyle = "rgba(183,192,205,0.9)"; ctx.lineWidth = 5; rr(q0 + 14, q0 + 14, qs - 28, qs - 28, 36); ctx.stroke();
    ctx.strokeStyle = "rgba(183,192,205,0.7)"; ctx.lineWidth = 5; line(q0 + 30, q0 + 30, q0 + qs - 30, q0 + qs - 30); line(q0 + qs - 30, q0 + 30, q0 + 30, q0 + qs - 30);
    ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.beginPath(); ctx.arc(w / 2, h / 2, 30, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(183,192,205,0.95)"; ctx.lineWidth = 6; ctx.beginPath(); ctx.arc(w / 2, h / 2, 30, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "rgba(183,192,205,0.8)"; ctx.beginPath(); ctx.arc(w / 2, h / 2, 9, 0, Math.PI * 2); ctx.fill();
    // Blocks in every other cell: a rounded footprint with a light top-left edge, like the CSS cards.
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const inQuad = i >= 3 && i <= 4 && j >= 3 && j <= 4; if (inQuad) continue;
      const x0 = i * cell + gap, y0 = j * cell + gap, size = cell - 2 * gap, k = i * 8 + j;
      const kind = hash(k, 13);
      if (kind < 0.18) continue; // a lawn
      const parts = kind < 0.55 ? [[0, 0, 1, 1]] : kind < 0.8 ? [[0, 0, 1, 0.42], [0, 0.58, 0.55, 0.42]] : [[0, 0, 0.44, 1], [0.56, 0, 0.44, 0.62]];
      for (const [px, py, pw, ph] of parts) {
        const x = x0 + px * size, y = y0 + py * size, ww = pw * size, hh = ph * size;
        ctx.fillStyle = "rgba(183,192,205,0.42)"; rr(x, y, ww, hh, 12); ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x + 8, y + hh - 10); ctx.lineTo(x + 8, y + 8); ctx.lineTo(x + ww - 10, y + 8); ctx.stroke();
      }
    }
    // Trees along the avenues.
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

export async function createWorld(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(W, H, false); renderer.setPixelRatio(1);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.VSMShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = EXPOSURE;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, W / H, 0.1, 200);

  // Image-based light: a neutral room, blurred, at a whisper, so matte surfaces pick up soft real reflections.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture; scene.environmentIntensity = ENV_SCENE;
  pmrem.dispose();

  // Light: one soft sky, one key from the top left (the same light the CSS shadows pretend), a weak fill from the right.
  scene.add(new THREE.HemisphereLight(LIGHT, SHADE, 1.4));
  const key = new THREE.DirectionalLight(LIGHT, 0.7); key.position.set(-10, 15, 11); key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -26; key.shadow.camera.right = 26; key.shadow.camera.top = 22; key.shadow.camera.bottom = -22; key.shadow.camera.near = 0.5; key.shadow.camera.far = 70;
  key.shadow.radius = 7; key.shadow.blurSamples = 16; key.shadow.bias = -0.0003; key.shadow.normalBias = 0.02;
  scene.add(key);
  const fill = new THREE.DirectionalLight(LIGHT, 0.5); fill.position.set(7, 4, -6); scene.add(fill);

  const flat = (extra = {}) => new THREE.MeshBasicMaterial({ toneMapped: false, ...extra }); // exact sRGB colours: labels, maps, plates, accent lines
  const matte = (extra = {}) => new THREE.MeshStandardMaterial({ color: SURFACE_3D, roughness: 0.82, metalness: 0, envMapIntensity: ENV_INTENSITY, emissive: ACCENT, emissiveIntensity: 0, ...extra });

  // Desk: shadow receiver (invisible except for the shadow) + the soft contact shadow under the box.
  const desk = new THREE.Mesh(new THREE.PlaneGeometry(90, 90), new THREE.ShadowMaterial({ opacity: 0.22, color: SHADE, depthWrite: false })); // invisible: must never occlude
  desk.rotation.x = -Math.PI / 2; desk.position.y = DESK_Y; desk.receiveShadow = true; desk.renderOrder = -5; scene.add(desk);
  const contact = new THREE.Mesh(new THREE.PlaneGeometry(5.8, 4.6), flat({ map: shadowTexture(), transparent: true, depthWrite: false }));
  contact.rotation.x = -Math.PI / 2; contact.position.set(0.14, DESK_Y + 0.004, 0.12); contact.renderOrder = -4; scene.add(contact);

  // The box: four slabs, cameras at the bottom, agent at the top. One object: tight stack, soft radius, a faint seam.
  const box = new THREE.Group(); box.position.y = SLAB.boxY; scene.add(box);
  const slabGeo = new RoundedBoxGeometry(SLAB.w, SLAB.h, SLAB.d, 7, SLAB.r);
  const seamGeo = new RoundedBoxGeometry(SLAB.w - 0.3, 0.17, SLAB.d - 0.3, 2, 0.04);
  // A thin accent band around each slab's waist: the seam glow of a lit layer (flat colour, exact accent).
  const rimGeo = new RoundedBoxGeometry(SLAB.w + 0.05, 0.05, SLAB.d + 0.05, 2, 0.02);
  const layers = [], seams = [], rims = [];
  for (let i = 0; i < 4; i++) {
    const m = new THREE.Mesh(slabGeo, matte()); m.castShadow = true; m.receiveShadow = true; m.position.y = slabY(i); box.add(m); layers.push(m);
    const rim = new THREE.Mesh(rimGeo, flat({ color: ACCENT, transparent: true, opacity: 0, depthWrite: false })); rim.renderOrder = 5; m.add(rim); rims.push(rim);
    if (i < 3) { const s = new THREE.Mesh(seamGeo, new THREE.MeshStandardMaterial({ color: 0xc9d1db, roughness: 1, envMapIntensity: 0.1, transparent: true })); box.add(s); seams.push(s); }
  }
  // Two camera eyes on the front of the bottom layer: a bezel, a domed lens that catches the room, a glint, a halo.
  const lensGeo = new THREE.SphereGeometry(0.056, 28, 16);
  function makeEye(x) {
    const g = new THREE.Group(); g.position.set(x, 0, SLAB.d / 2 + 0.003);
    const bezel = new THREE.Mesh(new THREE.CircleGeometry(0.08, 48), new THREE.MeshStandardMaterial({ color: SHADE, roughness: 0.5, envMapIntensity: 0.5 }));
    const lens = new THREE.Mesh(lensGeo, new THREE.MeshStandardMaterial({ color: INK, roughness: 0.16, metalness: 0.05, envMapIntensity: 1.1, emissive: ACCENT, emissiveIntensity: 0 }));
    lens.position.z = 0.004; lens.scale.set(1, 1, 0.42);
    const glint = new THREE.Mesh(new THREE.CircleGeometry(0.011, 16), flat({ color: LIGHT })); glint.position.set(-0.018, 0.02, 0.03);
    const halo = new THREE.Mesh(new THREE.RingGeometry(0.082, 0.1, 48), flat({ color: ACCENT, transparent: true, opacity: 0, depthWrite: false })); halo.position.z = 0.002;
    g.add(bezel, lens, glint, halo); g.userData.lens = lens; g.userData.halo = halo; return g;
  }
  const eyes = [-0.45, 0.45].map((x) => { const e = makeEye(x); layers[0].add(e); return e; });

  // The membrane: the privacy shell. Nothing leaves it uncounted. Front rim + a fainter back rim read as thickness.
  const shellGeo = new THREE.SphereGeometry(2.6, 80, 52);
  const membrane = new THREE.Mesh(shellGeo, shellMaterial(THREE.FrontSide, 1)); membrane.renderOrder = 10;
  const membraneBack = new THREE.Mesh(shellGeo, shellMaterial(THREE.BackSide, 0.45)); membraneBack.renderOrder = 9;
  membrane.add(membraneBack); membrane.position.y = 0.2; scene.add(membrane);

  // The time ring on the desk, and its beads.
  const ring = new THREE.Mesh(new THREE.TorusGeometry(4, 0.035, 12, 200), new THREE.MeshStandardMaterial({ color: MUTED, roughness: 0.7, envMapIntensity: 0.4, transparent: true, opacity: 0 }));
  ring.rotation.x = Math.PI / 2; ring.position.y = DESK_Y + 0.035; ring.renderOrder = -2; scene.add(ring);
  const BEADS = 48;
  const beads = new THREE.InstancedMesh(new RoundedBoxGeometry(0.22, 0.22, 0.22, 3, 0.07), matte({ roughness: 0.8 }), BEADS);
  beads.castShadow = true; beads.instanceMatrix.setUsage(THREE.DynamicDrawUsage); scene.add(beads);

  // The lab, far to the right: two compact AMD Strix Halo pods (front grille, name plate, one status light) and the loss ribbon.
  const lab = new THREE.Group(); lab.position.set(14, 0, -2); scene.add(lab);
  const POD = { w: 1.6, h: 1.2, d: 1.6, r: 0.17 }, podY = DESK_Y + POD.h / 2;
  const podGeo = new RoundedBoxGeometry(POD.w, POD.h, POD.d, 7, POD.r);
  const plateTex = plateTexture(), plateGeo = new THREE.PlaneGeometry(1.0, 0.25);
  const wellGeo = new THREE.BoxGeometry(1.12, 0.46, 0.02), wellMat = new THREE.MeshStandardMaterial({ color: 0x59616e, roughness: 0.95, envMapIntensity: 0.2 });
  const finGeo = new THREE.BoxGeometry(1.12, 0.032, 0.03), finMat = matte({ roughness: 0.75 });
  const FINS = 7, fins = new THREE.InstancedMesh(finGeo, finMat, 2 * FINS); let fi = 0; const fm = new THREE.Matrix4();
  const ledGeo = new THREE.SphereGeometry(0.028, 16, 12);
  const pods = [-1.15, 1.15].map((x) => {
    const p = new THREE.Mesh(podGeo, matte({ roughness: 0.8 })); p.position.set(x, podY, 0); p.castShadow = true; p.receiveShadow = true; lab.add(p);
    const well = new THREE.Mesh(wellGeo, wellMat); well.position.set(0, 0.16, POD.d / 2 + 0.004); p.add(well);
    for (let k = 0; k < FINS; k++) { fm.makeTranslation(x, podY + 0.16 + (k - (FINS - 1) / 2) * 0.064, POD.d / 2 + 0.03); fins.setMatrixAt(fi++, fm); }
    const plate = new THREE.Mesh(plateGeo, flat({ map: plateTex, transparent: true, depthWrite: false })); plate.position.set(0.02, -0.3, POD.d / 2 + 0.006); p.add(plate);
    const led = new THREE.Mesh(ledGeo, flat({ color: SHADE })); led.position.set(-0.6, -0.42, POD.d / 2 + 0.01); p.add(led); p.userData.led = led;
    const ring = new THREE.Mesh(new RoundedBoxGeometry(POD.w + 0.05, 0.05, POD.d + 0.05, 2, 0.02), flat({ color: ACCENT, transparent: true, opacity: 0, depthWrite: false })); ring.position.y = -POD.h / 2 + 0.2; ring.renderOrder = 5; p.add(ring); p.userData.ring = ring;
    return p;
  });
  fins.castShadow = false; lab.add(fins);
  const curvePts = []; for (let i = 0; i <= 60; i++) { const u = i / 60; curvePts.push(new THREE.Vector3(-2.5 + u * 5.0, 1.05 + Math.exp(-u * 3.2) * 1.5 + Math.sin(u * 18) * 0.08 * (1 - u), 0.4)); }
  const CURVE_SEG = 120, CURVE_RAD = 8;
  const curve = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(curvePts), CURVE_SEG, 0.05, CURVE_RAD, false), new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.5, toneMapped: false, envMapIntensity: 0.2, transparent: true, opacity: 0 }));
  curve.castShadow = true; lab.add(curve);

  // Friends' boxes around the desk, each in its own shell.
  const friends = new THREE.Group(); scene.add(friends);
  const friendBoxes = [0, 1, 2].map((i) => {
    const a = (i / 3) * Math.PI * 2 + 0.6, g = new THREE.Group(); g.position.set(Math.cos(a) * 9, SLAB.boxY, Math.sin(a) * 9); g.rotation.y = -a;
    for (let k = 0; k < 4; k++) { const m = new THREE.Mesh(slabGeo, matte()); m.position.y = slabY(k); m.castShadow = true; m.receiveShadow = true; g.add(m); }
    g.children[0].add(makeEye(-0.45), makeEye(0.45));
    const shell = new THREE.Mesh(new THREE.SphereGeometry(2.4, 56, 36), shellMaterial(THREE.FrontSide, 1)); shell.renderOrder = 10;
    shell.position.y = 0.2 - SLAB.boxY; g.add(shell); g.userData.shell = shell; friends.add(g); return g;
  });
  const linkGeo = new THREE.BufferGeometry(); const linkMat = new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0, toneMapped: false });
  const links = new THREE.LineSegments(linkGeo, linkMat); scene.add(links);

  // The campus map under everything, and the route drawn on it.
  const campus = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), flat({ map: campusTexture(), transparent: true, opacity: 0, depthWrite: false }));
  campus.rotation.x = -Math.PI / 2; campus.position.y = DESK_Y + 0.01; campus.renderOrder = -3; scene.add(campus);
  const routePts = [[-9, 8], [-9, 1], [-2, 1], [-2, -6], [6, -6], [6, -11], [11, -11]].map(([x, z]) => new THREE.Vector3(x, -0.9, z));
  const routeCurve = new THREE.CatmullRomCurve3(routePts, false, "catmullrom", 0.2);
  const route = new THREE.Mesh(new THREE.TubeGeometry(routeCurve, 200, 0.08, 8, false), new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.5, toneMapped: false, envMapIntensity: 0.2, transparent: true, opacity: 0 }));
  scene.add(route);

  // The network cable: a short lead out of the back-left of the bottom slab, drooping to the desk. Turns with the box.
  const cableRig = new THREE.Group(); scene.add(cableRig);
  const cableSlide = new THREE.Group(); cableRig.add(cableSlide);
  const y0 = SLAB.boxY + slabY(0);
  const cablePts = [[-1.0, y0, -0.5], [-1.52, y0 + 0.02, -0.62], [-2.05, y0 - 0.1, -0.86], [-2.65, DESK_Y + 0.046, -1.16], [-3.45, DESK_Y + 0.046, -1.5]].map((p) => new THREE.Vector3(...p));
  const cableMat = new THREE.MeshStandardMaterial({ color: INK, roughness: 0.5, envMapIntensity: 0.6, transparent: true });
  const cable = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(cablePts, false, "catmullrom", 0.5), 48, 0.045, 12, false), cableMat);
  cable.castShadow = true;
  const plug = new THREE.Mesh(new RoundedBoxGeometry(0.22, 0.11, 0.14, 2, 0.02), new THREE.MeshStandardMaterial({ color: INK, roughness: 0.45, envMapIntensity: 0.6, transparent: true }));
  plug.position.set(-1.24, y0, -0.5); plug.castShadow = true;
  cableSlide.add(cable, plug);

  const HOME = { x: 0, y: 2.2, z: 8.5, tx: 0, ty: 0.2, tz: 0, fov: 35 };
  const cam = { ...HOME };
  const state = {
    open: 0, spin: 0, layer: [0, 0, 0, 0], membrane: 0, ring: 0, beads: 0, bars: 0,
    lab: 0, labGlow: 0, labCurve: 0, friends: 0, friendsLink: 0, campus: 0, route: 0, cable: 0, pulse: 0, blur: 0,
    drop: 0, eyes: 0,
  };
  const P = {
    box: [0, 0.2, 0], lab: [14, 0.8, -2], friend: friendBoxes.map((g) => [g.position.x, 0.2, g.position.z]),
    ring: (a) => [Math.cos(a) * 4, -0.9, Math.sin(a) * 4],
  };
  /** World-space rest height of layer i (0 = cameras … 3 = agent), before `open` lifts it. */
  const layerY = (i) => SLAB.boxY + slabY(i);

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

  // Labels: soft cards at 3D points, always facing the camera.
  const labels = {}, labelSprites = {};
  function label(id, { text, at, size = 56, scale = 1 }) {
    if (labels[id]) return labels[id];
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTexture(text, size), transparent: true, opacity: 0, depthWrite: false, toneMapped: false }));
    sp.scale.set(4 * scale, 1 * scale, 1); sp.position.set(...at); sp.renderOrder = 20; scene.add(sp);
    labelSprites[id] = { sprite: sp, base: at.slice() }; labels[id] = { opacity: 0, lift: 0 };
    return labels[id];
  }

  const surfC = new THREE.Color(SURFACE_3D), accentC = new THREE.Color(ACCENT), shadeC = new THREE.Color(SHADE);
  const m4 = new THREE.Matrix4(), v3 = new THREE.Vector3(), q = new THREE.Quaternion(), s3 = new THREE.Vector3(), yAxis = new THREE.Vector3(0, 1, 0);

  function draw(t) {
    // Camera.
    camera.position.set(cam.x, cam.y, cam.z); camera.lookAt(cam.tx, cam.ty, cam.tz); camera.fov = cam.fov; camera.updateProjectionMatrix();
    // The box: layers spread with `open`, fall in with `drop` (bottom lands first), the whole box turns with `spin`,
    // each layer lights with `layer[i]`, the box breathes half a percent and beats with `pulse`.
    box.rotation.y = state.spin * Math.PI * 2;
    box.scale.setScalar(1 + 0.005 * Math.sin(t * 0.8));
    const d = state.drop, beat = heartbeat(t);
    for (let i = 0; i < 4; i++) {
      const lag = (3 - i) * 0.14, u = d <= 0 ? 0 : clamp01((d - lag) / (1 - lag));
      const fall = u * (3.2 + i * 0.9), squash = d < 0 ? d * i * 0.5 : 0;
      const L = layers[i];
      L.position.y = slabY(i) + state.open * i * SLAB.open + fall + squash;
      L.rotation.z = u * (hash(i, 21) - 0.5) * 0.26; L.rotation.x = u * (hash(i, 22) - 0.5) * 0.18;
      const lit = clamp01(state.layer[i]);
      L.material.color.copy(surfC).lerp(accentC, lit * 0.35); L.material.emissiveIntensity = lit * 0.22 + state.pulse * 0.28 * beat;
      rims[i].material.opacity = lit * 0.9; rims[i].visible = lit > 0.002;
    }
    for (let i = 0; i < 3; i++) { seams[i].position.y = layers[i].position.y + SLAB.pitch / 2; seams[i].material.opacity = 1 - clamp01(state.open * 4); seams[i].visible = state.open < 0.25 && d <= 0; }
    // Eyes: warm up with `eyes` (an accent glow in the lens and a halo), and blink now and then while awake.
    const ph = (t + 1.7) % 5.3, blink = ph < 0.22 ? Math.sin((ph / 0.22) * Math.PI) : 0;
    for (const e of eyes) {
      e.userData.lens.material.emissiveIntensity = state.eyes * (0.9 + 0.1 * Math.sin(t * 1.3));
      e.userData.halo.material.opacity = state.eyes * 0.75; e.userData.halo.visible = state.eyes > 0.002;
      e.scale.y = Math.max(0.06, 1 - blink * clamp01(state.eyes * 3));
    }
    contact.material.opacity = 1 - clamp01(d * 1.6);
    // The membrane drifts a little, like something alive.
    membrane.material.uniforms.uOpacity.value = state.membrane; membraneBack.material.uniforms.uOpacity.value = state.membrane;
    membrane.visible = state.membrane > 0.002;
    membrane.position.y = 0.2 + 0.03 * Math.sin(t * 0.45); membrane.scale.setScalar(1 + 0.008 * Math.sin(t * 0.7 + 1));
    // The ring and its beads: beads appear in order; `bars` stretches them upward into columns.
    ring.material.opacity = state.ring * 0.9; ring.visible = state.ring > 0.002;
    beads.visible = state.beads > 0.001;
    for (let i = 0; i < BEADS; i++) {
      const a = (i / BEADS) * Math.PI * 2 - Math.PI / 2;
      const on = clamp01(state.beads - i), h = 0.22 + state.bars * (0.3 + hash(i, 1) * 1.6);
      v3.set(Math.cos(a) * 4, DESK_Y + (h / 2) * on, Math.sin(a) * 4); q.setFromAxisAngle(yAxis, -a); s3.set(on, on * (h / 0.22), on);
      m4.compose(v3, q, s3); beads.setMatrixAt(i, m4);
    }
    beads.instanceMatrix.needsUpdate = true; beads.material.emissiveIntensity = state.bars * 0.3;
    // The lab: the pods rise and settle in; `labGlow` warms them (watts); `labCurve` draws the loss ribbon.
    lab.visible = state.lab > 0.001; lab.scale.setScalar(0.6 + 0.4 * state.lab); lab.position.y = (1 - state.lab) * -1.5;
    pods.forEach((p, i) => {
      const g = clamp01(state.labGlow) * (0.92 + 0.08 * Math.sin(t * 3 + i));
      p.material.color.copy(surfC).lerp(accentC, g * 0.32); p.material.emissiveIntensity = g * 0.24;
      p.userData.led.material.color.copy(shadeC).lerp(accentC, clamp01(state.labGlow * 3)); p.userData.led.material.opacity = 1;
      p.userData.ring.material.opacity = g * 0.85; p.userData.ring.visible = g > 0.002;
    });
    curve.material.opacity = state.labCurve > 0 ? 1 : 0; curve.visible = state.labCurve > 0.001;
    curve.geometry.setDrawRange(0, Math.floor(clamp01(state.labCurve) * CURVE_SEG) * CURVE_RAD * 6);
    // Friends and their links.
    friends.visible = state.friends > 0.001;
    friendBoxes.forEach((g) => { g.scale.setScalar(0.001 + state.friends); g.userData.shell.material.uniforms.uOpacity.value = state.friends * 0.85; });
    if (state.friendsLink > 0) {
      const pts = []; const c = new THREE.Vector3(0, 0.2, 0);
      friendBoxes.forEach((g) => { const f = new THREE.Vector3(g.position.x, 0.2, g.position.z); const e = c.clone().lerp(f, state.friendsLink); pts.push(c.x, c.y, c.z, e.x, e.y, e.z); });
      linkGeo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3)); linkMat.opacity = 0.9; links.visible = true;
    } else links.visible = false;
    // Campus and route.
    campus.material.opacity = state.campus * 0.9; campus.visible = state.campus > 0.002;
    route.material.opacity = state.route > 0 ? 1 : 0; route.geometry.setDrawRange(0, Math.floor(state.route * 200 * 8 * 6));
    // The cable turns with the box, slides out and swings away with `cable`, and is not there while the slabs are in the air.
    cableRig.rotation.y = box.rotation.y;
    cableSlide.position.x = -state.cable * 1.6; cableSlide.position.y = -state.cable * 0.08; cableSlide.rotation.y = state.cable * 0.35;
    cable.material.opacity = plug.material.opacity = 1 - clamp01(d * 2); cableRig.visible = d < 0.5;
    // Particle flows.
    for (const id in flowMeshes) {
      const { mesh, curve: cv, count, size } = flowMeshes[id], f = flows[id];
      mesh.material.opacity = f.density; mesh.visible = f.density > 0.001 && f.progress > 0;
      for (let i = 0; i < count; i++) {
        const u = clamp01(f.progress * (1 + f.spread) - hash(i, 5) * f.spread);
        cv.getPoint(u, v3); v3.x += (hash(i, 2) - 0.5) * 0.5; v3.y += (hash(i, 3) - 0.5) * 0.5; v3.z += (hash(i, 4) - 0.5) * 0.5;
        const sc = size * (u > 0 && u < 1 ? 1 : 0.001) * (0.7 + 0.6 * hash(i, 6));
        m4.compose(v3, q.identity(), s3.set(sc, sc, sc)); mesh.setMatrixAt(i, m4);
      }
      mesh.instanceMatrix.needsUpdate = true;
    }
    // Labels.
    for (const id in labelSprites) { const { sprite, base } = labelSprites[id], l = labels[id]; sprite.material.opacity = l.opacity; sprite.position.set(base[0], base[1] + l.lift, base[2]); sprite.visible = l.opacity > 0.001; }
    renderer.render(scene, camera);
  }

  // render(t) draws once per JavaScript task, after every tween of that tick has applied: the root timeline's clock
  // tween (the one that calls render) is the first child of the timeline, so it would otherwise draw before the slot
  // tweens of the same tick had updated cam/state. Still a pure function of (state, t); draw(t) is the synchronous form.
  let queued = false, lastT = 0;
  function render(t) { lastT = t; if (queued) return; queued = true; queueMicrotask(() => { queued = false; draw(lastT); }); }

  function leg(tl, at, to, dur = 1.6, ease = "power2.inOut") { tl.to(cam, { ...to, duration: dur, ease }, at); }

  const world = { THREE, scene, camera, renderer, cam, HOME, state, P, flows, flow, labels, label, render, draw, leg, box, layers, membrane, lab, pods, friends: friendBoxes, hash, SLAB, layerY };
  draw(0);
  return world;
}
