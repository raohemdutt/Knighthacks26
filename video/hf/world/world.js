// The Atlas world: one Three.js scene the whole video lives in. See WORLD.md for the contract.
// Everything that moves is a number on world.cam / world.state / world.flows[id] / world.labels[id];
// render(t) is a pure function of those numbers and t. No clocks, no Math.random.
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

const SURFACE = 0xe4e8ee, SHADE = 0xb7c0cd, LIGHT = 0xffffff, INK = 0x2b313b, ACCENT = 0xd9774a, MUTED = 0x6f7886;
const W = 1920, H = 1080;

/** Deterministic 0..1 hash of two integers (same every frame, every render). */
export function hash(i, j = 0) {
  let n = (i * 374761393 + j * 668265263) | 0;
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const lerp = (a, b, t) => a + (b - a) * t;

function softTexture(draw, w = 512, h = 512) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  return tex;
}

function shadowTexture() {
  return softTexture((ctx, w, h) => {
    const g = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    g.addColorStop(0, "rgba(183,192,205,0.55)"); g.addColorStop(0.6, "rgba(183,192,205,0.18)"); g.addColorStop(1, "rgba(183,192,205,0)");
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

function campusTexture() {
  return softTexture((ctx, w, h) => {
    ctx.fillStyle = "rgba(228,232,238,0)"; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(111,120,134,0.35)"; ctx.lineWidth = 10; ctx.lineCap = "round";
    const roads = [[[80, 520], [1000, 520]], [[540, 60], [540, 980]], [[200, 200], [860, 860]], [[160, 820], [900, 240]], [[300, 100], [300, 950]], [[800, 100], [800, 950]]];
    for (const [[x1, y1], [x2, y2]] of roads) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }
    ctx.fillStyle = "rgba(183,192,205,0.45)";
    for (let i = 0; i < 26; i++) { const x = 90 + hash(i, 7) * 880, y = 90 + hash(i, 9) * 880, s = 40 + hash(i, 11) * 90; ctx.beginPath(); ctx.roundRect(x, y, s, s * (0.6 + hash(i, 3) * 0.8), 14); ctx.fill(); }
  }, 1024, 1024);
}

export async function createWorld(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(W, H, false); renderer.setPixelRatio(1);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, W / H, 0.1, 200);

  // Light: one soft sky, one key from the top left (the same light the CSS shadows pretend).
  scene.add(new THREE.HemisphereLight(LIGHT, SHADE, 2.2));
  const key = new THREE.DirectionalLight(LIGHT, 1.6); key.position.set(-8, 12, 9); key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048); key.shadow.camera.left = -14; key.shadow.camera.right = 14; key.shadow.camera.top = 14; key.shadow.camera.bottom = -14; key.shadow.radius = 6; key.shadow.bias = -0.0005;
  scene.add(key);
  const fill = new THREE.DirectionalLight(LIGHT, 0.5); fill.position.set(6, 4, -6); scene.add(fill);

  // Desk: shadow receiver (invisible except for the shadow) + the soft contact shadow.
  const desk = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.ShadowMaterial({ opacity: 0.22, color: SHADE }));
  desk.rotation.x = -Math.PI / 2; desk.position.y = -1.0; desk.receiveShadow = true; scene.add(desk);
  const contact = new THREE.Mesh(new THREE.PlaneGeometry(6.4, 6.4), new THREE.MeshBasicMaterial({ map: shadowTexture(), transparent: true, depthWrite: false }));
  contact.rotation.x = -Math.PI / 2; contact.position.y = -0.995; scene.add(contact);

  // The box: four layers, cameras at the bottom, agent at the top.
  const box = new THREE.Group(); scene.add(box);
  const slabGeo = new RoundedBoxGeometry(2.4, 0.36, 1.6, 5, 0.09);
  const layers = [];
  for (let i = 0; i < 4; i++) {
    const m = new THREE.Mesh(slabGeo, new THREE.MeshStandardMaterial({ color: SURFACE, roughness: 0.82, metalness: 0.0, emissive: ACCENT, emissiveIntensity: 0 }));
    m.castShadow = true; m.receiveShadow = true; box.add(m); layers.push(m);
  }
  // Two camera eyes on the front of the bottom layer.
  const eyeGeo = new THREE.CircleGeometry(0.09, 32), eyeMat = new THREE.MeshBasicMaterial({ color: INK });
  const eyes = [-0.45, 0.45].map((x) => { const e = new THREE.Mesh(eyeGeo, eyeMat); e.position.set(x, 0, 0.81); layers[0].add(e); return e; });

  // The membrane: the privacy shell. Nothing leaves it uncounted.
  const membrane = new THREE.Mesh(new THREE.SphereGeometry(2.6, 64, 48), new THREE.MeshPhysicalMaterial({
    color: LIGHT, roughness: 0.15, metalness: 0, transparent: true, opacity: 0, transmission: 0, clearcoat: 1, clearcoatRoughness: 0.2, side: THREE.DoubleSide, depthWrite: false,
  }));
  membrane.position.y = 0.2; scene.add(membrane);

  // The time ring on the desk, and its beads.
  const ring = new THREE.Mesh(new THREE.TorusGeometry(4, 0.035, 12, 160), new THREE.MeshStandardMaterial({ color: MUTED, roughness: 0.6, transparent: true, opacity: 0 }));
  ring.rotation.x = Math.PI / 2; ring.position.y = -0.98; scene.add(ring);
  const BEADS = 48;
  const beads = new THREE.InstancedMesh(new RoundedBoxGeometry(0.22, 0.22, 0.22, 3, 0.06), new THREE.MeshStandardMaterial({ color: SURFACE, roughness: 0.8, emissive: ACCENT, emissiveIntensity: 0.0 }), BEADS);
  beads.castShadow = true; beads.instanceMatrix.setUsage(THREE.DynamicDrawUsage); scene.add(beads);

  // The lab, far to the right: two standing R9700 slabs and a loss ribbon.
  const lab = new THREE.Group(); lab.position.set(14, 0, -2); scene.add(lab);
  const cardGeo = new RoundedBoxGeometry(0.5, 3.2, 5.2, 5, 0.12);
  const cards = [-1.1, 1.1].map((x) => { const c = new THREE.Mesh(cardGeo, new THREE.MeshStandardMaterial({ color: SURFACE, roughness: 0.8, emissive: ACCENT, emissiveIntensity: 0 })); c.position.set(x, 0.6, 0); c.castShadow = true; lab.add(c); return c; });
  const curvePts = []; for (let i = 0; i <= 60; i++) { const u = i / 60; curvePts.push(new THREE.Vector3(-2.6 + u * 5.2, 2.9 + Math.exp(-u * 3.2) * 1.6 + Math.sin(u * 18) * 0.08 * (1 - u), 3.2)); }
  const curve = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(curvePts), 120, 0.05, 8, false), new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.5, transparent: true, opacity: 0 }));
  lab.add(curve);

  // Friends' boxes around the desk, each in its own shell.
  const friends = new THREE.Group(); scene.add(friends);
  const friendBoxes = [0, 1, 2].map((i) => {
    const a = (i / 3) * Math.PI * 2 + 0.6, g = new THREE.Group(); g.position.set(Math.cos(a) * 9, 0, Math.sin(a) * 9); g.rotation.y = -a;
    for (let k = 0; k < 4; k++) { const m = new THREE.Mesh(slabGeo, new THREE.MeshStandardMaterial({ color: SURFACE, roughness: 0.82 })); m.position.y = k * 0.4 - 0.6; m.castShadow = true; g.add(m); }
    const shell = new THREE.Mesh(new THREE.SphereGeometry(2.4, 48, 32), new THREE.MeshPhysicalMaterial({ color: LIGHT, roughness: 0.15, transparent: true, opacity: 0, clearcoat: 1, side: THREE.DoubleSide, depthWrite: false }));
    shell.position.y = 0.2; g.add(shell); g.userData.shell = shell; friends.add(g); return g;
  });
  const linkGeo = new THREE.BufferGeometry(); const linkMat = new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0 });
  const links = new THREE.LineSegments(linkGeo, linkMat); scene.add(links);

  // The campus map under everything, and the route drawn on it.
  const campus = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshBasicMaterial({ map: campusTexture(), transparent: true, opacity: 0, depthWrite: false }));
  campus.rotation.x = -Math.PI / 2; campus.position.y = -0.99; scene.add(campus);
  const routePts = [[-9, 8], [-9, 1], [-2, 1], [-2, -6], [6, -6], [6, -11], [11, -11]].map(([x, z]) => new THREE.Vector3(x, -0.9, z));
  const routeCurve = new THREE.CatmullRomCurve3(routePts, false, "catmullrom", 0.2);
  const route = new THREE.Mesh(new THREE.TubeGeometry(routeCurve, 200, 0.08, 8, false), new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.5, transparent: true, opacity: 0 }));
  scene.add(route);

  // The network cable out of the back-left of the box.
  const cable = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 6, 12), new THREE.MeshStandardMaterial({ color: INK, roughness: 0.6 }));
  cable.rotation.z = Math.PI / 2; cable.position.set(-4.2, -0.6, -0.4); cable.castShadow = true; scene.add(cable);

  const HOME = { x: 0, y: 2.2, z: 8.5, tx: 0, ty: 0.2, tz: 0, fov: 35 };
  const cam = { ...HOME };
  const state = {
    open: 0, spin: 0, layer: [0, 0, 0, 0], membrane: 0, ring: 0, beads: 0, bars: 0,
    lab: 0, labGlow: 0, labCurve: 0, friends: 0, friendsLink: 0, campus: 0, route: 0, cable: 0, pulse: 0, blur: 0,
  };
  const P = {
    box: [0, 0.2, 0], lab: [14, 0.8, -2], friend: friendBoxes.map((g) => [g.position.x, 0.2, g.position.z]),
    ring: (a) => [Math.cos(a) * 4, -0.9, Math.sin(a) * 4],
  };

  // Particle flows: declared once, driven by flows[id].progress / density.
  const flows = {}, flowMeshes = {};
  const dot = new THREE.SphereGeometry(0.07, 10, 8);
  function flow(id, { from, to, count = 24, color = ACCENT, size = 1, lift = 2.2 }) {
    if (flows[id]) return flows[id];
    const mesh = new THREE.InstancedMesh(dot, new THREE.MeshStandardMaterial({ color, roughness: 0.4, emissive: color, emissiveIntensity: 0.35, transparent: true, opacity: 0 }), count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); mesh.castShadow = false; scene.add(mesh);
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
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTexture(text, size), transparent: true, opacity: 0, depthWrite: false }));
    sp.scale.set(4 * scale, 1 * scale, 1); sp.position.set(...at); scene.add(sp);
    labelSprites[id] = { sprite: sp, base: at.slice() }; labels[id] = { opacity: 0, lift: 0 };
    return labels[id];
  }

  const m4 = new THREE.Matrix4(), v3 = new THREE.Vector3(), q = new THREE.Quaternion(), s3 = new THREE.Vector3();

  function render(t) {
    // Camera.
    camera.position.set(cam.x, cam.y, cam.z); camera.lookAt(cam.tx, cam.ty, cam.tz); camera.fov = cam.fov; camera.updateProjectionMatrix();
    // The box: layers spread with `open`, the whole box turns with `spin`, each layer lights with `layer[i]`.
    box.rotation.y = state.spin * Math.PI * 2;
    for (let i = 0; i < 4; i++) {
      layers[i].position.y = i * 0.4 - 0.6 + state.open * i * 0.55;
      layers[i].material.emissiveIntensity = state.layer[i] * 0.55 + state.pulse * 0.25 * (0.5 + 0.5 * Math.sin(t * 2.4 + i));
    }
    membrane.material.opacity = state.membrane * 0.28;
    membrane.scale.setScalar(1 + state.membrane * 0.02 * Math.sin(t * 1.3));
    // The ring and its beads: beads appear in order; `bars` stretches them upward into columns.
    ring.material.opacity = state.ring * 0.9;
    for (let i = 0; i < BEADS; i++) {
      const a = (i / BEADS) * Math.PI * 2 - Math.PI / 2;
      const on = clamp01(state.beads - i), h = 0.22 + state.bars * (0.3 + hash(i, 1) * 1.6);
      v3.set(Math.cos(a) * 4, -0.98 + (h / 2) * on, Math.sin(a) * 4); q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -a); s3.set(on, on * (h / 0.22), on);
      m4.compose(v3, q, s3); beads.setMatrixAt(i, m4);
    }
    beads.instanceMatrix.needsUpdate = true; beads.material.emissiveIntensity = state.bars * 0.3;
    // The lab.
    lab.visible = state.lab > 0.001; lab.scale.setScalar(0.6 + 0.4 * state.lab); lab.position.y = (1 - state.lab) * -1.5;
    cards.forEach((c, i) => { c.material.emissiveIntensity = state.labGlow * (0.35 + 0.1 * Math.sin(t * 3 + i)); c.material.opacity = 1; });
    curve.material.opacity = state.labCurve; curve.scale.set(Math.max(0.001, state.labCurve), 1, 1); curve.position.x = -2.6 * (1 - Math.max(0.001, state.labCurve));
    // Friends and their links.
    friends.visible = state.friends > 0.001;
    friendBoxes.forEach((g, i) => { g.scale.setScalar(0.001 + state.friends); g.userData.shell.material.opacity = state.friends * 0.22; g.children.forEach((m, k) => { if (k < 4) m.position.y = k * 0.4 - 0.6; }); });
    if (state.friendsLink > 0) {
      const pts = []; const c = new THREE.Vector3(0, 0.2, 0);
      friendBoxes.forEach((g, i) => { const f = new THREE.Vector3(g.position.x, 0.2, g.position.z); const e = c.clone().lerp(f, state.friendsLink); pts.push(c.x, c.y, c.z, e.x, e.y, e.z); });
      linkGeo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3)); linkMat.opacity = 0.9; links.visible = true;
    } else links.visible = false;
    // Campus and route.
    campus.material.opacity = state.campus * 0.9;
    route.material.opacity = state.route > 0 ? 1 : 0; route.geometry.setDrawRange(0, Math.floor(state.route * 200 * 8 * 6));
    // The cable slides out and droops.
    cable.position.x = -4.2 - state.cable * 2.2; cable.position.y = -0.6 - state.cable * 0.3; cable.rotation.y = state.cable * 0.25;
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

  function leg(tl, at, to, dur = 1.6, ease = "power2.inOut") { tl.to(cam, { ...to, duration: dur, ease }, at); }

  const world = { THREE, scene, camera, renderer, cam, HOME, state, P, flows, flow, labels, label, render, leg, box, layers, membrane, lab, friends: friendBoxes, hash };
  render(0);
  return world;
}
