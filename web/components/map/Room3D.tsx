"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { cameraName, humanize } from "@/lib/format";
import { createFsrPipeline, type FsrInfo, type FsrPipeline } from "@/lib/fsr";
import type { Point, RoomCamera, RoomPin } from "@/lib/room";
import ui from "../ui.module.css";
import dr from "../dayrun/dayrun.module.css";
import css from "./room3d.module.css";

/**
 * The room in 3D: every camera's view is a floor panel and its zones are
 * extruded from their polygons. Drag to rotate and tilt, pinch or scroll to
 * zoom. This file is the only one that imports three.js, and the Room map
 * loads it on demand, so no other screen pays for it.
 *
 * The heights are for reading the picture only; nothing on the box measures them.
 *
 * Rendering: the scene can be drawn below the screen's resolution and upscaled with the
 * FSR 1 (EASU + RCAS) port in lib/fsr.ts. The control under the stage picks the mode, and
 * "Compare" splits the picture between a plain bilinear upscale and FSR.
 *
 * Presenter and test links (they win over the remembered choice):
 *   /map/?view=3d&fsr=off | quality | performance     the render mode
 *   /map/?view=3d&fsr=quality&compare=1               the split (&split=0.3 moves the divider; 1 is all bilinear)
 *   /map/?view=3d&gl=1  (or gl=badshader)             force the fallback to normal rendering
 *   /map/?view=3d&fsrbench=1                          time 120 frames in each mode
 */

const PANEL_W = 4 / 3; // a camera frame is drawn 4:3, like the Sort It grid (32 x 24)
const PANEL_D = 1;
const GAP = 0.22;
const ZONE_H = 0.14;
const NAME_STEP = 0.15; // how much higher a zone's name stands when it would run into its neighbour's
const PIN_STEP = 0.09; // how much taller a pin stands for each row of pins in front of it
const HALO = 0.3; // how far a panel's soft shadow reaches past its edge

const COLOR = { floor: 0xdbe6f8, edge: 0xc2d0e8, zone: 0x8e9bb0, zoneOn: 0x3b82f6, text: "#1b2433", muted: "#566176", pin: 0x566176, accent: 0x2563eb };

export type Mode = "native" | "quality" | "performance";
export const MODES: { id: Mode; name: string; sub: string; scale: number }[] = [
  { id: "native", name: "Native", sub: "100%", scale: 1 },
  { id: "quality", name: "FSR Quality", sub: "67%", scale: 0.67 },
  { id: "performance", name: "FSR Performance", sub: "50%", scale: 0.5 },
];
export const scaleOf = (mode: Mode) => MODES.find((m) => m.id === mode)?.scale ?? 1;
export const STORE_KEY = "atlas.map.render";
export const BENCH_FRAMES = 120;
export const BENCH: { name: string; scale: number; direct?: boolean }[] = [
  { name: "Straight to the canvas (no pipeline)", scale: 1, direct: true },
  { name: "Native", scale: 1 },
  { name: "FSR Ultra Quality (77%)", scale: 0.77 },
  { name: "FSR Quality (67%)", scale: 0.67 },
  { name: "FSR Performance (50%)", scale: 0.5 },
];

export interface BenchRow {
  name: string;
  ms: number;
  internal: [number, number];
  output: [number, number];
}

export const dims = (s: [number, number]) => `${s[0]}×${s[1]}`;

export function remembered(): { mode: Mode; compare: boolean } {
  let mode: Mode = "quality";
  let compare = false;
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORE_KEY) ?? "null") as { mode?: string; compare?: boolean } | null;
    if (saved && MODES.some((m) => m.id === saved.mode)) mode = saved.mode as Mode;
    if (saved?.compare === true) compare = true;
  } catch {
    // Private mode or a bad value: start from the default.
  }
  const q = new URLSearchParams(window.location.search);
  const asked = q.get("fsr");
  if (asked === "off" || asked === "native") mode = "native";
  else if (asked === "quality" || asked === "performance") mode = asked;
  if (q.get("compare") === "1") compare = true;
  if (q.get("compare") === "0") compare = false;
  return { mode, compare };
}

function label(text: string, color: string, size = 0.11, outline = true): THREE.Sprite {
  const pad = 12;
  const font = "600 44px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
  const canvas = document.createElement("canvas");
  const probe = canvas.getContext("2d");
  if (probe) probe.font = font;
  const w = Math.ceil((probe?.measureText(text).width ?? text.length * 24) + pad * 2);
  canvas.width = w;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.font = font;
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineWidth = 8;
    ctx.strokeStyle = "#eaf1fc";
    if (outline) ctx.strokeText(text, pad, 34);
    ctx.fillStyle = color;
    ctx.fillText(text, pad, 34);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
  sprite.scale.set((size * w) / 64, size, 1);
  sprite.renderOrder = 10;
  return sprite;
}

/**
 * A soft shadow painted on a canvas and laid flat under something. It stands in for ambient
 * occlusion and contact shadows: no shadow pass, no extra work per frame.
 */
function shadowPlane(width: number, depth: number, pixels: [number, number], order: number, paint: (ctx: CanvasRenderingContext2D) => void): THREE.Mesh {
  const canvas = document.createElement("canvas");
  canvas.width = pixels[0];
  canvas.height = pixels[1];
  const ctx = canvas.getContext("2d");
  if (ctx) paint(ctx);
  const geometry = new THREE.PlaneGeometry(width, depth);
  geometry.rotateX(-Math.PI / 2); // flat, with the top of the canvas at the far edge
  const plane = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false }));
  plane.renderOrder = order;
  return plane;
}

/** A round soft shadow, for under a pin or a ball. */
function blob(radius: number, strength: number): THREE.Mesh {
  return shadowPlane(radius * 2, radius * 2, [64, 64], 3, (ctx) => {
    const fade = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    fade.addColorStop(0, `rgba(0,0,0,${strength})`);
    fade.addColorStop(0.35, `rgba(0,0,0,${strength * 0.55})`);
    fade.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = fade;
    ctx.fillRect(0, 0, 64, 64);
  });
}

function dispose(root: THREE.Object3D) {
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const mats = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
    for (const m of mats) {
      (m as THREE.SpriteMaterial).map?.dispose?.();
      m.dispose();
    }
  });
}

export interface Room3DProps {
  cameras: RoomCamera[];
  pins: RoomPin[];
  selected: string | null; // a pin's label
  route: { camera: string; points: Point[]; bin: string } | null;
}

export default function Room3D({ cameras, pins, selected, route }: Room3DProps) {
  const host = useRef<HTMLDivElement>(null);
  const stage = useRef<{ renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera; pipeline: FsrPipeline; draw: () => void; home: (ease?: boolean) => void; report: () => void } | null>(null);
  const content = useRef<THREE.Group | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("quality");
  const [compare, setCompare] = useState(false);
  const [info, setInfo] = useState<FsrInfo | null>(null);
  const [bench, setBench] = useState<BenchRow[] | "running" | null>(null);
  const choice = useRef<{ mode: Mode; compare: boolean }>({ mode: "quality", compare: false });

  const total = cameras.length * PANEL_W + Math.max(0, cameras.length - 1) * GAP;
  const originX = (i: number) => -total / 2 + i * (PANEL_W + GAP);

  // The remembered choice, or the one in the link. Read before the renderer is made.
  useEffect(() => {
    choice.current = remembered();
    setMode(choice.current.mode);
    setCompare(choice.current.compare);
  }, []);

  const choose = (next: { mode?: Mode; compare?: boolean }) => {
    choice.current = { ...choice.current, ...next };
    setMode(choice.current.mode);
    setCompare(choice.current.compare);
    try {
      window.localStorage.setItem(STORE_KEY, JSON.stringify(choice.current));
    } catch {
      // Nowhere to remember it; the choice still holds for this visit.
    }
    const s = stage.current;
    if (!s) return;
    s.pipeline.setScale(scaleOf(choice.current.mode));
    s.pipeline.setCompare(choice.current.compare);
    s.report();
    s.draw();
  };

  // The renderer, camera and controls: made once.
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    } catch (err) {
      setFailed(err instanceof Error ? err.message : String(err));
      return;
    }
    // Opaque, in the stage's own colour: the upscaler works on colour only, not on alpha.
    const backdrop = new THREE.Color(COLOR.floor);
    try {
      backdrop.setStyle(getComputedStyle(el).backgroundColor);
    } catch {
      // Keep the default.
    }
    renderer.setClearColor(backdrop, 1);
    el.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("aria-label", "The room in 3D. Drag to rotate and tilt.");
    renderer.domElement.setAttribute("role", "img");

    const query = new URLSearchParams(window.location.search);
    const gl = query.get("gl");
    const pipeline = createFsrPipeline(renderer, { scale: scaleOf(choice.current.mode), simulate: gl === "1" ? "webgl1" : gl === "badshader" ? "badshader" : null });
    const at = Number(query.get("split") ?? "0.5");
    pipeline.setCompare(choice.current.compare, Number.isFinite(at) ? at : 0.5);

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.9);
    sun.position.set(-1.5, 3, 2);
    scene.add(sun);

    const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 50);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.enableDamping = true; // a drag coasts to a stop instead of halting dead
    controls.dampingFactor = 0.09;
    controls.maxPolarAngle = Math.PI / 2 - 0.08; // never under the floor
    controls.minDistance = 1.2;
    controls.maxDistance = 9;

    // Frames are drawn only while something moves: a drag, its coasting, or the eased reset.
    const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    let frame = 0;
    let dirty = true;
    let still = 0;
    let glide: { from: THREE.Vector3; to: THREE.Vector3; target: THREE.Vector3; aim: THREE.Vector3; start: number } | null = null;
    const paint = () => {
      dirty = false;
      pipeline.render(scene, camera);
    };
    const tick = (now: number) => {
      frame = 0;
      if (glide) {
        const t = Math.min(1, (now - glide.start) / 650);
        const k = 1 - Math.pow(1 - t, 3); // ease out
        camera.position.lerpVectors(glide.from, glide.to, k);
        controls.target.lerpVectors(glide.target, glide.aim, k);
        if (t >= 1) glide = null;
        dirty = true;
      }
      const moved = controls.update();
      if (moved || dirty) {
        paint();
        still = 0;
      } else still++;
      // `controls.update()` may already have asked for the next frame through its change event.
      if (!frame && (glide || still < 12)) frame = requestAnimationFrame(tick);
    };
    const draw = () => {
      dirty = true;
      still = 0;
      if (!frame) frame = requestAnimationFrame(tick);
    };
    const report = () => setInfo(pipeline.info());
    const home = (ease = false) => {
      // Just far enough back that every panel fits across the stage, whatever its shape.
      const half = total / 2 + 0.12;
      const tan = Math.tan((camera.fov * Math.PI) / 360);
      const distance = Math.max(half / (tan * camera.aspect), (PANEL_D * 0.9) / tan) * 1.1; // a margin for the nearer edge and the labels
      const from = new THREE.Vector3(-0.1, 0.74, 0.66).normalize().multiplyScalar(distance);
      const aim = new THREE.Vector3(0, 0.05, PANEL_D / 2);
      const to = new THREE.Vector3(from.x, from.y, PANEL_D / 2 + from.z);
      if (ease && !calm) {
        glide = { from: camera.position.clone(), to, target: controls.target.clone(), aim, start: performance.now() };
      } else {
        glide = null;
        controls.target.copy(aim);
        camera.position.copy(to);
        controls.update();
      }
      draw();
    };
    const size = () => {
      const w = el.clientWidth || 1;
      const h = el.clientHeight || 1;
      pipeline.setSize(w, h, window.devicePixelRatio || 1);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      report();
      paint(); // at once: resizing clears the canvas
      draw();
    };
    const grabbed = () => {
      glide = null; // a touch takes over from the eased reset
    };
    controls.addEventListener("change", draw);
    controls.addEventListener("start", grabbed);
    const watcher = new ResizeObserver(size);
    watcher.observe(el);
    stage.current = { renderer, scene, camera, pipeline, draw, home, report };
    size();
    home();

    // Hidden: /map/?view=3d&fsrbench=1 times BENCH_FRAMES frames in each mode. Every frame is
    // read back, so the time is what the GPU took, not how long it took to queue the work.
    let benchTimer = 0;
    let gone = false;
    if (query.get("fsrbench") === "1") {
      const context = renderer.getContext();
      const pixel = new Uint8Array(4);
      const frames = (n: number) => {
        for (let i = 0; i < n; i++) {
          pipeline.render(scene, camera);
          context.readPixels(0, 0, 1, 1, context.RGBA, context.UNSIGNED_BYTE, pixel);
        }
      };
      const pause = () => new Promise<void>((resolve) => window.setTimeout(resolve, 30));
      const run = async () => {
        setBench("running");
        await pause();
        pipeline.setCompare(false);
        const rows: BenchRow[] = [];
        for (const m of BENCH) {
          if (gone) return;
          pipeline.setBypass(m.direct === true);
          pipeline.setScale(m.scale);
          frames(10); // compile and warm up
          const t0 = performance.now();
          frames(BENCH_FRAMES);
          const now = pipeline.info();
          rows.push({ name: m.name, ms: (performance.now() - t0) / BENCH_FRAMES, internal: now.internal, output: now.output });
          await pause();
        }
        if (gone) return;
        pipeline.setBypass(false);
        pipeline.setScale(scaleOf(choice.current.mode));
        pipeline.setCompare(choice.current.compare);
        report();
        draw();
        setBench(rows);
        console.info("fsrbench", JSON.stringify(rows));
      };
      benchTimer = window.setTimeout(run, 1500); // after the zones and pins are in
    }

    return () => {
      gone = true;
      window.clearTimeout(benchTimer);
      if (frame) cancelAnimationFrame(frame);
      watcher.disconnect();
      controls.dispose();
      dispose(scene);
      pipeline.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      stage.current = null;
      content.current = null;
    };
  }, [total]);

  // What is in the room: rebuilt whenever the zones, pins, selection or route change.
  useEffect(() => {
    const s = stage.current;
    if (!s) return;
    if (content.current) {
      s.scene.remove(content.current);
      dispose(content.current);
    }
    const group = new THREE.Group();
    content.current = group;
    const selectedPin = pins.find((p) => p.label === selected);

    cameras.forEach((cam, i) => {
      const ox = originX(i);
      const place = (p: Point, y: number) => new THREE.Vector3(ox + p[0] * PANEL_W, y, p[1] * PANEL_D);

      const floor = new THREE.Mesh(new THREE.BoxGeometry(PANEL_W, 0.02, PANEL_D), new THREE.MeshStandardMaterial({ color: COLOR.floor, roughness: 0.95 }));
      floor.position.set(ox + PANEL_W / 2, -0.01, PANEL_D / 2);
      group.add(floor);
      const rim = new THREE.LineSegments(new THREE.EdgesGeometry(floor.geometry), new THREE.LineBasicMaterial({ color: COLOR.edge }));
      rim.position.copy(floor.position);
      group.add(rim);

      // Contact shading, painted once: the panel darkens the ground around it, and every zone
      // block darkens the floor at its foot (and a little under itself, seen through the block).
      const ground = shadowPlane(PANEL_W + HALO * 2, PANEL_D + HALO * 2, [256, 208], 0, (ctx) => {
        const sx = 256 / (PANEL_W + HALO * 2);
        const sy = 208 / (PANEL_D + HALO * 2);
        ctx.shadowColor = "rgba(70,85,122,0.5)";
        ctx.shadowBlur = HALO * sx * 0.75;
        ctx.fillStyle = "#46557a";
        ctx.fillRect(HALO * sx, HALO * sy, PANEL_W * sx, PANEL_D * sy);
      });
      ground.position.set(ox + PANEL_W / 2, -0.021, PANEL_D / 2);
      group.add(ground);
      const feet = shadowPlane(PANEL_W, PANEL_D, [512, 384], 1, (ctx) => {
        const trace = (polygon: Point[]) => {
          ctx.beginPath();
          polygon.forEach(([x, y], n) => (n === 0 ? ctx.moveTo(x * 512, y * 384) : ctx.lineTo(x * 512, y * 384)));
          ctx.closePath();
        };
        ctx.fillStyle = "#46557a";
        ctx.shadowColor = "rgba(70,85,122,0.55)";
        ctx.shadowBlur = 20;
        for (const zone of cam.zones) {
          trace(zone.polygon);
          ctx.fill();
        }
        ctx.shadowColor = "transparent";
        ctx.shadowBlur = 0;
        ctx.globalCompositeOperation = "destination-out";
        ctx.fillStyle = "rgba(0,0,0,0.7)"; // lighten the inside: the shadow belongs at the foot
        for (const zone of cam.zones) {
          trace(zone.polygon);
          ctx.fill();
        }
      });
      feet.position.set(ox + PANEL_W / 2, 0.002, PANEL_D / 2);
      group.add(feet);

      const name = label(cameraName(cam.id), COLOR.muted, 0.12);
      name.position.set(ox + PANEL_W / 2, 0.08, -0.06); // along the far edge, behind the zones
      group.add(name);

      const names: { level: number; span: [number, number] }[] = []; // zone names placed so far in this panel
      for (const zone of cam.zones) {
        const on = (selectedPin?.camera === cam.id && selectedPin.zone === zone.id) || (route?.camera === cam.id && route.bin === zone.id);
        const shape = new THREE.Shape(zone.polygon.map(([x, y]) => new THREE.Vector2(x * PANEL_W, y * PANEL_D)));
        const geometry = new THREE.ExtrudeGeometry(shape, { depth: ZONE_H, bevelEnabled: false });
        geometry.rotateX(Math.PI / 2); // the outline lies on the floor, the extrusion points down...
        geometry.translate(ox, ZONE_H, 0); // ...so lift it by its own height
        const block = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: on ? COLOR.zoneOn : COLOR.zone, roughness: 0.8, transparent: true, opacity: on ? 0.6 : 0.5 }));
        block.renderOrder = 2;
        group.add(block);
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color: on ? COLOR.accent : 0x8e9bb0 }));
        group.add(edges);
        // The name stands at the near edge of the zone, clear of any pins in its middle. It is drawn no
        // wider than it must be, and when it would still run into a neighbour's name it stands a step
        // higher, on a thin line down to its own zone.
        const xs = zone.polygon.map((q) => q[0]);
        const near = Math.max(...zone.polygon.map((q) => q[1]));
        const mid = (Math.min(...xs) + Math.max(...xs)) / 2;
        const text = label(humanize(zone.id), COLOR.text, 0.15);
        const fit = Math.max(0.72, Math.min(1, ((Math.max(...xs) - Math.min(...xs)) * PANEL_W) / text.scale.x));
        text.scale.multiplyScalar(fit);
        const span: [number, number] = [mid * PANEL_W - text.scale.x / 2, mid * PANEL_W + text.scale.x / 2];
        let level = 0;
        while (names.some((n) => n.level === level && n.span[0] < span[1] && span[0] < n.span[1])) level++;
        names.push({ level, span });
        text.position.copy(place([mid, near], ZONE_H + 0.1 + level * NAME_STEP));
        group.add(text);
        if (level > 0) {
          const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([place([mid, near], ZONE_H), place([mid, near], ZONE_H + 0.04 + level * NAME_STEP)]),
            new THREE.LineBasicMaterial({ color: 0x8e9bb0 }),
          );
          group.add(line);
        }
      }

      // Each pin is a ball with its number on it (the list under the map says which item that is). Pins
      // in a row behind another stand taller, so no ball hides the one in front of it. Only the pin in
      // focus has its name written out, above all of them.
      const mine = pins.filter((p) => p.camera === cam.id);
      const tallest = Math.max(0, ...mine.map((p) => p.lift));
      for (const pin of mine) {
        const on = pin.label === selected;
        const color = on ? COLOR.accent : COLOR.pin;
        const top = 0.2 + pin.lift * PIN_STEP;
        const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, top, 8), new THREE.MeshStandardMaterial({ color }));
        stem.position.copy(place(pin.at, ZONE_H + top / 2));
        group.add(stem);
        const shade = blob(0.085, 0.8); // where the pin stands on its zone
        shade.position.copy(place(pin.at, ZONE_H + 0.003));
        group.add(shade);
        const head = new THREE.Mesh(new THREE.SphereGeometry(on ? 0.058 : 0.05, 20, 14), new THREE.MeshStandardMaterial({ color, roughness: 0.4 }));
        head.position.copy(place(pin.at, ZONE_H + top));
        group.add(head);
        const number = label(String(pin.n), "#ffffff", 0.1, false);
        number.position.copy(head.position);
        group.add(number);
        if (on) {
          const text = label(humanize(pin.label), "#456cb3", 0.13);
          text.position.copy(place(pin.at, ZONE_H + 0.2 + tallest * PIN_STEP + 0.19));
          text.renderOrder = 11;
          group.add(text);
        }
      }

      if (route && route.camera === cam.id && route.points.length > 1) {
        const path = new THREE.CurvePath<THREE.Vector3>();
        for (let n = 1; n < route.points.length; n++) path.add(new THREE.LineCurve3(place(route.points[n - 1] as Point, ZONE_H + 0.03), place(route.points[n] as Point, ZONE_H + 0.03)));
        group.add(new THREE.Mesh(new THREE.TubeGeometry(path, route.points.length * 4, 0.014, 8, false), new THREE.MeshStandardMaterial({ color: COLOR.accent, emissive: COLOR.accent, emissiveIntensity: 0.4 })));
        const you = new THREE.Mesh(new THREE.SphereGeometry(0.04, 20, 14), new THREE.MeshStandardMaterial({ color: COLOR.pin }));
        you.position.copy(place(route.points[0] as Point, ZONE_H + 0.03));
        group.add(you);
        const shade = blob(0.08, 0.75);
        shade.position.copy(place(route.points[0] as Point, 0.004));
        group.add(shade);
        const text = label("You", COLOR.text, 0.13);
        text.position.copy(place(route.points[0] as Point, ZONE_H + 0.17));
        group.add(text);
      }
    });

    s.scene.add(group);
    s.draw();
    // `originX` only depends on the camera list, which is in the dependencies.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameras, pins, selected, route, failed]);

  if (failed) {
    return (
      <div className={`${ui.state} ${ui.stateError}`} role="alert">
        <p className={ui.stateTitle}>The 3D view can&apos;t start on this phone</p>
        <p className={ui.breakWords}>{failed}. The 2D view shows the same zones and pins.</p>
      </div>
    );
  }

  const fsrOn = info?.active ?? false;
  const splitOn = info?.compare ?? false;
  const unavailable = info !== null && !info.supported;
  const shown: Mode = unavailable ? "native" : mode; // nothing but normal rendering can be on

  return (
    <>
      <div ref={host} className={dr.stage}>
        {splitOn ? (
          <>
            {info && info.split > 0.2 ? (
              <span className={css.side} style={{ right: `calc(${(1 - info.split) * 100}% + 8px)` }}>
                Bilinear
              </span>
            ) : null}
            {info && info.split < 0.8 ? (
              <span className={css.side} style={{ left: `calc(${info.split * 100}% + 8px)` }}>
                FSR 1
              </span>
            ) : null}
          </>
        ) : null}
      </div>
      <div className={css.bar}>
        <span className={css.barName} id="render-mode">
          Render
        </span>
        <div className={`${ui.segmented} ${css.modes}`} role="group" aria-labelledby="render-mode">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              className={`${ui.seg} ${css.mode} ${shown === m.id ? ui.segOn : ""}`}
              aria-pressed={shown === m.id}
              aria-label={m.id === "native" ? "Native" : `${m.name} (${m.sub})`}
              disabled={unavailable && m.id !== "native"}
              onClick={() => choose({ mode: m.id })}
            >
              <span>{m.name}</span>
              <span className={css.modeSub}>{m.sub}</span>
            </button>
          ))}
        </div>
        <button type="button" className={`${dr.pill} ${compare && fsrOn ? dr.pillOn : ""}`} aria-pressed={compare && fsrOn} disabled={!fsrOn} onClick={() => choose({ compare: !compare })}>
          Compare
        </button>
      </div>
      <p className={css.caption} data-fsr-caption>
        {info === null
          ? "Starting the 3D view."
          : unavailable
            ? `FSR is not available here, so the room is drawn normally at ${dims(info.output)}: ${info.reason}.`
            : fsrOn
              ? `Upscaled with AMD FidelityFX Super Resolution 1 (EASU + RCAS): ${dims(info.internal)} → ${dims(info.output)}${splitOn ? ". Left half: the same low-resolution picture, stretched the plain (bilinear) way." : ""}`
              : `Native rendering at ${dims(info.output)}. FSR is off.`}
      </p>
      {bench === "running" ? (
        <p className={css.caption} data-fsr-bench="running">
          Timing {BENCH_FRAMES} frames in each mode.
        </p>
      ) : bench ? (
        <ul className={css.bench} data-fsr-bench="done" aria-label={`Average time for one frame, over ${BENCH_FRAMES} frames`}>
          {bench.map((row) => (
            <li key={row.name}>
              {row.name}: {row.ms.toFixed(2)} ms a frame ({dims(row.internal)} → {dims(row.output)})
            </li>
          ))}
        </ul>
      ) : null}
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <p className={`${ui.muted} ${ui.small}`}>Drag to rotate and tilt. Pinch or scroll to zoom.</p>
        <button type="button" className={dr.pill} onClick={() => stage.current?.home(true)}>
          Reset the view
        </button>
      </div>
    </>
  );
}
