"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { cameraName, humanize } from "@/lib/format";
import type { Point, RoomCamera, RoomPin } from "@/lib/room";
import ui from "../ui.module.css";
import dr from "../dayrun/dayrun.module.css";

/**
 * The room in 3D: every camera's view is a floor panel and its zones are
 * extruded from their polygons. Drag to rotate and tilt, pinch or scroll to
 * zoom. This file is the only one that imports three.js, and the Room map
 * loads it on demand, so no other screen pays for it.
 *
 * The heights are for reading the picture only; nothing on the box measures them.
 */

const PANEL_W = 4 / 3; // a camera frame is drawn 4:3, like the Sort It grid (32 x 24)
const PANEL_D = 1;
const GAP = 0.22;
const ZONE_H = 0.14;
const NAME_STEP = 0.15; // how much higher a zone's name stands when it would run into its neighbour's
const PIN_STEP = 0.09; // how much taller a pin stands for each row of pins in front of it

const COLOR = { floor: 0x1c2229, edge: 0x2b333d, zone: 0x5f6b78, zoneOn: 0x5ce1c6, text: "#f3f5f7", muted: "#a6b0bb", pin: 0xf3f5f7, accent: 0x5ce1c6 };

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
    ctx.strokeStyle = "#13171c";
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
  const stage = useRef<{ renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera; controls: OrbitControls; draw: () => void; home: () => void } | null>(null);
  const content = useRef<THREE.Group | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  const total = cameras.length * PANEL_W + Math.max(0, cameras.length - 1) * GAP;
  const originX = (i: number) => -total / 2 + i * (PANEL_W + GAP);

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
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    el.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("aria-label", "The room in 3D. Drag to rotate and tilt.");
    renderer.domElement.setAttribute("role", "img");

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.9);
    sun.position.set(-1.5, 3, 2);
    scene.add(sun);

    const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 50);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.maxPolarAngle = Math.PI / 2 - 0.08; // never under the floor
    controls.minDistance = 1.2;
    controls.maxDistance = 9;

    const draw = () => renderer.render(scene, camera);
    const home = () => {
      // Just far enough back that every panel fits across the stage, whatever its shape.
      const half = total / 2 + 0.12;
      const tan = Math.tan((camera.fov * Math.PI) / 360);
      const distance = Math.max(half / (tan * camera.aspect), (PANEL_D * 0.9) / tan) * 1.1; // a margin for the nearer edge and the labels
      const from = new THREE.Vector3(-0.1, 0.74, 0.66).normalize().multiplyScalar(distance);
      controls.target.set(0, 0.05, PANEL_D / 2);
      camera.position.set(from.x, from.y, PANEL_D / 2 + from.z);
      controls.update();
      draw();
    };
    const size = () => {
      const w = el.clientWidth || 1;
      const h = el.clientHeight || 1;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      draw();
    };
    controls.addEventListener("change", draw);
    const watcher = new ResizeObserver(size);
    watcher.observe(el);
    stage.current = { renderer, scene, camera, controls, draw, home };
    size();
    home();

    return () => {
      watcher.disconnect();
      controls.dispose();
      dispose(scene);
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

      const name = label(cameraName(cam.id), COLOR.muted, 0.12);
      name.position.set(ox + PANEL_W / 2, 0.08, -0.06); // along the far edge, behind the zones
      group.add(name);

      const names: { level: number; span: [number, number] }[] = []; // zone names placed so far in this panel
      for (const zone of cam.zones) {
        const on =(selectedPin?.camera === cam.id && selectedPin.zone === zone.id) || (route?.camera === cam.id && route.bin === zone.id);
        const shape = new THREE.Shape(zone.polygon.map(([x, y]) => new THREE.Vector2(x * PANEL_W, y * PANEL_D)));
        const geometry = new THREE.ExtrudeGeometry(shape, { depth: ZONE_H, bevelEnabled: false });
        geometry.rotateX(Math.PI / 2); // the outline lies on the floor, the extrusion points down...
        geometry.translate(ox, ZONE_H, 0); // ...so lift it by its own height
        const block = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: on ? COLOR.zoneOn : COLOR.zone, roughness: 0.8, transparent: true, opacity: on ? 0.6 : 0.5 }));
        group.add(block);
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color: on ? COLOR.accent : 0xa6b0bb }));
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
            new THREE.LineBasicMaterial({ color: 0xa6b0bb }),
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
        const head = new THREE.Mesh(new THREE.SphereGeometry(on ? 0.058 : 0.05, 20, 14), new THREE.MeshStandardMaterial({ color, roughness: 0.4 }));
        head.position.copy(place(pin.at, ZONE_H + top));
        group.add(head);
        const number = label(String(pin.n), "#13171c", 0.1, false);
        number.position.copy(head.position);
        group.add(number);
        if (on) {
          const text = label(humanize(pin.label), "#5ce1c6", 0.13);
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

  return (
    <>
      <div ref={host} className={dr.stage} />
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <p className={`${ui.muted} ${ui.small}`}>Drag to rotate and tilt. Pinch or scroll to zoom.</p>
        <button type="button" className={dr.pill} onClick={() => stage.current?.home()}>
          Reset the view
        </button>
      </div>
    </>
  );
}
