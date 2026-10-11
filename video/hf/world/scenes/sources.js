// Scene "sources" (beat 3): sparks (the overlay names them) fly into the box; the Jetson warms; then the camera's
// lens lights as the narrator says what the camera captures (the Encompass story). Hard cut in.
import { cut, at, drift } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { state, HOME, P } = world, t0 = scene.start, tEnd = t0 + scene.duration, L = scene.lines;
  const SHOT = { ...HOME, x: 3.6, y: 1.4, z: 5.6, fov: 32 };
  cut(tl, world, t0, SHOT, { pulse: 0.4 });
  drift(tl, world, t0, scene.duration, SHOT, { x: -0.8, z: -1.0, fov: -2 });
  const f1 = world.flow("src-a", { from: [-6, 2.5, 5], to: P.box, count: 28, lift: 2.6 });
  const f2 = world.flow("src-b", { from: [6, 3.0, 4], to: P.box, count: 28, lift: 2.2 });
  tl.set([f1, f2], { progress: 0, density: 1 }, t0);
  tl.to([f1, f2], { progress: 1, duration: Math.max(1, L[0].duration), ease: "power1.inOut" }, at(L, 0, 0));
  tl.to([f1, f2], { density: 0, duration: 0.5 }, at(L, 1, 0));
  tl.set([f1, f2], { progress: 0 }, tEnd - 0.05);
  tl.to(state, { pulse: 1, duration: 0.6 }, at(L, 0, 0.7));
  tl.to(state, { eyes: 1, xray: 0.45, duration: 0.6 }, at(L, 1, 0));
  tl.to(state.hl, { webcam: 1, duration: 0.5 }, at(L, 1, 0.05));
  cut(tl, world, tEnd - 0.02, HOME);
}
