// Scene "plant" (beat 15): same engine, different data. The campus map plane stands in for the plant's floor plan;
// the route draws like the pointer walking SPOT's paths; the box pulses as the findings are computed. Hard cut in.
import { cut, at, drift } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { state, HOME } = world, t0 = scene.start, tEnd = t0 + scene.duration, L = scene.lines;
  const HERO = { x: 0, y: 6, z: 16, tx: 0, ty: 0, tz: 0, fov: 38 };
  cut(tl, world, t0, HERO, { campus: 1, pulse: 0.6, eyes: 1 });
  tl.to(state, { route: 1, duration: Math.max(1.5, L[0].duration * 0.8), ease: "none" }, at(L, 0, 0.35));
  tl.to(state, { ring: 1, beads: 48, duration: 2.0, ease: "sine.inOut" }, at(L, 1, 0)); // the whole ring of beads (48), not half
  drift(tl, world, t0, scene.duration, HERO, { y: -0.8, z: -2.4, fov: -1 });
  cut(tl, world, tEnd - 0.02, HOME);
}
