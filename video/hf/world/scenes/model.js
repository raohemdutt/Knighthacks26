// Scene "model" (beat 4): see-through around the Jetson; the overlay says Qwen, Fast, Accurate, Private, 48 / 48.
import { cut, drift } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { state, SHOTS, HOME } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  cut(tl, world, t0, SHOTS.xray, { xray: 1, pulse: 1, hl: { jetson: 1 } });
  drift(tl, world, t0, scene.duration, SHOTS.xray, { x: -0.6, z: 0.4, fov: -2 });
  // The walls (and the front panel) come back before the cut, instead of popping back on it (Dom).
  tl.to(state, { xray: 0, duration: 1.3, ease: "sine.inOut" }, tEnd - 1.5);
  tl.to(state.hl, { jetson: 0, duration: 1.3, ease: "sine.inOut" }, tEnd - 1.5);
  cut(tl, world, tEnd - 0.02, HOME);
}
