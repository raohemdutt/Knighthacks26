// Scene "faces" (beat 6): the privacy shell round the box; the overlay shows clear on your own screen, blurred when
// saved, blurred when it leaves. Hard cut in.
import { cut, drift } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { state, HOME } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  const SHOT = { ...HOME, x: 3.8, y: 2.2, z: 6.4, ty: -0.2, fov: 34 };
  cut(tl, world, t0, SHOT, { eyes: 1 });
  tl.to(state, { membrane: 1, duration: 1.2, ease: "power2.out" }, t0 + 0.2);
  drift(tl, world, t0, scene.duration, SHOT, { x: -1.0, z: 0.6 });
  cut(tl, world, tEnd - 0.02, HOME);
}
