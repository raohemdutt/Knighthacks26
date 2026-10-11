// Scene "shift" (beat 16): Xander's shift. The Inspection page carries the scene in the app window (triage, the
// pointer on the floor plan, the trust line, the approval, Shift Run); the box on the console listens and answers.
import { cut, drift } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { HOME } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  const SHOT = { ...HOME, x: -2.6, y: 0.9, z: 5.2, ty: -0.4, fov: 32 };
  cut(tl, world, t0, SHOT, { eyes: 1, pulse: 0.6 });
  drift(tl, world, t0, scene.duration * 0.5, SHOT, { x: 1.4, z: -0.4 });
  world.leg(tl, t0 + scene.duration * 0.5, { ...HOME, x: 3.4, y: 1.4, z: 5.6, fov: 33 }, scene.duration * 0.48, "power2.inOut");
  cut(tl, world, tEnd - 0.02, HOME);
}
