// Scene "shift" (beat 16): Xander's shift in the control room (sets/control.js). Behind Xander at the console as
// Xander asks; the console Atlas answers; through the window SPOT walks to heat exchanger 2 when it's sent back; the
// trainee steps in for Shift Run at the end. The app window carries the Inspection page. Hard cut in.
import { cut, at, drift } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { state, SHOTS, HOME } = world, t0 = scene.start, tEnd = t0 + scene.duration, L = scene.lines;
  cut(tl, world, t0, SHOTS.controlWide, { control: 1 });
  // Xander talks on Xander's lines (0, 2, 4); the console Atlas lights when it answers (1, 3, 5).
  for (const i of [0, 2, 4]) {
    if (!L[i] || !L[i].text) continue;
    tl.to(state, { xTalk: 1, duration: 0.15 }, at(L, i, 0));
    tl.to(state, { xTalk: 0, duration: 0.2 }, at(L, i, 1));
  }
  tl.to(state, { consoleCount: 1, duration: 0.5 }, at(L, 1, 0));
  drift(tl, world, t0, at(L, 1, 0) - t0, SHOTS.controlWide, { x: -1.2, z: -1.5 });
  // "Three things...": onto the console Atlas.
  world.leg(tl, at(L, 1, 0), SHOTS.consoleClose, 1.6, "power2.inOut");
  drift(tl, world, at(L, 1, 0) + 1.6, Math.max(0.5, at(L, 2, 0) - at(L, 1, 0) - 1.6), SHOTS.consoleClose, { x: 0.6, z: 0.3 });
  // "Where is it?" ... the pointer walks (overlay); "Send SPOT back to it": through the window, SPOT walks to HX-202.
  world.leg(tl, at(L, 2, 0), SHOTS.controlWide, 1.4, "power2.inOut");
  world.leg(tl, at(L, 4, 0), SHOTS.plantView, 1.4, "power2.inOut");
  tl.fromTo(state, { spotWalk: 0 }, { spotWalk: 0.42, duration: Math.max(1.5, tEnd - at(L, 4, 0) - 2.0), ease: "none", immediateRender: false }, at(L, 4, 0.3));
  // Shift Run: the trainee steps in; back behind Xander for the handoff.
  tl.to(state, { trainee: 1, duration: 0.6 }, at(L, 5, 0.5));
  world.leg(tl, at(L, 5, 0.6), { ...SHOTS.controlWide, x: SHOTS.controlWide.x + 2, z: SHOTS.controlWide.z - 2 }, Math.max(0.6, tEnd - at(L, 5, 0.6) - 0.1), "power2.inOut");
  cut(tl, world, tEnd - 0.02, HOME);
}
