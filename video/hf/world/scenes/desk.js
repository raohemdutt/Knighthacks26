// Scene "desk" (beat 5): afternoon, he has been sitting a long time, strained; Atlas checks in; on "Go fill your
// bottle" he stands up. Hard cut in.
import { cut, at, drift } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { state, SHOTS, HOME } = world, t0 = scene.start, tEnd = t0 + scene.duration, L = scene.lines;
  const { person, SPOTS } = world.room;
  cut(tl, world, t0, SHOTS.overShoulder, { room: 1, person: 1, personStrain: 1 });
  const ex = person.duration("Sitting_Exit"), up = Math.min(tEnd - ex - 0.2, at(L, 2, 0.85));
  person.act([
    { t0, t1: up, clip: "Sitting_Idle_Loop", loop: true, speed: 0.4, from: SPOTS.seat, yaw: [SPOTS.faceDesk], fade: 0 },
    { t0: up, t1: up + ex, clip: "Sitting_Exit", from: SPOTS.seat, to: SPOTS.standDesk, yaw: [SPOTS.faceDesk], fade: 0.2 },
    { t0: up + ex, t1: 1e9, clip: "Idle_Loop", loop: true, from: SPOTS.standDesk, yaw: [SPOTS.faceDesk + 0.4], fade: 0.25 },
  ]);
  tl.to(state, { eyes: 1, pulse: 0.6, duration: 0.6 }, Math.max(t0, at(L, 0, 0) - 0.4));
  tl.to(state, { personLook: 0.8, duration: 0.7, ease: "power2.inOut" }, at(L, 0, 0.1));
  tl.to(state, { personLook: 0.3, duration: 0.6, ease: "power2.inOut" }, at(L, 1, 0.2));
  tl.to(state, { personLook: 0.9, personStrain: 0.3, duration: 0.6, ease: "power2.inOut" }, at(L, 2, 0.1));
  drift(tl, world, t0, scene.duration * 0.6, SHOTS.overShoulder, { x: -1.2, z: -1.6, fov: -2 });
  world.leg(tl, t0 + scene.duration * 0.6, { ...SHOTS.roomWide, x: 24, z: 34, fov: 38 }, scene.duration * 0.38, "power2.inOut");
  cut(tl, world, tEnd - 0.02, HOME);
}
