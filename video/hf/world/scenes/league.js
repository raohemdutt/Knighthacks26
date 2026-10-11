// Scene "league" (beat 12): Sunday evening at the desk; the league, three calendars agreeing, dinner with Ilan and
// the meals goal met all play in the app window. Hard cut in.
import { cut, at, drift } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { state, SHOTS, HOME } = world, t0 = scene.start, tEnd = t0 + scene.duration, L = scene.lines;
  const { person, SPOTS } = world.room;
  cut(tl, world, t0, SHOTS.overShoulder, { room: 1, person: 1, eyes: 1, pulse: 0.5 });
  person.act([{ t0, t1: 1e9, clip: "Sitting_Idle_Loop", loop: true, speed: 0.6, from: SPOTS.seat, yaw: [SPOTS.faceDesk], fade: 0 }]);
  tl.to(state, { personLook: 0.85, duration: 0.6 }, at(L, 1, 0));
  tl.to(state, { personTalk: 1, duration: 0.15 }, at(L, 2, 0));
  tl.to(state, { personTalk: 0, personSmile: 1, duration: 0.3 }, at(L, 2, 1));
  drift(tl, world, t0, scene.duration, SHOTS.overShoulder, { x: -1.5, y: -0.6, z: -2.0, fov: -2 });
  cut(tl, world, tEnd - 0.02, HOME);
}
