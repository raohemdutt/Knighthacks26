// Scene "problem" (beat 0): the dorm in morning light, the person asleep in bed; the scattered app windows
// (overlay) drift together into the box on its shelf as the camera pushes in to it. Opens on a hard cut.
import { cut, at, camPath } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { SHOTS, HOME } = world, t0 = scene.start, tEnd = t0 + scene.duration, L = scene.lines;
  const { person, SPOTS } = world.room;
  const WIDE = { ...SHOTS.roomWide, x: 27, z: 38, fov: 38 };
  cut(tl, world, t0, SHOTS.roomWide, { room: 1, person: 1, personSleepy: 1 });
  person.act([{ t0, t1: 1e9, clip: "Idle_Loop", loop: true, speed: 0.3, from: SPOTS.bedLie, yaw: [0], tip: [Math.PI / 2], fade: 0 }]); // asleep
  world.leg(tl, t0, WIDE, Math.max(0.5, at(L, 1, 0) - t0), "none");
  camPath(tl, world, at(L, 1, 0), Math.max(1, tEnd - at(L, 1, 0) - 0.2),
    [WIDE, { x: 8, y: 1.5, z: 14, tx: 0.4, ty: -1.0, tz: 1.0, fov: 34 }, SHOTS.deskClose], "power2.inOut");
  tl.to(world.state, { eyes: 1, pulse: 0.6, duration: 0.8 }, at(L, 1, 0.5));
  cut(tl, world, tEnd - 0.02, HOME);
}
