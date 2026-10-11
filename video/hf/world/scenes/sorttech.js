// Scene "sorttech" (beat 9): the bin camera's own view, then the counter Atlas counting; the overlay names YOLO11 at
// 41 fps, the tracking, the grid route, UCF's bin rules and the look guide. Hard cut in.
import { cut, at, drift } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { state, SHOTS, HOME } = world, t0 = scene.start, tEnd = t0 + scene.duration, L = scene.lines;
  cut(tl, world, t0, SHOTS.binCam, { kitchen: 1 });
  tl.to(state, { binCount: 1, duration: 0.6 }, at(L, 0, 0.5));
  tl.set(world.cam, { ...SHOTS.counterAtlas }, at(L, 1, 0));
  drift(tl, world, at(L, 1, 0), tEnd - at(L, 1, 0), SHOTS.counterAtlas, { x: 0.5, z: 0.35, fov: -1 });
  cut(tl, world, tEnd - 0.02, HOME);
}
