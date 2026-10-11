// Scene "sortit" (beat 8): the kitchen; Atlas offers a round; he holds up a cup (the overlay shows the route and the
// score), then the wall camera's own view. Hard cut in.
import { cut, drift } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { SHOTS, HOME } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  cut(tl, world, t0, SHOTS.kitchenWide, { kitchen: 1, binCount: 1 });
  drift(tl, world, t0, scene.duration * 0.45, SHOTS.kitchenWide, { x: 1.5, z: -3 });
  tl.set(world.cam, { ...SHOTS.binCam }, t0 + scene.duration * 0.45);
  cut(tl, world, tEnd - 0.02, HOME);
}
