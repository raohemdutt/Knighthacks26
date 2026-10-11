// Scene "end": Atlas. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
//
// Inherits the dayrun hero shot ({ x: 0, y: 6, z: 16, tx: 0, ty: 0, tz: 0, fov: 38 }, campus 1, ring 1, membrane 1)
// and does one slow push in while the box's heartbeat comes up. Nothing is reset: the film ends here.
export default function choreograph(tl, world, scene) {
  const { state } = world, t0 = scene.start, dur = scene.duration;
  world.leg(tl, t0, { x: 0, y: 5.3, z: 13.6, tx: 0, ty: 0.1, tz: 0, fov: 37 }, dur - 0.1, "power1.inOut");
  tl.to(state, { pulse: 0.7, duration: Math.min(2.4, dur * 0.5), ease: "power2.inOut" }, t0 + 0.6);
}
