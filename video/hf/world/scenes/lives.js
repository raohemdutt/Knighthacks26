// Scene "lives" (beat 14): "Not just for students: a night-shift worker's upside-down day, a family's chores, a plant
// floor, at twelve watts idle." One line. The camera rises high over the campus while the three other lives pop up
// (the overlay cards), then pulls in toward the box. The Day Run route and its stop labels belonged to a line that was
// cut (v4); they are gone, and nothing here is scheduled past the end of the scene.
// Camera + world choreography on the ROOT timeline (absolute seconds). See WORLD.md.
export default function choreograph(tl, world, scene) {
  const { state } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  const HIGH = { x: 7.0, y: 25.0, z: 21.0, tx: 0.5, ty: -1.0, tz: -1.2, fov: 40 };
  const HIGH_DRIFT = { x: 4.5, y: 24.0, z: 22.5, tx: 0.5, ty: -1.0, tz: -1.0, fov: 40 };
  const rise = Math.min(2.4, scene.duration * 0.35);
  tl.to(state, { campus: 1, duration: 1.2, ease: "power2.out" }, t0);
  tl.to(state, { friends: 0, friendsLink: 0, room: 0, kitchen: 0, duration: 0.8, ease: "power2.inOut" }, t0);
  world.leg(tl, t0, HIGH, rise, "power2.inOut");
  world.leg(tl, t0 + rise, HIGH_DRIFT, Math.max(0.3, tEnd - 0.1 - (t0 + rise)), "none");
}
