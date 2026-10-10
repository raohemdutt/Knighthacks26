// Slot 0: Atlas. Camera + world choreography on the ROOT timeline; every time is t0 + seconds. See WORLD.md.
export default function choreograph(tl, world, t0) {
  const { cam, state, HOME } = world;
  tl.set(cam, { ...HOME }, t0);
  world.leg(tl, t0 + 1.5, { x: 2.5, y: 2.0, z: 7.5, fov: 34 }, 4, "power2.inOut");
  tl.to(state, { spin: 0.12, duration: 20, ease: "none" }, t0 + 1.5);
  world.leg(tl, t0 + 52, { ...HOME }, 4, "power2.inOut");
  tl.to(state, { spin: 0, duration: 4, ease: "power2.inOut" }, t0 + 54);
}
