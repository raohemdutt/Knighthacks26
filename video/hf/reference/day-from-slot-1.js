// Slot 1: It runs your day. The TIME axis: the day rises as a ring on the desk, classes and deadlines become
// beads, the camera tilts to a top view while the real phone recording floats at the right (overlay).
// Every position is t0 + seconds, read from voice.json: line 0 at 1.5, line 1 at 8.96 ("re-plans" ~14.0),
// line 2 at 18.88 ("tap approve" ~21.6). See WORLD.md.
export default function choreograph(tl, world, t0) {
  const { cam, state, HOME, P, hash } = world;
  const A = (i) => (i / 48) * Math.PI * 2 - Math.PI / 2; // bead i sits at this angle on the ring
  // The renderer jitters particle 0 by a fixed hash offset; cancel it so a single dot lands exactly on a bead.
  const off = [(hash(0, 2) - 0.5) * 0.5, (hash(0, 3) - 0.5) * 0.5, (hash(0, 4) - 0.5) * 0.5];
  const exact = (p) => [p[0] - off[0], p[1] - off[1], p[2] - off[2]];
  const atY = (p, y) => [p[0], y, p[2]];

  // One bead jumps to a later slot when the plan slips; one approval travels from the box to the calendar bead.
  const replan = world.flow("s1-replan", { from: exact(atY(P.ring(A(19)), -0.86)), to: exact(atY(P.ring(A(29)), -0.86)), count: 1, lift: 1.3, size: 2.2 });
  const approve = world.flow("s1-approve", { from: exact(P.box), to: exact(atY(P.ring(A(38)), -0.86)), count: 1, lift: 1.8, size: 2.4 });
  replan.spread = 0; approve.spread = 0;

  const lClasses = world.label("s1-classes", { text: "Classes", at: atY(P.ring(A(4)), 0.2), size: 84, scale: 1.3 });
  const lDeadline = world.label("s1-deadline", { text: "Deadline", at: atY(P.ring(A(9)), 0.2), size: 84, scale: 1.3 });
  const lReplan = world.label("s1-replan", { text: "Re-planned", at: atY(P.ring(A(29)), 0.2), size: 84, scale: 1.3 });
  const lCal = world.label("s1-cal", { text: "On your calendar", at: atY(P.ring(A(38)), 0.2), size: 72, scale: 1.6 });

  const RISE = { x: 0, y: 3.8, z: 9.2, tx: 0, ty: -0.3, tz: 0, fov: 35 };
  const TOP = { x: 3.3, y: 13.6, z: 3.6, tx: 3.3, ty: -0.9, tz: -0.6, fov: 40 };
  const TOP_DRIFT = { x: 2.5, y: 13.0, z: 5.4, tx: 3.1, ty: -0.9, tz: -0.3, fov: 40 };

  // Title card: calm at home.
  tl.set(cam, { ...HOME }, t0);

  // "Point it at your week": the day rises as a ring; the camera lifts to see it.
  tl.to(state, { ring: 1, duration: 1.6, ease: "power2.out" }, t0 + 1.5);
  world.leg(tl, t0 + 1.5, RISE, 3.0, "power2.inOut");

  // "reads your classes and deadlines": beads appear in order around the ring.
  tl.to(state, { beads: 10, duration: 2.6, ease: "power1.inOut" }, t0 + 2.6);
  tl.fromTo(lClasses, { opacity: 0, lift: -0.25 }, { opacity: 1, lift: 0, duration: 0.6, ease: "power3.out" }, t0 + 3.3);
  tl.fromTo(lDeadline, { opacity: 0, lift: -0.25 }, { opacity: 1, lift: 0, duration: 0.6, ease: "power3.out" }, t0 + 4.5);

  // "watches how your day really goes at the desk": tilt down to the top view, the ring on the left.
  world.leg(tl, t0 + 5.6, TOP, 3.2, "power4.out");

  // "Each morning it writes one plan ... in the evening": the day fills in, bead by bead, until the evening.
  tl.to(state, { beads: 48, duration: 8.5, ease: "none" }, t0 + 9.5);
  tl.to(lClasses, { opacity: 0, duration: 0.8, ease: "power2.in" }, t0 + 12.4);
  tl.to(lDeadline, { opacity: 0, duration: 0.8, ease: "power2.in" }, t0 + 12.6);

  // "When something slips, it re-plans" (~14.0): one bead hops to a later slot.
  tl.to(replan, { progress: 1, duration: 1.0, ease: "power2.inOut" }, t0 + 13.9);
  tl.fromTo(lReplan, { opacity: 0, lift: -0.25 }, { opacity: 1, lift: 0, duration: 0.6, ease: "power3.out" }, t0 + 14.7);
  tl.to(lReplan, { opacity: 0, duration: 0.8, ease: "power2.in" }, t0 + 18.6);

  // "until you tap approve" (~21.6): the approval leaves the box and settles on one bead, which stays orange.
  tl.to(approve, { progress: 0.999, duration: 0.9, ease: "power2.out" }, t0 + 21.6);
  tl.fromTo(lCal, { opacity: 0, lift: -0.25 }, { opacity: 1, lift: 0, duration: 0.6, ease: "power3.out" }, t0 + 22.5);
  tl.to(lCal, { opacity: 0, duration: 0.8, ease: "power2.in" }, t0 + 27.5);

  // The recording keeps playing: a very slow drift around the top view so the shot never dies.
  world.leg(tl, t0 + 25, TOP_DRIFT, 26, "none");

  // Back home, then reset everything this slot touched before the next title card.
  world.leg(tl, t0 + 52, { ...HOME }, 4, "power2.inOut");
  tl.to(state, { ring: 0, beads: 0, duration: 1.5, ease: "power2.inOut" }, t0 + 56.5);
  tl.to(approve, { density: 0, duration: 0.6, ease: "power2.in" }, t0 + 56.5);
  tl.to(approve, { progress: 0, duration: 0.4, ease: "none" }, t0 + 57.4);
  tl.to(approve, { density: 1, duration: 0.3, ease: "none" }, t0 + 58.2);
  tl.to(replan, { progress: 0, duration: 0.4, ease: "none" }, t0 + 57.4);
}
