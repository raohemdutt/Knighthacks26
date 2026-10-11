// Scene "day": It runs your day. The TIME axis: the day rises as a ring on the desk, classes and deadlines become
// beads, the camera tilts down to a high three-quarter view. One bead hops when the plan slips; one approval flies
// from the box to the calendar bead. Camera + world choreography on the ROOT timeline. See WORLD.md.
// scene = { id, title, start, duration, lines: [{ text, start, duration }] } with every start in ABSOLUTE seconds.
export default function choreograph(tl, world, scene) {
  const { state, HOME, P, hash } = world, t0 = scene.start, tEnd = t0 + scene.duration;
  const L = scene.lines;
  const at = (i, f) => L[i].start + f * L[i].duration; // a point inside line i, as a fraction of the spoken line

  const A = (i) => (i / 48) * Math.PI * 2 - Math.PI / 2; // bead i sits at this angle on the ring
  const BEAD_Y = -0.89; // bead centre height (desk -1 + half a bead)
  // The renderer jitters particle 0 by a fixed hash offset; cancel it so a single dot lands exactly on a bead.
  const off = [(hash(0, 2) - 0.5) * 0.5, (hash(0, 3) - 0.5) * 0.5, (hash(0, 4) - 0.5) * 0.5];
  const exact = (p) => [p[0] - off[0], p[1] - off[1], p[2] - off[2]];
  const atY = (p, y) => [p[0], y, p[2]];

  // One bead hops to a later slot when the plan slips; one approval travels from the box to that slot.
  const replan = world.flow("day-replan", { from: exact(atY(P.ring(A(6)), BEAD_Y)), to: exact(atY(P.ring(A(13)), BEAD_Y)), count: 1, lift: 1.2, size: 2.2 });
  const approve = world.flow("day-approve", { from: exact(P.box), to: exact(atY(P.ring(A(13)), BEAD_Y)), count: 1, lift: 1.7, size: 2.6 });
  replan.spread = 0; approve.spread = 0;

  // Labels sit a little inside the ring (toward the box) so the week strip at the right never covers them.
  const inward = (p, k, y) => [p[0] * k, y, p[2] * k];
  const lClasses = world.label("day-classes", { text: "Classes", at: inward(P.ring(A(1)), 1.0, 0.1), size: 84, scale: 1.1 });
  const lDeadline = world.label("day-deadline", { text: "Deadline", at: inward(P.ring(A(7)), 0.78, -0.2), size: 84, scale: 1.1 });
  const lReplan = world.label("day-replan", { text: "Re-planned", at: inward(P.ring(A(13)), 0.7, -0.1), size: 84, scale: 1.1 });

  // Shots. HIGH: a high three-quarter from the front right, the ring weighted left so the week strip owns the right.
  const HIGH = { x: 4.0, y: 10.6, z: 9.0, tx: 1.7, ty: -1.0, tz: -0.3, fov: 38 };
  const HIGH_DRIFT = { x: 3.3, y: 10.2, z: 9.6, tx: 1.6, ty: -1.0, tz: -0.2, fov: 38 };
  const CLOSE = { x: 3.0, y: 8.4, z: 8.8, tx: 1.8, ty: -0.8, tz: 0.0, fov: 37 };

  // ---- Line 0: "Point it at your week": the day rises as a ring; the camera tilts down to see it. ----
  tl.to(state, { ring: 1, duration: 1.5, ease: "power2.out" }, at(0, 0.03));
  world.leg(tl, at(0, 0.03), HIGH, 3.6, "power2.inOut");
  // "reads your classes and deadlines": beads appear in order around the ring, two labels name them.
  tl.to(state, { beads: 10, duration: 2.6, ease: "power2.out" }, at(0, 0.3));
  tl.fromTo(lClasses, { opacity: 0, lift: -0.3 }, { opacity: 1, lift: 0, duration: 0.7, ease: "power3.out" }, at(0, 0.4));
  tl.fromTo(lDeadline, { opacity: 0, lift: -0.3 }, { opacity: 1, lift: 0, duration: 0.7, ease: "power3.out" }, at(0, 0.55));
  // "watches how your day really goes at the desk": the camera settles.

  // ---- Line 1: "Each morning it writes one plan": a slow drift keeps the shot alive while the plan card writes. ----
  world.leg(tl, L[1].start, HIGH_DRIFT, L[1].duration, "none");
  tl.to(lClasses, { opacity: 0, duration: 0.6, ease: "power2.in" }, at(1, 0.05));
  tl.to(lDeadline, { opacity: 0, duration: 0.6, ease: "power2.in" }, at(1, 0.1));
  // "When something slips, it re-plans": one bead hops to a later slot; the label lands when it lands.
  tl.fromTo(replan, { progress: 0 }, { progress: 0.999, duration: 0.9, ease: "power2.inOut" }, at(1, 0.8));
  tl.fromTo(lReplan, { opacity: 0, lift: -0.3 }, { opacity: 1, lift: 0, duration: 0.6, ease: "power3.out" }, at(1, 0.8) + 0.8);

  // ---- Line 2: "Nothing is written to your calendar until you tap approve". ----
  world.leg(tl, L[2].start, CLOSE, 2.2, "power2.inOut");
  tl.to(lReplan, { opacity: 0, duration: 0.5, ease: "power2.in" }, at(2, 0.62));
  // "tap approve": the approval leaves the box and settles on the re-planned bead.
  tl.fromTo(approve, { progress: 0 }, { progress: 0.999, duration: 0.8, ease: "power2.out" }, at(2, 0.7));

  // ---- Hand over: back to HOME, ring and beads gone, flows parked, before the scene ends. ----
  const back = Math.min(at(2, 0.7) + 0.4, tEnd - 1.8);
  world.leg(tl, back, { ...HOME }, 1.7, "power2.inOut");
  tl.to(state, { ring: 0, beads: 0, duration: 1.0, ease: "power2.inOut" }, tEnd - 1.1);
  tl.to(approve, { density: 0, duration: 0.5, ease: "power2.in" }, tEnd - 1.0);
  tl.to(replan, { density: 0, duration: 0.5, ease: "power2.in" }, tEnd - 1.0);
  tl.to(approve, { progress: 0, duration: 0.2, ease: "none" }, tEnd - 0.45);
  tl.to(replan, { progress: 0, duration: 0.2, ease: "none" }, tEnd - 0.45);
  tl.to(approve, { density: 1, duration: 0.15, ease: "none" }, tEnd - 0.2);
  tl.to(replan, { density: 1, duration: 0.15, ease: "none" }, tEnd - 0.2);
}
