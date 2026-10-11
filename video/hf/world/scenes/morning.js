// Scene "morning" (beat 2): he wakes in bed and sits up; "Good morning" from the bed's edge, the brief while he walks to the desk and sits, the meals
// goal, then "where are my keys?" and a soft label on the keys behind the mug. Hard cut in.
import { cut, at, camPath, yawTo } from "./_kit.js";

export default function choreograph(tl, world, scene) {
  const { state, SHOTS, HOME, P } = world, t0 = scene.start, tEnd = t0 + scene.duration, L = scene.lines;
  const { person, SPOTS } = world.room, F = world.room.F;
  const D = (c) => person.duration(c);
  cut(tl, world, t0, SHOTS.roomWide, { room: 1, person: 1, personSleepy: 1, light: 0.22 });
  // Dark while he sleeps and gets up; the lights come up as he says good morning.
  tl.to(state, { light: 1, duration: 1.6, ease: "power2.inOut" }, at(L, 0, 0.2));

  // The person: on the bed's edge for "Good morning", up as Atlas answers, round the chair, seated for the brief.
  const exitAt = at(L, 1, 0.1);
  const W1 = [SPOTS.bedFront[0], F, SPOTS.bedFront[2]], W2 = [2.8, F, 13.6], W3 = SPOTS.standDesk;
  const w1 = exitAt + D("Sitting_Exit"), w2 = w1 + 1.35, w3 = w2 + 0.75, sit = w3 + D("Sitting_Enter");
  // Getting up the way a person does: lying on his back, he sits up in place (legs along the bed), swings round to the
  // mattress's edge, says good morning, and then stands.
  const upAt = Math.max(t0 + 0.3, at(L, 0, 0) - 2.3), swing = upAt + 1.1, edge = swing + 1.2;
  person.act([
    { t0, t1: upAt, clip: "Idle_Loop", loop: true, speed: 0.3, from: SPOTS.bedLie, yaw: [0], tip: [Math.PI / 2], fade: 0 },
    { t0: upAt, t1: swing, clip: "Sitting_Idle_Loop", loop: true, speed: 0.5, from: SPOTS.bedSitUp, yaw: [0], tip: [0], fade: 1.1 },
    { t0: swing, t1: edge, clip: "Sitting_Idle_Loop", loop: true, speed: 0.5, offset: 0.55, from: SPOTS.bedSitUp, to: SPOTS.bedEdge, yaw: [0, SPOTS.faceDesk], fade: 0 },
    { t0: edge, t1: exitAt, clip: "Sitting_Idle_Loop", loop: true, speed: 0.5, offset: 1.15, from: SPOTS.bedEdge, yaw: [SPOTS.faceDesk], fade: 0 },
    { t0: exitAt, t1: w1, clip: "Sitting_Exit", from: SPOTS.bedEdge, to: SPOTS.bedFront, yaw: [SPOTS.faceDesk], fade: 0.15 },
    { t0: w1, t1: w2, clip: "Walk_Loop", loop: true, speed: 0.95, from: W1, to: W2, yaw: [SPOTS.faceDesk, yawTo(W1, W2)], ease: "none", fade: 0.3 },
    { t0: w2, t1: w3, clip: "Walk_Loop", loop: true, speed: 0.95, offset: (w2 - w1) * 0.95, from: W2, to: W3, yaw: [yawTo(W1, W2), yawTo(W2, W3)], ease: "none", fade: 0 },
    { t0: w3, t1: sit, clip: "Sitting_Enter", from: W3, to: SPOTS.seat, yaw: [yawTo(W2, W3), SPOTS.faceDesk], fade: 0.25 },
    { t0: sit, t1: 1e9, clip: "Sitting_Idle_Loop", loop: true, speed: 0.6, from: SPOTS.seat, yaw: [SPOTS.faceDesk], fade: 0.2 },
  ]);
  // Faces: sleepy, waking, a smile when the plan is locked; his mouth moves on his own lines.
  tl.to(state, { personSleepy: 0, duration: 1.2, ease: "power2.inOut" }, exitAt + 0.4);
  for (const i of [0, 6, 8]) {
    if (!L[i] || !L[i].text) continue;
    tl.to(state, { personTalk: 1, duration: 0.15 }, at(L, i, 0));
    tl.to(state, { personTalk: 0, duration: 0.2 }, at(L, i, 1));
  }
  tl.to(state, { personSmile: 1, duration: 0.5 }, at(L, 7, 0.9));
  tl.to(state, { personSmile: 0, duration: 0.5 }, at(L, 8, 0));

  // Atlas wakes on its shelf when he says good morning.
  tl.to(state, { eyes: 1, pulse: 0.7, duration: 0.6, ease: "power2.out" }, at(L, 0, 0.6));

  // Camera: the room wide on the bed; over the shoulder once he has walked over; a slow drift through the brief.
  const WIDE = { ...SHOTS.roomWide, x: 28, z: 39, fov: 38 };
  const OVER_DRIFT = { x: 14.2, y: 8.2, z: 25.6, tx: -3.5, ty: -4.0, tz: 5.2, fov: 34 };
  const walkCam = Math.max(w1, t0 + 0.5), seated = Math.max(sit, walkCam + 1.5);
  world.leg(tl, t0, WIDE, walkCam - t0, "none");
  camPath(tl, world, walkCam, seated - walkCam, [WIDE, SHOTS.overShoulder], "power2.inOut");
  world.leg(tl, seated, OVER_DRIFT, Math.max(0.5, at(L, 9, 0) - seated), "none");
  // He looks at Atlas when he answers it, and when he asks for his keys.
  tl.to(state, { personLook: 0.85, duration: 0.6, ease: "power2.inOut" }, at(L, 5, 0.6));
  tl.to(state, { personLook: 0, duration: 0.6, ease: "power2.inOut" }, at(L, 7, 1));
  tl.to(state, { personLook: 0.9, duration: 0.6, ease: "power2.inOut" }, at(L, 8, 0));
  // "Behind the mug on your desk": the camera finds the desk; a soft label lands on the keys.
  const DESK = { x: -2.6, y: 1.6, z: 16.4, tx: -7.0, ty: -4.4, tz: 12.6, fov: 30 }; // looking down: the mug and the keys behind it
  world.leg(tl, at(L, 9, 0), DESK, 1.4, "power2.inOut");
  const keys = world.label("keys-here", { text: "Your keys", at: [P.keys[0], P.keys[1] + 0.9, P.keys[2]], scale: 0.5 });
  tl.set(keys, { opacity: 0 }, t0);
  tl.to(keys, { opacity: 1, duration: 0.4 }, at(L, 9, 0.5));
  tl.to(keys, { opacity: 0, duration: 0.3 }, tEnd - 0.4);
  cut(tl, world, tEnd - 0.02, HOME);
}
