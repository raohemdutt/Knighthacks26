// THROWAWAY PREVIEW of the world's sets (written by the world agent). The intro agent replaces this file.
// It tweens state and the camera through: assembly (drop), studio three-quarter, x-ray + Jetson highlight, the exploded
// diagram with labels, each highlight, the room reveal (deskClose -> roomWide -> overShoulder), the kitchen
// (kitchenWide -> binCam, the can throw, binCount), then back to HOME with everything at 0 (the hand-over).
// Times are scene-local seconds from t0; the narration is ignored here.
export default function choreograph(tl, world, scene) {
  const { cam, state, HOME, SHOTS } = world, t0 = scene.start, END = t0 + scene.duration;
  const T = (s) => t0 + s;
  const hl = state.hl;

  tl.set(cam, { ...HOME, x: HOME.x + 0.5, y: HOME.y + 0.3, z: HOME.z + 0.8 }, t0);
  tl.fromTo(state, { drop: 1 }, { drop: 0, duration: 2.0, ease: "power3.out" }, T(0.1));
  world.leg(tl, T(0.1), { ...HOME }, 2.4, "power2.out");

  // X-ray, then the Jetson lights up.
  world.leg(tl, T(3.1), SHOTS.xray, 1.4, "power2.inOut");
  tl.to(state, { xray: 1, duration: 1.0, ease: "power2.inOut" }, T(3.3));
  tl.to(hl, { jetson: 1, duration: 0.5, ease: "power2.out" }, T(4.5));
  tl.to(state, { pulse: 1, duration: 0.5 }, T(4.5));
  tl.to(hl, { jetson: 0, duration: 0.4 }, T(6.0));
  tl.to(state, { xray: 0, pulse: 0, duration: 0.6 }, T(6.0));

  // The exploded diagram, labelled.
  world.leg(tl, T(6.0), SHOTS.exploded, 1.8, "power2.inOut");
  tl.to(state, { explode: 1, duration: 1.6, ease: "power3.inOut" }, T(6.2));
  const names = [["webcam", "Webcam"], ["lid", "Lid"], ["jetson", "Jetson Orin Nano"], ["speaker", "Speaker"], ["front", "Honeycomb front"]];
  const off = { webcam: [1.15, 0.05, 0], lid: [1.15, 0.05, 0], jetson: [1.25, 0.1, -0.1], speaker: [1.15, -0.05, 0.1], front: [0.0, -0.62, 0.15] };
  const labels = names.map(([k, text]) => { const p = world.at(k, 1); const o = off[k]; return world.label(`preview-${k}`, { text, at: [p[0] + o[0], p[1] + o[1], p[2] + o[2]], scale: 0.24 }); });
  labels.forEach((l, i) => tl.fromTo(l, { opacity: 0, lift: -0.3 }, { opacity: 1, lift: 0, duration: 0.5, ease: "power3.out" }, T(7.7 + i * 0.12)));
  labels.forEach((l) => tl.to(l, { opacity: 0, duration: 0.3 }, T(9.6)));
  // Each highlight in turn.
  ["webcam", "speaker", "lid", "front"].forEach((k, i) => {
    tl.to(hl, { [k]: 1, duration: 0.25 }, T(9.8 + i * 0.6));
    tl.to(hl, { [k]: 0, duration: 0.25 }, T(10.15 + i * 0.6));
  });
  tl.to(state, { explode: 0, duration: 1.2, ease: "power3.inOut" }, T(12.3));
  world.leg(tl, T(12.3), SHOTS.deskClose, 1.6, "power2.inOut");

  // The room rises in around the desk; the person appears and turns toward the device.
  tl.to(state, { room: 1, duration: 1.6, ease: "power2.out" }, T(14.0));
  tl.to(state, { person: 1, duration: 0.8, ease: "power2.out" }, T(15.0));
  world.leg(tl, T(14.8), SHOTS.roomWide, 2.2, "power2.inOut");
  tl.to(state, { personLook: 0.75, eyes: 1, duration: 0.8, ease: "power2.inOut" }, T(16.4));
  world.leg(tl, T(18.1), SHOTS.overShoulder, 1.3, "power2.inOut");
  tl.to(state, { personLook: 0, duration: 0.6 }, T(18.1));

  // To the kitchen.
  tl.to(state, { kitchen: 1, duration: 1.0, ease: "power2.out" }, T(19.4));
  tl.to(state, { room: 0, person: 0, duration: 0.8 }, T(19.6));
  world.leg(tl, T(19.5), SHOTS.kitchenWide, 1.6, "power2.inOut");
  world.leg(tl, T(22.0), SHOTS.binCam, 0.9, "power2.inOut");
  tl.to(state, { canThrow: 1, duration: 1.6, ease: "none" }, T(22.9));
  tl.to(state, { binCount: 1, duration: 0.5 }, T(24.5));

  // Hand-over: back to HOME, everything at 0.
  tl.to(state, { kitchen: 0, binCount: 0, eyes: 0, duration: 0.8 }, T(25.0));
  tl.set(state, { canThrow: 0 }, T(25.9));
  world.leg(tl, T(25.0), { ...HOME }, Math.min(1.7, END - T(25.0) - 0.05), "power2.inOut");
}
