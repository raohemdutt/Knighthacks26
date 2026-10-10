import type { Scene } from "./Diagram";

/**
 * One bird's-eye picture per slot, for the minutes that have no recording yet (and the explainer cut). Simple words:
 * a judge reads a node in a glance. Positions are in the 1920x1080 frame; nodes default to 300x120.
 */

const row = (y: number, xs: number[]) => xs.map((x) => ({ x, y }));

export const SCENES: Record<number, Scene> = {
  1: {
    nodes: [
      { id: "classes", label: "Your classes & deadlines", sub: "Canvas, calendar", ...row(300, [200])[0], icon: "📅" },
      { id: "desk", label: "The desk camera", sub: "how the day really goes", ...row(620, [200])[0], icon: "👁" },
      { id: "plan", label: "Atlas writes one plan", sub: "with its reasons, re-plans when something slips", x: 760, y: 440, w: 400, h: 160, sunk: true, icon: "🗒" },
      { id: "ok", label: "You tap approve", sub: "or deny", x: 1320, y: 300, icon: "✅" },
      { id: "cal", label: "Your calendar", sub: "written only after the tap", x: 1320, y: 620, icon: "📆" },
    ],
    edges: [
      { from: "classes", to: "plan" },
      { from: "desk", to: "plan" },
      { from: "plan", to: "ok", label: "proposes" },
      { from: "ok", to: "cal", label: "then writes" },
    ],
  },
  2: {
    nodes: [
      { id: "teach", label: "You tap Teach", sub: "name a new skill", x: 120, y: 470, icon: "🎓" },
      { id: "pics", label: "A few blurred pictures", sub: "leave only after your approval", x: 540, y: 200, w: 360, sunk: true, icon: "🖼" },
      { id: "teacher", label: "Big teacher model", sub: "labels them in the lab", x: 1080, y: 200, w: 340, icon: "🧠" },
      { id: "tune", label: "Small model learns", sub: "fine-tuned on the answers", x: 1500, y: 470, w: 340, icon: "✨" },
      { id: "test", label: "A test decides", sub: "better than before? only then ship", x: 1080, y: 740, w: 340, icon: "⚖️" },
    ],
    edges: [
      { from: "teach", to: "pics" },
      { from: "pics", to: "teacher" },
      { from: "teacher", to: "tune", label: "answers" },
      { from: "tune", to: "test" },
      { from: "test", to: "teach", label: "ships back to the box", dashed: true, labelBelow: true },
    ],
  },
  3: {
    groups: [{ label: "The lab: two AMD Radeon R9700s on ROCm", x: 560, y: 180, w: 1240, h: 680 }],
    nodes: [
      { id: "box", label: "The box", sub: "Jetson on your desk", x: 80, y: 460, icon: "📦" },
      { id: "a", label: "GPU A: the teacher", sub: "27B Qwen, serving", x: 660, y: 290, w: 420, icon: "🧠" },
      { id: "b", label: "GPU B: the trainer", sub: "fine-tune, test, package", x: 660, y: 620, w: 420, icon: "🛠" },
      { id: "ledger", label: "Every run recorded", sub: "loss curve, minutes, watts", x: 1300, y: 455, w: 400, sunk: true, icon: "📈" },
    ],
    edges: [
      { from: "box", to: "a", label: "a dataset out" },
      { from: "b", to: "box", label: "a model package back", labelBelow: true },
      { from: "a", to: "b", label: "labels" },
      { from: "b", to: "ledger" },
    ],
  },
  4: {
    nodes: [
      { id: "pkg", label: "A small package arrives", sub: "new model, one file set", x: 120, y: 260, w: 360, icon: "📦" },
      { id: "check", label: "The box checks it", sub: "signatures match?", x: 620, y: 260, icon: "🔍" },
      { id: "swap", label: "Swaps it in", sub: "back by itself if it goes quiet", x: 1060, y: 260, w: 360, icon: "🔁" },
      { id: "wj", label: "Watts on the Jetson", x: 320, y: 700, icon: "⚡" },
      { id: "wg", label: "Watts on the GPUs", x: 760, y: 700, icon: "⚡" },
      { id: "j", label: "Joules per answer", sub: "what one question costs", x: 1240, y: 700, w: 340, sunk: true, icon: "🧮" },
    ],
    edges: [
      { from: "pkg", to: "check" },
      { from: "check", to: "swap" },
      { from: "wj", to: "j" },
      { from: "wg", to: "j" },
    ],
  },
  5: {
    groups: [{ label: "All of this runs on the Jetson", x: 120, y: 180, w: 1300, h: 540 }],
    nodes: [
      { id: "cam", label: "Cameras", x: 190, y: 400, icon: "📷" },
      { id: "blur", label: "Faces blurred", sub: "before anything is saved", x: 580, y: 400, icon: "🙈" },
      { id: "vl", label: "Vision model", sub: "understands the scene", x: 970, y: 400, icon: "🧠" },
      { id: "ans", label: "Answers & alerts", sub: "in a few seconds", x: 1520, y: 400, sunk: true, icon: "💬" },
      { id: "net", label: "The internet", sub: "unplugged: nothing changes", x: 1540, y: 800, w: 360, icon: "🔌" },
    ],
    edges: [
      { from: "cam", to: "blur" },
      { from: "blur", to: "vl" },
      { from: "vl", to: "ans" },
      { from: "net", to: "ans", dashed: true, label: "only for learning", labelBelow: true },
    ],
  },
  6: {
    nodes: [
      { id: "frames", label: "Frames", sub: "15 a second", x: 120, y: 300, icon: "🎞" },
      { id: "events", label: "Events", sub: "someone came, a box arrived", x: 560, y: 300, icon: "📍" },
      { id: "eps", label: "Episodes", sub: "what you ask about", x: 1000, y: 300, icon: "🧩" },
      { id: "q", label: "A question, an answer", sub: "with the picture it came from", x: 1440, y: 300, w: 340, sunk: true, icon: "💬" },
      { id: "bins", label: "The bins", x: 560, y: 720, icon: "🗑" },
      { id: "counts", label: "Counts", sub: "what was thrown away, and where it landed", x: 1000, y: 720, w: 380, icon: "🔢" },
    ],
    edges: [
      { from: "frames", to: "events" },
      { from: "events", to: "eps" },
      { from: "eps", to: "q" },
      { from: "bins", to: "counts" },
      { from: "frames", to: "bins", dashed: true },
    ],
  },
  7: {
    nodes: [
      { id: "ta", label: "Table A", sub: "its counts", x: 160, y: 300, icon: "🧮" },
      { id: "tb", label: "Table B", sub: "its counts", x: 160, y: 640, icon: "🧮" },
      { id: "neon", label: "Shared counts", sub: "Neon Postgres: numbers only", x: 700, y: 470, w: 380, sunk: true, icon: "☁️" },
      { id: "rate", label: "Sorted right: before vs after", sub: "a number a campus can act on", x: 1260, y: 300, w: 420, icon: "📊" },
      { id: "board", label: "Leaderboard", sub: "who is winning this week", x: 1260, y: 640, w: 420, icon: "🏆" },
      { id: "home", label: "Pictures stay home", x: 700, y: 860, w: 380, icon: "🏠" },
    ],
    edges: [
      { from: "ta", to: "neon" },
      { from: "tb", to: "neon" },
      { from: "neon", to: "rate" },
      { from: "neon", to: "board" },
    ],
  },
  8: {
    nodes: [
      { id: "you", label: "Your Atlas", sub: "your calendar stays here", x: 160, y: 300, icon: "📦" },
      { id: "f1", label: "A friend's Atlas", sub: "their calendar stays there", x: 160, y: 640, icon: "📦" },
      { id: "neon", label: "Only free or busy", sub: "crosses, through Neon", x: 700, y: 470, w: 380, sunk: true, icon: "☁️" },
      { id: "time", label: "A time that works", sub: "the boxes agree", x: 1260, y: 300, w: 380, icon: "🤝" },
      { id: "tap", label: "One tap", sub: "then it lands in everyone's plan", x: 1260, y: 640, w: 380, icon: "✅" },
    ],
    edges: [
      { from: "you", to: "neon" },
      { from: "f1", to: "neon" },
      { from: "neon", to: "time" },
      { from: "time", to: "tap" },
    ],
  },
  9: {
    nodes: [
      { id: "in", label: "Your calendar & habits", sub: "the real numbers", x: 120, y: 460, w: 360, icon: "📅" },
      { id: "map", label: "A map of your campus", sub: "every walk routed", x: 620, y: 460, w: 360, icon: "🗺" },
      { id: "play", label: "Play tomorrow", sub: "before you live it", x: 1120, y: 260, w: 340, sunk: true, icon: "🎮" },
      { id: "hard", label: "Hard mode", sub: "a class moves: re-route", x: 1120, y: 660, w: 340, icon: "🌪" },
      { id: "stars", label: "Three stars", sub: "or the reasons you failed", x: 1580, y: 460, icon: "⭐" },
    ],
    edges: [
      { from: "in", to: "map" },
      { from: "map", to: "play" },
      { from: "map", to: "hard", dashed: true },
      { from: "play", to: "stars" },
      { from: "hard", to: "stars" },
    ],
  },
};
