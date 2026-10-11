"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { Fragment, useEffect, useState, type ReactNode } from "react";
import { DeviceStage } from "@/components/home/Device";
import css from "@/components/home/home.module.css";
import { CountUp, Reveal } from "@/components/home/motion";

/**
 * Home: the landing screen. The device pulls apart as the page scrolls, then what Atlas does (each tile with a small
 * picture of it working), the numbers that matter, how a question travels through the box, and a box to ask it
 * something. Everything on it is static copy except the question box, which hands its question to Ask Atlas.
 */

const TITLE = ["Your", "desk,"];
const TITLE_ACCENT = ["remembered."];

const TRUST: { label: string; icon: ReactNode }[] = [
  { label: "Runs on a Jetson Orin Nano", icon: <path d="M5 7h14v10H5zM9 4v3M15 4v3M9 17v3M15 17v3M2 10h3M2 14h3M19 10h3M19 14h3" /> },
  { label: "0 bytes of video out", icon: <path d="M4 7h11v10H4zM15 10l5-3v10l-5-3M3 3l18 18" /> },
  { label: "Every action needs your tap", icon: <path d="M12 3l7.5 3v5.5c0 4.5-3.2 8-7.5 9.5-4.3-1.5-7.5-5-7.5-9.5V6zM8.5 12l2.5 2.5 4.5-5" /> },
];

const ARROW = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M5 12h14M13 6l6 6-6 6" />
  </svg>
);

// ---------- the tiles' pictures: small, static scenes of each feature at work (decorative, hidden from readers) ----------

function SeesArt() {
  const boxes = [
    { label: "laptop", score: "0.97", x: 30, y: 30, w: 38, h: 42 },
    { label: "mug", score: "0.94", x: 73, y: 44, w: 14, h: 26 },
    { label: "keys", score: "0.91", x: 12, y: 66, w: 13, h: 14 },
  ];
  return (
    <div className={css.cam}>
      <div className={css.camDesk} />
      <div className={css.camLaptop} />
      <div className={css.camMug} />
      <div className={css.camKeys} />
      {boxes.map((b, i) => (
        <span key={b.label} className={css.camBox} style={{ left: `${b.x}%`, top: `${b.y}%`, width: `${b.w}%`, height: `${b.h}%`, "--i": i } as React.CSSProperties}>
          <b>
            {b.label} <em>{b.score}</em>
          </b>
        </span>
      ))}
      <span className={css.camFace}>
        <b>face blurred</b>
      </span>
      <span className={css.camScan} />
      <span className={css.camBadge}>
        <i />
        desk · live
      </span>
      <span className={css.camZone}>zone: desk</span>
    </div>
  );
}

function RemembersArt() {
  return (
    <div className={css.memory}>
      <div className={css.memoryAsk}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <circle cx="11" cy="11" r="6" />
          <path d="M20 20l-4.5-4.5" />
        </svg>
        where are my keys?
      </div>
      <div className={css.memoryHit}>
        <span className={css.memoryThumb}>
          <i />
        </span>
        <span className={css.memoryText}>
          <b>On the desk, left of the laptop</b>
          <span>8:42 PM · desk camera · with the frame</span>
        </span>
      </div>
    </div>
  );
}

function AsksArt() {
  return (
    <div className={css.ask}>
      <div className={css.askCard}>
        <span className={css.askHead}>
          <i />
          Atlas wants to send a message
        </span>
        <span className={css.askBody}>To Sam: &ldquo;Running ten minutes late, start without me.&rdquo;</span>
        <span className={css.askButtons}>
          <span className={css.askYes}>Approve</span>
          <span className={css.askNo}>Not now</span>
        </span>
      </div>
    </div>
  );
}

function PlansArt() {
  // Two lanes, so each block is wide enough to read.
  const blocks = [
    { at: "10:30", what: "Lecture", where: "leave by 10:12", x: 4, w: 40, lane: 0, tone: "a" },
    { at: "1:00", what: "Lab", where: "Engineering II", x: 38, w: 32, lane: 1, tone: "b" },
    { at: "4:00", what: "Essay due", where: "on Canvas", x: 66, w: 31, lane: 0, tone: "c" },
  ];
  return (
    <div className={css.plan}>
      <div className={css.planHours}>
        {["9", "11", "1", "3", "5"].map((h) => (
          <span key={h}>{h}</span>
        ))}
      </div>
      <div className={css.planTrack}>
        {blocks.map((b, i) => (
          <span key={b.what} className={css.planBlock} data-tone={b.tone} data-lane={b.lane} style={{ left: `${b.x}%`, width: `${b.w}%`, "--i": i } as React.CSSProperties}>
            <b>
              {b.at} {b.what}
            </b>
            <span>{b.where}</span>
          </span>
        ))}
        <span className={css.planNow}>
          <i>now</i>
        </span>
      </div>
    </div>
  );
}

function LearnsArt() {
  const bars = [
    { label: "Before", w: 48 },
    { label: "After", w: 82, hot: true },
    { label: "Teacher", w: 90 },
  ];
  return (
    <div className={css.learn}>
      {bars.map((b, i) => (
        <span key={b.label} className={css.learnRow}>
          <span className={css.learnLabel}>{b.label}</span>
          <span className={css.learnTrack}>
            <span className={css.learnFill} data-hot={b.hot ? "true" : undefined} style={{ "--w": `${b.w}%`, "--i": i } as React.CSSProperties} />
          </span>
        </span>
      ))}
      <span className={css.learnNote}>A new skill ships only if the after beats the before.</span>
    </div>
  );
}

function SecretsArt() {
  const rows = [
    { host: "calendar", amount: "14 KB", w: 34 },
    { host: "canvas", amount: "9 KB", w: 22 },
    { host: "video", amount: "0 B", w: 0, locked: true },
  ];
  return (
    <div className={css.egress}>
      {rows.map((r, i) => (
        <span key={r.host} className={css.egressRow} data-locked={r.locked ? "true" : undefined}>
          <span className={css.egressHost}>{r.host}</span>
          <span className={css.egressTrack}>
            <span className={css.egressFill} style={{ "--w": `${r.w}%`, "--i": i } as React.CSSProperties} />
          </span>
          <span className={css.egressAmount}>
            {r.locked ? (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="5" y="11" width="14" height="9" rx="2" />
                <path d="M8 11V8a4 4 0 0 1 8 0v3" />
              </svg>
            ) : null}
            {r.amount}
          </span>
        </span>
      ))}
    </div>
  );
}

const FEATURES: { title: string; text: string; art: ReactNode; size: "wide" | "narrow" | "half" }[] = [
  {
    title: "It sees",
    text: "Two cameras, one Jetson. Objects, people and zones are found on the desk itself, and faces are blurred before a single frame is kept.",
    art: <SeesArt />,
    size: "wide",
  },
  {
    title: "It remembers",
    text: "Every episode is kept as plain words and keyframes you can search. Ask where your keys are and get the frame that proves it.",
    art: <RemembersArt />,
    size: "narrow",
  },
  {
    title: "It asks first",
    text: "A message, a calendar write or an export never happens without a tap on your phone. Say no out loud and it stops.",
    art: <AsksArt />,
    size: "narrow",
  },
  {
    title: "It plans the day",
    text: "Canvas, your calendar and the campus map become one plan, with a leave-by time for every walk and a nudge when it slips.",
    art: <PlansArt />,
    size: "wide",
  },
  {
    title: "It learns",
    text: "The lab teaches the on-device model from Atlas's own data. New skills arrive as a package, with the score before and after.",
    art: <LearnsArt />,
    size: "half",
  },
  {
    title: "It keeps secrets",
    text: "An egress firewall lets through only the sources you connected. Video never leaves the box. Not for training, not for anyone.",
    art: <SecretsArt />,
    size: "half",
  },
];

const NUMBERS: { value: number; suffix?: string; label: string }[] = [
  { value: 2, label: "cameras, watching the desk and the door" },
  { value: 8, suffix: " GB", label: "on the desk: the whole model and memory fit on the Jetson" },
  { value: 0, suffix: " bytes", label: "of video ever leave the box" },
  { value: 1, suffix: " tap", label: "to approve anything Atlas wants to do" },
];

const FLOW: { label: string; sub: string; icon: ReactNode }[] = [
  { label: "See", sub: "two cameras", icon: <path d="M3 7h13v11H3zM16 10.5l5-3v10l-5-3" /> },
  { label: "Detect", sub: "on the Jetson", icon: <path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4M9 12a3 3 0 1 0 6 0a3 3 0 1 0 -6 0" /> },
  { label: "Remember", sub: "episodes and frames", icon: <path d="M5 6a7 2.5 0 1 0 14 0a7 2.5 0 1 0 -14 0M5 6v12a7 2.5 0 0 0 14 0V6M5 12a7 2.5 0 0 0 14 0" /> },
  { label: "Ask", sub: "in plain words", icon: <path d="M4 5h16v11h-9l-5 4v-4H4z" /> },
  { label: "Answer", sub: "with the proof", icon: <path d="M4 5h16v14H4zM8 12l3 3 5-6" /> },
  { label: "Act", sub: "after your yes", icon: <path d="M12 3l7.5 3v5.5c0 4.5-3.2 8-7.5 9.5-4.3-1.5-7.5-5-7.5-9.5V6zM8.5 12l2.5 2.5 4.5-5" /> },
];

const QUESTIONS = ["Where are my keys?", "What is due this week?", "What happened at the desk overnight?", "When do I need to leave for class?"];

/** A question box that hands its question to Ask Atlas (/?q=…), which asks it at once. The placeholder cycles. */
function AskBox() {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [hint, setHint] = useState(0);
  useEffect(() => {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const id = window.setInterval(() => setHint((i) => (i + 1) % QUESTIONS.length), 3200);
    return () => window.clearInterval(id);
  }, []);
  const go = (text: string) => {
    const t = text.trim();
    router.push(t ? `/?q=${encodeURIComponent(t)}` : "/");
  };
  return (
    <div className={css.askBox}>
      <form
        className={css.askForm}
        onSubmit={(e) => {
          e.preventDefault();
          go(q || QUESTIONS[hint] || "");
        }}
      >
        <input className={css.askInput} value={q} onChange={(e) => setQ(e.target.value)} placeholder={QUESTIONS[hint]} aria-label="Ask Atlas something" enterKeyHint="send" />
        <button type="submit" className={`${css.cta} ${css.ctaPrimary} ${css.askSend}`}>
          Ask {ARROW}
        </button>
      </form>
      <div className={css.askChips}>
        {QUESTIONS.map((text) => (
          <button key={text} type="button" className={css.askChip} onClick={() => go(text)}>
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}

/** The tile under the pointer keeps a light where the pointer is. */
function spotlight(e: React.PointerEvent<HTMLElement>) {
  if (e.pointerType !== "mouse") return;
  const tile = (e.target as HTMLElement).closest<HTMLElement>("[data-tile]");
  if (!tile) return;
  const r = tile.getBoundingClientRect();
  tile.style.setProperty("--mx", `${e.clientX - r.left}px`);
  tile.style.setProperty("--my", `${e.clientY - r.top}px`);
}

export default function HomePage() {
  let w = 0;
  return (
    <div className={css.home}>
      <DeviceStage>
        <span className={css.eyebrow}>
          <span className={css.eyebrowDot} aria-hidden="true" />
          Atlas · on-device, always on
        </span>
        <h1 className={css.title}>
          {TITLE.map((word, i) => (
            <Fragment key={word}>
              {i ? " " : null}
              <span className={css.word} style={{ "--i": w++ } as React.CSSProperties}>
                <span>{word}</span>
              </span>
            </Fragment>
          ))}
          <br />
          {TITLE_ACCENT.map((word) => (
            <span key={word} className={`${css.word} ${css.accentWord}`} style={{ "--i": w++ } as React.CSSProperties}>
              <span>{word}</span>
            </span>
          ))}
        </h1>
        <p className={css.lead}>A screenless box that watches your space, remembers what matters, and acts only when you say yes. Nothing leaves the room.</p>
        <div className={css.ctas}>
          <Link href="/" className={`${css.cta} ${css.ctaPrimary}`}>
            Ask Atlas {ARROW}
          </Link>
          <Link href="/map/?view=3d" className={css.cta}>
            Walk the room {ARROW}
          </Link>
        </div>
        <ul className={css.trust}>
          {TRUST.map((item) => (
            <li key={item.label}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                {item.icon}
              </svg>
              {item.label}
            </li>
          ))}
        </ul>
      </DeviceStage>

      <section className={css.section} aria-labelledby="what">
        <Reveal className={css.sectionHead}>
          <p className={css.kicker}>What it does</p>
          <h2 id="what" className={css.h2}>
            Eyes on the desk. A memory that answers back.
          </h2>
          <p className={css.sub}>Atlas lives on the table, not in a data centre. Everything below happens on the box in front of you.</p>
        </Reveal>
        <ul className={css.bento} onPointerMove={spotlight}>
          {FEATURES.map((f, i) => (
            <Reveal key={f.title} as="li" className={`${css.tile} ${css[`tile_${f.size}`] ?? ""}`} index={i % 2}>
              <div className={css.tileInner} data-tile="">
                <div className={css.tileArt} aria-hidden="true">
                  {f.art}
                </div>
                <div className={css.tileCopy}>
                  <h3 className={css.tileTitle}>{f.title}</h3>
                  <p className={css.tileText}>{f.text}</p>
                </div>
              </div>
            </Reveal>
          ))}
        </ul>
      </section>

      <section className={css.section} aria-label="By the numbers">
        <Reveal className={css.numbers}>
          {NUMBERS.map((n) => (
            <div key={n.label} className={css.number}>
              <span className={css.numberValue}>
                <CountUp to={n.value} suffix={n.suffix} />
              </span>
              <span className={css.numberLabel}>{n.label}</span>
            </div>
          ))}
        </Reveal>
      </section>

      <section className={css.section} aria-labelledby="flow">
        <Reveal className={css.sectionHead}>
          <p className={css.kicker}>How a question travels</p>
          <h2 id="flow" className={css.h2}>
            From the camera to your phone, without leaving the desk.
          </h2>
        </Reveal>
        <Reveal className={css.rail} index={1}>
          <ol className={css.railSteps} style={{ "--n": FLOW.length } as React.CSSProperties}>
            {FLOW.map((s, i) => (
              <li key={s.label} className={css.railStep} style={{ "--i": i } as React.CSSProperties}>
                <span className={css.railNode}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    {s.icon}
                  </svg>
                </span>
                <span className={css.railText}>
                  <b>{s.label}</b>
                  <span>{s.sub}</span>
                </span>
              </li>
            ))}
          </ol>
        </Reveal>
      </section>

      <section className={css.section} aria-labelledby="go">
        <Reveal className={css.closing}>
          <p className={css.kicker}>Try it</p>
          <h2 id="go" className={css.closingTitle}>
            Ask it something.
          </h2>
          <p className={css.sub}>The box answers from what it saw, with the frame to prove it.</p>
          <AskBox />
          <Link href="/now" className={css.closingLink}>
            Or see today {ARROW}
          </Link>
        </Reveal>
      </section>

      <p className={css.foot}>Atlas · Knight Hacks 2026 · built on an NVIDIA Jetson Orin Nano</p>
    </div>
  );
}
