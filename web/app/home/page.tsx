"use client";

import Link from "next/link";
import { Fragment, type ReactNode } from "react";
import { DeviceStage } from "@/components/home/Device";
import css from "@/components/home/home.module.css";
import { CountUp, Reveal } from "@/components/home/motion";

/**
 * Home: the landing screen. The device pulls apart as the page scrolls, then what Atlas does, the numbers that
 * matter, how a question travels through the box, and a way in. Everything on it is static copy; the data screens
 * are one tap away.
 */

const TITLE = ["Your", "desk,"];
const TITLE_ACCENT = ["remembered."];

const FEATURES: { title: string; text: string; icon: ReactNode }[] = [
  {
    title: "It sees",
    text: "Two cameras, one Jetson. Objects, people and zones are found on the desk itself, and faces are blurred before a single frame is kept.",
    icon: (
      <>
        <rect x="3" y="6" width="18" height="13" rx="2.5" />
        <circle cx="12" cy="12.5" r="3.4" />
        <path d="M8 6l1.4-2h5.2L16 6" />
      </>
    ),
  },
  {
    title: "It remembers",
    text: "Every episode is kept as plain words and keyframes you can search. Ask where your keys are and get the frame that proves it.",
    icon: (
      <>
        <ellipse cx="12" cy="6" rx="7" ry="3" />
        <path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6" />
        <path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3" />
      </>
    ),
  },
  {
    title: "It plans the day",
    text: "Canvas, your calendar and the campus map become one plan, with a leave-by time for every walk and a nudge when it slips.",
    icon: (
      <>
        <rect x="3.5" y="5" width="17" height="15" rx="2.5" />
        <path d="M3.5 10h17M8 3v4M16 3v4" />
        <path d="M8 14.5l2.5 2.5 5-5" />
      </>
    ),
  },
  {
    title: "It asks first",
    text: "A message, a calendar write or an export never happens without a tap on your phone. Say no out loud and it stops.",
    icon: (
      <>
        <path d="M12 3l7.5 3v5.5c0 4.5-3.2 8-7.5 9.5-4.3-1.5-7.5-5-7.5-9.5V6z" />
        <path d="M8.5 12l2.5 2.5 4.5-5" />
      </>
    ),
  },
  {
    title: "It learns",
    text: "The lab teaches the on-device model from Atlas's own data. New skills arrive as a package, with the score before and after.",
    icon: (
      <>
        <path d="M3 9l9-4.5L21 9l-9 4.5z" />
        <path d="M7 11.2V15c0 1.4 2.2 2.8 5 2.8s5-1.4 5-2.8v-3.8M21 9v5" />
      </>
    ),
  },
  {
    title: "It keeps secrets",
    text: "An egress firewall lets through only the sources you connected. Video never leaves the box. Not for training, not for anyone.",
    icon: (
      <>
        <rect x="5" y="10" width="14" height="10" rx="2.5" />
        <path d="M8 10V7a4 4 0 0 1 8 0v3" />
        <circle cx="12" cy="15" r="1.4" />
      </>
    ),
  },
];

const NUMBERS: { value: number; suffix?: string; label: string }[] = [
  { value: 2, label: "cameras, watching the desk and the door" },
  { value: 8, suffix: " GB", label: "on the desk: the whole model and memory fit on the Jetson" },
  { value: 0, suffix: " bytes", label: "of video ever leave the box" },
  { value: 1, suffix: " tap", label: "to approve anything Atlas wants to do" },
];

const FLOW_PATH = "M100 100 C 160 100, 190 60, 250 60 S 340 140, 400 140 S 490 60, 550 60 S 640 140, 700 140 S 790 100, 900 100";

const FLOW: { label: string; sub: string; x: number; y: number; icon: ReactNode }[] = [
  { label: "See", sub: "two cameras", x: 100, y: 100, icon: <path d="M-8 -3 h16 v10 h-16 z M-5 -3 l1.5 -2 h7 l1.5 2 M0 5.5 m-3 0 a3 3 0 1 0 6 0 a3 3 0 1 0 -6 0" /> },
  { label: "Detect", sub: "on the Jetson", x: 250, y: 60, icon: <path d="M-8 -5 v-3 h3 M5 -8 h3 v3 M8 5 v3 h-3 M-5 8 h-3 v-3 M-3 0 a3 3 0 1 0 6 0 a3 3 0 1 0 -6 0" /> },
  { label: "Remember", sub: "episodes and frames", x: 400, y: 140, icon: <path d="M-7 -5 a7 2.5 0 1 0 14 0 a7 2.5 0 1 0 -14 0 M-7 -5 v10 a7 2.5 0 0 0 14 0 v-10 M-7 0 a7 2.5 0 0 0 14 0" /> },
  { label: "Ask", sub: "in plain words", x: 550, y: 60, icon: <path d="M-8 -6 h16 v10 h-9 l-4 3 v-3 h-3 z" /> },
  { label: "Answer", sub: "with the proof", x: 700, y: 140, icon: <path d="M-8 -6 h16 v12 h-16 z M-5 1 l3 3 l5 -6" /> },
  { label: "Act", sub: "after your yes", x: 900, y: 100, icon: <path d="M0 -8 l7 3 v5 c0 4 -3 7 -7 8.5 c-4 -1.5 -7 -4.5 -7 -8.5 v-5 z M-3 0.5 l2.5 2.5 l4.5 -5" /> },
];

const ARROW = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M5 12h14M13 6l6 6-6 6" />
  </svg>
);

export default function HomePage() {
  let w = 0;
  return (
    <div className={css.home}>
      <DeviceStage>
        <div className={css.hero}>
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
        </div>
      </DeviceStage>

      <section className={css.section} aria-labelledby="what">
        <Reveal className={css.sectionHead}>
          <p className={css.kicker}>What it does</p>
          <h2 id="what" className={css.h2}>
            Eyes on the desk. A memory that answers back.
          </h2>
          <p className={css.sub}>Atlas lives on the table, not in a data centre. Everything below happens on the box in front of you.</p>
        </Reveal>
        <ul className={css.grid}>
          {FEATURES.map((f, i) => (
            <Reveal key={f.title} as="li" className={css.feature} index={i}>
              <span className={css.icon} aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  {f.icon}
                </svg>
              </span>
              <h3 className={css.featureTitle}>{f.title}</h3>
              <p className={css.featureText}>{f.text}</p>
            </Reveal>
          ))}
        </ul>
      </section>

      <section className={css.section} aria-label="By the numbers">
        <div className={css.numbers}>
          {NUMBERS.map((n, i) => (
            <Reveal key={n.label} className={css.number} index={i}>
              <span className={css.numberValue}>
                <CountUp to={n.value} suffix={n.suffix} />
              </span>
              <span className={css.numberLabel}>{n.label}</span>
            </Reveal>
          ))}
        </div>
      </section>

      <section className={css.section} aria-labelledby="flow">
        <Reveal className={css.sectionHead}>
          <p className={css.kicker}>How a question travels</p>
          <h2 id="flow" className={css.h2}>
            From the camera to your phone, without leaving the desk.
          </h2>
        </Reveal>
        <Reveal className={css.flow} index={1}>
          <svg className={css.flowSvg} viewBox="0 0 1000 220" role="img" aria-label="See, detect, remember, ask, answer, act">
            <path className={css.flowPath} d={FLOW_PATH} />
            <path className={css.flowGlow} d={FLOW_PATH} pathLength={1} />
            {FLOW.map((n) => (
              <g key={n.label} transform={`translate(${n.x} ${n.y})`}>
                <circle className={css.flowNode} r="26" />
                <g className={css.flowNodeIcon}>{n.icon}</g>
                <text className={css.flowLabel} y="52">
                  {n.label}
                </text>
                <text className={css.flowSubLabel} y="72">
                  {n.sub}
                </text>
              </g>
            ))}
            <circle className={css.flowDot} r="6">
              <animateMotion dur="7s" repeatCount="indefinite" path={FLOW_PATH} keyPoints="0;1" keyTimes="0;1" calcMode="spline" keySplines="0.45 0 0.55 1" />
            </circle>
          </svg>
        </Reveal>
      </section>

      <section className={css.section} aria-labelledby="go">
        <Reveal className={css.closing}>
          <p className={css.kicker}>Try it</p>
          <h2 id="go" className={css.closingTitle}>
            Ask it something.
          </h2>
          <p className={css.sub}>Where are my keys. What is due this week. What happened at the table overnight. The box answers from what it saw, with the frame to prove it.</p>
          <div className={css.ctas}>
            <Link href="/" className={`${css.cta} ${css.ctaPrimary}`}>
              Ask Atlas {ARROW}
            </Link>
            <Link href="/now" className={css.cta}>
              See today {ARROW}
            </Link>
          </div>
        </Reveal>
      </section>

      <p className={css.foot}>Atlas · Knight Hacks 2026 · built on an NVIDIA Jetson Orin Nano</p>
    </div>
  );
}
