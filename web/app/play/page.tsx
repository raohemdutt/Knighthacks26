"use client";

import { useEffect, useRef, useState } from "react";
import { TwinStage } from "@/components/map/TwinStage";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type Cell, type PlayRoute } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { humanize, percent } from "@/lib/format";
import { binPick } from "@/lib/guideBin";
import { SORT_ITEMS } from "@/lib/room";
import { type TwinPin, useRoomTwin } from "@/lib/roomTwin";
import { useFetch } from "@/lib/useFetch";
import { useVoice } from "@/lib/voice";

/** The items the box knows how to sort; anything it does not know goes to the trash. Shared with the Room map. */
const ITEMS = SORT_ITEMS;

/**
 * The route endpoint returns cells, not where the bins are. Asking for one item
 * per bin tells us where each bin's route ends, which is enough to mark both.
 */
const BIN_PROBES = ["can", "cup"];

/** Sort It shows no memory pins: the floor is for the route. */
const NO_PINS: TwinPin[] = [];

/** SVG units per grid cell. Kept large so label text is a normal font size (browsers clamp very small ones). */
const S = 10;

interface BinMark {
  bin: string;
  at: Cell;
}

export default function PlayPage() {
  const [item, setItem] = useState<string>(ITEMS[0] ?? "can");
  const route = useFetch((s) => api.playRoute(item, s), item);
  // "which bin for the cup?" asked anywhere: the look guide picks the item here, and the room lights up its bin
  useEffect(() => (binPick.get() ? setItem(binPick.get()!) : undefined, binPick.subscribe(setItem)), []);
  const probes = useFetch((s) => Promise.all(BIN_PROBES.map((p) => api.playRoute(p, s))));
  const accuracy = useFetch((s) => api.playAccuracy(s));
  // The room twin is the main picture when the box has a floor plan: the live round, or the way for the picked item.
  const twin = useRoomTwin();

  // This week's mistakes at the bins (the Impact screen's practice set), most first. Nothing when Waste Watch is off.
  const practice = useFetch((s) => api.impactPractice(undefined, s));
  const voice = useVoice();
  const top = practice.data?.items[0];
  const [practising, setPractising] = useState<string | null>(null);
  const [result, setResult] = useState<{ item: string; correct: boolean } | null>(null);
  const practisingRef = useRef(practising);
  practisingRef.current = practising;
  const rules = new Map((practice.data?.items ?? []).map((p) => [p.item, p.rule]));
  const rulesRef = useRef(rules);
  rulesRef.current = rules;

  const practise = (it: string) => {
    setPractising(it);
    setResult(null);
    setItem(it); // the same preview any picked item gets: the right bin and the way to it
  };

  // A round started on the box: follow along with its item. A round scored: refresh the numbers, and after a
  // round of the item being practised say why it goes where it goes.
  useAtlasEvents(["play.round.", "impact.practice.", "impact.item.disposed"], (e) => {
    if (e.type === "play.round.started" && typeof e.item === "string") setItem(e.item);
    if (e.type === "play.round.scored") {
      accuracy.reload();
      if (typeof e.item === "string" && e.item === practisingRef.current) {
        setResult({ item: e.item, correct: e.correct === true });
        const rule = rulesRef.current.get(e.item);
        if (rule) voice.say(rule, { tag: "sort-it-rule", replace: true });
      }
    }
    if (e.type.startsWith("impact.")) practice.reload();
  });

  const off = route.error?.notConfigured ?? false;

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Sort It</h1>
        <p className={ui.muted}>
          {twin.plan
            ? "Hold something up to the bin camera for two seconds, or pick it below. Atlas shows the right bin and the way there."
            : "Pick what you are holding. Atlas shows the right bin and the way there."}
        </p>
      </div>

      {twin.plan ? (
        <div className={ui.card} data-guide="play:twin">
          <TwinStage twin={twin} pins={NO_PINS} previewItem={item} speak start="top" shape="tall" />
        </div>
      ) : null}

      {top ? (
        <section className={ui.card} aria-label="Practise what you got wrong">
          <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
            <h2>Practise what you got wrong</h2>
            <span className={ui.tag}>
              {practice.data?.size} this week
            </span>
          </div>
          <p className={ui.breakWords}>
            <strong>{humanize(top.item)}</strong>
            {top.mistakes > 0
              ? `: ${top.mistakes === 1 ? "once" : `${top.mistakes} times`} in the wrong bin this week${top.went_to ? ` (it went in ${top.went_to})` : ""}.`
              : ": you asked to be taught this one."}
          </p>
          <div className={`${ui.row} ${ui.wrap}`}>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} aria-pressed={practising === top.item} onClick={() => practise(top.item)}>
              {practising === top.item ? `Practising: ${humanize(top.item)}` : `Practise: ${humanize(top.item)}`}
            </button>
            {(practice.data?.items ?? []).slice(1, 4).map((p) => (
              <button key={p.item} type="button" className={ui.btn} aria-pressed={practising === p.item} onClick={() => practise(p.item)}>
                {humanize(p.item)}
              </button>
            ))}
          </div>
          {practising && rules.get(practising) ? (
            <p className={ui.nudge} role="status">
              {result?.item === practising ? (result.correct ? "Right bin. " : "Not that bin. ") : ""}
              {rules.get(practising)}
            </p>
          ) : null}
          <p className={`${ui.muted} ${ui.small}`}>
            Pick one to see its bin and the way there on this screen. Hold it up to the bin camera to play the round.
          </p>
        </section>
      ) : null}

      {off ? (
        <Empty title="Sort It is not set up on this box">
          It needs a camera that can see the bins and the sorting rules. Once those are configured, the route to the right bin is drawn here.
        </Empty>
      ) : (
        <>
          <div className={ui.chips} role="group" aria-label="Item">
            {ITEMS.map((i) => (
              <button key={i} type="button" className={`${ui.chip} ${item === i ? ui.chipOn : ""}`} aria-pressed={item === i} onClick={() => setItem(i)}>
                {humanize(i)}
              </button>
            ))}
            {ITEMS.includes(item) ? null : (
              <button type="button" className={`${ui.chip} ${ui.chipOn}`} aria-pressed="true">
                {humanize(item)}
              </button>
            )}
          </div>

          {route.error ? (
            <ErrorBox what="the route" error={route.error} onRetry={route.reload} />
          ) : !route.data ? (
            <Loading what="the route" />
          ) : (
            <>
              {twin.plan ? <h2 className={ui.sectionTitle}>The bin camera&apos;s own view</h2> : null}
              <RouteCard route={route.data} bins={binMarks(route.data, probes.data ?? [])} />
            </>
          )}
        </>
      )}

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Are you getting better?</h2>
        {accuracy.error ? (
          accuracy.error.notConfigured ? (
            off ? null : <Empty title="No accuracy to show">Sort It is not set up on this box.</Empty>
          ) : (
            <ErrorBox what="your accuracy" error={accuracy.error} onRetry={accuracy.reload} />
          )
        ) : !accuracy.data ? (
          <Loading what="your accuracy" />
        ) : (
          <>
            <div className={ui.stats}>
              <div className={ui.stat}>
                <p className={ui.statNum}>{accuracy.data.first_round === null ? "–" : percent(accuracy.data.first_round)}</p>
                <p className={ui.statLabel}>right bin, round 1</p>
              </div>
              <div className={ui.stat}>
                <p className={ui.statNum}>{accuracy.data.round_5_on === null ? "–" : percent(accuracy.data.round_5_on)}</p>
                <p className={ui.statLabel}>right bin, round 5 onward</p>
              </div>
            </div>
            <p className={`${ui.muted} ${ui.small}`}>
              {accuracy.data.rounds === 0
                ? "No rounds played in this sitting yet. Face the camera with an item to start one; the comparison fills in after five rounds."
                : accuracy.data.first_round === null
                  ? `${accuracy.data.rounds} ${accuracy.data.rounds === 1 ? "round" : "rounds"} played in this sitting. The comparison fills in after five.`
                  : `${accuracy.data.rounds} rounds played in this sitting.`}
            </p>
          </>
        )}
      </section>
    </>
  );
}

/** Where each bin sits on the grid: the last cell of a route that ends in it. The selected item's own route wins. */
function binMarks(route: PlayRoute, probes: PlayRoute[]): BinMark[] {
  const marks = new Map<string, Cell>();
  for (const r of [...probes, route]) {
    const end = r.path[r.path.length - 1];
    if (end && r.rows === route.rows && r.cols === route.cols) marks.set(r.bin, end);
  }
  return [...marks].map(([bin, at]) => ({ bin, at }));
}

function RouteCard({ route, bins }: { route: PlayRoute; bins: BinMark[] }) {
  const { rows, cols, path } = route;
  const start = path[0];
  const points = path.map((c) => `${(c.col + 0.5) * S},${(c.row + 0.5) * S}`).join(" ");

  return (
    <div className={ui.card}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <h2>
          {humanize(route.item)} goes in <span style={{ color: "var(--accent-text)" }}>{route.bin}</span>
        </h2>
        {path.length > 1 ? <span className={ui.tag}>{path.length - 1} steps</span> : null}
      </div>

      {path.length === 0 ? (
        <Empty title="No route found">Something is blocking every way to the {route.bin} bin from where the player stands.</Empty>
      ) : (
        <>
          <svg className={ui.board} viewBox={`0 0 ${cols * S} ${rows * S}`} role="img" aria-label={`Camera view as a ${cols} by ${rows} grid, with the route to the ${route.bin} bin`}>
            {Array.from({ length: cols - 1 }, (_, i) => (
              <line key={`v${i}`} className={ui.gridLine} x1={(i + 1) * S} y1={0} x2={(i + 1) * S} y2={rows * S} />
            ))}
            {Array.from({ length: rows - 1 }, (_, i) => (
              <line key={`h${i}`} className={ui.gridLine} x1={0} y1={(i + 1) * S} x2={cols * S} y2={(i + 1) * S} />
            ))}

            {bins
              .filter((b) => b.bin !== route.bin)
              .map((b) => (
                <BinMarker key={b.bin} mark={b} target={false} cols={cols} />
              ))}

            {path.length > 1 ? <polyline className={ui.route} points={points} /> : null}

            {bins
              .filter((b) => b.bin === route.bin)
              .map((b) => (
                <BinMarker key={b.bin} mark={b} target cols={cols} />
              ))}

            {start ? (
              <>
                <circle className={ui.startMark} cx={(start.col + 0.5) * S} cy={(start.row + 0.5) * S} r={7} />
                <text className={ui.boardLabel} x={(start.col + 0.5) * S} y={(start.row - 0.7) * S} textAnchor="middle">
                  You
                </text>
              </>
            ) : null}
          </svg>
          <p className={ui.legend}>
            <span>
              <span className={ui.legendKey} />
              where you stand
            </span>
            <span>
              <span className={`${ui.legendKey} ${ui.legendAccent}`} />
              the right bin and the way there
            </span>
            {bins.some((b) => b.bin !== route.bin) ? (
              <span>
                <span className={`${ui.legendKey} ${ui.legendHollow}`} />
                the other bin
              </span>
            ) : null}
          </p>
          <p className={`${ui.muted} ${ui.small}`}>
            The grid is the bin camera&apos;s view, {cols} by {rows} cells. Each bin is marked where its route arrives.
          </p>
        </>
      )}
    </div>
  );
}

function BinMarker({ mark, target, cols }: { mark: BinMark; target: boolean; cols: number }) {
  const cx = (mark.at.col + 0.5) * S;
  const cy = (mark.at.row + 0.5) * S;
  // Keep the label inside the board when a bin sits against the left or right edge.
  const anchor = cx < 4 * S ? "start" : cx > (cols - 4) * S ? "end" : "middle";
  const tx = anchor === "start" ? cx - 12 : anchor === "end" ? cx + 12 : cx;
  return (
    <g>
      <rect className={target ? ui.binTarget : ui.binOther} x={cx - 12} y={cy - 12} width={24} height={24} rx={5} />
      <text className={`${ui.boardLabel} ${target ? "" : ui.boardLabelMuted}`} x={tx} y={cy + 27} textAnchor={anchor}>
        {humanize(mark.bin)}
      </text>
    </g>
  );
}
