"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { KIND, OTHER_KIND } from "@/components/calendar/DayBar";
import type { DayBlock, Explained, Fix, Move, MoveOption, NumberSource, OpenSlot, PlanBoard, PlannedBlock, SimResult, Suggestion } from "@/lib/api";
import { ms } from "@/lib/dayrun";
import { blockTitle, clockAt, dur, freeTime, insertAt, isoAt, moveTo, placeShort, rowsOf, t12, type BlockRow, type GapRow, type Row } from "@/lib/plan";
import { Sheet } from "../Portal";
import ui from "../ui.module.css";
import pl from "./plan.module.css";
import { SourceChip } from "./PlanDetails";
import { RouteSheet } from "./RouteSheet";

/**
 * Your plan, as an agenda in the Calendar screen's row style: the time, a dot, and a card with a title and one
 * line ("Student Union · 30 min · 7 min walk"). Open gaps offer "+ Add"; a "Near you" event that would fall in a
 * gap is offered there, dashed, and is never a commitment until it is added.
 *
 * Tap a block to select it: the map lights its walk, and the block opens its controls (length, earlier, later,
 * remove). Every change is handed up as a new list of moves, and the page runs it at once.
 */

const STEP = 15;
const DEFAULT_MINUTES: Record<string, number | null> = { study: 90, eat: 30, rest: 30, sleep: null, event: null };

interface Props {
  board: PlanBoard;
  result: SimResult;
  moves: Move[];
  /** The word this box uses for the main activity ("Study", "Work"). */
  activity: string;
  /** A past day, as the log has it: shown read-only, with what the review measured. */
  past?: DayBlock[];
  readOnly?: boolean;
  selected: string | null;
  onSelect: (key: string | null) => void;
  onHover: (key: string | null) => void;
  /** The move an edit just touched: scrolled to and pulsed once. */
  flash: { index: number; n: number } | null;
  /** Moves the day's disruptions changed (hard mode). */
  moved: Set<number>;
  fixes: Fix[];
  explained: Explained | undefined;
  onChange: (moves: Move[], changed: number | null) => void;
}

export const rowKey = (b: Pick<PlannedBlock, "id" | "start">) => b.id + b.start;

export function PlanAgenda({ board, result, moves, activity, past, readOnly = false, selected, onSelect, onHover, flash, moved, fixes, explained, onChange }: Props) {
  const [adding, setAdding] = useState<GapRow | null>(null);
  const [openRuns, setOpenRuns] = useState<Set<string>>(new Set());
  const [why, setWhy] = useState<string | null>(null);
  const list = useRef<HTMLUListElement>(null);

  const rows: Row[] = useMemo(() => {
    if (!past) return rowsOf(board, result, moves);
    // a past day: the blocks as they were planned, matched to this run of them for the map; and what did not fit
    const stuck: Row[] = result.rejected.map((r) => ({ type: "stuck", key: `stuck-${r.index}`, index: r.index, move: r.move, slot: board.slots.find((s) => s.id === r.move.slot), reason: r.reason }));
    const planned = past.map((b): BlockRow => {
      const sim = result.blocks.find((x) => x.title === b.title && x.start === b.start) ?? result.blocks.find((x) => x.title === b.title && Math.abs(ms(x.start) - ms(b.start)) < 20 * 60_000);
      const block: PlannedBlock = sim ?? { id: b.id, kind: b.kind, title: b.title, start: b.start, end: b.end, minutes: b.minutes, place: b.place ?? "", place_name: b.place_name ?? "", fixed: b.fixed, move: null, with_friend: b.with_friend };
      const walk = sim ? result.routes.find((r) => (sim.move !== null ? r.move === sim.move : r.move === null && r.end === sim.place && Math.abs(ms(r.arrive) - ms(sim.start)) <= 30 * 60_000)) : undefined;
      return { type: "block", key: rowKey(block), block: { ...block, start: b.start, end: b.end, minutes: b.minutes }, walk };
    });
    const when = (r: Row) => (r.type === "block" ? ms(r.block.start) : r.type === "stuck" && r.slot ? ms(r.slot.start) - 1 : 0);
    return [...planned, ...stuck].sort((x, y) => when(x) - when(y));
  }, [board, result, moves, past]);
  const marks = useMemo(() => new Map((past ?? []).map((b) => [b.title + b.start, b])), [past]);

  // scroll to, and pulse, the row an edit just changed
  const flashKey = flash ? rows.find((r) => r.type === "block" && r.block.move === flash.index)?.key : undefined;
  useEffect(() => {
    if (!flash || !flashKey) return;
    const el = list.current?.querySelector<HTMLElement>(`[data-row="${CSS.escape(flashKey)}"]`);
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    el?.scrollIntoView({ block: "center", behavior: calm ? "auto" : "smooth" });
    // only when a new edit lands, not when the rows are rebuilt
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flash?.n, flashKey]);

  // long runs of work and short rests in one place fold into one row
  const groups = useMemo(() => {
    const out: (Row | { type: "run"; key: string; rows: BlockRow[] })[] = [];
    let run: BlockRow[] = [];
    const flush = () => {
      if (run.length >= 3) out.push({ type: "run", key: `run-${run[0]!.key}`, rows: run });
      else out.push(...run);
      run = [];
    };
    for (const r of rows) {
      const fits = r.type === "block" && !r.block.fixed && (r.block.kind === "study" || r.block.kind === "rest") && (run.length === 0 || run[0]!.block.place === r.block.place);
      if (fits) run.push(r as BlockRow);
      else {
        flush();
        if (r.type === "block" && !r.block.fixed && (r.block.kind === "study" || r.block.kind === "rest")) run.push(r);
        else out.push(r);
      }
    }
    flush();
    return out;
  }, [rows]);

  const fixFor = (index: number) => fixes.find((f) => f.id === `move:${index}`);
  const change = (next: Move[], changed: number | null) => onChange(next, changed);

  const blockRow = (r: BlockRow): ReactNode => {
    const b = r.block;
    const k = KIND[b.kind] ?? OTHER_KIND;
    const isOn = selected === r.key;
    const move = b.move !== null ? moves[b.move] : undefined;
    const mark = marks.get(b.title + b.start);
    const walkMin = r.walk ? Math.max(1, Math.round(r.walk.minutes)) : 0;
    // a calendar block with no place the map knows happens "wherever you are": no place is claimed for it
    const nowhere = b.fixed && board.blocks.some((x) => x.id === b.id && x.place === null);
    const meta = [nowhere ? null : placeShort(b.place_name), dur(b.minutes), walkMin > 0 ? `${walkMin} min walk` : null].filter(Boolean).join(" · ");
    const isMoved = mark?.moved || (b.move !== null && moved.has(b.move));
    const tags = [
      mark?.mark === "done" ? <span key="m" className={`${pl.tagPill} ${pl.tagGood}`}>Happened</span> : null,
      mark?.mark === "missed" ? <span key="m" className={`${pl.tagPill} ${pl.tagWarn}`}>Missed</span> : null,
      isMoved ? <span key="v" className={pl.tagPill}>Moved</span> : null,
    ].filter(Boolean);
    const editable = !readOnly && move !== undefined && b.move !== null;
    const length = move?.minutes ?? b.minutes;
    const shift = (dir: 1 | -1) => {
      if (b.move === null) return;
      const want = ms(b.start) + dir * 30 * 60_000;
      const open = board.slots.some((s) => ms(s.start) <= want && want < ms(s.end));
      const t = open ? want : freeTime(board, ms(b.start), dir);
      const out = t === null ? null : moveTo(moves, b.move, t, board, result);
      if (out) change(out.moves, out.index);
    };
    return (
      <li key={r.key} className={pl.row} data-row={r.key} data-guide={`plan:row:${b.id}`}>
        <div className={pl.rowTime}>
          <b>{t12(b.start)}</b>
          <span>{t12(b.end)}</span>
        </div>
        <div className={pl.rowRail} aria-hidden="true">
          <i className={`${pl.rowDot} ${b.fixed ? pl.rowDotFixed : ""}`} style={{ "--c": k.color } as CSSProperties} />
        </div>
        <div className={pl.rowCell}>
          <button
            type="button"
            className={`${pl.rowCard} ${pl.rowBlock} ${isOn ? pl.rowOn : ""} ${flash && b.move === flash.index ? pl.rowFlash : ""}`}
            key={flash && b.move === flash.index ? `f${flash.n}` : "c"}
            style={{ "--c": k.color } as CSSProperties}
            aria-expanded={isOn}
            onClick={() => onSelect(isOn ? null : r.key)}
            onPointerEnter={(e) => e.pointerType === "mouse" && onHover(r.key)}
            onPointerLeave={(e) => e.pointerType === "mouse" && onHover(null)}
          >
            <span className={pl.rowTop}>
              <span className={pl.rowTitle}>{blockTitle(b, activity)}</span>
              {tags.length > 0 ? <span className={pl.rowTags}>{tags}</span> : null}
            </span>
            <span className={pl.rowMeta}>{meta}</span>
            {isOn ? (
              <span className={pl.rowReason}>
                {r.walk && walkMin > 0 ? `${walkMin} min walk · leave by ${t12(r.walk.depart)}` : "No walk: you are already there."}
                {b.fixed && !past ? " On your calendar, so Atlas plans around it." : ""}
              </span>
            ) : null}
          </button>
          {isOn ? (
            <div className={pl.edit}>
              {editable ? (
                <>
                  <span className={pl.stepper} role="group" aria-label="Length">
                    <button type="button" aria-label="15 minutes shorter" disabled={length <= STEP} onClick={() => change(moves.map((m, i) => (i === b.move ? { ...m, minutes: Math.max(STEP, length - STEP) } : m)), b.move)}>
                      −
                    </button>
                    <span>{dur(length)}</span>
                    <button type="button" aria-label="15 minutes longer" onClick={() => change(moves.map((m, i) => (i === b.move ? { ...m, minutes: length + STEP } : m)), b.move)}>
                      +
                    </button>
                  </span>
                  <button type="button" className={pl.miniBtn} onClick={() => shift(-1)}>
                    Earlier
                  </button>
                  <button type="button" className={pl.miniBtn} onClick={() => shift(1)}>
                    Later
                  </button>
                  <button type="button" className={`${pl.miniBtn} ${pl.miniDanger}`} onClick={() => (onSelect(null), change(moves.filter((_, i) => i !== b.move), null))}>
                    Remove
                  </button>
                </>
              ) : null}
              {r.walk && walkMin > 0 ? (
                <>
                  <RouteSheet from={r.walk.start} to={r.walk.end} arriveBy={b.start} now={board.start} what={blockTitle(b, activity)} walkMinutes={walkMin} />
                  {explained?.walks[String(r.walk.index)] ? (
                    explained.sources?.walks[String(r.walk.index)] ? (
                      <span className={pl.srcFor}>
                        {walkMin} min walk
                        <SourceChip source={explained.sources.walks[String(r.walk.index)] as NumberSource} about={`${walkMin} min walk`} open={why === r.key} onToggle={() => setWhy(why === r.key ? null : r.key)} />
                      </span>
                    ) : (
                      <button type="button" className={pl.whyBtn} aria-expanded={why === r.key} onClick={() => setWhy(why === r.key ? null : r.key)}>
                        Why {walkMin} min?
                      </button>
                    )
                  ) : null}
                </>
              ) : null}
            </div>
          ) : null}
          {isOn && why === r.key && r.walk ? <p className={pl.srcNote}>{explained?.walks[String(r.walk.index)]}</p> : null}
        </div>
      </li>
    );
  };

  return (
    <>
      <ul className={pl.rows} ref={list}>
        {groups.map((g) => {
          if (g.type === "run") {
            const first = g.rows[0]!.block;
            const last = g.rows[g.rows.length - 1]!.block;
            const inside = g.rows.some((r) => r.key === selected || (flash && r.block.move === flash.index));
            const open = openRuns.has(g.key) || inside;
            if (open) return g.rows.map(blockRow);
            const work = g.rows.filter((r) => r.block.kind === "study");
            const k = KIND.study ?? OTHER_KIND;
            return (
              <li key={g.key} className={pl.row}>
                <div className={pl.rowTime}>
                  <b>{t12(first.start)}</b>
                  <span>{t12(last.end)}</span>
                </div>
                <div className={pl.rowRail} aria-hidden="true">
                  <i className={pl.rowDot} style={{ "--c": k.color } as CSSProperties} />
                </div>
                <div className={pl.rowCell}>
                  <button type="button" className={`${pl.rowCard} ${pl.rowBlock}`} style={{ "--c": k.color } as CSSProperties} aria-expanded={false} onClick={() => setOpenRuns(new Set([...openRuns, g.key]))}>
                    <span className={pl.rowTop}>
                      <span className={pl.rowTitle}>{activity} with breaks</span>
                      <span className={pl.rowTags}>
                        <span className={pl.tagPill}>{g.rows.length} blocks +</span>
                      </span>
                    </span>
                    <span className={pl.rowMeta}>
                      {placeShort(first.place_name)} · {dur(work.reduce((n, r) => n + r.block.minutes, 0))} of {activity.toLowerCase()}
                    </span>
                  </button>
                </div>
              </li>
            );
          }
          if (g.type === "block") return blockRow(g);
          if (g.type === "stuck") {
            const fix = fixFor(g.index);
            const option = g.slot?.moves.find((o) => o.kind === g.move.kind && (o.place ?? null) === (g.move.place ?? null));
            const title = g.move.kind === "eat" ? "Meal" : g.move.kind === "study" ? activity : g.move.kind === "sleep" ? "Sleep" : g.move.kind === "rest" ? "Rest" : "Event";
            return (
              <li key={g.key} className={pl.row} data-row={g.key}>
                <div className={pl.rowTime}>
                  <b>{g.slot ? t12(g.slot.start) : ""}</b>
                </div>
                <div className={pl.rowRail} aria-hidden="true">
                  <i className={pl.rowDot} />
                </div>
                <div className={pl.rowCell}>
                  <div className={`${pl.rowCard} ${pl.rowWarn}`}>
                    <span className={pl.rowTop}>
                      <span className={pl.rowTitle}>{title}</span>
                      <span className={pl.rowTags}>
                        <span className={`${pl.tagPill} ${pl.tagWarn}`}>Doesn&apos;t fit</span>
                      </span>
                    </span>
                    <span className={pl.rowMeta}>
                      {[placeShort(option?.place_name), g.move.minutes ? dur(g.move.minutes) : null].filter(Boolean).join(" · ")}
                    </span>
                    <span className={pl.rowReason}>{fix?.problem ?? "It doesn't fit where it is."}</span>
                  </div>
                  {readOnly ? null : (
                    <div className={pl.edit}>
                      {fix?.moves && fix.label ? (
                        <button type="button" className={pl.miniBtn} onClick={() => change(fix.moves as Move[], fix.changed)}>
                          {fix.label}
                        </button>
                      ) : null}
                      <button type="button" className={`${pl.miniBtn} ${pl.miniDanger}`} onClick={() => change(moves.filter((_, i) => i !== g.index), null)}>
                        Remove
                      </button>
                    </div>
                  )}
                </div>
              </li>
            );
          }
          if (readOnly) return null;
          return (
            <li key={g.key} className={pl.row}>
              <div className={pl.rowTime}>
                <span>{clockAt(g.start, board.start)}</span>
              </div>
              <div className={pl.rowRail} aria-hidden="true">
                <i className={`${pl.rowDot} ${pl.rowDotGap}`} />
              </div>
              <div className={pl.rowCell}>
                <button type="button" className={`${pl.rowCard} ${pl.rowAdd}`} aria-haspopup="dialog" onClick={() => setAdding(g)}>
                  <b>+ Add</b>
                  <span>{dur((g.end - g.start) / 60_000)} free</span>
                </button>
                {g.suggestions.map((s) => (
                  <SuggestRow key={s.id} s={s} onAdd={() => {
                    const out = insertAt(moves, { slot: s.slot ?? g.slot.id, kind: "event", event: s.id }, board, result, ms(s.start));
                    change(out.moves, out.index);
                  }} />
                ))}
              </div>
            </li>
          );
        })}
      </ul>
      {adding ? (
        <AddSheet
          slot={adding.slot}
          from={clockAt(adding.start, board.start)}
          free={(adding.end - adding.start) / 60_000}
          activity={activity}
          onClose={() => setAdding(null)}
          onPick={(o) => {
            const want = DEFAULT_MINUTES[o.kind] ?? null;
            const out = insertAt(
              moves,
              { slot: adding.slot.id, kind: o.kind, place: o.place, minutes: want === null ? null : Math.max(STEP, Math.min(want, o.max_minutes)), start: isoAt(adding.start, board.start), event: o.event },
              board,
              result,
              adding.start,
            );
            setAdding(null);
            change(out.moves, out.index);
          }}
        />
      ) : null}
    </>
  );
}

/** A "Near you" event in the gap where it would fall: dashed and sunken, one tap adds it. */
function SuggestRow({ s, onAdd }: { s: Suggestion; onAdd: () => void }) {
  const walk = s.walk_minutes ? `${Math.round(s.walk_minutes)} min walk` : null;
  return (
    <button type="button" className={`${pl.rowCard} ${pl.rowSuggest}`} onClick={onAdd}>
      <span className={pl.rowTitle}>
        <b>+ Add</b> {s.title.replace(/\s*\(optional\)\s*$/i, "")}
      </span>
      <span className={pl.rowMeta}>{[t12(s.start), placeShort(s.place_name), walk].filter(Boolean).join(" · ")}</span>
    </button>
  );
}

const ORDER = ["eat", "study", "rest", "sleep", "event"];
const PER_KIND: Record<string, number> = { eat: 3, study: 3, rest: 2, sleep: 1, event: 4 };

function AddSheet({ slot, from, free, activity, onPick, onClose }: { slot: OpenSlot; from: string; free: number; activity: string; onPick: (o: MoveOption) => void; onClose: () => void }) {
  const options = ORDER.flatMap((kind) =>
    slot.moves
      .filter((o) => o.kind === kind && o.feasible)
      .sort((a, b) => a.walk_minutes + a.back_minutes - (b.walk_minutes + b.back_minutes))
      .slice(0, PER_KIND[kind] ?? 2),
  );
  const title = (o: MoveOption) => {
    const place = placeShort(o.place_name);
    if (o.kind === "eat") return `A meal at ${place}`;
    if (o.kind === "study") return `${activity} at ${place}`;
    if (o.kind === "sleep") return "Sleep";
    if (o.kind === "rest") return `A rest at ${place}`;
    return o.title.replace(/^Go to: /, "");
  };
  return (
    <Sheet label="Add to your plan" onClose={onClose}>
      <div className={`${ui.row} ${ui.between}`}>
        <h2>Add at {from}</h2>
        <button type="button" className={ui.btn} onClick={onClose}>
          Close
        </button>
      </div>
      <p className={ui.muted}>{dur(free)} free here.</p>
      {options.length === 0 ? (
        <p className={ui.muted}>Nothing fits in this gap once the walks are taken out.</p>
      ) : (
        <div className={pl.options}>
          {options.map((o) => (
            <button key={`${o.kind}-${o.place}-${o.event}`} type="button" className={pl.option} onClick={() => onPick(o)}>
              <b>{title(o)}</b>
              <span>{o.walk_minutes > 0 ? `${Math.round(o.walk_minutes)} min walk` : "right here"}</span>
            </button>
          ))}
        </div>
      )}
    </Sheet>
  );
}
