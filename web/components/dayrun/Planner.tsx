"use client";

import { useEffect, useState } from "react";
import type { Board, Change, Move, MoveKind, MoveOption, OpenSlot, SimResult } from "@/lib/api";
import { addMove, blockOfMove, clock, clockSpan, minutes, nextClock, placeName, shiftMove, type PlaceIndex } from "@/lib/dayrun";
import { useWords, type Words } from "@/lib/profile";
import ui from "../ui.module.css";
import dr from "./dayrun.module.css";

/**
 * The open slots of the board and the moves planned in each. Options, walking
 * minutes and limits all come from the board; what a move turned into (its real
 * start and end, or why it was rejected) comes from the last simulation.
 */

/** The name of a move kind in the profile's words: "study" is the activity ("Study", "Work", "Homework"). */
const kindName = (w: Words, kind: MoveKind): string => (kind === "study" ? w.Activity : { eat: "Eat", sleep: "Sleep", rest: "Rest", event: "Event" }[kind]);
const KINDS: MoveKind[] = ["study", "eat", "sleep", "rest", "event"];
/** A starting length for a new move; the box clamps and reports what really fits. `null` fills the slot. */
const DEFAULT_MINUTES: Record<MoveKind, number | null> = { study: 90, eat: 30, sleep: null, rest: null, event: null };
const STEP = 15;

function optionFor(slot: OpenSlot, move: Move): MoveOption | undefined {
  return (
    slot.moves.find((o) => o.kind === move.kind && (move.kind === "event" ? o.event === (move.event ?? null) : o.place === (move.place ?? null))) ??
    (move.place ? undefined : slot.moves.find((o) => o.kind === move.kind))
  );
}

function moveTitle(slot: OpenSlot | undefined, move: Move, places: PlaceIndex, w: Words): string {
  const option = slot ? optionFor(slot, move) : undefined;
  if (option && (move.place || move.kind === "event")) return option.title;
  if (move.place) return `${kindName(w, move.kind)} at ${placeName(places, move.place)}`;
  return move.kind === "sleep" ? "Sleep" : `${kindName(w, move.kind)} where you are`;
}

export function Planner({
  board,
  moves,
  onChange,
  result,
  changes,
  places,
}: {
  board: Board;
  moves: Move[];
  onChange: (moves: Move[]) => void;
  result: SimResult | undefined;
  changes: Change[];
  places: PlaceIndex;
}) {
  const w = useWords();
  const [adding, setAdding] = useState<string | null>(null);
  const [editing, setEditing] = useState<number | null>(null); // the one move whose controls are open
  const slotIds = new Set(board.slots.map((s) => s.id));
  const strays = moves.map((m, i) => ({ m, i })).filter(({ m }) => !slotIds.has(m.slot));
  const addingSlot = board.slots.find((s) => s.id === adding);

  const update = (index: number, patch: Partial<Move>) => onChange(moves.map((m, i) => (i === index ? { ...m, ...patch } : m)));
  const remove = (index: number) => {
    setEditing(null);
    onChange(moves.filter((_, i) => i !== index));
  };
  const shift = (index: number, dir: -1 | 1) => {
    setEditing(index + dir);
    onChange(shiftMove(moves, index, dir));
  };

  const row = (m: Move, i: number, slot: OpenSlot | undefined, first: boolean, last: boolean) => {
    const option = slot ? optionFor(slot, m) : undefined;
    const block = blockOfMove(result, i);
    const rejected = result?.rejected.find((r) => r.index === i);
    const notes = changes.filter((c) => c.move === i && c.kind !== "rejected");
    const friend = option?.friends[0];
    const walk = rejected ? undefined : result?.routes.find((r) => r.move === i);
    const bedtime = m.start ? clock(m.start) : "";
    const open = editing === i;
    return (
      <li key={i} className={`${dr.move} ${rejected ? dr.moveRejected : ""}`}>
        <div className={`${ui.row} ${ui.between}`}>
          <button type="button" className={dr.moveHead} aria-expanded={open} onClick={() => setEditing(open ? null : i)}>
            <span className={dr.moveTitle}>
              {/* What the box made of the move (its real place, after any re-route), else what was asked for. */}
              {block ? block.title : `${moveTitle(slot, m, places, w)}${m.with_friend ? ` with ${m.with_friend}` : ""}`}
            </span>
            <span className={dr.moveEdit}>{open ? "Done" : "Change"}</span>
          </button>
          <div className={dr.moveTools}>
            {open ? (
              <>
                <button type="button" className={dr.tool} aria-label="Move earlier" disabled={first} onClick={() => shift(i, -1)}>
                  ↑
                </button>
                <button type="button" className={dr.tool} aria-label="Move later" disabled={last} onClick={() => shift(i, 1)}>
                  ↓
                </button>
              </>
            ) : null}
            <button type="button" className={dr.tool} aria-label="Remove this move" onClick={() => remove(i)}>
              ×
            </button>
          </div>
        </div>

        {rejected ? (
          <p className={dr.rejected} role="alert">
            <strong>Rejected:</strong> {rejected.reason}
          </p>
        ) : block ? (
          <p className={`${ui.muted} ${ui.small}`}>
            Runs {clockSpan(block.start, block.end)} · {minutes(block.minutes)}
            {walk ? ` · ${minutes(walk.minutes)} walk first, leaving ${clock(walk.depart)}` : ""}
          </p>
        ) : result ? (
          <p className={`${ui.muted} ${ui.small}`}>Not in the simulated day.</p>
        ) : null}
        {notes.map((c, n) => (
          <p key={n} className={dr.changed}>
            <span className={ui.tag}>{c.kind.replace("_", " ")}</span> {c.text}
          </p>
        ))}

        {open ? (
        <div className={`${ui.row} ${ui.wrap}`}>
          <div className={dr.stepper} role="group" aria-label="Minutes">
            <button type="button" className={dr.tool} aria-label="15 minutes less" disabled={!m.minutes || m.minutes <= STEP} onClick={() => update(i, { minutes: Math.max(STEP, (m.minutes ?? STEP) - STEP) })}>
              −
            </button>
            <span className={dr.stepValue}>{m.minutes ? minutes(m.minutes) : m.kind === "event" ? "whole event" : "fills the slot"}</span>
            <button
              type="button"
              className={dr.tool}
              aria-label="15 minutes more"
              onClick={() => update(i, { minutes: m.minutes ? m.minutes + STEP : Math.min(60, Math.max(STEP, option?.max_minutes ?? 60)) })}
            >
              +
            </button>
          </div>
          {m.minutes ? (
            <button type="button" className={dr.pill} onClick={() => update(i, { minutes: null })}>
              {m.kind === "event" ? "Whole event" : "Fill the slot"}
            </button>
          ) : null}
          {friend || m.with_friend ? (
            <button
              type="button"
              className={`${dr.pill} ${m.with_friend ? dr.pillOn : ""}`}
              aria-pressed={Boolean(m.with_friend)}
              onClick={() => update(i, { with_friend: m.with_friend ? null : (friend ?? null) })}
            >
              With {m.with_friend ?? friend}
            </button>
          ) : null}
          {m.kind === "sleep" && slot ? (
            <label className={dr.bedtime}>
              <span>Bedtime</span>
              <input
                type="time"
                className={dr.timeInput}
                value={bedtime}
                onChange={(e) => update(i, { start: e.target.value ? nextClock(slot.start, e.target.value) : null })}
              />
            </label>
          ) : m.start ? (
            <button type="button" className={`${dr.pill} ${dr.pillOn}`} onClick={() => update(i, { start: null })}>
              Starts {bedtime} ×
            </button>
          ) : null}
        </div>
        ) : null}
        {open && option && !rejected && !block ? (
          <p className={`${ui.muted} ${ui.small}`}>
            {option.walk_minutes > 0 ? `${minutes(option.walk_minutes)} walk from the start of the slot` : "No walk from the start of the slot"}
            {option.back_minutes > 0 ? `, ${minutes(option.back_minutes)} on to the next block` : ""}.
          </p>
        ) : null}
      </li>
    );
  };

  return (
    <>
      <ul className={ui.list}>
        {board.slots.map((slot) => {
          const mine = moves.map((m, i) => ({ m, i })).filter(({ m }) => m.slot === slot.id);
          return (
            <li key={slot.id} className={ui.card}>
              <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
                <h3>
                  {clockSpan(slot.start, slot.end)} <span className={ui.muted}>· {minutes(slot.minutes)} open</span>
                </h3>
                <span className={ui.tag}>{slot.id}</span>
              </div>
              <p className={`${ui.muted} ${ui.small}`}>
                Starts at {placeName(places, slot.from_place)}
                {slot.to_place ? `, then you must be at ${placeName(places, slot.to_place)}` : ""}.
              </p>
              {mine.length === 0 ? <p className={ui.muted}>Nothing planned here yet.</p> : <ul className={dr.moves}>{mine.map(({ m, i }, n) => row(m, i, slot, n === 0, n === mine.length - 1))}</ul>}
              <button type="button" className={ui.btn} onClick={() => setAdding(slot.id)}>
                Add a move
              </button>
            </li>
          );
        })}
        {strays.length > 0 ? (
          <li className={ui.card}>
            <h3>Moves for slots this board does not have</h3>
            <p className={`${ui.muted} ${ui.small}`}>They are still sent to the box, which decides what to do with them.</p>
            <ul className={dr.moves}>{strays.map(({ m, i }) => row(m, i, undefined, true, true))}</ul>
          </li>
        ) : null}
      </ul>

      {addingSlot ? (
        <AddSheet
          slot={addingSlot}
          onClose={() => setAdding(null)}
          onPick={(o) => {
            const want = DEFAULT_MINUTES[o.kind];
            onChange(
              addMove(
                moves,
                {
                  slot: addingSlot.id,
                  kind: o.kind,
                  place: o.place,
                  minutes: want === null ? null : Math.max(1, Math.min(want, o.max_minutes)),
                  with_friend: o.friends[0] ?? null,
                  event: o.event,
                },
                board.slots,
              ),
            );
            setAdding(null);
            setEditing(null);
          }}
        />
      ) : null}
    </>
  );
}

function AddSheet({ slot, onPick, onClose }: { slot: OpenSlot; onPick: (o: MoveOption) => void; onClose: () => void }) {
  const w = useWords();
  const kinds = KINDS.filter((k) => slot.moves.some((o) => o.kind === k));
  const [kind, setKind] = useState<MoveKind>(kinds[0] ?? "study");
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const options = slot.moves.filter((o) => o.kind === kind);

  return (
    <div className={ui.overlay} onClick={onClose}>
      <div className={ui.sheet} role="dialog" aria-modal="true" aria-label="Add a move" onClick={(e) => e.stopPropagation()}>
        <div className={`${ui.row} ${ui.between}`}>
          <h2>
            Add to {clockSpan(slot.start, slot.end)}
          </h2>
          <button type="button" className={ui.btn} onClick={onClose}>
            Close
          </button>
        </div>
        {slot.moves.length === 0 ? (
          <p className={ui.muted}>The box offers no moves for this slot.</p>
        ) : (
          <>
            <div className={ui.chips} role="group" aria-label="Kind of move">
              {kinds.map((k) => (
                <button key={k} type="button" className={`${ui.chip} ${k === kind ? ui.chipOn : ""}`} aria-pressed={k === kind} onClick={() => setKind(k)}>
                  {kindName(w, k)}
                </button>
              ))}
            </div>
            <ul className={dr.options}>
              {options.map((o, n) => (
                <li key={`${o.kind}-${o.place}-${o.event}-${n}`}>
                  <button type="button" className={dr.option} disabled={!o.feasible} onClick={() => onPick(o)}>
                    <span className={dr.optionTitle}>
                      {o.title}
                      {o.friends.length > 0 ? ` · with ${o.friends.join(", ")}` : ""}
                    </span>
                    {o.feasible ? (
                      <span className={dr.optionMeta}>
                        {o.walk_minutes > 0 ? `${minutes(o.walk_minutes)} walk there` : "no walk"}
                        {o.back_minutes > 0 ? ` · ${minutes(o.back_minutes)} on to the next block` : ""} · up to {minutes(o.max_minutes)}
                      </span>
                    ) : (
                      <span className={dr.optionMeta}>Does not fit: {o.reason ?? "the box gave no reason"}</span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}
