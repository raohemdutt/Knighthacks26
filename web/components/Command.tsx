"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import ui from "./ui.module.css";
import { PLAN_EDIT, hasScreenIntent, screenIntent } from "@/lib/voice";

/** One thing the palette can do: open a screen. The groups come from the shell's sections. */
export interface Command {
  href: string;
  label: string;
  group: string;
  icon: ReactNode;
  /** Other words someone might type for it. */
  keywords?: string;
}

/** The event the palette sends when a question is typed into it on the Ask screen itself. */
export const ASK_EVENT = "atlas:ask";

type Row = { kind: "go"; command: Command } | { kind: "ask"; question: string };

const QUESTION = /\?$|^(what|where|when|who|how|is|are|did|do|does|which|show|tell|was|were|has|have|can|should)\b/;

/**
 * The command palette: Ctrl or Cmd + K anywhere, or the search field in the sidebar. Type to filter the screens;
 * type a question and Enter sends it to Ask. Arrow keys move, Enter goes, Escape closes.
 */
export function CommandPalette({ commands, path, onClose }: { commands: Command[]; path: string; onClose: () => void }) {
  const router = useRouter();
  const [text, setText] = useState("");
  const [cursor, setCursor] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => input.current?.focus(), []);

  const query = text.trim();
  const needle = query.toLowerCase();
  const rows = useMemo<Row[]>(() => {
    const hits = commands.filter((c) => !needle || `${c.label} ${c.group} ${c.keywords ?? ""}`.toLowerCase().includes(needle));
    const out: Row[] = hits.map((command) => ({ kind: "go", command }));
    // A question, or anything no screen matches, can be asked.
    // On a screen that takes commands of its own (Today's Plan), so can a change to it: "drop the second study block".
    const forScreen = hasScreenIntent(path) && PLAN_EDIT.test(needle);
    if (query.length > 2 && (hits.length === 0 || QUESTION.test(needle) || forScreen)) out.unshift({ kind: "ask", question: query });
    return out;
  }, [commands, needle, query, path]);

  useEffect(() => setCursor(0), [needle]);

  const run = (row: Row) => {
    onClose();
    if (row.kind === "go") {
      router.push(row.command.href);
      return;
    }
    const toAsk = () => {
      if (path === "/") window.dispatchEvent(new CustomEvent(ASK_EVENT, { detail: row.question }));
      else router.push(`/?q=${encodeURIComponent(row.question)}`);
    };
    // The screen this was typed on is offered the words first (Today's Plan: "move lunch to one" changes the plan
    // there). Everywhere else, and when the screen says they are not for it, they go to Ask as before.
    if (hasScreenIntent(path)) void screenIntent(path, row.question).then((taken) => (taken === null ? toAsk() : undefined));
    else toAsk();
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowDown") {
        e.preventDefault();
        setCursor((c) => Math.min(c + 1, rows.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setCursor((c) => Math.max(c - 1, 0));
      } else if (e.key === "Enter") {
        const row = rows[cursor];
        if (row) run(row);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // `run` closes over the latest props through the effect's own closure.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, cursor, onClose]);

  // Keep the chosen row in view when the arrows move it.
  useEffect(() => {
    list.current?.querySelector<HTMLElement>("[data-on='true']")?.scrollIntoView({ block: "nearest" });
  }, [cursor]);

  // Group headings only while browsing, not while filtering.
  let lastGroup: string | null = null;

  return (
    <div className={ui.paletteOverlay} onClick={onClose}>
      <div className={ui.palette} role="dialog" aria-modal="true" aria-label="Jump to a screen or ask a question" onClick={(e) => e.stopPropagation()}>
        <div className={ui.paletteHead}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <circle cx="11" cy="11" r="7" />
            <path d="M20 20l-3.5-3.5" />
          </svg>
          <input
            ref={input}
            className={ui.paletteInput}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Jump to a screen, or ask Atlas anything"
            aria-label="Search"
            autoComplete="off"
            spellCheck={false}
          />
          <span className={ui.kbd}>esc</span>
        </div>
        <div ref={list} className={ui.paletteList} role="listbox">
          {rows.length === 0 ? <p className={`${ui.muted} ${ui.paletteGroup}`}>Nothing matches. Keep typing to ask it instead.</p> : null}
          {rows.map((row, i) => {
            const on = i === cursor;
            if (row.kind === "ask")
              return (
                <button key="ask" type="button" role="option" aria-selected={on} data-on={on} className={`${ui.paletteItem} ${on ? ui.paletteItemOn : ""}`} onMouseEnter={() => setCursor(i)} onClick={() => run(row)}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M4 5.5h16v10.5H10l-4.5 3.5V16H4z" />
                  </svg>
                  <span>
                    Ask Atlas: <em>{row.question}</em>
                  </span>
                  <span className={ui.paletteSub}>Enter</span>
                </button>
              );
            const { command } = row;
            const heading = !needle && command.group !== lastGroup ? command.group : null;
            lastGroup = command.group;
            return (
              <div key={command.href}>
                {heading ? <p className={ui.paletteGroup}>{heading}</p> : null}
                <button type="button" role="option" aria-selected={on} data-on={on} className={`${ui.paletteItem} ${on ? ui.paletteItemOn : ""}`} onMouseEnter={() => setCursor(i)} onClick={() => run(row)}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    {command.icon}
                  </svg>
                  <span>{command.label}</span>
                  <span className={ui.paletteSub}>{command.href === path ? "here" : needle ? command.group : ""}</span>
                </button>
              </div>
            );
          })}
        </div>
        <div className={ui.paletteHint}>
          <span>
            <span className={ui.kbd}>↑</span> <span className={ui.kbd}>↓</span> move
          </span>
          <span>
            <span className={ui.kbd}>↵</span> open
          </span>
          <span>type a question to ask it</span>
        </div>
      </div>
    </div>
  );
}
