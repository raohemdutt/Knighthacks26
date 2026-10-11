"use client";

import ui from "@/components/ui.module.css";
import { LevelIcon } from "./FloorMap";
import o from "./op.module.css";
import type { Shift } from "./types";

/**
 * The Equipment view of the Inspection workspace: every piece of equipment, worst first, in the same three levels as the
 * floor plan. The same rows and words as the "All equipment" sheet (Shift.tsx, AllSheet), laid out as a page.
 */
export function EquipmentList({ shift, onPick }: { shift: Shift; onPick: (id: string) => void }) {
  return (
    <section className={o.card} aria-label="All equipment">
      <div className={o.cardHead}>
        <h2>All equipment</h2>
      </div>
      {(shift.levels ?? []).map((lv) => {
        const rows = shift.assets.filter((a) => a.level === lv.level);
        if (rows.length === 0) return null;
        return (
          <section key={lv.level} className={ui.section}>
            <h3 className={o.groupTitle}>
              <LevelIcon level={lv.level} size={22} /> {lv.words} · {rows.length}
            </h3>
            <p className={o.note}>{lv.rule}</p>
            <ul className={o.rows}>
              {rows.map((a) => (
                <li key={a.asset_id}>
                  <button type="button" className={o.rowBtn} onClick={() => onPick(a.asset_id)}>
                    <b>{a.label}</b>
                    <span>{a.place}</span>
                    <span className={o.rowSentence}>{a.sentence}</span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </section>
  );
}
