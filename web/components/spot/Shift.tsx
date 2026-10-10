"use client";

import { useEffect, useState, type ReactNode } from "react";
import ui from "@/components/ui.module.css";
import { ApiError, api, spotUrl } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { LevelIcon } from "./FloorMap";
import o from "./op.module.css";
import type { AssetState, CompareVisual, Level, OpAction, PlainFinding, Shift, Target, WorkOrder } from "./types";

/**
 * Layer 1, "Your shift": what a person starting a shift needs in ten seconds. Every sentence here comes from
 * GET /api/spot/shift (contexts/spot/application/plain.py); this file only lays it out. "Why?" opens the evidence
 * panel that backs the sentence.
 */

export function Why({ to, onWhy, children }: { to: Target; onWhy: (t: Target) => void; children?: ReactNode }) {
  return (
    <button type="button" className={o.why} onClick={() => onWhy(to)}>
      {children ?? "Why?"}
    </button>
  );
}

/**
 * The one button on an item. It raises an approval card on the box (the same flow as before): nothing is written until
 * the person says yes, here or on the Approvals screen, and then only a file on this box.
 */
export function ActionButton({ action, orders, quiet }: { action: OpAction; orders: WorkOrder[]; quiet?: boolean }) {
  const { approvals } = useAtlas();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isOrder = action.kind === "work_order";
  const target = isOrder ? "spot.work_order" : "spot.reinspect";
  const mine = approvals.items.filter((r) => r.target === target && (isOrder ? r.params.anomaly_id === action.anomaly_id : r.params.asset_id === action.asset_id));
  const written = orders.filter((w) => (isOrder ? w.anomaly_id === action.anomaly_id && w.kind !== "reinspect" : w.asset_id === action.asset_id && w.kind === "reinspect"));

  const run = (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    fn()
      .then(() => approvals.reload())
      .catch((e: unknown) => setError(e instanceof ApiError ? (e.needsToken ? "This box asks for a token first (Settings)." : e.message) : String(e)))
      .finally(() => setBusy(false));
  };
  const ask = () => run(() => (isOrder && action.anomaly_id ? api.spotWorkOrder(action.anomaly_id) : api.spotReinspect(action.asset_id)));

  if (mine.length > 0) {
    const r = mine[0]!;
    return (
      <div className={o.confirm} role="group" aria-label="Confirm">
        <p>
          <b>Ready to send.</b> {isOrder ? "A work order for a technician" : "A request to send SPOT back"} will be written on this box. Nothing leaves the building.
        </p>
        <div className={o.confirmRow}>
          <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy} onClick={() => run(() => api.decide(r.id, "approve"))}>
            Yes, do it
          </button>
          <button type="button" className={ui.btn} disabled={busy} onClick={() => run(() => api.decide(r.id, "deny"))}>
            Not now
          </button>
        </div>
      </div>
    );
  }
  return (
    <>
      <button type="button" className={`${ui.btn} ${quiet ? "" : ui.btnPrimary} ${o.act}`} disabled={busy} onClick={ask}>
        {action.label}
      </button>
      {written.length > 0 ? (
        <p className={o.done}>
          <LevelIcon level="ok" size={18} /> Done: {written[0]!.work_order_id} is written on this box.
        </p>
      ) : null}
      {error ? <p className={o.error}>{error}</p> : null}
    </>
  );
}

export function StatusLine({ shift }: { shift: Shift }) {
  const st = shift.status;
  if (!st) return null;
  const [lead, ...rest] = st.sentence.split(". ");
  return (
    <header className={o.status}>
      <p className={o.eyebrow}>Your shift · Inspection</p>
      <h1 className={o.statusText}>
        {lead}.{" "}
        {rest.map((part, i) => (
          <span key={i} className={i === 0 && st.act > 0 ? o.statusAct : undefined}>
            {part}
            {part.endsWith(".") ? " " : ". "}
          </span>
        ))}
      </h1>
      <p className={o.statusSub}>
        {st.as_of}. {st.last_round ? `${st.last_round}.` : ""}
      </p>
    </header>
  );
}

/** "How much can you trust a completed round?": one honest line with a row of tiles, one per completed round. */
export function TrustLine({ trust, onWhy }: { trust: NonNullable<Shift["trust"]>; onWhy: (t: Target) => void }) {
  const k = trust.facts.k ?? 0;
  const n = trust.facts.n ?? 0;
  return (
    <button type="button" className={o.trust} onClick={() => onWhy(trust.why_ref)}>
      <span className={o.trustTiles} aria-hidden="true">
        {Array.from({ length: n }, (_, i) => (
          <i key={i} className={i < k ? o.tileBad : o.tileOk} />
        ))}
      </span>
      <span className={o.trustText}>
        <b>{trust.question}</b> {trust.sentence}
      </span>
      <span className={o.whyInline}>Why?</span>
    </button>
  );
}

function CardHead({ level, title, hint }: { level: Level; title: string; hint?: string }) {
  return (
    <div className={o.cardHead}>
      <LevelIcon level={level} size={26} />
      <h2>{title}</h2>
      {hint ? <span className={o.hint}>{hint}</span> : null}
    </div>
  );
}

export function DoFirst({ shift, orders, onAsset, onWhy, onAll }: { shift: Shift; orders: WorkOrder[]; onAsset: (id: string) => void; onWhy: (t: Target) => void; onAll: () => void }) {
  const items = shift.do_first;
  const more = (shift.do_first_total ?? items.length) - items.length;
  return (
    <section className={`${o.card} ${o.cardAct}`} aria-label="Do this first">
      <CardHead level="act" title="Do this first" hint={items.length ? "in this order" : undefined} />
      {items.length === 0 ? <p className={o.sentence}>Nothing needs you right now.</p> : null}
      <ol className={o.items}>
        {items.map((it, i) => (
          <li key={it.id} className={o.item}>
            <span className={o.num} aria-hidden="true">
              {i + 1}
            </span>
            <div className={o.itemBody}>
              <button type="button" className={o.itemTitle} onClick={() => onAsset(it.asset_id)}>
                {it.headline}
              </button>
              <p className={o.sentence}>{it.sentence}</p>
              <div className={o.itemActs}>
                {it.action ? <ActionButton action={it.action} orders={orders} /> : null}
                <Why to={it.why_ref} onWhy={onWhy} />
              </div>
            </div>
          </li>
        ))}
      </ol>
      <p className={o.foot}>
        An <b>alarm</b> means SPOT flagged something unusual. <b>Not confirmed</b> means the check that raised it failed.
        {more > 0 ? (
          <>
            {" "}
            <button type="button" className={o.link} onClick={onAll}>
              {more} more need{more === 1 ? "s" : ""} you
            </button>
          </>
        ) : null}
      </p>
    </section>
  );
}

/** Two shares side by side: the tiny picture next to a pattern. */
export function MiniBars({ v }: { v: CompareVisual }) {
  const share = (b: { k: number; n: number }) => (b.n ? b.k / b.n : 0);
  const top = Math.max(0.01, ...v.bars.map(share));
  return (
    <div className={o.bars} role="img" aria-label={v.bars.map((b) => `${b.label}: ${b.k} of ${b.n} ${v.unit}`).join("; ")}>
      {v.bars.map((b, i) => (
        <div key={b.label} className={o.barRow}>
          <span className={o.barLabel}>{b.label}</span>
          <span className={o.barTrack}>
            <i className={i === 0 ? o.barHot : o.barCool} style={{ width: `${Math.max(3, (100 * share(b)) / top)}%` }} />
          </span>
          <span className={o.barVal}>
            {b.k} of {b.n}
          </span>
        </div>
      ))}
    </div>
  );
}

export function WatchOut({ items, onWhy }: { items: PlainFinding[]; onWhy: (t: Target) => void }) {
  return (
    <section className={o.card} aria-label="Watch out">
      <CardHead level="watch" title="Watch out" hint="patterns that affect today" />
      {items.length === 0 ? <p className={o.sentence}>No pattern stands out in the rounds so far.</p> : null}
      <ul className={o.items}>
        {items.map((it) => (
          <li key={it.id} className={o.pattern}>
            <p className={o.sentence}>{it.sentence}</p>
            {it.visual ? <MiniBars v={it.visual} /> : null}
            <div className={o.itemActs}>
              {it.note ? <span className={o.note}>{it.note}</span> : null}
              <Why to={{ kind: "finding", id: it.finding_id }} onWhy={onWhy} />
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function LooksFine({ shift, onAll }: { shift: Shift; onAll: () => void }) {
  const fine = shift.fine;
  const st = shift.status;
  if (!fine || !st) return null;
  return (
    <section className={o.card} aria-label="Looks fine">
      <CardHead level="ok" title="Looks fine" />
      <div className={o.fineRow}>
        <p className={o.fineNum}>
          {fine.count}
          <span> of {fine.total}</span>
        </p>
        <div className={o.split} role="img" aria-label={`${st.act} need you, ${st.watch} to keep an eye on, ${st.ok} look fine`}>
          <i className={o.splitAct} style={{ flexGrow: st.act }} />
          <i className={o.splitWatch} style={{ flexGrow: st.watch }} />
          <i className={o.splitOk} style={{ flexGrow: st.ok }} />
        </div>
      </div>
      <p className={o.sentence}>{fine.sentence}</p>
      <button type="button" className={`${ui.btn} ${o.act}`} onClick={onAll}>
        See all {fine.total}
      </button>
    </section>
  );
}

export function Sheet({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className={ui.overlay} onClick={onClose}>
      <div className={ui.sheet} role="dialog" aria-modal="true" aria-label={title} onClick={(e) => e.stopPropagation()}>
        <div className={`${ui.row} ${ui.between}`}>
          <h2 className={o.sheetTitle}>{title}</h2>
          <button type="button" className={ui.btn} onClick={onClose}>
            Close
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className={o.fact}>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

/** The short panel behind a mark on the floor plan: what it is, the last check, the last reading, the alarm, one button. */
export function AssetSheet({ asset, orders, onClose, onWhy }: { asset: AssetState; orders: WorkOrder[]; onClose: () => void; onWhy: (t: Target) => void }) {
  return (
    <Sheet title={asset.label} onClose={onClose}>
      <p className={`${o.levelTag} ${o[`tag_${asset.level}`]}`}>
        <LevelIcon level={asset.level} size={22} /> {asset.level_words}
        {asset.reasons.length ? <span>Because: {asset.reasons.join("; ")}.</span> : null}
      </p>
      <p className={o.sheetLead}>{asset.sentence}</p>
      <dl className={o.facts}>
        <Fact label="What it is">{asset.what}</Fact>
        <Fact label="Last check">
          <LevelIcon level={asset.last_check.worked === false ? "act" : "ok"} size={18} /> {asset.last_check.sentence}
        </Fact>
        <Fact label="Last reading">
          {asset.reading.sentence}
          {asset.reading.range ? <small>{asset.reading.range.words}</small> : null}
        </Fact>
        <Fact label="Alarm">
          {asset.alarm.sentence} {asset.alarm.count_words}
        </Fact>
      </dl>
      {asset.sample ? (
        <figure className={o.sample}>
          <img src={spotUrl(asset.sample.url)} alt="" loading="lazy" />
          <figcaption>
            <b>Sample.</b> {asset.sample.label}
          </figcaption>
        </figure>
      ) : null}
      <div className={o.itemActs}>
        {asset.action ? <ActionButton action={asset.action} orders={orders} quiet={asset.level === "ok"} /> : null}
        <Why to={asset.why_ref} onWhy={onWhy}>
          Why? See every check
        </Why>
      </div>
    </Sheet>
  );
}

/** Every piece of equipment, worst first, in the same three levels as the floor plan. */
export function AllSheet({ shift, onPick, onClose }: { shift: Shift; onPick: (id: string) => void; onClose: () => void }) {
  return (
    <Sheet title="All equipment" onClose={onClose}>
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
    </Sheet>
  );
}
