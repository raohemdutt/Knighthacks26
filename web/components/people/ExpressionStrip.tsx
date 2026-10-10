"use client";

import ui from "@/components/ui.module.css";
import { api, type Looked } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { percent, time } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import p from "./people.module.css";

/** The only words used for a face here: how it looked. Never what someone felt. */
const WORDS: Record<Looked, string> = { upbeat: "upbeat", neutral: "neutral", tired: "tired", tense: "tense" };
const ORDER: Looked[] = ["upbeat", "neutral", "tired", "tense"];

/**
 * The last few hours as a strip: one block per stretch of time, coloured by how the face looked most of that
 * stretch. `memberId` null is the anonymous room. Apparent expression is a noisy reading of a face's shape.
 */
export function ExpressionStrip({ memberId, hours = 6 }: { memberId: string | null; hours?: number }) {
  const series = useFetch((s) => api.expressions(memberId, hours, s), `${memberId ?? "room"}:${hours}`);
  useAtlasEvents(["perception.expression"], series.reload);
  const data = series.data;
  if (!data) return series.error ? <p className={`${ui.muted} ${ui.small}`}>The expression history could not be loaded.</p> : null;
  const first = data.windows[0];
  const last = data.windows[data.windows.length - 1];
  if (!first || !last)
    return (
      <p className={`${ui.muted} ${ui.small}`}>
        Nothing in the last {hours} hours yet. A stretch appears after about half a minute of {memberId ? "this face" : "faces"} in view of a camera.
      </p>
    );
  const most = ORDER.filter((k) => data.looked[k]).sort((a, b) => (data.looked[b] ?? 0) - (data.looked[a] ?? 0));
  const people = memberId === null ? Math.max(...data.windows.map((w) => w.people ?? 1)) : 0;
  const summary =
    (memberId === null ? `${people} ${people === 1 ? "person" : "people"} at most, mostly ` : "Looked mostly ") +
    most.slice(0, 2).map((k) => `${WORDS[k]} (${percent(data.looked[k] ?? 0)})`).join(", then ");
  return (
    <div>
      <p>{summary}</p>
      <div className={p.strip} role="img" aria-label={`${summary}. From ${time(first.start)} to ${time(last.end)}.`}>
        {data.windows.map((w) => (
          <span
            key={`${w.camera_id}-${w.start}`}
            className={p[w.looked]}
            style={{ flexGrow: Math.max(new Date(w.end).getTime() - new Date(w.start).getTime(), 1000) }}
            title={`${time(w.start)} to ${time(w.end)}: looked ${WORDS[w.looked]}`}
          />
        ))}
      </div>
      <ul className={p.stripLegend}>
        {ORDER.map((k) => (
          <li key={k}>
            <i className={p[k]} />
            looked {WORDS[k]}
          </li>
        ))}
      </ul>
      <p className={`${ui.muted} ${ui.small}`}>
        {time(first.start)} to {time(last.end)}, from {data.samples} looks. This is how a face looked to a small
        model, added up over many looks: a rough signal, often wrong on any one look, and not a reading of feelings or health.
      </p>
    </div>
  );
}
