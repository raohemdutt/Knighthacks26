"use client";

import { useEffect, useMemo, useState } from "react";
import { ItemRow } from "@/components/ItemRow";
import { Keyframe } from "@/components/Keyframe";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, ApiError, type DailyDigest, type Episode, type SearchHit } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { amount, cameraName, dayLabel, dayTime, humanize, safeUrl, span } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

type View = "camera" | "apps";

const KIND_LABELS: Record<string, string> = {
  assignment: "Assignments",
  calendar_event: "Calendar",
  message: "Messages",
  post: "Posts",
  page_change: "Page changes",
  search_result: "Search results",
};

export default function TimelinePage() {
  const [view, setView] = useState<View>("camera");
  return (
    <>
      <div className={ui.pageHead}>
        <h1>Timeline</h1>
      </div>
      <div className={ui.segmented} role="tablist" aria-label="What to show">
        <button type="button" role="tab" aria-selected={view === "camera"} className={`${ui.seg} ${view === "camera" ? ui.segOn : ""}`} onClick={() => setView("camera")}>
          Camera
        </button>
        <button type="button" role="tab" aria-selected={view === "apps"} className={`${ui.seg} ${view === "apps" ? ui.segOn : ""}`} onClick={() => setView("apps")}>
          Connected apps
        </button>
      </div>
      {view === "camera" ? <CameraView /> : <AppsView />}
    </>
  );
}

function CameraView() {
  const stats = useFetch((s) => api.stats(s));
  const episodes = useFetch((s) => api.episodes({}, s));
  const [camera, setCamera] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const [query, setQuery] = useState("");

  // Deep link: /timeline/?q=keys fills the box and runs the search. Read once, after the first
  // render (the page is a static export, so the query is only known in the browser).
  useEffect(() => {
    const q = new URLSearchParams(window.location.search).get("q")?.trim();
    if (!q) return;
    setTyped(q);
    setQuery(q);
  }, []);

  // Search as you type, once typing pauses.
  useEffect(() => {
    const t = window.setTimeout(() => setQuery(typed.trim()), 350);
    return () => window.clearTimeout(t);
  }, [typed]);

  // An episode shows up without a reload, both when it starts ("happening now") and when it finishes.
  useAtlasEvents(["memory.episode.opened", "memory.episode.closed"], () => {
    episodes.reload();
    stats.reload();
  });

  const cameras = useMemo(() => [...new Set((episodes.data ?? []).map((e) => e.camera_id))].sort(), [episodes.data]);
  const shown = useMemo(
    () =>
      (episodes.data ?? [])
        .filter((e) => !camera || e.camera_id === camera)
        .sort((a, b) => b.start.localeCompare(a.start)),
    [episodes.data, camera],
  );

  return (
    <>
      {stats.data ? (
        <div className={ui.stats}>
          <div className={ui.stat}>
            <p className={ui.statNum}>{amount(stats.data.hours_recorded)}</p>
            <p className={ui.statLabel}>hours recorded</p>
          </div>
          <div className={ui.stat}>
            <p className={ui.statNum}>{stats.data.episodes}</p>
            <p className={ui.statLabel}>
              episodes{stats.data.first_event ? ` since ${dayTime(stats.data.first_event)}` : ""}
            </p>
          </div>
        </div>
      ) : stats.error ? (
        <ErrorBox what="the recording totals" error={stats.error} onRetry={stats.reload} />
      ) : (
        <Loading what="recording totals" />
      )}

      <DigestCard />

      <form role="search" onSubmit={(e) => (e.preventDefault(), setQuery(typed.trim()))}>
        <input
          className={ui.input}
          type="search"
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          placeholder="Search what Atlas saw and read"
          aria-label="Search memory"
          enterKeyHint="search"
        />
      </form>

      {query ? (
        <SearchResults query={query} />
      ) : (
        <>
          {cameras.length > 1 ? (
            <div className={ui.chips} role="group" aria-label="Filter by camera">
              <button type="button" className={`${ui.chip} ${camera === null ? ui.chipOn : ""}`} aria-pressed={camera === null} onClick={() => setCamera(null)}>
                All cameras
              </button>
              {cameras.map((c) => (
                <button key={c} type="button" className={`${ui.chip} ${camera === c ? ui.chipOn : ""}`} aria-pressed={camera === c} onClick={() => setCamera(c)}>
                  {cameraName(c)}
                </button>
              ))}
            </div>
          ) : null}

          {episodes.error ? (
            <ErrorBox what="the timeline" error={episodes.error} onRetry={episodes.reload} />
          ) : !episodes.data ? (
            <Loading what="episodes" />
          ) : shown.length === 0 ? (
            <Empty title="Nothing recorded yet">
              When a camera sees something happen (someone sits down, a package arrives), it shows up here as a card with the time, a
              one-line description and a frame.
            </Empty>
          ) : (
            <EpisodeList episodes={shown} />
          )}
        </>
      )}
    </>
  );
}

/** The daily digest (A6): one tap, the day's episodes in a paragraph written from the data, with its moments. */
function DigestCard() {
  const [digest, setDigest] = useState<DailyDigest | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const load = async () => {
    setBusy(true);
    setError(null);
    try {
      setDigest(await api.digest());
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(0, e instanceof Error ? e.message : String(e)));
    } finally {
      setBusy(false);
    }
  };
  if (error) return <ErrorBox what="today's digest" error={error} onRetry={load} />;
  if (!digest)
    return (
      <div className={ui.row}>
        <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={load} disabled={busy}>
          {busy ? "Summing up today..." : "Today's digest"}
        </button>
      </div>
    );
  return (
    <section className={`${ui.card} ${ui.section}`} aria-label="Today's digest">
      <div className={`${ui.row} ${ui.between}`}>
        <h2 className={ui.sectionTitle}>Today's digest</h2>
        <button type="button" className={ui.btn} onClick={() => setDigest(null)}>
          Close
        </button>
      </div>
      <p className={ui.answerText}>{digest.text}</p>
      {digest.moments.length > 0 ? (
        <div className={ui.frames}>
          {digest.moments.map((m) =>
            m.citation.frame_ref ? <Keyframe key={m.hour} frameRef={m.citation.frame_ref} compact /> : null,
          )}
        </div>
      ) : null}
    </section>
  );
}

function EpisodeList({ episodes }: { episodes: Episode[] }) {
  // Episodes arrive sorted newest first; start a new heading whenever the local day changes.
  const groups: { day: string; items: Episode[] }[] = [];
  for (const e of episodes) {
    const day = dayLabel(e.start);
    const last = groups[groups.length - 1];
    if (last && last.day === day) last.items.push(e);
    else groups.push({ day, items: [e] });
  }
  return (
    <>
      {groups.map((g) => (
        <section key={g.day} className={ui.section}>
          <h2 className={`${ui.sectionTitle} ${ui.dayHeading}`}>{g.day}</h2>
          <ul className={ui.list}>
            {g.items.map((e) => (
              <li key={e.id} className={`${ui.card} ${ui.episode}`}>
                {e.keyframe_refs[0] ? <Keyframe frameRef={e.keyframe_refs[0]} compact /> : <div />}
                <div className={ui.episodeBody}>
                  <div className={`${ui.row} ${ui.wrap}`}>
                    <strong>{span(e.start, e.end)}</strong>
                    <span className={ui.tag}>{cameraName(e.camera_id)}</span>
                    {e.end === null ? <span className={`${ui.tag} ${ui.tagAccent}`}>happening now</span> : null}
                  </div>
                  <p className={ui.episodeText}>{e.summary || e.captions[0] || "No description yet. Atlas adds one once the episode is captioned."}</p>
                  {e.labels.length > 0 ? <p className={`${ui.muted} ${ui.small}`}>{e.labels.join(" · ")}</p> : null}
                </div>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </>
  );
}

function SearchResults({ query }: { query: string }) {
  const hits = useFetch((s) => api.search(query, 20, s), query);
  if (hits.error) return <ErrorBox what="search results" error={hits.error} onRetry={hits.reload} />;
  if (!hits.data) return <Loading what={`matches for "${query}"`} />;
  if (hits.data.length === 0)
    return (
      <Empty title={`No matches for "${query}"`}>
        Search looks through what the cameras saw and what Atlas read from your classes, calendar and group chat. Try an object or a course
        name.
      </Empty>
    );
  return (
    <section className={ui.section}>
      <h2 className={ui.sectionTitle}>
        {hits.data.length} {hits.data.length === 1 ? "match" : "matches"}, best first
      </h2>
      <ul className={ui.list}>
        {hits.data.map((h, i) => (
          <SearchHitCard key={`${h.kind}-${h.episode_id}-${i}`} hit={h} />
        ))}
      </ul>
    </section>
  );
}

function SearchHitCard({ hit }: { hit: SearchHit }) {
  const url = safeUrl(hit.url);
  // "note" is the box's hourly summary of what a camera saw: it carries a camera and a keyframe, not a connector.
  if (hit.kind === "episode" || hit.kind === "note")
    return (
      <li className={`${ui.card} ${ui.episode}`}>
        {hit.frame_ref ? <Keyframe frameRef={hit.frame_ref} compact /> : <div />}
        <div className={ui.episodeBody}>
          <div className={`${ui.row} ${ui.wrap}`}>
            <strong>{dayTime(hit.ts)}</strong>
            {hit.camera_id ? <span className={ui.tag}>{cameraName(hit.camera_id)}</span> : null}
            {hit.kind === "note" ? <span className={ui.tag}>Hour summary</span> : null}
          </div>
          <p className={ui.episodeText}>{hit.text}</p>
        </div>
      </li>
    );
  // For connector items the box puts the connector name in camera_id and starts the text with the title.
  const rest = hit.title && hit.text.startsWith(hit.title) ? hit.text.slice(hit.title.length).replace(/^[.\s]+/, "") : hit.text;
  return (
    <li className={ui.card}>
      <div className={`${ui.row} ${ui.wrap}`}>
        <span className={ui.tag}>{humanize(hit.kind)}</span>
        {hit.camera_id ? <span className={ui.tag}>{humanize(hit.camera_id)}</span> : null}
        <span className={`${ui.muted} ${ui.small}`}>{dayTime(hit.ts)}</span>
      </div>
      {hit.title ? <p className={ui.episodeText}>{url ? <a href={url} target="_blank" rel="noreferrer noopener">{hit.title}</a> : hit.title}</p> : null}
      {rest ? <p className={`${ui.muted} ${ui.breakWords}`}>{rest}</p> : null}
      {!hit.title && url ? (
        <a href={url} target="_blank" rel="noreferrer noopener">
          Open source
        </a>
      ) : null}
    </li>
  );
}

function AppsView() {
  const items = useFetch((s) => api.items({}, s));
  const [kind, setKind] = useState<string | null>(null);

  useAtlasEvents(["connector.item.synced"], () => items.reload());

  const kinds = useMemo(() => [...new Set((items.data ?? []).map((i) => i.kind))].sort(), [items.data]);
  const shown = useMemo(
    () =>
      (items.data ?? [])
        .filter((i) => !kind || i.kind === kind)
        .sort((a, b) => (b.when ?? "").localeCompare(a.when ?? "")),
    [items.data, kind],
  );

  if (items.error) return <ErrorBox what="items from your connected apps" error={items.error} onRetry={items.reload} />;
  if (!items.data) return <Loading what="items from your connected apps" />;
  if (items.data.length === 0)
    return (
      <Empty title="Nothing synced yet">
        Assignments from Canvas, calendar events and group chat messages appear here once the box has read them.
      </Empty>
    );

  return (
    <>
      {kinds.length > 1 ? (
        <div className={ui.chips} role="group" aria-label="Filter by type">
          <button type="button" className={`${ui.chip} ${kind === null ? ui.chipOn : ""}`} aria-pressed={kind === null} onClick={() => setKind(null)}>
            Everything
          </button>
          {kinds.map((k) => (
            <button key={k} type="button" className={`${ui.chip} ${kind === k ? ui.chipOn : ""}`} aria-pressed={kind === k} onClick={() => setKind(k)}>
              {KIND_LABELS[k] ?? humanize(k)}
            </button>
          ))}
        </div>
      ) : null}
      <ul className={ui.list}>
        {shown.map((i) => (
          <ItemRow key={`${i.connector}:${i.external_id}`} item={i} showKind />
        ))}
      </ul>
    </>
  );
}
