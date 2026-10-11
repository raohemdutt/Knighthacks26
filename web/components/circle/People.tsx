"use client";

import { useState, type FormEvent } from "react";
import { RouteFacts, RouteMap, RoutePlaces } from "@/components/map/RouteMap";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type CircleMember, type CircleMembers, type FaceShareFriend, type FaceShareStatus, type LeagueStanding, type PlacePack, type PlaceRoute, type PlaceShared } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { relative, time } from "@/lib/format";
import { useFetch, type Loadable } from "@/lib/useFetch";
import c from "./circle.module.css";
import { colourOf } from "./shared";

/**
 * People: every member of the Circle as one card. Where they are now (only if they shared a named place,
 * with when it expires), how their League week is going, whether a box of theirs has been heard from, and
 * the walk to where they are.
 *
 * The walk is the same one every other screen draws: GET /api/places/route (the mapped footpaths, the same
 * minutes Today's Plan counts) on the shared campus map (components/map/RouteMap.tsx), inside the friend's card.
 * It is only offered for a friend who shared a named place that has not run out, and the box checks that again
 * before it answers: this screen never learns a place that was not shared.
 *
 * GET /api/circle/members answers in every state. "Has a box" is what this box's own log shows: a member
 * whose Atlas sent a message or a summary, or whose desk camera shared a place. Otherwise it says so
 * plainly instead of guessing.
 *
 * "Here now" is this box's own cameras recognising someone who enrolled on the People screen and was linked to
 * a Circle member (GET /identity/present: ids only). It says nothing about anyone who did not enroll here, and it
 * is not shared with the Circle.
 *
 * "Can recognise you" and "Recognised here" are face share (off by default; set up on the People screen): you let
 * that friend's Atlas recognise you, or they let this box recognise them. No chip while it is off.
 */

/**
 * The household's named places (config/agent.yaml `places`). The gateway has no endpoint that lists them,
 * so these are suggestions; the box still has the last word and answers 422 with its own list.
 */
const SUGGESTED_PLACES = ["Our table", "Library", "Dorm", "Student Union", "Gym"];
const HOURS = [1, 2, 4];

const BOX: Record<string, string> = {
  agent: "its Atlas has answered a plan",
  summary: "it shares its week with the League",
  desk: "its desk camera shared a place",
  "this box": "this one",
};

export function People({ members, standings, packs, faceShare, onChanged }: {
  members: Loadable<CircleMembers>;
  standings: LeagueStanding[];
  packs: PlacePack[] | undefined;
  /** GET /api/circle/face-share; null or undefined while it is off or not loaded. */
  faceShare?: FaceShareStatus | null;
  onChanged: () => void;
}) {
  // Who this box recognises right now. No recognition on this box (or it is off): nobody, and no message about it.
  const seen = useFetch((s) => api.presentNow(s).catch(() => ({ present: [] })));
  useAtlasEvents(["perception.person", "perception.identity"], seen.reload);
  const here = new Set((seen.data?.present ?? []).map((x) => x.circle_member_id).filter((id): id is string => !!id));
  if (members.error) return <ErrorBox what="the Circle's people" error={members.error} onRetry={members.reload} />;
  if (!members.data) return <Loading what="the Circle's people" />;
  const data = members.data;
  if (!data.configured)
    return (
      <Empty title="No one here yet">
        Once this box joins a Circle, each friend appears here with the place they chose to share, their League week and the walk to where they are.
      </Empty>
    );
  const mine = data.members.find((m) => m.mine);
  return (
    <>
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>
          {data.members.length} in {data.circle_id}
        </h2>
        <ul className={ui.list}>
          {data.members.map((m) => (
            <Person key={m.member_id} member={m} here={here.has(m.member_id)} all={data.members} standing={standings.find((s) => s.member_id === m.member_id)} packs={packs} from={mine?.place?.place} face={faceShare?.friends.find((f) => f.member_id === m.member_id)} />
          ))}
        </ul>
      </section>
      <ShareMyPlace onShared={onChanged} />
    </>
  );
}

/** "Towers (dorm)" -> "Towers" */
function bare(name: string): string {
  return (name.split(" (")[0] ?? name).trim();
}

/** "ucf.library" for a place name or alias ("Library"), campus packs first. */
function placeId(packs: PlacePack[] | undefined, name: string | undefined): string | null {
  if (!name || !packs) return null;
  const want = name.trim().toLowerCase();
  for (const pack of [...packs].sort((a, b) => Number(a.kind !== "campus") - Number(b.kind !== "campus")))
    for (const [id, p] of Object.entries(pack.places)) {
      const names = [p.name, bare(p.name), ...(p.aliases ?? [])].map((n) => n.trim().toLowerCase());
      if (names.includes(want)) return `${pack.name}.${id}`;
    }
  return null;
}

/** Where a walk starts: where I said I am, else the campus pack's entry. */
function startOf(packs: PlacePack[] | undefined, mine: string | undefined): string | null {
  const here = placeId(packs, mine);
  if (here) return here;
  const campus = packs?.find((p) => p.kind === "campus" && p.entry) ?? packs?.find((p) => p.entry);
  return campus?.entry ? `${campus.name}.${campus.entry}` : null;
}

function Person({ member: m, here, all, standing, packs, from, face }: {
  member: CircleMember;
  face: FaceShareFriend | undefined;
  here: boolean;
  all: CircleMember[];
  standing: LeagueStanding | undefined;
  packs: PlacePack[] | undefined;
  from: string | undefined;
}) {
  const start = startOf(packs, from);
  const [route, setRoute] = useState<PlaceRoute | null>(null);
  const [routeError, setRouteError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const findRoute = async () => {
    if (!start || loading) return;
    setLoading(true);
    setRouteError(null);
    try {
      setRoute(await api.placeRoute({ from: start, to: `friend:${m.member_id}` }));
    } catch (err) {
      setRouteError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <li className={`${ui.card} ${c.person} ${colourOf(all, m.member_id)}`}>
      <span className={c.avatar} aria-hidden="true">
        {m.display_name.slice(0, 1).toUpperCase()}
      </span>
      <div className={c.personBody}>
        <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
          <h3 className={ui.breakWords}>
            {m.display_name}
            {m.mine ? <span className={ui.muted}> (you)</span> : null}
          </h3>
          {standing ? (
            <span className={`${ui.tag} ${standing.rank === 1 && standing.score !== null ? ui.tagAccent : ""}`}>
              {standing.score === null ? "No League days yet" : `#${standing.rank} this week · ${Math.round(standing.score)}%`}
            </span>
          ) : null}
        </div>
        {m.place ? (
          <>
            <p className={c.where}>{m.place.place}</p>
            <p className={`${ui.muted} ${ui.small}`}>
              {m.place.source === "desk" ? "Seen at the desk" : m.mine ? "Shared from your phone" : "Shared from their phone"} at {time(m.place.shared_at)} · until{" "}
              {time(m.place.expires_at)} ({relative(m.place.expires_at)})
            </p>
          </>
        ) : (
          <p className={ui.muted}>{m.mine ? "You are not sharing a place right now." : "Not sharing a place right now."}</p>
        )}
        <div className={c.facts}>
          {here ? <span className={`${ui.tag} ${ui.tagAccent}`}>Here now</span> : null}
          {face?.outgoing.state === "shared" ? <span className={ui.tag}>Can recognise you</span> : null}
          {face?.incoming.state === "accepted" ? <span className={ui.tag}>Recognised here</span> : null}
          <span className={ui.tag}>{m.has_box ? (m.mine ? "This box" : "Has a box") : "Phone only so far"}</span>
        </div>
        <p className={`${ui.muted} ${ui.small}`}>
          {m.has_box
            ? m.mine
              ? "Your Atlas answers friends' plans from your calendar, on this box."
              : `${m.display_name}'s box is on the Circle: ${BOX[m.box_seen ?? ""] ?? "it has been heard from"}.`
            : `This box has not heard from a box of ${m.display_name}'s, so no Atlas answers plans for them yet.`}
        </p>
        {!m.mine && m.place ? (
          <>
            {route ? (
              <div className={c.walk} data-guide={`circle:route:${m.member_id}`}>
                <p className={ui.nudge} role="status">
                  <strong>{route.walk_minutes === 0 ? "You are already there" : `${route.walk_minutes} min walk`}</strong> to {m.display_name} at {m.place.place}
                  {route.walk_minutes === 0 ? "." : `, from ${route.from_name}.`}
                </p>
                <RouteFacts route={route} />
                <RouteMap route={route} packs={packs} />
                <RoutePlaces route={route} packs={packs} />
              </div>
            ) : null}
            {routeError ? <p className={`${ui.muted} ${ui.small}`}>No route: {routeError}</p> : null}
            <div className={`${ui.row} ${ui.wrap}`}>
              {route ? (
                <button type="button" className={ui.btn} onClick={() => setRoute(null)}>
                  Hide the route
                </button>
              ) : (
                <button type="button" className={ui.btn} onClick={findRoute} disabled={!start || loading}>
                  {loading ? "Finding the way…" : `Route to ${m.display_name}`}
                </button>
              )}
            </div>
          </>
        ) : null}
      </div>
    </li>
  );
}

function ShareMyPlace({ onShared }: { onShared: () => void }) {
  const [place, setPlace] = useState(SUGGESTED_PLACES[0] ?? "");
  const [hours, setHours] = useState(2);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [shared, setShared] = useState<PlaceShared | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!place.trim() || busy) return;
    setBusy(true);
    setError(null);
    setShared(null);
    try {
      setShared(await api.sharePlace(place.trim(), hours));
      onShared();
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className={ui.section}>
      <h2 className={ui.sectionTitle}>Share where you are</h2>
      <form className={ui.card} onSubmit={submit}>
        <p className={`${ui.muted} ${ui.small}`}>Pick a named place. Friends see the name only, and it disappears when the time is up.</p>
        <div className={ui.chips} role="group" aria-label="Place">
          {SUGGESTED_PLACES.map((p) => (
            <button key={p} type="button" className={`${ui.chip} ${place === p ? ui.chipOn : ""}`} aria-pressed={place === p} onClick={() => setPlace(p)}>
              {p}
            </button>
          ))}
        </div>
        <input className={ui.input} value={place} onChange={(e) => setPlace(e.target.value)} placeholder="Place name" aria-label="Place name" />
        <div className={ui.chips} role="group" aria-label="For how long">
          {HOURS.map((h) => (
            <button key={h} type="button" className={`${ui.chip} ${hours === h ? ui.chipOn : ""}`} aria-pressed={hours === h} onClick={() => setHours(h)}>
              {h} {h === 1 ? "hour" : "hours"}
            </button>
          ))}
        </div>
        {error ? <WriteError error={error} /> : null}
        {shared ? (
          <p className={ui.nudge} role="status">
            Sharing <strong>{shared.place}</strong> until {time(shared.expires_at)}.
          </p>
        ) : null}
        <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy || !place.trim()}>
          {busy ? "Sharing…" : "Share this place"}
        </button>
      </form>
    </section>
  );
}
