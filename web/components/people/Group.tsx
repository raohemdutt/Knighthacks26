"use client";

import Link from "next/link";
import ui from "@/components/ui.module.css";
import { api, type CircleMember, type FaceShareStatus } from "@/lib/api";
import { relative } from "@/lib/format";
import { REHEARSAL } from "@/lib/rehearsal";
import { useFetch } from "@/lib/useFetch";
import p from "./people.module.css";

/**
 * "Your group" on the People screen: the people who share the network's GPUs with you.
 *
 * Read-only, and nothing here is made up. Each figure comes from one of three places, named under the card:
 *   - who is in the group and whether their box was heard from: GET /api/circle/members (the Circle screen's list);
 *   - where the shared model runs and what the last answer cost: GET /api/circle/brain/power (the Shared brain card);
 *   - alone against with a friend: GET /api/learning/lab/compare (the Learning lab's Together tab), or, on a box that
 *     recorded no comparison, the team's recorded rehearsal (lib/rehearsal.ts), tagged as such.
 * On a box with no Circle the rows are three example people, tagged "Demo data".
 *
 * The Shared brain card itself lives on the Circle screen; this links to it and to the Learning lab.
 */

const DEMO_GROUP: CircleMember[] = ["You", "Maya", "Dominick"].map((name, i) => ({
  member_id: `demo-${i}`,
  display_name: name,
  mine: false,
  has_box: null,
  box_seen: null,
  place: null,
})) as unknown as CircleMember[];

const kj = (j: number) => `${(j / 1000).toFixed(1)} kJ`;
const whole = (n: number) => `${Math.round(n)}`;

export function Group({ sharing }: { sharing?: FaceShareStatus | null }) {
  const circle = useFetch((s) => api.circleMembers(s));
  const power = useFetch((s) => api.circleBrainPower(undefined, s));
  const compare = useFetch((s) => api.labCompare(null, null, s));

  const real = circle.data?.configured ? circle.data.members : [];
  const demo = real.length === 0;
  const members = demo ? DEMO_GROUP : real;

  // Alone against together: this box's own recorded comparison when it has one, else the recorded rehearsal.
  const a = compare.data?.a;
  const b = compare.data?.b;
  const own = a?.j_per_example != null && b?.j_per_example != null && b.pods_joules != null && a.examples != null && b.examples != null;
  const solo = own ? { examples: a!.examples!, j_per_example: a!.j_per_example!, pods_joules: a!.pods_joules ?? 0 } : REHEARSAL.solo;
  const duo = own ? { examples: b!.examples!, j_per_example: b!.j_per_example!, pods_joules: b!.pods_joules! } : REHEARSAL.duo;
  const ratio = own ? { examples: duo.examples / solo.examples, pods_joules: solo.pods_joules ? duo.pods_joules / solo.pods_joules : null } : REHEARSAL.ratio;
  // How many people the shared run's energy is split between: what the run says, else how many times the solo run's examples it held.
  const split = Math.max(1, (own ? b!.members : null) ?? Math.round(duo.examples / solo.examples));
  const share = duo.pods_joules / split;
  const teacher = own ? (b!.labeling?.node ?? null) : REHEARSAL.teacher;
  const trainer = own ? (b!.training?.node ?? null) : REHEARSAL.trainer;

  // Where the shared model answers from, when the box has a shared brain configured (the Shared brain card's own source).
  const brain = power.data?.configured ? power.data : null;
  const pod = brain?.node ?? teacher;
  const last = brain?.last_answer && !brain.last_answer.pending && brain.last_answer.joules != null ? brain.last_answer : null;
  const watts = brain?.samples.at(-1)?.w ?? null;
  const training = new Set((sharing?.friends ?? []).filter((f) => f.outgoing.state === "shared" && f.incoming.state === "accepted").map((f) => f.member_id));

  return (
    <div className={`${ui.card} ${p.stack}`} data-guide="people:group">
      <div className={p.nameRow}>
        <h3>Your group</h3>
        {demo ? (
          <span className={`${ui.tag} ${p.demoTag}`} title="This box is not in a Circle, so these three are example people.">
            Demo data
          </span>
        ) : null}
        {!own ? (
          <span className={`${ui.tag} ${p.demoTag}`} title={`Copied from ${REHEARSAL.source}. This box has not recorded a comparison of its own.`}>
            Recorded rehearsal
          </span>
        ) : null}
      </div>
      <p className={p.headline} data-guide="people:group-headline">
        With a friend, each example costs <strong>{whole(duo.j_per_example)} J</strong> instead of <strong>{whole(solo.j_per_example)}</strong>.
      </p>
      <p className={`${ui.muted} ${ui.small}`}>
        {ratio.examples === 2 ? "Twice" : `${ratio.examples.toFixed(1)} times`} the examples
        {ratio.pods_joules != null ? ` for ${ratio.pods_joules.toFixed(2)}× the energy` : ""}: {duo.examples} examples took {kj(duo.pods_joules)} on the pods together, {solo.examples} took{" "}
        {kj(solo.pods_joules)} alone.
      </p>

      <ul className={p.group}>
        {members.map((m, i) => {
          const inRun = i < split; // the shared run's energy is split evenly between the people in it
          return (
            <li key={m.member_id} className={p.groupRow}>
              <div className={p.nameRow}>
                <strong className={ui.breakWords}>{m.display_name}</strong>
                {m.mine ? <span className={`${ui.tag} ${ui.tagAccent}`}>You</span> : null}
                {training.has(m.member_id) ? <span className={ui.tag}>Recognise each other</span> : null}
              </div>
              <p className={`${ui.muted} ${ui.small}`}>
                {demo
                  ? "Example person: no box to hear from."
                  : m.has_box
                    ? `Box heard from${m.box_seen && relative(m.box_seen) ? ` ${relative(m.box_seen)}` : ""}.`
                    : m.has_box === false
                      ? "Box not heard from yet."
                      : "No box seen from here (phone only so far)."}{" "}
                {pod ? `Shares ${pod}${trainer && trainer !== pod ? ` and ${trainer}` : ""}.` : "No shared pod set up on this box."}
              </p>
              <p className={ui.small}>
                {inRun ? (
                  <>
                    Energy share of the shared model: <strong>{kj(share)}</strong> <span className={ui.muted}>(an even {split === 2 ? "half" : `1/${split}`} of {kj(duo.pods_joules)})</span>
                  </>
                ) : (
                  <span className={ui.muted}>Not in the last shared run, so no energy share yet.</span>
                )}
              </p>
            </li>
          );
        })}
      </ul>

      {brain ? (
        <p className={ui.small}>
          The shared brain runs {brain.where}
          {watts != null ? `, drawing ${whole(watts)} W now` : ""}
          {last ? `. The last answer cost ${whole(last.joules!)} J` : ""}.
        </p>
      ) : (
        <p className={`${ui.muted} ${ui.small}`}>No shared brain is set up on this box yet, so there are no live watts to show.</p>
      )}

      <div className={`${ui.row} ${ui.wrap}`}>
        <Link className={ui.btn} href="/circle/" data-guide="people:group-brain">
          Open the shared brain
        </Link>
        <Link className={ui.btn} href="/lab/?tab=compare" data-guide="people:group-lab">
          Open the Learning lab
        </Link>
      </div>
      <p className={`${ui.muted} ${ui.small}`}>
        Sources: {demo ? "example people (this box is in no Circle)" : "your Circle's members"};{" "}
        {own ? "this box's own recorded runs (Learning lab, Together)" : `the rehearsal recorded ${REHEARSAL.recorded} (${REHEARSAL.source})`}
        {brain ? "; the shared brain's power samples" : ""}.
      </p>
    </div>
  );
}
