"use client";

import { ApprovalCard } from "@/components/ApprovalCard";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { useAtlas } from "@/lib/atlas";
import { isSpotApproval } from "@/lib/spotView";

/**
 * Inspection's approvals: the cards Inspection itself raised (a re-inspection, a work order, a mission request), with the
 * same card and the same Approve and Deny as the Approvals screen. Every other card stays on that screen.
 */
export function SpotApprovals() {
  const { approvals, link } = useAtlas();
  const { loading, error, reload } = approvals;
  const items = approvals.items.filter(isSpotApproval);
  return (
    <section aria-label="Approvals">
      <div className={ui.pageHead}>
        <h2>Approvals</h2>
        <p className={ui.muted}>Atlas never sends, writes or exports anything until you approve it here.</p>
      </div>
      {error && items.length === 0 ? (
        <ErrorBox what="pending approvals" error={error} onRetry={reload} />
      ) : loading && items.length === 0 ? (
        <Loading what="pending approvals" />
      ) : items.length === 0 ? (
        <Empty title="Nothing waiting for you">{link === "live" ? "New requests show up on their own." : "The live connection is down, so pull up this screen again to check."}</Empty>
      ) : (
        <ul className={ui.list}>
          {items.map((r) => (
            <ApprovalCard key={r.id} request={r} onDecided={reload} />
          ))}
        </ul>
      )}
    </section>
  );
}
