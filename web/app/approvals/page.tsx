"use client";

import { ApprovalCard } from "@/components/ApprovalCard";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { useAtlas } from "@/lib/atlas";

export default function ApprovalsPage() {
  // The pending list lives in the app-wide provider: it reloads itself when /ws reports a new request or a decision.
  const { approvals, link } = useAtlas();
  const { items, loading, error, reload } = approvals;

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Approvals</h1>
        <p className={ui.muted}>Atlas never sends, writes or exports anything until you approve it here.</p>
      </div>

      {error && items.length === 0 ? (
        <ErrorBox what="pending approvals" error={error} onRetry={reload} />
      ) : loading && items.length === 0 ? (
        <Loading what="pending approvals" />
      ) : items.length === 0 ? (
        <Empty title="Nothing waiting for you">
          When Atlas wants to message someone, add a calendar event, share with your Circle or export training data, a card appears here
          with Approve and Deny.{" "}
          {link === "live" ? "New requests show up on their own." : "The live connection is down, so pull up this screen again to check."}
        </Empty>
      ) : (
        <ul className={ui.list}>
          {items.map((r) => (
            <ApprovalCard key={r.id} request={r} onDecided={reload} />
          ))}
        </ul>
      )}
    </>
  );
}
