import Link from "next/link";
import ui from "@/components/ui.module.css";

export default function NotFound() {
  return (
    <div className={ui.state}>
      <p className={ui.stateTitle}>There is no screen at this address</p>
      <p>
        Use the tabs below, or go back to <Link href="/">Ask</Link>.
      </p>
    </div>
  );
}
