import { useEffect, useState } from "react";
import { MascotImage } from "./SkillMascot";

export function ConversationActivity() {
  const [startedAt] = useState(() => Date.now());
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [startedAt]);
  return <div className="conversation-activity" role="status" aria-label="Alphonse is working. Your request is in progress.">
    <MascotImage state="pending" className="conversation-activity-mascot" />
    <div className="conversation-activity-body">
      <div className="conversation-activity-heading"><span className="conversation-activity-pulse" aria-hidden="true" />
        <strong>Alphonse is working</strong>
        <span className="conversation-activity-grid" aria-hidden="true">{Array.from({ length: 9 }, (_, index) => <i key={index} />)}</span>
        <span className="conversation-activity-time" aria-hidden="true">{Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}</span>
      </div>
      <span className="conversation-activity-subtitle">Your request is in progress.</span>
    </div>
  </div>;
}
