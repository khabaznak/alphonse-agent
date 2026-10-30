import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { DecisionContext } from "./DecisionContext";
import type { A2uiSurface } from "./a2ui";
import { progressStatus, terminalProgressStates } from "./taskProgress";

export function activityStateLabel(status: string, archived: boolean): string {
  const labels: Record<string, string> = { completed: "Completed", done: "Completed", success: "Completed", failed: "Failed", cancelled: "Cancelled", canceled: "Cancelled", abandoned: "Abandoned", interrupted: "Interrupted · outcome unconfirmed", waiting_user: "Waiting for your answer", waiting: "Waiting", retrying: "Waiting to retry" };
  return labels[status] || (archived ? "Stopped · outcome unconfirmed" : "Alphonse is working");
}

export function TaskProgressBubble({ surface, archived = false, connected = true, timezone = "UTC" }: { surface: A2uiSurface; archived?: boolean; connected?: boolean; timezone?: string }) {
  const text = (id: string) => String(surface.components[id]?.text || "").trim();
  const steps = Object.values(surface.components).filter((component) => component.id.startsWith("step_") && component.text).sort((left, right) => left.id.localeCompare(right.id, undefined, { numeric: true }));
  if (!connected) archived = true;
  const status = connected ? progressStatus(surface) : "interrupted";
  archived = archived || terminalProgressStates.has(status);
  const stateLabel = activityStateLabel(status, archived);
  const updated = String(surface.dataModel.updated_at || "");
  const timestamp = updated && Number.isFinite(Date.parse(updated)) ? new Date(updated) : null;
  const criteria = text("criteria").replace(/^Acceptance criteria\n?/, "").trim();
  const visibleCriteria = criteria === "- (none)" ? "" : criteria;
  const Container = archived ? "details" : "div";
  return <article className="message assistant task-progress-bubble" data-task-id={String(surface.dataModel.task_id || surface.surfaceId.slice("task-progress:".length))} aria-label="Task activity" aria-live="polite">
    <Container className="task-progress-log">
      {archived && <summary>Work log · {stateLabel}{text("summary") ? ` · ${text("summary")}` : ""}</summary>}
    <div className="task-progress-content">
      {timestamp && <small>Updated <time dateTime={updated}>{timestamp.toLocaleString(undefined, { timeZone: timezone })}</time></small>}
      {archived && <p>{status === "failed" ? "Review the failed action before retrying in conversation." : status === "cancelled" || status === "abandoned" ? "Work stopped. Ask in conversation to start another attempt." : status === "interrupted" ? "Reconnect to check the task status before retrying." : ""}</p>}
      {!archived && <div className="task-progress-heading">{status !== "waiting_user" && <span className="task-progress-spinner" aria-hidden="true">◌</span>}<strong>{stateLabel}</strong></div>}
      {!archived && text("summary") && <p>{text("summary")}</p>}
      {visibleCriteria && <section><small>Acceptance criteria</small><div className="task-progress-checklist"><ReactMarkdown remarkPlugins={[remarkGfm]}>{acceptanceCriteriaMarkdown(visibleCriteria)}</ReactMarkdown></div></section>}
      {text("intention") && <section><small>Intention</small><p>{text("intention").replace(/^Intention:\s*/, "")}</p></section>}
      {text("tool") && <p>{text("tool")}</p>}
      {text("blocker") && <p role="status">{text("blocker")}</p>}
      {steps.length ? <details className="task-progress-trace"><summary>Past actions · {steps.length}</summary>{steps.map((step) => {
        const [summary, ...detail] = String(step.text || "").split("\n");
        return <details className="task-progress-step" key={step.id}><summary>{summary}</summary><pre className="task-progress-detail">{detail.filter((line) => line !== "Details:").join("\n") || "No additional action details."}</pre></details>;
      })}</details> : <>{text("arguments") && <pre className="task-progress-detail">{text("arguments")}</pre>}{text("result") && <pre className="task-progress-detail">{text("result")}</pre>}</>}
      <DecisionContext summary={text("decision_context")} manifests={(surface.dataModel.decision_context_manifests || []) as Parameters<typeof DecisionContext>[0]["manifests"]} actorUserId={String(surface.dataModel.owner_user_id || "")} taskId={String(surface.dataModel.task_id || "")} />
    </div>
    </Container>
  </article>;
}

function acceptanceCriteriaMarkdown(value: string): string {
  return value.split("\n").map((line) => line.replace(/^\s*\d+\s*\.\s*-?\s*(\[[ xX]\])\s*/, "- $1 ")).join("\n");
}

