import { useEffect, useRef, useState } from "react";
import { type OwnerRequest } from "./ownerConversation";

type WorkEvent = { label: string; outcome: string; gmail_label?: string };
type WorkLog = {
  status: string; failure_code: string | null; notice: string; events: WorkEvent[];
  model_parse_stages?: string[];
  activity?: { recorded_action_count: number; unconfirmed_effect_count: number; pending_tool: boolean; pending_intent?: string | null; outcome_counts?: { tool_kind: string; outcome: string; count: number }[]; latest?: { tool_kind: string; outcome: string; effects: string; detail?: string | null; gmail_label?: string } | null };
};

const TOOL_LABELS: Record<string, string> = {
  conversation_search: "Conversation history search", conversation_source: "Conversation source",
  web_search: "Web search", web_source: "Web source", local_command: "Local command",
  clock: "Clock check", schedules: "Schedule check", mail: "Unread mail check", mail_search: "Gmail search",
  calendar: "Google Calendar check", skills: "Skill catalog check", named_skill: "Named skill check", other: "Tool",
};
const OUTCOME_LABELS: Record<string, string> = {
  recorded: "an observation was recorded", no_results: "no results were returned", source_failed: "could not be completed",
  partial: "returned partial results because some search engines failed",
  timed_out: "timed out", interrupted: "was interrupted", exit_zero: "ended; what it achieved is unknown", exit_nonzero: "ended with a nonzero status; what that means for this request is unknown", unknown: "outcome is unknown",
};
const COMMAND_OUTCOMES: Record<string, string> = {
  exit_zero: "reported a normal command exit", exit_nonzero: "returned a code that can indicate a problem",
  timed_out: "timed out", interrupted: "were interrupted", unknown: "had an unknown outcome",
};
const INTENT_LABELS: Record<string, string> = {
  find_information: "find information", check_state: "check a state", make_change: "make a change", other: "use a tool",
};
const DETAIL_LABELS: Record<string, string> = {
  timeout: "The request timed out", http_error: "An HTTP error was reported", invalid_response: "The response could not be read",
  upstream_search_failed: "Upstream search engines failed",
  unresolvable_host: "The source address could not be resolved", output_too_large: "The response was too large",
};
const PARSE_STAGE_LABELS: Record<string, string> = {
  outer_object: "The response could not be read as a JSON object",
  arguments_not_string: "Tool arguments had the wrong format",
  arguments_invalid_json: "Tool arguments could not be decoded as JSON",
  arguments_not_object: "Tool arguments did not decode to an object",
};

function validGmailLabel(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value === value.trim() && Array.from(value).length <= 256;
}

export function focusRequestWorkLog(messageId: string): boolean {
  const details = document.getElementById(`conversation-work-log-${messageId}`);
  if (!(details instanceof HTMLDetailsElement)) return false;
  details.open = true;
  const summary = details.querySelector<HTMLElement>("summary");
  summary?.focus();
  return document.activeElement === summary;
}

/** Render only bounded descriptions and counts returned by the owner-scoped endpoint. */
export function ConversationWorkLog({ request, messageId, active, revealFailure = false, summary = "Work log", showEvidenceLimits = false, labelContextId, disclosureId, onActiveUpdate }: { request: OwnerRequest; messageId: string; active: boolean; revealFailure?: boolean; summary?: string; showEvidenceLimits?: boolean; labelContextId?: string; disclosureId?: string; onActiveUpdate?: () => void }) {
  const [open, setOpen] = useState(active || revealFailure);
  const wasVisible = useRef({ active, revealFailure });
  useEffect(() => {
    if ((active && !wasVisible.current.active) || (revealFailure && !wasVisible.current.revealFailure)) setOpen(true);
    wasVisible.current = { active, revealFailure };
  }, [active, revealFailure]);
  const [log, setLog] = useState<WorkLog | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  useEffect(() => {
    if (!open && !active) return;
    let cancelled = false;
    let issued = 0;
    let applied = 0;
    const load = async () => {
      const sequence = ++issued;
      try {
        const result = await request<WorkLog>("experimental_owner_work_log", { message_id: messageId });
        if (!result || !Array.isArray(result.events) || result.events.length > 101
            || !result.events.every((event) => typeof event.label === "string" && event.label.length <= 100
              && typeof event.outcome === "string" && event.outcome.length <= 100
              && (event.gmail_label === undefined || validGmailLabel(event.gmail_label)))
            || typeof result.notice !== "string" || result.notice.length > 200
            || !["running", "blocked", "completed", "waiting_user", "cancelled"].includes(result.status)
            || (result.failure_code !== null && typeof result.failure_code !== "string")
            || (result.model_parse_stages !== undefined && (!Array.isArray(result.model_parse_stages)
              || result.model_parse_stages.length > 4 || !result.model_parse_stages.every((stage) =>
                typeof stage === "string" && Object.hasOwn(PARSE_STAGE_LABELS, stage))))
            || (result.activity !== undefined && (!result.activity ||
              !Number.isSafeInteger(result.activity.recorded_action_count) || result.activity.recorded_action_count < 0 ||
              !Number.isSafeInteger(result.activity.unconfirmed_effect_count) || result.activity.unconfirmed_effect_count < 0 ||
              typeof result.activity.pending_tool !== "boolean" ||
              (result.activity.outcome_counts !== undefined && (!Array.isArray(result.activity.outcome_counts) ||
                result.activity.outcome_counts.length > 100 || !result.activity.outcome_counts.every((item) => item &&
                  typeof item.tool_kind === "string" && item.tool_kind.length <= 40 && typeof item.outcome === "string" && item.outcome.length <= 40 &&
                  Number.isSafeInteger(item.count) && item.count > 0 && item.count <= 100) ||
                result.activity.outcome_counts.reduce((total, item) => total + item.count, 0) > 100))))) throw new Error("invalid");
        if (!cancelled && sequence > applied) { applied = sequence; setLog(result); setUnavailable(false); }
      } catch { if (!cancelled && sequence > applied) { applied = sequence; setLog(null); setUnavailable(true); } }
    };
    void load();
    const timer = active ? window.setInterval(() => void load(), 2000) : undefined;
    return () => { cancelled = true; if (timer) window.clearInterval(timer); };
  }, [request, messageId, active, open, revealFailure]);

  useEffect(() => { if (active) onActiveUpdate?.(); }, [active, log, unavailable, onActiveUpdate]);

  const groups: { event: WorkEvent; count: number }[] = [];
  for (const event of log?.events ?? []) {
    // Preserve order and do not merge observations separated by a different event.
    const last = groups[groups.length - 1];
    if (last && last.event.label === event.label && last.event.outcome === event.outcome
        && last.event.gmail_label === event.gmail_label) last.count++;
    else groups.push({ event, count: 1 });
  }
  const count = log?.activity?.recorded_action_count;
  const latest = log?.activity?.latest;
  const latestLabel = latest && Object.hasOwn(TOOL_LABELS, latest.tool_kind) ? TOOL_LABELS[latest.tool_kind] : undefined;
  const latestGmailLabel = latest?.tool_kind === "mail_search" && validGmailLabel(latest.gmail_label) ? latest.gmail_label : undefined;
  const latestOutcome = latest && Object.hasOwn(OUTCOME_LABELS, latest.outcome) ? OUTCOME_LABELS[latest.outcome] : undefined;
  const gmailLabels = (log?.events ?? []).flatMap(({ gmail_label }) => gmail_label ? [gmail_label] : []);
  const gmailSearchOrder = gmailLabels.length > 1 ? gmailLabels.join(" → ") : null;
  const latestIncludedInSearchOrder = gmailSearchOrder && latestGmailLabel && latest?.outcome === "recorded";
  const commands = (log?.activity?.outcome_counts ?? []).filter((item) => item.tool_kind === "local_command");
  const commandSummary = commands.filter((item) => Object.hasOwn(COMMAND_OUTCOMES, item.outcome))
    .map((item) => `${item.count} ${item.outcome === "interrupted" && item.count === 1 ? "was interrupted" : COMMAND_OUTCOMES[item.outcome]}`).join("; ");
  const intent = active && log?.status === "running" && log.activity?.pending_tool && log.activity.pending_intent;
  const intentLabel = intent && Object.hasOwn(INTENT_LABELS, intent) ? INTENT_LABELS[intent] : undefined;
  const latestDetail = latest?.detail && Object.hasOwn(DETAIL_LABELS, latest.detail) ? DETAIL_LABELS[latest.detail] : undefined;
  const parseStages = log?.model_parse_stages ?? [];
  return <details id={disclosureId} className="conversation-work-log" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary id={labelContextId ? `work-log-summary-${messageId}` : undefined} aria-labelledby={labelContextId ? `work-log-summary-${messageId} ${labelContextId}` : undefined}>{summary}</summary>
    {showEvidenceLimits && <p>These are recorded observations, not source content or proof of an action’s effects.</p>}
    {unavailable ? <p>Work log is unavailable or not recorded for this request.</p> : !log ? <p>Checking recorded work…</p> : <>
      <p>{count === undefined ? (log.events.length ? "Recorded work is available in technical details." : "No tool observations recorded yet.") : count ? `${count} tool ${count === 1 ? "observation" : "observations"} recorded.` : "No tool observations recorded yet."}</p>
      {commandSummary && <p>Local commands: {commandSummary}. Some commands use these codes for other reasons. They do not establish what happened with your request.</p>}
      {gmailSearchOrder && <p>Gmail search · {gmailSearchOrder}</p>}
      {latestLabel && latestOutcome && !latestIncludedInSearchOrder && !(latest?.tool_kind === "local_command" && commandSummary) && <p>{latestLabel}{latestGmailLabel && ` · ${latestGmailLabel}`}: {latestOutcome}.{latestDetail && ` ${latestDetail}.`}</p>}
      {Boolean(log.activity?.pending_tool) && <p>{intentLabel && `Alphonse says it plans to ${intentLabel}. `}A tool request was recorded; whether it ran is unconfirmed.</p>}
      {Boolean(log.activity?.unconfirmed_effect_count) && <p>The log cannot confirm whether the recorded local commands changed anything.</p>}
      {(groups.length > 0 || parseStages.length > 0) && <details className="conversation-work-log-details">
        <summary>Technical details</summary>
        {groups.length > 0 && <ol>{groups.map(({ event, count: groupCount }, index) => <li key={index}>{event.label}{event.gmail_label && ` · ${event.gmail_label}`} · {event.outcome}{groupCount > 1 && ` (${groupCount} times)`}</li>)}</ol>}
        {parseStages.length > 0 && <ul>{parseStages.map((stage) => <li key={stage}>{PARSE_STAGE_LABELS[stage]}</li>)}</ul>}
      </details>}
      <small>{log.notice}</small>
    </>}
  </details>;
}
