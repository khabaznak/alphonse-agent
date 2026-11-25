import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { draftQuestionAboutRequest, quotedRequestText } from "../../src/conversationFollowup";
import { ConversationTimelineEntry } from "../../src/ConversationTimelineEntry";
import { FailedReferenceNotice } from "../../src/FailedReferenceNotice";
import { ConversationActivity } from "../../src/ConversationActivity";
import { focusConversationRequest, focusOwnerRecovery, OwnerRecovery } from "../../src/OwnerRecoveryPanel";
import type { OwnerRequest } from "../../src/ownerConversation";
import "../../src/styles.css";

const id = "a".repeat(32);
const secondId = "b".repeat(32);
const log = { status: "running", failure_code: null, notice: "This log shows recorded tool observations; it may not include every activity.",
  activity: { recorded_action_count: 3, unconfirmed_effect_count: 2, pending_tool: false, latest: { tool_kind: "web_search", outcome: "no_results", effects: "not_reported" } },
  events: [{ label: "Local command", outcome: "Command returned exit code 1; effects are unconfirmed" }, { label: "Local command", outcome: "Command returned exit code 1; effects are unconfirmed" }, { label: "Searched the web", outcome: "No results returned" }] };
function Fixture() {
  const [mode, setMode] = useState("live");
  const [dismissed, setDismissed] = useState(false);
  const [recovery, setRecovery] = useState(false);
  const [draft, setDraft] = useState("Existing draft");
  const [reference, setReference] = useState<{ id: string; createdAt: string } | null>(null);
  const [staleNotice, setStaleNotice] = useState(false);
  const draftRef = useRef<HTMLTextAreaElement>(null);
  const removeRef = useRef<HTMLButtonElement>(null);
  const request = useCallback((async (_method: string, args: { message_id: string }) => { if (args.message_id === secondId) throw new Error("fixture unavailable"); if (mode === "mixed commands") return { ...log, activity: { recorded_action_count: 6, unconfirmed_effect_count: 6, pending_tool: false, pending_intent: null, latest: { tool_kind: "local_command", outcome: "exit_zero", effects: "unconfirmed" }, outcome_counts: [{ tool_kind: "local_command", outcome: "exit_nonzero", count: 3 }, { tool_kind: "local_command", outcome: "exit_zero", count: 3 }] }, events: [...Array(3).fill({ label: "Local command", outcome: "Command returned exit code 1; effects are unconfirmed" }), ...Array(3).fill({ label: "Local command", outcome: "Command returned exit code 0; effects are unconfirmed" })] };
    if (mode === "command only") return { ...log, activity: { recorded_action_count: 1, unconfirmed_effect_count: 1, pending_tool: false, latest: { tool_kind: "local_command", outcome: "exit_nonzero", effects: "unconfirmed" } }, events: [log.events[0]] };
    if (mode === "pending") return { ...log, activity: { recorded_action_count: 0, unconfirmed_effect_count: 0, pending_tool: true, pending_intent: "check_state", latest: null }, events: [] };
    if (mode === "expired reservation") return { ...log, activity: { recorded_action_count: 0, unconfirmed_effect_count: 0, pending_tool: true, pending_intent: null, latest: null }, events: [] };
    if (mode === "settled") return { ...log, activity: { recorded_action_count: 0, unconfirmed_effect_count: 0, pending_tool: false, pending_intent: null, latest: null }, events: [] };
    if (mode === "empty") return { ...log, activity: { recorded_action_count: 0, unconfirmed_effect_count: 0, pending_tool: false, latest: null }, events: [] };
    return mode.startsWith("failed") ? { ...log, status: "blocked", activity: { ...log.activity, latest: { tool_kind: "web_source", outcome: "source_failed", effects: "not_reported", detail: "timeout" } } } : log; }) as OwnerRequest, [mode]);
  useEffect(() => { if (recovery) focusOwnerRecovery(id); }, [recovery]);
  return <main style={{maxWidth: 950, margin:"auto", padding:24}}><h1>Disposable activity fixture</h1>
    <p>Fixture data only; no household connection. Recovery navigation and actions are simulated; this is not the production App route.</p>
    <nav>{["live", "command only", "mixed commands", "pending", "expired reservation", "settled", "empty", "failed", "failed without recovery", "non-admin failed", "disconnected"].map((value) => <button key={value} onClick={() => { setMode(value); setRecovery(false); setDismissed(false); }}>{value}</button>)}</nav>
    {recovery ? <OwnerRecovery workLogRequest={request} onReturnToConversation={() => { setRecovery(false); window.requestAnimationFrame(() => focusConversationRequest(id)); }} items={[{message_id:id, prompt:"Find the latest household update", status:"action_unconfirmed", can_abandon:false}, {message_id:secondId, prompt:"Inspect another unrelated update", status:"action_unconfirmed", can_abandon:false}]} stopping="" onStop={() => {}} onShow={() => {}} /> :
      <section className="conversation"><div id="conversation-timeline" tabIndex={-1} className="timeline" style={{minHeight: 350}}><ConversationTimelineEntry message={{id, role:"user", content:"Find the latest household update", source:"desktop", created_at:"2026-11-25T14:00:00Z"}} questions={[]} user="fixture" timezone="UTC" onDone={async () => {}} projectName="" memoize={false} renderMarkdown={false} failures={(mode.startsWith("failed") || mode === "non-admin failed") ? [{ messageId:id, failureCode:"time_limit", dismissed }] : []} onFailureDismissedChange={(_, value) => setDismissed(value)} workLogRequest={mode === "disconnected" || mode === "non-admin failed" ? undefined : request} active={["live", "command only", "mixed commands", "pending", "expired reservation", "settled", "empty"].includes(mode)} onDraftQuestion={mode === "failed" || mode === "failed without recovery" ? (selectedId, content, createdAt) => { setReference({ id: selectedId, createdAt: createdAt || "" }); setStaleNotice(false); setDraft((current) => draftQuestionAboutRequest(current, content)); window.requestAnimationFrame(() => draftRef.current?.focus()); } : undefined} onReviewRecovery={mode === "failed" ? () => setRecovery(true) : undefined} />
        {mode === "failed" && <ConversationTimelineEntry message={{id:secondId, role:"user", content:"Find the latest household update", source:"desktop", created_at:"2026-11-25T14:05:00Z"}} questions={[]} user="fixture" timezone="UTC" onDone={async () => {}} projectName="" memoize={false} renderMarkdown={false} failures={[{messageId:secondId, failureCode:"time_limit", dismissed:false}]} onFailureDismissedChange={() => {}} workLogRequest={request} onDraftQuestion={(selectedId, content, createdAt) => { setReference({ id: selectedId, createdAt: createdAt || "" }); setStaleNotice(false); setDraft((current) => draftQuestionAboutRequest(current, content)); window.requestAnimationFrame(() => draftRef.current?.focus()); }} />}
        {["live", "command only", "mixed commands", "pending", "expired reservation", "settled", "empty"].includes(mode) && <ConversationActivity />}</div>
        <section className="input-dock">{mode === "disconnected" && <p className="conversation-active-task" role="status">Connection lost. The status of this request is unknown until Alphonse reconnects.</p>}{reference && <FailedReferenceNotice id={reference.id} createdAt={reference.createdAt} timezone="UTC" unavailable={staleNotice} removeRef={removeRef} onRemove={() => { setReference(null); setStaleNotice(false); draftRef.current?.focus(); }} />}{reference && <button type="button" onClick={() => { setStaleNotice(true); window.requestAnimationFrame(() => removeRef.current?.focus()); }}>Simulate stale-reference rejection</button>}<label>Editable draft (fixture only)<textarea ref={draftRef} value={draft} onChange={(event) => { const value = event.target.value; setDraft(value); if (!value.includes(quotedRequestText("Find the latest household update"))) setReference(null); }} /></label></section></section>}
  </main>;
}
createRoot(document.getElementById("root")!).render(<Fixture />);
