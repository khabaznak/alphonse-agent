export type Member = { user_id: string; display_name: string };
export type Source = { skill_id: string; role: string; binding_id: string; connection_id: string; grant_id: string; operation: string; resources: string[]; recipient_user_ids: string[]; project_id: string | null; display_name: string };
export type Connection = { connection_id: string; provider: string; display_name: string; revision: number; status: string; health_code: string; capabilities: string[]; credential_health?: string; credential_deletion_pending?: boolean; revocation_pending?: boolean };
export function groupAccountRecords(connections: Connection[]): { connected: Connection[]; other: Connection[] } {
  return { connected: connections.filter(c => c.status === 'connected'), other: connections.filter(c => c.status !== 'connected') };
}
export type Grant = { grant_id: string; connection_id: string; grantor_user_id: string; beneficiary_user_id: string; project_id: string | null; operations: string[]; resources: string[]; recipient_user_ids: string[]; expires_at: string; revision: number; revoked_at: string };
export type Catalog = { actor_user_id: string; can_configure_oauth: boolean; connections: Connection[]; grants: Grant[]; members: Member[]; providers: { provider: string; capabilities: string[]; status: string; source: string }[]; sources: Source[]; skills: { skill_id: string; name: string }[]; bindings: { binding_id: string; skill_id: string; grant_id: string }[]; projects: { project_id: string; name: string }[] };
export type Definition = { project_id: string; source_bindings: Omit<Source, "display_name" | "recipient_user_ids" | "project_id">[]; query: { text: string }; trigger: { kind: "schedule"; weekdays: number[]; local_time: string }; timezone: string; retrieval_window: { weekdays: number[]; start: string; end: string }; duration: { ongoing: boolean; starts_at: string; ends_at: string | null }; budgets: { max_pages: number; max_items: number; max_bytes: number; max_model_calls: number; max_runtime_seconds: number }; initial_lookback_days: number; recipient_user_ids: string[]; skill_hints: string[]; notification_mode: string; missed_run?: { kind: string; max_lateness_seconds?: number } };
export type Policy = Definition & { policy_id: string; revision: number; status: string };
export type Run = { check_run_id: string; state: string; outcome: string; error_code: string; window_start: string; window_end: string; stale: boolean; offline: boolean; pending_analysis: boolean | null; processing_details_expired?: boolean; coverage: Record<string, unknown> };
export type ProjectChecks = { checks: { policy: Policy; runs: Run[]; next_run_at: string | null }[]; findings: { finding_id: string; summary: string; state: string; due_at: string | null; timezone: string | null; uncertainty: string[]; coverage: Record<string, unknown>; deliveries: Record<string, { state: string }> }[]; sources: Source[] };
export const labels: Record<string, string> = { mail_read: "Read mail", calendar_discover: "Find calendars", calendar_read: "Read calendar events", classroom_read: "Read coursework", classroom_announcements_read: "Read classroom announcements", read: "Read", search: "Search", list: "List", changes: "Read changes", 'attachments.read': "Read attachments" };
export const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
export function defaultDefinition(projectId: string): Definition {
  return { project_id: projectId, source_bindings: [], query: { text: "" }, trigger: { kind: "schedule", weekdays: [0, 1, 2, 3, 4], local_time: "08:00" }, timezone: Intl.DateTimeFormat().resolvedOptions().timeZone, retrieval_window: { weekdays: [0, 1, 2, 3, 4], start: "08:00", end: "09:00" }, duration: { ongoing: false, starts_at: new Date().toISOString(), ends_at: null }, budgets: { max_pages: 5, max_items: 100, max_bytes: 1000000, max_model_calls: 2, max_runtime_seconds: 60 }, initial_lookback_days: 30, recipient_user_ids: [], skill_hints: [], notification_mode: "material_changes", missed_run: { kind: "next_allowed_window" } };
}
export function sourceDefinition(s: Source): Definition['source_bindings'][number] {
  const { skill_id, role, binding_id, connection_id, grant_id, operation, resources } = s;
  return { skill_id, role, binding_id, connection_id, grant_id, operation, resources };
}
export function editableDefinition(p: Policy): Definition {
  return { project_id: p.project_id, source_bindings: p.source_bindings.map(s => sourceDefinition(s as Source)), query: p.query, trigger: p.trigger, timezone: p.timezone, retrieval_window: p.retrieval_window, duration: p.duration, budgets: p.budgets, initial_lookback_days: p.initial_lookback_days, recipient_user_ids: p.recipient_user_ids, skill_hints: p.skill_hints, notification_mode: p.notification_mode, missed_run: p.missed_run };
}
export function validateDefinition(d: Definition): string | null {
  if (!d.query.text.trim()) return "Describe the goal for this check.";
  if (!d.source_bindings.length) return "Choose at least one authorized source.";
  if (!d.recipient_user_ids.length) return "Choose who may receive findings.";
  if (!d.trigger.weekdays.length) return "Choose at least one day.";
  if (!(d.retrieval_window.start <= d.trigger.local_time && d.trigger.local_time < d.retrieval_window.end)) return "The check time must fall inside the retrieval window, including a narrow window.";
  if (!d.duration.ongoing && (!d.duration.ends_at || Date.parse(d.duration.ends_at) <= Date.parse(d.duration.starts_at))) return "Choose an end after the start, or explicitly choose ongoing.";
  try { new Intl.DateTimeFormat("en", { timeZone: d.timezone }); } catch { return "Enter a valid timezone."; }
  if (!Number.isInteger(d.initial_lookback_days) || d.initial_lookback_days < 1 || d.initial_lookback_days > 365) return "Lookback must be 1–365 days.";
  if (Object.entries(d.budgets).some(([key, v]) => !Number.isInteger(v) || v < (key === 'max_model_calls' ? 0 : 1))) return "Enter whole-number budgets within their allowed bounds.";
  return null;
}
// A component owns one fence. Invalidation on unmount/identity change prevents
// late requests (including an earlier refresh) from replacing current state.
export class ResponseFence {
  private version = 0;
  start(): number { return ++this.version; }
  current(version: number): boolean { return version === this.version; }
  invalidate(): void { ++this.version; }
}
export function friendly(value: string): string { return value.replaceAll('_', ' '); }
