/** Layout metadata only. Never persist widget payloads or treat a saved ID as permission. */
export type WidgetMode = "contextual" | "pinned" | "locked";
export type WidgetPlacement = { id: string; kind: string; mode: WidgetMode; expiresAt?: number };
export type DrawerState = { open: boolean; widgets: WidgetPlacement[]; dismissed: string[] };
export type ContextWidget = { id: string; kind: string; expiresAt: number };
export const WIDGET_CATALOG = [
  { kind: "workload", title: "Workload", description: "Waiting and processing work over the last 30 minutes of this session." },
  { kind: "pdca", title: "PDCA activity", description: "Planning, doing, checking and acting over this session." },
  { kind: "care", title: "Treatments & care", description: "Authorized Doctor records, routines and recorded adherence." },
  { kind: "sensors", title: "Home sensors", description: "Authorized readings with source and freshness." },
  { kind: "emergency_contacts", title: "Emergency contacts", description: "Configured contacts and manual copy actions." },
  { kind: "todos", title: "To-dos", description: "Active work and authorized skill lists." },
  { kind: "shopping", title: "Shopping list", description: "A list supplied by its owning skill." },
  { kind: "reminders", title: "Reminders", description: "Native reminder schedules and their controls." },
  { kind: "appointments", title: "Appointments", description: "Appointments supplied by a connected skill." },
  { kind: "questions", title: "Pending questions", description: "Questions waiting for your input." },
  { kind: "agenda", title: "Agenda", description: "Month, Week and Today views of your authorized scheduled work and history." },
] as const;
export const knownWidget = (kind: string) => kind === "skill" || WIDGET_CATALOG.some((entry) => entry.kind === kind);
export const defaultDrawer = (): DrawerState => ({ open: false, widgets: WIDGET_CATALOG.filter(({ kind }) => ["questions", "agenda"].includes(kind)).map(({ kind }) => ({ id: kind, kind, mode: "pinned" })), dismissed: [] });
export const drawerStorageKey = (member: string) => `alphonse.desktop.atHand.v1.${encodeURIComponent(member)}`;
export function readDrawer(storage: Pick<Storage, "getItem">, member: string): DrawerState {
  if (!member) return defaultDrawer();
  try {
    const raw = storage.getItem(drawerStorageKey(member));
    if (!raw) return defaultDrawer();
    const value = JSON.parse(raw);
    if (!Array.isArray(value.widgets) || !Array.isArray(value.dismissed)) return defaultDrawer();
    const ids = new Set<string>();
    const widgets: WidgetPlacement[] = [];
    for (const entry of value.widgets) {
      if (!entry || typeof entry.id !== "string" || !entry.id || typeof entry.kind !== "string" || !["contextual", "pinned", "locked"].includes(entry.mode) || ids.has(entry.id)) continue;
      ids.add(entry.id);
      widgets.push({ id: entry.id, kind: entry.kind, mode: entry.mode, ...(Number.isFinite(entry.expiresAt) ? { expiresAt: entry.expiresAt } : {}) });
    }
    return { open: value.open === true, widgets, dismissed: [...new Set<string>(value.dismissed.filter((id: unknown) => typeof id === "string"))] };
  } catch { return defaultDrawer(); }
}
export function saveDrawer(storage: Pick<Storage, "setItem">, member: string, state: DrawerState): boolean {
  if (!member) return false;
  try { storage.setItem(drawerStorageKey(member), JSON.stringify(state)); return true; } catch { return false; }
}
export type DrawerAction =
  | { type: "visibility"; open: boolean }
  | { type: "add"; kind: string }
  | { type: "mode"; id: string; mode: WidgetMode; expiresAt?: number }
  | { type: "remove"; id: string }
  | { type: "move"; id: string; direction: -1 | 1 };
/** Explicit member actions. Locked positions cannot be crossed or removed until unlocked. */
export function changeDrawer(state: DrawerState, action: DrawerAction): DrawerState {
  if (action.type === "visibility") return { ...state, open: action.open };
  if (action.type === "add") {
    if (!knownWidget(action.kind) || state.widgets.some((widget) => widget.id === action.kind)) return state;
    return { ...state, widgets: [...state.widgets, { id: action.kind, kind: action.kind, mode: "pinned" }], dismissed: state.dismissed.filter((id) => id !== action.kind) };
  }
  const index = state.widgets.findIndex((widget) => widget.id === action.id);
  if (index < 0) return state;
  const widget = state.widgets[index];
  if (action.type === "mode") {
    return { ...state, widgets: state.widgets.map((entry) => entry.id === action.id ? { ...entry, mode: action.mode, ...(action.expiresAt !== undefined ? { expiresAt: action.expiresAt } : {}) } : entry) };
  }
  if (widget.mode === "locked") return state;
  if (action.type === "remove") {
    return { ...state, widgets: state.widgets.filter((entry) => entry.id !== action.id), dismissed: [...new Set([...state.dismissed, action.id])] };
  }
  const target = index + action.direction;
  if (target < 0 || target >= state.widgets.length || state.widgets[target].mode === "locked") return state;
  const widgets = [...state.widgets];
  [widgets[index], widgets[target]] = [widgets[target], widgets[index]];
  return { ...state, widgets };
}
/** #69 supplies authorized metadata here. Automatic updates only append new identities;
 * they cannot change saved modes/order, remove widgets, reopen the drawer or undo dismissal.
 * Contextual expiry is rendered as a placeholder until explicit dismissal; pins/locks do not expire.
 */
export function reconcileContext(state: DrawerState, candidates: ContextWidget[], now: number): DrawerState {
  const ids = new Set([...state.widgets.map((widget) => widget.id), ...state.dismissed]);
  const widgets = [...state.widgets];
  for (const candidate of candidates) {
    if (!candidate.id || ids.has(candidate.id) || !knownWidget(candidate.kind) || !Number.isFinite(candidate.expiresAt) || candidate.expiresAt <= now) continue;
    ids.add(candidate.id);
    widgets.push({ id: candidate.id, kind: candidate.kind, expiresAt: candidate.expiresAt, mode: "contextual" });
  }
  return widgets.length === state.widgets.length ? state : { ...state, widgets };
}
export function widgetAvailability(widget: WidgetPlacement, member: string, connected: boolean, now: number): "loading" | "unavailable" | "expired" | "ready" {
  if (!member) return "loading";
  if (!knownWidget(widget.kind)) return "unavailable";
  if (widget.mode === "contextual" && (!widget.expiresAt || widget.expiresAt <= now)) return "expired";
  return connected ? "ready" : "unavailable";
}
