// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { beforeEach, describe, expect, it, vi } from "vitest";

const service = vi.hoisted(() => ({ request: vi.fn(), recoveryFails: false, pollFails: false, actionable: false, recoveryWait: null as Promise<void> | null }));
vi.mock("./api", () => ({
  ensureDaemon: async () => {}, playAlertSound: async () => {}, showInFinder: async () => {}, stopDaemon: async () => {},
  daemonRequest: async (method: string, params: Record<string, unknown> = {}) => {
    service.request(method, params);
    if (method === "current_user") return { onboarded: true, workspace_configured: true, user: { user_id: "fixture", display_name: "Fixture", role: "admin", is_active: true } };
    if (method === "desktop_poll" && service.pollFails) throw new Error("disconnected");
    if (method === "desktop_poll") return { next_sequence: 0, next_ui_sequence: 0, daemon_id: "fixture", questions: [], events: [], deliveries: [], ui_events: [], legacy_notification_review: { owner_scoped: 1, ambiguous: 1, unassigned: 1 }, status: { queue: { ready: 0, processing: 0 }, activity: { state: "idle" }, active_work: {} } };
    if (method === "experimental_owner_status") return { processing_ready: true, owner_only: true, paused_legacy_requests: 4 };
    if (method === "experimental_owner_recovery_list") {
      if (service.recoveryWait) await service.recoveryWait;
      if (service.recoveryFails) throw new Error("unavailable");
      return { items: [{ message_id: "a".repeat(32), prompt: "Check the fixture device", status: service.actionable ? "unavailable" : "action_unconfirmed", can_abandon: service.actionable }] };
    }
    if (method === "legacy_notification_list") return { items: [] };
    if (method === "desktop_conversation_history") return { messages: [] };
    if (method === "owner_scheduled_results_list") return { results: [] };
    if (method === "timezone_settings") return { timezone: "UTC" };
    if (method === "manageable_projects") return { projects: [] };
    if (method.startsWith("experimental_owner_")) return { items: [] };
    return {};
  },
}));
import App from "./App";
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const flush = () => new Promise(resolve => setTimeout(resolve, 30));
async function mountAttention() {
  const el = document.createElement("div"); document.body.appendChild(el); const root = createRoot(el);
  await act(async () => { root.render(<App />); await flush(); });
  await act(async () => { await flush(); });
  const button = [...el.querySelectorAll<HTMLButtonElement>("button")].find(b => b.title === "Needs your attention")!;
  await act(async () => { button.click(); await flush(); });
  return { el, root };
}

describe("attention workspace integration", () => {
  beforeEach(() => {
    const data = new Map<string, string>();
    Object.defineProperty(window, "localStorage", { configurable: true, value: { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => data.set(key, String(value)), removeItem: (key: string) => data.delete(key) } });
    window.matchMedia = vi.fn().mockReturnValue({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() });
    HTMLElement.prototype.scrollTo = vi.fn(); service.request.mockReset(); service.recoveryFails = false; service.pollFails = false; service.actionable = false; service.recoveryWait = null;
  });
  it("separates questions, notification review, paused work, and uncertain requests without running recovery actions", async () => {
    const { el, root } = await mountAttention();
    const page = el.querySelector<HTMLElement>('[aria-label="Needs your attention workspace"]')!;
    expect(page.textContent).not.toContain("No pending questions");
    expect(page.textContent).toContain("Notifications to review");
    expect(page.textContent).toContain("3 earlier notifications need review");
    expect(page.textContent).toContain("4 earlier requests are paused");
    expect(page.textContent).toContain("An action may have happened; its result is unknown");
    expect([...page.querySelectorAll('details')].find(detail => detail.querySelector('summary')?.textContent === 'Earlier work records and technical details')?.hasAttribute('open')).toBe(false);
    expect(service.request.mock.calls.some(([method]) => ["legacy_state_inventory", "legacy_state_items", "legacy_notification_list", "experimental_owner_abandon"].includes(method))).toBe(false);
    await act(async () => root.unmount()); el.remove();
  });
  it("does not present a failed recovery check as an empty request list", async () => {
    service.recoveryFails = true;
    const { el, root } = await mountAttention();
    const page = el.querySelector<HTMLElement>('[aria-label="Needs your attention workspace"]')!;
    expect(page.textContent).toContain("Earlier requests could not be checked");
    expect(page.textContent).not.toContain("No unfinished requests were found");
    const refresh = [...page.querySelectorAll("button")].find(button => button.textContent === "Check requests again")!;
    service.recoveryFails = false;
    await act(async () => { refresh.click(); await flush(); });
    expect(page.textContent).toContain("An action may have happened; its result is unknown");
    expect(service.request.mock.calls.some(([method]) => method === "experimental_owner_abandon")).toBe(false);
    await act(async () => root.unmount()); el.remove();
  });
  it("withholds stale actions after reconnect until a fresh recovery check succeeds", async () => {
    service.actionable = true;
    const { el, root } = await mountAttention();
    const page = el.querySelector<HTMLElement>('[aria-label="Needs your attention workspace"]')!;
    const stop = () => page.querySelector<HTMLButtonElement>('.owner-request-card-actions button');
    expect(stop()?.textContent).toBe("Stop request");
    service.pollFails = true;
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 950)); });
    expect(page.textContent).toContain("Disconnected. Reconnect to check earlier requests.");
    expect(stop()).toBeNull();
    let confirmRecovery!: () => void;
    service.recoveryWait = new Promise<void>(resolve => { confirmRecovery = resolve; });
    service.pollFails = false;
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 950)); });
    expect(page.textContent).toContain("Checking earlier requests; any list shown is an older snapshot.");
    expect(stop()).toBeNull();
    service.recoveryWait = null;
    await act(async () => { confirmRecovery(); await flush(); });
    expect(stop()?.textContent).toBe("Stop request");
    expect(service.request.mock.calls.some(([method]) => method === "experimental_owner_abandon")).toBe(false);
    await act(async () => root.unmount()); el.remove();
  });

});
