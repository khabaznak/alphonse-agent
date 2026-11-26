// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const service = vi.hoisted(() => ({ ingress: vi.fn(), followup: vi.fn(), steer: vi.fn(), followupList: vi.fn(), history: vi.fn(), recoveryList: vi.fn(), drafts: vi.fn(), ownerStatus: vi.fn(), takeDraft: vi.fn(), optionalManagement: vi.fn() }));
vi.mock("./api", () => ({
  ensureDaemon: async () => {}, playAlertSound: async () => {}, showInFinder: async () => {}, stopDaemon: async () => {},
  optionalManagement: service.optionalManagement,
  daemonRequest: async (method: string, params: Record<string, unknown> = {}) => {
    if (method === "current_user") return { onboarded: true, workspace_configured: true, user: { user_id: "fixture", display_name: "Fixture", role: "admin", is_active: true } };
    if (method === "desktop_poll") return { next_sequence: 0, next_ui_sequence: 0, daemon_id: "fixture", questions: [], events: [], deliveries: [], ui_events: [], status: { queue: { ready: 0, processing: 0 }, activity: { state: "idle" }, active_work: {} } };
    if (method === "experimental_owner_status") return params.message_id ? service.ownerStatus(params.message_id) : { processing_ready: true, owner_only: true, paused_legacy_requests: 0 };
    if (method === "experimental_owner_ingress") return service.ingress(params);
    if (method === "experimental_owner_take_draft") return service.takeDraft(params.message_id);
    if (method === "experimental_owner_ack_draft") return { status: "client_acknowledged" };
    if (method === "experimental_owner_followup_list") return service.followupList();
    if (method === "experimental_owner_recovery_list") return service.recoveryList();
    if (method === "experimental_owner_composer_list") return service.drafts();
    if (method === "experimental_owner_composer_save") return { id: params.id, status: "saved" };
    if (method === "experimental_owner_followup_add") return service.followup(params);
    if (method === "experimental_owner_steer_add") return service.steer(params);
    if (method === "desktop_conversation_history") return service.history();
    if (method === "owner_scheduled_results_list") return { results: [] };
    if (method === "timezone_settings") return { timezone: "UTC" };
    if (method === "manageable_projects") return { projects: [] };
    if (method.startsWith("experimental_owner_")) return { items: [] };
    return {};
  },
}));
import App from "./App";
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const flush = () => new Promise((resolve) => setTimeout(resolve, 20));
function typeValue(textarea: HTMLTextAreaElement, value: string) {
  Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(textarea, value);
  textarea.dispatchEvent(new Event("input", { bubbles: true }));
}

describe("confirmed conversation sends", () => {
  beforeEach(() => {
    const data = new Map<string, string>();
    const storage = { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => data.set(key, String(value)), removeItem: (key: string) => data.delete(key), clear: () => data.clear() };
    Object.defineProperty(window, "localStorage", { configurable: true, value: storage });
    window.matchMedia = vi.fn().mockReturnValue({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() });
    HTMLElement.prototype.scrollTo = vi.fn();
    service.ingress.mockReset(); service.followup.mockReset(); service.steer.mockReset(); service.followupList.mockReset();
    service.followupList.mockResolvedValue({ items: [] }); service.history.mockReset(); service.history.mockResolvedValue({ messages: [] }); service.recoveryList.mockReset(); service.recoveryList.mockResolvedValue({ items: [] }); service.drafts.mockReset(); service.drafts.mockResolvedValue({ items: [] });
    service.followup.mockResolvedValue({ status: "queued" }); service.steer.mockResolvedValue({ status: "accepted" });
    service.ownerStatus.mockReset(); service.ownerStatus.mockResolvedValue({ status: "pending" });
    service.takeDraft.mockReset(); service.optionalManagement.mockReset();
  });
  afterEach(() => vi.restoreAllMocks());
  it("focuses the composer and Outlook heading when handing off in both directions", async () => {
    service.optionalManagement.mockResolvedValue({
      actor_user_id: "fixture", owner_outlook_inbox_setup: true,
      providers: [{ provider: "microsoft", capabilities: ["outlook_inbox_unread"] }],
      connections: [{ connection_id: "hotmail", provider: "microsoft", display_name: "Hotmail", revision: 1,
        status: "connected", capabilities: ["outlook_inbox_unread"] }],
      grants: [{ grant_id: "inbox", connection_id: "hotmail", grantor_user_id: "fixture", beneficiary_user_id: "fixture",
        project_id: null, operations: ["outlook_inbox_unread"], resources: ["outlook:inbox"], recipient_user_ids: ["fixture"],
        expires_at: "2099-01-01T00:00:00Z", revoked_at: "", revision: 1 }],
      bindings: [], members: [{ user_id: "fixture", display_name: "Fixture" }], projects: [], skills: [], sources: [],
    });
    const element = document.body.appendChild(document.createElement("div"));
    const root = createRoot(element);
    try {
      await act(async () => { root.render(<App />); await flush(); });
      const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
      await act(async () => typeValue(textarea, "Existing draft"));
      await act(async () => element.querySelector<HTMLButtonElement>('button[title="Settings"]')!.click());
      await act(async () => element.querySelector<HTMLButtonElement>('button[title="Integrations"]')!.click());
      await act(async () => element.querySelector<HTMLButtonElement>('button[aria-label="Manage Outlook / Hotmail"]')!.click());
      const open = [...element.querySelectorAll("button")].find(button => button.textContent === "Open Conversation")!;
      open.focus();
      // Keyboard input belongs to the composer as soon as navigation commits.
      // No animation frame should be needed to override the workspace focus.
      vi.spyOn(window, "requestAnimationFrame").mockImplementation(() => 1);
      await act(async () => open.click());
      expect(document.activeElement).toBe(textarea);
      expect(textarea.value).toBe("Existing draft");
      expect(service.ingress).not.toHaveBeenCalled();

      const messageId = "a".repeat(32);
      service.ingress.mockResolvedValue({ message_id: messageId });
      service.ownerStatus.mockResolvedValue({ status: "ready" });
      service.takeDraft.mockResolvedValue({ message_id: messageId, message: "Refresh Microsoft access.", kind: "answer",
        outlook_inbox_status: "refresh_required", acknowledgement_token: "fixture-token", delivery_status: "started_unconfirmed" });
      await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
      const recover = [...element.querySelectorAll("button")].find(button => button.textContent === "Open Outlook / Hotmail settings")!;
      recover.focus();
      await act(async () => recover.click());
      expect(document.activeElement).toBe(element.querySelector("[data-outlook-settings-heading]"));
      expect(document.activeElement?.textContent).toBe("Outlook / Hotmail");
    } finally {
      await act(async () => root.unmount());
      element.remove();
    }
  });
  it("clears the exact whitespace-padded draft after confirmed send", async () => {
    let confirm!: (value: unknown) => void;
    service.ingress.mockImplementation(() => new Promise((resolve) => { confirm = resolve; }));
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); });
    await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "  A message with space  \n"));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(service.ingress).toHaveBeenCalledWith({ prompt: "A message with space" });
    await act(async () => { confirm({ message_id: "a".repeat(32) }); await flush(); });
    expect(textarea.value).toBe("");
    expect(element.textContent).not.toContain("The active task changed while you were writing.");
    await act(async () => root.unmount()); element.remove();
  });
  it("preserves a draft edited while its earlier submission is in flight", async () => {
    let confirm!: (value: unknown) => void;
    service.ingress.mockImplementation(() => new Promise((resolve) => { confirm = resolve; }));
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); });
    await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "  original  "));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    await act(async () => typeValue(textarea, "new text while sending"));
    await act(async () => { confirm({ message_id: "a".repeat(32) }); await flush(); });
    expect(textarea.value).toBe("new text while sending");
    await act(async () => root.unmount()); element.remove();
  });
  it("preserves the draft when submission is not confirmed", async () => {
    service.ingress.mockRejectedValue(new Error("network unavailable"));
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); });
    await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "  keep this draft  "));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(textarea.value).toBe("  keep this draft  ");
    expect(element.textContent).toContain("Message submission could not be confirmed");
    await act(async () => root.unmount()); element.remove();
  });
  it.each(["queue", "steer"] as const)("clears a whitespace-padded %s after acceptance", async (mode) => {
    service.ingress.mockResolvedValue({ message_id: "a".repeat(32) });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); });
    await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "first request"));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(textarea.value).toBe("");
    await act(async () => typeValue(textarea, "  later details  \n"));
    await act(async () => {
      if (mode === "queue") textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
      else element.querySelector<HTMLButtonElement>(".conversation-send-alternative")!.click();
      await flush();
    });
    const sender = mode === "queue" ? service.followup : service.steer;
    expect(sender).toHaveBeenCalledWith(expect.objectContaining({ prompt: "later details", active_message_id: "a".repeat(32) }));
    expect(textarea.value).toBe("");
    await act(async () => root.unmount()); element.remove();
  });

  it.each([false, true])("reconciles an uncertain queued send without erasing later edits: %s", async (edited) => {
    service.ingress.mockResolvedValue({ message_id: "a".repeat(32) });
    service.followup.mockRejectedValue(new Error("reply lost"));
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); });
    await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "first request"));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    await act(async () => typeValue(textarea, "  queued message  "));
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(textarea.value).toBe("  queued message  ");
    const id = service.followup.mock.calls[0][0].request_id;
    if (edited) await act(async () => typeValue(textarea, "newer draft"));
    service.followupList.mockResolvedValue({ items: [{ id, prompt: "queued message", status: "queued", can_remove: true }] });
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 950)); });
    expect(textarea.value).toBe(edited ? "newer draft" : "");
    expect(service.followup).toHaveBeenCalledTimes(1);
    await act(async () => root.unmount()); element.remove();
  });

  it("binds the selected one of two identical failed requests, preserving and clearing draft context safely", async () => {
    const first = "c".repeat(32); const second = "d".repeat(32);
    window.localStorage.setItem("alphonse.desktop.conversationFailures.fixture", JSON.stringify([first, second].map((messageId) => ({ messageId, failureCode: "time_limit", dismissed: false }))));
    service.history.mockResolvedValue({ messages: [first, second].map((id, index) => ({ id, role: "user", source: "desktop", content: "Same request", created_at: new Date(Date.now() + index).toISOString() })) });
    service.ingress.mockResolvedValue({ message_id: "a".repeat(32) });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); }); await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => typeValue(textarea, "Existing draft"));
    const buttons = [...element.querySelectorAll("button")].filter((button) => button.textContent === "Reference this failed request in draft");
    expect(buttons).toHaveLength(2);
    await act(async () => { buttons[1].click(); await flush(); });
    expect(textarea.value).toContain("Existing draft"); expect(element.textContent).toContain(`ID: ${second}`);
    expect(service.ingress).not.toHaveBeenCalled();
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(service.ingress).toHaveBeenCalledWith(expect.objectContaining({ failed_request_message_id: second }));
    await act(async () => root.unmount()); element.remove();
  });

  it("lets the user remove a failed-request reference without erasing the draft", async () => {
    const id = "f".repeat(32); window.localStorage.setItem("alphonse.desktop.conversationFailures.fixture", JSON.stringify([{ messageId: id, failureCode: "time_limit", dismissed: false }]));
    service.history.mockResolvedValue({ messages: [{ id, role: "user", source: "desktop", content: "Previous request", created_at: new Date().toISOString() }] });
    service.ingress.mockResolvedValue({ message_id: "a".repeat(32) });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); }); await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => { [...element.querySelectorAll("button")].find((item) => item.textContent === "Reference this failed request in draft")!.click(); await flush(); });
    const savedDraft = textarea.value;
    await act(async () => { [...element.querySelectorAll("button")].find((item) => item.textContent === "Remove reference")!.click(); });
    expect(textarea.value).toBe(savedDraft); expect(document.activeElement).toBe(textarea); expect(element.textContent).not.toContain(`ID: ${id}`);
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(service.ingress).toHaveBeenCalledWith({ prompt: savedDraft.trim() });
    await act(async () => root.unmount()); element.remove();
  });

  it("preserves the draft and allows removal when a failed reference was rejected before enqueue", async () => {
    const id = "1".repeat(32); window.localStorage.setItem("alphonse.desktop.conversationFailures.fixture", JSON.stringify([{ messageId: id, failureCode: "time_limit", dismissed: false }]));
    service.history.mockResolvedValue({ messages: [{ id, role: "user", source: "desktop", content: "Previous request", created_at: new Date().toISOString() }] });
    service.ingress.mockRejectedValueOnce(new Error("ValueError: owner_failure_reference_unavailable")).mockResolvedValueOnce({ message_id: "a".repeat(32) });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); }); await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    await act(async () => { [...element.querySelectorAll("button")].find((item) => item.textContent === "Reference this failed request in draft")!.click(); await flush(); });
    const draft = textarea.value;
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(textarea.value).toBe(draft); expect(element.textContent).toContain("Your draft was not sent");
    expect(element.textContent).not.toContain("Message submission could not be confirmed");
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 30)); });
    expect(document.activeElement?.textContent).toBe("Remove reference");
    await act(async () => { [...element.querySelectorAll("button")].find((item) => item.textContent === "Remove reference")!.click(); });
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(service.ingress).toHaveBeenLastCalledWith({ prompt: draft.trim() });
    await act(async () => root.unmount()); element.remove();
  });

  it("removes a failed-request reference if its quoted text is deleted", async () => {
    const id = "e".repeat(32); window.localStorage.setItem("alphonse.desktop.conversationFailures.fixture", JSON.stringify([{ messageId: id, failureCode: "time_limit", dismissed: false }]));
    service.history.mockResolvedValue({ messages: [{ id, role: "user", source: "desktop", content: "Same request", created_at: new Date().toISOString() }] });
    service.ingress.mockResolvedValue({ message_id: "a".repeat(32) });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); }); await act(async () => { await flush(); });
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    const button = [...element.querySelectorAll("button")].find((item) => item.textContent === "Reference this failed request in draft")!;
    await act(async () => { button.click(); await flush(); });
    await act(async () => typeValue(textarea, "A different question"));
    expect(element.textContent).not.toContain(`ID: ${id}`);
    await act(async () => { textarea.closest("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    expect(service.ingress).toHaveBeenCalledWith({ prompt: "A different question" });
    await act(async () => root.unmount()); element.remove();
  });

  it("refreshes a recovery item before showing it and focuses the updated request", async () => {
    const id = "2".repeat(32); window.localStorage.setItem("alphonse.desktop.conversationFailures.fixture", JSON.stringify([{ messageId: id, failureCode: "time_limit", dismissed: false }]));
    service.history.mockResolvedValue({ messages: [{ id, role: "user", source: "desktop", content: "Previous request", created_at: new Date().toISOString() }] });
    const pending = { message_id: id, prompt: "Previous request", status: "pending", can_abandon: false };
    let resolve!: (value: unknown) => void;
    service.recoveryList.mockImplementation(() => new Promise((done) => { resolve = done; }));
    // Keep the first discovery immediate, then defer the review refresh.
    service.recoveryList.mockResolvedValueOnce({ items: [pending] });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); }); await act(async () => { await flush(); });
    const button = [...element.querySelectorAll("button")].find((item) => item.textContent === "Review request status")!;
    expect(button).toBeTruthy();
    await act(async () => { button.click(); await flush(); });
    const panel = element.querySelector('[aria-label="Needs your attention workspace"]')!;
    expect(panel.textContent).toContain("Checking request status");
    expect(panel.textContent).not.toContain("Waiting to be processed");
    await act(async () => { resolve({ items: [{ ...pending, status: "failed" }] }); await flush(); });
    await act(async () => { await new Promise((done) => setTimeout(done, 30)); });
    expect(panel.textContent).toContain("Request needs attention");
    expect(document.activeElement?.id).toBe(`owner-recovery-${id}`);
    await act(async () => root.unmount()); element.remove();
  });

  it("keeps a contextual failure in order between normal messages and restores it from owner history", async () => {
    const id = "b".repeat(32);
    const nextId = "c".repeat(32);
    const request = "¿Podemos retomar el diagnóstico del boiler sintético? Creo que ayer comprobaste si respondía a la red. ¿Qué sigue?";
    const recovery = {
      reply: "**Todavía no puedo confirmar si puedo controlar el boiler.**\n\nBusqué en el historial y no pude verificar el ping de ayer.\n\n1. Identificar el modelo.\n2. Revisar las opciones documentadas.\n3. Si hay una opción compatible, consultar el estado.\n\n**¿Qué marca y modelo tiene?**",
      language: "es-MX",
      activity: { label: "Ver actividad", detail: "Se registró una búsqueda en el historial. El registro visible no indica el resultado." },
    };
    service.ingress.mockResolvedValueOnce({ message_id: id }).mockResolvedValueOnce({ message_id: nextId });
    service.ownerStatus.mockImplementation(async (messageId) => messageId === id
      ? { status: "failed", failure_code: "source_unavailable", failure_recovery: recovery }
      : { status: "pending" });
    const element = document.createElement("div");
    document.body.appendChild(element);
    let root = createRoot(element);
    const open = async () => {
      await act(async () => { root.render(<App />); await flush(); });
      await act(async () => { await flush(); });
    };
    const send = async (content: string) => {
      const editor = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
      await act(async () => typeValue(editor, content));
      await act(async () => { editor.form!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); await flush(); });
    };
    const inlineReply = () => element.querySelector<HTMLElement>(".conversation-failure-recovery")!;
    await open();
    await send(request);
    expect(inlineReply().previousElementSibling?.textContent).toContain(request);
    expect(inlineReply().lang).toBe("es-MX");
    expect(inlineReply().querySelectorAll("ol > li")).toHaveLength(3);
    expect(inlineReply().querySelector("details")?.open).toBe(false);
    expect(inlineReply().querySelector("summary")?.textContent).toBe("Ver actividad");
    await act(async () => inlineReply().querySelector("summary")!.click());
    expect(inlineReply().querySelector("details")?.open).toBe(true);
    expect(inlineReply().textContent).toContain(recovery.activity.detail);
    expect(element.querySelector(".conversation-failure")).toBeNull();
    expect(window.localStorage.getItem("alphonse.desktop.conversationFailures.fixture")).not.toContain("boiler");
    await send("Lo controlo desde la app del fabricante.");
    expect(service.ingress.mock.calls).toEqual([[{ prompt: request }], [{ prompt: "Lo controlo desde la app del fabricante." }]]);
    expect(inlineReply().nextElementSibling?.textContent).toContain("Lo controlo desde la app del fabricante.");
    await act(async () => root.unmount());

    window.localStorage.removeItem("alphonse.desktop.conversationFailures.fixture");
    service.history.mockResolvedValue({ messages: [
      { id, content: request, role: "user", source: "desktop", created_at: new Date().toISOString(), failure_recovery: recovery },
    ] });
    root = createRoot(element);
    await open();
    expect(element.querySelectorAll(".conversation-failure-recovery")).toHaveLength(1);
    expect(inlineReply().querySelector("details")?.open).toBe(false);
    await act(async () => root.unmount());
    element.remove();
  });

  it("keeps saved-draft controls accessible separately from the message editor", async () => {
    service.drafts.mockResolvedValue({ items: Array.from({ length: 6 }, (_, index) => ({ id: String(index).repeat(32), active_message_id: "a".repeat(32), prompt: `Saved draft ${index}`, resolution: "not_submitted" })) });
    const element = document.createElement("div"); document.body.appendChild(element); const root = createRoot(element);
    await act(async () => { root.render(<App />); await flush(); }); await act(async () => { await flush(); });
    const context = element.querySelector<HTMLElement>('[aria-label="Request context"]')!;
    const textarea = element.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message Alphonse"]')!;
    expect(context.tabIndex).toBe(0); expect(context.textContent).toContain("Saved draft 5");
    expect(context.contains(textarea)).toBe(false);
    expect([...context.querySelectorAll("button")].filter((button) => button.textContent === "Restore draft")).toHaveLength(6);
    await act(async () => root.unmount()); element.remove();
  });

});
