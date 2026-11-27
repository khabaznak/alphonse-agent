import { mockIPC, mockWindows } from "/desktop/node_modules/@tauri-apps/api/mocks.js";
import surface from "./private-gmail.json";

const stateKey = "alphonse.synthetic-window-blink.v1";
const restored = new URLSearchParams(location.search).get("history") === "restore";
const state = JSON.parse(sessionStorage.getItem(stateKey) || '{"boot":0,"tasks":[],"calls":[],"unseen":false}');
state.boot += 1;
const persist = () => sessionStorage.setItem(stateKey, JSON.stringify(state));
const idFor = (n) => String(n + 1).repeat(32);
const receipt = "Gmail Inbox matches were displayed privately in Desktop.";
const promptFor = (n) => "Synthetic request " + (n + 1) + " for a private Inbox card.";
const titleFor = (n) => "Synthetic Inbox card " + (n + 1);
const recovery = "Needs your attention";
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const timeline = () => document.querySelector("#conversation-timeline");
const working = () => Boolean(document.querySelector(".conversation-activity"));
const user = { user_id: "fixture", display_name: "Synthetic Workshop", role: "admin", is_active: true };
let draftsChecked = false;
const record = (method, id) => { state.calls.push({ method, id, boot: state.boot }); persist(); };

if (state.boot > 1) {
  // The independent daemon fixture finishes a held job during navigation.
  // Keep the request so the reloaded App sees an outstanding ready answer.
  for (const task of state.tasks) if (task.status === "pending") task.status = "ready";
}
persist();

function card(task) {
  const value = structuredClone(surface);
  value[0].createSurface.surfaceId = "private-gmail:" + task.id;
  value[1].updateComponents.surfaceId = "private-gmail:" + task.id;
  const title = value[1].updateComponents.components.find((item) => item.id === "title");
  title.text = titleFor(task.index);
  return value;
}

function history() {
  return state.tasks.flatMap((task) => [
    { id: task.id, role: "user", content: task.prompt, source: "desktop", created_at: task.start, sequence: task.index * 2 + 1 },
    ...(task.status === "client_acknowledged" ? [{
      id: "owner:" + task.id, role: "assistant", content: receipt, source: "desktop", created_at: task.end,
      sequence: task.index * 2 + 2, ...(restored ? { private_card: { provider: "gmail", envelopes: card(task) } } : {}),
    }] : []),
  ]);
}

mockWindows("main");
mockIPC(async (command, payload) => {
  if (command === "desktop_diagnostic_mode") return "normal";
  if (command === "desktop_diagnostic_project_id") return "";
  if (command === "ensure_daemon") return null;
  if (command !== "daemon_request") return null;
  const { method, params = {} } = payload;
  const task = state.tasks.find((item) => item.id === params.message_id);
  switch (method) {
    case "current_user": record(method); return { onboarded: true, workspace_configured: true, user };
    case "timezone_settings": return { timezone: "UTC" };
    case "manageable_projects": return { projects: [] };
    case "desktop_poll":
      return {
        daemon_id: "window-blink-fixture", next_sequence: 0, next_ui_sequence: 0,
        status: { queue: { ready: 0, processing: 0 }, activity: { state: "idle" }, active_work: {} },
        questions: [], events: [], deliveries: [], ui_events: [], conversation_has_unseen_messages: state.unseen,
      };
    case "desktop_conversation_history":
      record(method);
      return { messages: history(), ...(restored && params.include_private_cards === true ? { private_cards_checked: true } : {}) };
    case "desktop_mark_conversation_seen": state.unseen = false; persist(); return {};
    case "experimental_owner_composer_list": draftsChecked = true; return { items: [] };
    case "experimental_owner_status":
      if (!params.message_id) return { processing_ready: true, owner_only: true, paused_legacy_requests: 0 };
      return { status: task?.status || "unavailable" };
    case "experimental_owner_ingress": {
      const index = state.tasks.length;
      if (params.prompt !== promptFor(index) || index > 2) throw new Error("Unexpected synthetic submission");
      const item = { id: idFor(index), index, prompt: params.prompt, start: new Date().toISOString(), status: index < 2 ? "ready" : "pending" };
      state.tasks.push(item); record(method, item.id);
      return { message_id: item.id };
    }
    case "experimental_owner_take_draft":
      if (task?.status !== "ready") throw new Error("Synthetic answer is not ready");
      task.status = "started_unconfirmed"; record(method, task.id);
      return { message_id: task.id, kind: "answer", delivery_status: task.status, acknowledgement_token: "fixture-token",
        message: receipt, private_gmail: true, private_a2ui: card(task) };
    case "experimental_owner_ack_draft":
      task.status = "client_acknowledged"; task.end = new Date().toISOString(); state.unseen = true;
      record(method, task.id); return { status: task.status };
    case "experimental_owner_recovery_list":
      return { items: state.tasks.filter((item) => item.status === "ready").map((item) => ({
        message_id: item.id, prompt: item.prompt, status: item.status, can_abandon: false,
      })) };
    case "experimental_owner_work_log":
      return { status: "running", events: [], failure_code: null, notice: "Synthetic fixture activity." };
    case "owner_scheduled_results_list": return { results: [] };
    default: return { items: [] };
  }
}, { shouldMockEvents: true });

async function waitFor(predicate, label) {
  const deadline = performance.now() + 8500;
  while (!predicate()) {
    if (performance.now() > deadline) throw new Error("Waiting for " + label + ": " + JSON.stringify(metrics()));
    await sleep(30);
  }
}

function metrics() {
  const text = timeline()?.textContent || "";
  return {
    boot: state.boot,
    sidebar: Boolean(document.querySelector('button[title="Conversation"]')),
    working: working(),
    cards: [0, 1, 2].filter((n) => text.includes(titleFor(n))).map((n) => n + 1),
    receiptOnlyRows: (text.match(/Gmail Inbox matches were displayed privately in Desktop\./g) || []).length,
    tasks: state.tasks.map(({ index, status }) => ({ index, status })),
    pickups: state.calls.filter((call) => call.method === "experimental_owner_take_draft").map(({ id, boot }) => ({ id, boot })),
    submissions: state.calls.filter((call) => call.method === "experimental_owner_ingress").length,
    errorText: document.querySelector(".error")?.textContent || null,
  };
}

async function prepare() {
  await waitFor(() => draftsChecked, "owner draft status");
  await sleep(180);
  for (let index = 0; index < 3; index++) {
    await waitFor(() => document.querySelector('button[aria-label="Send"]:not(:disabled)'), "composer");
    const textarea = document.querySelector('textarea[aria-label="Message Alphonse"]');
    textarea.focus();
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set.call(textarea, promptFor(index));
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    await new Promise(requestAnimationFrame);
    textarea.form.requestSubmit();
    if (index < 2) {
      await waitFor(() => state.tasks[index]?.status === "client_acknowledged" && metrics().cards.includes(index + 1),
        "acknowledged private card " + (index + 1));
      await sleep(120);
    } else await waitFor(() => state.tasks[index]?.status === "pending" && working(), "live third reply");
  }
  return metrics();
}

async function settleAfterReload() {
  await waitFor(() => metrics().sidebar && timeline()?.querySelectorAll(".message").length >= 5, "restored timeline");
  await sleep(650);
  const beforeRecovery = metrics();
  document.querySelector('button[title="' + recovery + '"]').click();
  await waitFor(() => [...document.querySelectorAll('button')].some((button) => button.textContent === "Show answer"), "Show answer");
  [...document.querySelectorAll('button')].find((button) => button.textContent === "Show answer").click();
  await waitFor(() => metrics().cards.includes(3) && state.tasks[2].status === "client_acknowledged", "manual recovery");
  return { beforeRecovery, afterRecovery: metrics() };
}

window.reloadProbe = {
  metrics, prepare, settleAfterReload,
  async finishNormally() {
    state.tasks[2].status = "ready"; persist();
    await waitFor(() => metrics().cards.includes(3) && state.tasks[2].status === "client_acknowledged", "normal third reply");
    return metrics();
  },
  start(method) {
    this.result = null; this.error = null;
    void this[method]().then((result) => { this.result = result; }, (error) => { this.error = String(error.message); });
    return true;
  },
};
await import("/desktop/src/main.tsx");
