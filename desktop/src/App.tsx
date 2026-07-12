import { FormEvent, KeyboardEvent, ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { daemonRequest, ensureDaemon, stopDaemon } from "./api";
import { matchingCommands } from "./commands";
import { A2uiSurfaceView, applyA2uiEvent, DESKTOP_CATALOG_ID } from "./a2ui";
import { agentStateLabel, capdActivityLabel, projectKey } from "./layoutState";
import type { ActivityEvent, AgentDocument, ChatMessage, InferenceSettings, Project, Question } from "./types";

type Modal = "projects" | "project-context" | "integrations" | "model" | "agent-config" | "settings" | "users" | "onboarding" | null;
type PollResponse = {
  events: ActivityEvent[];
  next_sequence: number;
  ui_events?: Array<{ sequence?: number; event: { type: string; name?: string; value?: unknown } }>;
  next_ui_sequence?: number;
  deliveries: Array<{ outbox_message_id: string; message: string }>;
  questions: Question[];
  status: { active_work: Record<string, string>; activity: { state?: string } };
};
type HistoryResponse = { messages: ChatMessage[] };
type Provider = { provider_key: string; display_name: string; models: Array<{ model_id: string; display_name: string }> };

export default function App() {
  const clientId = useRef(crypto.randomUUID()).current;
  const sequence = useRef(0);
  const uiSequence = useRef(0);
  const delivered = useRef(new Set<string>());
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([{ id: "welcome", role: "assistant", content: "Alphonse Desktop is connected locally." }]);
  const [messageBuckets, setMessageBuckets] = useState<Record<string, ChatMessage[]>>({});
  const [prompt, setPrompt] = useState("");
  const [connected, setConnected] = useState(false);
  const [activity, setActivity] = useState("idle");
  const [agentState, setAgentState] = useState<"Idle" | "Working" | "Error" | "Disconnected">("Disconnected");
  const [questions, setQuestions] = useState<Question[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [modal, setModal] = useState<Modal>(null);
  const [user, setUser] = useState("");
  const [error, setError] = useState("");
  const [surfaces, setSurfaces] = useState<Record<string, import("./a2ui").A2uiSurface>>({});
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [enterToSend, setEnterToSend] = useState(() => window.localStorage.getItem("alphonse.desktop.enterToSend") !== "false");
  const currentProjectKey = projectKey(project?.project_id);

  const appendMessage = useCallback((message: ChatMessage) => {
    setMessages((current) => {
      const next = [...current, message];
      setMessageBuckets((buckets) => ({ ...buckets, [currentProjectKey]: next }));
      return next;
    });
  }, [currentProjectKey]);

  const poll = useCallback(async () => {
    try {
      const response = await daemonRequest<PollResponse>("desktop_poll", {
        client_id: clientId,
        user,
        after_sequence: sequence.current,
        after_ui_sequence: uiSequence.current,
        client_capabilities: { supportedCatalogIds: [DESKTOP_CATALOG_ID] },
        limit: 50,
      });
      sequence.current = response.next_sequence;
      uiSequence.current = response.next_ui_sequence ?? uiSequence.current;
      setConnected(true);
      setError("");
      setQuestions(response.questions);
      setSurfaces((current) => (response.ui_events || []).reduce((next, item) => applyA2uiEvent(next, item.event), current));
      const latest = response.events.at(-1);
      setActivity(capdActivityLabel(latest, response.status.activity.state || "idle"));
      setAgentState(agentStateLabel(true, false, Object.keys(response.status.active_work || {}).length));
      for (const delivery of response.deliveries) {
        if (delivered.current.has(delivery.outbox_message_id)) continue;
        delivered.current.add(delivery.outbox_message_id);
        appendMessage({ id: delivery.outbox_message_id, role: "assistant", content: delivery.message });
        await daemonRequest("desktop_ack_delivery", { client_id: clientId, outbox_message_id: delivery.outbox_message_id });
      }
    } catch (cause) {
      setConnected(false);
      setError(cause instanceof Error ? cause.message : "Daemon connection lost");
      setActivity("idle");
      setAgentState("Disconnected");
    }
  }, [appendMessage, clientId, user]);

  useEffect(() => {
    ensureDaemon().then(async () => {
      const current = await daemonRequest<{ onboarded: boolean; user: { user_id: string } | null }>("current_user");
      if (!current.onboarded || !current.user) setModal("onboarding"); else setUser(current.user.user_id);
      await poll();
    }).catch((cause: unknown) => setError(cause instanceof Error ? cause.message : "Unable to start daemon"));
    const timer = window.setInterval(() => void poll(), 800);
    return () => window.clearInterval(timer);
  }, [poll]);

  useEffect(() => {
    window.localStorage.setItem("alphonse.desktop.enterToSend", String(enterToSend));
  }, [enterToSend]);

  useEffect(() => {
    const element = timelineRef.current;
    if (!element) return;
    element.scrollTop = element.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 10 * 22 + 24)}px`;
  }, [prompt]);

  const selectProject = useCallback(async (next: Project) => {
    const nextKey = projectKey(next.project_id);
    setMessageBuckets((buckets) => ({ ...buckets, [currentProjectKey]: messages }));
    setProject(next);
    setModal(null);
    setSurfaces({});
    setQuestions([]);
    delivered.current.clear();
    setMessages(messageBuckets[nextKey] || []);
    try {
      const history = await daemonRequest<HistoryResponse>("desktop_conversation_history", { user, project_id: next.project_id, limit: 100 });
      setMessages((current) => current.length > 0 ? current : history.messages);
      setMessageBuckets((buckets) => ({ ...buckets, [nextKey]: buckets[nextKey]?.length ? buckets[nextKey] : history.messages }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Conversation history could not be loaded");
    }
  }, [currentProjectKey, messageBuckets, messages, user]);

  const submitPrompt = async () => {
    const value = prompt.trim();
    if (!value) return;
    setPrompt("");
    if (value.startsWith("/")) {
      await runCommand(value.split(/\s+/, 1)[0]);
      return;
    }
    appendMessage({ id: crypto.randomUUID(), role: "user", content: value });
    try {
      await daemonRequest("queue_message", {
        prompt: value,
        user,
        project_id: project?.project_id || "",
        integration_id: "desktop",
        provider_key: "tui",
        channel_target: user,
      });
      setActivity("doing");
      await poll();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Message could not be queued");
      setAgentState("Error");
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    await submitPrompt();
  };

  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (!enterToSend || event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    void submitPrompt();
  };

  const runCommand = async (command: string) => {
    if (command === "/project") return setModal("projects");
    if (command === "/project-context") return setModal("project-context");
    if (command === "/integrations") return setModal("integrations");
    if (command === "/model" || command === "/model-provider") return setModal("model");
    if (command === "/agent-config") return setModal("agent-config");
    if (command === "/settings") return setModal("settings");
    if (command === "/users") return setModal("users");
    if (command === "/stop") {
      if (window.confirm("Stop the Alphonse daemon?")) await stopDaemon();
      return;
    }
    if (command === "/exit" || command === "/quit") window.close();
  };

  const suggestions = useMemo(() => matchingCommands(prompt), [prompt]);
  const activeSurface = Object.values(surfaces)[0];
  const fallbackQuestion = questions.find((question) => !surfaces[`question:${question.question_id}`]);

  return (
    <main className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="sidebar" aria-label="Alphonse navigation">
        <div className="brand">
          <span className="mark">A</span>
          <span className="brand-name">Alphonse</span>
          <button className="collapse-toggle" type="button" onClick={() => setSidebarCollapsed((value) => !value)} aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}>{sidebarCollapsed ? ">" : "<"}</button>
        </div>
        <button title="Projects" onClick={() => setModal("projects")}><span>Project</span><small>{project?.name || "Home"}</small></button>
        <button title="Integrations" onClick={() => setModal("integrations")}><span>Integrations</span></button>
        <button title="Model" onClick={() => setModal("model")}><span>Model</span></button>
        <button title="Agent configuration" onClick={() => setModal("agent-config")}><span>Agent configuration</span></button>
        <button title="Users" onClick={() => setModal("users")}><span>Users</span></button>
        <button title="Settings" onClick={() => setModal("settings")}><span>Settings</span></button>
      </aside>

      <section className="conversation">
        <header className="topbar">
          <div className="topbar-project"><p className="eyebrow">Project</p><h1>{project?.name || "Home"}</h1></div>
          <StatusPill label="Activity" value={activity} />
          <StatusPill label="State" value={agentState} tone={agentState.toLowerCase()} />
          <StatusPill label="Connection" value={connected ? "Connected" : "Disconnected"} tone={connected ? "online" : "offline"} />
        </header>
        {error && <div className="error" role="alert">{error}</div>}
        <div className="timeline" ref={timelineRef} aria-live="polite">
          {messages.map((message) => <article className={`message ${message.role}`} key={message.id}>{message.content}</article>)}
        </div>
        <section className="input-dock">
          {activeSurface ? (
            <A2uiSurfaceView surface={activeSurface} clientId={clientId} user={user} onDone={poll} />
          ) : fallbackQuestion ? (
            <QuestionCard question={fallbackQuestion} onDone={poll} />
          ) : (
            <form className="composer" onSubmit={submit}>
              {suggestions.length > 0 && <div className="suggestions">{suggestions.map((item) => <button type="button" key={item} onClick={() => setPrompt(item)}>{item}</button>)}</div>}
              <textarea ref={textareaRef} value={prompt} onKeyDown={onComposerKeyDown} onChange={(event) => setPrompt(event.target.value)} placeholder="Message Alphonse... Type / for commands" rows={1} />
              <button className="send" type="submit">Send</button>
            </form>
          )}
        </section>
      </section>

      {modal === "projects" && <ProjectsModal user={user} active={project} onSelect={(next) => void selectProject(next)} onClose={() => setModal(null)} />}
      {modal === "project-context" && <ProjectContextModal user={user} project={project} onClose={() => setModal(null)} />}
      {modal === "integrations" && <IntegrationsModal user={user} onClose={() => setModal(null)} />}
      {modal === "model" && <ModelModal onClose={() => setModal(null)} />}
      {modal === "agent-config" && <AgentConfigModal onClose={() => setModal(null)} />}
      {modal === "settings" && <SettingsModal enterToSend={enterToSend} onEnterToSendChange={setEnterToSend} onClose={() => setModal(null)} />}
      {modal === "users" && <UsersModal onClose={() => setModal(null)} />}
      {modal === "onboarding" && <OnboardingModal onComplete={(next) => { setUser(next); setModal(null); void poll(); }} />}
    </main>
  );
}

function StatusPill({ label, value, tone = "" }: { label: string; value: string; tone?: string }) {
  return <div className={`status-pill ${tone}`}><span>{label}</span><strong>{value}</strong></div>;
}

function QuestionCard({ question, onDone }: { question: Question; onDone: () => Promise<void> }) {
  const [value, setValue] = useState("");
  const answer = async (payload: Record<string, unknown>, text = "") => {
    await daemonRequest("answer_question", { question_id: question.question_id, payload, text });
    await onDone();
  };
  return <section className="question-card"><strong>Alphonse needs your input</strong><p>{question.message}</p>
    {question.kind === "yes_no" && <div><button onClick={() => void answer({ answer: true }, "yes")}>Yes</button><button onClick={() => void answer({ answer: false }, "no")}>No</button></div>}
    {question.kind === "single_choice" && <div>{question.choices.map((choice) => <button key={choice.id} onClick={() => void answer({ choice_id: choice.id }, choice.label)}>{choice.label}</button>)}</div>}
    {question.kind === "open_text" && <form onSubmit={(event) => { event.preventDefault(); void answer({ text: value }, value); }}><input value={value} onChange={(event) => setValue(event.target.value)} placeholder="Your answer" /><button>Answer</button></form>}
  </section>;
}

function ModalFrame({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  return <div className="modal-backdrop" role="presentation"><section className="modal" role="dialog" aria-modal="true" aria-label={title}><header><h2>{title}</h2><button onClick={onClose}>Close</button></header>{children}</section></div>;
}

function OnboardingModal({ onComplete }: { onComplete: (userId: string) => void }) {
  const [name, setName] = useState(""); const [root, setRoot] = useState("~/.alphonse/users"); const [importV1, setImportV1] = useState(true); const [error, setError] = useState("");
  const save = async () => { try { const result = await daemonRequest<{ admin_user: { user_id: string } }>("onboard", { display_name: name, users_root: root, import_v1: importV1 }); onComplete(result.admin_user.user_id); } catch (cause) { setError(cause instanceof Error ? cause.message : "Onboarding failed"); } };
  return <ModalFrame title="Set up Alphonse" onClose={() => undefined}><p>Create the local administrator and choose where user data is stored.</p><input value={name} onChange={(event) => setName(event.target.value)} placeholder="Your name" /><input value={root} onChange={(event) => setRoot(event.target.value)} placeholder="Users root" /><label><input type="checkbox" checked={importV1} onChange={(event) => setImportV1(event.target.checked)} /> Import existing v1 identity data</label><button onClick={() => void save()}>Create administrator</button><p>{error}</p></ModalFrame>;
}

function SettingsModal({ enterToSend, onEnterToSendChange, onClose }: { enterToSend: boolean; onEnterToSendChange: (value: boolean) => void; onClose: () => void }) {
  const [root, setRoot] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => { void daemonRequest<{ users_root: string }>("settings").then((result) => setRoot(result.users_root)); }, []);
  const save = async () => { const result = await daemonRequest<{ users_root: string; warning_repository_path?: boolean }>("save_settings", { users_root: root }); setNotice(result.warning_repository_path ? "Saved. This path is inside the repository; keep it ignored." : "Saved."); };
  return <ModalFrame title="Settings" onClose={onClose}><label className="setting-row"><input type="checkbox" checked={enterToSend} onChange={(event) => onEnterToSendChange(event.target.checked)} /> Enter sends message</label><label>Users root<input value={root} onChange={(event) => setRoot(event.target.value)} /></label><button onClick={() => void save()}>Save</button><p>{notice}</p></ModalFrame>;
}

function UsersModal({ onClose }: { onClose: () => void }) {
  const [users, setUsers] = useState<Array<{ user_id: string; display_name: string; role: string; is_active: boolean }>>([]); const [name, setName] = useState(""); const [role, setRole] = useState("member");
  const load = useCallback(() => daemonRequest<{ users: Array<{ user_id: string; display_name: string; role: string; is_active: boolean }> }>("users").then((result) => setUsers(result.users)), []);
  useEffect(() => { void load(); }, [load]);
  const create = async () => { await daemonRequest("create_user", { display_name: name, role }); setName(""); await load(); };
  return <ModalFrame title="Users" onClose={onClose}><div className="stack">{users.map((item) => <p key={item.user_id}>{item.display_name} ({item.role})<small>{item.user_id}</small></p>)}</div><input value={name} onChange={(event) => setName(event.target.value)} placeholder="Name" /><select value={role} onChange={(event) => setRole(event.target.value)}><option value="member">Member</option><option value="caregiver">Caregiver</option></select><button onClick={() => void create()}>Add user</button></ModalFrame>;
}

function ProjectsModal({ user, active, onSelect, onClose }: { user: string; active: Project | null; onSelect: (project: Project) => void; onClose: () => void }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState(""); const [rootPath, setRootPath] = useState("");
  const load = useCallback(() => daemonRequest<{ projects: Project[] }>("projects", { user }).then((result) => setProjects(result.projects)), [user]);
  useEffect(() => { void load(); }, [load]);
  const create = async (event: FormEvent) => { event.preventDefault(); const result = await daemonRequest<{ project: Project }>("create_project", { user, name, description: "", root_path: rootPath, visibility: "private" }); onSelect(result.project); };
  return <ModalFrame title="Projects" onClose={onClose}><div className="stack">{projects.map((item) => <button className={item.project_id === active?.project_id ? "selected" : ""} key={item.project_id} onClick={() => onSelect(item)}>{item.name}<small>{item.root_path}</small></button>)}</div><form className="stack" onSubmit={create}><input value={name} onChange={(event) => setName(event.target.value)} placeholder="New project name" required /><input value={rootPath} onChange={(event) => setRootPath(event.target.value)} placeholder="Directory path" required /><button>Create project</button></form></ModalFrame>;
}

function ProjectContextModal({ user, project, onClose }: { user: string; project: Project | null; onClose: () => void }) {
  const [content, setContent] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => { if (project) void daemonRequest<{ content: string }>("project_context", { user, project_id: project.project_id }).then((result) => setContent(result.content)); }, [project, user]);
  if (!project) return <ModalFrame title="Project context" onClose={onClose}><p>Select a project before editing its context.</p></ModalFrame>;
  return <ModalFrame title={`${project.name} context`} onClose={onClose}><textarea value={content} onChange={(event) => setContent(event.target.value)} rows={12} /><button onClick={() => void daemonRequest("save_project_context", { user, project_id: project.project_id, content }).then(() => setNotice("Saved."))}>Save context</button><p>{notice}</p></ModalFrame>;
}

function IntegrationsModal({ user, onClose }: { user: string; onClose: () => void }) {
  const [integrationId, setIntegrationId] = useState("telegram-home"); const [displayName, setDisplayName] = useState("Telegram");
  const [token, setToken] = useState(""); const [telegramUserId, setTelegramUserId] = useState(""); const [allowedChatIds, setAllowedChatIds] = useState("");
  const [pollInterval, setPollInterval] = useState("1"); const [enabled, setEnabled] = useState(false); const [presenceEnabled, setPresenceEnabled] = useState(true); const [notice, setNotice] = useState("");
  useEffect(() => { void daemonRequest<{ integrations: Array<{ integration: Record<string, unknown> | null }> }>("integrations").then((result) => {
    const integration = result.integrations[0]?.integration; if (!integration) return;
    const config = (integration.config as Record<string, unknown> | undefined) ?? {};
    setIntegrationId(String(integration.integration_id || "telegram-home")); setDisplayName(String(integration.display_name || "Telegram")); setEnabled(Boolean(integration.enabled));
    setTelegramUserId(String(config.telegram_user_id || "")); setAllowedChatIds(Array.isArray(config.allowed_chat_ids) ? config.allowed_chat_ids.join(", ") : "");
    setPollInterval(String(config.poll_interval_sec || 1)); setPresenceEnabled(config.presence_enabled !== false);
  }); }, []);
  const save = async () => { await daemonRequest("save_telegram_integration", { user, values: { integration_id: integrationId, display_name: displayName, enabled, bot_token: token, telegram_user_id: telegramUserId, allowed_chat_ids: allowedChatIds, poll_interval_sec: pollInterval, presence_enabled: presenceEnabled } }); setToken(""); setNotice("Saved and integrations restarted."); };
  return <ModalFrame title="Telegram integration" onClose={onClose}><input value={integrationId} onChange={(event) => setIntegrationId(event.target.value)} placeholder="Integration id" /><input value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="Display name" /><input type="password" value={token} onChange={(event) => setToken(event.target.value)} placeholder="Bot token (leave blank to keep current)" /><input value={telegramUserId} onChange={(event) => setTelegramUserId(event.target.value)} placeholder="Telegram user id" /><input value={allowedChatIds} onChange={(event) => setAllowedChatIds(event.target.value)} placeholder="Allowed chat IDs, comma separated" /><input value={pollInterval} onChange={(event) => setPollInterval(event.target.value)} placeholder="Poll interval seconds" /><label><input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} /> Enabled</label><label><input type="checkbox" checked={presenceEnabled} onChange={(event) => setPresenceEnabled(event.target.checked)} /> Presence enabled</label><button onClick={() => void save()}>Save integration</button><p>{notice}</p></ModalFrame>;
}

function ModelModal({ onClose }: { onClose: () => void }) {
  const [providers, setProviders] = useState<Provider[]>([]); const [settings, setSettings] = useState<InferenceSettings | null>(null); const [provider, setProvider] = useState(""); const [model, setModel] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => { void Promise.all([daemonRequest<{ providers: Provider[] }>("inference_providers"), daemonRequest<{ settings: InferenceSettings }>("inference_settings")]).then(([catalog, current]) => { setProviders(catalog.providers); setSettings(current.settings); setProvider(current.settings.provider_key); setModel(current.settings.model_id); }); }, []);
  const selected = providers.find((item) => item.provider_key === provider);
  return <ModalFrame title="Inference model" onClose={onClose}><select value={provider} onChange={(event) => { setProvider(event.target.value); setModel(""); }}>{providers.map((item) => <option value={item.provider_key} key={item.provider_key}>{item.display_name}</option>)}</select><select value={model} onChange={(event) => setModel(event.target.value)}>{selected?.models.map((item) => <option value={item.model_id} key={item.model_id}>{item.display_name}</option>)}</select><button onClick={() => void daemonRequest<{ settings: InferenceSettings }>("set_inference_settings", { provider_key: provider, model_id: model }).then((result) => { setSettings(result.settings); setNotice("Validated and saved."); }).catch((cause: unknown) => setNotice(cause instanceof Error ? cause.message : "Validation failed"))}>Validate & save</button><p>{notice || settings?.validation_error}</p></ModalFrame>;
}

function AgentConfigModal({ onClose }: { onClose: () => void }) {
  const [documents, setDocuments] = useState<AgentDocument[]>([]); const [fileName, setFileName] = useState(""); const [content, setContent] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => { void daemonRequest<{ documents: AgentDocument[] }>("agent_config_documents").then((result) => { setDocuments(result.documents); setFileName(result.documents[0]?.file_name || ""); }); }, []);
  useEffect(() => { if (fileName) void daemonRequest<{ document: AgentDocument }>("read_agent_config", { file_name: fileName }).then((result) => setContent(result.document.content || "")); }, [fileName]);
  return <ModalFrame title="Agent configuration" onClose={onClose}><select value={fileName} onChange={(event) => setFileName(event.target.value)}>{documents.map((item) => <option value={item.file_name} key={item.file_name}>{item.display_name}</option>)}</select><textarea value={content} onChange={(event) => setContent(event.target.value)} rows={14} /><button onClick={() => void daemonRequest("save_agent_config", { file_name: fileName, content }).then(() => setNotice("Saved. Restart the daemon before new tasks use these changes."))}>Save configuration</button><p>{notice}</p></ModalFrame>;
}
