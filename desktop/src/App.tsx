import { FormEvent, ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { daemonRequest, ensureDaemon, stopDaemon } from "./api";
import { matchingCommands } from "./commands";
import { A2uiSurfaceHost, applyA2uiEvent, DESKTOP_CATALOG_ID } from "./a2ui";
import type { ActivityEvent, AgentDocument, ChatMessage, InferenceSettings, Project, Question } from "./types";

const USER = "local";

type Modal = "projects" | "project-context" | "integrations" | "model" | "agent-config" | null;
type PollResponse = {
  events: ActivityEvent[];
  next_sequence: number;
  ui_events?: Array<{ sequence?: number; event: { type: string; name?: string; value?: unknown } }>;
  next_ui_sequence?: number;
  deliveries: Array<{ outbox_message_id: string; message: string }>;
  questions: Question[];
  status: { active_work: Record<string, string>; activity: { state?: string } };
};
type Provider = { provider_key: string; display_name: string; models: Array<{ model_id: string; display_name: string }> };

export default function App() {
  const clientId = useRef(crypto.randomUUID()).current;
  const sequence = useRef(0);
  const uiSequence = useRef(0);
  const delivered = useRef(new Set<string>());
  const [messages, setMessages] = useState<ChatMessage[]>([{ id: "welcome", role: "assistant", content: "Alphonse Desktop is connected locally." }]);
  const [prompt, setPrompt] = useState("");
  const [connected, setConnected] = useState(false);
  const [activity, setActivity] = useState("Starting daemon…");
  const [questions, setQuestions] = useState<Question[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [modal, setModal] = useState<Modal>(null);
  const [error, setError] = useState("");
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [surfaces, setSurfaces] = useState<Record<string, import("./a2ui").A2uiSurface>>({});

  const poll = useCallback(async () => {
    try {
      const response = await daemonRequest<PollResponse>("desktop_poll", {
        client_id: clientId,
        user: USER,
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
      setEvents((current) => [...current, ...response.events].slice(-12));
      const latest = response.events.at(-1);
      setActivity(latest ? `${latest.label || latest.phase}: ${latest.message || "Working"}` : response.status.activity.state || "idle");
      for (const delivery of response.deliveries) {
        if (delivered.current.has(delivery.outbox_message_id)) continue;
        delivered.current.add(delivery.outbox_message_id);
        setMessages((current) => [...current, { id: delivery.outbox_message_id, role: "assistant", content: delivery.message }]);
        await daemonRequest("desktop_ack_delivery", { client_id: clientId, outbox_message_id: delivery.outbox_message_id });
      }
    } catch (cause) {
      setConnected(false);
      setError(cause instanceof Error ? cause.message : "Daemon connection lost");
      setActivity("Disconnected");
    }
  }, [clientId]);

  useEffect(() => {
    ensureDaemon().then(() => poll()).catch((cause: unknown) => setError(cause instanceof Error ? cause.message : "Unable to start daemon"));
    const timer = window.setInterval(() => void poll(), 800);
    return () => window.clearInterval(timer);
  }, [poll]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const value = prompt.trim();
    if (!value) return;
    setPrompt("");
    if (value.startsWith("/")) {
      await runCommand(value.split(/\s+/, 1)[0]);
      return;
    }
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", content: value }]);
    try {
      await daemonRequest("queue_message", {
        prompt: value,
        user: USER,
        project_id: project?.project_id || "",
        integration_id: "desktop",
        provider_key: "tui",
        channel_target: USER,
      });
      setActivity("Queued");
      await poll();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Message could not be queued");
    }
  };

  const runCommand = async (command: string) => {
    if (command === "/project") return setModal("projects");
    if (command === "/project-context") return setModal("project-context");
    if (command === "/integrations") return setModal("integrations");
    if (command === "/model" || command === "/model-provider") return setModal("model");
    if (command === "/agent-config") return setModal("agent-config");
    if (command === "/stop") {
      if (window.confirm("Stop the Alphonse daemon?")) await stopDaemon();
      return;
    }
    if (command === "/exit" || command === "/quit") window.close();
  };

  const suggestions = useMemo(() => matchingCommands(prompt), [prompt]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="mark">A</span><span>Alphonse</span></div>
        <p className={`connection ${connected ? "online" : "offline"}`}>{connected ? "Daemon connected" : "Daemon unavailable"}</p>
        <button onClick={() => setModal("projects")}>Project{project ? `: ${project.name}` : ""}</button>
        <button onClick={() => setModal("integrations")}>Integrations</button>
        <button onClick={() => setModal("model")}>Model</button>
        <button onClick={() => setModal("agent-config")}>Agent configuration</button>
        <div className="event-log">
          <h2>Activity</h2>
          {events.slice(-5).reverse().map((entry) => <p key={entry.sequence}>{entry.label || entry.phase}</p>)}
        </div>
      </aside>

      <section className="conversation">
        <header>
          <div><p className="eyebrow">LOCAL PRESENCE</p><h1>{project?.name || "Home"}</h1></div>
          <p className="activity">{activity}</p>
        </header>
        {error && <div className="error" role="alert">{error}</div>}
        <div className="timeline" aria-live="polite">
          {messages.map((message) => <article className={`message ${message.role}`} key={message.id}>{message.content}</article>)}
        </div>
        <A2uiSurfaceHost surfaces={surfaces} clientId={clientId} user={USER} onDone={poll} />
        {questions.filter((question) => !surfaces[`question:${question.question_id}`]).map((question) => <QuestionCard key={question.question_id} question={question} onDone={poll} />)}
        <form className="composer" onSubmit={submit}>
          {suggestions.length > 0 && <div className="suggestions">{suggestions.map((item) => <button type="button" key={item} onClick={() => setPrompt(item)}>{item}</button>)}</div>}
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Message Alphonse…  Type / for commands" rows={3} />
          <button className="send" type="submit">Send</button>
        </form>
      </section>

      {modal === "projects" && <ProjectsModal active={project} onSelect={(next) => { setProject(next); setModal(null); }} onClose={() => setModal(null)} />}
      {modal === "project-context" && <ProjectContextModal project={project} onClose={() => setModal(null)} />}
      {modal === "integrations" && <IntegrationsModal onClose={() => setModal(null)} />}
      {modal === "model" && <ModelModal onClose={() => setModal(null)} />}
      {modal === "agent-config" && <AgentConfigModal onClose={() => setModal(null)} />}
    </main>
  );
}

function QuestionCard({ question, onDone }: { question: Question; onDone: () => Promise<void> }) {
  const [value, setValue] = useState("");
  const answer = async (payload: Record<string, unknown>, text = "") => {
    await daemonRequest("answer_question", { user: USER, question_id: question.question_id, payload, text });
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

function ProjectsModal({ active, onSelect, onClose }: { active: Project | null; onSelect: (project: Project) => void; onClose: () => void }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState(""); const [rootPath, setRootPath] = useState("");
  const load = useCallback(() => daemonRequest<{ projects: Project[] }>("projects", { user: USER }).then((result) => setProjects(result.projects)), []);
  useEffect(() => { void load(); }, [load]);
  const create = async (event: FormEvent) => { event.preventDefault(); const result = await daemonRequest<{ project: Project }>("create_project", { user: USER, name, description: "", root_path: rootPath, visibility: "private" }); onSelect(result.project); };
  return <ModalFrame title="Projects" onClose={onClose}><div className="stack">{projects.map((item) => <button className={item.project_id === active?.project_id ? "selected" : ""} key={item.project_id} onClick={() => onSelect(item)}>{item.name}<small>{item.root_path}</small></button>)}</div><form className="stack" onSubmit={create}><input value={name} onChange={(event) => setName(event.target.value)} placeholder="New project name" required /><input value={rootPath} onChange={(event) => setRootPath(event.target.value)} placeholder="Directory path" required /><button>Create project</button></form></ModalFrame>;
}

function ProjectContextModal({ project, onClose }: { project: Project | null; onClose: () => void }) {
  const [content, setContent] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => { if (project) void daemonRequest<{ content: string }>("project_context", { user: USER, project_id: project.project_id }).then((result) => setContent(result.content)); }, [project]);
  if (!project) return <ModalFrame title="Project context" onClose={onClose}><p>Select a project before editing its context.</p></ModalFrame>;
  return <ModalFrame title={`${project.name} context`} onClose={onClose}><textarea value={content} onChange={(event) => setContent(event.target.value)} rows={12} /><button onClick={() => void daemonRequest("save_project_context", { user: USER, project_id: project.project_id, content }).then(() => setNotice("Saved."))}>Save context</button><p>{notice}</p></ModalFrame>;
}

function IntegrationsModal({ onClose }: { onClose: () => void }) {
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
  const save = async () => { await daemonRequest("save_telegram_integration", { user: USER, values: { integration_id: integrationId, display_name: displayName, enabled, bot_token: token, telegram_user_id: telegramUserId, allowed_chat_ids: allowedChatIds, poll_interval_sec: pollInterval, presence_enabled: presenceEnabled } }); setToken(""); setNotice("Saved and integrations restarted."); };
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
