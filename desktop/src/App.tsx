import { FormEvent, KeyboardEvent, ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { daemonRequest, ensureDaemon, showInFinder, stopDaemon } from "./api";
import { matchingCommands } from "./commands";
import { mergeFreshConversationHistory } from "./conversationHistory";
import { buildConversationTimeline } from "./conversationTimeline";
import { A2uiSurfaceView, applyA2uiEvent, DESKTOP_CATALOG_ID, type A2uiSurface } from "./a2ui";
import { DESKTOP_STYLE_STORAGE_KEY, parseDesktopStyle, type DesktopStyle } from "./desktopStyle";
import { readDismissedScheduledSurfaces, rememberDismissedScheduledSurface, withoutDismissedSurfaces, withoutSurface } from "./dismissedSurfaces";
import { agentStateLabel, capdActivityLabel, projectKey } from "./layoutState";
import { formatMessageTime } from "./messageTime";
import type { ActivityEvent, AgentDocument, ChatMessage, InferenceSettings, MediaToolsSettings, MemorySettings, Project, Question, WebToolsSettings } from "./types";

type Modal = "projects" | "project-settings" | "project-context" | "scheduled-tasks" | "settings" | "users" | "onboarding" | null;
type SettingsTab = "general" | "appearance" | "tools" | "artifacts" | "integrations" | "automations" | "model" | "agent-config";
type ManagedProject = Project & { owner?: { display_name?: string; user_id?: string } | null };
type PollResponse = {
  events: ActivityEvent[];
  next_sequence: number;
  ui_events?: Array<{ sequence?: number; event: { type: string; name?: string; value?: unknown } }>;
  next_ui_sequence?: number;
  deliveries: Array<{ outbox_message_id: string; message: string; task_id?: string; project_id: string; created_at: string; conversation_sequence: number }>;
  questions: Question[];
  project_attention?: ProjectAttention;
  status: { active_work: Record<string, string>; activity: { state?: string }; queue?: { ready?: number; processing?: number } };
};
type HistoryResponse = { messages: ChatMessage[] };
type RecentFilesResponse = { files: Array<{ name: string; kind: "file" | "directory"; modified_at: string }> };
type Provider = { provider_key: string; display_name: string; models: Array<{ model_id: string; display_name: string }> };
type ProjectAttention = Record<string, { unread_messages: number; pending_questions: number; total: number }>;

export default function App() {
  const clientId = useRef(crypto.randomUUID()).current;
  const sequence = useRef(0);
  const uiSequence = useRef(0);
  const delivered = useRef(new Set<string>());
  const progressTaskIdsRef = useRef(new Set<string>());
  const progressStartedAtRef = useRef(new Map<string, number>());
  const progressMeaningfulAtRef = useRef(new Map<string, number>());
  const progressCompletionTimersRef = useRef(new Map<string, number>());
  const projectHistoryRequestRef = useRef(0);
  const projectHistoryLoadingRef = useRef(0);
  const activeHistoryRefreshRef = useRef(false);
  const activeProjectKeyRef = useRef(projectKey());
  const messagesDuringHistoryReloadRef = useRef(new Map<number, ChatMessage[]>());
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const surfacesRef = useRef<Record<string, A2uiSurface>>({});
  const dismissedScheduledSurfacesRef = useRef(readDismissedScheduledSurfaces(window.localStorage));
  const [messages, setMessages] = useState<ChatMessage[]>([{ id: "welcome", role: "assistant", content: "Alphonse Desktop is connected locally." }]);
  const [messageBuckets, setMessageBuckets] = useState<Record<string, ChatMessage[]>>({});
  const [prompt, setPrompt] = useState("");
  const [connected, setConnected] = useState(false);
  const [activity, setActivity] = useState("idle");
  const [agentState, setAgentState] = useState<"Idle" | "Working" | "Error" | "Disconnected">("Disconnected");
  const [questions, setQuestions] = useState<Question[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [projectForSettings, setProjectForSettings] = useState<ManagedProject | null>(null);
  const [scheduledTaskForView, setScheduledTaskForView] = useState("");
  const [modal, setModal] = useState<Modal>(null);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("general");
  const [user, setUser] = useState("");
  const [error, setError] = useState("");
  const [surfaces, setSurfaces] = useState<Record<string, import("./a2ui").A2uiSurface>>({});
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [recentFilesOpen, setRecentFilesOpen] = useState(false);
  const [recentFiles, setRecentFiles] = useState<RecentFilesResponse["files"]>([]);
  const [recentFilesError, setRecentFilesError] = useState("");
  const [queueStatus, setQueueStatus] = useState({ ready: 0, processing: 0 });
  const [projectAttention, setProjectAttention] = useState<ProjectAttention>({});
  const [timezone, setTimezone] = useState("UTC");
  const [progressTaskIds, setProgressTaskIds] = useState<string[]>([]);
  const [pendingProgressMessages, setPendingProgressMessages] = useState<Record<string, ChatMessage>>({});
  const [morphedMessageTaskIds, setMorphedMessageTaskIds] = useState<Record<string, string>>({});
  const [heldProgressSurfaces, setHeldProgressSurfaces] = useState<Record<string, A2uiSurface>>({});
  const [enterToSend, setEnterToSend] = useState(() => window.localStorage.getItem("alphonse.desktop.enterToSend") !== "false");
  const [desktopStyle, setDesktopStyle] = useState<DesktopStyle>(() => parseDesktopStyle(window.localStorage.getItem(DESKTOP_STYLE_STORAGE_KEY)));
  const currentProjectKey = projectKey(project?.project_id);
  activeProjectKeyRef.current = currentProjectKey;

  const dismissScheduledSurface = useCallback((surfaceId: string) => {
    dismissedScheduledSurfacesRef.current = rememberDismissedScheduledSurface(
      window.localStorage,
      dismissedScheduledSurfacesRef.current,
      surfaceId,
    );
    surfacesRef.current = withoutSurface(surfacesRef.current, surfaceId);
    setSurfaces((current) => withoutSurface(current, surfaceId));
  }, []);

  const appendMessage = useCallback((message: ChatMessage) => {
    const activeKey = activeProjectKeyRef.current;
    const loadingRequest = projectHistoryLoadingRef.current;
    if (loadingRequest) {
      const pending = messagesDuringHistoryReloadRef.current.get(loadingRequest) || [];
      messagesDuringHistoryReloadRef.current.set(loadingRequest, [...pending, message]);
    }
    setMessages((current) => {
      const next = mergeFreshConversationHistory(current, [message]);
      setMessageBuckets((buckets) => ({ ...buckets, [activeKey]: next }));
      return next;
    });
  }, []);

  const poll = useCallback(async () => {
    try {
      const requestedProjectId = project?.project_id || "";
      const response = await daemonRequest<PollResponse>("desktop_poll", {
        client_id: clientId,
        user,
        project_id: requestedProjectId,
        after_sequence: sequence.current,
        after_ui_sequence: uiSequence.current,
        client_capabilities: { supportedCatalogIds: [DESKTOP_CATALOG_ID] },
        limit: 50,
      });
      sequence.current = response.next_sequence;
      uiSequence.current = response.next_ui_sequence ?? uiSequence.current;
      setConnected(true);
      setError("");
      const responseIsForActiveProject = projectKey(requestedProjectId) === activeProjectKeyRef.current;
      if (responseIsForActiveProject) setQuestions(response.questions);
      setProjectAttention(response.project_attention || {});
      setQueueStatus({ ready: response.status.queue?.ready || 0, processing: response.status.queue?.processing || 0 });
      const newProgressTaskIds = responseIsForActiveProject ? taskProgressIds(response.ui_events || []) : [];
      if (newProgressTaskIds.length) {
        newProgressTaskIds.forEach((taskId) => {
          progressTaskIdsRef.current.add(taskId);
          if (!progressStartedAtRef.current.has(taskId)) progressStartedAtRef.current.set(taskId, Date.now());
        });
        setProgressTaskIds((current) => [...current, ...newProgressTaskIds.filter((taskId) => !current.includes(taskId))]);
      }
      const projectedSurfaces = responseIsForActiveProject
        ? withoutDismissedSurfaces(
            (response.ui_events || []).reduce((next, item) => applyA2uiEvent(next, item.event), surfacesRef.current),
            dismissedScheduledSurfacesRef.current,
          )
        : surfacesRef.current;
      if (responseIsForActiveProject) {
        surfacesRef.current = projectedSurfaces;
        setSurfaces(projectedSurfaces);
      }
      newProgressTaskIds.forEach((taskId) => {
        const components = projectedSurfaces[`task-progress:${taskId}`]?.components || {};
        if (["criteria", "intention", "tool", "result"].some((componentId) => String(components[componentId]?.text || "").trim()) || Object.keys(components).some((componentId) => componentId.startsWith("step_"))) {
          progressMeaningfulAtRef.current.set(taskId, Date.now());
        }
      });
      const latest = response.events.at(-1);
      setActivity(capdActivityLabel(latest, response.status.activity.state || "idle"));
      setAgentState(agentStateLabel(true, false, Object.keys(response.status.active_work || {}).length));
      for (const delivery of response.deliveries) {
        if (delivered.current.has(delivery.outbox_message_id)) continue;
        delivered.current.add(delivery.outbox_message_id);
        const deliveryProjectKey = projectKey(delivery.project_id);
        const completedMessage: ChatMessage = {
          id: delivery.outbox_message_id,
          role: "assistant",
          content: delivery.message,
          created_at: delivery.created_at,
          project_id: delivery.project_id,
          sequence: delivery.conversation_sequence,
        };
        if (delivery.task_id && deliveryProjectKey === activeProjectKeyRef.current) {
          if (progressTaskIdsRef.current.has(delivery.task_id)) {
            const taskId = delivery.task_id;
            const progressSurface = surfacesRef.current[`task-progress:${taskId}`];
            if (progressSurface) setHeldProgressSurfaces((current) => ({ ...current, [taskId]: progressSurface }));
            setPendingProgressMessages((current) => ({ ...current, [taskId]: completedMessage }));
            const visibleSince = progressMeaningfulAtRef.current.get(taskId) || progressStartedAtRef.current.get(taskId) || Date.now();
            const elapsed = Date.now() - visibleSince;
            const finish = () => {
              progressCompletionTimersRef.current.delete(taskId);
              progressTaskIdsRef.current.delete(taskId);
              progressStartedAtRef.current.delete(taskId);
              progressMeaningfulAtRef.current.delete(taskId);
              setMorphedMessageTaskIds((current) => ({ ...current, [completedMessage.id]: taskId }));
              setProgressTaskIds((current) => current.filter((value) => value !== taskId));
              setPendingProgressMessages((current) => {
                const next = { ...current };
                delete next[taskId];
                return next;
              });
              setHeldProgressSurfaces((current) => { const next = { ...current }; delete next[taskId]; return next; });
              const nextSurfaces = { ...surfacesRef.current };
              delete nextSurfaces[`task-progress:${taskId}`];
              surfacesRef.current = nextSurfaces;
              setSurfaces(nextSurfaces);
              appendMessage(completedMessage);
            };
            const remaining = Math.max(0, 1500 - elapsed);
            if (remaining) progressCompletionTimersRef.current.set(taskId, window.setTimeout(finish, remaining)); else finish();
            await daemonRequest("desktop_ack_delivery", { client_id: clientId, outbox_message_id: delivery.outbox_message_id });
            await daemonRequest("desktop_mark_project_seen", { user, project_id: delivery.project_id, through_sequence: delivery.conversation_sequence });
            setProjectAttention((current) => clearUnreadAttention(current, delivery.project_id));
            continue;
          }
        }
        if (deliveryProjectKey === activeProjectKeyRef.current) appendMessage(completedMessage);
        else setMessageBuckets((current) => ({
          ...current,
          [deliveryProjectKey]: mergeFreshConversationHistory(current[deliveryProjectKey] || [], [completedMessage]),
        }));
        await daemonRequest("desktop_ack_delivery", { client_id: clientId, outbox_message_id: delivery.outbox_message_id });
        if (deliveryProjectKey === activeProjectKeyRef.current) {
          await daemonRequest("desktop_mark_project_seen", { user, project_id: delivery.project_id, through_sequence: delivery.conversation_sequence });
          setProjectAttention((current) => clearUnreadAttention(current, delivery.project_id));
        }
      }
    } catch (cause) {
      setConnected(false);
      setError(cause instanceof Error ? cause.message : "Daemon connection lost");
      setActivity("idle");
      setAgentState("Disconnected");
    }
  }, [appendMessage, clientId, project?.project_id, user]);

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
    window.localStorage.setItem(DESKTOP_STYLE_STORAGE_KEY, desktopStyle);
    document.documentElement.dataset.alphonseStyle = desktopStyle;
  }, [desktopStyle]);

  useEffect(() => {
    if (!user) return;
    void daemonRequest<{ timezone: string }>("timezone_settings", { actor_user_id: user })
      .then((result) => setTimezone(result.timezone || "UTC"))
      .catch(() => setTimezone("UTC"));
  }, [user]);

  useEffect(() => {
    const activeProjectId = project?.project_id || "";
    if (!user || !(projectAttention[activeProjectId]?.unread_messages > 0) || activeHistoryRefreshRef.current || projectHistoryLoadingRef.current) return;
    activeHistoryRefreshRef.current = true;
    void daemonRequest<HistoryResponse>("desktop_conversation_history", { user, project_id: activeProjectId, limit: 100 })
      .then(async (history) => {
        setMessages((current) => {
          const refreshed = mergeFreshConversationHistory(history.messages, current);
          setMessageBuckets((buckets) => ({ ...buckets, [projectKey(activeProjectId)]: refreshed }));
          return refreshed;
        });
        const throughSequence = history.messages.reduce((latest, message) => Math.max(latest, message.sequence || 0), 0);
        await daemonRequest("desktop_mark_project_seen", { user, project_id: activeProjectId, through_sequence: throughSequence });
        setProjectAttention((current) => ({
          ...current,
          [activeProjectId]: {
            unread_messages: 0,
            pending_questions: current[activeProjectId]?.pending_questions || 0,
            total: current[activeProjectId]?.pending_questions || 0,
          },
        }));
      })
      .catch((cause: unknown) => setError(cause instanceof Error ? cause.message : "Conversation history could not be refreshed"))
      .finally(() => { activeHistoryRefreshRef.current = false; });
  }, [project?.project_id, projectAttention, user]);

  useEffect(() => {
    const element = timelineRef.current;
    if (!element) return;
    element.scrollTop = element.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 4 * 22 + 24)}px`;
  }, [prompt]);

  useEffect(() => {
    if (!recentFilesOpen || !project) return;
    let active = true;
    setRecentFilesError("");
    void daemonRequest<RecentFilesResponse>("project_recent_files", { user, project_id: project.project_id, limit: 4 })
      .then((result) => { if (active) setRecentFiles(result.files); })
      .catch((cause: unknown) => {
        if (!active) return;
        setRecentFiles([]);
        setRecentFilesError(cause instanceof Error ? cause.message : "Recent files could not be loaded");
      });
    return () => { active = false; };
  }, [project, recentFilesOpen, user]);

  const selectProject = useCallback(async (next: Project) => {
    const nextKey = projectKey(next.project_id);
    activeProjectKeyRef.current = nextKey;
    const previousHistoryRequest = projectHistoryLoadingRef.current;
    if (previousHistoryRequest) messagesDuringHistoryReloadRef.current.delete(previousHistoryRequest);
    const historyRequest = projectHistoryRequestRef.current + 1;
    projectHistoryRequestRef.current = historyRequest;
    projectHistoryLoadingRef.current = historyRequest;
    messagesDuringHistoryReloadRef.current.set(historyRequest, []);
    setMessageBuckets((buckets) => ({ ...buckets, [currentProjectKey]: messages }));
    setProject(next);
    setRecentFilesOpen(false);
    setRecentFiles([]);
    setRecentFilesError("");
    setModal(null);
    setSurfaces({});
    surfacesRef.current = {};
    progressTaskIdsRef.current.clear();
    progressCompletionTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    progressCompletionTimersRef.current.clear();
    progressStartedAtRef.current.clear();
    progressMeaningfulAtRef.current.clear();
    setProgressTaskIds([]);
    setPendingProgressMessages({});
    setMorphedMessageTaskIds({});
    setHeldProgressSurfaces({});
    setQuestions([]);
    delivered.current.clear();
    setMessages([]);
    try {
      const history = await daemonRequest<HistoryResponse>("desktop_conversation_history", { user, project_id: next.project_id, limit: 100 });
      if (projectHistoryRequestRef.current !== historyRequest) {
        messagesDuringHistoryReloadRef.current.delete(historyRequest);
        return;
      }
      const pending = messagesDuringHistoryReloadRef.current.get(historyRequest) || [];
      const refreshed = mergeFreshConversationHistory(history.messages, pending);
      projectHistoryLoadingRef.current = 0;
      messagesDuringHistoryReloadRef.current.delete(historyRequest);
      setMessages(refreshed);
      setMessageBuckets((buckets) => ({ ...buckets, [nextKey]: refreshed }));
      const throughSequence = refreshed.reduce((latest, message) => Math.max(latest, message.sequence || 0), 0);
      await daemonRequest("desktop_mark_project_seen", {
        user,
        project_id: next.project_id,
        through_sequence: throughSequence,
      });
      setProjectAttention((current) => ({
        ...current,
        [next.project_id]: {
          unread_messages: 0,
          pending_questions: current[next.project_id]?.pending_questions || 0,
          total: current[next.project_id]?.pending_questions || 0,
        },
      }));
    } catch (cause) {
      if (projectHistoryRequestRef.current !== historyRequest) {
        messagesDuringHistoryReloadRef.current.delete(historyRequest);
        return;
      }
      const pending = messagesDuringHistoryReloadRef.current.get(historyRequest) || [];
      projectHistoryLoadingRef.current = 0;
      messagesDuringHistoryReloadRef.current.delete(historyRequest);
      setMessages(pending);
      setError(cause instanceof Error ? cause.message : "Conversation history could not be loaded");
    }
  }, [currentProjectKey, messages, user]);

  const submitPrompt = async () => {
    const value = prompt.trim();
    if (!value) return;
    setPrompt("");
    if (value.startsWith("/")) {
      await runCommand(value.split(/\s+/, 1)[0]);
      return;
    }
    try {
      const queued = await daemonRequest<{ message_id: string; project_id: string; created_at: string; conversation_sequence: number }>("queue_message", {
        prompt: value,
        user,
        project_id: project?.project_id || "",
        integration_id: "desktop",
        provider_key: "tui",
        channel_target: user,
      });
      appendMessage({
        id: queued.message_id,
        role: "user",
        content: value,
        created_at: queued.created_at,
        project_id: queued.project_id,
        sequence: queued.conversation_sequence,
      });
      await daemonRequest("desktop_mark_project_seen", { user, project_id: queued.project_id, through_sequence: queued.conversation_sequence });
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

  const revealProjectInFinder = async () => {
    if (!project) return;
    try {
      await showInFinder(project.root_path);
      setRecentFilesError("");
    } catch (cause) {
      setRecentFilesError(cause instanceof Error ? cause.message : "Finder could not be opened");
    }
  };

  const runCommand = async (command: string) => {
    if (command === "/project") return setModal("projects");
    if (command === "/project-context") return setModal("project-context");
    if (command === "/integrations") { setSettingsTab("integrations"); return setModal("settings"); }
    if (command === "/model" || command === "/model-provider") { setSettingsTab("model"); return setModal("settings"); }
    if (command === "/agent-config") { setSettingsTab("agent-config"); return setModal("settings"); }
    if (command === "/scheduled-tasks") return setModal("scheduled-tasks");
    if (command === "/settings") { setSettingsTab("general"); return setModal("settings"); }
    if (command === "/users") return setModal("users");
    if (command === "/stop") {
      if (window.confirm("Stop the Alphonse daemon?")) await stopDaemon();
      return;
    }
    if (command === "/exit" || command === "/quit") window.close();
  };

  const suggestions = useMemo(() => matchingCommands(prompt), [prompt]);
  const activeSurface = Object.values(surfaces).find((surface) => !surface.surfaceId.startsWith("task-progress:") && !progressTaskIds.includes(questionTaskId(surface)));
  const fallbackQuestion = questions.find((question) => !surfaces[`question:${question.question_id}`]);
  const timelineNodes: ReactNode[] = buildConversationTimeline(
    messages,
    progressTaskIds,
    pendingProgressMessages,
    morphedMessageTaskIds,
  ).map((item) => {
    if (item.kind === "message") return <MessageBubble key={item.key} message={item.message} timezone={timezone} />;
    const surface = surfaces[`task-progress:${item.taskId}`] || heldProgressSurfaces[item.taskId];
    const questionSurface = Object.values(surfaces).find((candidate) => questionTaskId(candidate) === item.taskId);
    if (questionSurface) {
      return <article className="message assistant task-question-bubble" key={item.key}><A2uiSurfaceView surface={questionSurface} clientId={clientId} user={user} onDone={poll} /></article>;
    }
    return surface ? <TaskProgressBubble key={item.key} surface={surface} /> : null;
  });

  return (
    <main className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="sidebar" aria-label="Alphonse navigation">
        <div className="brand">
          <span className="mark">A</span>
          <span className="brand-name">Alphonse</span>
          <button className="collapse-toggle" type="button" onClick={() => setSidebarCollapsed((value) => !value)} aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}>{sidebarCollapsed ? ">" : "<"}</button>
        </div>
        <section className="project-sidebar-section">
          <div className="project-sidebar-header">
            <button className="project-selector" title="Projects" onClick={() => setModal("projects")}><span className="nav-icon" aria-hidden="true">🗂️</span><span className="nav-label">Project</span><small>{project?.name || "Home"}</small>{attentionTotal(projectAttention) > 0 && <span className="attention-badge" aria-label={`${attentionTotal(projectAttention)} project items need attention`}>{attentionTotal(projectAttention)}</span>}</button>
            {project && <button className="project-disclosure" type="button" title={recentFilesOpen ? "Hide recent files" : "Show recent files"} aria-label={recentFilesOpen ? "Hide recent files" : "Show recent files"} aria-expanded={recentFilesOpen} onClick={() => setRecentFilesOpen((open) => !open)}>{recentFilesOpen ? "⌄" : "›"}</button>}
          </div>
          {project && recentFilesOpen && <div className="recent-files-panel">
            <div className="recent-files-heading"><span>Recent files</span><button type="button" onClick={() => void revealProjectInFinder()}>Show in Finder</button></div>
            {recentFilesError && <p className="recent-files-error" role="alert">{recentFilesError}</p>}
            {!recentFilesError && (recentFiles.length ? <ul>{recentFiles.map((file) => <li key={`${file.kind}:${file.name}`}><span className="recent-file-icon" aria-hidden="true">{file.kind === "directory" ? "📁" : "📄"}</span><span className="recent-file-name" title={file.name}>{file.name}</span><small>{dateLabel(file.modified_at)}</small></li>)}</ul> : <p className="recent-files-empty">No accessible files yet.</p>)}
          </div>}
        </section>
        <button title="Scheduled tasks" onClick={() => setModal("scheduled-tasks")}><span className="nav-icon" aria-hidden="true">◷</span><span className="nav-label">Scheduled tasks</span></button>
        <button title="Users" onClick={() => setModal("users")}><span className="nav-icon" aria-hidden="true">♙</span><span className="nav-label">Users</span></button>
        <button title="Settings" onClick={() => { setSettingsTab("general"); setModal("settings"); }}><span className="nav-icon" aria-hidden="true">⚙</span><span className="nav-label">Settings</span></button>
        <div className="queue-sidebar-status" title="Inbound message queue"><span className="queue-icon" aria-hidden="true">☷</span><span className="queue-label">Queue</span><small>{queueStatus.ready} waiting{queueStatus.processing ? ` · ${queueStatus.processing} working` : ""}</small></div>
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
          {timelineNodes}
        </div>
        <section className="input-dock">
          {activeSurface ? (
            <A2uiSurfaceView surface={activeSurface} clientId={clientId} user={user} onDone={poll} onDismiss={dismissScheduledSurface} onAction={(result) => { if (result.action === "view_scheduled_task" && typeof result.scheduled_task_id === "string") { dismissScheduledSurface(activeSurface.surfaceId); setScheduledTaskForView(result.scheduled_task_id); setModal("scheduled-tasks"); } }} />
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

      {modal === "projects" && <ProjectsModal user={user} active={project} attention={projectAttention} onSelect={(next) => void selectProject(next)} onSettings={(next) => { setProjectForSettings(next); setModal("project-settings"); }} onClose={() => setModal(null)} />}
      {modal === "project-settings" && projectForSettings && <ProjectSettingsModal user={user} project={projectForSettings} onBack={() => setModal("projects")} onClose={() => setModal(null)} />}
      {modal === "project-context" && <ProjectContextModal user={user} project={project} onClose={() => setModal(null)} />}
      {modal === "scheduled-tasks" && <ScheduledTasksModal actorUserId={user} initialTaskId={scheduledTaskForView} onClose={() => { setScheduledTaskForView(""); setModal(null); }} />}
      {modal === "settings" && <SettingsModal user={user} initialTab={settingsTab} enterToSend={enterToSend} desktopStyle={desktopStyle} onDesktopStyleChange={setDesktopStyle} onEnterToSendChange={setEnterToSend} onTimezoneChange={setTimezone} onClose={() => setModal(null)} />}
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
  const cancel = async () => {
    await daemonRequest("cancel_question", { question_id: question.question_id });
    await onDone();
  };
  return <section className="question-card"><strong>Alphonse needs your input</strong><p>{question.message}</p>
    {question.kind === "yes_no" && <div className="question-actions"><button onClick={() => void answer({ answer: true }, "yes")}>Yes</button><button onClick={() => void answer({ answer: false }, "no")}>No</button><button className="question-cancel" onClick={() => void cancel()}>Cancel</button></div>}
    {question.kind === "single_choice" && <><div>{question.choices.map((choice) => <button key={choice.id} onClick={() => void answer({ choice_id: choice.id }, choice.label)}>{choice.label}</button>)}</div><div className="question-actions"><button className="question-cancel" onClick={() => void cancel()}>Cancel</button></div></>}
    {question.kind === "open_text" && <form onSubmit={(event) => { event.preventDefault(); void answer({ text: value }, value); }}><input value={value} onChange={(event) => setValue(event.target.value)} placeholder="Your answer" /><span className="question-actions"><button>Answer</button><button type="button" className="question-cancel" onClick={() => void cancel()}>Cancel</button></span></form>}
  </section>;
}

function ModalFrame({ title, children, tabs, onClose }: { title: string; children: ReactNode; tabs?: ReactNode; onClose: () => void }) {
  return <div className="modal-backdrop" role="presentation"><section className="modal" role="dialog" aria-modal="true" aria-label={title}><header><h2>{title}</h2><button onClick={onClose}>Close</button></header>{tabs && <div className="modal-tabs">{tabs}</div>}<div className="modal-content">{children}</div></section></div>;
}

function OnboardingModal({ onComplete }: { onComplete: (userId: string) => void }) {
  const [name, setName] = useState(""); const [root, setRoot] = useState("~/.alphonse/users"); const [importV1, setImportV1] = useState(true); const [error, setError] = useState("");
  const save = async () => { try { const result = await daemonRequest<{ admin_user: { user_id: string } }>("onboard", { display_name: name, users_root: root, import_v1: importV1 }); onComplete(result.admin_user.user_id); } catch (cause) { setError(cause instanceof Error ? cause.message : "Onboarding failed"); } };
  return <ModalFrame title="Set up Alphonse" onClose={() => undefined}><p>Create the local administrator and choose where user data is stored.</p><input value={name} onChange={(event) => setName(event.target.value)} placeholder="Your name" /><input value={root} onChange={(event) => setRoot(event.target.value)} placeholder="Users root" /><label><input type="checkbox" checked={importV1} onChange={(event) => setImportV1(event.target.checked)} /> Import existing v1 identity data</label><button onClick={() => void save()}>Create administrator</button><p>{error}</p></ModalFrame>;
}

function SettingsModal({ user, initialTab, enterToSend, desktopStyle, onDesktopStyleChange, onEnterToSendChange, onTimezoneChange, onClose }: { user: string; initialTab: SettingsTab; enterToSend: boolean; desktopStyle: DesktopStyle; onDesktopStyleChange: (value: DesktopStyle) => void; onEnterToSendChange: (value: boolean) => void; onTimezoneChange: (value: string) => void; onClose: () => void }) {
  const [root, setRoot] = useState(""); const [timezone, setTimezone] = useState("UTC"); const [mirrorAutomationMessages, setMirrorAutomationMessages] = useState(false); const [notice, setNotice] = useState(""); const [tab, setTab] = useState<SettingsTab>(initialTab);
  const [web, setWeb] = useState<WebToolsSettings | null>(null); const [webNotice, setWebNotice] = useState("");
  const [memory, setMemory] = useState<MemorySettings | null>(null); const [memoryNotice, setMemoryNotice] = useState("");
  useEffect(() => { void daemonRequest<{ users_root: string; mirror_automation_messages_to_preferred_channel: boolean }>("settings").then((result) => { setRoot(result.users_root); setMirrorAutomationMessages(Boolean(result.mirror_automation_messages_to_preferred_channel)); }); void daemonRequest<{ timezone: string }>("timezone_settings", { actor_user_id: user }).then((result) => setTimezone(result.timezone)).catch((cause: unknown) => setNotice(timezoneSettingsError(cause))); }, [user]);
  useEffect(() => { setTab(initialTab); }, [initialTab]);
  useEffect(() => { if (tab === "tools") void daemonRequest<{ user: { user_id: string } | null }>("current_user").then(async (current) => { if (!current.user) return; const result = await daemonRequest<{ settings: WebToolsSettings }>("web_tools_settings", { actor_user_id: current.user.user_id }); setWeb(result.settings); }).catch((cause: unknown) => setWebNotice(cause instanceof Error ? cause.message : "Web Tools unavailable")); }, [tab]);
  useEffect(() => { void daemonRequest<{ user: { user_id: string } | null }>("current_user").then(async (current) => { if (!current.user) return; const result = await daemonRequest<{ settings: MemorySettings }>("memory_settings", { actor_user_id: current.user.user_id }); setMemory(result.settings); }).catch((cause: unknown) => setMemoryNotice(cause instanceof Error ? cause.message : "Memory settings unavailable")); }, []);
  const save = async () => { try { const timezoneResult = await daemonRequest<{ timezone: string }>("save_timezone_settings", { actor_user_id: user, timezone }); if (timezoneResult.timezone !== timezone.trim()) throw new Error("The daemon did not persist the requested timezone"); setTimezone(timezoneResult.timezone); onTimezoneChange(timezoneResult.timezone); const result = await daemonRequest<{ users_root: string; warning_repository_path?: boolean; mirror_automation_messages_to_preferred_channel: boolean }>("save_settings", { users_root: root, mirror_automation_messages_to_preferred_channel: mirrorAutomationMessages }); setMirrorAutomationMessages(Boolean(result.mirror_automation_messages_to_preferred_channel)); setNotice(result.warning_repository_path ? "Saved. This path is inside the repository; keep it ignored." : "Saved."); } catch (cause) { setNotice(timezoneSettingsError(cause)); } };
  const saveWeb = async () => { if (!web) return; try { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ settings: WebToolsSettings }>("save_web_tools_settings", { actor_user_id: current.user.user_id, values: web }); setWeb(result.settings); setWebNotice("Saved. Newly started tasks can use enabled Web Tools."); } catch (cause) { setWebNotice(cause instanceof Error ? cause.message : "Save failed"); } };
  const verify = async (kind: "search" | "fetch") => { try { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ result: { exception?: { message?: string } } }>("verify_web_tools", { actor_user_id: current.user.user_id, kind }); setWebNotice(result.result.exception?.message || `${kind === "search" ? "SearXNG search" : "Public fetch"} verified.`); } catch (cause) { setWebNotice(cause instanceof Error ? cause.message : "Verification failed"); } };
  const saveMemory = async () => { if (!memory) return; try { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ settings: MemorySettings }>("save_memory_settings", { actor_user_id: current.user.user_id, values: memory }); setMemory(result.settings); setMemoryNotice("Saved. New tasks use these limits."); } catch (cause) { setMemoryNotice(cause instanceof Error ? cause.message : "Memory settings could not be saved"); } };
  const tabs = <div className="settings-tabs" role="tablist" aria-label="Settings sections">{(["general", "appearance", "tools", "artifacts", "integrations", "automations", "model", "agent-config"] as SettingsTab[]).map((item) => <button key={item} type="button" role="tab" aria-selected={tab === item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>{item === "agent-config" ? "Agent configuration" : item[0].toUpperCase() + item.slice(1)}</button>)}</div>;
  return <ModalFrame title="Settings" tabs={tabs} onClose={onClose}>
    {tab === "general" && <div className="general-settings">
      <section className="setting-group">
        <div className="setting-group-heading"><h3>Messaging</h3><p>Choose how messages are sent from the desktop conversation.</p></div>
        <label className="setting-row"><input type="checkbox" checked={enterToSend} onChange={(event) => onEnterToSendChange(event.target.checked)} /> Enter sends message</label>
      </section>
      <section className="setting-group">
        <div className="setting-group-heading"><h3>Workspace &amp; time</h3><p>Set where Alphonse stores user data and the default timezone for new scheduled tasks.</p></div>
        <label>Users root<input value={root} onChange={(event) => setRoot(event.target.value)} /></label>
        <label>Timezone<input value={timezone} placeholder="America/Mexico_City" onChange={(event) => setTimezone(event.target.value)} /><small>Use an IANA timezone. New scheduled tasks use this unless one is explicitly specified.</small></label>
      </section>
      <section className="setting-group">
        <div className="setting-group-heading"><h3>Delivery</h3><p>Control whether automation messages are also delivered outside of Desktop.</p></div>
        <label className="setting-row"><input type="checkbox" checked={mirrorAutomationMessages} onChange={(event) => setMirrorAutomationMessages(event.target.checked)} /> Also send reminders and scheduled-task messages to the recipient’s preferred channel</label>
        <small>Desktop delivery remains unchanged. When enabled, Alphonse sends one additional copy to the recipient’s preferred configured channel.</small>
        <div className="settings-save-actions"><button onClick={() => void save()}>Save general settings</button>{notice && <p role="status">{notice}</p>}</div>
      </section>
      <section className="setting-group">
        <div className="setting-group-heading"><h3>Conversation memory</h3><p>Set how much conversation history Alphonse retains before it compacts the ledger.</p></div>
        {memory && <><label>Ledger limit (bytes)<input type="number" value={memory.max_ledger_bytes} onChange={(event) => setMemory({ ...memory, max_ledger_bytes: Number(event.target.value) })} /></label><label>Compaction limit (words)<input type="number" value={memory.compaction_summary_max_words} onChange={(event) => setMemory({ ...memory, compaction_summary_max_words: Number(event.target.value) })} /></label><div className="settings-save-actions"><button onClick={() => void saveMemory()}>Save memory settings</button>{memoryNotice && <p role="status">{memoryNotice}</p>}</div></>}
        {!memory && memoryNotice && <p role="status">{memoryNotice}</p>}
      </section>
    </div>}
    {tab === "appearance" && <AppearanceSettingsSection value={desktopStyle} onChange={onDesktopStyleChange} />}
    {tab === "tools" && <><section><h3>Web Tools</h3><p>Run SearXNG separately in Docker with JSON output enabled. Alphonse connects to it; it does not manage Docker.</p>{web && <><label className="setting-row"><input type="checkbox" checked={web.enabled} onChange={(event) => setWeb({ ...web, enabled: event.target.checked })} /> Enable Web Search and Fetch</label><label>SearXNG URL<input value={web.searxng_base_url} placeholder="http://127.0.0.1:8080" onChange={(event) => setWeb({ ...web, searxng_base_url: event.target.value })} /></label><label>Search timeout (seconds)<input type="number" value={web.search_timeout_seconds} onChange={(event) => setWeb({ ...web, search_timeout_seconds: Number(event.target.value) })} /></label><label>Fetch timeout (seconds)<input type="number" value={web.fetch_timeout_seconds} onChange={(event) => setWeb({ ...web, fetch_timeout_seconds: Number(event.target.value) })} /></label><label>Fetch text limit<input type="number" value={web.fetch_max_chars} onChange={(event) => setWeb({ ...web, fetch_max_chars: Number(event.target.value) })} /></label><button onClick={() => void saveWeb()}>Save Web Tools</button><button onClick={() => void verify("search")}>Verify SearXNG</button><button onClick={() => void verify("fetch")}>Verify Fetch</button></>}<p>{webNotice}</p></section><MediaToolsSettingsSection /></>}
    {tab === "artifacts" && <ArtifactsSettingsSection user={user} />}
    {tab === "integrations" && <IntegrationsSettingsSection user={user} />}
    {tab === "automations" && <AutomationsSettingsSection user={user} />}
    {tab === "model" && <ModelSettingsSection />}
    {tab === "agent-config" && <AgentConfigSettingsSection />}
  </ModalFrame>;
}

function AppearanceSettingsSection({ value, onChange }: { value: DesktopStyle; onChange: (value: DesktopStyle) => void }) {
  const options: Array<{ value: DesktopStyle; label: string; description: string }> = [
    { value: "classic", label: "Classic", description: "The current green editorial style with Fraunces and DM Mono." },
    { value: "modern", label: "Modern", description: "A crisp Helvetica interface with neutral surfaces and teal accents." },
  ];
  return <section className="settings-panel appearance-settings">
    <div><h3>Appearance</h3><p>Choose how Alphonse Desktop looks on this Mac. Your selection applies immediately.</p></div>
    <div className="style-picker" role="radiogroup" aria-label="Desktop style">
      {options.map((option) => <button key={option.value} type="button" role="radio" aria-checked={value === option.value} className={`style-preview ${option.value}${value === option.value ? " selected" : ""}`} onClick={() => onChange(option.value)}>
        <span className="style-preview-window" aria-hidden="true"><span className="style-preview-sidebar" /><span className="style-preview-content"><span /><span /><i /></span></span>
        <span className="style-preview-copy"><strong>{option.label}</strong><small>{option.description}</small></span>
        <span className="style-preview-selected">{value === option.value ? "Selected" : "Select"}</span>
      </button>)}
    </div>
  </section>;
}

type Artifact = { artifact_id: string; name: string; description: string; project_id: string; entrypoint_path: string; enabled: boolean; timeout_seconds: number };
function ArtifactsSettingsSection({ user }: { user: string }) {
  const [items, setItems] = useState<Artifact[]>([]); const [notice, setNotice] = useState("");
  const load = async () => { try { const result = await daemonRequest<{ artifacts: Artifact[] }>("artifacts", { actor_user_id: user }); setItems(result.artifacts); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Artifacts unavailable"); } };
  useEffect(() => { void load(); }, [user]);
  const update = async (item: Artifact) => { try { await daemonRequest("update_artifact", { actor_user_id: user, artifact_id: item.artifact_id, name: item.name, description: item.description }); setNotice("Artifact saved."); await load(); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Save failed"); } };
  const toggle = async (item: Artifact) => { try { await daemonRequest("set_artifact_enabled", { actor_user_id: user, artifact_id: item.artifact_id, enabled: !item.enabled }); await load(); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Update failed"); } };
  const remove = async (item: Artifact) => { if (!window.confirm(`Unregister ${item.name}? Program files and data will not be deleted.`)) return; try { await daemonRequest("delete_artifact", { actor_user_id: user, artifact_id: item.artifact_id }); await load(); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Unregister failed"); } };
  return <section className="settings-panel"><h3>Artifacts</h3><p>Registered project-local programs become reusable Alphonse tools. Unregistering leaves their files and data untouched.</p>{items.length ? <div className="stack">{items.map((item, index) => <article key={item.artifact_id} className="project-card"><span className="project-card-summary"><strong>{item.artifact_id}</strong><small>Project: {item.project_id} · Entry point: {item.entrypoint_path}</small><label>Name<input value={item.name} onChange={(event) => setItems((current) => current.map((value, position) => position === index ? { ...value, name: event.target.value } : value))} /></label><label>Description<input value={item.description} onChange={(event) => setItems((current) => current.map((value, position) => position === index ? { ...value, description: event.target.value } : value))} /></label></span><span className="project-card-actions"><button onClick={() => void update(item)}>Save</button><button className="secondary" onClick={() => void toggle(item)}>{item.enabled ? "Turn off" : "Turn on"}</button><button className="secondary" onClick={() => void remove(item)}>Unregister</button></span></article>)}</div> : <p>No artifacts registered. Ask Alphonse to register an executable created in an active project.</p>}<p>{notice}</p></section>;
}

function MediaToolsSettingsSection() {
  const [settings, setSettings] = useState<MediaToolsSettings | null>(null); const [notice, setNotice] = useState(""); const [verifying, setVerifying] = useState<"" | "tts" | "stt" | "ocr">(""); const [samples, setSamples] = useState<Record<string, string>>({ tts: "Alphonse text-to-speech verification.", stt: "", ocr: "" });
  const load = async () => { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ settings: MediaToolsSettings }>("media_tools_settings", { actor_user_id: current.user.user_id }); setSettings(result.settings); };
  useEffect(() => { void load().catch((cause: unknown) => setNotice(cause instanceof Error ? cause.message : "Media Tools unavailable")); }, []);
  const save = async (kind: "tts" | "stt" | "ocr") => { if (!settings) return; try { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ settings: MediaToolsSettings }>("save_media_tools_settings", { actor_user_id: current.user.user_id, kind, values: settings[kind] }); setSettings(result.settings); setNotice(`${kind.toUpperCase()} saved; verify it to mark it ready.`); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Save failed"); } };
  const verify = async (kind: "tts" | "stt" | "ocr") => { setVerifying(kind); setNotice(`Verifying ${kind.toUpperCase()}… this may take a while while the local model loads.`); try { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ settings: MediaToolsSettings; result: { exception?: { message?: string } } }>("verify_media_tools", { actor_user_id: current.user.user_id, kind, sample: samples[kind] }); setSettings(result.settings); setNotice(result.result.exception?.message || `${kind.toUpperCase()} verified.`); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Verification failed"); } finally { setVerifying(""); } };
  if (!settings) return <section><h3>Local Media Tools</h3><p>{notice || "Loading…"}</p></section>;
  const field = (kind: "tts" | "stt" | "ocr", key: string, label: string, type = "text") => <label>{label}<input type={type} value={String((settings[kind] as Record<string, unknown>)[key] ?? "")} onChange={(event) => setSettings({ ...settings, [kind]: { ...settings[kind], [key]: type === "number" ? Number(event.target.value) : event.target.value } })} /></label>;
  const toggle = (kind: "tts" | "stt" | "ocr", key: string, label: string) => <label className="setting-row"><input type="checkbox" checked={Boolean((settings[kind] as Record<string, unknown>)[key])} onChange={(event) => setSettings({ ...settings, [kind]: { ...settings[kind], [key]: event.target.checked } })} /> {label}</label>;
  const verification = (kind: "tts" | "stt" | "ocr") => <p>{verifying === kind && <><span className="verification-spinner" aria-label="Verifying" /> Verifying {kind.toUpperCase()}… </>}Ready: {String(settings[kind].available)}. {settings[kind].verification.error || settings[kind].verification.preview || "Not yet verified."}</p>;
  const verifyLabel = (kind: "tts" | "stt" | "ocr", label: string) => verifying === kind ? `Verifying ${kind.toUpperCase()}…` : label;
  return <section><h3>Local Media Tools</h3><p>Install models and runtimes separately, then save and verify each backend. These tools are not yet exposed to conversations.</p><h4>Qwen TTS</h4>{toggle("tts", "enabled", "Enable Qwen TTS")}{field("tts", "model_id", "Model ID or local path")}{field("tts", "device_map", "Device map")}{field("tts", "dtype", "Precision")}{field("tts", "language", "Language")}{field("tts", "speaker", "Speaker")}{field("tts", "instruct", "Voice instruction")}{field("tts", "attn_implementation", "Attention implementation")}{toggle("tts", "local_files_only", "Use local model files only")}{verification("tts")}<label>Test phrase<input disabled={verifying !== ""} value={samples.tts} onChange={(event) => setSamples({ ...samples, tts: event.target.value })} /></label><button disabled={verifying !== ""} onClick={() => void save("tts")}>Save TTS</button><button disabled={verifying !== ""} onClick={() => void verify("tts")}>{verifyLabel("tts", "Verify Qwen TTS")}</button>{settings.platform === "darwin" && <p>macOS say fallback: {settings.say_available ? "available" : "not found"}.</p>}<h4>Whisper STT</h4>{toggle("stt", "enabled", "Enable Whisper STT")}{field("stt", "executable_path", "Whisper executable (blank uses PATH)")}{field("stt", "model", "Model")}{field("stt", "default_language", "Default language")}{verification("stt")}<label>Audio sample path<input disabled={verifying !== ""} value={samples.stt} onChange={(event) => setSamples({ ...samples, stt: event.target.value })} /></label><button disabled={verifying !== ""} onClick={() => void save("stt")}>Save STT</button><button disabled={verifying !== ""} onClick={() => void verify("stt")}>{verifyLabel("stt", "Verify Whisper STT")}</button><h4>Ollama OCR</h4>{toggle("ocr", "enabled", "Enable OCR")}{field("ocr", "ollama_base_url", "Ollama URL")}{field("ocr", "model_id", "Vision model")}{field("ocr", "timeout_seconds", "Timeout seconds", "number")}{verification("ocr")}<label>Image sample path<input disabled={verifying !== ""} value={samples.ocr} onChange={(event) => setSamples({ ...samples, ocr: event.target.value })} /></label><button disabled={verifying !== ""} onClick={() => void save("ocr")}>Save OCR</button><button disabled={verifying !== ""} onClick={() => void verify("ocr")}>{verifyLabel("ocr", "Verify OCR")}</button><p>{notice}</p></section>;
}

type ManagedAddress = { address_id: string; integration_id: string; provider_key: string; provider_user_id: string; channel_target: string; is_preferred: boolean };
type ManagedUser = { user_id: string; display_name: string; role: string; is_active: boolean; addresses?: ManagedAddress[] };
type ScheduledTask = {
  scheduled_task_id: string; owner_user_id: string; name: string; description: string; prompt: string;
  schedule: Record<string, unknown>; timezone: string; status: string; next_run_at: string | null; last_run_at: string | null;
  latest_execution?: { status: string; started_at: string; finished_at: string | null; error: string; last_error?: string } | null;
};
type ScheduledExecution = { run_id: string; status: string; started_at: string; finished_at: string | null; error: string; last_error?: string; attempt_count: number };

function scheduleLabel(schedule: Record<string, unknown>): string {
  return schedule.kind === "once" ? `Once at ${String(schedule.run_at || "unknown")}` : `Repeats: ${String(schedule.rrule || "unknown")}`;
}

function dateLabel(value: string | null | undefined, timezone?: string): string {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString(undefined, timezone ? { timeZone: timezone } : undefined);
  } catch {
    return new Date(value).toLocaleString();
  }
}

function calendarDateLabel(value: string | null | undefined, timezone: string): { month: string; day: string; pending: boolean } {
  if (!value) return { month: "CAL", day: "—", pending: true };
  try {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) throw new Error("invalid date");
    return {
      month: new Intl.DateTimeFormat(undefined, { month: "short", timeZone: timezone || "UTC" }).format(date).toUpperCase(),
      day: new Intl.DateTimeFormat(undefined, { day: "numeric", timeZone: timezone || "UTC" }).format(date),
      pending: false,
    };
  } catch {
    return { month: "CAL", day: "—", pending: true };
  }
}

function sourceLabel(source: string): string {
  return source.replace(/[-_]/g, " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function attentionTotal(attention: ProjectAttention): number {
  return Object.values(attention).reduce((total, item) => total + item.total, 0);
}

function clearUnreadAttention(attention: ProjectAttention, projectId: string): ProjectAttention {
  const pending = attention[projectId]?.pending_questions || 0;
  return {
    ...attention,
    [projectId]: { unread_messages: 0, pending_questions: pending, total: pending },
  };
}

function timezoneSettingsError(cause: unknown): string {
  const message = cause instanceof Error ? cause.message : "Timezone settings unavailable";
  return message.includes("unknown_method") ? "Restart the Alphonse daemon once to enable timezone settings." : message;
}

function taskProgressIds(events: Array<{ event: { type: string; name?: string; value?: unknown } }>): string[] {
  const ids = new Set<string>();
  for (const item of events) {
    const value = item.event.name === "a2ui.envelope" && typeof item.event.value === "object" && item.event.value !== null ? item.event.value as { createSurface?: { surfaceId?: unknown } } : null;
    const surfaceId = value?.createSurface?.surfaceId;
    if (typeof surfaceId === "string" && surfaceId.startsWith("task-progress:")) ids.add(surfaceId.slice("task-progress:".length));
  }
  return [...ids];
}

function MessageBubble({ message, timezone }: { message: ChatMessage; timezone: string }) {
  const timestamp = message.created_at ? formatMessageTime(message.created_at, timezone) : null;
  return <article className={`message ${message.role}`}>
    {message.source && !["desktop", "ledger"].includes(message.source) && <small className="message-source" title={`Sent from ${message.source}`}>↗ {sourceLabel(message.source)}</small>}
    {message.role === "assistant" ? <div className="message-markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div> : message.content}
    {timestamp && <time className="message-timestamp" dateTime={message.created_at} title={timestamp.tooltip}>{timestamp.visible}</time>}
  </article>;
}

function TaskProgressBubble({ surface }: { surface: A2uiSurface }) {
  const text = (id: string) => String(surface.components[id]?.text || "").trim();
  const steps = Object.values(surface.components).filter((component) => component.id.startsWith("step_") && component.text).sort((left, right) => left.id.localeCompare(right.id, undefined, { numeric: true }));
  const criteria = text("criteria").replace(/^Acceptance criteria\n?/, "").trim();
  const visibleCriteria = criteria === "- (none)" ? "" : criteria;
  const hasConcreteProgress = Boolean(visibleCriteria || text("intention") || steps.length || text("tool"));
  return <article className="message assistant task-progress-bubble" aria-live="polite">
    <div className="task-progress-content">
      <div className="task-progress-heading"><span className="task-progress-spinner" aria-hidden="true">◌</span><strong>Alphonse is working</strong></div>
      {!hasConcreteProgress && text("summary") && <p>{text("summary")}</p>}
      {visibleCriteria && <section><small>Acceptance criteria</small><div className="task-progress-checklist"><ReactMarkdown remarkPlugins={[remarkGfm]}>{acceptanceCriteriaMarkdown(visibleCriteria)}</ReactMarkdown></div></section>}
      {text("intention") && <section><small>Intention</small><p>{text("intention").replace(/^Intention:\s*/, "")}</p></section>}
      {steps.length ? <section className="task-progress-trace"><small>Work log</small>{steps.map((step) => <pre className="task-progress-detail" key={step.id}>{step.text}</pre>)}</section> : <>{text("tool") && <p className="task-progress-detail">{text("tool")}</p>}{text("arguments") && <pre className="task-progress-detail">{text("arguments")}</pre>}{text("result") && <pre className="task-progress-detail">{text("result")}</pre>}</>}
    </div>
  </article>;
}

function questionTaskId(surface: A2uiSurface): string {
  const question = surface.dataModel.question;
  return typeof question === "object" && question !== null && "task_id" in question ? String((question as { task_id?: unknown }).task_id || "") : "";
}

function acceptanceCriteriaMarkdown(value: string): string {
  return value.split("\n").map((line) => line.replace(/^\s*\d+\s*\.\s*-?\s*(\[[ xX]\])\s*/, "- $1 ")).join("\n");
}

function ScheduledTasksModal({ actorUserId, initialTaskId = "", onClose }: { actorUserId: string; initialTaskId?: string; onClose: () => void }) {
  const [users, setUsers] = useState<ManagedUser[]>([]); const [ownerId, setOwnerId] = useState(actorUserId); const [status, setStatus] = useState("");
  const [tasks, setTasks] = useState<ScheduledTask[]>([]); const [selected, setSelected] = useState<ScheduledTask | null>(null); const [executions, setExecutions] = useState<ScheduledExecution[]>([]);
  const [name, setName] = useState(""); const [prompt, setPrompt] = useState(""); const [notice, setNotice] = useState(""); const [removeOpen, setRemoveOpen] = useState(false);
  const isAdmin = users.some((item) => item.user_id === actorUserId && item.role === "admin");
  const load = useCallback(async () => {
    try {
      const [taskResult, userResult] = await Promise.all([
        daemonRequest<{ tasks: ScheduledTask[] }>("scheduled_tasks", { actor_user_id: actorUserId, owner_user_id: ownerId, status }),
        daemonRequest<{ users: ManagedUser[] }>("users"),
      ]);
      setTasks(taskResult.tasks); setUsers(userResult.users); setNotice("");
      setSelected((current) => current ? taskResult.tasks.find((task) => task.scheduled_task_id === current.scheduled_task_id) || null : null);
    } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Scheduled tasks could not be loaded."); }
  }, [actorUserId, ownerId, status]);
  useEffect(() => { void load(); }, [load]);
  const select = async (task: ScheduledTask) => {
    setSelected(task); setName(task.name); setPrompt(task.prompt); setRemoveOpen(false);
    try { const result = await daemonRequest<{ executions: ScheduledExecution[] }>("scheduled_executions", { actor_user_id: actorUserId, scheduled_task_id: task.scheduled_task_id, limit: 8 }); setExecutions(result.executions); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Execution history could not be loaded."); }
  };
  useEffect(() => {
    const task = tasks.find((item) => item.scheduled_task_id === initialTaskId);
    if (task && selected?.scheduled_task_id !== task.scheduled_task_id) void select(task);
  }, [initialTaskId, tasks]);
  const mutate = async (method: string, extra: Record<string, unknown> = {}) => {
    if (!selected) return;
    try {
      await daemonRequest(method, { actor_user_id: actorUserId, scheduled_task_id: selected.scheduled_task_id, ...extra });
      setNotice(method === "delete_scheduled_task" ? "Task permanently deleted." : "Task updated."); setRemoveOpen(false);
      if (method === "delete_scheduled_task") { setSelected(null); setExecutions([]); }
      await load();
    } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Scheduled task could not be updated."); }
  };
  const editable = selected?.status === "active" || selected?.status === "paused";
  return <ModalFrame title="Scheduled tasks" onClose={onClose}>
    {isAdmin && <div className="form-field"><label htmlFor="scheduled-owner">User</label><select id="scheduled-owner" value={ownerId} onChange={(event) => { setOwnerId(event.target.value); setSelected(null); setExecutions([]); }}>{users.map((item) => <option key={item.user_id} value={item.user_id}>{item.display_name}</option>)}</select></div>}
    <div className="form-field"><label htmlFor="scheduled-status">Status</label><select id="scheduled-status" value={status} onChange={(event) => { setStatus(event.target.value); setSelected(null); setExecutions([]); }}><option value="">All statuses</option>{["active", "paused", "completed", "cancelled", "failed"].map((value) => <option key={value} value={value}>{value}</option>)}</select></div>
    <div className="stack scheduled-task-list">{tasks.length ? tasks.map((task) => {
      const calendar = calendarDateLabel(task.next_run_at, task.timezone);
      return <button className={`scheduled-task-row${task.scheduled_task_id === selected?.scheduled_task_id ? " selected" : ""}`} key={task.scheduled_task_id} onClick={() => void select(task)}>
        <span className="scheduled-task-row-calendar" aria-label={calendar.pending ? "Schedule pending" : `Next run on ${calendar.month} ${calendar.day}`}><span>{calendar.month}</span><strong>{calendar.day}</strong></span>
        <span className="scheduled-task-row-summary"><span className="scheduled-task-row-title"><strong>{task.name}</strong><small className="scheduled-task-row-status">{task.status}</small></span><small>{calendar.pending ? "Schedule pending" : `Next: ${dateLabel(task.next_run_at, task.timezone)}`}</small><small>{scheduleLabel(task.schedule)} · {task.timezone}</small><small>Owner: {users.find((item) => item.user_id === task.owner_user_id)?.display_name || task.owner_user_id} · Latest: {task.latest_execution?.status || "not run"}</small></span>
      </button>;
    }) : <p>No scheduled tasks match this filter.</p>}</div>
    {selected && <section className="scheduled-task-detail">
      <div className="form-field"><label htmlFor="scheduled-name">Task name</label><input id="scheduled-name" value={name} disabled={!editable} onChange={(event) => setName(event.target.value)} /></div>
      <div className="form-field"><label htmlFor="scheduled-prompt">Stored prompt</label><textarea id="scheduled-prompt" value={prompt} disabled={!editable} onChange={(event) => setPrompt(event.target.value)} rows={6} /></div>
      <div className="form-field task-readonly"><label>Owner</label><output>{users.find((item) => item.user_id === selected.owner_user_id)?.display_name || selected.owner_user_id}</output><small>{selected.owner_user_id}</small></div>
      <div className="form-field task-readonly"><label>Schedule</label><output>{scheduleLabel(selected.schedule)} ({selected.timezone})</output><small>Next run: {dateLabel(selected.next_run_at)} · Last run: {dateLabel(selected.last_run_at)}</small></div>
      {editable && <button onClick={() => void mutate("update_scheduled_task", { name, prompt })}>Save name and prompt</button>}
      <div className="dialog-actions">{selected.status === "active" && <button onClick={() => void mutate("pause_schedule")}>Suspend</button>}{selected.status === "paused" && <button onClick={() => void mutate("resume_schedule")}>Resume</button>}<button onClick={() => setRemoveOpen((value) => !value)}>Remove…</button></div>
      {removeOpen && <div className="form-field destructive-action"><p>Cancel keeps this task and its execution history. Permanent deletion removes both.</p><div className="dialog-actions">{["active", "paused"].includes(selected.status) && <button onClick={() => void mutate("cancel_schedule")}>Cancel task</button>}<button onClick={() => void mutate("delete_scheduled_task")}>Delete permanently</button></div></div>}
      <h3>Recent executions</h3><div className="execution-list">{executions.length ? executions.map((execution) => <p key={execution.run_id}><strong>{execution.status}</strong> · {dateLabel(execution.started_at)}{execution.error || execution.last_error ? ` · ${execution.error || execution.last_error}` : ""}</p>) : <p>No executions yet.</p>}</div>
    </section>}
    <p>{notice}</p>
  </ModalFrame>;
}

function UsersModal({ onClose }: { onClose: () => void }) {
  const [users, setUsers] = useState<ManagedUser[]>([]); const [selectedId, setSelectedId] = useState("");
  const [name, setName] = useState(""); const [role, setRole] = useState("member"); const [active, setActive] = useState(true); const [context, setContext] = useState(""); const [notice, setNotice] = useState(""); const [addresses, setAddresses] = useState<ManagedAddress[]>([]); const [communicationOptions, setCommunicationOptions] = useState<Array<{ integration_id: string; provider_key: string; label: string }>>([]); const [showCommunicationForm, setShowCommunicationForm] = useState(false); const [integrationId, setIntegrationId] = useState(""); const [providerUserId, setProviderUserId] = useState(""); const [channelTarget, setChannelTarget] = useState(""); const [confirmation, setConfirmation] = useState("");
  const load = useCallback(async () => { const result = await daemonRequest<{ users: ManagedUser[] }>("users"); setUsers(result.users); return result.users; }, []);
  useEffect(() => { void load(); void daemonRequest<{ integrations: Array<{ provider_key: string; display_name: string; integration: Record<string, unknown> | null }> }>("integrations").then((result) => setCommunicationOptions(result.integrations.map((item) => ({ integration_id: String(item.integration?.integration_id || `${item.provider_key}-home`), provider_key: item.provider_key, label: String(item.integration?.display_name || item.display_name) })))); }, [load]);
  const select = async (item: ManagedUser) => { setSelectedId(item.user_id); setName(item.display_name); setRole(item.role); setActive(item.is_active); setAddresses(item.addresses || []); setNotice(""); setConfirmation(""); const result = await daemonRequest<{ content: string }>("user_context", { user_id: item.user_id }); setContext(result.content); };
  const create = async () => { const result = await daemonRequest<{ user: ManagedUser }>("create_user", { display_name: name, role }); await load(); await select(result.user); setNotice("User created."); };
  const save = async () => { if (!selectedId) return create(); await daemonRequest("update_user", { user_id: selectedId, display_name: name, role, is_active: active }); await daemonRequest("save_user_context", { user_id: selectedId, content: context }); await load(); setNotice("User saved."); };
  const deactivate = async () => { if (!selectedId) return; await daemonRequest("update_user", { user_id: selectedId, is_active: !active }); setActive(!active); await load(); setNotice(active ? "User deactivated. Existing data is preserved." : "User reactivated."); };
  const selectCommunication = (nextIntegrationId: string) => { setIntegrationId(nextIntegrationId); const existing = addresses.find((address) => address.integration_id === nextIntegrationId); setProviderUserId(existing?.provider_user_id || ""); setChannelTarget(existing?.channel_target || ""); };
  const bind = async () => { if (!selectedId || !integrationId || !providerUserId) return; const option = communicationOptions.find((item) => item.integration_id === integrationId); if (!option) return; await daemonRequest("bind_user_address", { user_id: selectedId, integration_id: option.integration_id, provider_key: option.provider_key, provider_user_id: providerUserId, channel_target: channelTarget || providerUserId, is_preferred: true }); setShowCommunicationForm(false); await load(); setAddresses((current) => [{ address_id: `pending-${integrationId}`, integration_id: option.integration_id, provider_key: option.provider_key, provider_user_id: providerUserId, channel_target: channelTarget || providerUserId, is_preferred: true }, ...current.filter((address) => address.integration_id !== option.integration_id)]); setNotice("Preferred communication address saved."); };
  const makePreferred = async (address: ManagedAddress) => { await daemonRequest("bind_user_address", { user_id: selectedId, integration_id: address.integration_id, provider_key: address.provider_key, provider_user_id: address.provider_user_id, channel_target: address.channel_target, is_preferred: true }); setAddresses((current) => current.map((item) => ({ ...item, is_preferred: item.address_id === address.address_id }))); };
  const removeAddress = async (addressId: string) => { await daemonRequest("remove_user_address", { address_id: addressId }); await load(); setAddresses((current) => current.filter((item) => item.address_id !== addressId)); };
  const remove = async () => { if (!selectedId) return; try { const result = await daemonRequest<{ deleted: number }>("delete_user", { user_id: selectedId, confirmation }); if (result.deleted) { beginCreate(); await load(); setNotice("User and all owned data permanently deleted."); } } catch (cause) { setNotice(cause instanceof Error ? cause.message : "User could not be deleted."); } };
  const beginCreate = () => { setSelectedId(""); setName(""); setRole("member"); setActive(true); setContext("# User Context\n"); setAddresses([]); setShowCommunicationForm(false); setNotice(""); setConfirmation(""); };
  return <ModalFrame title="Users" onClose={onClose}>
    <div className="stack">{users.map((item) => <button className={item.user_id === selectedId ? "selected user-card" : "user-card"} key={item.user_id} onClick={() => void select(item)}><span className="user-card-icon" aria-hidden="true">👤</span><span>{item.display_name}<small>{item.is_active ? item.role : "inactive"}</small></span></button>)}<button onClick={beginCreate}>New user</button></div>
    <div className="form-field"><label htmlFor="user-name">Name</label><input id="user-name" value={name} onChange={(event) => setName(event.target.value)} placeholder="Name" /></div>
    <div className="form-field"><label htmlFor="user-role">Role</label><select id="user-role" value={role} onChange={(event) => setRole(event.target.value)}><option value="member">Member</option><option value="caregiver">Caregiver</option><option value="admin">Admin</option></select></div>
    <div className="form-field"><label htmlFor="user-context">User context</label><textarea id="user-context" value={context} onChange={(event) => setContext(event.target.value)} rows={6} /></div>
    {selectedId && <><div className="form-field"><label>Canonical user ID</label><output>{selectedId}</output></div><h3>Preferred communication</h3>{addresses.map((address) => <div className="form-field" key={address.address_id}><label>{address.provider_key}: {address.provider_user_id}{address.is_preferred ? " (preferred)" : ""}</label><div className="dialog-actions">{!address.is_preferred && <button onClick={() => void makePreferred(address)}>Set preferred</button>}<button onClick={() => void removeAddress(address.address_id)}>Remove address</button></div></div>)}<button onClick={() => { setShowCommunicationForm(true); selectCommunication(communicationOptions[0]?.integration_id || ""); }}>Add communication</button>{showCommunicationForm && <div className="form-field"><label htmlFor="address-integration">Integration</label><select id="address-integration" value={integrationId} onChange={(event) => selectCommunication(event.target.value)}>{communicationOptions.map((option) => <option key={option.integration_id} value={option.integration_id}>{option.label}</option>)}</select><label htmlFor="address-user-id">Provider user ID</label><input id="address-user-id" value={providerUserId} onChange={(event) => setProviderUserId(event.target.value)} /><label htmlFor="address-target">Channel target</label><input id="address-target" value={channelTarget} onChange={(event) => setChannelTarget(event.target.value)} placeholder="Optional; provider user ID by default" /><button onClick={() => void bind()}>Save and set preferred</button></div>}</>}
    <div className="dialog-actions"><button onClick={() => void save()}>{selectedId ? "Save user" : "Create user"}</button>{selectedId && <button onClick={() => void deactivate()}>{active ? "Deactivate user" : "Reactivate user"}</button>}</div>
    {selectedId && <><h3>Permanent deletion</h3><div className="form-field"><p>This permanently deletes the user, profile, managed projects, schedules, pending questions, and channel mappings.</p><label htmlFor="delete-confirmation">Type this canonical user ID to confirm deletion</label><input id="delete-confirmation" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder={selectedId} /><button disabled={confirmation !== selectedId} onClick={() => void remove()}>Delete permanently</button></div></>}
    <p>{notice}</p>
  </ModalFrame>;
}

function ProjectsModal({ user, active, attention, onSelect, onSettings, onClose }: { user: string; active: Project | null; attention: ProjectAttention; onSelect: (project: Project) => void; onSettings: (project: ManagedProject) => void; onClose: () => void }) {
  const [projects, setProjects] = useState<ManagedProject[]>([]); const [status, setStatus] = useState(""); const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState(""); const [description, setDescription] = useState(""); const [visibility, setVisibility] = useState<"private" | "shared">("private"); const [rootPath, setRootPath] = useState(""); const [mode, setMode] = useState<"create" | "import">("create"); const [notice, setNotice] = useState("");
  const load = useCallback(async () => { try { const result = await daemonRequest<{ projects: ManagedProject[] }>("manageable_projects", { user, status }); setProjects(result.projects); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Projects could not be loaded."); } }, [status, user]);
  useEffect(() => { void load(); }, [load]);
  const createOrImport = async (event: FormEvent) => { event.preventDefault(); try { const method = mode === "import" ? "import_project" : "create_project"; const result = await daemonRequest<{ project: ManagedProject }>(method, { user, name, description, root_path: rootPath, visibility }); setNotice(mode === "import" ? "Project imported." : "Project created."); setShowCreate(false); await load(); onSettings(result.project); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Project could not be created."); } };
  return <ModalFrame title="Projects" onClose={onClose}>
    <div className="project-list-toolbar"><div className="form-field"><label htmlFor="project-status-filter">Show</label><select id="project-status-filter" value={status} onChange={(event) => setStatus(event.target.value)}><option value="">All projects</option><option value="active">Active</option><option value="archived">Archived</option></select></div><button onClick={() => setShowCreate((value) => !value)}>{showCreate ? "Cancel" : "New project"}</button></div>
    {showCreate && <form className="project-create" onSubmit={createOrImport}><h3>{mode === "import" ? "Import existing folder" : "New project"}</h3><div className="dialog-actions"><button type="button" onClick={() => setMode("create")}>New</button><button type="button" onClick={() => setMode("import")}>Import</button></div><div className="form-field"><label htmlFor="new-project-name">Name</label><input id="new-project-name" value={name} onChange={(event) => setName(event.target.value)} required /></div><div className="form-field"><label htmlFor="new-project-description">Description</label><input id="new-project-description" value={description} onChange={(event) => setDescription(event.target.value)} /></div><div className="form-field"><label htmlFor="new-project-visibility">Visibility</label><select id="new-project-visibility" value={visibility} onChange={(event) => setVisibility(event.target.value as "private" | "shared")}><option value="private">Private</option><option value="shared">Shared</option></select></div><div className="form-field"><label htmlFor="new-project-path">{mode === "import" ? "Existing folder" : "Parent directory (optional)"}</label><input id="new-project-path" value={rootPath} onChange={(event) => setRootPath(event.target.value)} required={mode === "import"} /></div><button>{mode === "import" ? "Import project" : "Create project"}</button></form>}
    <div className="stack project-list">{projects.length ? projects.map((item) => <article className={item.project_id === active?.project_id ? "selected project-card" : "project-card"} key={item.project_id}><span className="project-card-icon" aria-hidden="true">🗂️</span><span className="project-card-summary"><strong>{item.name}{(attention[item.project_id]?.total || 0) > 0 && <span className="attention-badge">{attention[item.project_id].total}</span>}</strong><small>👤 {item.owner?.display_name || item.owner_user_id}</small><small>{item.status} · {item.visibility} · Updated {dateLabel(item.updated_at)}</small>{(attention[item.project_id]?.total || 0) > 0 && <small>{attention[item.project_id].unread_messages} new message{attention[item.project_id].unread_messages === 1 ? "" : "s"} · {attention[item.project_id].pending_questions} pending question{attention[item.project_id].pending_questions === 1 ? "" : "s"}</small>}</span><span className="project-card-actions">{item.status === "active" && <button onClick={() => onSelect(item)}>Open project</button>}<button className="secondary" onClick={() => onSettings(item)}>Settings</button></span></article>) : <p>No projects match this filter.</p>}</div>
    <p>{notice}</p>
  </ModalFrame>;
}

function ProjectSettingsModal({ user, project, onBack, onClose }: { user: string; project: ManagedProject; onBack: () => void; onClose: () => void }) {
  const [name, setName] = useState(project.name); const [description, setDescription] = useState(project.description); const [visibility, setVisibility] = useState(project.visibility); const [context, setContext] = useState(""); const [confirmation, setConfirmation] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => { void daemonRequest<{ content: string }>("project_context", { user, project_id: project.project_id }).then((result) => setContext(result.content)).catch((cause: unknown) => setNotice(cause instanceof Error ? cause.message : "Project context could not be loaded.")); }, [project.project_id, user]);
  const save = async () => { try { await daemonRequest("update_project", { user, project_id: project.project_id, name, description, visibility }); await daemonRequest("save_project_context", { user, project_id: project.project_id, content: context }); setNotice("Project settings saved."); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Project could not be saved."); } };
  const lifecycle = async (method: "archive_project" | "restore_project" | "delete_project") => { try { await daemonRequest(method, method === "delete_project" ? { user, project_id: project.project_id, confirmation } : { user, project_id: project.project_id }); setNotice(method === "archive_project" ? "Project archived." : method === "restore_project" ? "Project restored." : "Project removed."); if (method === "delete_project") onBack(); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Project lifecycle action failed."); } };
  const archived = project.status === "archived";
  return <ModalFrame title={`${project.name} settings`} onClose={onClose}>
    <button className="back-button" onClick={onBack}>← All projects</button>
    <section className="project-detail"><div className="form-field"><label htmlFor="project-name">Name</label><input id="project-name" disabled={archived} value={name} onChange={(event) => setName(event.target.value)} /></div><div className="form-field"><label htmlFor="project-description">Description</label><input id="project-description" disabled={archived} value={description} onChange={(event) => setDescription(event.target.value)} /></div><div className="form-field"><label htmlFor="project-visibility">Visibility</label><select id="project-visibility" disabled={archived} value={visibility} onChange={(event) => setVisibility(event.target.value as "private" | "shared")}><option value="private">Private</option><option value="shared">Shared</option></select></div><div className="form-field task-readonly"><label>Owner</label><output>👤 {project.owner?.display_name || project.owner_user_id}</output><small>{project.owner_user_id}</small></div><div className="form-field task-readonly"><label>Project directory</label><output>{project.root_path}</output><small>Created {dateLabel(project.created_at)} · Updated {dateLabel(project.updated_at)}</small></div><div className="form-field"><label htmlFor="project-context">Project context</label><textarea id="project-context" disabled={archived} rows={8} value={context} onChange={(event) => setContext(event.target.value)} /></div>{!archived ? <div className="dialog-actions"><button onClick={() => void save()}>Save changes</button><button className="secondary" onClick={() => void lifecycle("archive_project")}>Archive</button></div> : <div className="dialog-actions"><button onClick={() => void lifecycle("restore_project")}>Restore</button></div>}<div className="form-field destructive-action"><label htmlFor="project-delete-confirmation">Type the project ID to remove it</label><input id="project-delete-confirmation" value={confirmation} placeholder={project.project_id} onChange={(event) => setConfirmation(event.target.value)} /><button disabled={confirmation !== project.project_id} onClick={() => void lifecycle("delete_project")}>Remove permanently</button></div></section>
    <p>{notice}</p>
  </ModalFrame>;
}

function ProjectContextModal({ user, project, onClose }: { user: string; project: Project | null; onClose: () => void }) {
  const [content, setContent] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => { if (project) void daemonRequest<{ content: string }>("project_context", { user, project_id: project.project_id }).then((result) => setContent(result.content)); }, [project, user]);
  if (!project) return <ModalFrame title="Project context" onClose={onClose}><p>Select a project before editing its context.</p></ModalFrame>;
  return <ModalFrame title={`${project.name} context`} onClose={onClose}><textarea value={content} onChange={(event) => setContent(event.target.value)} rows={12} /><button onClick={() => void daemonRequest("save_project_context", { user, project_id: project.project_id, content }).then(() => setNotice("Saved."))}>Save context</button><p>{notice}</p></ModalFrame>;
}

function IntegrationsSettingsSection({ user }: { user: string }) {
  const [integrationId, setIntegrationId] = useState("telegram-home"); const [displayName, setDisplayName] = useState("Telegram");
  const [token, setToken] = useState(""); const [telegramUserId, setTelegramUserId] = useState(""); const [allowedChatIds, setAllowedChatIds] = useState("");
  const [pollInterval, setPollInterval] = useState("1"); const [enabled, setEnabled] = useState(false); const [presenceEnabled, setPresenceEnabled] = useState(true); const [notice, setNotice] = useState("");
  useEffect(() => { void daemonRequest<{ integrations: Array<{ provider_key: string; integration: Record<string, unknown> | null }> }>("integrations").then((result) => {
    const integration = result.integrations.find((item) => item.provider_key === "telegram")?.integration; if (!integration) return;
    const config = (integration.config as Record<string, unknown> | undefined) ?? {};
    setIntegrationId(String(integration.integration_id || "telegram-home")); setDisplayName(String(integration.display_name || "Telegram")); setEnabled(Boolean(integration.enabled));
    setTelegramUserId(String(config.telegram_user_id || "")); setAllowedChatIds(Array.isArray(config.allowed_chat_ids) ? config.allowed_chat_ids.join(", ") : "");
    setPollInterval(String(config.poll_interval_sec || 1)); setPresenceEnabled(config.presence_enabled !== false);
  }); }, []);
  const save = async () => { await daemonRequest("save_telegram_integration", { user, values: { integration_id: integrationId, display_name: displayName, enabled, bot_token: token, telegram_user_id: telegramUserId, allowed_chat_ids: allowedChatIds, poll_interval_sec: pollInterval, presence_enabled: presenceEnabled } }); setToken(""); setNotice("Saved and integrations restarted."); };
  return <><section className="settings-panel">
    <h3>Telegram integration</h3>
    <div className="form-field">
      <label htmlFor="telegram-integration-id">Integration ID</label>
      <input id="telegram-integration-id" value={integrationId} onChange={(event) => setIntegrationId(event.target.value)} />
    </div>
    <div className="form-field">
      <label htmlFor="telegram-display-name">Display name</label>
      <input id="telegram-display-name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} />
    </div>
    <div className="form-field">
      <label htmlFor="telegram-bot-token">Bot token <span className="field-help" title="Create this token with Telegram's @BotFather. Leave this field blank to keep the token already saved for this integration.">?</span></label>
      <input id="telegram-bot-token" type="password" value={token} onChange={(event) => setToken(event.target.value)} placeholder="Leave blank to keep the current token" autoComplete="new-password" />
    </div>
    <div className="form-field">
      <label htmlFor="telegram-user-id">Telegram user ID <span className="field-help" title="The Telegram numeric user ID to associate with the currently selected Alphonse user.">?</span></label>
      <input id="telegram-user-id" value={telegramUserId} onChange={(event) => setTelegramUserId(event.target.value)} />
    </div>
    <div className="form-field">
      <label htmlFor="telegram-allowed-chat-ids">Allowed chat IDs <span className="field-help" title="Enter one or more Telegram chat IDs, separated by commas. Leave empty to allow all chats.">?</span></label>
      <input id="telegram-allowed-chat-ids" value={allowedChatIds} onChange={(event) => setAllowedChatIds(event.target.value)} placeholder="123456, -1001234567890" />
    </div>
    <div className="form-field">
      <label htmlFor="telegram-poll-interval">Poll interval (seconds) <span className="field-help" title="How often Alphonse checks Telegram for new messages. Use a positive number; the default is 1 second.">?</span></label>
      <input id="telegram-poll-interval" inputMode="decimal" value={pollInterval} onChange={(event) => setPollInterval(event.target.value)} />
    </div>
    <div className="form-field checkbox-field">
      <label htmlFor="telegram-enabled"><input id="telegram-enabled" type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} /> Enable Telegram integration</label>
      <small>Starts the Telegram bridge to receive messages and send replies. A bot token is required.</small>
    </div>
    <div className="form-field checkbox-field">
      <label htmlFor="telegram-presence-enabled"><input id="telegram-presence-enabled" type="checkbox" checked={presenceEnabled} onChange={(event) => setPresenceEnabled(event.target.checked)} /> Show Telegram presence</label>
      <small>Shows typing indicators and status reactions while Alphonse is working on a Telegram message.</small>
    </div>
    <button onClick={() => void save()}>Save integration</button><p>{notice}</p>
  </section><DiscordIntegrationSettingsSection user={user} /></>;
}

function DiscordIntegrationSettingsSection({ user }: { user: string }) {
  const [integrationId, setIntegrationId] = useState("discord-home"); const [displayName, setDisplayName] = useState("Discord");
  const [token, setToken] = useState(""); const [discordUserId, setDiscordUserId] = useState(""); const [allowedGuildIds, setAllowedGuildIds] = useState(""); const [allowedChannelIds, setAllowedChannelIds] = useState("");
  const [enabled, setEnabled] = useState(false); const [presenceEnabled, setPresenceEnabled] = useState(true); const [notice, setNotice] = useState("");
  useEffect(() => { void daemonRequest<{ integrations: Array<{ provider_key: string; integration: Record<string, unknown> | null }> }>("integrations").then((result) => {
    const integration = result.integrations.find((item) => item.provider_key === "discord")?.integration; if (!integration) return;
    const config = (integration.config as Record<string, unknown> | undefined) ?? {};
    setIntegrationId(String(integration.integration_id || "discord-home")); setDisplayName(String(integration.display_name || "Discord")); setEnabled(Boolean(integration.enabled));
    setDiscordUserId(String(config.discord_user_id || "")); setAllowedGuildIds(Array.isArray(config.allowed_guild_ids) ? config.allowed_guild_ids.join(", ") : ""); setAllowedChannelIds(Array.isArray(config.allowed_channel_ids) ? config.allowed_channel_ids.join(", ") : ""); setPresenceEnabled(config.presence_enabled !== false);
  }).catch((cause: unknown) => setNotice(cause instanceof Error ? cause.message : "Discord settings unavailable")); }, []);
  const save = async () => { try { await daemonRequest("save_discord_integration", { user, values: { integration_id: integrationId, display_name: displayName, enabled, bot_token: token, discord_user_id: discordUserId, allowed_guild_ids: allowedGuildIds, allowed_channel_ids: allowedChannelIds, presence_enabled: presenceEnabled } }); setToken(""); setNotice("Saved and integrations restarted."); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Save failed"); } };
  return <section className="settings-panel">
    <h3>Discord integration</h3>
    <div className="form-field"><label htmlFor="discord-integration-id">Integration ID</label><input id="discord-integration-id" value={integrationId} onChange={(event) => setIntegrationId(event.target.value)} /></div>
    <div className="form-field"><label htmlFor="discord-display-name">Display name</label><input id="discord-display-name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} /></div>
    <div className="form-field"><label htmlFor="discord-bot-token">Bot token <span className="field-help" title="Create a bot token in the Discord Developer Portal. Leave blank to keep the saved token.">?</span></label><input id="discord-bot-token" type="password" value={token} onChange={(event) => setToken(event.target.value)} placeholder="Leave blank to keep the current token" autoComplete="new-password" /></div>
    <div className="form-field"><label htmlFor="discord-user-id">Discord user ID <span className="field-help" title="The Discord user ID to associate with the selected Alphonse user.">?</span></label><input id="discord-user-id" value={discordUserId} onChange={(event) => setDiscordUserId(event.target.value)} /></div>
    <div className="form-field"><label htmlFor="discord-allowed-guild-ids">Allowed guild IDs <span className="field-help" title="Optional comma-separated allow-list. Leave blank to allow every guild the bot can read.">?</span></label><input id="discord-allowed-guild-ids" value={allowedGuildIds} onChange={(event) => setAllowedGuildIds(event.target.value)} placeholder="123456789012345678" /></div>
    <div className="form-field"><label htmlFor="discord-allowed-channel-ids">Allowed channel IDs <span className="field-help" title="Optional comma-separated allow-list. Leave blank to allow every readable channel.">?</span></label><input id="discord-allowed-channel-ids" value={allowedChannelIds} onChange={(event) => setAllowedChannelIds(event.target.value)} placeholder="123456789012345678" /></div>
    <div className="form-field checkbox-field"><label htmlFor="discord-enabled"><input id="discord-enabled" type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} /> Enable Discord integration</label><small>Starts the Discord Gateway bridge. A bot token and Message Content Intent are required.</small></div>
    <div className="form-field checkbox-field"><label htmlFor="discord-presence-enabled"><input id="discord-presence-enabled" type="checkbox" checked={presenceEnabled} onChange={(event) => setPresenceEnabled(event.target.checked)} /> Show Discord presence</label><small>Shows typing indicators and status reactions while Alphonse is working.</small></div>
    <button onClick={() => void save()}>Save integration</button><p>{notice}</p>
  </section>;
}

type AutomationCatalog = { workers: Array<{ worker_id: string; display_name: string; allowed_event_types: string[]; enabled: boolean }>; event_types: Array<{ event_type: string; version: string; schema: Record<string, unknown>; max_history: number; enabled: boolean }>; automations: Array<{ automation_id: string; name: string; trigger_kind: string; trigger: { event_type?: string; event_version?: string; filters?: Record<string, unknown> }; status: string }>; events: Array<{ source_event_id: string; event_type: string; event_version: string; occurred_at: string; payload: Record<string, unknown>; dispatch_count: number }> };

function AutomationsSettingsSection({ user }: { user: string }) {
  const [catalog, setCatalog] = useState<AutomationCatalog | null>(null); const [notice, setNotice] = useState("");
  const [workerId, setWorkerId] = useState(""); const [workerName, setWorkerName] = useState(""); const [workerTypes, setWorkerTypes] = useState("");
  const [eventType, setEventType] = useState(""); const [eventVersion, setEventVersion] = useState("1"); const [schema, setSchema] = useState('{"type":"object"}'); const [maxHistory, setMaxHistory] = useState("500");
  const [automationName, setAutomationName] = useState(""); const [automationPrompt, setAutomationPrompt] = useState(""); const [automationType, setAutomationType] = useState(""); const [automationVersion, setAutomationVersion] = useState("1"); const [filters, setFilters] = useState("{}");
  const load = async () => { try { setCatalog(await daemonRequest<AutomationCatalog>("automation_catalog")); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Automations unavailable"); } };
  useEffect(() => { void load(); }, []);
  const saveWorker = async () => { try { await daemonRequest("register_event_worker", { worker_id: workerId, display_name: workerName, allowed_event_types: workerTypes.split(",").map((item) => item.trim()).filter(Boolean), enabled: true }); setNotice("Worker registered."); setWorkerId(""); setWorkerName(""); setWorkerTypes(""); await load(); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Worker save failed"); } };
  const saveEventType = async () => { try { await daemonRequest("register_event_type", { event_type: eventType, version: eventVersion, schema: JSON.parse(schema), max_history: Number(maxHistory), enabled: true }); setNotice("Event type registered."); setEventType(""); await load(); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Event type save failed"); } };
  const saveAutomation = async () => { try { await daemonRequest("create_event_automation", { user, name: automationName, prompt: automationPrompt, event_type: automationType, event_version: automationVersion, filters: JSON.parse(filters), enabled: true }); setNotice("Event automation created."); setAutomationName(""); setAutomationPrompt(""); setFilters("{}"); await load(); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Automation save failed"); } };
  return <section className="settings-panel">
    <h3>Automations</h3><p>Scheduled tasks and event automations share this catalog. Event automations run a fixed prompt when a registered local worker publishes a matching validated event.</p>
    <h4>Workers</h4><div className="form-field"><label>Worker ID</label><input value={workerId} onChange={(event) => setWorkerId(event.target.value)} placeholder="plant-monitor" /></div><div className="form-field"><label>Display name</label><input value={workerName} onChange={(event) => setWorkerName(event.target.value)} /></div><div className="form-field"><label>Allowed event types</label><input value={workerTypes} onChange={(event) => setWorkerTypes(event.target.value)} placeholder="plant.soil_humidity, doorbell.rang" /></div><button onClick={() => void saveWorker()}>Register worker</button>
    <h4>Event types</h4><div className="form-field"><label>Event type</label><input value={eventType} onChange={(event) => setEventType(event.target.value)} placeholder="plant.soil_humidity" /></div><div className="form-field"><label>Version</label><input value={eventVersion} onChange={(event) => setEventVersion(event.target.value)} /></div><div className="form-field"><label>JSON Schema</label><textarea value={schema} rows={5} onChange={(event) => setSchema(event.target.value)} /></div><div className="form-field"><label>History limit</label><input type="number" value={maxHistory} onChange={(event) => setMaxHistory(event.target.value)} /></div><button onClick={() => void saveEventType()}>Register event type</button>
    <h4>Event automation</h4><div className="form-field"><label>Name</label><input value={automationName} onChange={(event) => setAutomationName(event.target.value)} /></div><div className="form-field"><label>Fixed prompt</label><textarea value={automationPrompt} rows={3} onChange={(event) => setAutomationPrompt(event.target.value)} /></div><div className="form-field"><label>Event type</label><input value={automationType} onChange={(event) => setAutomationType(event.target.value)} placeholder="plant.soil_humidity" /></div><div className="form-field"><label>Version</label><input value={automationVersion} onChange={(event) => setAutomationVersion(event.target.value)} /></div><div className="form-field"><label>Exact payload filters (JSON object)</label><textarea value={filters} rows={3} onChange={(event) => setFilters(event.target.value)} /></div><button onClick={() => void saveAutomation()}>Create event automation</button>
    <h4>Registered</h4>{catalog && <><p>Workers: {catalog.workers.map((item) => `${item.display_name} (${item.worker_id})`).join(", ") || "none"}</p><p>Event types: {catalog.event_types.map((item) => `${item.event_type}@${item.version}`).join(", ") || "none"}</p><p>Automations: {catalog.automations.map((item) => `${item.name} [${item.trigger_kind}, ${item.status}]`).join(", ") || "none"}</p><h4>Recent events</h4>{catalog.events.map((item) => <p key={`${item.event_type}:${item.source_event_id}`}>{item.event_type}@{item.event_version} · {item.source_event_id} · {item.dispatch_count} automation(s)</p>)}</>}<p>{notice}</p>
  </section>;
}

function ModelSettingsSection() {
  const [providers, setProviders] = useState<Provider[]>([]); const [settings, setSettings] = useState<InferenceSettings | null>(null); const [provider, setProvider] = useState(""); const [model, setModel] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => { void Promise.all([daemonRequest<{ providers: Provider[] }>("inference_providers"), daemonRequest<{ settings: InferenceSettings }>("inference_settings")]).then(([catalog, current]) => { setProviders(catalog.providers); setSettings(current.settings); setProvider(current.settings.provider_key); setModel(current.settings.model_id); }); }, []);
  const selected = providers.find((item) => item.provider_key === provider);
  return <section className="settings-panel"><h3>Inference model</h3><select value={provider} onChange={(event) => { setProvider(event.target.value); setModel(""); }}>{providers.map((item) => <option value={item.provider_key} key={item.provider_key}>{item.display_name}</option>)}</select><select value={model} onChange={(event) => setModel(event.target.value)}>{selected?.models.map((item) => <option value={item.model_id} key={item.model_id}>{item.display_name}</option>)}</select><button onClick={() => void daemonRequest<{ settings: InferenceSettings }>("set_inference_settings", { provider_key: provider, model_id: model }).then((result) => { setSettings(result.settings); setNotice("Validated and saved."); }).catch((cause: unknown) => setNotice(cause instanceof Error ? cause.message : "Validation failed"))}>Validate & save</button><p>{notice || settings?.validation_error}</p></section>;
}

function AgentConfigSettingsSection() {
  const [documents, setDocuments] = useState<AgentDocument[]>([]); const [fileName, setFileName] = useState(""); const [content, setContent] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => { void daemonRequest<{ documents: AgentDocument[] }>("agent_config_documents").then((result) => { setDocuments(result.documents); setFileName(result.documents[0]?.file_name || ""); }); }, []);
  useEffect(() => { if (fileName) void daemonRequest<{ document: AgentDocument }>("read_agent_config", { file_name: fileName }).then((result) => setContent(result.document.content || "")); }, [fileName]);
  return <section className="settings-panel"><h3>Agent configuration</h3><select value={fileName} onChange={(event) => setFileName(event.target.value)}>{documents.map((item) => <option value={item.file_name} key={item.file_name}>{item.display_name}</option>)}</select><textarea value={content} onChange={(event) => setContent(event.target.value)} rows={14} /><button onClick={() => void daemonRequest("save_agent_config", { file_name: fileName, content }).then(() => setNotice("Saved. Restart the daemon before new tasks use these changes."))}>Save configuration</button><p>{notice}</p></section>;
}
