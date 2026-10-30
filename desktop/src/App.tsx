import { SkillMascot, SkillAppearanceSettings } from "./SkillMascot";
import { BASE_CONTEXT, type AppearanceContext, type SkillAppearance } from "./skillAppearances";
import { DecisionContext } from "./DecisionContext";
import { FormEvent, KeyboardEvent, memo, ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CalendarClock, ChevronLeft, ChevronRight, FolderKanban, FolderOpen, Settings, UsersRound, Sparkles, MessageCircle, CircleHelp, PanelRightClose, PanelRightOpen, Palette, Wrench, Plug, Brain, Gauge, FileText } from "lucide-react";
import { open as openFileDialog } from "@tauri-apps/plugin-dialog";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { onAction } from "@tauri-apps/plugin-notification";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { daemonRequest, ensureDaemon, playAlertSound, showInFinder, stopDaemon } from "./api";
import { matchingCommands } from "./commands";
import { mergeFreshConversationHistory } from "./conversationHistory";
import { projectTaskProgress } from "./taskProgress";
import { buildConversationTimeline } from "./conversationTimeline";
import { A2uiSurfaceView, DESKTOP_CATALOG_ID, type A2uiSurface } from "./a2ui";
import { followDesktopStyle, readDesktopStyle, saveDesktopStyle, type DesktopStyle } from "./desktopStyle";
import { desktopDiagnosticBehavior, type DesktopDiagnosticMode } from "./diagnosticMode";
import { createDesktopNotifier, DESKTOP_NOTIFICATION_PREFERENCES_KEY, notificationForCompletion, notificationForQuestion, readDesktopNotificationPreferences, type DesktopNotificationPreferences } from "./desktopNotifications";
import { readDismissedScheduledSurfaces, rememberDismissedScheduledSurface, withoutDismissedSurfaces, withoutSurface } from "./dismissedSurfaces";
import { avatarState, avatarStateLabel, capdActivityLabel, type Workspace, SETTINGS_SECTIONS } from "./layoutState";
import { ConversationMessage, MemoizedConversationMessage } from "./ConversationMessage";
import { reuseQuestions, withoutTaskProgressSurfaces } from "./pollState";
import { QueueWorkloadChart } from "./QueueWorkloadChart";
import { appendQueueSample, type QueueSample } from "./queueHistory";
import { PdcaActivityChart } from "./PdcaActivityChart";
import { appendPdcaActivity, type PdcaSample } from "./pdcaHistory";
import type { ActivityEvent, AgentDocument, ChatMessage, CodeModeSettings, InferenceSettings, MediaToolsSettings, MemorySettings, Project, Question, SystemOneSettings, WebToolsSettings } from "./types";

type Modal = "onboarding" | null;
type SettingsTab = (typeof SETTINGS_SECTIONS)[number][0];
type ManagedProject = Project & { owner?: { display_name?: string; user_id?: string } | null };
type PollResponse = {
  mascot?: AppearanceContext;
  daemon_id?: string;
  daemon_changed?: boolean;
  events: ActivityEvent[];
  next_sequence: number;
  ui_events?: Array<{ sequence?: number; event: { type: string; name?: string; value?: unknown } }>;
  next_ui_sequence?: number;
  deliveries: Array<{ outbox_message_id: string; message: string; task_id?: string; project_id: string; created_at: string; conversation_sequence: number }>;
  questions: Question[];
  conversation_has_unseen_messages?: boolean;
  status: { active_work: Record<string, string>; activity: { state?: string }; queue?: { ready?: number; processing?: number } };
};
type HistoryResponse = { messages: ChatMessage[] };
type Provider = {
  provider_key: string;
  display_name: string;
  models: Array<{ model_id: string; display_name: string }>;
  catalog_cli_version?: string;
  cli_version?: string;
  cli_path?: string;
  catalog_cli_matches_runtime?: boolean | null;
};
const SETTINGS_ICONS = { general: Settings, appearance: Palette, tools: Wrench, integrations: Plug, automations: CalendarClock, model: Brain, "system-one": Gauge, "agent-config": FileText, skills: Sparkles, members: UsersRound };
const localDay = (timezone: string) => {
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone: timezone || "UTC", year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(new Date());
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
};
const messageLocalDay = (value: string | undefined, timezone: string) => {
  if (!value) return "";
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone: timezone || "UTC", year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(new Date(value));
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
};
const shiftDay = (value: string, amount: number) => { const date = new Date(`${value}T12:00:00Z`); date.setUTCDate(date.getUTCDate() + amount); return date.toISOString().slice(0, 10); };

export default function App({ diagnosticMode = "normal", diagnosticProjectId = "" }: { diagnosticMode?: DesktopDiagnosticMode; diagnosticProjectId?: string }) {
  const diagnosticBehavior = useMemo(() => desktopDiagnosticBehavior(diagnosticMode), [diagnosticMode]);
  const clientId = useRef(crypto.randomUUID()).current;
  const sequence = useRef(0);
  const uiSequence = useRef(0);
  const daemonId = useRef("");
  const delivered = useRef(new Set<string>());
  const progressTaskIdsRef = useRef(new Set<string>());
  const autoScrolledProgressTaskIdsRef = useRef(new Set<string>());
  const progressStartedAtRef = useRef(new Map<string, number>());
  const progressMeaningfulAtRef = useRef(new Map<string, number>());
  const progressCompletionTimersRef = useRef(new Map<string, number>());
  const conversationHistoryRequestRef = useRef(0);
  const conversationHistoryLoadingRef = useRef(0);
  const activeHistoryRefreshRef = useRef(false);
  const messagesDuringConversationReloadRef = useRef(new Map<number, ChatMessage[]>());
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const surfacesRef = useRef<Record<string, A2uiSurface>>({});
  const dismissedScheduledSurfacesRef = useRef(readDismissedScheduledSurfaces(window.localStorage));
  const windowFocusedRef = useRef(document.hasFocus());
  const desktopNotifierRef = useRef(createDesktopNotifier({
    isFocused: () => windowFocusedRef.current,
    isNativeRuntime: () => "__TAURI_INTERNALS__" in window,
    playSound: playAlertSound,
  }));
  const [messages, setMessages] = useState<ChatMessage[]>([{ id: "welcome", role: "assistant", content: "Alphonse Desktop is connected locally." }]);
  const [prompt, setPrompt] = useState("");
  const [connected, setConnected] = useState(false);
  const [activity, setActivity] = useState("idle");
  const [mascotContext, setMascotContext] = useState(BASE_CONTEXT);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [contextProjects, setContextProjects] = useState<Project[]>([]);
  const [conversationDay, setConversationDay] = useState(() => localDay("UTC"));
  const conversationDayRef = useRef(conversationDay);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("general");
  const [workspace, setWorkspace] = useState<Workspace>("conversation");
  // Mount once per visit history so navigation preserves local forms and scroll.
  const [visited, setVisited] = useState<Set<Workspace>>(() => new Set(["conversation"]));
  const [returnWorkspace, setReturnWorkspace] = useState<Workspace>("conversation");
  const [atHandOpen, setAtHandOpen] = useState(false);
  const [projectForView, setProjectForView] = useState({ id: "", request: 0 });
  const [memberName, setMemberName] = useState("");
  const [isAdmin, setIsAdmin] = useState(false);
  const workspaceRef = useRef(workspace);
  workspaceRef.current = workspace;
  const workspaceHostRef = useRef<HTMLDivElement>(null);
  const drawerToggleRef = useRef<HTMLButtonElement>(null);
  const drawerCloseRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (atHandOpen) drawerCloseRef.current?.focus();
  }, [atHandOpen]);
  const navigate = (next: Workspace) => {
    if (workspaceRequired && next !== "settings") return;
    if (next === "settings" && workspace !== "settings") setReturnWorkspace(workspace);
    setVisited((current) => new Set([...current, next]));
    setWorkspace(next);
  };
  useEffect(() => {
    workspaceHostRef.current?.querySelector<HTMLElement>(".workspace-panel:not([hidden])")?.focus({ preventScroll: true });
  }, [workspace, settingsTab]);
  const [scheduledTaskForView, setScheduledTaskForView] = useState("");
  const [modal, setModal] = useState<Modal>(null);
  const [user, setUser] = useState("");
  const [workspaceRequired, setWorkspaceRequired] = useState(false);
  const [error, setError] = useState("");
  const [surfaces, setSurfaces] = useState<Record<string, import("./a2ui").A2uiSurface>>({});
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [queueHistory, setQueueHistory] = useState<QueueSample[]>([]);
  const [pdcaHistory, setPdcaHistory] = useState<PdcaSample[]>([]);
  const [conversationHasUnseenMessages, setConversationHasUnseenMessages] = useState(false);
  const [timezone, setTimezone] = useState("UTC");
  const timezoneRef = useRef(timezone);
  const archivedProgressRef = useRef<Record<string, A2uiSurface>>({});
  const [archivedProgress, setArchivedProgress] = useState<Record<string, A2uiSurface>>({});
  const [progressTaskIds, setProgressTaskIds] = useState<string[]>([]);
  const [pendingProgressMessages, setPendingProgressMessages] = useState<Record<string, ChatMessage>>({});
  const [morphedMessageTaskIds, setMorphedMessageTaskIds] = useState<Record<string, string>>({});
  const [heldProgressSurfaces, setHeldProgressSurfaces] = useState<Record<string, A2uiSurface>>({});
  const [enterToSend, setEnterToSend] = useState(() => window.localStorage.getItem("alphonse.desktop.enterToSend") !== "false");
  const [desktopStyle, setDesktopStyle] = useState<DesktopStyle>("system");
  const [desktopNotifications, setDesktopNotifications] = useState<DesktopNotificationPreferences>(() => readDesktopNotificationPreferences(window.localStorage));
  const [diagnosticRenderTick, forceDiagnosticRender] = useState(0);
  const currentAvatarState = avatarState(connected, error, activity);
  const currentAvatarStateLabel = avatarStateLabel(currentAvatarState);
  conversationDayRef.current = conversationDay;
  timezoneRef.current = timezone;

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
    const messageDay = messageLocalDay(message.created_at, timezoneRef.current);
    if (messageDay && messageDay !== conversationDayRef.current) return;
    const loadingRequest = conversationHistoryLoadingRef.current;
    if (loadingRequest) {
      const pending = messagesDuringConversationReloadRef.current.get(loadingRequest) || [];
      messagesDuringConversationReloadRef.current.set(loadingRequest, [...pending, message]);
    }
    setMessages((current) => {
      // Polls repeat pending questions. Preserve identity (and reading position)
      // when that message is already present instead of triggering auto-scroll.
      if (current.some((existing) => existing.id === message.id)) return current;
      const next = mergeFreshConversationHistory(current, [message]);
      return next;
    });
  }, []);

  const pollTransport = useCallback(async () => {
    const response = await daemonRequest<PollResponse>("desktop_poll", {
        client_id: clientId,
        user,
        project_id: "",
        local_date: conversationDay,
        after_sequence: sequence.current,
        after_ui_sequence: uiSequence.current,
        daemon_id: daemonId.current,
        client_capabilities: { supportedCatalogIds: [DESKTOP_CATALOG_ID] },
        limit: 50,
    });
    const previousDaemonId = daemonId.current;
    const daemonChanged = Boolean(response.daemon_changed || (previousDaemonId && response.daemon_id && previousDaemonId !== response.daemon_id));
    if (response.daemon_id) daemonId.current = response.daemon_id;
    sequence.current = response.next_sequence;
    uiSequence.current = response.next_ui_sequence ?? uiSequence.current;
    return { response, daemonChanged };
  }, [clientId, conversationDay, user]);

  const poll = useCallback(async () => {
    try {
      const { response, daemonChanged } = await pollTransport();
      if (!diagnosticBehavior.commitsPollResponses) return;
      if (daemonChanged) {
        progressCompletionTimersRef.current.forEach((timer) => window.clearTimeout(timer));
        progressCompletionTimersRef.current.clear();
        progressTaskIdsRef.current.clear();
        autoScrolledProgressTaskIdsRef.current.clear();
        progressStartedAtRef.current.clear();
        progressMeaningfulAtRef.current.clear();
        archivedProgressRef.current = {};
        setArchivedProgress({});
        setProgressTaskIds([]);
        setPendingProgressMessages({});
        setHeldProgressSurfaces({});
        const retainedSurfaces = withoutTaskProgressSurfaces(surfacesRef.current);
        surfacesRef.current = retainedSurfaces;
        setSurfaces(retainedSurfaces);
      }
      setConnected(true);
      setError("");
      setMascotContext(response.mascot || BASE_CONTEXT);
      setQuestions((current) => reuseQuestions(current, response.questions));
      response.questions.forEach((question) => appendMessage({
        id: `question:${question.question_id}`,
        role: "assistant",
        content: question.message,
        source: "desktop",
        created_at: question.created_at,
        project_id: question.project_id,
        sequence: question.conversation_sequence,
      }));
      response.questions.forEach((question) => {
        const projectName = contextProjects.find((item) => item.project_id === question.project_id)?.name || "";
        void desktopNotifierRef.current.notify(notificationForQuestion(question, projectName), desktopNotifications);
      });
      setConversationHasUnseenMessages(Boolean(response.conversation_has_unseen_messages));
      const polledAt = Date.now();
      setQueueHistory((current) => appendQueueSample(current, {
        ready: response.status.queue?.ready || 0,
        processing: response.status.queue?.processing || 0,
      }, polledAt));
      setPdcaHistory((current) => appendPdcaActivity(
        daemonChanged ? [] : current,
        response.events,
        response.status.activity.state || "idle",
        polledAt,
      ));
      const newProgressTaskIds = taskProgressIds(response.ui_events || []);
      if (newProgressTaskIds.length) {
        newProgressTaskIds.forEach((taskId) => {
          progressTaskIdsRef.current.add(taskId);
          if (!progressStartedAtRef.current.has(taskId)) progressStartedAtRef.current.set(taskId, Date.now());
        });
        setProgressTaskIds((current) => [...current, ...newProgressTaskIds.filter((taskId) => !current.includes(taskId))]);
      }
      const progressProjection = projectTaskProgress(
        surfacesRef.current,
        archivedProgressRef.current,
        response.ui_events || [],
        ["idle", "error"].includes(response.status.activity.state || "idle")
          && !Object.keys(response.status.active_work).length
          && !response.status.queue?.processing,
      );
      archivedProgressRef.current = progressProjection.archived;
      setArchivedProgress(progressProjection.archived);
      Object.keys(progressProjection.archived).forEach((taskId) => {
        progressTaskIdsRef.current.delete(taskId);
        progressStartedAtRef.current.delete(taskId);
        progressMeaningfulAtRef.current.delete(taskId);
      });
      const projectedSurfaces = withoutDismissedSurfaces(
        progressProjection.surfaces,
        dismissedScheduledSurfacesRef.current,
      );
      surfacesRef.current = projectedSurfaces;
      setSurfaces(projectedSurfaces);
      newProgressTaskIds.forEach((taskId) => {
        if (progressProjection.archived[taskId]) return;
        const components = projectedSurfaces[`task-progress:${taskId}`]?.components || {};
        if (["criteria", "intention", "tool", "result"].some((componentId) => String(components[componentId]?.text || "").trim()) || Object.keys(components).some((componentId) => componentId.startsWith("step_"))) {
          progressMeaningfulAtRef.current.set(taskId, Date.now());
        }
      });
      const latest = response.events.at(-1);
      setActivity(capdActivityLabel(latest, response.status.activity.state || "idle"));
      for (const delivery of response.deliveries) {
        if (delivered.current.has(delivery.outbox_message_id)) continue;
        delivered.current.add(delivery.outbox_message_id);
        void desktopNotifierRef.current.notify(
          notificationForCompletion(delivery, contextProjects.find((item) => item.project_id === delivery.project_id)?.name || ""),
          desktopNotifications,
        );
        const completedMessage: ChatMessage = {
          id: delivery.outbox_message_id,
          role: "assistant",
          content: delivery.message,
          created_at: delivery.created_at,
          project_id: delivery.project_id,
          sequence: delivery.conversation_sequence,
        };
        if (delivery.task_id) {
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
            continue;
          }
        }
        appendMessage(completedMessage);
        await daemonRequest("desktop_ack_delivery", { client_id: clientId, outbox_message_id: delivery.outbox_message_id });
      }
    } catch (cause) {
      if (!diagnosticBehavior.commitsPollResponses) return;
      setConnected(false);
      setMascotContext(BASE_CONTEXT);
      setError(cause instanceof Error ? cause.message : "Daemon connection lost");
      setActivity("idle");
    }
  }, [appendMessage, clientId, contextProjects, desktopNotifications, diagnosticBehavior.commitsPollResponses, pollTransport, user]);

  useEffect(() => {
    const runCycle = diagnosticBehavior.cycle === "ping"
      ? () => daemonRequest("ping")
      : diagnosticBehavior.cycle === "poll" ? poll : null;
    if (diagnosticBehavior.startsDaemon) void ensureDaemon().then(async () => {
      if (diagnosticBehavior.loadsHistory) {
        const current = await daemonRequest<{ onboarded: boolean; user: { user_id: string } | null }>("current_user");
        if (!current.user) throw new Error("Diagnostic history requires an onboarded user");
        const history = await daemonRequest<HistoryResponse>("desktop_conversation_history", {
          user: current.user.user_id,
          project_id: diagnosticProjectId,
          limit: 100,
        });
        setUser(current.user.user_id);
        setMessages(history.messages);
      } else if (diagnosticMode === "normal") {
        const current = await daemonRequest<{ onboarded: boolean; workspace_configured?: boolean; user: { user_id: string; display_name?: string; role?: string; is_active?: boolean } | null }>("current_user");
        setMemberName(current.user?.display_name || current.user?.user_id || "");
        setIsAdmin(current.user?.role === "admin" && current.user?.is_active !== false);
        if (!current.onboarded || !current.user) setModal("onboarding"); else {
          setUser(current.user.user_id);
          if (!current.workspace_configured) { setWorkspaceRequired(true); setSettingsTab("general"); setVisited((value) => new Set([...value, "settings"])); setWorkspace("settings"); }
        }
      }
      if (runCycle) await runCycle();
    }).catch((cause: unknown) => {
      if (diagnosticBehavior.commitsPollResponses || diagnosticBehavior.loadsHistory) {
        setError(cause instanceof Error ? cause.message : "Unable to start daemon");
      }
    });
    if (diagnosticBehavior.cycle === "render") {
      const timer = window.setInterval(() => forceDiagnosticRender((current) => current + 1), 800);
      return () => window.clearInterval(timer);
    }
    if (runCycle) {
      const timer = window.setInterval(() => void runCycle(), 800);
      return () => window.clearInterval(timer);
    }
  }, [diagnosticBehavior, diagnosticMode, diagnosticProjectId, poll]);

  useEffect(() => {
    window.localStorage.setItem("alphonse.desktop.enterToSend", String(enterToSend));
  }, [enterToSend]);

  useEffect(() => {
    setDesktopStyle(readDesktopStyle(window.localStorage, user));
  }, [user]);

  useEffect(() => followDesktopStyle(
    document.documentElement, window.matchMedia("(prefers-color-scheme: dark)"), desktopStyle,
  ), [desktopStyle]);

  const changeDesktopStyle = (next: DesktopStyle) => {
    saveDesktopStyle(window.localStorage, user, next);
    setDesktopStyle(next);
  };

  useEffect(() => {
    window.localStorage.setItem(DESKTOP_NOTIFICATION_PREFERENCES_KEY, JSON.stringify(desktopNotifications));
  }, [desktopNotifications]);

  useEffect(() => {
    const setFocused = (focused: boolean) => { windowFocusedRef.current = focused; };
    const focus = () => setFocused(true);
    const blur = () => setFocused(false);
    window.addEventListener("focus", focus);
    window.addEventListener("blur", blur);
    if (!("__TAURI_INTERNALS__" in window)) return () => {
      window.removeEventListener("focus", focus);
      window.removeEventListener("blur", blur);
    };
    let unlistenFocus: (() => void) | undefined;
    let notificationActionListener: { unregister: () => Promise<void> } | undefined;
    void getCurrentWindow().onFocusChanged(({ payload }) => setFocused(payload)).then((stop) => { unlistenFocus = stop; }).catch(() => undefined);
    void onAction(() => { void getCurrentWindow().show().then(() => getCurrentWindow().setFocus()); }).then((listener) => { notificationActionListener = listener; }).catch(() => undefined);
    return () => {
      window.removeEventListener("focus", focus);
      window.removeEventListener("blur", blur);
      unlistenFocus?.();
      void notificationActionListener?.unregister();
    };
  }, []);

  useEffect(() => {
    if (!user) return;
    void daemonRequest<{ timezone: string }>("timezone_settings", { actor_user_id: user })
      .then((result) => setTimezone(result.timezone || "UTC"))
      .catch(() => setTimezone("UTC"));
  }, [user]);

  useEffect(() => {
    if (!user) return;
    setConversationDay(localDay(timezone));
    void daemonRequest<{ projects: Project[] }>("manageable_projects", { user }).then((result) => setContextProjects(result.projects)).catch(() => setContextProjects([]));
  }, [timezone, user]);

  useEffect(() => {
    if (!user) return;
    const request = ++conversationHistoryRequestRef.current;
    conversationHistoryLoadingRef.current = request;
    messagesDuringConversationReloadRef.current.set(request, []);
    setMessages([]);
    void daemonRequest<HistoryResponse>("desktop_conversation_history", { user, local_date: conversationDay, limit: 500 })
      .then(async (history) => {
        if (conversationHistoryRequestRef.current !== request) return;
        const pending = messagesDuringConversationReloadRef.current.get(request) || [];
        const refreshed = mergeFreshConversationHistory(history.messages, pending);
        setMessages(refreshed);
        conversationHistoryLoadingRef.current = 0;
        messagesDuringConversationReloadRef.current.delete(request);
        await daemonRequest("desktop_mark_conversation_seen", { user, local_date: conversationDay });
        setConversationHasUnseenMessages(false);
      })
      .catch((cause: unknown) => { if (conversationHistoryRequestRef.current === request) setError(cause instanceof Error ? cause.message : "Conversation history could not be loaded"); })
      .finally(() => { if (conversationHistoryRequestRef.current === request) { conversationHistoryLoadingRef.current = 0; messagesDuringConversationReloadRef.current.delete(request); } });
  }, [conversationDay, user, timezone]);

  useEffect(() => {
    if (!user || !conversationHasUnseenMessages || activeHistoryRefreshRef.current || conversationHistoryLoadingRef.current) return;
    activeHistoryRefreshRef.current = true;
    void daemonRequest<HistoryResponse>("desktop_conversation_history", { user, local_date: conversationDay, limit: 500 })
      .then(async (history) => {
        setMessages((current) => {
          const refreshed = mergeFreshConversationHistory(history.messages, current);
          return refreshed;
        });
        await daemonRequest("desktop_mark_conversation_seen", { user, local_date: conversationDay });
        setConversationHasUnseenMessages(false);
      })
      .catch((cause: unknown) => setError(cause instanceof Error ? cause.message : "Conversation history could not be refreshed"))
      .finally(() => { activeHistoryRefreshRef.current = false; });
  }, [conversationDay, conversationHasUnseenMessages, user]);

  useEffect(() => {
    const element = timelineRef.current;
    if (!element || workspaceRef.current !== "conversation") return;
    element.scrollTop = element.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const unseen = progressTaskIds.filter((taskId) => !autoScrolledProgressTaskIdsRef.current.has(taskId));
    if (!unseen.length || workspaceRef.current !== "conversation") return;
    unseen.forEach((taskId) => autoScrolledProgressTaskIdsRef.current.add(taskId));
    const frame = window.requestAnimationFrame(() => {
      const element = timelineRef.current;
      element?.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [progressTaskIds]);

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 4 * 22 + 24)}px`;
  }, [prompt, workspace]);

  const submitPrompt = async () => {
    const value = prompt.trim();
    if (!value) return;
    if (value.startsWith("/")) {
      const command = value.split(/\s+/, 1)[0].toLowerCase();
      if (command === "/killswitch") {
        setPrompt("");
        try {
          const result = await daemonRequest<{ status: string }>("trigger_killswitch", { actor_user_id: user });
          appendMessage({
            id: `killswitch:${crypto.randomUUID()}`,
            role: "assistant",
            content: result.status === "cancel_requested" ? "Kill switch engaged. The active task is being cancelled." : "Kill switch checked: no active task is running.",
            created_at: new Date().toISOString(),
            project_id: "",
          });
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : "Kill switch failed");
        }
        return;
      }
      if (matchingCommands(command).includes(command)) {
        setPrompt("");
        await runCommand(command);
        return;
      }
    }
    try {
      const messagePrompt = value;
      const queued = await daemonRequest<{ message_id: string; project_id: string; created_at: string; conversation_sequence: number }>("queue_message", {
        prompt: messagePrompt,
        user,
        project_id: "",
        integration_id: "desktop",
        provider_key: "tui",
        channel_target: user,
        metadata: {},
      });
      setPrompt("");
      const today = localDay(timezone);
      conversationDayRef.current = today;
      setConversationDay(today);
      appendMessage({
        id: queued.message_id,
        role: "user",
        content: messagePrompt,
        created_at: queued.created_at,
        project_id: queued.project_id,
        sequence: queued.conversation_sequence,
      });
      await daemonRequest("desktop_mark_conversation_seen", { user, local_date: today });
      setActivity("doing");
      await poll();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Message could not be queued");
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
    if (command === "/project") return navigate("projects");
    if (command === "/integrations") { setSettingsTab("integrations"); return navigate("settings"); }
    if (command === "/model" || command === "/model-provider") { setSettingsTab("model"); return navigate("settings"); }
    if (command === "/agent-config") { setSettingsTab("agent-config"); return navigate("settings"); }
    if (command === "/scheduled-tasks") return navigate("agenda");
    if (command === "/settings") { setSettingsTab("general"); return navigate("settings"); }
    if (command === "/users") return navigate("household");
    if (command === "/stop") {
      if (window.confirm("Stop the Alphonse daemon?")) await stopDaemon();
      return;
    }
    if (command === "/exit" || command === "/quit") window.close();
  };

  const suggestions = useMemo(() => matchingCommands(prompt), [prompt]);
  const activeSurface = Object.values(surfaces).find((surface) => !surface.surfaceId.startsWith("task-progress:") && !progressTaskIds.includes(questionTaskId(surface)));
  const fallbackQuestion = questions.find((question) => !surfaces[`question:${question.question_id}`]);
  const timelineRenderVersion = diagnosticBehavior.memoizesTimeline ? 0 : diagnosticRenderTick;
  const questionTaskIds = new Set(questions.filter((question) => surfaces[`question:${question.question_id}`]).map((question) => question.task_id));
  const timelineNodes: ReactNode[] = useMemo(() => buildConversationTimeline(
      messages,
      progressTaskIds.filter((taskId) => !questionTaskIds.has(taskId)),
      pendingProgressMessages,
      morphedMessageTaskIds,
      Object.keys(archivedProgress),
    ).map((item) => {
      if (item.kind === "message") {
        const question = questions.find((candidate) => item.message.id === `question:${candidate.question_id}`);
        const questionSurface = question ? surfaces[`question:${question.question_id}`] : undefined;
        if (questionSurface) return <article data-question-id={question?.question_id} className="message assistant task-question-bubble" aria-label="Question from Alphonse" key={item.key}><A2uiSurfaceView surface={questionSurface} clientId={clientId} user={user} onDone={poll} /></article>;
        const Bubble = diagnosticBehavior.memoizesMessages ? MemoizedConversationMessage : ConversationMessage;
        const bubble = <Bubble key={item.key} message={item.message} timezone={timezone} projectName={contextProjects.find((projectItem) => projectItem.project_id === item.message.project_id)?.name || ""} renderMarkdown={!diagnosticBehavior.plainTextMessages} />;
        return question ? <div className="timeline-message" data-question-id={question.question_id} key={item.key}>{bubble}</div> : bubble;
      }
      const archivedSurface = archivedProgress[item.taskId];
      const surface = archivedSurface || surfaces[`task-progress:${item.taskId}`] || heldProgressSurfaces[item.taskId];
      const questionSurface = Object.values(surfaces).find((candidate) => questionTaskId(candidate) === item.taskId);
      if (questionSurface) {
        return <article data-question-id={questions.find((question) => question.task_id === item.taskId)?.question_id} className="message assistant task-question-bubble" aria-label="Question from Alphonse" key={item.key}><A2uiSurfaceView surface={questionSurface} clientId={clientId} user={user} onDone={poll} /></article>;
      }
      return surface ? <TaskProgressBubble key={item.key} surface={surface} archived={Boolean(archivedSurface)} /> : null;
    }), [archivedProgress, clientId, contextProjects, diagnosticBehavior.memoizesMessages, diagnosticBehavior.plainTextMessages, heldProgressSurfaces, messages, morphedMessageTaskIds, pendingProgressMessages, poll, progressTaskIds, questionTaskIds, questions, surfaces, timelineRenderVersion, timezone, user]);

  return (
    <main className={`app-shell desktop-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""} ${atHandOpen ? "at-hand-open" : ""}`}>
      <aside className="sidebar" aria-label="Alphonse navigation">
        <div className="brand">
          <div className="brand-avatar" title={`Alphonse is ${currentAvatarStateLabel}`}>
            <SkillMascot user={user} state={currentAvatarState} context={mascotContext} />
            {currentAvatarState === "disconnected" && <span className="brand-avatar-status">Disconnected</span>}
          </div>
          <span className="brand-name">Alphonse</span>
          <button className="collapse-toggle" type="button" onClick={() => setSidebarCollapsed((value) => !value)} aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}>{sidebarCollapsed ? <ChevronRight aria-hidden="true" /> : <ChevronLeft aria-hidden="true" />}</button>
        </div>
        <div className="member-context"><strong>{memberName || user || "Local household"}</strong><small>{connected ? "Connected locally" : "Disconnected"} · {currentAvatarStateLabel}</small></div>
        <nav className="shell-navigation" aria-label={workspace === "settings" ? "Settings navigation" : "Workspaces"}>
          {workspace === "settings" ? <>
            <button disabled={workspaceRequired} title="Back to workspace" onClick={() => navigate(returnWorkspace)}><ChevronLeft /><span>Back to workspace</span></button>
            {SETTINGS_SECTIONS.filter(([id]) => (!workspaceRequired || id === "general") && (isAdmin || ["general", "appearance"].includes(id))).map(([id, label]) => { const Icon = SETTINGS_ICONS[id]; return <button key={id} title={label} aria-current={workspace === "settings" && settingsTab === id ? "page" : undefined} onClick={() => { setSettingsTab(id); navigate("settings"); }}><Icon /><span>{label}</span></button>; })}
          </> : <>
            {([["conversation", "Conversation", MessageCircle], ["household", "Household", UsersRound], ["projects", "Projects & Tasks", FolderKanban], ["agenda", "Agenda", CalendarClock], ["attention", "Needs your attention", CircleHelp]] as const).map(([id, label, Icon]) => <button key={id} title={label} aria-current={workspace === id ? "page" : undefined} onClick={() => navigate(id)}><Icon /><span>{label}</span>{id === "attention" && questions.length > 0 && <small>{questions.length}</small>}</button>)}
            <div className="in-progress-projects"><small>In progress</small>{contextProjects.filter((item) => item.status === "active" && !item.is_system_home).map((item) => <button key={item.project_id} title={item.name} onClick={() => { setProjectForView((current) => ({ id: item.project_id, request: current.request + 1 })); navigate("projects"); }}><FolderKanban /><span>{item.name}</span></button>)}</div>
            <button title="Settings" onClick={() => navigate("settings")}><Settings /><span>Settings</span></button>
          </>}
        </nav>
        <QueueWorkloadChart samples={queueHistory} />
        <PdcaActivityChart samples={pdcaHistory} now={queueHistory.at(-1)?.at || Date.now()} />
      </aside>
      <div className="workspace-host" ref={workspaceHostRef}>
        <div className="shell-toolbar"><span>{memberName || user || "Local household"} · {connected ? currentAvatarStateLabel : "Disconnected"}</span><button ref={drawerToggleRef} aria-label="Toggle At hand" aria-controls="at-hand" aria-expanded={atHandOpen} onClick={() => setAtHandOpen((value) => !value)}>{atHandOpen ? <PanelRightClose /> : <PanelRightOpen />} At hand</button></div>
        <section className="workspace-panel" tabIndex={-1} aria-label="Conversation workspace" hidden={workspace !== "conversation"}>
        <section className="conversation">
        <header className="topbar">
          <div className="topbar-project"><p className="eyebrow">Main conversation</p><h1>Conversation</h1></div>
          <div className="conversation-day-controls" aria-label="Conversation day history">
            <button type="button" aria-label="Previous day" onClick={() => setConversationDay((day) => shiftDay(day, -1))}>‹</button>
            <input aria-label="Conversation date" type="date" value={conversationDay} max={localDay(timezone)} onChange={(event) => event.target.value && setConversationDay(event.target.value)} />
            <button type="button" aria-label="Next day" disabled={conversationDay >= localDay(timezone)} onClick={() => setConversationDay((day) => shiftDay(day, 1))}>›</button>
            {conversationDay !== localDay(timezone) && <button type="button" onClick={() => setConversationDay(localDay(timezone))}>Today</button>}
          </div>
        </header>
        {error && <div className="error" role="alert">{error}</div>}
        <div className="timeline" ref={timelineRef} aria-live="polite">
          {timelineNodes}
        </div>
        <section className="input-dock">
          {activeSurface ? (
            <A2uiSurfaceView surface={activeSurface} clientId={clientId} user={user} onDone={poll} onDismiss={dismissScheduledSurface} onAction={(result) => { if (result.action === "view_scheduled_task" && typeof result.scheduled_task_id === "string") { dismissScheduledSurface(activeSurface.surfaceId); setScheduledTaskForView(result.scheduled_task_id); navigate("agenda"); } }} />
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
        </section></section>
        {visited.has("projects") && <section className="workspace-panel" tabIndex={-1} aria-label="Projects and tasks workspace" hidden={workspace !== "projects"}><ProjectsWorkspace user={user} initialProjectId={projectForView.id} selectionRequest={projectForView.request} onClose={() => navigate("conversation")} /></section>}
        {visited.has("household") && <section className="workspace-panel" tabIndex={-1} aria-label="Household workspace" hidden={workspace !== "household"}>{isAdmin ? <UsersModal embedded onClose={() => navigate("conversation")} /> : <p>Member management requires an administrator.</p>}</section>}
        {visited.has("agenda") && <section className="workspace-panel" tabIndex={-1} aria-label="Agenda workspace" hidden={workspace !== "agenda"}><ScheduledTasksModal embedded actorUserId={user} initialTaskId={scheduledTaskForView} onClose={() => navigate("conversation")} /></section>}
        {visited.has("attention") && <section className="workspace-panel attention-workspace" tabIndex={-1} aria-label="Needs your attention workspace" hidden={workspace !== "attention"}><h1>Needs your attention</h1>{questions.length ? questions.map((question) => <article className="card" key={question.question_id}><p>{question.message}</p><button onClick={() => { navigate("conversation"); window.requestAnimationFrame(() => {
              const target = Array.from(document.querySelectorAll<HTMLElement>("[data-question-id]")).find((element) => element.dataset.questionId === question.question_id);
              (target?.querySelector<HTMLElement>(".message") || target)?.scrollIntoView({ block: "center" });
              target?.querySelector<HTMLElement>("input, textarea, button")?.focus({ preventScroll: true });
            }); }}>View in conversation</button></article>) : <p>No pending questions.</p>}</section>}
      {visited.has("settings") && <section className="workspace-panel" tabIndex={-1} aria-label="Settings workspace" hidden={workspace !== "settings"}><SettingsWorkspace isAdmin={isAdmin} user={user} initialTab={settingsTab} workspaceRequired={workspaceRequired} enterToSend={enterToSend} desktopStyle={desktopStyle} desktopNotifications={desktopNotifications} onDesktopStyleChange={changeDesktopStyle} onDesktopNotificationsChange={setDesktopNotifications} onEnterToSendChange={setEnterToSend} onTimezoneChange={setTimezone} onWorkspaceConfigured={() => setWorkspaceRequired(false)} onTestDesktopNotification={() => void desktopNotifierRef.current.notify({ id: `test:${Date.now()}`, title: "Alphonse test alert", body: "Desktop notifications are ready." }, { ...desktopNotifications, onlyWhenUnfocused: false })} onClose={() => { navigate(returnWorkspace); }} /></section>}
      </div>
      <aside id="at-hand" className="at-hand-drawer" aria-label="At hand" hidden={!atHandOpen} onKeyDown={(event) => { if (event.key === "Escape") { setAtHandOpen(false); drawerToggleRef.current?.focus(); } }}>
        <header><h2>At hand</h2><button ref={drawerCloseRef} aria-label="Close At hand" onClick={() => { setAtHandOpen(false); drawerToggleRef.current?.focus(); }}><PanelRightClose /></button></header>
        <div className="card"><h3>Pending questions</h3><p>{questions.length ? `${questions.length} waiting for your input` : "No pending questions"}</p><button onClick={() => navigate("attention")}>Needs your attention</button></div>
        <div className="card"><h3>Household agenda</h3><p>Review scheduled work and its history.</p><button onClick={() => navigate("agenda")}>Open Agenda</button></div>
      </aside>
      {modal === "onboarding" && <OnboardingModal onComplete={(next) => { setUser(next); setIsAdmin(true); setWorkspaceRequired(false); setModal(null); void poll(); }} />}
    </main>
  );
}

function QuestionCard({ question, onDone }: { question: Question; onDone: () => Promise<void> }) {
  const [value, setValue] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
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
    {question.kind === "multi_choice" && <form onSubmit={(event) => { event.preventDefault(); void answer({ choice_ids: selected }, selected.join(", ")); }}><fieldset className="a2ui-choice-picker"><legend>Choose one or more options</legend>{question.choices.map((choice) => <label key={choice.id}><input type="checkbox" checked={selected.includes(choice.id)} onChange={(event) => setSelected((current) => event.target.checked ? [...current, choice.id] : current.filter((id) => id !== choice.id))} />{choice.label}</label>)}</fieldset><span className="question-actions"><button disabled={!selected.length}>Answer</button><button type="button" className="question-cancel" onClick={() => void cancel()}>Cancel</button></span></form>}
    {question.kind === "datetime" && <form onSubmit={(event) => { event.preventDefault(); void answer({ datetime: value }, value); }}><input type="datetime-local" value={value} onChange={(event) => setValue(event.target.value)} aria-label="Date and time" /><span className="question-actions"><button disabled={!value}>Answer</button><button type="button" className="question-cancel" onClick={() => void cancel()}>Cancel</button></span></form>}
    {question.kind === "open_text" && <form onSubmit={(event) => { event.preventDefault(); void answer({ text: value }, value); }}><input value={value} onChange={(event) => setValue(event.target.value)} placeholder="Your answer" /><span className="question-actions"><button>Answer</button><button type="button" className="question-cancel" onClick={() => void cancel()}>Cancel</button></span></form>}
  </section>;
}

function ModalFrame({ title, children, tabs, className = "", embedded = false, onClose }: { title: string; children: ReactNode; tabs?: ReactNode; className?: string; embedded?: boolean; onClose: () => void }) {
  if (embedded) return <section className={`modal embedded-workspace ${className}`}><header><h2>{title}</h2><button onClick={onClose}>← Back</button></header>{tabs && <div className="modal-tabs">{tabs}</div>}<div className="modal-content">{children}</div></section>;
  return <div className="modal-backdrop" role="presentation"><section className={`modal ${className}`} role="dialog" aria-modal="true" aria-label={title}><header><h2>{title}</h2><button onClick={onClose}>Close</button></header>{tabs && <div className="modal-tabs">{tabs}</div>}<div className="modal-content">{children}</div></section></div>;
}

function OnboardingModal({ onComplete }: { onComplete: (userId: string) => void }) {
  const [name, setName] = useState(""); const [root, setRoot] = useState(""); const [error, setError] = useState("");
  const save = async () => { try { const result = await daemonRequest<{ admin_user: { user_id: string } }>("onboard", { display_name: name, workspace_root: root }); onComplete(result.admin_user.user_id); } catch (cause) { setError(cause instanceof Error ? cause.message : "Onboarding failed"); } };
  const chooseRoot = async () => { try { const selected = await openFileDialog({ title: "Choose Alphonse Workspace", multiple: false, directory: true }); if (typeof selected === "string") setRoot(selected); } catch (cause) { setError(cause instanceof Error ? cause.message : "Folder picker could not be opened."); } };
  return <ModalFrame title="Set up Alphonse" onClose={() => undefined}><p>Create the local administrator and choose Alphonse’s Workspace directory. This is the root for users, projects, skills, and agent configuration.</p><label>Workspace directory</label><div className="workspace-picker"><input aria-label="Selected Workspace directory" readOnly value={root} placeholder="Choose a folder in Finder" /><button type="button" onClick={() => void chooseRoot()}>Choose folder…</button></div><label>Your name<input value={name} onChange={(event) => setName(event.target.value)} placeholder="Your name" /></label><button disabled={!name.trim() || !root.trim()} onClick={() => void save()}>Create administrator</button><p>{error}</p></ModalFrame>;
}

function SettingsWorkspace({ isAdmin, user, initialTab, workspaceRequired, enterToSend, desktopStyle, desktopNotifications, onDesktopStyleChange, onDesktopNotificationsChange, onEnterToSendChange, onTimezoneChange, onWorkspaceConfigured, onTestDesktopNotification, onClose }: { isAdmin: boolean; user: string; initialTab: SettingsTab; workspaceRequired: boolean; enterToSend: boolean; desktopStyle: DesktopStyle; desktopNotifications: DesktopNotificationPreferences; onDesktopStyleChange: (value: DesktopStyle) => void; onDesktopNotificationsChange: (value: DesktopNotificationPreferences) => void; onEnterToSendChange: (value: boolean) => void; onTimezoneChange: (value: string) => void; onWorkspaceConfigured: () => void; onTestDesktopNotification: () => void; onClose: () => void }) {
  const [root, setRoot] = useState(""); const [timezone, setTimezone] = useState("UTC"); const [mirrorAutomationMessages, setMirrorAutomationMessages] = useState(false); const [notice, setNotice] = useState(""); const [workspaceNotice, setWorkspaceNotice] = useState(""); const [savingWorkspace, setSavingWorkspace] = useState(false); const tab = workspaceRequired ? "general" : initialTab;
  const [visitedSections, setVisitedSections] = useState<Set<SettingsTab>>(() => new Set([tab]));
  const headingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    setVisitedSections((value) => new Set([...value, tab]));
    headingRef.current?.focus({ preventScroll: true });
  }, [tab]);
  const [web, setWeb] = useState<WebToolsSettings | null>(null); const [webNotice, setWebNotice] = useState("");
  const [memory, setMemory] = useState<MemorySettings | null>(null); const [memoryNotice, setMemoryNotice] = useState("");
  useEffect(() => { if (!isAdmin) return; void daemonRequest<{ workspace_root: string; workspace_configured: boolean; legacy_users_root: string; mirror_automation_messages_to_preferred_channel: boolean }>("settings").then((result) => { setRoot(result.workspace_root || ""); setMirrorAutomationMessages(Boolean(result.mirror_automation_messages_to_preferred_channel)); if (!result.workspace_configured) setNotice("Workspace directory is not configured. Set it here to continue."); }).catch((cause: unknown) => setWorkspaceNotice(timezoneSettingsError(cause))); void daemonRequest<{ timezone: string }>("timezone_settings", { actor_user_id: user }).then((result) => setTimezone(result.timezone)).catch((cause: unknown) => setNotice(timezoneSettingsError(cause))); }, [user, isAdmin]);
  useEffect(() => { if (isAdmin && tab === "tools" && !web) void daemonRequest<{ user: { user_id: string } | null }>("current_user").then(async (current) => { if (!current.user) return; const result = await daemonRequest<{ settings: WebToolsSettings }>("web_tools_settings", { actor_user_id: current.user.user_id }); setWeb(result.settings); }).catch((cause: unknown) => setWebNotice(cause instanceof Error ? cause.message : "Web Tools unavailable")); }, [tab, isAdmin]);
  useEffect(() => { if (!isAdmin) return; void daemonRequest<{ user: { user_id: string } | null }>("current_user").then(async (current) => { if (!current.user) return; const result = await daemonRequest<{ settings: MemorySettings }>("memory_settings", { actor_user_id: current.user.user_id }); setMemory(result.settings); }).catch((cause: unknown) => setMemoryNotice(cause instanceof Error ? cause.message : "Memory settings unavailable")); }, [user, isAdmin]);
  const save = async () => { try { const persisted = await daemonRequest<{ workspace_root: string }>("settings"); const result = await daemonRequest<{ workspace_root: string; warning_repository_path?: boolean; mirror_automation_messages_to_preferred_channel: boolean }>("save_settings", { actor_user_id: user, workspace_root: persisted.workspace_root, mirror_automation_messages_to_preferred_channel: mirrorAutomationMessages }); setMirrorAutomationMessages(Boolean(result.mirror_automation_messages_to_preferred_channel)); onWorkspaceConfigured(); setNotice(result.warning_repository_path ? "Saved. This path is inside the repository; keep it ignored." : "Delivery settings saved."); } catch (cause) { setNotice(timezoneSettingsError(cause)); } };
  const saveWorkspace = async () => { if (!root.trim() || savingWorkspace) return; setSavingWorkspace(true); setWorkspaceNotice("Checking and migrating existing files if needed. The first save can take a few minutes while skills and managed project files are copied."); let workspaceSaved = false; try { const result = await daemonRequest<{ workspace_root: string; warning_repository_path?: boolean }>("save_settings", { actor_user_id: user, workspace_root: root }); setRoot(result.workspace_root); onWorkspaceConfigured(); workspaceSaved = true; setWorkspaceNotice("Workspace saved. Saving timezone…"); const timezoneResult = await daemonRequest<{ timezone: string }>("save_timezone_settings", { actor_user_id: user, timezone }); if (timezoneResult.timezone !== timezone.trim()) throw new Error("The daemon did not persist the requested timezone"); setTimezone(timezoneResult.timezone); onTimezoneChange(timezoneResult.timezone); setWorkspaceNotice(result.warning_repository_path ? `Workspace and timezone saved. Workspace: ${result.workspace_root}. This path is inside the repository; keep it ignored.` : `Workspace and timezone saved. Workspace: ${result.workspace_root}.`); } catch (cause) { const message = cause instanceof Error ? cause.message : "Settings could not be saved."; setWorkspaceNotice(workspaceSaved ? `Workspace saved, but timezone could not be saved: ${message}` : timezoneSettingsError(cause)); } finally { setSavingWorkspace(false); } };
  const chooseWorkspace = async () => { try { const selected = await openFileDialog({ title: "Choose Alphonse Workspace", multiple: false, directory: true }); if (typeof selected === "string") { setRoot(selected); setWorkspaceNotice(""); } } catch (cause) { setWorkspaceNotice(cause instanceof Error ? cause.message : "Folder picker could not be opened."); } };
  const saveWeb = async () => { if (!web) return; try { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ settings: WebToolsSettings }>("save_web_tools_settings", { actor_user_id: current.user.user_id, values: web }); setWeb(result.settings); setWebNotice("Saved. Newly started tasks can use enabled Web Tools."); } catch (cause) { setWebNotice(cause instanceof Error ? cause.message : "Save failed"); } };
  const chooseNotificationSound = async () => {
    const selected = await openFileDialog({
      title: "Choose notification sound",
      multiple: false,
      directory: false,
      filters: [{ name: "Audio", extensions: ["aif", "aiff", "caf", "m4a", "mp3", "wav"] }],
    });
    if (typeof selected === "string") onDesktopNotificationsChange({ ...desktopNotifications, soundFile: selected });
  };
  const verify = async (kind: "search" | "fetch") => { try { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ result: { exception?: { message?: string } } }>("verify_web_tools", { actor_user_id: current.user.user_id, kind }); setWebNotice(result.result.exception?.message || `${kind === "search" ? "SearXNG search" : "Public fetch"} verified.`); } catch (cause) { setWebNotice(cause instanceof Error ? cause.message : "Verification failed"); } };
  const saveMemory = async () => { if (!memory) return; try { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ settings: MemorySettings }>("save_memory_settings", { actor_user_id: current.user.user_id, values: memory }); setMemory(result.settings); setMemoryNotice("Saved. New tasks use these limits."); } catch (cause) { setMemoryNotice(cause instanceof Error ? cause.message : "Memory settings could not be saved"); } };
  const allowed = isAdmin || ["general", "appearance"].includes(tab);
  return <section className="settings-workspace">
    <header className="settings-workspace-header"><div><p className="eyebrow">Settings</p><h1 ref={headingRef} tabIndex={-1}>{workspaceRequired ? "Set up Alphonse Workspace" : SETTINGS_SECTIONS.find(([id]) => id === tab)?.[1]}</h1><p>Edits stay here while you navigate, until you close the app. Use each section’s save button to apply changes. Appearance, messaging and alerts save immediately.</p></div><button disabled={workspaceRequired} onClick={onClose}>← Back to workspace</button></header>
    {!allowed && <p role="alert">This section requires an administrator.</p>}
    {workspaceRequired && <p className="settings-setup-notice" role="alert">{isAdmin ? "Choose and save a Workspace directory to continue." : "Ask the administrator to configure the Workspace directory."}</p>}
    {visitedSections.has("general") && <div hidden={tab !== "general"} className="general-settings">
      <section className="setting-group">
        <div className="setting-group-heading"><h3>Messaging</h3><p>Choose how messages are sent from the desktop conversation.</p></div>
        <label className="setting-row"><input type="checkbox" checked={enterToSend} onChange={(event) => onEnterToSendChange(event.target.checked)} /> Enter sends message</label>
      </section>
      <section className="setting-group">
        <div className="setting-group-heading"><h3>Desktop alerts</h3><p>When Alphonse needs your input or completes a task, notify you while this window is in the background.</p></div>
        <label className="setting-row"><input type="checkbox" checked={desktopNotifications.enabled} onChange={(event) => onDesktopNotificationsChange({ ...desktopNotifications, enabled: event.target.checked })} /> Show native notifications</label>
        <label className="setting-row"><input type="checkbox" checked={desktopNotifications.sound} disabled={!desktopNotifications.enabled} onChange={(event) => onDesktopNotificationsChange({ ...desktopNotifications, sound: event.target.checked })} /> Play an alert sound</label>
        <div className="input-row"><label>Notification sound<input readOnly value={desktopNotifications.soundFile || "Use macOS default alert sound"} aria-label="Notification sound" /></label></div>
        <div className="settings-save-actions"><button type="button" disabled={!desktopNotifications.enabled || !desktopNotifications.sound} onClick={() => void chooseNotificationSound()}>Choose sound file</button><button type="button" disabled={!desktopNotifications.enabled || !desktopNotifications.sound || !desktopNotifications.soundFile} onClick={() => onDesktopNotificationsChange({ ...desktopNotifications, soundFile: "" })}>Use macOS default</button></div>
        <label className="setting-row"><input type="checkbox" checked={desktopNotifications.onlyWhenUnfocused} disabled={!desktopNotifications.enabled} onChange={(event) => onDesktopNotificationsChange({ ...desktopNotifications, onlyWhenUnfocused: event.target.checked })} /> Notify only when Alphonse is not focused</label>
        <div className="settings-save-actions"><button type="button" disabled={!desktopNotifications.enabled} onClick={onTestDesktopNotification}>Test alert</button></div>
      </section>
      {isAdmin && <><section className="setting-group">
        <div className="setting-group-heading"><h3>Workspace &amp; time</h3><p>Set the root directory Alphonse uses for users, projects, skills, and agent configuration. This must match the Workspace directory selected by the administrator.</p></div>
        {!root && <p role="alert">No Workspace directory has been configured. Choose one to enable normal use.</p>}
        <label>Workspace directory</label><div className="workspace-picker"><input aria-label="Selected Workspace directory" readOnly value={root} placeholder="Choose a folder in Finder" /><button type="button" onClick={() => void chooseWorkspace()}>Choose folder…</button></div>
        <label>Timezone<input value={timezone} placeholder="America/Mexico_City" onChange={(event) => setTimezone(event.target.value)} /><small>Use an IANA timezone. New scheduled tasks use this unless one is explicitly specified.</small></label>
        <div className="settings-save-actions"><button type="button" disabled={!root.trim() || savingWorkspace} onClick={() => void saveWorkspace()}>{savingWorkspace ? "Migrating & saving…" : "Save Workspace & time"}</button>{workspaceNotice && <p role="status" aria-live="polite">{workspaceNotice}</p>}</div>
      </section>
      <section className="setting-group">
        <div className="setting-group-heading"><h3>Delivery</h3><p>Control whether automation messages are also delivered outside of Desktop.</p></div>
        <label className="setting-row"><input type="checkbox" checked={mirrorAutomationMessages} onChange={(event) => setMirrorAutomationMessages(event.target.checked)} /> Also send reminders and scheduled-task messages to the recipient’s preferred channel</label>
        <small>Desktop delivery remains unchanged. When enabled, Alphonse sends one additional copy to the recipient’s preferred configured channel.</small>
        <div className="settings-save-actions"><button onClick={() => void save()}>Save delivery settings</button>{notice && <p role="status">{notice}</p>}</div>
      </section>
      <section className="setting-group">
        <div className="setting-group-heading"><h3>Conversation memory</h3><p>Set how much conversation history Alphonse retains before it compacts the ledger.</p></div>
        {memory && <><label>Ledger limit (bytes)<input type="number" value={memory.max_ledger_bytes} onChange={(event) => setMemory({ ...memory, max_ledger_bytes: Number(event.target.value) })} /></label><label>Compaction limit (words)<input type="number" value={memory.compaction_summary_max_words} onChange={(event) => setMemory({ ...memory, compaction_summary_max_words: Number(event.target.value) })} /></label><div className="settings-save-actions"><button onClick={() => void saveMemory()}>Save memory settings</button>{memoryNotice && <p role="status">{memoryNotice}</p>}</div></>}
        {!memory && memoryNotice && <p role="status">{memoryNotice}</p>}
      </section>
      </>}
    </div>}
    {visitedSections.has("appearance") && <div hidden={tab !== "appearance"}><AppearanceSettingsSection value={desktopStyle} onChange={onDesktopStyleChange} /></div>}
    {visitedSections.has("tools") && isAdmin && <div hidden={tab !== "tools"} className="tools-settings">
      <CodeModeSettingsSection user={user} />
      <section className="setting-group">
        <div className="setting-group-heading"><h3>Web Tools</h3><p>Run SearXNG separately in Docker with JSON output enabled. Alphonse connects to it; it does not manage Docker.</p></div>
        {web && <>
          <label className="setting-row"><input type="checkbox" checked={web.enabled} onChange={(event) => setWeb({ ...web, enabled: event.target.checked })} /> Enable Web Search and Fetch</label>
          <div className="input-row"><label>SearXNG URL<input value={web.searxng_base_url} placeholder="http://127.0.0.1:8080" onChange={(event) => setWeb({ ...web, searxng_base_url: event.target.value })} /></label></div>
          <div className="input-row"><label>Search timeout (seconds)<input type="number" value={web.search_timeout_seconds} onChange={(event) => setWeb({ ...web, search_timeout_seconds: Number(event.target.value) })} /></label></div>
          <div className="input-row"><label>Fetch timeout (seconds)<input type="number" value={web.fetch_timeout_seconds} onChange={(event) => setWeb({ ...web, fetch_timeout_seconds: Number(event.target.value) })} /></label></div>
          <div className="input-row"><label>Fetch text limit<input type="number" value={web.fetch_max_chars} onChange={(event) => setWeb({ ...web, fetch_max_chars: Number(event.target.value) })} /></label></div>
          <div className="settings-save-actions"><button onClick={() => void saveWeb()}>Save Web Tools</button><button onClick={() => void verify("search")}>Verify SearXNG</button><button onClick={() => void verify("fetch")}>Verify Fetch</button>{webNotice && <p role="status">{webNotice}</p>}</div>
        </>}
        {!web && webNotice && <p role="status">{webNotice}</p>}
      </section>
      <div className="setting-group"><MediaToolsSettingsSection /></div>
    </div>}
    {visitedSections.has("integrations") && isAdmin && <div hidden={tab !== "integrations"}><IntegrationsSettingsSection user={user} /></div>}
    {visitedSections.has("automations") && isAdmin && <div hidden={tab !== "automations"}><AutomationsSettingsSection user={user} /></div>}
    {visitedSections.has("model") && isAdmin && <div hidden={tab !== "model"}><ModelSettingsSection /></div>}
    {visitedSections.has("system-one") && isAdmin && <div hidden={tab !== "system-one"}><SystemOneSettingsSection user={user} /></div>}
    {visitedSections.has("agent-config") && isAdmin && <div hidden={tab !== "agent-config"}><AgentConfigSettingsSection /></div>}
    {visitedSections.has("skills") && isAdmin && <div className="settings-skills-screen" hidden={tab !== "skills"}><SkillsWorkspace user={user} /></div>}
    {visitedSections.has("members") && isAdmin && <div hidden={tab !== "members"}><UsersModal embedded onClose={onClose} /></div>}
  </section>;
}

function CodeModeSettingsSection({ user }: { user: string }) {
  const [settings, setSettings] = useState<CodeModeSettings | null>(null);
  const [notice, setNotice] = useState("");
  const [verifying, setVerifying] = useState(false);
  const load = async () => {
    const result = await daemonRequest<{ settings: CodeModeSettings }>("code_mode_settings", { actor_user_id: user });
    setSettings(result.settings);
  };
  useEffect(() => { void load().catch((cause: unknown) => setNotice(cause instanceof Error ? cause.message : "Code Mode settings unavailable")); }, [user]);
  const save = async () => {
    if (!settings) return;
    const weakened = [!settings.network_disabled && "network access", !settings.read_only_filesystem && "a writable container filesystem", !settings.run_as_non_root && "root container user", !settings.drop_all_capabilities && "Linux capabilities", !settings.no_new_privileges && "privilege escalation protection"].filter(Boolean) as string[];
    if (weakened.length && !window.confirm(`Advanced Sandbox Config weakens: ${weakened.join(", ")}. Generated programs may gain access to local resources. Save this configuration?`)) return;
    try {
      const result = await daemonRequest<{ settings: CodeModeSettings }>("save_code_mode_settings", { actor_user_id: user, values: settings, acknowledge_unsafe: weakened.length > 0 });
      setSettings(result.settings); setNotice("Saved. New program phases use this configuration.");
    } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Code Mode settings could not be saved"); }
  };
  const verify = async () => {
    setVerifying(true); setNotice("Verifying Docker and the configured Python image…");
    try {
      const result = await daemonRequest<{ settings: CodeModeSettings; result: { ready: boolean; error?: string; preview?: string } }>("verify_code_mode", { actor_user_id: user });
      setSettings(result.settings); setNotice(result.result.ready ? (result.result.preview || "Docker verified.") : (result.result.error || "Docker verification failed."));
    } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Docker verification failed"); } finally { setVerifying(false); }
  };
  const restoreSafeDefaults = () => {
    if (!settings || !window.confirm("Restore all Code Mode settings to their safe defaults? Save afterward to apply them.")) return;
    setSettings({ ...settings, enabled: false, docker_bin: "docker", image: "python:3.11-slim", timeout_seconds: 60, max_tool_calls: 16, max_parallel_calls: 4, memory_mb: 256, cpu_count: 0.5, pid_limit: 64, tmpfs_mb: 64, network_disabled: true, read_only_filesystem: true, run_as_non_root: true, drop_all_capabilities: true, no_new_privileges: true, weakened_protections: [] });
    setNotice("Safe defaults restored locally. Save to apply them.");
  };
  const toggle = (field: "network_disabled" | "read_only_filesystem" | "run_as_non_root" | "drop_all_capabilities" | "no_new_privileges", label: string) => settings && <label className="setting-row"><input type="checkbox" checked={settings[field]} onChange={(event) => setSettings({ ...settings, [field]: event.target.checked })} /> {label}</label>;
  if (!settings) return <section className="setting-group"><div className="setting-group-heading"><h3>Code Mode</h3><p>{notice || "Loading…"}</p></div></section>;
  return <section className="setting-group">
    <div className="setting-group-heading"><h3>Code Mode</h3><p>Run a bounded Python program that can call Alphonse’s host-side tools. Docker is required; no `.env` credentials are used.</p></div>
    <label className="setting-row"><input type="checkbox" checked={settings.enabled} onChange={(event) => setSettings({ ...settings, enabled: event.target.checked })} /> Enable Code Mode</label>
    <p role="status">Docker status: {settings.verification_ready ? "verified" : "not verified"}{settings.verification_error ? ` — ${settings.verification_error}` : ""}</p>
    <div className="input-row"><label>Docker binary<input value={settings.docker_bin} onChange={(event) => setSettings({ ...settings, docker_bin: event.target.value })} /></label></div>
    <div className="input-row"><label>Python image<input value={settings.image} onChange={(event) => setSettings({ ...settings, image: event.target.value })} /></label></div>
    <div className="input-row"><label>Program timeout (seconds)<input type="number" value={settings.timeout_seconds} onChange={(event) => setSettings({ ...settings, timeout_seconds: Number(event.target.value) })} /></label></div>
    <div className="input-row"><label>Tool-call limit<input type="number" value={settings.max_tool_calls} onChange={(event) => setSettings({ ...settings, max_tool_calls: Number(event.target.value) })} /></label></div>
    <div className="input-row"><label>Parallel-call limit<input type="number" value={settings.max_parallel_calls} onChange={(event) => setSettings({ ...settings, max_parallel_calls: Number(event.target.value) })} /></label></div>
    <section className="advanced-sandbox-config" aria-labelledby="advanced-sandbox-config-title">
      <div className="advanced-sandbox-config-heading"><h4 id="advanced-sandbox-config-title">Advanced Sandbox Config</h4><p role="alert">Warning: weakening these protections can expose the local machine, network, credentials, or family data to generated programs. Each unsafe save requires confirmation.</p></div>
      <div className="input-row"><label>Memory limit (MB)<input type="number" value={settings.memory_mb} onChange={(event) => setSettings({ ...settings, memory_mb: Number(event.target.value) })} /></label></div>
      <div className="input-row"><label>CPU limit<input type="number" step="0.1" value={settings.cpu_count} onChange={(event) => setSettings({ ...settings, cpu_count: Number(event.target.value) })} /></label></div>
      <div className="input-row"><label>PID limit<input type="number" value={settings.pid_limit} onChange={(event) => setSettings({ ...settings, pid_limit: Number(event.target.value) })} /></label></div>
      <div className="input-row"><label>Temporary filesystem (MB)<input type="number" value={settings.tmpfs_mb} onChange={(event) => setSettings({ ...settings, tmpfs_mb: Number(event.target.value) })} /></label></div>
      <div className="sandbox-protections">{toggle("network_disabled", "Disable network access")}{toggle("read_only_filesystem", "Use a read-only container filesystem")}{toggle("run_as_non_root", "Run as a non-root user")}{toggle("drop_all_capabilities", "Drop all Linux capabilities")}{toggle("no_new_privileges", "Prevent new privileges")}</div>
      <div className="sandbox-restore-actions"><button type="button" onClick={restoreSafeDefaults}>Restore safe defaults</button></div>
    </section>
    <div className="settings-save-actions"><button onClick={() => void save()}>Save Code Mode</button><button disabled={verifying} onClick={() => void verify()}>{verifying ? "Verifying Docker…" : "Verify Docker"}</button>{notice && <p role="status">{notice}</p>}</div>
  </section>;
}

function AppearanceSettingsSection({ value, onChange }: { value: DesktopStyle; onChange: (value: DesktopStyle) => void }) {
  return <section className="settings-panel appearance-settings">
    <div><h3>Appearance</h3><p>Warm, quiet surfaces for everyday life. Your preference applies immediately and is saved for your household profile on this Mac.</p></div>
    <label>Color appearance
      <select value={value} onChange={(event) => onChange(event.target.value as DesktopStyle)}>
        <option value="system">Follow system</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </label>
    <div className="typography-preview"><h4>A little help, every day.</h4><p>Manrope headings · DM Sans reading and controls</p><small>Bundled fonts work offline. Motion follows your system’s reduced-motion preference.</small></div>
  </section>;
}

function MediaToolsSettingsSection() {
  const [settings, setSettings] = useState<MediaToolsSettings | null>(null); const [notice, setNotice] = useState(""); const [verifying, setVerifying] = useState<"" | "tts" | "stt">(""); const [samples, setSamples] = useState<Record<string, string>>({ tts: "Alphonse text-to-speech verification.", stt: "" });
  const [recording, setRecording] = useState(false); const recorder = useRef<MediaRecorder | null>(null); const recordingStartedAt = useRef(0); const recordingTimeout = useRef<number | null>(null);
  const load = async () => { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ settings: MediaToolsSettings }>("media_tools_settings", { actor_user_id: current.user.user_id }); setSettings(result.settings); };
  useEffect(() => { void load().catch((cause: unknown) => setNotice(cause instanceof Error ? cause.message : "Media Tools unavailable")); }, []);
  const save = async (kind: "tts" | "stt") => { if (!settings) return; try { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ settings: MediaToolsSettings }>("save_media_tools_settings", { actor_user_id: current.user.user_id, kind, values: settings[kind] }); setSettings(result.settings); setNotice(`${kind.toUpperCase()} saved; verify it to mark it ready.`); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Save failed"); } };
  const verify = async (kind: "tts" | "stt") => { setVerifying(kind); setNotice(`Verifying ${kind.toUpperCase()}… this may take a while while the local model loads.`); try { const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return; const result = await daemonRequest<{ settings: MediaToolsSettings; result: { exception?: { message?: string } } }>("verify_media_tools", { actor_user_id: current.user.user_id, kind, sample: samples[kind] }); setSettings(result.settings); setNotice(result.result.exception?.message || `${kind.toUpperCase()} verified.`); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Verification failed"); } finally { setVerifying(""); } };
  const stopTracks = (stream: MediaStream | null) => stream?.getTracks().forEach((track) => track.stop());
  const stopRecording = () => { if (recordingTimeout.current !== null) { window.clearTimeout(recordingTimeout.current); recordingTimeout.current = null; } if (recorder.current?.state === "recording") recorder.current.stop(); };
  const startRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") { setNotice("This Alphonse Desktop build does not expose microphone capture. Update or rebuild the desktop app so macOS can grant microphone access."); return; }
    try {
      const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user || !settings) return;
      const saved = await daemonRequest<{ settings: MediaToolsSettings }>("save_media_tools_settings", { actor_user_id: current.user.user_id, kind: "stt", values: settings.stt }); setSettings(saved.settings);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferredType = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"].find((value) => MediaRecorder.isTypeSupported(value)) || "";
      const chunks: BlobPart[] = []; const instance = new MediaRecorder(stream, preferredType ? { mimeType: preferredType } : undefined);
      recorder.current = instance; recordingStartedAt.current = Date.now(); setRecording(true); setNotice("Recording… speak a short message, then stop and verify.");
      instance.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      instance.onerror = () => { stopTracks(stream); setRecording(false); setNotice("The microphone recording failed."); };
      instance.onstop = async () => {
        recorder.current = null; stopTracks(stream); setRecording(false);
        const durationMs = Date.now() - recordingStartedAt.current; const blob = new Blob(chunks, { type: instance.mimeType || preferredType });
        if (!blob.size) { setNotice("No audio was recorded. Please try again."); return; }
        setVerifying("stt"); setNotice("Transcribing your recording…");
        try {
          const current = await daemonRequest<{ user: { user_id: string } | null }>("current_user"); if (!current.user) return;
          const result = await daemonRequest<{ settings: MediaToolsSettings; result: { exception?: { message?: string } } }>("verify_stt_recording", { actor_user_id: current.user.user_id, audio_base64: await blobToBase64(blob), mime_type: blob.type || preferredType, duration_ms: durationMs });
          setSettings(result.settings); setNotice(result.result.exception?.message || `STT verified. Transcript: ${result.settings.stt.verification.preview}`);
        } catch (cause) { setNotice(cause instanceof Error ? cause.message : "STT verification failed"); } finally { setVerifying(""); }
      };
      instance.start(); recordingTimeout.current = window.setTimeout(stopRecording, 30_000);
    } catch (cause) { setRecording(false); setNotice(cause instanceof DOMException && cause.name === "NotAllowedError" ? "Microphone permission was denied. Allow microphone access and try again." : cause instanceof Error ? cause.message : "Could not start the microphone."); }
  };
  useEffect(() => () => { if (recordingTimeout.current !== null) window.clearTimeout(recordingTimeout.current); stopRecording(); }, []);
  if (!settings) return <section><h3>Local Media Tools</h3><p>{notice || "Loading…"}</p></section>;
  const field = (kind: "tts" | "stt", key: string, label: string, type = "text") => <div className="input-row"><label>{label}<input type={type} value={String((settings[kind] as Record<string, unknown>)[key] ?? "")} onChange={(event) => setSettings({ ...settings, [kind]: { ...settings[kind], [key]: type === "number" ? Number(event.target.value) : event.target.value } })} /></label></div>;
  const toggle = (kind: "tts" | "stt", key: string, label: string) => <label className="setting-row"><input type="checkbox" checked={Boolean((settings[kind] as Record<string, unknown>)[key])} onChange={(event) => setSettings({ ...settings, [kind]: { ...settings[kind], [key]: event.target.checked } })} /> {label}</label>;
  const verification = (kind: "tts" | "stt") => <p>{verifying === kind && <><span className="verification-spinner" aria-label="Verifying" /> Verifying {kind.toUpperCase()}… </>}Ready: {String(settings[kind].available)}. {settings[kind].verification.error || settings[kind].verification.preview || "Not yet verified."}</p>;
  const verifyLabel = (kind: "tts" | "stt", label: string) => verifying === kind ? `Verifying ${kind.toUpperCase()}…` : label;
  return <section className="media-tools-settings">
    <div className="setting-group-heading"><h3>Local Media Tools</h3><p>Install models and runtimes separately, then save and verify each backend. Images are processed by the selected inference model; skills guide their interpretation.</p></div>
    <section className="media-tool-panel">
      <div className="media-tool-panel-heading"><h4>Qwen TTS</h4><span className={`feature-status ${settings.tts.enabled ? "enabled" : "disabled"}`}>{settings.tts.enabled ? "Enabled" : "Disabled"}</span></div>{toggle("tts", "enabled", "Enable Qwen TTS")}{field("tts", "model_id", "Model ID or local path")}{field("tts", "device_map", "Device map")}{field("tts", "dtype", "Precision")}{field("tts", "language", "Language")}{field("tts", "speaker", "Speaker")}{field("tts", "instruct", "Voice instruction")}{field("tts", "attn_implementation", "Attention implementation")}{toggle("tts", "local_files_only", "Use local model files only")}{verification("tts")}
      <div className="input-row"><label>Test phrase<input disabled={verifying !== ""} value={samples.tts} onChange={(event) => setSamples({ ...samples, tts: event.target.value })} /></label></div>
      <div className="settings-save-actions"><button disabled={verifying !== ""} onClick={() => void save("tts")}>Save TTS</button><button disabled={verifying !== ""} onClick={() => void verify("tts")}>{verifyLabel("tts", "Verify Qwen TTS")}</button></div>
      {settings.platform === "darwin" && <p>macOS say fallback: {settings.say_available ? "available" : "not found"}.</p>}
    </section>
    <section className="media-tool-panel">
      <div className="media-tool-panel-heading"><h4>Whisper STT</h4><span className={`feature-status ${settings.stt.enabled ? "enabled" : "disabled"}`}>{settings.stt.enabled ? "Enabled" : "Disabled"}</span></div>{toggle("stt", "enabled", "Enable Whisper STT")}{field("stt", "executable_path", "Whisper executable")}{field("stt", "model", "Model")}{field("stt", "default_language", "Default language")}{verification("stt")}
      <p>Record a short message to confirm that Whisper can hear and transcribe it. The recording is deleted immediately after verification.</p>
      <div className="settings-save-actions"><button disabled={verifying !== ""} onClick={() => void (recording ? stopRecording() : startRecording())}>{recording ? "Stop & verify recording" : "Start microphone test"}</button></div>
      <div className="input-row"><label>Audio sample path (advanced)<input disabled={verifying !== "" || recording} value={samples.stt} onChange={(event) => setSamples({ ...samples, stt: event.target.value })} /></label></div>
      <div className="settings-save-actions"><button disabled={verifying !== "" || recording} onClick={() => void save("stt")}>Save STT</button><button disabled={verifying !== "" || recording} onClick={() => void verify("stt")}>{verifyLabel("stt", "Verify Whisper file")}</button></div>
    </section>
    {notice && <p role="status">{notice}</p>}
  </section>;
}

async function blobToBase64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer()); let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  return btoa(binary);
}

type ManagedAddress = { address_id: string; integration_id: string; provider_key: string; provider_user_id: string; channel_target: string; is_preferred: boolean };
type ManagedUser = { user_id: string; display_name: string; role: string; is_active: boolean; addresses?: ManagedAddress[] };
type ProjectConfig = { scope: string; goals: string[]; duration: { ongoing: boolean; start_date: string | null; end_date: string | null }; directory_structure: string[] };
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

function TaskProgressBubble({ surface, archived = false }: { surface: A2uiSurface; archived?: boolean }) {
  const text = (id: string) => String(surface.components[id]?.text || "").trim();
  const steps = Object.values(surface.components).filter((component) => component.id.startsWith("step_") && component.text).sort((left, right) => left.id.localeCompare(right.id, undefined, { numeric: true }));
  const taskContext = text("task_context");
  const criteria = text("criteria").replace(/^Acceptance criteria\n?/, "").trim();
  const visibleCriteria = criteria === "- (none)" ? "" : criteria;
  const Container = archived ? "details" : "div";
  return <article className="message assistant task-progress-bubble" aria-label="Task activity" aria-live="polite">
    <Container className="task-progress-log">
      {archived && <summary>Work log{text("summary") ? ` · ${text("summary")}` : ""}</summary>}
    <div className="task-progress-content">
      {!archived && <div className="task-progress-heading"><span className="task-progress-spinner" aria-hidden="true">◌</span><strong>Alphonse is working</strong></div>}
      {!archived && text("summary") && <p>{text("summary")}</p>}
      {visibleCriteria && <section><small>Acceptance criteria</small><div className="task-progress-checklist"><ReactMarkdown remarkPlugins={[remarkGfm]}>{acceptanceCriteriaMarkdown(visibleCriteria)}</ReactMarkdown></div></section>}
      {text("intention") && <section><small>Intention</small><p>{text("intention").replace(/^Intention:\s*/, "")}</p></section>}
      {steps.length ? <section className="task-progress-trace"><small>Execution audit</small>{steps.map((step) => {
        const [summary, ...detail] = String(step.text || "").split("\n");
        return <details className="task-progress-step" key={step.id}><summary>{summary}</summary><pre className="task-progress-detail">{detail.filter((line) => line !== "Details:").join("\n") || "No additional action details."}</pre></details>;
      })}</section> : <>{text("tool") && <p className="task-progress-detail">{text("tool")}</p>}{text("arguments") && <pre className="task-progress-detail">{text("arguments")}</pre>}{text("result") && <pre className="task-progress-detail">{text("result")}</pre>}</>}
      <DecisionContext summary={text("decision_context")} manifests={(surface.dataModel.decision_context_manifests || []) as Parameters<typeof DecisionContext>[0]["manifests"]} actorUserId={String(surface.dataModel.owner_user_id || "")} taskId={String(surface.dataModel.task_id || "")} />
      {taskContext && <details className="task-progress-context"><summary>Task state diagnostics</summary><pre className="task-progress-detail">{taskContext}</pre></details>}
    </div>
    </Container>
  </article>;
}

function questionTaskId(surface: A2uiSurface): string {
  const question = surface.dataModel.question;
  return typeof question === "object" && question !== null && "task_id" in question ? String((question as { task_id?: unknown }).task_id || "") : "";
}

function acceptanceCriteriaMarkdown(value: string): string {
  return value.split("\n").map((line) => line.replace(/^\s*\d+\s*\.\s*-?\s*(\[[ xX]\])\s*/, "- $1 ")).join("\n");
}

function ScheduledTasksModal({ embedded = false, actorUserId, initialTaskId = "", onClose }: { embedded?: boolean; actorUserId: string; initialTaskId?: string; onClose: () => void }) {
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
  return <ModalFrame embedded={embedded} title={embedded ? "Alphonse’s Agenda" : "Scheduled tasks"} onClose={onClose}>
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

function UsersModal({ embedded = false, onClose }: { embedded?: boolean; onClose: () => void }) {
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
  return <ModalFrame embedded={embedded} title={embedded ? "Household" : "Users"} onClose={onClose}>
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

function ProjectsWorkspace({ user, initialProjectId = "", selectionRequest = 0, onClose }: { user: string; initialProjectId?: string; selectionRequest?: number; onClose: () => void }) {
  const [projects, setProjects] = useState<ManagedProject[]>([]); const [users, setUsers] = useState<ManagedUser[]>([]); const [ownerId, setOwnerId] = useState(user); const [filter, setFilter] = useState(""); const [showCreate, setShowCreate] = useState(false);
  const [selectedProject, setSelectedProject] = useState<ManagedProject | null>(null);
  const [name, setName] = useState(""); const [description, setDescription] = useState(""); const [visibility, setVisibility] = useState<"private" | "shared">("private"); const [rootPath, setRootPath] = useState(""); const [mode, setMode] = useState<"create" | "import">("create"); const [notice, setNotice] = useState(""); const [newMemberIds, setNewMemberIds] = useState<string[]>([]);
  const load = useCallback(async () => { try { const [projectResult, userResult] = await Promise.all([daemonRequest<{ projects: ManagedProject[] }>("manageable_projects", { user, status: filter }), daemonRequest<{ users: ManagedUser[] }>("users")]); setProjects(projectResult.projects); setUsers(userResult.users); setOwnerId((current) => userResult.users.some((item) => item.user_id === current) ? current : userResult.users.find((item) => item.user_id === user)?.user_id || userResult.users.find((item) => item.role === "admin")?.user_id || ""); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Projects could not be loaded."); } }, [filter, user]);
  useEffect(() => { void load(); }, [load]);
  const refreshProjects = async (selectedId?: string) => { await load(); if (selectedId) { const latest = await daemonRequest<{ projects: ManagedProject[] }>("manageable_projects", { user }); setSelectedProject(latest.projects.find((item) => item.project_id === selectedId) || null); } };
  const createOrImport = async (event: FormEvent) => { event.preventDefault(); try { const method = mode === "import" ? "import_project" : "create_project"; const result = await daemonRequest<{ project: ManagedProject }>(method, { user, name, description, root_path: rootPath, visibility }); if (visibility === "shared" && newMemberIds.length) await daemonRequest("update_project", { user, project_id: result.project.project_id, name, description, visibility, member_user_ids: newMemberIds }); setNotice(mode === "import" ? "Project imported." : "Project created."); setShowCreate(false); await refreshProjects(result.project.project_id); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Project could not be created."); } };
  const appliedProjectId = useRef("");
  useEffect(() => {
    if (initialProjectId) setFilter("");
  }, [initialProjectId, selectionRequest]);
  useEffect(() => {
    const requestKey = `${initialProjectId}:${selectionRequest}`;
    if (!initialProjectId || appliedProjectId.current === requestKey) return;
    const project = projects.find((item) => item.project_id === initialProjectId);
    if (project) { appliedProjectId.current = requestKey; setSelectedProject(project); setShowCreate(false); setOwnerId(project.owner_user_id); }
  }, [initialProjectId, selectionRequest, projects]);
  const visibleProjects = projects.filter((item) => item.visibility === "shared" || item.owner_user_id === ownerId);
  return <div className="projects-workspace">
    <aside className="projects-workspace-sidebar">
      <header className="projects-workspace-heading"><button className="back-button" onClick={onClose}>← Conversation</button><h2>Projects</h2><button onClick={() => { setShowCreate((value) => !value); setSelectedProject(null); }}>{showCreate ? "Cancel" : "+ New project"}</button></header>
      <div className="projects-workspace-filters"><label htmlFor="project-status-filter">Show</label><select id="project-status-filter" value={filter} onChange={(event) => setFilter(event.target.value)}><option value="">All projects</option><option value="active">Active</option><option value="paused">Paused</option><option value="completed">Completed</option><option value="archived">Archived</option></select><label htmlFor="project-owner-filter">Owner</label><select id="project-owner-filter" value={ownerId} onChange={(event) => setOwnerId(event.target.value)}>{users.map((item) => <option key={item.user_id} value={item.user_id}>{item.display_name}</option>)}</select></div>
      {showCreate && <form className="project-create" onSubmit={createOrImport}><h3>{mode === "import" ? "Import existing folder" : "New project"}</h3><div className="dialog-actions"><button type="button" onClick={() => setMode("create")}>New</button><button type="button" onClick={() => setMode("import")}>Import</button></div><div className="form-field"><label htmlFor="new-project-name">Name</label><input id="new-project-name" value={name} onChange={(event) => setName(event.target.value)} required /></div><div className="form-field"><label htmlFor="new-project-description">Description</label><input id="new-project-description" value={description} onChange={(event) => setDescription(event.target.value)} /></div><div className="form-field"><label htmlFor="new-project-visibility">Visibility</label><select id="new-project-visibility" value={visibility} onChange={(event) => setVisibility(event.target.value as "private" | "shared")}><option value="private">Private</option><option value="shared">Shared</option></select></div>{visibility === "shared" && <fieldset className="form-field"><legend>Share with family members</legend>{users.filter((item) => item.is_active && item.user_id !== ownerId).map((member) => <label key={member.user_id}><input type="checkbox" checked={newMemberIds.includes(member.user_id)} onChange={(event) => setNewMemberIds((current) => event.target.checked ? [...current, member.user_id] : current.filter((id) => id !== member.user_id))} /> {member.display_name}</label>)}</fieldset>}<div className="form-field"><label htmlFor="new-project-path">{mode === "import" ? "Existing folder" : "Parent directory (optional)"}</label><input id="new-project-path" value={rootPath} onChange={(event) => setRootPath(event.target.value)} required={mode === "import"} /></div><button>{mode === "import" ? "Import project" : "Create project"}</button></form>}
      <nav className="projects-workspace-list" aria-label="Available projects">{visibleProjects.map((item) => <button type="button" key={item.project_id} className={selectedProject?.project_id === item.project_id ? "active" : ""} onClick={() => { setShowCreate(false); setSelectedProject(item); }}><FolderKanban size={17} /><span>{item.name}<small>{item.status} · {item.visibility === "shared" ? "Shared" : "Private"}</small></span><ChevronRight size={16} /></button>)}{!visibleProjects.length && <p>No projects found.</p>}</nav>
      {notice && <p className="projects-workspace-notice">{notice}</p>}
    </aside>
    <section className="projects-workspace-main">{showCreate ? <div className="projects-workspace-empty"><h1>Create a project</h1><p>Fill out the project details in the sidebar.</p></div> : selectedProject ? <ProjectEditorPanel key={selectedProject.project_id} user={user} project={selectedProject} onBack={() => { setSelectedProject(null); void load(); }} onUpdated={() => void refreshProjects(selectedProject.project_id)} onClose={onClose} /> : <div className="projects-workspace-empty"><FolderKanban size={36} /><h1>Choose a project</h1><p>Select a project from the list to manage its context and files.</p></div>}</section>
  </div>;
}

const SKILL_NAME_MAX_LENGTH = 64;
const SKILL_DESCRIPTION_MAX_LENGTH = 512;
const SKILL_INSTRUCTIONS_MAX_LENGTH = 12_000;
const SKILL_INSTRUCTIONS_TEMPLATE = "# Purpose\n[What reusable expertise does this skill provide?]\n\n## When to use\n- [Describe the requests that should use this skill.]\n\n## Workflow\n1. [Describe the steps Alphonse should follow.]\n\n## Files and tools\n- [List useful files relative to this skill directory and explain how to use them.]\n\n## Boundaries\n- [Describe safety, privacy, authorization, and uncertainty limits.]\n\n## Response\n- [Describe what a helpful answer or result should include.]";

type ArtifactWorkspaceRecord = { artifact_id: string; name: string; description: string; entrypoint_path: string; project_id: string; enabled: boolean; storage_kind: "shared" | "skill" | "project"; execution: { working_directory?: string; runtime_path?: string; configuration_paths?: string[]; required_files?: string[] }; shared_with_user_ids: string[]; resource: { status: string; issues: string[]; storage_root?: string; resolved_entrypoint_path?: string; working_directory?: string; runtime_path?: string } };
type SkillWorkspaceRecord = { appearance?: SkillAppearance | null; appearance_configured?: boolean; id: string; title: string; description: string; directory: string; editable: boolean; instructions: string; files: Array<{ path: string; size_bytes: number }>; artifacts: ArtifactWorkspaceRecord[] };
function ArtifactLocationEditor({ artifact, user, onSaved }: { artifact: ArtifactWorkspaceRecord; user: string; onSaved: () => void }) {
  const [kind, setKind] = useState(artifact.storage_kind || "project");
  const [entrypoint, setEntrypoint] = useState(artifact.entrypoint_path);
  const [cwd, setCwd] = useState(artifact.execution?.working_directory || ".");
  const [runtime, setRuntime] = useState(artifact.execution?.runtime_path || "");
  const [configuration, setConfiguration] = useState((artifact.execution?.configuration_paths || []).join("\n"));
  const [required, setRequired] = useState((artifact.execution?.required_files || []).join("\n"));
  const [sharing, setSharing] = useState((artifact.shared_with_user_ids || []).join(", "));
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const lines = (value: string) => value.split("\n").map((item) => item.trim()).filter(Boolean);
  const save = async () => { setBusy(true); try {
    await daemonRequest("relocate_artifact", { actor_user_id: user, artifact_id: artifact.artifact_id, storage_kind: kind, entrypoint_path: entrypoint, execution: { working_directory: cwd, runtime_path: runtime, configuration_paths: lines(configuration), required_files: lines(required) }, shared_with_user_ids: sharing.split(",").map((item) => item.trim()).filter(Boolean) });
    setNotice("Registration updated. Program and data files were preserved."); onSaved();
  } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Location could not be saved."); } finally { setBusy(false); } };
  const inspect = async () => { try { const result = await daemonRequest<{ artifacts: Array<{ artifact_id: string; reconciliation: string; location_candidates: Array<{ resolved_entrypoint_path: string }> }> }>("reconcile_artifacts", { actor_user_id: user }); const found = result.artifacts.find((item) => item.artifact_id === artifact.artifact_id); setNotice(found ? `${found.reconciliation}: ${found.location_candidates.map((item) => item.resolved_entrypoint_path).join(", ") || "No registered location found. Choose the intended file explicitly."}` : "No location report available."); } catch (cause) { setNotice(String(cause)); } };
  return <details><summary>Storage · {artifact.storage_kind || "project"} · {artifact.resource?.status || "unknown"}</summary>
    <p>{artifact.resource?.resolved_entrypoint_path || artifact.entrypoint_path}</p>
    {!!artifact.resource?.issues?.length && <p role="status">{artifact.resource.issues.join(", ")}</p>}
    <label>Storage<select value={kind} onChange={(event) => setKind(event.target.value as typeof kind)}><option value="shared">Shared skill repository</option><option value="skill">Installed skill package</option><option value="project">Legacy project location</option></select></label>
    <label>Entry point relative to storage root<input value={entrypoint} onChange={(event) => setEntrypoint(event.target.value)} /></label>
    <label>Working directory relative to storage root<input value={cwd} onChange={(event) => setCwd(event.target.value)} /></label>
    <label>Runtime executable (absolute or relative to storage root)<input value={runtime} onChange={(event) => setRuntime(event.target.value)} /></label>
    <label>Configuration paths (one relative path per line)<textarea value={configuration} onChange={(event) => setConfiguration(event.target.value)} /></label>
    <label>Required files (one relative path per line)<textarea value={required} onChange={(event) => setRequired(event.target.value)} /></label>
    <label>Share with household user IDs (comma-separated)<input value={sharing} onChange={(event) => setSharing(event.target.value)} /></label>
    <p>Saving updates the registration only. It does not move files or repair the program.</p>
    <button disabled={busy} onClick={() => void save()}>Save location</button><button disabled={busy} onClick={() => void inspect()}>Check known locations</button>
    {notice && <p role="status">{notice}</p>}
  </details>;
}

function SkillsWorkspace({ user }: { user: string }) {
  const [skills, setSkills] = useState<SkillWorkspaceRecord[]>([]); const [unassigned, setUnassigned] = useState<Array<{ artifact_id: string; name: string; description: string; entrypoint_path: string; project_id: string }>>([]); const [selected, setSelected] = useState<SkillWorkspaceRecord | null>(null); const [showCreate, setShowCreate] = useState(false); const [name, setName] = useState(""); const [description, setDescription] = useState(""); const [instructions, setInstructions] = useState(""); const [filePath, setFilePath] = useState("SKILL.md"); const [fileContent, setFileContent] = useState(""); const [artifactInstructions, setArtifactInstructions] = useState(""); const [notice, setNotice] = useState(""); const [busy, setBusy] = useState(false);
  const [savedFile, setSavedFile] = useState({ path: "SKILL.md", content: "" });
  const fileRequest = useRef(0);
  const fileDirty = filePath !== savedFile.path || fileContent !== savedFile.content;
  const dirty = showCreate ? Boolean(name || description || instructions) : Boolean(selected && (description !== selected.description || instructions !== selected.instructions || fileDirty || artifactInstructions));
  const discardEdits = () => !dirty || window.confirm("Discard unsaved skill edits? Cancel to keep editing, or save each edited section first.");
  const load = useCallback(async (chooseId?: string) => { try { const result = await daemonRequest<{ skills: SkillWorkspaceRecord[]; unassigned_artifacts: typeof unassigned }>("skills", { actor_user_id: user }); setSkills(result.skills); setUnassigned(result.unassigned_artifacts); setSelected((current) => result.skills.find((item) => item.id === (chooseId || current?.id)) || null); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Skills could not be loaded."); } }, [user]);
  useEffect(() => { void load(); }, [load]);
  const create = async (event: FormEvent) => { event.preventDefault(); const createdDescription = description; const createdInstructions = instructions; setBusy(true); try { const result = await daemonRequest<{ skill: SkillWorkspaceRecord }>("create_skill", { actor_user_id: user, name, description, instructions }); setName(""); setDescription(""); setInstructions(""); setShowCreate(false); setNotice("Skill created."); await load(result.skill.id); setDescription(createdDescription); setInstructions(createdInstructions); setFilePath("SKILL.md"); void selectFileForSkill(result.skill); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Skill could not be created."); } finally { setBusy(false); } };
  const saveDescription = async () => { if (!selected) return; try { await daemonRequest("update_skill", { actor_user_id: user, skill_id: selected.id, description }); setNotice("Skill details saved."); await load(selected.id); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Skill details could not be saved."); } };
  const saveDefinition = async () => { if (!selected) return; setBusy(true); try { const result = await daemonRequest<{ content: string }>("read_skill_file", { actor_user_id: user, skill_id: selected.id, path: "SKILL.md" }); const next = result.content.replace(/(^---\s*[\s\S]*?---\s*)[\s\S]*/, `$1\n\n${instructions.trim()}\n`); await daemonRequest("write_skill_file", { actor_user_id: user, skill_id: selected.id, path: "SKILL.md", content: next }); setNotice("Skill definition saved."); await load(selected.id); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Definition could not be saved."); } finally { setBusy(false); } };
  const selectFile = async (path: string) => { if (!selected || (fileDirty && !window.confirm("Discard unsaved file edits?"))) return; const request = ++fileRequest.current; try { const result = await daemonRequest<{ content: string }>("read_skill_file", { actor_user_id: user, skill_id: selected.id, path }); if (request !== fileRequest.current) return; setFilePath(path); setFileContent(result.content); setSavedFile({ path, content: result.content }); setNotice(""); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "File could not be opened."); } };
  const saveFile = async () => { if (!selected) return; try { await daemonRequest("write_skill_file", { actor_user_id: user, skill_id: selected.id, path: filePath, content: fileContent }); setSavedFile({ path: filePath, content: fileContent }); setNotice("File saved."); await load(selected.id); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "File could not be saved."); } };
  const addFile = async () => { if (!selected) return; try { const picked = await openFileDialog({ title: "Choose a file to add to this skill", multiple: false, directory: false }); if (!picked || Array.isArray(picked)) return; setBusy(true); const result = await daemonRequest<{ path: string }>("add_skill_file", { actor_user_id: user, skill_id: selected.id, source_path: picked }); await load(selected.id); setNotice(`Added ${result.path} to ${selected.title}.`); await selectFile(result.path); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "File could not be added."); } finally { setBusy(false); } };
  const deleteFile = async () => { if (!selected || filePath === "SKILL.md" || !window.confirm(`Delete ${filePath}?`)) return; try { await daemonRequest("delete_skill_file", { actor_user_id: user, skill_id: selected.id, path: filePath }); await selectFileForSkill(selected); await load(selected.id); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "File could not be deleted."); } };
  const uninstall = async () => { if (!selected || !window.confirm(`Uninstall ${selected.title} and remove its installed package?`)) return; try { await daemonRequest("delete_skill", { actor_user_id: user, skill_id: selected.id }); setSelected(null); await load(); setNotice("Skill uninstalled."); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Skill could not be uninstalled."); } };
  const attachArtifact = async (artifact: (typeof unassigned)[number]) => { if (!selected) return; try { await daemonRequest("attach_artifact_to_skill", { actor_user_id: user, artifact_id: artifact.artifact_id, skill_id: selected.id, instructions: artifactInstructions }); setUnassigned((current) => current.filter((item) => item.artifact_id !== artifact.artifact_id)); setArtifactInstructions(""); setNotice(`${artifact.name} added to ${selected.title}.`); await load(selected.id); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Artifact could not be assigned."); } };
  const deleteArtifact = async (artifact: { artifact_id: string; name: string }) => { if (!window.confirm(`Remove ${artifact.name} from Alphonse's artifact catalog? Program and data files will remain.`)) return; try { await daemonRequest("delete_artifact", { actor_user_id: user, artifact_id: artifact.artifact_id }); setNotice(`${artifact.name} removed from the artifact catalog.`); await load(selected?.id); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Artifact could not be deleted."); } };
  const openSkillFolder = async (item: SkillWorkspaceRecord) => { try { await showInFinder(item.directory); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Skill folder could not be opened."); } };
  const importPackage = async (replace = false) => {
    if (busy || !discardEdits() || (replace && !selected)) return;
    try {
      const source = await openFileDialog({ title: replace ? "Choose updated skill package" : "Install skill from folder", directory: true, multiple: false });
      if (typeof source !== "string") return;
      if (replace && !window.confirm(`Replace ${selected?.title} with the selected package?`)) return;
      setBusy(true);
      const result = await daemonRequest<{ skill: { id: string } }>(replace ? "replace_skill" : "install_skill", { actor_user_id: user, ...(replace ? { skill_id: selected?.id } : {}), source_directory: source });
      const catalog = await daemonRequest<{ skills: SkillWorkspaceRecord[]; unassigned_artifacts: typeof unassigned }>("skills", { actor_user_id: user });
      const installed = catalog.skills.find((item) => item.id === result.skill.id);
      setSkills(catalog.skills); setUnassigned(catalog.unassigned_artifacts);
      setSelected(installed || null); setShowCreate(false); setArtifactInstructions("");
      if (installed) { setDescription(installed.description); setInstructions(installed.instructions); await selectFileForSkill(installed); }
      setNotice(replace ? "Skill package updated." : "Skill installed.");
    } catch (cause) { setNotice(errorNotice(cause, "Skill package could not be installed.")); }
    finally { setBusy(false); }
  };
  const startCreate = () => { setName(""); setDescription(""); setInstructions(""); setSelected(null); setShowCreate(true); setNotice(""); };
  return <div className="projects-workspace skills-workspace"><aside className="projects-workspace-sidebar"><header className="projects-workspace-heading"><h2>Skills</h2><button disabled={busy} onClick={() => void importPackage()}>Install from folder…</button><button onClick={() => { if (!discardEdits()) return; showCreate ? setShowCreate(false) : startCreate(); }}>{showCreate ? "Cancel" : "+ New skill"}</button></header>
    <nav className="projects-workspace-list" aria-label="Available skills">{skills.map((item) => <button type="button" key={item.id} className={selected?.id === item.id ? "active" : ""} onClick={() => { if (busy || !discardEdits()) return; setShowCreate(false); setSelected(item); setDescription(item.description); setInstructions(item.instructions); void selectFileForSkill(item); }}><Sparkles size={17} /><span>{item.title}<small>{item.description}</small></span><ChevronRight size={16} /></button>)}{!skills.length && <p>No skills installed.</p>}</nav>{notice && <p className="projects-workspace-notice">{notice}</p>}</aside>
    <section className="projects-workspace-main">{showCreate ? <article className="skill-editor skill-create-editor"><header><div><p className="eyebrow">New skill</p><h1>Create a skill</h1><p>Keep the instructions focused so Alphonse can load them alongside other relevant skills.</p></div></header><form onSubmit={(event) => void create(event)}><section><h2>Skill name</h2><label htmlFor="new-skill-name">Use a short lowercase name with hyphens between words.</label><input id="new-skill-name" value={name} onChange={(event) => setName(event.target.value)} maxLength={SKILL_NAME_MAX_LENGTH} placeholder="family-doctor" required /><small className="skill-character-count">{name.length}/{SKILL_NAME_MAX_LENGTH} characters</small></section><section><h2>Description</h2><label htmlFor="new-skill-description">Summarize the skill’s expertise and when Alphonse should use it.</label><textarea id="new-skill-description" value={description} onChange={(event) => setDescription(event.target.value)} maxLength={SKILL_DESCRIPTION_MAX_LENGTH} placeholder="Family health guidance, medical history, medications, and appointments. Use for health records or family medical questions." required rows={3} /><small className="skill-character-count">{description.length}/{SKILL_DESCRIPTION_MAX_LENGTH} characters</small></section><section><h2>Instructions</h2><label htmlFor="new-skill-instructions">Describe a practical workflow, relevant files, tools, and boundaries. Keep this under 12,000 characters.</label><textarea id="new-skill-instructions" className="skill-instructions-input" value={instructions} onChange={(event) => setInstructions(event.target.value)} maxLength={SKILL_INSTRUCTIONS_MAX_LENGTH} placeholder={SKILL_INSTRUCTIONS_TEMPLATE} required rows={18} /><small className="skill-character-count">{instructions.length}/{SKILL_INSTRUCTIONS_MAX_LENGTH} characters</small></section><div className="dialog-actions"><button type="submit" disabled={busy || !name.trim() || !description.trim() || !instructions.trim()}>Create skill</button></div></form></article> : selected ? <article className="skill-editor"><header><div><p className="eyebrow">Skill</p><div className="skill-title-row"><h1>{selected.title}</h1><button type="button" className="skill-open-folder" aria-label={`Open ${selected.title} folder in Finder`} title="Open skill folder in Finder" onClick={() => void openSkillFolder(selected)}><FolderOpen size={17} /></button></div><p>{selected.description}</p></div>{selected.editable && <div className="dialog-actions"><button disabled={busy} onClick={() => void importPackage(true)}>Update from folder…</button><button disabled={busy} className="secondary" onClick={() => void uninstall()}>Uninstall</button></div>}</header><SkillAppearanceSettings user={user} skill={selected} beforeSave={() => filePath !== "SKILL.md" || !fileDirty || window.confirm("Save the outfit and discard unsaved SKILL.md file edits?")} onSaved={async () => { await load(); if (filePath === "SKILL.md") await selectFileForSkill(selected); }} /><section><h2>Skill details</h2><label>Description<input disabled={!selected.editable} value={description} onChange={(event) => setDescription(event.target.value)} maxLength={SKILL_DESCRIPTION_MAX_LENGTH} /></label><button onClick={() => void saveDescription()} disabled={busy || !selected.editable}>Save details</button></section><section><h2>Definition</h2><textarea disabled={!selected.editable} value={instructions} onChange={(event) => setInstructions(event.target.value)} rows={12} /><button onClick={() => void saveDefinition()} disabled={busy || !selected.editable}>Save definition</button></section><section><div className="skill-files-heading"><h2>Files</h2>{selected.editable && <button type="button" className="secondary" disabled={busy} onClick={() => void addFile()}>+ Add file</button>}</div><div className="skill-file-tabs">{selected.files.map((file) => <button type="button" className={filePath === file.path ? "active" : ""} key={file.path} onClick={() => void selectFile(file.path)}>{file.path}</button>)}</div><input aria-label="File path" disabled={!selected.editable} value={filePath} onChange={(event) => setFilePath(event.target.value)} /><textarea aria-label="File contents" disabled={!selected.editable} value={fileContent} onChange={(event) => setFileContent(event.target.value)} rows={12} /><div className="dialog-actions">{selected.editable && <button onClick={() => void saveFile()}>Save file</button>}{selected.editable && filePath !== "SKILL.md" && <button className="secondary" onClick={() => void deleteFile()}>Delete file</button>}</div></section><section><h2>Artifacts</h2><p>Artifacts are part of this skill's operating knowledge and are not independently curated as tools.</p>{selected.artifacts.length ? selected.artifacts.map((artifact) => <article className="artifact-card" key={artifact.artifact_id}><strong>{artifact.name}</strong><p>{artifact.description}</p><ArtifactLocationEditor artifact={artifact} user={user} onSaved={() => void load(selected.id)} /><button type="button" className="secondary" onClick={() => void deleteArtifact(artifact)}>Delete artifact</button></article>) : <p>No artifacts are registered to this skill.</p>}{unassigned.length > 0 && <div className="unassigned-artifacts"><h3>Unassigned catalog artifacts</h3><p>These existing records need a skill and operating instructions.</p>{unassigned.map((artifact) => <article className="artifact-card" key={artifact.artifact_id}><strong>{artifact.name}</strong><p>{artifact.description}</p><small>{artifact.project_id} · {artifact.entrypoint_path}</small>{selected.editable && <><textarea aria-label={`Instructions for ${artifact.name}`} placeholder="How should this skill use the artifact?" value={artifactInstructions} onChange={(event) => setArtifactInstructions(event.target.value)} /><button disabled={!artifactInstructions.trim()} onClick={() => void attachArtifact(artifact)}>Add to this skill</button></>}<button type="button" className="secondary" onClick={() => void deleteArtifact(artifact)}>Delete artifact</button></article>)}</div>}</section></article> : <div className="projects-workspace-empty"><Sparkles size={36} /><h1>Choose a skill</h1><p>Select a skill to inspect its definition, package files, and associated artifacts.</p><SkillAppearanceSettings user={user} skill={null} onSaved={() => void load()} /></div>}</section>
  </div>;
  async function selectFileForSkill(item: SkillWorkspaceRecord) { const request = ++fileRequest.current; setFilePath("SKILL.md"); setFileContent(""); setSavedFile({ path: "SKILL.md", content: "" }); try { const result = await daemonRequest<{ content: string }>("read_skill_file", { actor_user_id: user, skill_id: item.id, path: "SKILL.md" }); if (request !== fileRequest.current) return; setFilePath("SKILL.md"); setFileContent(result.content); setSavedFile({ path: "SKILL.md", content: result.content }); } catch { /* selection still opens the skill definition */ } }
}

function ProjectEditorPanel({ user, project, onBack, onUpdated, onClose }: { user: string; project: ManagedProject; onBack: () => void; onUpdated: () => void; onClose: () => void }) {
  const [name, setName] = useState(project.name); const [description, setDescription] = useState(project.description); const [visibility, setVisibility] = useState(project.visibility); const [status, setStatus] = useState(project.status); const [members, setMembers] = useState<ManagedUser[]>([]); const [memberIds, setMemberIds] = useState<string[]>([]); const [config, setConfig] = useState<ProjectConfig>({ scope: "", goals: [], duration: { ongoing: true, start_date: null, end_date: null }, directory_structure: [] }); const [files, setFiles] = useState<Array<{ name: string; kind: string }>>([]); const [confirmation, setConfirmation] = useState(""); const [notice, setNotice] = useState("");
  const loadFiles = async () => { const result = await daemonRequest<{ files: Array<{ name: string; kind: string }> }>("project_files", { user, project_id: project.project_id }); setFiles(result.files); };
  useEffect(() => { void Promise.all([daemonRequest<{ config: Record<string, unknown> }>("project_config", { user, project_id: project.project_id }), daemonRequest<{ users: ManagedUser[] }>("users"), daemonRequest<{ members: string[] }>("project_members", { project_id: project.project_id }), daemonRequest<{ files: Array<{ name: string; kind: string }> }>("project_files", { user, project_id: project.project_id })]).then(([cfg, allUsers, selected, projectFiles]) => { const duration = cfg.config.duration && typeof cfg.config.duration === "object" ? cfg.config.duration as Record<string, unknown> : {}; setConfig({ scope: String(cfg.config.scope || ""), goals: Array.isArray(cfg.config.goals) ? cfg.config.goals.map(String) : [], duration: { ongoing: duration.ongoing !== false, start_date: typeof duration.start_date === "string" ? duration.start_date : null, end_date: typeof duration.end_date === "string" ? duration.end_date : null }, directory_structure: Array.isArray(cfg.config.directory_structure) ? cfg.config.directory_structure.map(String) : [] }); setMembers(allUsers.users); setMemberIds(selected.members); setFiles(projectFiles.files); }).catch((cause: unknown) => setNotice(cause instanceof Error ? cause.message : "Project settings could not be loaded.")); }, [project.project_id, user]);
  const addFiles = async () => { try { const selected = await openFileDialog({ title: "Add files to this project", multiple: true, directory: false }); const paths = selected ? (Array.isArray(selected) ? selected : [selected]) : []; if (!paths.length) return; await daemonRequest("copy_desktop_project_files", { user, project_id: project.project_id, source_paths: paths }); await loadFiles(); setNotice("Project files added."); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Project files could not be added."); } };
  const removeFile = async (fileName: string) => { if (!window.confirm(`Remove ${fileName} from this project folder?`)) return; try { await daemonRequest("remove_project_file", { user, project_id: project.project_id, name: fileName }); await loadFiles(); setNotice(`${fileName} removed.`); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Project file could not be removed."); } };
  const save = async () => { try { await daemonRequest("update_project", { user, project_id: project.project_id, name, description, visibility, member_user_ids: visibility === "shared" ? memberIds : [] }); await daemonRequest("save_project_config", { user, project_id: project.project_id, config }); setNotice("Project settings saved."); onUpdated(); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Project settings could not be saved."); } };
  const lifecycle = async (nextStatus: "active" | "paused" | "completed" | "archived") => { try { await daemonRequest("set_project_status", { user, project_id: project.project_id, status: nextStatus }); setStatus(nextStatus); setNotice(`Project marked ${nextStatus}.`); onUpdated(); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Project status could not be changed."); } };
  const remove = async () => { try { await daemonRequest("delete_project", { user, project_id: project.project_id, confirmation }); setNotice("Project removed from Alphonse."); onBack(); } catch (cause) { setNotice(cause instanceof Error ? cause.message : "Project could not be removed."); } };
  const archived = status === "archived";
  return <div className="project-editor">
    <header className="project-editor-header"><div><p className="eyebrow">Project</p><h1>{project.name}</h1></div><button className="secondary" onClick={onClose}>Back to conversation</button></header>
    <section className="project-detail"><div className="form-field"><label htmlFor="project-name">Name</label><input id="project-name" disabled={archived} value={name} onChange={(event) => setName(event.target.value)} /></div><div className="form-field"><label htmlFor="project-description">Description</label><input id="project-description" disabled={archived} value={description} onChange={(event) => setDescription(event.target.value)} /></div><div className="form-field"><label htmlFor="project-visibility">Visibility</label><select id="project-visibility" disabled={archived} value={visibility} onChange={(event) => setVisibility(event.target.value as "private" | "shared")}><option value="private">Private</option><option value="shared">Selected family members</option></select></div>{visibility === "shared" && <fieldset className="form-field"><legend>Shared with</legend>{members.map((member) => member.is_active && member.user_id !== project.owner_user_id && <label key={member.user_id}><input type="checkbox" checked={memberIds.includes(member.user_id)} onChange={(event) => setMemberIds((current) => event.target.checked ? [...current, member.user_id] : current.filter((id) => id !== member.user_id))} /> {member.display_name}</label>)}</fieldset>}<div className="form-field task-readonly"><label>Owner user</label><output>👤 {project.owner?.display_name || project.owner_user_id}</output><small>{project.owner_user_id}</small></div><div className="form-field task-readonly"><label>Project directory</label><output>{project.root_path}</output><button type="button" className="secondary" onClick={() => void showInFinder(project.root_path).catch((cause: unknown) => setNotice(cause instanceof Error ? cause.message : "Project folder could not be opened."))}>Open folder in Finder</button><small>Legacy project_context.md is preserved and excluded from prompts.</small></div><div className="form-field"><label htmlFor="project-scope">Scope</label><textarea id="project-scope" disabled={archived} rows={4} value={config.scope} onChange={(event) => setConfig((current) => ({ ...current, scope: event.target.value }))} /></div><div className="form-field"><label htmlFor="project-goals">Goals (one per line)</label><textarea id="project-goals" disabled={archived} rows={5} value={config.goals.join("\n")} onChange={(event) => setConfig((current) => ({ ...current, goals: event.target.value.split("\n").map((item) => item.trim()).filter(Boolean) }))} /></div><fieldset className="form-field"><legend>Duration</legend><label><input type="checkbox" checked={config.duration.ongoing} disabled={archived} onChange={(event) => setConfig((current) => ({ ...current, duration: { ...current.duration, ongoing: event.target.checked } }))} /> Ongoing</label><label htmlFor="project-start-date">Start date</label><input id="project-start-date" type="date" disabled={archived} value={config.duration.start_date || ""} onChange={(event) => setConfig((current) => ({ ...current, duration: { ...current.duration, start_date: event.target.value || null } }))} /><label htmlFor="project-end-date">End date</label><input id="project-end-date" type="date" disabled={archived} value={config.duration.end_date || ""} onChange={(event) => setConfig((current) => ({ ...current, duration: { ...current.duration, end_date: event.target.value || null } }))} /></fieldset><div className="form-field"><label htmlFor="project-structure">Directory structure (one relative path per line)</label><textarea id="project-structure" disabled={archived} rows={5} value={config.directory_structure.join("\n")} onChange={(event) => setConfig((current) => ({ ...current, directory_structure: event.target.value.split("\n").map((item) => item.trim()).filter(Boolean) }))} /></div><div className="form-field"><label htmlFor="project-status">Lifecycle</label><select id="project-status" value={status} onChange={(event) => void lifecycle(event.target.value as "active" | "paused" | "completed" | "archived")}><option value="active">Active</option><option value="paused">Paused</option><option value="completed">Completed</option><option value="archived">Archived</option></select></div><div className="dialog-actions"><button onClick={() => void save()}>Save changes</button></div><div className="form-field destructive-action"><label htmlFor="project-delete-confirmation">Type the project ID to remove it</label><input id="project-delete-confirmation" value={confirmation} placeholder={project.project_id} onChange={(event) => setConfirmation(event.target.value)} /><button disabled={confirmation !== project.project_id} onClick={() => void remove()}>Remove from Alphonse</button></div></section>
    <section className="project-detail"><h3>Project files</h3><button type="button" disabled={archived} onClick={() => void addFiles()}>Add files</button>{files.length ? <ul>{files.map((file) => <li key={file.name}>{file.name} <button type="button" className="secondary" disabled={archived} onClick={() => void removeFile(file.name)}>Remove</button></li>)}</ul> : <p>No project files yet.</p>}</section>
    <p>{notice}</p>
  </div>;
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
  const eventAutomations = catalog?.automations.filter((item) => item.trigger_kind === "event") || [];
  return <section className="settings-panel">
    <h3>Automations</h3><p>Create an event automation in three steps: first, run a local worker that detects a fact—such as a doorbell press, geofence arrival, or low-temperature threshold—and register the event types it may publish. Next, register each event type with a version and JSON Schema describing its payload. Finally, create an event automation with a fixed prompt and optional exact payload filters. When the worker publishes a valid matching event, Alphonse queues that prompt with the event details attached. Scheduled tasks use the same catalog, but are triggered by time rather than an external event.</p>
    <h4>Workers</h4><div className="form-field"><label>Worker ID</label><input value={workerId} onChange={(event) => setWorkerId(event.target.value)} placeholder="plant-monitor" /></div><div className="form-field"><label>Display name</label><input value={workerName} onChange={(event) => setWorkerName(event.target.value)} /></div><div className="form-field"><label>Allowed event types</label><input value={workerTypes} onChange={(event) => setWorkerTypes(event.target.value)} placeholder="plant.soil_humidity, doorbell.rang" /></div><button onClick={() => void saveWorker()}>Register worker</button>
    <h4>Event types</h4><div className="form-field"><label>Event type</label><input value={eventType} onChange={(event) => setEventType(event.target.value)} placeholder="plant.soil_humidity" /></div><div className="form-field"><label>Version</label><input value={eventVersion} onChange={(event) => setEventVersion(event.target.value)} /></div><div className="form-field"><label>JSON Schema</label><textarea value={schema} rows={5} onChange={(event) => setSchema(event.target.value)} /></div><div className="form-field"><label>History limit</label><input type="number" value={maxHistory} onChange={(event) => setMaxHistory(event.target.value)} /></div><button onClick={() => void saveEventType()}>Register event type</button>
    <h4>Event automation</h4><div className="form-field"><label>Name</label><input value={automationName} onChange={(event) => setAutomationName(event.target.value)} /></div><div className="form-field"><label>Fixed prompt</label><textarea value={automationPrompt} rows={3} onChange={(event) => setAutomationPrompt(event.target.value)} /></div><div className="form-field"><label>Event type</label><input value={automationType} onChange={(event) => setAutomationType(event.target.value)} placeholder="plant.soil_humidity" /></div><div className="form-field"><label>Version</label><input value={automationVersion} onChange={(event) => setAutomationVersion(event.target.value)} /></div><div className="form-field"><label>Exact payload filters (JSON object)</label><textarea value={filters} rows={3} onChange={(event) => setFilters(event.target.value)} /></div><button onClick={() => void saveAutomation()}>Create event automation</button>
    <h4>Registered event automations ({eventAutomations.length})</h4>{catalog && <><div className="automation-table-wrap"><table className="automation-table"><thead><tr><th scope="col">Name</th><th scope="col">Trigger event</th><th scope="col">Filters</th><th scope="col">Status</th></tr></thead><tbody>{eventAutomations.map((item) => <tr key={item.automation_id}><td>{item.name}</td><td><code>{item.trigger.event_type}@{item.trigger.event_version}</code></td><td>{Object.keys(item.trigger.filters || {}).length > 0 ? "Exact payload filters" : "None"}</td><td>{item.status}</td></tr>)}{eventAutomations.length === 0 && <tr><td colSpan={4}>No event automations registered.</td></tr>}</tbody></table></div><h4>Recent events</h4>{catalog.events.map((item) => <p key={`${item.event_type}:${item.source_event_id}`}>{item.event_type}@{item.event_version} · {item.source_event_id} · {item.dispatch_count} automation(s)</p>)}</>}<p>{notice}</p>
  </section>;
}

function ModelSettingsSection() {
  const [providers, setProviders] = useState<Provider[]>([]); const [settings, setSettings] = useState<InferenceSettings | null>(null); const [provider, setProvider] = useState(""); const [model, setModel] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => { void Promise.all([daemonRequest<{ providers: Provider[] }>("inference_providers"), daemonRequest<{ settings: InferenceSettings }>("inference_settings")]).then(([catalog, current]) => { const selectedProvider = catalog.providers.find((item) => item.provider_key === current.settings.provider_key) || catalog.providers[0]; setProviders(catalog.providers); setSettings(current.settings); setProvider(selectedProvider?.provider_key || ""); setModel(current.settings.model_id || selectedProvider?.models[0]?.model_id || ""); }); }, []);
  const selected = providers.find((item) => item.provider_key === provider);
  const savedStatus = settings?.validation_error?.startsWith("openai_codex_model_access_rejected") || settings?.validation_error?.startsWith("openai_codex_model_unavailable")
    ? "Codex temporarily rejected this model. Alphonse will keep the last successful validation and retry new tasks."
    : settings?.validation_error || (settings?.validated_at ? `Validated ${new Date(settings.validated_at).toLocaleString()}.` : (!settings?.model_id ? "Choose and save a model before starting a task." : "Validation required."));
  return <section className="settings-panel"><h3>Agent model</h3><p>Models in this list are advertised by the Codex catalog. Validate &amp; save makes a live request to verify current access before Alphonse uses the selection for new tasks.</p><select value={provider} onChange={(event) => { const nextProvider = providers.find((item) => item.provider_key === event.target.value); setProvider(event.target.value); setModel(nextProvider?.models[0]?.model_id || ""); }}>{providers.map((item) => <option value={item.provider_key} key={item.provider_key}>{item.display_name}</option>)}</select><select value={model} onChange={(event) => setModel(event.target.value)} disabled={!selected?.models.length}>{selected?.models.map((item) => <option value={item.model_id} key={item.model_id}>{item.display_name}</option>)}</select><button disabled={!provider || !model} onClick={() => void daemonRequest<{ settings: InferenceSettings }>("set_inference_settings", { provider_key: provider, model_id: model }).then((result) => { setSettings(result.settings); setNotice("Validated and saved for new tasks."); }).catch((cause: unknown) => setNotice(inferenceValidationNotice(cause)))}>Validate &amp; save</button><p>{notice || savedStatus}</p>{selected?.catalog_cli_matches_runtime === false && <p>Catalog source {selected.catalog_cli_version || "unknown"}; runtime {selected.cli_version || "unknown"} at <code>{selected.cli_path || "codex"}</code>. Update or configure the runtime CLI so both use the same Codex version.</p>}</section>;
}

function SystemOneSettingsSection({ user }: { user: string }) {
  const [settings, setSettings] = useState<SystemOneSettings | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [notice, setNotice] = useState("");
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    void daemonRequest<{ settings: SystemOneSettings }>("system_one_settings", { actor_user_id: user })
      .then((result) => setSettings(result.settings))
      .catch((cause: unknown) => setNotice(errorNotice(cause, "System One settings unavailable")));
  }, [user]);
  const save = async () => {
    if (!settings) return;
    setSaving(true);
    setNotice(settings.enabled ? "Validating Jev connection…" : "Saving…");
    try {
      const result = await daemonRequest<{ settings: SystemOneSettings }>("save_system_one_settings", {
        actor_user_id: user,
        values: { ...settings, api_key: apiKey },
      });
      setSettings(result.settings);
      setApiKey("");
      setNotice(result.settings.enabled ? "Validated and enabled for newly started V3 tasks." : "Saved. System One is disabled.");
    } catch (cause) {
      setNotice(errorNotice(cause, "System One validation failed"));
    } finally {
      setSaving(false);
    }
  };
  const status = settings?.validation_error || (settings?.validated_at
    ? `Last validated ${new Date(settings.validated_at).toLocaleString()}.`
    : "Not validated. Alphonse will continue using the existing System Two review until validation succeeds.");
  return <section className="settings-panel">
    <h3>System One</h3>
    <p>Use TypeSafe.ai Jev for V3 tool curation, task admission, Check acceptance-evidence decisions, and Act routing. Each decision has an independent calibration threshold. Deterministic safety gates remain authoritative; unavailable or unresolved evidence reviews fall back to the agent model.</p>
    {settings && <>
      <div className="form-field checkbox-field"><label htmlFor="system-one-enabled"><input id="system-one-enabled" type="checkbox" checked={settings.enabled} onChange={(event) => setSettings({ ...settings, enabled: event.target.checked })} /> Enable System One for V3 admission, Do, Check, and Act</label></div>
      <div className="form-field"><label htmlFor="system-one-url">API URL</label><input id="system-one-url" value={settings.api_url} onChange={(event) => setSettings({ ...settings, api_url: event.target.value })} /></div>
      <div className="form-field"><label htmlFor="system-one-model">Model</label><input id="system-one-model" value={settings.model} onChange={(event) => setSettings({ ...settings, model: event.target.value })} /></div>
      <div className="form-field"><label htmlFor="system-one-key">API key</label><input id="system-one-key" type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder={settings.has_api_key ? "Leave blank to keep the saved key" : "Enter TypeSafe.ai API key"} autoComplete="new-password" /></div>
      <details><summary>Decision thresholds</summary>
        <div className="form-field"><label htmlFor="system-one-tools">Tool selection minimum threshold</label><input id="system-one-tools" type="range" min="0" max="1" step="0.01" value={settings.tool_selection_threshold} onChange={(event) => setSettings({ ...settings, tool_selection_threshold: Number(event.target.value) })} /><output htmlFor="system-one-tools">{settings.tool_selection_threshold.toFixed(2)}</output><small>Select tools at or above this value; reject tools below it.</small></div>
        <div className="form-field"><label htmlFor="system-one-admission">Task admission threshold</label><input id="system-one-admission" type="range" min="0" max="1" step="0.01" value={settings.task_admission_threshold} onChange={(event) => setSettings({ ...settings, task_admission_threshold: Number(event.target.value) })} /><output htmlFor="system-one-admission">{settings.task_admission_threshold.toFixed(2)}</output><small>Scores at or above this value start a task; scores below it receive a direct conversational reply.</small></div>
        <div className="form-field"><label htmlFor="system-one-check">Check completion threshold</label><input id="system-one-check" type="range" min={Math.min(1, Math.round((settings.no_threshold + 0.01) * 100) / 100)} max="1" step="0.01" value={settings.check_completion_threshold} onChange={(event) => setSettings({ ...settings, check_completion_threshold: Number(event.target.value) })} /><output htmlFor="system-one-check">{settings.check_completion_threshold.toFixed(2)}</output><small>Evidence at or above this value can satisfy an acceptance criterion. Must be higher than the evidence no threshold.</small></div>
        <div className="form-field"><label htmlFor="system-one-yes">Tactical evidence yes threshold</label><input id="system-one-yes" type="number" min="0" max="1" step="0.01" value={settings.yes_threshold} onChange={(event) => setSettings({ ...settings, yes_threshold: Number(event.target.value) })} /></div>
        <div className="form-field"><label htmlFor="system-one-no">Evidence no threshold</label><input id="system-one-no" type="number" min="0" max="1" step="0.01" value={settings.no_threshold} onChange={(event) => setSettings({ ...settings, no_threshold: Number(event.target.value) })} /></div>
        <div className="form-field"><label htmlFor="system-one-route">Act route confidence</label><input id="system-one-route" type="range" min="0" max="1" step="0.01" value={settings.act_route_confidence_threshold} onChange={(event) => setSettings({ ...settings, act_route_confidence_threshold: Number(event.target.value) })} /><output htmlFor="system-one-route">{settings.act_route_confidence_threshold.toFixed(2)}</output></div>
      </details>
      <p><small>When enabled, bounded phase objectives, acceptance criteria, and verified evidence summaries are sent to the configured TypeSafe.ai endpoint. The API key stays in local settings and is never returned to the desktop.</small></p>
      <button disabled={saving || (settings.enabled && !settings.has_api_key && !apiKey)} onClick={() => void save()}>{settings.enabled ? "Validate & save" : "Save disabled settings"}</button>
      <p role="status">{notice || status}</p>
    </>}
    {!settings && notice && <p role="status">{notice}</p>}
  </section>;
}

function inferenceValidationNotice(cause: unknown): string {
  const message = errorNotice(cause, "Validation failed");
  return message.startsWith("openai_codex_model_access_rejected") || message.startsWith("openai_codex_model_unavailable")
    ? "Codex temporarily rejected this advertised model. Your saved model and its last successful validation were not changed."
    : message;
}

function errorNotice(cause: unknown, fallback: string): string {
  if (cause instanceof Error && cause.message.trim()) return cause.message;
  if (typeof cause === "string" && cause.trim()) return cause.trim();
  if (cause && typeof cause === "object" && "message" in cause) {
    const message = (cause as { message?: unknown }).message;
    if (typeof message === "string" && message.trim()) return message.trim();
  }
  return fallback;
}

function AgentConfigSettingsSection() {
  const [documents, setDocuments] = useState<AgentDocument[]>([]); const [fileName, setFileName] = useState(""); const [content, setContent] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => { void daemonRequest<{ documents: AgentDocument[] }>("agent_config_documents").then((result) => { setDocuments(result.documents); setFileName(result.documents[0]?.file_name || ""); }).catch((cause: unknown) => setNotice(errorNotice(cause, "Configuration unavailable"))); }, []);
  const drafts = useRef<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!fileName) return;
    if (fileName in drafts.current) { setLoading(false); setContent(drafts.current[fileName]); return; }
    let active = true;
    setLoading(true); setContent("");
    void daemonRequest<{ document: AgentDocument }>("read_agent_config", { file_name: fileName }).then((result) => {
      if (!active) return;
      drafts.current[fileName] = result.document.content || ""; setContent(drafts.current[fileName]);
    }).catch((cause: unknown) => { if (active) setNotice(errorNotice(cause, "Configuration could not be loaded")); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [fileName]);
  return <section className="settings-panel"><h3>Agent configuration</h3><select aria-label="Configuration document" value={fileName} onChange={(event) => setFileName(event.target.value)}>{documents.map((item) => <option value={item.file_name} key={item.file_name}>{item.display_name}</option>)}</select>{fileName === "GlobalContext.md" && <p className="settings-help">Use the household sections: Family definition; Household location and setup; Household norms; Alphonse's household-wide interaction defaults; Privacy and sharing boundaries; Context maintenance. Keep language, accessibility, communication, and address preferences per member in user_context.md or user settings. This file is editable here.</p>}<textarea aria-label="Configuration contents" disabled={loading || !(fileName in drafts.current)} value={content} onChange={(event) => { drafts.current[fileName] = event.target.value; setContent(event.target.value); }} rows={14} /><button disabled={loading || !(fileName in drafts.current)} onClick={() => void daemonRequest("save_agent_config", { file_name: fileName, content }).then(() => setNotice("Saved. New tasks will read the updated configuration.")).catch((cause: unknown) => setNotice(cause instanceof Error ? cause.message : "Configuration could not be saved."))}>Save configuration</button><p>{notice}</p></section>;
}
