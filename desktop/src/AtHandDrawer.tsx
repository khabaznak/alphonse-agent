import { HomeSetup, HomeWidget } from "./HomeWidgets";
import { ActivityWidget } from "./ActivityWidgets";
import { CareWidget, CareUnavailable } from "./CareWidget";
import { Agenda } from "./Agenda";
import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { CoordinationList, NativeCoordinationBody, useNativeCoordination, type CoordinationRoutes } from "./CoordinationWidgets";
import { ArrowDown, ArrowUp, LockKeyhole, PanelRightClose, Pin, Plus, UnlockKeyhole, X } from "lucide-react";
import { changeDrawer, reconcileContext, readDrawer, saveDrawer, widgetAvailability, WIDGET_CATALOG, type DrawerAction, type DrawerState } from "./atHandState";

import { authorizedWidgets, suggestedPlacements, type WidgetContext, type SkillWidget } from "./widgetContext";

export function useAtHand(member: string) {
  // An identity-tagged snapshot avoids writing the previous member's layout into a new key.
  const [snapshot, setSnapshot] = useState(() => ({ member, state: readDrawer(window.localStorage, member) }));
  const [saveFailed, setSaveFailed] = useState(false);
  const state = snapshot.member === member ? snapshot.state : readDrawer(window.localStorage, member);
  const act = (action: DrawerAction) => {
    if (!member) return;
    const next = changeDrawer(state, action);
    setSnapshot({ member, state: next });
    setSaveFailed(!saveDrawer(window.localStorage, member, next));
  };
  const memberRef = useRef(member);
  memberRef.current = member;
  const reconcile = useCallback((context: WidgetContext | undefined) => {
    const currentMember = memberRef.current;
    if (!context || context.member !== currentMember) return;
    const candidates = suggestedPlacements(authorizedWidgets(context, currentMember));
    setSnapshot((previous) => {
      const current = previous.member === currentMember ? previous.state : readDrawer(window.localStorage, currentMember);
      const next = reconcileContext(current, candidates, Date.now());
      if (next === current && previous.member === currentMember) return previous;
      setSaveFailed(!saveDrawer(window.localStorage, currentMember, next));
      return { member: currentMember, state: next };
    });
  }, []);
  return { state, act, saveFailed, reconcile };
}

export function AtHandDrawer({ state, act, member, connected, questionCount, questionsMember, onAttention, onAgenda, toggleRef, saveFailed, widgetContext, onWork, onRequest, onItem, timezone, onSchedule, onProjects }: CoordinationRoutes & {
  onSchedule?: (id: string) => void; onProjects?: () => void;
  state: DrawerState; act: (action: DrawerAction) => void; member: string; connected: boolean;
  questionCount: number; questionsMember: string; onAttention: () => void;
  toggleRef: RefObject<HTMLButtonElement | null>; saveFailed: boolean; widgetContext?: WidgetContext;
}) {
  const rows = authorizedWidgets(widgetContext, member);
  const native = useNativeCoordination(member, connected && state.open && state.widgets.some((widget) => ["todos", "reminders"].includes(widget.kind)));
  const routes = { onWork, onRequest, onItem, onAgenda, timezone };
  const closeRef = useRef<HTMLButtonElement>(null);
  const catalogRef = useRef<HTMLDetailsElement>(null);
  const openCatalog = () => { if (catalogRef.current) { catalogRef.current.open = true; catalogRef.current.scrollIntoView({ block: "nearest" }); catalogRef.current.querySelector("summary")?.focus(); } };
  const [now, setNow] = useState(Date.now);
  useEffect(() => { if (state.open) closeRef.current?.focus(); }, [state.open]);
  useEffect(() => { const timer = window.setInterval(() => setNow(Date.now()), 1000); return () => window.clearInterval(timer); }, []);
  const close = () => { act({ type: "visibility", open: false }); toggleRef.current?.focus(); };
  return <aside id="at-hand" className="at-hand-drawer" aria-label="At hand" hidden={!state.open} onKeyDown={(event) => { if (event.key === "Escape") { event.stopPropagation(); close(); } }}>
    <header className="at-hand-heading"><h2>At hand <span className="widget-count">{state.widgets.length}</span></h2><div><button aria-label="Add widget" title="Add widget" onClick={openCatalog}><Plus /></button><button ref={closeRef} aria-label="Close At hand" title="Close At hand" onClick={close}><PanelRightClose /></button></div></header>
    {saveFailed && <p role="status">Layout could not be saved. Changes last for this session.</p>}
    <div className="at-hand-cards" aria-label="Widgets">
      <p className="drawer-context">Pinned for you · Available across workspaces</p>
      {!state.widgets.length && <p>No widgets. Use Add widget to choose one.</p>}
      {state.widgets.map((widget, index) => {
        const data = rows.find((row) => row.id === widget.id);
        const title = data?.title || WIDGET_CATALOG.find((entry) => entry.kind === widget.kind)?.title || "Unavailable widget";
        const availability = widget.kind === "skill" && (!data || data.status !== "ready") ? "unavailable" : widgetAvailability(widget, member, connected, now);
        const locked = widget.mode === "locked";
        const agendaBody = <Agenda key={member} compact active={state.open} member={member} timezone={timezone || "UTC"} onSchedule={onSchedule || ((id) => onAgenda(id))} onProjects={onProjects || (() => onWork?.(""))} onOpen={() => onAgenda()} />;
        return <article className="card at-hand-card" key={widget.id} aria-label={title}>
          <div className="widget-heading"><h3>{title}</h3><div className="widget-controls" role="group" aria-label={`${title} controls`}>
            <button aria-label={`${locked ? "Unlock" : "Lock"} ${title}`} title={locked ? "User locked · Unlock" : "Lock in place"} aria-pressed={locked} onClick={() => act({ type: "mode", id: widget.id, mode: locked ? "pinned" : "locked" })}>{locked ? <LockKeyhole /> : <UnlockKeyhole />}</button>
            {!locked && <button aria-label={`${widget.mode === "pinned" ? "Unpin" : "Pin"} ${title}`} title={widget.mode === "pinned" ? "Pinned · Unpin" : "Keep pinned"} aria-pressed={widget.mode === "pinned"} onClick={() => act({ type: "mode", id: widget.id, mode: widget.mode === "pinned" ? "contextual" : "pinned", expiresAt: Date.now() + 30 * 60 * 1000 })}><Pin /></button>}
            <button aria-label={`Move ${title} up`} title="Move up" disabled={locked || index === 0 || state.widgets[index - 1]?.mode === "locked"} onClick={() => act({ type: "move", id: widget.id, direction: -1 })}><ArrowUp /></button>
            <button aria-label={`Move ${title} down`} title="Move down" disabled={locked || index === state.widgets.length - 1 || state.widgets[index + 1]?.mode === "locked"} onClick={() => act({ type: "move", id: widget.id, direction: 1 })}><ArrowDown /></button>
            <button aria-label={`Remove ${title}`} title={locked ? "Unlock before removing" : "Remove card"} disabled={locked} onClick={() => act({ type: "remove", id: widget.id })}><X /></button>
          </div></div>
          <span className="widget-mode">{locked ? "User locked" : widget.mode === "pinned" ? "Pinned" : "Temporary context"}</span>
          <div className="widget-body" tabIndex={0} aria-label={`${title} content`}>
            {availability === "loading" ? <p role="status">Loading member…</p> : availability === "expired" ? <p role="status">Context expired. Remove this card or pin it to keep it.</p> : availability === "unavailable" ? widget.kind === "care" || data?.binding === "care" ? <CareUnavailable /> : <p role="status">Unavailable. Reconnect or check access in the workspace.</p> : widget.kind === "care" ? <>{rows.filter((row) => row.status === "ready" && row.binding === "care").map((row) => <CareWidget key={row.id} data={row.care} member={member} timezone={timezone} onRequest={onRequest} />)}{!rows.some((row) => row.status === "ready" && row.binding === "care") && <CareUnavailable />}</> : widget.kind === "skill" ? data!.binding === "agenda" ? agendaBody : <SkillWidgetBody now={now} data={data!} {...routes} /> : ["sensors", "emergency_contacts"].includes(widget.kind) ? <>{rows.filter((row) => row.status === "ready" && row.binding === "home" && row.home?.family === widget.kind).map((row) => <HomeWidget key={row.id} data={row.home!} timezone={timezone} onRequest={onRequest} />)}{!rows.some((row) => row.status === "ready" && row.home?.family === widget.kind) && <HomeSetup family={widget.kind as "sensors" | "emergency_contacts"} onRequest={onRequest} />}</> : ["todos", "reminders", "shopping", "appointments"].includes(widget.kind) ? <>{rows.filter((row) => row.status === "ready" && row.binding === "coordination" && row.coordination?.family === widget.kind).map((row) => <CoordinationList key={row.id} data={row.coordination!} {...routes} />)}{["todos", "reminders"].includes(widget.kind) ? <NativeCoordinationBody family={widget.kind} controller={native} {...routes} /> : !rows.some((row) => row.status === "ready" && row.coordination?.family === widget.kind) && <p role="status">Unavailable. Install the owning skill and connect an authorized, current list source.</p>}</> : widget.kind === "questions" ? questionsMember !== member ? <p role="status">Loading questions…</p> : <><p>{questionCount ? `${questionCount} waiting for your input` : "No pending questions"}</p><button className="widget-text-link" onClick={onAttention}>Needs your attention <ArrowUp aria-hidden="true" /></button></> : agendaBody}
          </div>
        </article>;
      })}
      <details ref={catalogRef} className="widget-catalog"><summary><Plus aria-hidden="true" /> Add widget</summary><p>Choose a widget to keep at hand.</p>{WIDGET_CATALOG.map((entry) => <div key={entry.kind}><strong>{entry.title}</strong><p>{entry.description}</p><button disabled={!member || state.widgets.some((widget) => widget.id === entry.kind)} onClick={() => act({ type: "add", kind: entry.kind })}>Add {entry.title}</button></div>)}<p className="widget-help">Pins stay until removed. Unpinned cards expire after 30 minutes. Unlock before moving or removing a locked card. Removing a card only changes this drawer.</p></details>
    </div>
  </aside>;
}

function SkillWidgetBody({ data, now, ...routes }: CoordinationRoutes & { data: SkillWidget; now: number }) {
  const { onAgenda } = routes;
  if (data.binding === "home") return <HomeWidget data={data.home!} timezone={routes.timezone} onRequest={routes.onRequest} />;
  if (data.binding === "activity") return <ActivityWidget data={data.activity!} now={now} timezone={routes.timezone} onRequest={routes.onRequest} />;
  if (data.binding === "care") return <CareWidget data={data.care} member={data.member} timezone={routes.timezone} onRequest={routes.onRequest} />;
  if (data.binding === "coordination") return <><CoordinationList data={data.coordination!} {...routes} /><details className="widget-source-details"><summary>Skill context</summary><small>Skill: {data.skillId}{data.contextId && <> · Context: {data.contextId}</>}</small></details></>;
  return <>{data.historical && <small>Historical context · not a current measurement</small>}<p style={{ whiteSpace: "pre-wrap" }}>{data.text}</p>
    {data.binding === "agenda" && <button onClick={() => onAgenda()}>Open Agenda</button>}
    {data.relevant && ["pin", "remove", "expire"].includes(data.suggestion) && <p role="status">Alphonse suggests {data.suggestion === "pin" ? "keeping this card pinned" : data.suggestion === "remove" ? "removing this card" : "letting this context expire"}. Use the card controls to apply.</p>}
  </>;
}
