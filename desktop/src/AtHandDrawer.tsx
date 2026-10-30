import { useEffect, useRef, useState, type RefObject } from "react";
import { PanelRightClose } from "lucide-react";
import { changeDrawer, readDrawer, saveDrawer, widgetAvailability, WIDGET_CATALOG, type DrawerAction, type DrawerState } from "./atHandState";

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
  return { state, act, saveFailed };
}

export function AtHandDrawer({ state, act, member, connected, questionCount, questionsMember, onAttention, onAgenda, toggleRef, saveFailed }: {
  state: DrawerState; act: (action: DrawerAction) => void; member: string; connected: boolean;
  questionCount: number; questionsMember: string; onAttention: () => void; onAgenda: () => void;
  toggleRef: RefObject<HTMLButtonElement | null>; saveFailed: boolean;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const [now, setNow] = useState(Date.now);
  useEffect(() => { if (state.open) closeRef.current?.focus(); }, [state.open]);
  useEffect(() => { const timer = window.setInterval(() => setNow(Date.now()), 1000); return () => window.clearInterval(timer); }, []);
  const close = () => { act({ type: "visibility", open: false }); toggleRef.current?.focus(); };
  return <aside id="at-hand" className="at-hand-drawer" aria-label="At hand" hidden={!state.open} onKeyDown={(event) => { if (event.key === "Escape") { event.stopPropagation(); close(); } }}>
    <header><h2>At hand</h2><button ref={closeRef} aria-label="Close At hand" onClick={close}><PanelRightClose /></button></header>
    {saveFailed && <p role="status">Layout could not be saved. Changes last for this session.</p>}
    <details className="widget-catalog"><summary>Add widget</summary><p>Choose a widget to keep at hand.</p>{WIDGET_CATALOG.map((entry) => <div key={entry.kind}><strong>{entry.title}</strong><p>{entry.description}</p><button disabled={!member || state.widgets.some((widget) => widget.id === entry.kind)} onClick={() => act({ type: "add", kind: entry.kind })}>Add {entry.title}</button></div>)}</details>
    <p className="widget-help">Pins stay until removed. Unpinned cards expire after 30 minutes. Unlock before moving or removing a locked card. Removing a card only changes this drawer.</p>
    <div className="at-hand-cards" aria-label="Widgets">
      {!state.widgets.length && <p>No widgets. Use Add widget to choose one.</p>}
      {state.widgets.map((widget, index) => {
        const title = WIDGET_CATALOG.find((entry) => entry.kind === widget.kind)?.title || "Unavailable widget";
        const availability = widgetAvailability(widget, member, connected, now);
        const locked = widget.mode === "locked";
        return <article className="card at-hand-card" key={widget.id} aria-label={title}>
          <h3>{title}</h3><small className="widget-mode">{locked ? "User locked" : widget.mode === "pinned" ? "Pinned" : "Temporary context"}</small>
          <div className="widget-body" tabIndex={0} aria-label={`${title} content`}>
            {availability === "loading" ? <p role="status">Loading member…</p> : availability === "expired" ? <p role="status">Context expired. Remove this card or pin it to keep it.</p> : availability === "unavailable" ? <p role="status">Unavailable. Reconnect or check access in the workspace.</p> : widget.kind === "questions" ? questionsMember !== member ? <p role="status">Loading questions…</p> : <><p>{questionCount ? `${questionCount} waiting for your input` : "No pending questions"}</p><button onClick={onAttention}>Needs your attention</button></> : <><p>Review scheduled work and its history.</p><button onClick={onAgenda}>Open Agenda</button></>}
          </div>
          <div className="widget-controls" role="group" aria-label={`${title} controls`}>
            <button aria-label={`${locked ? "Unlock" : "Lock"} ${title}`} onClick={() => act({ type: "mode", id: widget.id, mode: locked ? "pinned" : "locked" })}>{locked ? "Unlock" : "Lock"}</button>
            {!locked && <button aria-label={`${widget.mode === "pinned" ? "Unpin" : "Pin"} ${title}`} onClick={() => act({ type: "mode", id: widget.id, mode: widget.mode === "pinned" ? "contextual" : "pinned", expiresAt: Date.now() + 30 * 60 * 1000 })}>{widget.mode === "pinned" ? "Unpin" : "Pin"}</button>}
            <button aria-label={`Move ${title} up`} disabled={locked || index === 0 || state.widgets[index - 1]?.mode === "locked"} onClick={() => act({ type: "move", id: widget.id, direction: -1 })}>↑</button>
            <button aria-label={`Move ${title} down`} disabled={locked || index === state.widgets.length - 1 || state.widgets[index + 1]?.mode === "locked"} onClick={() => act({ type: "move", id: widget.id, direction: 1 })}>↓</button>
            <button aria-label={`Remove ${title}`} disabled={locked} onClick={() => act({ type: "remove", id: widget.id })}>Remove</button>
          </div>
        </article>;
      })}
    </div>
  </aside>;
}
