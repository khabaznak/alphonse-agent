import { type ReactNode, useState } from "react";
import { daemonRequest } from "./api";

export const DESKTOP_CATALOG_ID = "alphonse.desktop.catalog.v1";

export type A2uiComponent = {
  id: string;
  component: "Card" | "Container" | "Text" | "Button" | "ChoiceList" | "TextInput" | "Status";
  children?: string[];
  text?: string;
  label?: string;
  action?: { name: string; context: Record<string, unknown> };
  value?: { path: string };
};

export type A2uiSurface = { surfaceId: string; catalogId: string; components: Record<string, A2uiComponent>; dataModel: Record<string, unknown> };
export type AgUiEvent = { type: string; name?: string; value?: unknown };

export function applyA2uiEvent(surfaces: Record<string, A2uiSurface>, event: AgUiEvent): Record<string, A2uiSurface> {
  if (event.type !== "CUSTOM" || event.name !== "a2ui.envelope" || !isRecord(event.value)) return surfaces;
  const envelope = event.value;
  if (envelope.version !== "v0.9.1") return surfaces;
  if (isRecord(envelope.createSurface)) {
    const { surfaceId, catalogId } = envelope.createSurface;
    if (typeof surfaceId !== "string" || catalogId !== DESKTOP_CATALOG_ID) return surfaces;
    return { ...surfaces, [surfaceId]: { surfaceId, catalogId, components: {}, dataModel: {} } };
  }
  if (isRecord(envelope.deleteSurface) && typeof envelope.deleteSurface.surfaceId === "string") {
    const next = { ...surfaces }; delete next[envelope.deleteSurface.surfaceId]; return next;
  }
  if (isRecord(envelope.updateComponents) && typeof envelope.updateComponents.surfaceId === "string" && Array.isArray(envelope.updateComponents.components)) {
    const current = surfaces[envelope.updateComponents.surfaceId];
    if (!current) return surfaces;
    const components = envelope.updateComponents.components.filter(isComponent).reduce<Record<string, A2uiComponent>>((all, item) => ({ ...all, [item.id]: item }), {});
    return { ...surfaces, [current.surfaceId]: { ...current, components } };
  }
  if (isRecord(envelope.updateDataModel) && typeof envelope.updateDataModel.surfaceId === "string" && envelope.updateDataModel.path === "/" && isRecord(envelope.updateDataModel.value)) {
    const current = surfaces[envelope.updateDataModel.surfaceId];
    return current ? { ...surfaces, [current.surfaceId]: { ...current, dataModel: envelope.updateDataModel.value } } : surfaces;
  }
  return surfaces;
}

export function A2uiSurfaceHost({ surfaces, clientId, user, onDone }: { surfaces: Record<string, A2uiSurface>; clientId: string; user: string; onDone: () => Promise<void> }) {
  return <>{Object.values(surfaces).map((surface) => <Surface key={surface.surfaceId} surface={surface} clientId={clientId} user={user} onDone={onDone} />)}</>;
}

function Surface({ surface, clientId, user, onDone }: { surface: A2uiSurface; clientId: string; user: string; onDone: () => Promise<void> }) {
  const [dataModel, setDataModel] = useState(surface.dataModel);
  const component = (id: string): ReactNode => {
    const item = surface.components[id]; if (!item) return null;
    const children = <>{(item.children || []).map(component)}</>;
    if (item.component === "Card") return <section key={id} className="question-card a2ui-card">{children}</section>;
    if (item.component === "Container" || item.component === "ChoiceList") return <div key={id} className="a2ui-container">{children}</div>;
    if (item.component === "Text" || item.component === "Status") return <p key={id} className={item.component === "Status" ? "a2ui-status" : ""}>{item.text}</p>;
    if (item.component === "TextInput") return <input key={id} value={answerText(dataModel)} onChange={(event) => setDataModel({ ...dataModel, answer: { text: event.target.value } })} placeholder={item.label || "Your answer"} />;
    if (item.component === "Button" && item.action) return <button key={id} onClick={() => void act(item)}>{item.label || "Continue"}</button>;
    return null;
  };
  const act = async (item: A2uiComponent) => {
    if (!item.action) return;
    await daemonRequest("a2ui_action", {
      client_id: clientId, user, surface_id: surface.surfaceId, source_component_id: item.id,
      action_name: item.action.name, context: item.action.context, data_model: dataModel,
    });
    await onDone();
  };
  return component("root");
}

function answerText(dataModel: Record<string, unknown>): string {
  return isRecord(dataModel.answer) ? String(dataModel.answer.text || "") : "";
}
function isRecord(value: unknown): value is Record<string, any> { return typeof value === "object" && value !== null && !Array.isArray(value); }
function isComponent(value: unknown): value is A2uiComponent {
  return isRecord(value) && typeof value.id === "string" && ["Card", "Container", "Text", "Button", "ChoiceList", "TextInput", "Status"].includes(String(value.component));
}
