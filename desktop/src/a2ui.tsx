import { type ReactNode, useEffect, useState } from "react";
import { daemonRequest } from "./api";

export const DESKTOP_CATALOG_ID = "alphonse.desktop.catalog.v2";
const COMPONENTS = ["Card", "Row", "Column", "List", "Text", "Button", "TextInput", "Status", "Divider", "Icon", "CheckBox", "ChoicePicker", "DateTimeInput", "Table"] as const;
const ICONS: Record<string, string> = { check: "✓", info: "ⓘ", warning: "⚠", error: "✕", calendar: "◫", clock: "◷", list: "☷", progress: "◌" };

export type A2uiComponent = {
  id: string; component: typeof COMPONENTS[number]; children?: string[]; text?: string; label?: string;
  action?: { name: string; context: Record<string, unknown> }; value?: { path: string };
  justify?: "start" | "center" | "end" | "spaceBetween"; align?: "start" | "center" | "end" | "stretch";
  direction?: "vertical" | "horizontal"; axis?: "horizontal" | "vertical"; name?: keyof typeof ICONS;
  options?: { label: string; value: string }[]; maxAllowedSelections?: number; enableDate?: boolean; enableTime?: boolean;
  columns?: { id: string; label: string; align?: "left" | "center" | "right" }[]; rows?: { cells: Record<string, string | number | boolean | null> }[];
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
  if (isRecord(envelope.deleteSurface) && typeof envelope.deleteSurface.surfaceId === "string") { const next = { ...surfaces }; delete next[envelope.deleteSurface.surfaceId]; return next; }
  if (isRecord(envelope.updateComponents) && typeof envelope.updateComponents.surfaceId === "string" && Array.isArray(envelope.updateComponents.components)) {
    const current = surfaces[envelope.updateComponents.surfaceId]; if (!current) return surfaces;
    const components = envelope.updateComponents.components.filter(isComponent).reduce<Record<string, A2uiComponent>>((all, item) => ({ ...all, [item.id]: item }), {});
    return { ...surfaces, [current.surfaceId]: { ...current, components } };
  }
  if (isRecord(envelope.updateDataModel) && typeof envelope.updateDataModel.surfaceId === "string" && envelope.updateDataModel.path === "/" && isRecord(envelope.updateDataModel.value)) {
    const current = surfaces[envelope.updateDataModel.surfaceId]; return current ? { ...surfaces, [current.surfaceId]: { ...current, dataModel: envelope.updateDataModel.value } } : surfaces;
  }
  return surfaces;
}

export function A2uiSurfaceHost({ surfaces, clientId, user, onDone, onDismiss }: { surfaces: Record<string, A2uiSurface>; clientId: string; user: string; onDone: () => Promise<void>; onDismiss?: (surfaceId: string) => void }) {
  return <>{Object.values(surfaces).map((surface) => <A2uiSurfaceView key={surface.surfaceId} surface={surface} clientId={clientId} user={user} onDone={onDone} onDismiss={onDismiss} />)}</>;
}

export function A2uiSurfaceView({ surface, clientId, user, onDone, onAction, onDismiss }: { surface: A2uiSurface; clientId: string; user: string; onDone: () => Promise<void>; onAction?: (result: Record<string, unknown>) => void; onDismiss?: (surfaceId: string) => void }) {
  const [dataModel, setDataModel] = useState(surface.dataModel);
  useEffect(() => setDataModel(surface.dataModel), [surface.dataModel, surface.surfaceId]);
  const act = async (item: A2uiComponent) => {
    if (!item.action) return;
    if (item.action.name === "dismiss_surface") { const id = localDismissSurfaceId(surface, item); if (id) onDismiss?.(id); return; }
    const result = await daemonRequest<Record<string, unknown>>("a2ui_action", { client_id: clientId, user, surface_id: surface.surfaceId, source_component_id: item.id, action_name: item.action.name, context: item.action.context, data_model: dataModel });
    onAction?.(result); await onDone();
  };
  return <A2uiComponentTree surface={surface} dataModel={dataModel} setDataModel={setDataModel} onAction={act} />;
}

export function A2uiComponentTree({ surface, dataModel, setDataModel, onAction }: { surface: A2uiSurface; dataModel: Record<string, unknown>; setDataModel: (value: Record<string, unknown>) => void; onAction?: (item: A2uiComponent) => void }) {
  const node = (id: string): ReactNode => {
    const item = surface.components[id]; if (!item) return null;
    const children = <>{(item.children || []).map(node)}</>; const cls = `a2ui-${cssIdentifier(id)}`;
    if (item.component === "Card") return <section key={id} className={`question-card a2ui-card ${cls}${surface.surfaceId.startsWith("scheduled-task:") ? " scheduled-task-card" : ""}`}>{children}</section>;
    if (item.component === "Row" || item.component === "Column" || item.component === "List") return <div key={id} className={`a2ui-${item.component.toLowerCase()} ${cls}`} data-justify={item.justify} data-align={item.align}>{children}</div>;
    if (item.component === "Text" || item.component === "Status") return <p key={id} className={`${cls}${item.component === "Status" ? " a2ui-status" : ""}`}>{item.text}</p>;
    if (item.component === "Divider") return <hr key={id} className={`${cls} a2ui-divider`} aria-orientation={item.axis || "horizontal"} />;
    if (item.component === "Icon") return item.name && ICONS[item.name] ? <span key={id} className={`${cls} a2ui-icon`} role="img" aria-label={item.name}>{ICONS[item.name]}</span> : null;
    if (item.component === "TextInput") return <input key={id} value={String(readPointer(dataModel, item.value?.path) || "")} onChange={(e) => setDataModel(writePointer(dataModel, item.value?.path, e.target.value))} placeholder={item.label || "Your answer"} aria-label={item.label || "Your answer"} />;
    if (item.component === "CheckBox") return <label key={id} className={`${cls} a2ui-checkbox`}><input type="checkbox" checked={Boolean(readPointer(dataModel, item.value?.path))} onChange={(e) => setDataModel(writePointer(dataModel, item.value?.path, e.target.checked))} />{item.label}</label>;
    if (item.component === "DateTimeInput") return <input key={id} className={cls} type="datetime-local" value={String(readPointer(dataModel, item.value?.path) || "")} onChange={(e) => setDataModel(writePointer(dataModel, item.value?.path, e.target.value))} aria-label={item.label || "Date and time"} />;
    if (item.component === "ChoicePicker") { const selected = arrayValue(readPointer(dataModel, item.value?.path)); const multi = (item.maxAllowedSelections || 1) > 1; return <fieldset key={id} className={`${cls} a2ui-choice-picker`}><legend>{item.label}</legend>{(item.options || []).map((option) => <label key={option.value}><input type={multi ? "checkbox" : "radio"} name={id} checked={selected.includes(option.value)} onChange={(e) => setDataModel(writePointer(dataModel, item.value?.path, multi ? (e.target.checked ? [...selected, option.value] : selected.filter((value) => value !== option.value)) : [option.value]))} />{option.label}</label>)}</fieldset>; }
    if (item.component === "Table") return <div key={id} className={`${cls} a2ui-table-wrap`}><table className="a2ui-table"><thead><tr>{(item.columns || []).map((column) => <th key={column.id} scope="col" data-align={column.align}>{column.label}</th>)}</tr></thead><tbody>{(item.rows || []).map((row, index) => <tr key={index}>{(item.columns || []).map((column) => <td key={column.id} data-align={column.align}>{String(row.cells[column.id] ?? "")}</td>)}</tr>)}</tbody></table></div>;
    if (item.component === "Button" && item.action) return <button type="button" key={id} className={`${cls}${["cancel_question", "dismiss_surface"].includes(item.action.name) ? " question-cancel" : ""}`} onClick={() => onAction?.(item)}>{item.label || "Continue"}</button>;
    return null;
  };
  return <>{node("root")}</>;
}

export function readPointer(model: Record<string, unknown>, pointer?: string): unknown { if (!pointer || pointer === "/") return model; return pointer.split("/").slice(1).reduce<unknown>((value, key) => isRecord(value) ? value[key.replace(/~1/g, "/").replace(/~0/g, "~")] : undefined, model); }
export function writePointer(model: Record<string, unknown>, pointer: string | undefined, value: unknown): Record<string, unknown> { const keys = (pointer || "/").split("/").slice(1).map((key) => key.replace(/~1/g, "/").replace(/~0/g, "~")); if (!keys.length) return isRecord(value) ? value : model; const next: Record<string, unknown> = structuredClone(model); let target = next; keys.slice(0, -1).forEach((key) => { target[key] = isRecord(target[key]) ? { ...target[key] } : {}; target = target[key] as Record<string, unknown>; }); target[keys[keys.length - 1]] = value; return next; }
function arrayValue(value: unknown): string[] { return Array.isArray(value) ? value.map(String) : []; }
function isRecord(value: unknown): value is Record<string, any> { return typeof value === "object" && value !== null && !Array.isArray(value); }
function isComponent(value: unknown): value is A2uiComponent { if (!isRecord(value) || typeof value.id !== "string" || !COMPONENTS.includes(value.component as typeof COMPONENTS[number])) return false; if (value.component === "Icon" && (typeof value.name !== "string" || !(value.name in ICONS))) return false; if (value.component === "Table" && (!Array.isArray(value.columns) || !Array.isArray(value.rows))) return false; return true; }
function cssIdentifier(value: string): string { return value.replace(/[^a-zA-Z0-9_-]/g, "-"); }
export function localDismissSurfaceId(surface: A2uiSurface, item: A2uiComponent): string | null { if (!surface.surfaceId.startsWith("scheduled-task:") || item.id !== "dismiss" || item.action?.name !== "dismiss_surface") return null; return item.action.context.surface_id === surface.surfaceId ? surface.surfaceId : null; }
