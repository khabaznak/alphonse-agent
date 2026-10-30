import { validCare, type CareData } from "./CareWidget";
import { validHome, type HomeData } from "./HomeWidgets";
import { validCoordination, type CoordinationData } from "./CoordinationWidgets";
import type { ContextWidget } from "./atHandState";
export type SkillWidget = ContextWidget & { member: string; skillId: string; contextId: string; scope: "member"; title: string; binding: "agenda" | "memory" | "coordination" | "home" | "care"; care?: CareData; home?: HomeData; coordination?: CoordinationData; status: "ready" | "unavailable"; text: string; historical?: boolean; relevant: boolean; suggestion: "show" | "update" | "pin" | "remove" | "expire" };
export type WidgetContext = { version: number; member: string; widgets: SkillWidget[] };
/** Only fixed native renderers; data never becomes HTML, a URL or an executable action. */
export function authorizedWidgets(context: WidgetContext | undefined, member: string): SkillWidget[] {
  if (!context || context.version !== 1 || context.member !== member || !member || !Array.isArray(context.widgets)) return [];
  const ids = new Set<string>();
  return context.widgets.filter((row) => {
    if (!row || typeof row.id !== "string" || !/^skill-widget:[a-f0-9]{32}$/.test(row.id) || ids.has(row.id) || row.kind !== "skill" || row.member !== member || row.scope !== "member" || typeof row.skillId !== "string" || typeof row.title !== "string" || row.title.length > 100 || typeof row.text !== "string" || row.text.length > 4000 || !["agenda", "memory", "coordination", "home", "care"].includes(row.binding) || !["ready", "unavailable"].includes(row.status) || !["show", "update", "pin", "remove", "expire"].includes(row.suggestion) || !Number.isFinite(row.expiresAt)) return false;
    if (row.binding === "coordination" && row.status === "ready" && !validCoordination(row.coordination)) return false;
    if (row.binding === "home" && row.status === "ready" && !validHome(row.home)) return false;
    if (row.binding === "care" && row.status === "ready" && !validCare(row.care, member)) return false;
    ids.add(row.id); return true;
  });
}
export function suggestedPlacements(rows: SkillWidget[]): SkillWidget[] {
  return rows.filter((row) => row.relevant && row.status === "ready" && ["show", "pin"].includes(row.suggestion));
}
