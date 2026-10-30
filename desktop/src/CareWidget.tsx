import { widgetTime } from "./CoordinationWidgets";
export type CareItem = { id: string; definition: string; dueAt: string | null; adherence: "recorded" | "missed" | "unknown"; recordedAt: string | null; writeState: "none" | "verified" | "failed" | "unconfirmed" };
export type CareData = { member: string; source: string; recordedAt: string; expiresAt: number; items: CareItem[] };
const date = (value: unknown) => typeof value === "string" && /(?:Z|[+-]\d\d:\d\d)$/.test(value) && Number.isFinite(Date.parse(value));
export function validCare(data: CareData | undefined, member: string): data is CareData {
  return !!data && data.member === member && typeof data.source === "string" && data.source.length > 0 && data.source.length <= 200 && date(data.recordedAt) && Number.isFinite(data.expiresAt) && Array.isArray(data.items) && data.items.length <= 50 && new Set(data.items.map((item) => item?.id)).size === data.items.length && data.items.every((item) => item && typeof item.id === "string" && item.id.length > 0 && item.id.length <= 128 && typeof item.definition === "string" && item.definition.length > 0 && item.definition.length <= 500 && ["recorded", "missed", "unknown"].includes(item.adherence) && ["none", "verified", "failed", "unconfirmed"].includes(item.writeState) && (item.dueAt === null || date(item.dueAt)) && (item.recordedAt === null ? item.adherence === "unknown" : date(item.recordedAt)));
}
export function careDraft(data: CareData, item: CareItem, action: string) {
  return `Use the authorized Doctor skill workflow. Requested action: ${action}.\nMember: ${JSON.stringify(data.member)}; source: ${JSON.stringify(data.source)}; source date: ${data.recordedAt}; record ID: ${JSON.stringify(item.id)}.\nRe-read the underlying record, check current access and prerequisites. Do not infer medication, dosage, diagnosis or adherence from missing data. Ask for missing details before logging or correcting. Report success only after verifying the persisted effect; report failed or unconfirmed writes explicitly. Refresh the care projection after verification.`;
}
export function CareUnavailable() {
  return <p role="status">Care records unavailable. The protected Doctor capability must be installed, configured and authorize your access. Missing data does not mean missed care.</p>;
}
export function CareWidget({ data, member, timezone, onRequest }: { data?: CareData; member: string; timezone?: string; onRequest?: (text: string) => void }) {
  if (!validCare(data, member) || data.expiresAt <= Date.now()) return <CareUnavailable />;
  return <div className="coordination-list"><small>Member: {data.member} · Source: {data.source}</small><small>Source recorded: {widgetTime(data.recordedAt, timezone)}</small>
    {!data.items.length && <p>No treatment or routine definitions returned. Adherence is unknown.</p>}
    <ul>{data.items.map((item) => <li key={item.id}><strong>{item.definition}</strong><small>Record: {item.id}</small><small>{item.dueAt ? `Due as recorded: ${widgetTime(item.dueAt, timezone)}` : "Due time unknown"}</small><small>{item.adherence === "unknown" ? "Adherence unknown / not recorded" : item.adherence === "missed" ? "Explicitly recorded as missed" : "Adherence recorded"}{item.recordedAt && ` · ${widgetTime(item.recordedAt, timezone)}`}</small>
      {item.writeState !== "none" && <p role={item.writeState === "verified" ? "status" : "alert"}>{item.writeState === "verified" ? "Source confirms a verified write." : item.writeState === "failed" ? "Write failed. No successful change confirmed." : "Write unconfirmed. Re-read before retrying."}</p>}
      {onRequest && <div className="coordination-actions"><button onClick={() => onRequest(careDraft(data, item, "Open the underlying record or related question"))}>Open record / ask</button><button onClick={() => onRequest(careDraft(data, item, "Log adherence using the details I provide"))}>Request logging</button><button onClick={() => onRequest(careDraft(data, item, "Correct this record using the details I provide"))}>Request correction</button></div>}
    </li>)}</ul><p>Requests open a conversation draft. No record changes until the authorized workflow verifies a write. Follow the recorded care instructions; missing records supply no medical instructions.</p></div>;
}
