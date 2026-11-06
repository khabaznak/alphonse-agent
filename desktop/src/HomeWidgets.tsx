import { useState } from "react";
import { widgetTime } from "./CoordinationWidgets";

export type Sensor = { id: string; name: string; source: string; unit: string; value: number | null; observed_at: string | null; stale_after_seconds: number; state: "connected" | "stale" | "disconnected" | "error" };
export type EmergencyContact = { id: string; name: string; role: string; phone: string | null; email: string | null };
type Provenance = { artifactId: string; owner: string; source: string; updatedAt: string; expiresAt: number };
export type HomeData = Provenance & ({ family: "sensors"; items: Sensor[] } | { family: "emergency_contacts"; items: EmergencyContact[] });
const text = (value: unknown, limit: number): value is string => typeof value === "string" && value.trim().length > 0 && value.length <= limit;
const time = (value: unknown): value is string => typeof value === "string" && /(?:Z|[+-]\d{2}:\d{2})$/.test(value) && Number.isFinite(Date.parse(value));
const keys = (value: object, fields: string[]) => Object.keys(value).length === fields.length && fields.every((field) => Object.hasOwn(value, field));
export function validHome(data: HomeData | undefined): data is HomeData {
  if (!data || !["sensors", "emergency_contacts"].includes(data.family) || !text(data.artifactId, 256) || !data.artifactId.startsWith("artifact.") || !text(data.owner, 256) || !text(data.source, 300) || !time(data.updatedAt) || Date.parse(data.updatedAt) > Date.now() || !Number.isFinite(data.expiresAt) || data.expiresAt <= Date.parse(data.updatedAt) || data.expiresAt - Date.parse(data.updatedAt) > 86400000 || !Array.isArray(data.items) || data.items.length > 50) return false;
  if (new Set(data.items.map((item) => item?.id)).size !== data.items.length) return false;
  return data.items.every((item) => {
    if (!item || !text(item.id, 128) || !text(item.name, 200)) return false;
    if (data.family === "sensors") {
      const sensor = item as Sensor;
      return keys(sensor, ["id", "name", "source", "unit", "value", "observed_at", "stale_after_seconds", "state"]) && text(sensor.source, 200) && text(sensor.unit, 40) && ["connected", "stale", "disconnected", "error"].includes(sensor.state) && (sensor.value === null || typeof sensor.value === "number" && Number.isFinite(sensor.value)) && Number.isInteger(sensor.stale_after_seconds) && sensor.stale_after_seconds >= 1 && sensor.stale_after_seconds <= 3600 && (sensor.observed_at === null || time(sensor.observed_at) && Date.parse(sensor.observed_at) <= Date.parse(data.updatedAt)) && (sensor.value === null || sensor.observed_at !== null) && (!["connected", "stale"].includes(sensor.state) || sensor.value !== null && sensor.observed_at !== null);
    }
    const contact = item as EmergencyContact;
    return keys(contact, ["id", "name", "role", "phone", "email"]) && text(contact.role, 200) && (contact.phone === null || typeof contact.phone === "string" && /^\+?[0-9][0-9 ()-]{1,39}$/.test(contact.phone)) && (contact.email === null || text(contact.email, 254) && /^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$/.test(contact.email));
  });
}
export function sensorState(sensor: Sensor, expiresAt: number, now = Date.now()) {
  if (sensor.state === "error" || sensor.state === "disconnected") return sensor.state;
  return sensor.state === "stale" || expiresAt <= now || !sensor.observed_at || Date.parse(sensor.observed_at) + sensor.stale_after_seconds * 1000 <= now ? "stale" : "current";
}
export function HomeSetup({ family, onRequest }: { family: HomeData["family"]; onRequest?: (text: string) => void }) {
  return <div><p role="status">Unavailable. {family === "sensors" ? "Connect an authorized home sensor source through its installed skill." : "Configure emergency contacts through an installed skill and an authorized contact source."}</p><p>Ask Alphonse to use the owning skill to surface its source, then pin its card.</p>{onRequest && <button onClick={() => onRequest(`Help configure ${family === "sensors" ? "home sensor readings" : "emergency contacts"} using an installed owning skill and an authorized source. Check provider prerequisites and sharing before changing configuration. Do not invent readings or contact details.`)}>Set up with Alphonse</button>}</div>;
}
export function HomeWidget({ data, timezone, onRequest }: { data: HomeData; timezone?: string; onRequest?: (text: string) => void }) {
  const [notice, setNotice] = useState<{ revision: string; text: string }>();
  const revision = JSON.stringify(data);
  const expired = data.expiresAt <= Date.now();
  if (!validHome(data) || data.family === "emergency_contacts" && expired) return <p role="status">Unavailable. Refresh the source or check access.</p>;
  async function copy(value: string) {
    if (data.expiresAt <= Date.now()) { setNotice({ revision, text: "Contact source expired. Refresh before copying." }); return; }
    try {
      await navigator.clipboard.writeText(value);
      setNotice({ revision, text: "Copied. Choose your phone or mail app to contact them." });
    } catch { setNotice({ revision, text: "Copy unavailable. Select the displayed contact detail to copy manually." }); }
  }
  return <div className="home-widget"><small>{data.source} · Owner: {data.owner}</small><small>Source updated: {widgetTime(data.updatedAt, timezone)}</small>
    {data.family === "sensors" ? <>{expired && <p role="status">Stale source · last-known readings only</p>}{!data.items.length && <p>No sensors configured in this source.</p>}<ul>{data.items.map((sensor) => {
      const state = sensorState(sensor, data.expiresAt);
      return <li key={sensor.id}><strong>{sensor.name}</strong><span className={`sensor-state sensor-${state}`}>{state === "current" ? "Current reading" : state === "stale" ? "Stale · not live" : state === "disconnected" ? "Disconnected · not live" : "Error · not live"}</span><span>{sensor.value === null ? "No reading available" : <>{state !== "current" && "Last known: "}{sensor.value} {sensor.unit}</>}</span><small>Source: {sensor.source}</small><small>Last reading: {sensor.observed_at ? widgetTime(sensor.observed_at, timezone) : "Never received"}</small></li>;
    })}</ul></> : <>{!data.items.length && <p>No emergency contacts configured. Ask Alphonse to configure your chosen contacts.</p>}<ul>{data.items.map((contact) => <li key={contact.id}><strong>{contact.name}</strong><span>{contact.role}</span>{contact.phone && <div><span>{contact.phone}</span><button aria-label={`Copy phone for ${contact.name}`} onClick={() => void copy(contact.phone!)}>Copy phone</button></div>}{contact.email && <div><span>{contact.email}</span><button aria-label={`Copy email for ${contact.name}`} onClick={() => void copy(contact.email!)}>Copy email</button></div>}{!contact.phone && !contact.email && <small>No contact action configured.</small>}</li>)}</ul>{notice?.revision === revision && <p role="status">{notice.text}</p>}<p>Contact actions copy configured details. Use your phone or mail app to contact them.</p>{onRequest && <button onClick={() => onRequest(`Review or update my configured emergency contacts using the owning skill. Resource: ${JSON.stringify(data.artifactId)}; source revision: ${JSON.stringify(data.updatedAt)}. Re-read the source, check authorization and verify any changes before refreshing the widget.`)}>Review / edit with Alphonse</button>}</>}
  </div>;
}
