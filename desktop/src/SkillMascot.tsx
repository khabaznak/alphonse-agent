import { useEffect, useMemo, useRef, useState } from "react";
import { daemonRequest } from "./api";
import { avatarStateLabel, type AvatarState } from "./layoutState";
import { appearanceAllowed, BASE_LATCH, latchAppearance, readAppearancePreferences, saveAppearancePreferences, type AppearanceContext, type AppearancePreferences, type SkillAppearance } from "./skillAppearances";

const PREFERENCE_EVENT = "alphonse:mascot-preferences";
function useAppearancePreferences(user: string) {
  const [preferences, setPreferences] = useState(() => readAppearancePreferences(window.localStorage, user));
  useEffect(() => {
    const refresh = () => setPreferences(readAppearancePreferences(window.localStorage, user));
    refresh();
    window.addEventListener(PREFERENCE_EVENT, refresh);
    window.addEventListener("storage", refresh);
    return () => { window.removeEventListener(PREFERENCE_EVENT, refresh); window.removeEventListener("storage", refresh); };
  }, [user]);
  const save = (value: AppearancePreferences) => {
    saveAppearancePreferences(window.localStorage, user, value);
    setPreferences(value);
    window.dispatchEvent(new Event(PREFERENCE_EVENT));
  };
  return [preferences, save] as const;
}

function useAppearanceResource(user: string, descriptor: SkillAppearance | null) {
  const [loaded, setLoaded] = useState<{ user: string; appearance: SkillAppearance } | null>(null);
  const skill = descriptor?.skill_id;
  const revision = descriptor?.revision;
  useEffect(() => {
    let cancelled = false;
    setLoaded(null);
    if (user && skill) void daemonRequest<{ appearance: SkillAppearance | null }>("skill_appearance", { actor_user_id: user, skill_id: skill })
      .then((result) => { if (!cancelled && result.appearance && result.appearance.revision === revision) setLoaded({ user, appearance: result.appearance }); })
      .catch(() => { /* Optional costume: the base mascot remains available. */ });
    return () => { cancelled = true; };
  }, [user, skill, revision]);
  return loaded?.user === user && loaded.appearance.skill_id === skill && loaded.appearance.revision === revision ? loaded.appearance : null;
}

export function MascotImage({ state, appearance = null, className = "brand-mascot" }: { state: AvatarState; appearance?: SkillAppearance | null; className?: string }) {
  const entry = appearance?.states[state];
  const resource = entry ? appearance?.resources?.[entry.resource] : undefined;
  const [failed, setFailed] = useState("");
  const identity = `${appearance?.revision || "base"}:${state}`;
  const useCostume = Boolean(resource?.startsWith("data:image/png;base64,") && failed !== identity);
  const frame = useCostume ? entry?.frame : undefined;
  const label = `Alphonse is ${avatarStateLabel(state)}${useCostume ? ` · ${appearance?.label} appearance` : ""}.`;
  return <span className={`mascot-image ${className}`} role="img" aria-label={label} title={label}>
    <img key={`${identity}:${useCostume}`} src={useCostume ? resource : `/alphonse-states/${state}.png`} alt="" aria-hidden="true" onError={() => { if (useCostume) setFailed(identity); }} style={frame ? { width: `${frame[2] * 100}%`, height: `${frame[3] * 100}%`, maxWidth: "none", left: `${-frame[0] * 100}%`, top: `${-frame[1] * 100}%` } : undefined} />
  </span>;
}

export function SkillMascot({ state, user, context }: { state: AvatarState; user: string; context: AppearanceContext }) {
  const [preferences] = useAppearancePreferences(user);
  const [latch, setLatch] = useState(BASE_LATCH);
  // Derive immediately to prevent a one-frame stale costume at task/user changes.
  const next = useMemo(() => latchAppearance(latch, context), [latch, context]);
  useEffect(() => { setLatch((previous) => latchAppearance(previous, context)); }, [context]);
  const candidate = next.skill && appearanceAllowed(preferences, next.skill) ? context.appearance : null;
  const appearance = useAppearanceResource(user, candidate);
  return <MascotImage key={state} state={state} appearance={appearance} />;
}

export type AppearanceSkill = { id: string; title: string; editable: boolean; appearance?: SkillAppearance | null; appearance_configured?: boolean };
export function SkillAppearanceSettings({ user, skill, onSaved, beforeSave }: { user: string; skill: AppearanceSkill | null; onSaved: () => void | Promise<void>; beforeSave?: () => boolean }) {
  const [preferences, savePreferences] = useAppearancePreferences(user);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [preset, setPreset] = useState("");
  const selectedId = useRef(skill?.id);
  selectedId.current = skill?.id;
  const appearance = useAppearanceResource(user, skill?.appearance || null);
  useEffect(() => { setNotice(""); setPreset(skill?.appearance?.preset || (skill?.appearance ? "custom" : "")); }, [skill?.id, skill?.appearance?.revision]);
  const bind = async () => {
    if (!skill || (beforeSave && !beforeSave())) return;
    setBusy(true);
    try { await daemonRequest("set_skill_appearance", { actor_user_id: user, skill_id: skill.id, preset }); if (selectedId.current === skill.id) { await onSaved(); setNotice("Appearance saved."); } }
    catch (cause) { setNotice(cause instanceof Error ? cause.message : "Appearance could not be saved."); }
    finally { setBusy(false); }
  };
  return <section className="skill-appearance-settings"><h2>Mascot appearances</h2>
    <label><input type="checkbox" checked={preferences.enabled} onChange={(event) => savePreferences({ ...preferences, enabled: event.target.checked })} /> Use skill appearances</label>
    <p>Alphonse keeps its activity expressions. One leading skill chooses the outfit; ambiguous work uses the base mascot. Outfits convey context and do not represent professional credentials.</p>
    {skill && <>
      <label><input type="checkbox" checked={!preferences.disabledSkills.includes(skill.id)} onChange={(event) => savePreferences({ ...preferences, disabledSkills: event.target.checked ? preferences.disabledSkills.filter((id) => id !== skill.id) : [...preferences.disabledSkills, skill.id] })} /> Allow {skill.title}’s appearance</label>
      <p>{skill.appearance ? `Available: ${skill.appearance.label}` : skill.appearance_configured ? "Appearance unavailable or incompatible. Using the base mascot." : "This skill uses the base mascot."}</p>
      <div className="mascot-previews" aria-label="Mascot activity previews">{(["idle", "planning", "doing", "checking", "acting", "error", "disconnected"] as AvatarState[]).map((state) => <figure key={state}><MascotImage state={state} appearance={appearance} className="mascot-preview" /><figcaption>{avatarStateLabel(state)}</figcaption></figure>)}</div>
      {skill.editable && <><label>Skill outfit<select value={preset} onChange={(event) => setPreset(event.target.value)}>{skill.appearance && !skill.appearance.preset && <option value="custom" disabled>Custom package resources</option>}<option value="">Base mascot (remove binding)</option><option value="doctor">Family health assistant</option><option value="exercise-coach">Exercise coach</option></select></label><button type="button" disabled={busy || preset === "custom"} onClick={() => void bind()}>Save skill outfit</button><small>Saving replaces this skill’s optional appearance binding. Package updates may replace the binding.</small></>}
    </>}
    {notice && <p role="status">{notice}</p>}
  </section>;
}
