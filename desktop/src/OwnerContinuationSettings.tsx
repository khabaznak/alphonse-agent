import { useEffect, useState } from "react";
import { daemonRequest } from "./api";

type Settings = { enabled: false } | { enabled: true; metric: "actions" | "model_calls"; interval: number };

export function OwnerContinuationSettings() {
  const [enabled, setEnabled] = useState(false);
  const [metric, setMetric] = useState<"actions" | "model_calls">("actions");
  const [interval, setInterval] = useState(5);
  const [notice, setNotice] = useState("");
  useEffect(() => { let active = true;
    void daemonRequest<Settings>("get_owner_continuation_settings", {}).then(value => {
      if (!active) return;
      setEnabled(value.enabled === true);
      if (value.enabled) { setMetric(value.metric); setInterval(value.interval); }
    }).catch(() => { if (active) setNotice("Continuation settings are unavailable."); });
    return () => { active = false; };
  }, []);
  const save = async () => {
    try {
      const params = enabled ? { enabled: true, metric, interval } : { enabled: false };
      const value = await daemonRequest<Settings>("set_owner_continuation_settings", params);
      if (value.enabled !== enabled) throw new Error("unconfirmed");
      setNotice("Continuation settings saved.");
    } catch { setNotice("Continuation settings could not be saved."); }
  };
  return <section className="setting-group">
    <div className="setting-group-heading"><h3>Long-running conversations (local preview)</h3>
      <p>Ask before Alphonse uses another work window. Checkpoints count activity; they do not measure progress.</p></div>
    <label className="setting-row"><input type="checkbox" checked={enabled} onChange={e => setEnabled(e.target.checked)} /> Ask to continue at checkpoints</label>
    {enabled && <><label>Measure <select value={metric} onChange={e => setMetric(e.target.value as "actions" | "model_calls")}>
      <option value="actions">Tool actions</option><option value="model_calls">Model calls</option></select></label>
      <label>Ask after <input type="number" min={1} max={metric === "actions" ? 9 : 10} value={interval}
        onChange={e => setInterval(Number(e.target.value))} /></label></>}
    <div className="settings-save-actions"><button type="button" onClick={() => void save()}>Save checkpoints</button>{notice && <p role="status">{notice}</p>}</div>
  </section>;
}
