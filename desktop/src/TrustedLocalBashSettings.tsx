import { useEffect, useState } from "react";
import { daemonRequest, ensureDaemon, stopDaemon } from "./api";

export type TrustedLocalBashState = {
  enabled: boolean;
  active: boolean | null;
  restart_required: boolean | null;
};

type Request = (method: string, params?: Record<string, unknown>) => Promise<unknown>;

function validState(value: unknown): value is TrustedLocalBashState {
  if (!value || typeof value !== "object") return false;
  const state = value as Record<string, unknown>;
  return typeof state.enabled === "boolean"
    && (typeof state.active === "boolean" || state.active === null)
    && (typeof state.restart_required === "boolean" || state.restart_required === null);
}

export async function readTrustedLocalBash(request: Request): Promise<TrustedLocalBashState> {
  const result = await request("get_trusted_local_bash_settings", {});
  if (!validState(result)) throw new Error("Bash settings could not be checked.");
  return result;
}

export async function setTrustedLocalBash(request: Request, enabled: boolean): Promise<TrustedLocalBashState> {
  const result = await request("set_trusted_local_bash_settings", { enabled });
  if (!validState(result)) throw new Error("Bash settings could not be verified.");
  return result;
}

/** Restart once, reconciling lost stop replies against the daemon identity. */
export async function restartTrustedLocalDaemon(
  request: Request,
  stop: () => Promise<void>,
  ensure: () => Promise<void>,
  wait: () => Promise<void> = () => new Promise(resolve => setTimeout(resolve, 250)),
): Promise<TrustedLocalBashState> {
  const identity = await request("runtime_identity", {});
  const oldId = identity && typeof identity === "object" && "daemon_id" in identity
    ? (identity as { daemon_id: unknown }).daemon_id : null;
  if (typeof oldId !== "string" || !oldId) throw new Error("Could not identify the running daemon.");
  // A failed stop response may still mean the stop succeeded. Never repeat it.
  try { await stop(); } catch { /* Reconcile below. */ }
  let disconnected = false;
  for (let attempt = 0; attempt < 40; attempt++) {
    await wait();
    let current: unknown;
    try {
      current = await request("runtime_identity", {});
    } catch {
      disconnected = true;
      break;
    }
    if (current && typeof current === "object" && "daemon_id" in current
        && typeof current.daemon_id === "string" && current.daemon_id && current.daemon_id !== oldId) {
      return await readTrustedLocalBash(request);
    }
  }
  if (!disconnected) throw new Error("Daemon restart could not be confirmed. The previous daemon may still be running.");
  await ensure();
  const current = await request("runtime_identity", {});
  if (!current || typeof current !== "object" || !("daemon_id" in current)
      || typeof current.daemon_id !== "string" || !current.daemon_id || current.daemon_id === oldId) {
    throw new Error("Daemon restart could not be confirmed. The previous daemon may still be running.");
  }
  return await readTrustedLocalBash(request);
}

export function TrustedLocalBashView({ state, busy, notice, onChange, onRefresh, onRestart }: {
  state: TrustedLocalBashState | null;
  busy: boolean;
  notice: string;
  onChange: (enabled: boolean) => void;
  onRefresh: () => void;
  onRestart: () => void;
}) {
  return <section className="setting-group" aria-label="Trusted local Bash">
    <div className="setting-group-heading"><h3>Trusted local Bash</h3>
      <p>Let Alphonse use Bash in owner conversations on this Mac.</p></div>
    <p>Bash runs with your macOS user’s permissions and can access your files and network. This development mode does not enforce a protected household authorization boundary.</p>
    <label className="setting-row"><input type="checkbox" checked={state?.enabled ?? false}
      disabled={!state || busy} onChange={event => onChange(event.target.checked)} /> Enable trusted local Bash</label>
    {state && <p role="status">Saved: {state.enabled ? "On" : "Off"}. Running daemon: {state.active === null ? "Unknown" : state.active ? "On" : "Off"}.
      {state.restart_required === true && " Restart the Alphonse daemon to apply this change."}
      {state.restart_required === null && " The running daemon state could not be verified."}</p>}
    {state?.restart_required === true && <p>Restarting the daemon may interrupt active work. When turning Bash off, it remains available until the daemon restarts.</p>}
    <div className="settings-save-actions">
      {state?.restart_required === true && <button type="button" disabled={busy} onClick={onRestart}>Restart Alphonse daemon</button>}
      <button type="button" disabled={busy} onClick={onRefresh}>Refresh Bash status</button>
    </div>
    {notice && <p role="status">{notice}</p>}
  </section>;
}

export function TrustedLocalBashSettings() {
  const [state, setState] = useState<TrustedLocalBashState | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");

  const refresh = async () => {
    setBusy(true);
    try { setState(await readTrustedLocalBash(daemonRequest)); setNotice(""); }
    catch { setState(null); setNotice("Bash settings could not be checked. Try refreshing."); }
    finally { setBusy(false); }
  };
  useEffect(() => { void refresh(); }, []);

  const change = async (enabled: boolean) => {
    setBusy(true);
    try { setState(await setTrustedLocalBash(daemonRequest, enabled)); setNotice(""); }
    catch {
      // A failed response may follow a successful write. Read again rather
      // than claiming that the setting did or did not change.
      try { setState(await readTrustedLocalBash(daemonRequest)); setNotice("The change could not be confirmed; status was refreshed."); }
      catch { setState(null); setNotice("The change could not be confirmed. Refresh Bash status before trying again."); }
    }
    finally { setBusy(false); }
  };
  const restart = async () => {
    setBusy(true);
    setNotice("Restarting the daemon…");
    try {
      setState(await restartTrustedLocalDaemon(daemonRequest, stopDaemon, ensureDaemon));
      setNotice("Daemon restarted. Check the running state above.");
    } catch {
      try { setState(await readTrustedLocalBash(daemonRequest)); }
      catch { setState(null); }
      setNotice("Daemon restart could not be confirmed. The previous daemon may still be running; refresh the status before continuing.");
    } finally { setBusy(false); }
  };
  return <TrustedLocalBashView state={state} busy={busy} notice={notice} onChange={value => void change(value)} onRefresh={() => void refresh()} onRestart={() => void restart()} />;
}
