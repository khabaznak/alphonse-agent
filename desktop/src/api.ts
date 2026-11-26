import { invoke } from "@tauri-apps/api/core";
import { CALENDAR_CONFIGURATION_CHANGED } from "./agendaDates";

export async function ensureDaemon(): Promise<void> {
  await invoke("ensure_daemon");
}

export async function getDesktopDiagnosticMode(): Promise<string> {
  return invoke<string>("desktop_diagnostic_mode");
}

export async function getDesktopDiagnosticProjectId(): Promise<string> {
  return invoke<string>("desktop_diagnostic_project_id");
}

export async function daemonRequest<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  try {
    return await invoke<T>("daemon_request", { method, params });
  } catch (cause) {
    // Tauri command failures are commonly rejected as plain strings, while
    // callers display details only for Error instances. Normalize them here
    // so the daemon's validation or filesystem error reaches the UI.
    const message = cause instanceof Error
      ? cause.message
      : typeof cause === "string"
        ? cause
        : cause && typeof cause === "object" && "message" in cause
          ? String((cause as { message: unknown }).message)
          : String(cause || "Daemon request failed");
    throw new Error(message);
  }
}

export async function stopDaemon(): Promise<void> {
  await invoke("stop_daemon");
}

export async function showInFinder(path: string): Promise<void> {
  await invoke("show_in_finder", { path });
}

export async function playAlertSound(path = ""): Promise<void> {
  await invoke("play_alert_sound", { path });
}

// Identity is derived by the daemon from the OS peer, never from UI user fields.
export async function optionalManagement<T>(action: string, values: Record<string, unknown> = {}): Promise<T> {
  const result = await daemonRequest<T>("optional_management", { action, values });
  if (typeof window !== "undefined" && (["catalog", "pause", "disconnect", "revoke_grant"].includes(action)
      || (action === "save_grant" && Array.isArray(values.operations) && values.operations.includes("calendar_read")))) {
    window.dispatchEvent(new Event(CALENDAR_CONFIGURATION_CHANGED));
  }
  return result;
}

export async function openAccountConsent(url: string): Promise<void> {
  await invoke("open_account_consent", { url });
}
