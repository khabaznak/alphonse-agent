import { invoke } from "@tauri-apps/api/core";

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
