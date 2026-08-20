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
  return invoke<T>("daemon_request", { method, params });
}

export async function stopDaemon(): Promise<void> {
  await invoke("stop_daemon");
}

export async function showInFinder(path: string): Promise<void> {
  await invoke("show_in_finder", { path });
}
