use std::env;
use std::path::PathBuf;
use std::process::Command;

use serde_json::{json, Value};
use tauri::AppHandle;
use tauri_plugin_shell::ShellExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::time::{sleep, Duration, Instant};

const PROTOCOL_VERSION: u8 = 1;

#[tauri::command]
fn desktop_diagnostic_mode() -> String {
    env::var("ALPHONSE_DESKTOP_DIAGNOSTIC_MODE").unwrap_or_else(|_| "normal".to_owned())
}

#[tauri::command]
fn desktop_diagnostic_project_id() -> String {
    env::var("ALPHONSE_DESKTOP_DIAGNOSTIC_PROJECT_ID").unwrap_or_default()
}

#[tauri::command]
async fn ensure_daemon(app: AppHandle) -> Result<(), String> {
    if ipc_request("ping", json!({})).await.is_ok() {
        return Ok(());
    }

    if cfg!(debug_assertions) {
        let project_root = development_project_root();
        let python = project_root.join(".venv").join("bin").join("python");
        let executable = if python.is_file() { python } else { PathBuf::from("python3") };
        Command::new(executable)
            .args(["-m", "alphonse.agent_v2.daemon"])
            .current_dir(project_root)
            .spawn()
            .map_err(|error| format!("could not start the development daemon: {error}"))?;
    } else {
        app.shell()
            .sidecar("alphonse-daemon")
            .map_err(|error| format!("bundled daemon unavailable: {error}"))?
            .spawn()
            .map_err(|error| format!("could not start bundled daemon: {error}"))?;
    }

    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline {
        if ipc_request("ping", json!({})).await.is_ok() {
            return Ok(());
        }
        sleep(Duration::from_millis(150)).await;
    }
    Err("Alphonse daemon did not become ready".into())
}

fn development_project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[tauri::command]
async fn daemon_request(method: String, params: Value) -> Result<Value, String> {
    ipc_request(&method, params).await
}

#[tauri::command]
async fn stop_daemon() -> Result<(), String> {
    ipc_request("stop", json!({})).await.map(|_| ())
}

#[tauri::command]
fn show_in_finder(path: String) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let root = PathBuf::from(path);
        if !root.is_dir() {
            return Err("Project directory is unavailable".into());
        }
        let status = Command::new("open")
            .arg(&root)
            .status()
            .map_err(|error| format!("Could not open Finder: {error}"))?;
        if status.success() {
            Ok(())
        } else {
            Err("Finder could not open the project directory".into())
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = path;
        Err("Show in Finder is available on macOS only".into())
    }
}

#[tauri::command]
fn play_alert_sound(path: String) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let configured = path.trim();
        let mut command = if configured.is_empty() {
            let mut command = Command::new("osascript");
            command.args(["-e", "beep 1"]);
            command
        } else {
            let sound = PathBuf::from(configured);
            if !sound.is_file() {
                return Err("Selected notification sound is unavailable".into());
            }
            let mut command = Command::new("afplay");
            command.arg(sound);
            command
        };
        command
            .spawn()
            .map(|_| ())
            .map_err(|error| format!("Could not play alert sound: {error}"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = path;
        Ok(())
    }
}

async fn ipc_request(method: &str, params: Value) -> Result<Value, String> {
    let mut stream = UnixStream::connect(socket_path())
        .await
        .map_err(|error| format!("daemon unavailable: {error}"))?;
    let request = json!({
        "version": PROTOCOL_VERSION,
        "request_id": format!("desktop-{}", method),
        "method": method,
        "params": params,
    });
    let encoded = format!(
        "{}\n",
        serde_json::to_string(&request).map_err(|error| error.to_string())?
    );
    stream
        .write_all(encoded.as_bytes())
        .await
        .map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    stream
        .read_to_end(&mut bytes)
        .await
        .map_err(|error| error.to_string())?;
    let response: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid daemon response: {error}"))?;
    if response.get("ok").and_then(Value::as_bool) != Some(true) {
        return Err(response
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("daemon request failed")
            .to_owned());
    }
    Ok(response.get("result").cloned().unwrap_or_else(|| json!({})))
}

fn socket_path() -> PathBuf {
    env::var_os("ALPHONSE_V2_SOCKET_PATH")
        .map(PathBuf::from)
        .or_else(|| dirs_home().map(|home| home.join(".alphonse").join("v2-daemon.sock")))
        .unwrap_or_else(|| PathBuf::from("/tmp/alphonse-v2-daemon.sock"))
}

fn dirs_home() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .invoke_handler(tauri::generate_handler![
            desktop_diagnostic_mode,
            desktop_diagnostic_project_id,
            ensure_daemon,
            daemon_request,
            stop_daemon,
            show_in_finder,
            play_alert_sound
        ])
        .run(tauri::generate_context!())
        .expect("error while running Alphonse Desktop");
}
