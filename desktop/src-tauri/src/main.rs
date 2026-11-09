use std::env;
use std::path::PathBuf;
use std::process::Command;

use serde_json::{json, Value};
use tauri::AppHandle;
use tauri_plugin_shell::ShellExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::time::{sleep, timeout, Duration, Instant};

const PROTOCOL_VERSION: u8 = 1;
const DAEMON_START_TIMEOUT: Duration = Duration::from_secs(90);
const DAEMON_PING_TIMEOUT: &str = "daemon readiness check timed out";
static DAEMON_START_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn daemon_is_ready(value: &Value) -> bool {
    value.get("status").and_then(Value::as_str) == Some("ready")
}

async fn daemon_ping(path: PathBuf, limit: Duration) -> Result<Value, String> {
    timeout(limit, ipc_request_at(path, "ping", json!({})))
        .await
        .map_err(|_| DAEMON_PING_TIMEOUT.to_owned())?
}

fn should_spawn_daemon(result: &Result<Value, String>) -> bool {
    result.as_ref().err().is_some_and(|error| error != DAEMON_PING_TIMEOUT)
}

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
    // Multiple UI effects may ask at once. Only one should launch a sidecar;
    // waiters recheck the socket after the first startup attempt completes.
    let _startup = DAEMON_START_LOCK.lock().await;
    let existing = daemon_ping(socket_path(), Duration::from_secs(2)).await;
    if existing.as_ref().is_ok_and(daemon_is_ready) { return Ok(()); }

    if should_spawn_daemon(&existing) && cfg!(debug_assertions) {
        let project_root = development_project_root();
        let python = project_root.join(".venv").join("bin").join("python");
        let executable = if python.is_file() { python } else { PathBuf::from("python3") };
        Command::new(executable)
            .args(["-m", "alphonse.agent_v2.daemon"])
            .current_dir(project_root)
            .spawn()
            .map_err(|error| format!("could not start the development daemon: {error}"))?;
    } else if should_spawn_daemon(&existing) {
        app.shell()
            .sidecar("alphonse-daemon")
            .map_err(|error| format!("bundled daemon unavailable: {error}"))?
            .spawn()
            .map_err(|error| format!("could not start bundled daemon: {error}"))?;
    }

    // A packaged one-file daemon may need to unpack on a cold launch.
    let deadline = Instant::now() + DAEMON_START_TIMEOUT;
    while Instant::now() < deadline {
        if daemon_ping(socket_path(), Duration::from_secs(2)).await.as_ref().is_ok_and(daemon_is_ready) {
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
fn open_account_consent(url: String) -> Result<(), String> {
    let parsed = tauri::Url::parse(&url).map_err(|_| "Invalid consent URL")?;
    if parsed.scheme() != "https" || !parsed.username().is_empty() || parsed.password().is_some()
        || !matches!(parsed.host_str(), Some("accounts.google.com") | Some("login.microsoftonline.com")) {
        return Err("Unsupported consent URL".into());
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(url).spawn().map_err(|_| "Could not open consent browser")?;
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    { Err("Account consent browser is currently supported on macOS".into()) }
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
    ipc_request_at(socket_path(), method, params).await
}

async fn ipc_request_at(path: PathBuf, method: &str, params: Value) -> Result<Value, String> {
    let mut stream = UnixStream::connect(path)
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
            open_account_consent,
            stop_daemon,
            show_in_finder,
            play_alert_sound
        ])
        .run(tauri::generate_context!())
        .expect("error while running Alphonse Desktop");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_ready_ping_completes_startup() {
        assert!(daemon_is_ready(&json!({"status": "ready"})));
        assert!(!daemon_is_ready(&json!({"status": "starting"})));
        assert!(!daemon_is_ready(&json!({})));
        assert!(!should_spawn_daemon(&Ok(json!({"status": "starting"}))));
        assert!(!should_spawn_daemon(&Err(DAEMON_PING_TIMEOUT.to_owned())));
        assert!(should_spawn_daemon(&Err("daemon unavailable".to_owned())));
    }

    #[test]
    fn readiness_ping_times_out_when_peer_never_replies() {
        use std::os::unix::net::UnixListener;
        let directory = std::env::temp_dir().join(format!("alphonse-ping-test-{}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("daemon.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let peer = std::thread::spawn(move || {
            let (_connection, _) = listener.accept().unwrap();
            std::thread::sleep(Duration::from_millis(300));
        });
        let result = tauri::async_runtime::block_on(daemon_ping(path.clone(), Duration::from_millis(50)));
        assert_eq!(result.unwrap_err(), DAEMON_PING_TIMEOUT);
        peer.join().unwrap();
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
}
