//! Native, typed transport to a separately provisioned authority service.
//!
//! This module is not an owner-authentication mechanism. It is intentionally
//! not exported as a Tauri command: a signed application or an arbitrary
//! webview invocation does not prove that a person submitted a request.
//! Its binary, resources and complete launch configuration must be protected
//! from the account executing Bash before the service may trust this client.

use std::ffi::{CString, OsStr};
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::UnixStream;
use std::path::{Component, Path};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::json;

// Provisioning must place this configuration under protected root-owned
// ancestors. Neither the model, the webview nor an IPC argument chooses it.
const CONFIG_PATH: &str = "/private/var/db/alphonse/desktop-authority.json";
const MAX_FRAME: usize = 131_072;
const MAX_CONFIG: usize = 4096;
const TIMEOUT: Duration = Duration::from_secs(3);

/// Development/Vite builds must never submit protected-owner authority.
/// This is one necessary build property, not proof of a protected executable,
/// trustworthy frontend content, or a human-origin request.
fn protected_owner_build() -> bool {
    protected_owner_build_flags(cfg!(feature = "protected-owner"), cfg!(debug_assertions))
}

fn protected_owner_build_flags(feature_enabled: bool, debug: bool) -> bool {
    feature_enabled && !debug
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AuthorityUnavailable;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceConfig {
    version: u8,
    service_uid: u32,
    desktop_socket: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SubmitResult {
    request_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SubmitReply {
    ok: bool,
    result: SubmitResult,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BashCapability {
    pub(crate) enabled: bool,
    pub(crate) generation: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapabilityReply {
    ok: bool,
    result: BashCapability,
}

enum CapabilityOperation {
    Read,
    Grant(u64),
    Revoke(u64),
}

extern "C" {
    fn acl_get_fd_np(fd: libc::c_int, acl_type: libc::c_int) -> *mut libc::c_void;
    fn acl_free(value: *mut libc::c_void) -> libc::c_int;
    fn getpeereid(fd: libc::c_int, uid: *mut libc::uid_t, gid: *mut libc::gid_t) -> libc::c_int;
}

fn id_valid(id: &str) -> bool {
    id.len() == 32
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn config_valid(config: &ServiceConfig, owner_uid: u32) -> bool {
    let path = Path::new(&config.desktop_socket);
    config.version == 1
        && config.service_uid != 0
        && config.service_uid != owner_uid
        && config.service_uid != u32::MAX
        && path.is_absolute()
        && config.desktop_socket.len() <= 100
        && !config.desktop_socket.as_bytes().contains(&0)
        && path
            .components()
            .all(|part| matches!(part, Component::RootDir | Component::Normal(_)))
}

fn no_acl(fd: libc::c_int) -> bool {
    // Darwin ACL_TYPE_EXTENDED. Unknown errors are not evidence of no ACL.
    unsafe {
        *libc::__error() = 0;
        let acl = acl_get_fd_np(fd, 0x0000_0100);
        if !acl.is_null() {
            acl_free(acl);
            return false;
        }
        *libc::__error() == libc::ENOENT
    }
}

fn checked_fd(fd: libc::c_int, directory: bool) -> Result<(), AuthorityUnavailable> {
    let mut info: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut info) } != 0
        || info.st_uid != 0
        || info.st_mode & 0o022 != 0
        || !no_acl(fd)
    {
        return Err(AuthorityUnavailable);
    }
    let kind = info.st_mode & libc::S_IFMT;
    if (directory && kind != libc::S_IFDIR)
        || (!directory && (kind != libc::S_IFREG || info.st_nlink != 1))
    {
        return Err(AuthorityUnavailable);
    }
    Ok(())
}

fn open_at(
    parent: libc::c_int,
    component: &OsStr,
    directory: bool,
) -> Result<OwnedFd, AuthorityUnavailable> {
    let name = CString::new(component.as_bytes()).map_err(|_| AuthorityUnavailable)?;
    let flags = libc::O_RDONLY
        | libc::O_NOFOLLOW
        | libc::O_CLOEXEC
        | if directory { libc::O_DIRECTORY } else { 0 };
    let fd = unsafe { libc::openat(parent, name.as_ptr(), flags) };
    if fd < 0 {
        return Err(AuthorityUnavailable);
    }
    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
    checked_fd(owned.as_raw_fd(), directory)?;
    Ok(owned)
}

fn open_protected_file(path: &Path) -> Result<OwnedFd, AuthorityUnavailable> {
    if !path.is_absolute() {
        return Err(AuthorityUnavailable);
    }
    let parts: Vec<_> = path.components().collect();
    if parts.len() < 2
        || parts[0] != Component::RootDir
        || parts[1..]
            .iter()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(AuthorityUnavailable);
    }
    let root = CString::new("/").map_err(|_| AuthorityUnavailable)?;
    let raw = unsafe {
        libc::open(
            root.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(AuthorityUnavailable);
    }
    let mut current = unsafe { OwnedFd::from_raw_fd(raw) };
    checked_fd(current.as_raw_fd(), true)?;
    for (index, component) in parts[1..].iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err(AuthorityUnavailable);
        };
        current = open_at(current.as_raw_fd(), name, index + 1 < parts.len() - 1)?;
    }
    Ok(current)
}

fn read_protected_config(path: &Path) -> Result<ServiceConfig, AuthorityUnavailable> {
    let current = open_protected_file(path)?;
    let mut data = Vec::new();
    File::from(current)
        .take((MAX_CONFIG + 1) as u64)
        .read_to_end(&mut data)
        .map_err(|_| AuthorityUnavailable)?;
    if data.len() > MAX_CONFIG {
        return Err(AuthorityUnavailable);
    }
    let config: ServiceConfig = serde_json::from_slice(&data).map_err(|_| AuthorityUnavailable)?;
    if !config_valid(&config, unsafe { libc::geteuid() }) {
        return Err(AuthorityUnavailable);
    }
    Ok(config)
}

fn service_peer_uid(stream: &UnixStream) -> Result<u32, AuthorityUnavailable> {
    let mut uid: libc::uid_t = 0;
    let mut gid: libc::gid_t = 0;
    if unsafe { getpeereid(stream.as_raw_fd(), &mut uid, &mut gid) } != 0 {
        return Err(AuthorityUnavailable);
    }
    Ok(uid)
}

fn remaining(deadline: Instant) -> Result<Duration, AuthorityUnavailable> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or(AuthorityUnavailable)
}

fn wait_socket(
    fd: libc::c_int,
    events: libc::c_short,
    deadline: Instant,
) -> Result<(), AuthorityUnavailable> {
    loop {
        let millis = remaining(deadline)?
            .as_millis()
            .max(1)
            .min(i32::MAX as u128) as libc::c_int;
        let mut descriptor = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        let result = unsafe { libc::poll(&mut descriptor, 1, millis) };
        if result > 0 {
            return Ok(());
        }
        if result == 0 {
            return Err(AuthorityUnavailable);
        }
        if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
            return Err(AuthorityUnavailable);
        }
    }
}

fn connect_bounded(path: &str, deadline: Instant) -> Result<UnixStream, AuthorityUnavailable> {
    let name = path.as_bytes();
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    if name.is_empty() || name.len() >= address.sun_path.len() || name.contains(&0) {
        return Err(AuthorityUnavailable);
    }
    address.sun_family = libc::AF_UNIX as u8;
    for (target, source) in address.sun_path.iter_mut().zip(name) {
        *target = *source as libc::c_char;
    }
    let address_len = std::mem::offset_of!(libc::sockaddr_un, sun_path) + name.len() + 1;
    address.sun_len = u8::try_from(address_len).map_err(|_| AuthorityUnavailable)?;
    let raw = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) };
    if raw < 0 {
        return Err(AuthorityUnavailable);
    }
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    if unsafe { libc::fcntl(raw, libc::F_SETFD, libc::FD_CLOEXEC) } < 0
        || unsafe { libc::fcntl(raw, libc::F_SETFL, libc::O_NONBLOCK) } < 0
    {
        return Err(AuthorityUnavailable);
    }
    let result = unsafe {
        libc::connect(
            raw,
            &address as *const _ as *const libc::sockaddr,
            address_len as libc::socklen_t,
        )
    };
    if result != 0 {
        let code = std::io::Error::last_os_error().raw_os_error();
        if !matches!(code, Some(libc::EINPROGRESS) | Some(libc::EAGAIN)) {
            return Err(AuthorityUnavailable);
        }
        wait_socket(raw, libc::POLLOUT, deadline)?;
        let mut error: libc::c_int = 0;
        let mut len = std::mem::size_of_val(&error) as libc::socklen_t;
        if unsafe {
            libc::getsockopt(
                raw,
                libc::SOL_SOCKET,
                libc::SO_ERROR,
                &mut error as *mut _ as *mut libc::c_void,
                &mut len,
            )
        } != 0
            || error != 0
            || len as usize != std::mem::size_of_val(&error)
        {
            return Err(AuthorityUnavailable);
        }
    }
    Ok(UnixStream::from(fd))
}

fn read_exact_bounded(
    stream: &mut UnixStream,
    mut buffer: &mut [u8],
    deadline: Instant,
) -> Result<(), AuthorityUnavailable> {
    while !buffer.is_empty() {
        wait_socket(stream.as_raw_fd(), libc::POLLIN, deadline)?;
        match stream.read(buffer) {
            Ok(0) => return Err(AuthorityUnavailable),
            Ok(n) => buffer = &mut buffer[n..],
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::Interrupted | std::io::ErrorKind::WouldBlock
                ) =>
            {
                continue
            }
            Err(_) => return Err(AuthorityUnavailable),
        }
    }
    Ok(())
}

fn write_all_bounded(
    stream: &mut UnixStream,
    mut data: &[u8],
    deadline: Instant,
) -> Result<(), AuthorityUnavailable> {
    while !data.is_empty() {
        wait_socket(stream.as_raw_fd(), libc::POLLOUT, deadline)?;
        match stream.write(data) {
            Ok(0) => return Err(AuthorityUnavailable),
            Ok(n) => data = &data[n..],
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::Interrupted | std::io::ErrorKind::WouldBlock
                ) =>
            {
                continue
            }
            Err(_) => return Err(AuthorityUnavailable),
        }
    }
    Ok(())
}

fn exchange_submit(
    stream: &mut UnixStream,
    goal: &str,
    message_id: &str,
    deadline: Instant,
) -> Result<String, AuthorityUnavailable> {
    let reply: SubmitReply = exchange_frame(
        stream,
        &json!({"op":"submit", "arguments":{
            "goal": goal, "idempotency_key": message_id
        }}),
        deadline,
    )?;
    if !reply.ok || !id_valid(&reply.result.request_id) {
        return Err(AuthorityUnavailable);
    }
    Ok(reply.result.request_id)
}

fn exchange_frame<T: serde::de::DeserializeOwned>(
    stream: &mut UnixStream,
    request: &serde_json::Value,
    deadline: Instant,
) -> Result<T, AuthorityUnavailable> {
    let payload = serde_json::to_vec(request).map_err(|_| AuthorityUnavailable)?;
    if payload.is_empty() || payload.len() > MAX_FRAME {
        return Err(AuthorityUnavailable);
    }
    write_all_bounded(stream, &(payload.len() as u32).to_be_bytes(), deadline)?;
    write_all_bounded(stream, &payload, deadline)?;
    let mut header = [0u8; 4];
    read_exact_bounded(stream, &mut header, deadline)?;
    let size = u32::from_be_bytes(header) as usize;
    if size == 0 || size > MAX_FRAME {
        return Err(AuthorityUnavailable);
    }
    let mut reply = vec![0; size];
    read_exact_bounded(stream, &mut reply, deadline)?;
    serde_json::from_slice(&reply).map_err(|_| AuthorityUnavailable)
}

fn submit_using_config(
    config: &ServiceConfig,
    goal: &str,
    message_id: &str,
) -> Result<String, AuthorityUnavailable> {
    // The existing owner ingress stores the trimmed prompt; bind the service
    // and queue to the same canonical input without inventing a request.
    let goal = goal.trim();
    if !config_valid(config, unsafe { libc::geteuid() })
        || !id_valid(message_id)
        || goal.is_empty()
        || goal.len() > 65_536
    {
        return Err(AuthorityUnavailable);
    }
    let deadline = Instant::now() + TIMEOUT;
    let mut stream = connect_service(config, deadline)?;
    exchange_submit(&mut stream, goal, message_id, deadline)
}

fn connect_service(
    config: &ServiceConfig,
    deadline: Instant,
) -> Result<UnixStream, AuthorityUnavailable> {
    if !config_valid(config, unsafe { libc::geteuid() }) {
        return Err(AuthorityUnavailable);
    }
    let stream = connect_bounded(&config.desktop_socket, deadline)?;
    // No request bytes, handshakes, prompts or credentials leave before this
    // check. The protected UID authenticates an OS identity, not a human.
    if service_peer_uid(&stream)? != config.service_uid {
        return Err(AuthorityUnavailable);
    }
    Ok(stream)
}

/// Caller must persist the unique message ID before invoking; a transport
/// failure may mean the service already recorded it and must not be replayed
/// automatically. This function is not connected to the webview or daemon.
#[allow(dead_code)]
pub(crate) fn submit_owner_request(
    goal: &str,
    message_id: &str,
) -> Result<String, AuthorityUnavailable> {
    let config = protected_service_config()?;
    submit_using_config(&config, goal, message_id)
}

fn protected_service_config() -> Result<ServiceConfig, AuthorityUnavailable> {
    if !protected_owner_build() {
        return Err(AuthorityUnavailable);
    }
    // This only checks a necessary installation property. In particular it
    // does not authenticate a UI event or verify the running process image.
    let executable = std::env::current_exe().map_err(|_| AuthorityUnavailable)?;
    let _protected_executable = open_protected_file(&executable)?;
    read_protected_config(Path::new(CONFIG_PATH))
}

fn exchange_capability(
    stream: &mut UnixStream,
    operation: CapabilityOperation,
    deadline: Instant,
) -> Result<BashCapability, AuthorityUnavailable> {
    let (op, arguments) = match operation {
        CapabilityOperation::Read => ("bash_capability", json!({})),
        CapabilityOperation::Grant(generation) => ("grant_bash", json!({"expected_generation": generation})),
        CapabilityOperation::Revoke(generation) => ("revoke_bash", json!({"expected_generation": generation})),
    };
    let reply: CapabilityReply =
        exchange_frame(stream, &json!({"op": op, "arguments": arguments}), deadline)?;
    if !reply.ok {
        return Err(AuthorityUnavailable);
    }
    Ok(reply.result)
}

// Deliberately not Tauri commands: a webview invocation does not establish
// owner intent. A failed mutation may have reached the service; callers must
// inspect state rather than assume it failed or silently retry it.
#[allow(dead_code)]
fn bash_capability_operation(
    operation: CapabilityOperation,
) -> Result<BashCapability, AuthorityUnavailable> {
    let config = protected_service_config()?;
    let deadline = Instant::now() + TIMEOUT;
    let mut stream = connect_service(&config, deadline)?;
    exchange_capability(&mut stream, operation, deadline)
}

#[allow(dead_code)]
pub(crate) fn bash_capability() -> Result<BashCapability, AuthorityUnavailable> {
    bash_capability_operation(CapabilityOperation::Read)
}

#[allow(dead_code)]
pub(crate) fn grant_bash(expected_generation: u64) -> Result<BashCapability, AuthorityUnavailable> {
    bash_capability_operation(CapabilityOperation::Grant(expected_generation))
}

#[allow(dead_code)]
pub(crate) fn revoke_bash(expected_generation: u64) -> Result<BashCapability, AuthorityUnavailable> {
    bash_capability_operation(CapabilityOperation::Revoke(expected_generation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn debug_build_cannot_submit_protected_owner_requests() {
        assert!(!protected_owner_build_flags(false, true));
        assert!(!protected_owner_build_flags(false, false));
        assert!(!protected_owner_build_flags(true, true));
        assert!(protected_owner_build_flags(true, false));
        if cfg!(debug_assertions) {
            assert!(!protected_owner_build());
            assert!(submit_owner_request("synthetic", &"a".repeat(32)).is_err());
        }
    }

    #[test]
    fn user_writable_executable_path_is_rejected() {
        let path = std::env::temp_dir().join(format!("alpha-auth-binary-{}", std::process::id()));
        std::fs::write(&path, b"synthetic executable").unwrap();
        assert!(open_protected_file(&path).is_err());
        std::fs::remove_file(path).unwrap();
        assert!(open_protected_file(Path::new("relative/binary")).is_err());
    }

    #[test]
    fn capability_operations_are_typed_and_fail_closed() {
        for (operation, expected_op, expected_args, response, expected) in [
            (
                CapabilityOperation::Read,
                "bash_capability",
                json!({}),
                r#"{"ok":true,"result":{"enabled":false,"generation":0}}"#,
                Some(false),
            ),
            (
                CapabilityOperation::Grant(0),
                "grant_bash",
                json!({"expected_generation": 0}),
                r#"{"ok":true,"result":{"enabled":true,"generation":1}}"#,
                Some(true),
            ),
            (
                CapabilityOperation::Revoke(1),
                "revoke_bash",
                json!({"expected_generation": 1}),
                r#"{"ok":true,"result":{"enabled":false,"generation":2}}"#,
                Some(false),
            ),
            (
                CapabilityOperation::Grant(0),
                "grant_bash",
                json!({"expected_generation": 0}),
                r#"{"ok":true,"result":{"enabled":true,"enabled":false,"generation":1}}"#,
                None,
            ),
        ] {
            let (mut client, mut server) = UnixStream::pair().unwrap();
            client.set_nonblocking(true).unwrap();
            let worker = thread::spawn(move || {
                let mut header = [0u8; 4];
                server.read_exact(&mut header).unwrap();
                let mut body = vec![0; u32::from_be_bytes(header) as usize];
                server.read_exact(&mut body).unwrap();
                let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(request, json!({"op": expected_op, "arguments": expected_args}));
                server
                    .write_all(&(response.len() as u32).to_be_bytes())
                    .unwrap();
                server.write_all(response.as_bytes()).unwrap();
            });
            let result = exchange_capability(&mut client, operation, Instant::now() + TIMEOUT);
            assert_eq!(result.ok().map(|value| value.enabled), expected);
            worker.join().unwrap();
        }
        if cfg!(debug_assertions) {
            assert!(bash_capability().is_err());
            assert!(grant_bash(0).is_err());
            assert!(revoke_bash(0).is_err());
        }
    }

    #[test]
    fn configuration_rejects_same_user_paths_and_duplicate_fields() {
        let uid = unsafe { libc::geteuid() };
        let config = ServiceConfig {
            version: 1,
            service_uid: uid,
            desktop_socket: "/tmp/test.sock".into(),
        };
        assert!(!config_valid(&config, uid));
        let config = ServiceConfig {
            version: 1,
            service_uid: uid + 1,
            desktop_socket: "/tmp/../bad.sock".into(),
        };
        assert!(!config_valid(&config, uid));
        assert!(serde_json::from_str::<ServiceConfig>(
            r#"{"version":1,"version":1,"service_uid":123,"desktop_socket":"/tmp/a"}"#
        )
        .is_err());
        assert!(!id_valid("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"));
    }

    #[test]
    fn same_user_impostor_receives_no_request_bytes() {
        let path = Path::new("/private/tmp").join(format!(
            "alpha-auth-{}-{}.sock",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let listener = UnixListener::bind(&path).unwrap();
        let (sender, receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut data = [0u8; 8];
            sender.send(socket.read(&mut data).unwrap()).unwrap();
        });
        let config = ServiceConfig {
            version: 1,
            service_uid: unsafe { libc::geteuid() } + 1,
            desktop_socket: path.to_str().unwrap().into(),
        };
        assert!(submit_using_config(&config, "synthetic owner request", &"a".repeat(32)).is_err());
        assert_eq!(receiver.recv_timeout(Duration::from_secs(3)).unwrap(), 0);
        worker.join().unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_user_owned_configuration_before_reading() {
        let path = std::env::temp_dir().join(format!("alpha-auth-config-{}", std::process::id()));
        std::fs::write(&path, b"{}").unwrap();
        assert!(read_protected_config(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn framed_submit_is_typed_and_handles_escaped_owner_input() {
        let (mut client, mut server) = UnixStream::pair().unwrap();
        client.set_nonblocking(true).unwrap();
        let worker = thread::spawn(move || {
            let mut header = [0u8; 4];
            server.read_exact(&mut header).unwrap();
            let mut body = vec![0u8; u32::from_be_bytes(header) as usize];
            server.read_exact(&mut body).unwrap();
            let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(request["op"], "submit");
            assert_eq!(request["arguments"]["goal"], "line \"one\"\nnext");
            assert_eq!(request["arguments"]["idempotency_key"], "a".repeat(32));
            assert_eq!(request["arguments"].as_object().unwrap().len(), 2);
            let reply =
                br#"{"ok":true,"result":{"request_id":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}"#;
            server
                .write_all(&(reply.len() as u32).to_be_bytes())
                .unwrap();
            server.write_all(reply).unwrap();
        });
        assert_eq!(
            exchange_submit(
                &mut client,
                "line \"one\"\nnext",
                &"a".repeat(32),
                Instant::now() + TIMEOUT
            )
            .unwrap(),
            "b".repeat(32)
        );
        worker.join().unwrap();
    }

    #[test]
    fn incomplete_service_response_has_a_bounded_deadline() {
        let (mut client, _server) = UnixStream::pair().unwrap();
        client.set_nonblocking(true).unwrap();
        let started = Instant::now();
        assert!(read_exact_bounded(
            &mut client,
            &mut [0; 4],
            started + Duration::from_millis(40)
        )
        .is_err());
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn saturated_unix_listener_cannot_block_connect_indefinitely() {
        let path = Path::new("/private/tmp").join(format!(
            "alpha-full-{}-{}.sock",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let listener = UnixListener::bind(&path).unwrap();
        assert_eq!(unsafe { libc::listen(listener.as_raw_fd(), 1) }, 0);
        let started = Instant::now();
        let mut open = Vec::new();
        let mut refused_or_timed_out = false;
        for _ in 0..128 {
            match connect_bounded(
                path.to_str().unwrap(),
                Instant::now() + Duration::from_millis(40),
            ) {
                Ok(stream) => open.push(stream),
                Err(_) => {
                    refused_or_timed_out = true;
                    break;
                }
            }
        }
        assert!(
            refused_or_timed_out,
            "fixture did not saturate the listener"
        );
        assert!(started.elapsed() < Duration::from_secs(2));
        drop(open);
        drop(listener);
        std::fs::remove_file(path).unwrap();
    }
}
