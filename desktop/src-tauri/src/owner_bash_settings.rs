//! Private setup building block. This is not an exposed Tauri command.
//!
//! A protected Desktop can request administrator authorization once when
//! enabling Bash; ordinary owner requests do not use this code. The OS may
//! reuse cached authorization, so success is not proof of a fresh prompt or
//! of a particular human's identity. Protected deployment, WebView ingress and
//! a real system authorization test are required before wiring this to UI.

use crate::authority_client::{self, BashCapability};
use std::ffi::{c_char, c_void};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SettingsError {
    Unavailable,
    AuthorizationDenied,
    // A failed mutation may have reached the service. The caller must read
    // the state before offering another action, never silently replay it.
    OutcomeUnknown,
}

trait Service {
    fn status(&self) -> Result<BashCapability, SettingsError>;
    fn grant(&self, generation: u64) -> Result<BashCapability, SettingsError>;
    fn revoke(&self, generation: u64) -> Result<BashCapability, SettingsError>;
}

trait SetupAuthorization {
    fn authorize_grant(&self) -> Result<(), SettingsError>;
}

struct NativeService;
impl Service for NativeService {
    fn status(&self) -> Result<BashCapability, SettingsError> {
        authority_client::bash_capability().map_err(|_| SettingsError::Unavailable)
    }
    fn grant(&self, generation: u64) -> Result<BashCapability, SettingsError> {
        authority_client::grant_bash(generation).map_err(|_| SettingsError::OutcomeUnknown)
    }
    fn revoke(&self, generation: u64) -> Result<BashCapability, SettingsError> {
        authority_client::revoke_bash(generation).map_err(|_| SettingsError::OutcomeUnknown)
    }
}

fn update<S: Service, A: SetupAuthorization>(
    service: &S, authorization: &A, enabled: bool,
) -> Result<BashCapability, SettingsError> {
    let current = service.status()?;
    if enabled && current.enabled {
        return Ok(current);
    }
    if enabled {
        authorization.authorize_grant()?;
        service.grant(current.generation)
    } else {
        service.revoke(current.generation)
    }
}

#[repr(C)]
struct AuthorizationItem {
    name: *const c_char,
    value_length: usize,
    value: *mut c_void,
    flags: u32,
}
#[repr(C)]
struct AuthorizationItemSet {
    count: u32,
    items: *mut AuthorizationItem,
}
type AuthorizationRef = *const c_void;

#[link(name = "Security", kind = "framework")]
extern "C" {
    fn AuthorizationCreate(rights: *const AuthorizationItemSet, environment: *const AuthorizationItemSet,
                           flags: u32, authorization: *mut AuthorizationRef) -> i32;
    fn AuthorizationFree(authorization: AuthorizationRef, flags: u32) -> i32;
}

struct MacSetupAuthorization;
impl SetupAuthorization for MacSetupAuthorization {
    fn authorize_grant(&self) -> Result<(), SettingsError> {
        // kAuthorizationRightExecute, named by the macOS SDK as the standard
        // administrative right. Do not accept a right or flags from the UI.
        let mut item = AuthorizationItem {
            name: b"system.privilege.admin\0".as_ptr().cast(), value_length: 0,
            value: std::ptr::null_mut(), flags: 0,
        };
        let rights = AuthorizationItemSet { count: 1, items: &mut item };
        let mut reference: AuthorizationRef = std::ptr::null();
        // InteractionAllowed | ExtendRights; no PartialRights or PreAuthorize.
        let result = unsafe { AuthorizationCreate(&rights, std::ptr::null(), 3, &mut reference) };
        if result != 0 || reference.is_null() {
            if !reference.is_null() { unsafe { AuthorizationFree(reference, 0); } }
            return Err(SettingsError::AuthorizationDenied);
        }
        unsafe { AuthorizationFree(reference, 0); }
        Ok(())
    }
}

#[allow(dead_code)]
pub(crate) fn status() -> Result<BashCapability, SettingsError> {
    NativeService.status()
}

#[allow(dead_code)]
pub(crate) fn set_enabled(enabled: bool) -> Result<BashCapability, SettingsError> {
    update(&NativeService, &MacSetupAuthorization, enabled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct Fake { calls: RefCell<Vec<&'static str>>, enabled: bool, fail_status: bool, fail_change: bool }
    impl Service for Fake {
        fn status(&self) -> Result<BashCapability, SettingsError> {
            self.calls.borrow_mut().push("status");
            if self.fail_status { Err(SettingsError::Unavailable) }
            else { Ok(BashCapability { enabled: self.enabled, generation: 1 }) }
        }
        fn grant(&self, generation: u64) -> Result<BashCapability, SettingsError> {
            assert_eq!(generation, 1);
            self.calls.borrow_mut().push("grant");
            if self.fail_change { Err(SettingsError::OutcomeUnknown) }
            else { Ok(BashCapability { enabled: true, generation: 2 }) }
        }
        fn revoke(&self, generation: u64) -> Result<BashCapability, SettingsError> {
            assert_eq!(generation, 1);
            self.calls.borrow_mut().push("revoke");
            if self.fail_change { Err(SettingsError::OutcomeUnknown) }
            else { Ok(BashCapability { enabled: false, generation: 2 }) }
        }
    }
    struct Approval<'a> { calls: &'a RefCell<Vec<&'static str>>, approved: bool }
    impl SetupAuthorization for Approval<'_> {
        fn authorize_grant(&self) -> Result<(), SettingsError> {
            self.calls.borrow_mut().push("authorize");
            if self.approved { Ok(()) } else { Err(SettingsError::AuthorizationDenied) }
        }
    }
    fn fake(enabled: bool) -> Fake {
        Fake { calls: RefCell::new(Vec::new()), enabled, fail_status: false, fail_change: false }
    }
    #[test]
    fn grant_requires_approval_and_denial_never_mutates() {
        let service = fake(false);
        let denied = Approval { calls: &service.calls, approved: false };
        assert_eq!(update(&service, &denied, true), Err(SettingsError::AuthorizationDenied));
        assert_eq!(*service.calls.borrow(), ["status", "authorize"]);
        service.calls.borrow_mut().clear();
        let approved = Approval { calls: &service.calls, approved: true };
        assert!(update(&service, &approved, true).unwrap().enabled);
        assert_eq!(*service.calls.borrow(), ["status", "authorize", "grant"]);
    }
    #[test]
    fn revoke_and_unchanged_state_do_not_prompt() {
        let service = fake(true);
        let approval = Approval { calls: &service.calls, approved: false };
        assert!(update(&service, &approval, true).unwrap().enabled);
        assert_eq!(*service.calls.borrow(), ["status"]);
        service.calls.borrow_mut().clear();
        assert!(!update(&service, &approval, false).unwrap().enabled);
        assert_eq!(*service.calls.borrow(), ["status", "revoke"]);
        let service = fake(false);
        let approval = Approval { calls: &service.calls, approved: false };
        assert!(!update(&service, &approval, false).unwrap().enabled);
        assert_eq!(*service.calls.borrow(), ["status", "revoke"]);
    }
    #[test]
    fn errors_do_not_prompt_or_replay_mutations() {
        let mut service = fake(false);
        service.fail_status = true;
        let approval = Approval { calls: &service.calls, approved: true };
        assert_eq!(update(&service, &approval, true), Err(SettingsError::Unavailable));
        assert_eq!(*service.calls.borrow(), ["status"]);
        service.calls.borrow_mut().clear();
        service.fail_status = false;
        service.fail_change = true;
        let approval = Approval { calls: &service.calls, approved: true };
        assert_eq!(update(&service, &approval, true), Err(SettingsError::OutcomeUnknown));
        assert_eq!(*service.calls.borrow(), ["status", "authorize", "grant"]);
    }
}
