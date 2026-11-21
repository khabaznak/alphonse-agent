//! Private service-first owner ingress; not a Tauri command.
//!
//! This component neither authenticates a human nor establishes that its
//! caller is protected. The caller must durably retain the message ID before
//! using it. A lost response may have committed in either store: this code
//! returns an unconfirmed outcome and never retries or falls back to the
//! unauthenticated owner ingress.

use serde_json::{json, Value};
use std::future::Future;
use std::time::Duration;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SubmitError {
    InvalidInput,
    Unconfirmed,
}

fn valid_id(value: &str) -> bool {
    value.len() == 32
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

async fn submit_with<A, D, F>(
    prompt: &str,
    message_id: &str,
    reply_to: Option<&str>,
    authority: A,
    daemon: D,
) -> Result<Value, SubmitError>
where
    A: FnOnce(&str, &str) -> Result<String, ()>,
    D: FnOnce(Value) -> F,
    F: Future<Output = Result<Value, ()>>,
{
    let goal = prompt.trim();
    if goal.is_empty()
        || goal.len() > 65_536
        || !valid_id(message_id)
        || reply_to.is_some_and(|value| !valid_id(value))
    {
        return Err(SubmitError::InvalidInput);
    }
    let authority_id = authority(goal, message_id).map_err(|_| SubmitError::Unconfirmed)?;
    if !valid_id(&authority_id) {
        return Err(SubmitError::Unconfirmed);
    }
    let mut params = json!({
        "prompt": goal, "message_id": message_id, "authority_request_id": authority_id
    });
    if let Some(reply) = reply_to {
        params["reply_to_message_id"] = json!(reply);
    }
    let result = daemon(params).await.map_err(|_| SubmitError::Unconfirmed)?;
    if result.get("message_id").and_then(Value::as_str) != Some(message_id) {
        return Err(SubmitError::Unconfirmed);
    }
    Ok(result)
}

/// Private building block only. A protected caller must retain the message ID
/// and reconcile any unconfirmed outcome before attempting another request.
#[allow(dead_code)]
pub(crate) async fn submit(
    prompt: &str,
    message_id: &str,
    reply_to: Option<&str>,
) -> Result<Value, SubmitError> {
    submit_with(
        prompt,
        message_id,
        reply_to,
        |goal, id| crate::authority_client::submit_owner_request(goal, id).map_err(|_| ()),
        |params| async move {
            tokio::time::timeout(
                Duration::from_secs(5),
                crate::ipc_request("experimental_owner_ingress", params),
            )
            .await
            .map_err(|_| ())?
            .map_err(|_| ())
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const MESSAGE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const REQUEST: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[test]
    fn service_precedes_queue_and_binds_exact_ids_and_trimmed_goal() {
        tauri::async_runtime::block_on(async {
            let stage = Cell::new(0);
            let receipt = submit_with(
                "  hello  ",
                MESSAGE,
                Some(REQUEST),
                |goal, id| {
                    assert_eq!((goal, id), ("hello", MESSAGE));
                    stage.set(1);
                    Ok(REQUEST.into())
                },
                |params| {
                    let stage = &stage;
                    async move {
                        assert_eq!(stage.get(), 1);
                        assert_eq!(
                            params,
                            json!({"prompt":"hello", "message_id":MESSAGE,
                        "authority_request_id":REQUEST, "reply_to_message_id":REQUEST})
                        );
                        Ok(json!({"message_id":MESSAGE, "accepted_at":"instant"}))
                    }
                },
            )
            .await
            .unwrap();
            assert_eq!(receipt["message_id"], MESSAGE);
        });
    }

    #[test]
    fn invalid_input_and_uncertain_authority_never_enqueue() {
        tauri::async_runtime::block_on(async {
            let enqueued = Cell::new(false);
            let invalid = submit_with(
                "hello",
                MESSAGE,
                Some("invalid"),
                |_, _| panic!("no submit"),
                |_| async {
                    enqueued.set(true);
                    Ok(json!({}))
                },
            )
            .await;
            assert_eq!(invalid, Err(SubmitError::InvalidInput));
            let uncertain = submit_with(
                "hello",
                MESSAGE,
                None,
                |_, _| Err(()),
                |_| async {
                    enqueued.set(true);
                    Ok(json!({}))
                },
            )
            .await;
            assert_eq!(uncertain, Err(SubmitError::Unconfirmed));
            assert!(!enqueued.get());
        });
    }

    #[test]
    fn invalid_queue_reply_is_unconfirmed_and_not_retried() {
        tauri::async_runtime::block_on(async {
            let attempts = Cell::new(0);
            let result = submit_with(
                "hello",
                MESSAGE,
                None,
                |_, _| Ok(REQUEST.into()),
                |_| async {
                    attempts.set(attempts.get() + 1);
                    Ok(json!({"message_id":REQUEST}))
                },
            )
            .await;
            assert_eq!(result, Err(SubmitError::Unconfirmed));
            assert_eq!(attempts.get(), 1);
        });
    }
}
