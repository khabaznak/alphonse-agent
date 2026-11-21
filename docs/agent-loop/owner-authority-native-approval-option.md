# Possible owner approval for host Bash

This describes a possible first-owner workflow, not an implemented feature or
the selected product UX. In particular, it does not add an approval to normal
conversation turns. It needs a product decision and a real native interface
before it can authorize any action.

The development Desktop loads its webview from a same-user Vite server and a
user-writable project. If a Tauri command directly grants authority for a
webview-supplied prompt, a Bash child can alter that frontend and cause the
genuine Desktop process to submit a fabricated request. Checking the Desktop
executable's signature or pinned code hash alone cannot distinguish that call
from one initiated by the owner. A JavaScript `isTrusted` value or a
webview-provided boolean is not a trustworthy native approval event.

One possible narrow alternative is to defer approval until the first time the
worker selects host Bash for a request. A protected release Desktop could show
the exact request in a native confirmation view; only the native confirmation
callback would cause the protected service to bind the actor, original message
ID and exact displayed request. The resulting task-scoped grant could cover
subsequent Bash actions for that same request, without approving each command.
The owner would still need to understand that Bash has the ordinary host
account's file, network and credential access. A model judgment, a webview
flag, or a signed process without an actual accepted native event is not a
human approval. A trusted and immutable Desktop build, protected service and
trusted worker remain prerequisites.

This is a substantive UX choice: it interrupts a task the first time it uses
Bash, and may not be appropriate for Alphonse's intended autonomy. A protected
native input surface could be another design, but its event provenance and
protection from same-user manipulation would need their own demonstration.
Normal OS accessibility or automation privileges must be included in the
threat model; a native dialog alone is not proof that a human clicked it.

Before a native confirmation could be used, the implementation would need to:

1. Persist the candidate's pre-dispatch pause, the selected action and its
   reservation without running Bash. The current owner worker claims authority
   before choosing actions and hides host Bash when no bridge is attached;
   an explicit pending-approval state and safe continuation are not present.
2. Render the complete request in a native, scrollable, non-executable view.
   Long text, Unicode direction controls and unusual characters must not hide
   or change what is authorized. Localized UI text must not depend on English
   keywords, and the view must not execute or forward its displayed content.
3. Bind the native decision to the exact actor, message ID, request and paused
   candidate. Changes, cancellation, expiry or a different actor invalidate
   approval. A webview cannot supply or reuse an approval token; the native
   client must not become a general service-signing or forwarding interface.
4. Persist the message ID before service submission and handle lost replies,
   queue/ledger crash windows, cancellation and uncertain started effects
   without blindly repeating a command. The current read-only inspector does
   not perform automatic cross-store recovery.
5. Verify with a protected, signed or pinned production-equivalent Desktop,
   worker and separate service UID: normal conversation without Bash; one Bash
   task with approval; decline; changed request; replay; restart; attempted
   JavaScript/IPC fabrication; and the applicable OS automation permissions.
   Tests must distinguish synthetic native callbacks from real owner input.

The private native Desktop client in `desktop/src-tauri/src/authority_client.rs`
is only a typed transport building block. It is deliberately not a Tauri
command, approval view, trusted input surface or normal owner-conversation
path. It checks a protected service UID before sending data; it cannot establish
who asked the Desktop to make the request.
