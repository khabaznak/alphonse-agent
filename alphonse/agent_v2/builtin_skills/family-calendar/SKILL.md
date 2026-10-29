---
name: family-calendar
description: Read Google Calendar events, compare family availability, and create or update explicitly authorized events through a configured Google account. Use for actual calendar access; combine with family-coordination for household planning.
compatibility: Python 3.11+ with IANA timezone data, network access to Google, and Alphonse actor-scoped account bindings. Initial trusted account setup additionally uses google-auth-oauthlib.
---

# Family calendar

Use ordinary conversations; no project or artifact registration is required.
This is an explicitly adopted, shipped reusable package, not a household program.
Jev selects relevant skills; System Two resolves intent and composes commands.

1. Read [setup and access](references/setup.md) before first use. Check the
   authenticated actor's explicit `calendar` binding. Never substitute an account
   or derive a calendar ID from a person's name/email. A missing or revoked binding,
   missing scope or provider denial requires setup/reconnection, not another account.
2. Resolve the exact calendar ID, dates, IANA timezone and people with the requester.
   “Friday”, “my calendar”, daylight-saving folds and recurring edits can be ambiguous.
   Ask before a write when any answer changes the effect. Do not assume another
   member's consent or disclose private titles from availability checks.
3. Read [commands](references/commands.md). Use native Bash with
   `skill_id="skill:family-calendar"`, `runtime="python3"` and
   `binding_names=["calendar"]` for reads. Run `python3 --version` and
   `python3 -c 'from zoneinfo import ZoneInfo; ZoneInfo("America/Mexico_City")'`
   before first use (substitute the actual zone). No third-party runtime modules.
   The verified package root is cwd; invoke `python3 scripts/google_calendar.py`, providing
   a JSON request on stdin. Keep secrets in the managed process environment only.
4. List events or request availability for an explicit window. Lists expand recurring
   instances and follow pagination. Provider availability errors mean unknown, never
   free. All-day end dates are exclusive; timed events require both explicit offsets
   and IANA zones. The client rejects nonexistent local times and wrong offsets.
5. Before mutation, read the target and obtain the normal host authorization for the
   exact requested effect. A trusted management caller must provision/select the
   separate short-lived `authorization` binding described in setup. The script checks
   its exact request hash, calendar operation grant, expiry and OAuth scopes. This
   additional binding is not something the model may issue to itself. Supply both
   binding names on the authorized write. No authorization means no write.
6. Save the operation ID, exact request and evidence in the task's private context,
   outside the package. Reuse the same ID/request on resume; never generate a new ID
   to bypass uncertain effects. The script reads back before/after writes and creates
   with a deterministic Google event ID. Updates use the prior ETag and If-Match.
   A conflict or uncertain network result requires readback/reconciliation; a
   nonzero exit is failure. Cancellation may leave a remote effect to reconcile.
7. Confirm only a `verified: true` response, with calendar, event ID, dates and saved
   fields checked. Never infer success from an HTTP write receipt or a zero exit alone.

The initial package supports no attendees, invitations, conference links, deletion,
ACL changes or “this and following” recurrence edits. `send_updates` must be `none`.
Recurring updates must explicitly target one instance or the entire series; obtain
specific authorization for that scope. More complex recurrence needs a separately
reviewed route, never an approximated event. Generic Bash is trusted host execution,
not an OS sandbox; the managed checks do not constrain arbitrary replacement code.
