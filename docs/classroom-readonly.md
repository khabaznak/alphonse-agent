# Optional Classroom student reads — #120

The native `ClassroomAdapter` is lazy and read-only. Connecting an account starts
no monitoring. An active project check policy, retrieval window, authenticated
executor, course-scoped account grant and authorized recipients are required for
scheduled acquisition, using the existing scheduler, worker, evidence store and
private result events. No account or school has been contacted for this delivery.

## Setup and consent

Enable the Google Classroom API for the public OAuth client and obtain school
administrator approval where required. Use the existing managed Google PKCE
connection flow. The connected account must itself be an enrolled student in each
explicitly granted canonical course ID. This adapter intentionally supports student
access only, rather than broad teacher/admin access to other students' work.

`classroom_read` requests `classroom.coursework.me.readonly`;
`classroom_announcements_read` separately requests
`classroom.announcements.readonly`. Each also needs
`classroom.courses.readonly` and `classroom.rosters.readonly` for course metadata
and the connected student's own enrollment proof. All scope names have the prefix
`https://www.googleapis.com/auth/`. OpenID identifies the connection, and the
existing OAuth flow requests offline consent. No email/photo, guardian, Drive,
write, all-students coursework or separate submission scope is requested.

Create grants and skill account bindings with the chosen capability, exact course
IDs and intended recipients. A project may bind coursework and announcements as
separate sources when both are authorized. The supported query is `{"text":""}`:
this provider has no mail-like free-text search. Unsupported queries fail clearly.
The daemon dispatches by provider **and operation**, preserving Google mail reads.

Before each content page or direct read, GET course metadata and
`courses/<course-id>/students/me`. Only an ACTIVE course and successful enrollment
proof allow the subsequent content GET. Provider 403 (including school restrictions)
is `blocked/classroom_permission_denied`, 401 is
`blocked/classroom_reauth_required`, and missing enrollment is
`blocked/classroom_student_role_required`. Missing/inaccessible resources are
explicitly unavailable; transient/throttled transport is partial/unavailable.
No automatic retry, account switching or impersonation occurs inside the adapter.
Recheck grants and cancellation before and after every provider response.

A guardian link or a Classroom UI preview is not student API access. Do not connect
a child's account under a parent's identity, request domain-wide delegation, or
infer enrollment from guardian email. Guardian-only accounts must use parent
summary mail through the existing Gmail/Outlook adapters, with their own consent,
mail grants and retrieval policy. School restrictions cannot be bypassed by this
adapter. See Google's [user role guide](https://developers.google.com/workspace/classroom/guides/key-concepts/user-types)
and [student enrollment API](https://developers.google.com/workspace/classroom/reference/rest/v1/courses.students/get).

## Evidence and coverage

Published coursework and announcements only. Preserve provider course/item IDs,
kind, creation/update times, descriptions, due-date and due-time components and
maximum points as inert JSON inside the shared MIME evidence format. Source IDs
are `courseWork:<provider-id>` or `announcements:<provider-id>` within each course;
version is a deterministic hash of retained provider facts. Repeated evidence has
the same version; corrected facts get a new version without overwriting history.
The shared store adds account, owner, grant, project and observation provenance.
Missing due dates/times, maximum points and submission state remain null. A date
without a time remains date-only. Provider due-time components are UTC; no local
midnight is invented. Grading weight remains null: **30 points never means 30%**.
This adapter never reads submissions or grades, so it cannot assert completion.
If provider timestamps are absent, the required evidence timestamp is observation
time and `source_timestamp_known` is false; it is not a claimed provider revision
time. Version hashes exclude observation time.

Classroom has no coursework delta feed. `changes` repeats a paginated bounded
list refresh with the original baseline lower bound, filtering by provider update
or creation time; absent timestamps remain eligible as observed evidence. Cursors
bind version, operation, course set, connection, grant revision and query. Page
size and lower bound stay fixed across continuations. Each provider page consumes
one shared page allowance, with course/role probes charged to its byte/deadline
budget. Each course gets explicit pending/complete coverage. Budget exhaustion,
permission denial or expired cursors never mean no-match. Invalid continuation
tokens trigger the shared single bounded resync and coverage gap. List refreshes
cannot prove a deletion or an atomic snapshot across concurrent provider edits;
no tombstones or absence-based task completion are inferred. Attachments and
linked resources are unavailable and are never fetched with Classroom credentials.

## Validation

Use disposable synthetic wire/account/policy fixtures only:

```sh
python -m pytest -q tests/test_classroom_adapter.py tests/test_oauth_lifecycle.py \
  tests/test_source_acquisition.py tests/test_account_grants.py \
  tests/test_project_checks.py tests/test_project_check_scheduling.py
```

Fixtures cover enrollment before content, guardian-only and school-admin denial,
unknown fields, revoked consent/cancellation, budgets, separate announcement
consent, pagination, scoped cursor reset, repeated/corrected evidence and durable
crash recovery/publication. No provider writes, submissions, grade changes,
student impersonation or real-account access are included. Real-account rollout
still requires explicit owner consent and milestone readiness qualification.
