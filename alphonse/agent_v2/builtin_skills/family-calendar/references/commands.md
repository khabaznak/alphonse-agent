# Requests and evidence

Run `python3 scripts/google_calendar.py` with one JSON object on stdin; stdout is JSON,
stderr contains static error codes and failures exit nonzero. Native Bash supplies
`ALPHONSE_SKILL_BINDINGS`; never put credentials in shell text, requests or output.
Narrow the time window if native output is truncated; incomplete JSON is not complete
listing evidence. No script entry is registered. Read requests need only the `calendar` binding.

Examples use synthetic placeholders, not actual household records:

```json
{"operation":"list","calendar_id":"explicit-calendar-id","time_min":"2030-01-10T00:00:00-06:00","time_max":"2030-01-11T00:00:00-06:00"}
```

Replace `list` with `availability` for busy windows without event titles, or use
`{"operation":"get","calendar_id":"explicit-calendar-id","event_id":"explicit-event-id"}`
for readback. Requests do not resolve labels or people; setup/user clarification
must supply the exact ID. Results may contain private schedule data; restrict them
to the authorized conversation. Do not commit or export results.

Create (requires the `authorization` binding in addition to `calendar`):

```json
{"operation":"create","calendar_id":"explicit-calendar-id","operation_id":"stable-task-operation-0001","send_updates":"none","event":{"summary":"Synthetic validation","start":{"dateTime":"2030-01-10T10:00:00-06:00","timeZone":"America/Mexico_City"},"end":{"dateTime":"2030-01-10T10:30:00-06:00","timeZone":"America/Mexico_City"}}}
```

For update, change operation to `update`, include the explicit `event_id`, `etag`
from GET, and `recurrence_scope` of `event`, `instance` or `series`. Always include
both start/end even when changing a title, preserving the read-back times. Only
summary, description, location, start, end and recurrence may be written. Unspecified
fields are preserved by PATCH. Arrays replace existing values. A matched readback
on resume returns verified without another PATCH; changed fields with a stale ETag
fail. Creating an ID already used for different content fails readback rather than
creating another event. Never reuse an operation ID for a different logical create.

For all-day events use `{"date":"2030-01-10"}` and an exclusive end date such as
`{"date":"2030-01-11"}`. Recurrence accepts one bounded rule, e.g.
`"recurrence":["RRULE:FREQ=WEEKLY;COUNT=4"]`. DAILY, WEEKLY and MONTHLY with counts
1–999 are supported; arbitrary RRULEs, EXDATE and “following” splits are rejected.
Google expands instances using the supplied series IANA timezone. Listing returns
instance IDs for explicit single-occurrence edits. Removing recurrence or complex
series transformations are outside this initial package.

Google references: [events insert](https://developers.google.com/workspace/calendar/api/v3/reference/events/insert),
[events patch](https://developers.google.com/workspace/calendar/api/v3/reference/events/patch),
[events list](https://developers.google.com/workspace/calendar/api/v3/reference/events/list),
[free/busy](https://developers.google.com/workspace/calendar/api/v3/reference/freebusy/query).
Client-supplied hex event IDs are valid Google base32hex IDs; retain them across
retries. A timeout is uncertain, not evidence that Google rejected the write.
