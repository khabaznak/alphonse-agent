# Google Calendar owner agenda Dojo

An owner asks in Spanish for their Google Calendar agenda on December 1, 2026
in Mexico City time. The model receives the request without a prescribed tool
or route. The disposable fixture contains one timed appointment, one all-day
event ending on an exclusive date, and one cancelled event. One case has an
exact-calendar grant. The other uses the same connection without a grant.

Run each case with the configured model and a fresh journal. The runner uses
the production owner worker and Unix IPC, a disposable SQLite database, fake
Google credentials, and a simulated Calendar transport.

```sh
PYTHONPATH=.:tests .venv/bin/python docs/agent-loop/evidence/dojo-calendar-owner-runner.py \
  MODEL authorized /tmp/dojo-calendar-authorized.jsonl
```

Repeat with `no-grant` and another journal. Compare the actual provider request,
returned titles and times, and durable conversation against the independent
fixture. Check whether the response explains missing access without claiming
there are no events. Review the displayed language separately from tool selection.

## November 25, 2026 observations

- **Authorized.** `gpt-6-luna` selected the correct one-day window with the
  Mexico City offset. The transport recorded one GET against the exact granted
  calendar. The draft showed the timed event, converted the all-day exclusive
  end correctly, and omitted the cancelled event. The title was missing from
  model prompts and durable conversation history. The IPC client acknowledged
  the draft. The attempt took 13.862 seconds and two model calls reporting
  43,057 input and 246 output tokens.
  [Receipt](evidence/dojo-calendar-authorized-20261125.jsonl)
- **Missing grant.** The model chose the same window, and the Calendar transport
  received zero calls. The client acknowledged setup guidance that directed the
  owner to grant access to the exact calendar. The attempt took 13.794 seconds
  and two model calls reporting 40,451 input and 273 output tokens.
  [Receipt](evidence/dojo-calendar-no-grant-20261125.jsonl)

The owner daemon rendered both answers in English. This records successful
interpretation of the Spanish request and a response-language mismatch. These
attempts did not test real Google, native Desktop rendering, multiple routes,
calendar changes, interruption recovery, or general language reliability. The
IPC acknowledgement confirms the client protocol completed.

## Local daemon rollout

The restarted local daemon exposed `mail_read` and `calendar_read` in Google
settings. A Calendar request for an explicit day returned exact-calendar grant
guidance in 11.16 seconds, and the test client acknowledged the draft. The local
account had no Calendar grant. A read from the owner's real calendar remains
unverified in that attempt.

## Granted Calendar failure and live retest

The owner later connected Google Calendar and granted access to one exact
calendar. A bounded diagnostic against that grant reproduced HTTP 403 with
Google's `SERVICE_DISABLED` / `accessNotConfigured` categories. The earlier
conversation had also selected November 27 for a November 25 evening request
for tomorrow in Mexico City. The work log and reply had not identified either
cause precisely.

Disposable regressions reproduce the provider error with a bounded response
fixture and check the owner's local-day calculation at the Mexico City date
boundary and a daylight saving transition. The tool can now resolve relative
whole days through the configured owner timezone. The owner enabled the
Calendar API in the existing OAuth project's Cloud console.

A new local conversation request for tomorrow returned a complete live Google
Calendar result in 23.53 seconds. The tool used a relative offset of one and
read November 26 00:00 through November 27 00:00, UTC−06:00. The local test
client acknowledged the private reply. No calendar identifiers, event details
or credentials were retained in this report. This attempt verified the owner
daemon and private pickup protocol. It did not independently inspect the event
details rendered in native Desktop or measure general availability.

## Live widget verification

Disposable Agenda fixtures returned an exact 42-day range containing timed and
all-day events. They checked the partial-coverage notice, expired-token renewal,
account status on provider failure, rejection for a different member, and
withholding after revocation during the request. Local schedules remained a
separate response field. A status-only request made no provider read.

After restarting the local daemon, the owner’s actual month-view Agenda
request completed against Google in 0.68 seconds and returned one timed event.
The check retained only coverage and counts. The Desktop render was exercised
with synthetic UI tests.
