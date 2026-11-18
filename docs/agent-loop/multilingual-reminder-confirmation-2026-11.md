# Language-neutral reminder proposal boundary

The current conversational reminder mutation accepts only a narrow English
grammar. It does not support an ordinary multilingual reminder request, and
its English-only direct-request detection cannot reliably block an unverified
model claim in other languages. Do not expand that grammar with more phrase
lists or remove its checks and treat the model's interpretation as authority.

An actor-bound confirmation path is one possible design for a language-neutral
replacement, not a requirement chosen by the owner. The owner wants clear
requests to work without an extra confirmation. Any alternative must preserve
the authorization, timing, effect verification and recovery guarantees below.
This is an integration sketch, not a currently supported capability:

1. The model may propose a one-time local reminder with its content, due time,
   owner timezone, and the original request. A proposal does not create a
   reminder or establish that the owner intended the action. The host validates
   the format, current owner clock, timezone, bounds and source request; it
   rejects expired times and requires clarification for ambiguous local times.
2. If a confirmation is needed, persist the proposal and its conversational presentation atomically under
   the existing trusted owner claim and cancellation fence. Bind it to the
   owner, original request, exact proposed content, due time and a unique
   proposal/version. Preserve history and expose it as pending, never saved.
3. In that confirmation path, Desktop displays the precise proposed content, local date/time, timezone and
   delivery limitation, with explicit confirm and dismiss controls. A generic
   free-text response is not by itself a deterministic confirmation signal.
4. On a trusted owner confirmation, recheck active actor, exact proposal version,
   cancellation, due time and timezone inside the same serialized database
   transaction that records the confirmation and creates the reminder. A stale,
   changed, expired, dismissed or previously consumed proposal must not create
   another effect. Repeat submission must return an existing result only when
   it identifies exactly the same confirmed effect. Use the existing reminder
   and cancellation fencing; do not infer an effect from model output.
5. Publish a success only from the durable, verified creation result. Recovery
   must distinguish pending proposals, confirmed effects, expired proposals and
   uncertain delivery without replaying a side effect. Terminal, question and
   recovery publication guards must apply regardless of the user's language.

The full implementation also needs authorization and revocation tests, concurrent
confirm/dismiss and retry tests, crash tests on both sides of commit, elapsed and
daylight-saving cases, representative multilingual interpretation tests and
Desktop/IPC integration tests. All fixtures must use disposable databases and
fake inference. The current Desktop reminder checks every five seconds while
Desktop is active and connected; an offered-but-unacknowledged reminder remains
uncertain, and no external notification is guaranteed.
