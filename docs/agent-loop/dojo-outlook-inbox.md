# Personal Hotmail Inbox Dojo

The owner asks how many unread messages are in their Hotmail or Outlook.com
Inbox. Use a disposable owner database, a scripted Microsoft OAuth provider,
and a Graph response containing an independently known Inbox count. The model
can choose among the ordinary owner Desktop capabilities. No test requires a
particular wording or supplies the answer to the model.

Run the regression suite with `.venv/bin/python -m pytest -q tests/test_owner_outlook.py`.
Use this command to exercise a configured model with simulated Microsoft traffic.

```sh
PYTHONPATH=.:tests .venv/bin/python docs/agent-loop/evidence/dojo-outlook-inbox-runner.py \
  MODEL /tmp/outlook-inbox-receipt.jsonl
```

The runner requires an isolated Unix socket and inference access. It checks
the normal owner ingress, private pickup and acknowledgement protocol. Compare
the Inbox count and observation time with the fixture, inspect the actual
Microsoft paths, and check that missing or revoked access releases no count.
Inspect the model input and durable conversation history for the fixture markers.

## November 26, 2026 observations

- The [first real-model runs](evidence/dojo-outlook-inbox-20261126-initial.jsonl)
  used `gpt-6-luna`. The French request selected the right tool and returned the
  synthetic count of 7,149 after one Graph metadata request, but the reply was
  English. Missing and expired access produced distinct recovery states with
  zero Graph requests and no token refresh. A request for Archive returned a
  scope limitation and made zero Graph requests.
- The [final replay](evidence/dojo-outlook-inbox-20261126-qualified.jsonl) used
  the same model. The authorized French request produced a typed private
  receipt with the verified count, observation time and a French formatting
  locale. The model received none of those private values. The test client
  acknowledged the result after 11.239 seconds and two model calls with 29,907
  input and 94 output tokens. The French missing-grant request returned the
  deterministic recovery status and no provider request after 14.827 seconds
  and two calls with 29,900 input and 91 output tokens. Settings and recovery
  labels use Desktop’s English copy. Desktop formats the verified values locally.

The simulated provider runs exercise one bounded Inbox metadata request and the
owner conversation path. They supply no evidence about live Microsoft
registration, the provider's real callback and identity behavior, or arbitrary
mailbox reliability. [Readiness issue #123](https://github.com/khabaznak/alphonse-agent/issues/123)
tracks deployment qualification.

## Desktop UX review

The native Tauri window used the current Desktop components and an isolated
owner daemon. Microsoft sign-in handoff was intercepted and completed through
the normal loopback callback with scripted responses. The first cancelled
attempt rejected a late callback. A subsequent callback connected automatically.
On reconnect, a different account could not replace the saved identity. A
provider cancellation kept the reconnect action available.

In native Desktop, the grant offered a prefilled local expiration 30 days from
the test date. A normal composer request delivered the verified 7,149 in a
private Inbox result with the observation time. Revoking access made a new
request show setup guidance and a direct Settings action. Reloaded history
retained only the receipt for the earlier count. Both navigation focus
regressions were reproduced, fixed and retested in native WebKit. Opening
Conversation focused the composer for immediate typing. Returning through
the recovery action focused the Outlook heading even when its panel was
already mounted. Draft text survived the Settings round trip.

The Critic independently reviewed production components in Chromium at 350
and 670 px in light and dark. The grant editor distinguished the opener from the
confirmation action, 200% content zoom supported keyboard navigation, and an
unbroken 117-character account name wrapped without overflowing the account
row or editor. Native accessibility inspection confirmed names, roles, focus
and status text. The Critic scored comprehension 9/10, honesty and scope 10/10,
recovery 9/10, continuity 9/10, and accessibility and inclusion 8/10 pending
spoken VoiceOver testing. The overall score is 8/10 under the minimum-per-category
gate. Functional and visual review passed for the demonstrated fixture.
Full approval is pending the native spoken walkthrough.

### VoiceOver follow-up

With the owner's permission, the native walkthrough was repeated with macOS
VoiceOver enabled. System Settings verified the original off state, the on state
for testing, and the restored off state afterward. The disposable flow exposed
the named grant controls and save status, handed off to the composer, returned
the verified 7,149 with its observation time, and withheld a new count after
revocation. One recovery activation focused the Outlook heading. A later
activation in a conversation with older recovery buttons did not establish the
exact focus target.

The window capture did not expose spoken output or the caption panel. The
Critic kept the overall score at 8/10 and full approval pending direct
observation of the grant controls, save and revoke announcements, the result
and timestamp, and the Outlook heading after recovery. A fresh conversation
with one blocked request makes the recovery focus check unambiguous.
