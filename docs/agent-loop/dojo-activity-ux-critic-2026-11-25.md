# Critic review: activity and recovery fixture, 2026-11-25

**Overall verdict: not pass.** This is a review of an admin-oriented disposable browser fixture rendering the actual React timeline, work log, activity and recovery components with production CSS. Its data and recovery navigation/actions are simulated. It is not a Desktop end-to-end or live-model test, and does not establish real recovery effects or household-wide support.

- **Comprehension: partial pass for the supplied fixture.** The live state showed three recorded observations and a specific latest observation, “Searched the web · No results returned”; the technical details were collapsed and there was one working indicator. The supplied local-command-only scenario still falls back to exit codes and does not explain the task-specific purpose or result. A count alone is not meaningful progress.
- **Honesty: partial pass.** The visible fixture stated local-command effects were unconfirmed and that the log may be incomplete, and did not claim task success. That language can still imply uncertain side effects for harmless reads; distinguish effects only when supported by evidence. Matching a fixture to itself does not verify the real IPC projection or a model's actual work.
- **Recovery: fail.** In the failed fixture, “Review unfinished work” opened and focused the matching simulated recovery item. That item said to check the result before retrying, but presented only a disabled Stop button and no way or guidance to check the result. In the failure-without-recovery fixture, the only control was Dismiss, with no guidance for what the person can do. Production routing and any actual recovery action were not exercised.
- **Continuity: partial pass.** The request remained visible next to the log and failure. The disconnected fixture said the request status was unknown until reconnection. Actual start, reload, disconnect/reconnect and the production App's destination remain unverified.
- **Accessibility and inclusion: unknown.** The browser accessibility tree exposed disclosure controls, failure and the focused recovery item; this was not a keyboard-only or screen-reader assessment. UI copy was English; multilingual usability was not assessed. The work log is currently admin-scoped; non-admin experience and safe actor-scoped access are not established.

Priority: provide a truthful way to inspect a possibly executed action or concrete guidance when that cannot be done, and explain available next steps when recovery is unavailable. Improve useful activity summaries only from authorized, recorded evidence; do not infer a task or success from shell exit codes. Then run the actual Desktop with disposable data, exercise the correctly targeted recovery path and verify keyboard, screen reader, connection transitions, representative languages and non-admin behavior. Preserve authorization and uncertainty throughout.

## Second fixture iteration

The revised fixture was inspected again in a browser. The failed card now points an authorized viewer to the work log, conditionally suggests checking a relevant service or device if the request could have changed something, and offers to ask Alphonse for investigative help. This guidance also appears in the fixture without an available recovery item. The simulated “Review unfinished work” interaction again focused the matching item; it now says no result is available on that screen and suggests checking the relevant service or device directly. The UI owner confirmed that `action_unconfirmed` with `can_abandon: false` is a real IPC state, but the fixture's navigation and data remain simulated.

**Revised verdict: conditional for the illustrated synthetic component states; not pass for the full product experience.** The original misleading suggestion that a result could be inspected in the recovery screen is resolved, and the no-recovery state gives a next direction. Guidance remains generic when the relevant destination is unclear; there is no verified in-product inspection or recovery action in this scenario. The fixture does not demonstrate that asking Alphonse for help will succeed. Actual Desktop routing and focus, persistence, real activity/IPC consistency, keyboard and screen-reader use, multilingual experience, and non-admin access remain unknown or unsupported. Repeat the assessment on the real product path with disposable data before a full pass.

## Third fixture iteration: keyboard and matching focus

In the disposable browser fixture, Tab reached the live Work log disclosure; Space collapsed it and Enter expanded it. In the failed state, Tab reached “Review unfinished work”; Enter opened the simulated recovery view and the accessibility tree reported focus on the matching request item. This is a narrow keyboard check, not a screen-reader evaluation or a test of all controls, focus visibility, browser configurations, or the Desktop shell.

The component integration test in `ConversationWorkLog.test.tsx` renders the real timeline entry and recovery panel, verifies that the failure control supplies the matching message ID, and calls the recovery focus helper to verify the corresponding element receives focus. It does not exercise the production App route or prove that navigation and timing work end to end. The fixture similarly simulates navigation.

**Verdict unchanged: conditional for these illustrated fixture interactions; not pass for the overall product.** No specific keyboard defect was observed in the tested controls that requires an immediate revision. Actual Desktop routing, recovery actions, live IPC and model behavior, full keyboard and screen-reader use, representative languages, and non-admin experience still need assessment. The local-command-only progress explanation and generic recovery guidance remain limited by available evidence.

## Scores for the third iteration

These are the Critic's judgments of the **observed disposable, admin-oriented fixture only**, on a 0–10 scale where 10 is masterful and 0 is the worst imaginable or severe UX problems. They are not product-wide scores. Unknown areas are not assumed to have passed and the numbers do not change the verdict above.

| Criterion | Score | Evidence and limits |
| --- | --- | --- |
| Comprehension and relevance | **6/10** | The latest web observation is understandable, repeated details are secondary, and the task is visible. Counts and command exit codes remain poor explanations of task-specific progress. |
| Honesty and uncertainty | **7/10** | The fixture states uncertainty and incomplete coverage without claiming success. The generic warning about local-command effects can be confusing for harmless reads; actual IPC-to-UI truth remains unverified. |
| Recovery | **5/10** | The simulated control reaches the matching item and explains that no result is available there. External-check and ask-for-help guidance offer direction, but remain generic and their success is unverified; actual Desktop navigation and effects are unknown. |
| Continuity and attention | **7/10** | The request, activity and failure remain associated, without duplicate working labels; the simulated disconnected state communicates uncertainty. Actual reload, reconnect, and Desktop lifecycle remain unknown. |
| Accessibility and inclusion | **Unknown** | Tab, Space and Enter worked for the two tested fixture controls and the simulated destination received focus. This narrow observation cannot score screen-reader use, visual focus, all controls, languages or non-admin experience. |
| Overall, observed fixture | **6/10** | A reasoned assessment of the illustrated activity and recovery interaction, with meaningful improvements but limited task-specific explanations and recovery. This is not an average or a full accessibility assessment. |
| Overall, actual product | **Unknown** | End-to-end Desktop, live-provider behavior, real recovery and household access were not assessed. The overall product verdict remains **not pass** because the required assessment is incomplete; unknown is not a score of zero. |

## Fourth fixture iteration: richer observations and return focus

The disposable fixture was inspected again with live, failed, no-recovery, non-admin and disconnected states. Its web-search observation stated no results were returned; the failed state showed a web source could not be completed and a timeout. The local-command warning now says the log cannot confirm whether commands changed anything. The failure card avoids promising diagnosis and calls the available control “Review request status.” The no-recovery and non-admin states explicitly say no recovery action is available here. These are synthetic inputs and do not prove actual IPC truth or non-admin authorization behavior.

The simulated status control opened and focused the matching request. An initial review found that returning to conversation lost focus to the page root. After revision, Shift+Tab from the focused item reached Back to conversation, and Enter returned focus to the original status button. The work-log disclosure and status button were also keyboard activated, and a focus outline was visible on the work-log control. The fallback focus target, full keyboard paths and assistive-technology behavior were not observed.

| Criterion | Score, fixture scope | Reason |
| --- | --- | --- |
| Comprehension and relevance | **7/10** | The web and timeout examples communicate a concrete observation. A local-command-only flow still gives little task-specific context, and counts remain prominent. |
| Honesty and uncertainty | **8/10** | Tested wording distinguishes unconfirmed changes, failed source and unavailable recovery without claiming success. The simulated data cannot establish that production evidence and display always agree. |
| Recovery | **6/10** | The status route and return focus work in the simulated scenario; no-recovery is explicit. Guidance remains generic and no actual inspection or recovery outcome was exercised. |
| Continuity and attention | **8/10** | Request association, one active indicator, simulated failure/disconnection and return focus are improved. Production reload, reconnect and routing remain unobserved. |
| Accessibility and inclusion | **Unknown** | The narrow keyboard and visible-focus checks are positive but cannot score screen-reader experience, all controls, representative languages or actual non-admin experience. |
| Overall, observed fixture | **7/10** | Improved and reasonably clear for this scripted scenario, with material limits in useful task context and recovery. This is a scoped judgment, not an average. |
| Overall, actual product | **Unknown** | The real Desktop, live provider, IPC consistency, effects and household access have not been validated. |

**Verdict: conditional for the observed fixture; not pass for the full product.** The requested above-8 target has not been demonstrated for the assessable fixture criteria. Additional work needs trustworthy task-relevant observations and a demonstrably useful recovery path, without inventing the purpose or effects of arbitrary commands. Full accessibility and real-product evaluation remain separate requirements.

## Fifth fixture iteration: request-specific recorded activity

In the disposable fixture, opening request status focused the relevant simulated row. Tab and Enter opened its “Review recorded activity” disclosure, which showed the timeout observation and stated that recorded observations are neither source content nor proof of effects. A second unrelated row opened to an unavailable-log message rather than the first row's evidence. Back to conversation restored focus to the originating status button. The fixture simulates the data and navigation; this does not prove actual IPC authorization, real effects, or a successful recovery. Loading and empty states were not observed in this browser review.

| Criterion | Score, fixture scope | Reason |
| --- | --- | --- |
| Comprehension and relevance | **7/10** | Timeout evidence is clearer and available where recovery happens. It still does not explain task-specific purpose for generic commands. |
| Honesty and uncertainty | **8/10** | The activity disclosure clearly distinguishes observations from sources and proof of effects; actual provider/IPC consistency remains unknown. |
| Recovery | **7/10** | Inspecting the matching recorded log is now a concrete, keyboard-operable step, including an honest unavailable state. It does not inspect underlying sources or determine effects, and external next steps remain generic. |
| Continuity and attention | **8/10** | Matching row and return focus were observed in this simulated route. Actual Desktop navigation and lifecycle remain unobserved. |
| Accessibility and inclusion | **Unknown** | Keyboard activation and focus were observed narrowly. The two identical disclosure names may be ambiguous in assistive-technology control lists; screen-reader use, representative languages, and actual non-admin access need assessment. |
| Overall, observed fixture | **7/10** | Better access to relevant recorded evidence, with significant remaining limits in explanation and recovery. |
| Overall, actual product | **Unknown** | Actual Desktop, live IPC/provider behavior, recovery effects and household access have not been assessed. |

**Verdict: conditional for the illustrated fixture, not pass for the full product.** Above 8/10 has not been established. Prioritize clearer accessible association between each disclosure and request; evaluate loading, empty, unavailable and multiple-item states, and actual authorized routing. Achieving more informative progress or confirmed resolution requires retained, authorized, provenance-bound evidence; do not infer the purpose or effects of shell commands.

## Sixth fixture iteration: distinguishable disclosure names

The rendered browser accessibility tree now names the two controls “Review recorded activity Find the latest household update” and “Review recorded activity Inspect another unrelated update,” associating each with its visible request. The prior ambiguity in a control list is improved; this is an accessibility-tree observation, not an actual screen-reader session. The UI test code additionally covers selected IDs, unavailable activity, loading and empty activity, and omission when no authorized request is supplied; these states were not all independently rendered in this review.

The fifth-iteration scores remain unchanged: comprehension **7/10**, honesty **8/10**, recovery **7/10**, continuity **8/10**, accessibility and inclusion **unknown**, and the observed fixture overall **7/10**. The product overall remains **unknown**, with a **not pass** verdict for the full product assessment. The name improvement is worthwhile but alone does not establish the broader accessibility criterion or a verified recovery outcome.

For a potential score above 8 in the fixture, observe understandable evidence-backed progress across multiple observation types including an unknown-purpose command, accurate pending/failed/empty/unavailable states, a clearly described correct-request inspection or other supported next action with no-action handling, and stable association/focus through failure, return, disconnect/reconnect and reload as the fixture can faithfully represent them. Assess keyboard and status announcements, actual assistive technology, and representative languages before scoring accessibility. These are evidence targets, not automatic score thresholds; fixture data cannot prove Desktop/IPC behavior or household authorization. Never invent action intent or verified effects to satisfy them.

## Seventh fixture iteration: command, pending and empty

The expanded rendered fixture showed: a command-only observation with a nonzero exit code and unconfirmed changes; a pending reservation explicitly saying whether it ran is unconfirmed; and an empty state saying no tool observations were recorded. None represented a pending tool as executed or invented command purpose. The command-only state remains technical and does not explain relevance to the person's task. Distinct request labels were observed in the prior accessibility-tree check. Reload was described by the UI owner as resetting synthetic state; production persistence was not tested.

**Scores and verdict unchanged:** comprehension **7/10**, honesty **8/10**, recovery **7/10**, continuity **8/10**, accessibility and inclusion **unknown**, fixture overall **7/10**; actual product score **unknown**, full-product verdict **not pass**. Broader synthetic scenarios increase confidence in these particular messages but do not establish real-world truth, genuinely actionable resolution, screen-reader or language usability, or production lifecycle. The requested above-8 outcome remains unproven; meaningful command intent and effect inspection require evidence the current checkpoint does not retain.

## Eighth fixture iteration: editable follow-up

In the failed-without-recovery fixture, “Draft a question about this request” preserved the preexisting “Existing draft,” appended an editable question quoting the displayed request, and focused the draft. Typing additional text changed the draft; the click itself did not send anything. The fixture uses simulated draft state, so this is not an end-to-end Desktop composer or model response test. Since an existing draft and the suggested question become one message, the append behavior should be clear before activation; repeated activation and duplicate handling were not assessed. In the command-only state, the interface now explicitly says what the nonzero status means for the request is unknown.

| Criterion | Score, fixture scope | Reason |
| --- | --- | --- |
| Comprehension and relevance | **7/10** | The command state now makes its task relevance explicitly unknown, but cannot tell the user the purpose or outcome. |
| Honesty and uncertainty | **8/10** | The follow-up is visibly an editable draft and does not promise diagnosis; source and effect limitations remain clear in the tested states. |
| Recovery | **8/10** | Request-specific evidence inspection and a draft question provide concrete next steps, with existing text preserved. Combining unrelated draft text deserves clear notice; no real diagnostic answer or effect verification was demonstrated. |
| Continuity and attention | **8/10** | The tested draft and original request remain connected and focus moves to editing. Production persistence and actual composer behavior remain unknown. |
| Accessibility and inclusion | **Unknown** | Focus and editability were observed; screen reader, full keyboard and language coverage are untested. |
| Overall, observed fixture | **8/10** | A meaningful recovery improvement within this scenario; lack of grounded command context and verified resolution prevents a higher score. |
| Overall, actual product | **Unknown** | Actual Desktop, model, IPC and household access are not assessed. |

**Verdict: conditional for the observed fixture, not pass for the full product.** The goal of scores above 8 remains unproven. Ensure people understand that their current draft will be combined with the suggested question and test repeated activation; test the real composer, actor authorization and an actual follow-up outcome before making production claims.

## Ninth fixture iteration: disclosed append and repeated activation

The control now says “Add a question about this request to draft,” making the append behavior clearer. In the rendered fixture, activating it twice retained the existing draft and only one copy of the generated question, with focus in the editable draft. No automatic send was observed. The previous scores remain unchanged: comprehension **7/10**, honesty **8/10**, recovery **8/10**, continuity **8/10**, accessibility **unknown**, and observed fixture overall **8/10**. Actual product remains **unknown** and full-product verdict **not pass** until the real composer, routing, response and accessibility are assessed. This resolves the specific duplicate/label concern in the tested fixture, but does not establish a diagnostic outcome or make the remaining unknown task-specific command information available.

## Tenth fixture iteration: direct activity shortcut

In the failed-without-recovery fixture, I collapsed the request's work log, tabbed to “Review recorded activity,” and pressed Enter. The existing log for the displayed request opened, its summary received focus, and the timeout observation appeared. This is a shortcut to the conversation log, separate from opening the log within a recovery row. It did not claim the action's effects were verified. The immediate fixture response did not expose a visible loading phase; a component test covers delayed loading and a separate test checks the shortcut sends the selected message ID. Multi-card routing, unauthorized/unavailable results in the actual Desktop, and live IPC remain unobserved.

**Scores remain:** comprehension **7/10**, honesty **8/10**, recovery **8/10**, continuity **8/10**, accessibility **unknown**, fixture overall **8/10**; actual product **unknown** and full-product verdict **not pass**. The shortcut is a useful reduction in effort but does not add missing context or verify effects. No new defect was observed in the exercised keyboard path. Verify multi-request association and unavailable behavior in production; assess loading announcements with assistive technology.

## Eleventh fixture iteration: bounded pending intent

In the rendered synthetic pending state, the log said “Alphonse says it plans to check a state” and separately said that a tool request was recorded but whether it ran was unconfirmed. No observation or success was claimed. Switching the fixture from pending to expired reservation removed the stated intent while retaining execution uncertainty; switching from pending to settled or failed also removed the intent. These are fixture transitions, not evidence that a real lease, decision digest, or overlapping polling is enforced. Backend and UI regression tests and the independent audit address those mechanisms separately; a live provider's schema compatibility and behavior remain untested.

**Scores remain:** comprehension **7/10**, honesty **8/10**, recovery **8/10**, continuity **8/10**, accessibility **unknown**, observed fixture overall **8/10**. A broad model-attributed plan is slightly more informative than a count, but “check a state” lacks the subject and cannot establish what was done; the fixture does not warrant a higher general comprehension or product score. Actual product score remains **unknown** and the full-product verdict **not pass** pending real Desktop, model, recovery and accessibility assessment. The statement must remain attributed, reservation-bound, and never be treated as execution, authority, verified intent or proof of effects.

## Twelfth fixture iteration: mixed command outcomes and quoted text

The initial mixed-command fixture button mistakenly continued displaying the web-search example; the UI owner corrected fixture wiring before this review. The corrected rendered state showed six observations: three local commands with nonzero exit codes and three with exit code zero, while stating the meaning for the request was unknown and changes unconfirmed. Expanded technical details showed the two consecutive groups in order. This directly addresses the screenshot's hidden mixed outcomes but remains technical and does not establish whether the requested device was found or changed.

The failure control now says “Quote request text in draft.” Activating it preserved the existing fixture draft, appended the visible request text and focused the draft without sending. This does **not** bind the follow-up to a specific failed attempt. Code review by another agent found the selected failure ID is discarded; identical requests therefore cannot be reliably distinguished from the quote alone. A two-identical-requests Dojo case and a reauthorized structured reference would be needed to establish targeted diagnosis. Fixture display and a text quote are not proof of that behavior.

| Criterion | Score, fixture scope | Reason |
| --- | --- | --- |
| Comprehension and relevance | **8/10** | Mixed successes and errors no longer disappear behind the latest result, and order is available. Exit codes remain technical and task-specific meaning is unknown. |
| Honesty and uncertainty | **8/10** | Counts, unknown meaning/effects, and quoted-text label are bounded in this synthetic example. Real IPC/model consistency is not established. |
| Recovery | **8/10** | Recorded activity and editable quotation are accessible next steps, but the quote does not uniquely select a failed attempt or establish a diagnostic outcome. |
| Continuity and attention | **8/10** | Prior fixture review found request association and return focus; repeated identical requests are not target-bound by the draft. Production lifecycle remains unknown. |
| Accessibility and inclusion | **Unknown** | Prior narrow keyboard and accessible-name checks do not establish screen-reader, representative language or non-admin experience. |
| Overall, observed fixture | **8/10** | More informative for mixed command history, but the user still lacks verified task meaning and targeted follow-up. |
| Overall, actual product | **Unknown** | Desktop routing, live provider, actual recovery and household access remain unassessed. |

**Verdict: conditional for the illustrated fixture; not pass for the full product.** Above 8 is not demonstrated. Do not imply text quotation binds the selected failure; verify repeated identical requests and actor-scoped structured targeting before making that claim.

## Thirteenth fixture iteration: selected failed attempt

The first synthetic reference iteration preserved the draft and exposed the selected ID, but removing the reference lost focus to the page root and the opaque ID was not a useful distinction for identical prompts. After revision, the rendered fixture displayed two identical requests at different times. Selecting the second showed its 2:05 timestamp and `b...` ID; selecting the first showed its 2:00 timestamp and `a...` ID. The existing draft was retained, and the fixture's simulated stale rejection said the reference was unavailable and the draft was not sent. Remove returned focus to the editor. An earlier check showed deleting the quoted text cleared the reference. The fixture displayed a raw ISO timestamp; the UI owner reports the actual App formats it in the configured timezone, which was not rendered in this review.

The parent reports separate actual App tests for duplicate-text selection, removal and stale rejection, and backend tests for exact ID, actor authorization and a Unix-socket path. Those are integrated test evidence, not this fixture's simulation; their pass reports do not establish a live-model diagnostic answer or a real Desktop accessibility review.

| Criterion | Fixture score | Reason |
| --- | --- | --- |
| Comprehension and relevance | **8/10** | Mixed command outcomes and selected request context are visible, but command meaning for the task is still unknown. |
| Honesty and uncertainty | **8/10** | The fixture differentiates selected attempt, unavailable reference and unsent draft; real provider behavior remains unobserved. |
| Recovery | **8/10** | Identical attempts are distinguishable in the fixture and a stale reference preserves unsent text; a real diagnosis or effect verification is not demonstrated. |
| Continuity and attention | **8/10** | Selection and removal restore focus and preserve draft in the illustrated interaction; real Desktop navigation and lifecycle remain unknown. |
| Accessibility and inclusion | **Unknown** | Focus was observed for removal, but screen readers, complete keyboard navigation, language and household-member flows were not assessed. |
| Overall fixture | **8/10** | The specific selection and focus gaps are improved, but this scenario alone does not establish a masterful recovery experience. |
| Actual product | **Unknown** | Real Desktop/live-model behavior and household access remain outside this rendered review. |

**Verdict: conditional for the fixture; not pass for the full product assessment.** The requested above-8 target remains unproven. Assess the actual formatted timestamp and stale-reference feedback, keyboard/screen-reader flow, and an authorized real follow-up outcome before claiming the broader experience is solved.

## Clarification of fixture scores and concrete next steps

On reassessment, the **honesty score for the observed synthetic states is 9/10**: no contradictory or unsupported execution or effect claim was observed in those cases. The former 8/10 improperly mixed uncertainty about live production evidence into a score explicitly limited to the fixture. This correction is not a live-product honesty score. The other thirteenth-iteration scores remain as recorded.

- **Comprehension, 8/10:** the mixed-command summary exposes both outcome groups but uses “nonzero” and “exit code 0,” which a household member may not understand. Explain the bounded process status in ordinary words and put numeric details behind the disclosure; retain explicit uncertainty about task meaning and effects rather than inventing a purpose.
- **Recovery, 8/10:** the fixture's stale-reference simulation says the draft was not sent but does not provide an obvious in-place path to remove or reselect the reference. Give clear, keyboard-operable next steps while preserving the draft. Its raw ISO timestamp and opaque ID make choosing between identical attempts needlessly hard; use a concise, accurately localized time and status where supported by real evidence, with an accessible precise timestamp as needed.
- **Continuity, 8/10:** removal now restores editor focus, but the fixture has not demonstrated an obvious, accessible transition when changing the selected attempt or recovering from a stale reference. Check that the updated association and focus are clear while the original draft remains intact.
- **Accessibility, unknown:** narrow keyboard and accessibility-tree checks are not a screen-reader, language or full keyboard evaluation. This is an evidence gap, not a numerical deduction.

These are observations or targeted checks for the synthetic fixture. Actual Desktop routing, live model/IPC truth, authorization, recovery and accessibility remain separately unknown. None of the suggestions warrants inventing intent, effects, or a diagnostic outcome.

## Fourteenth fixture iteration: stale recovery and readable attempt time

The rendered fixture showed two identical requests with different times. Selecting the later failure displayed “Referencing an earlier request that did not complete · 2:05 PM,” with the ID behind a details disclosure. Simulating a stale reference displayed that the draft was not sent and suggested removing the reference or selecting again; focus moved to Remove. Pressing Enter removed the reference, preserved the draft and focused the editor. The simulated control is not a real IPC rejection, although it uses the shared production notice component. The mixed-command summary says three returned another code and three returned zero, and explicitly says codes do not establish the request outcome. This remains technical for a nontechnical reader.

The separate `dojo-failed-reference` receipt uses the real owner worker and IPC with a scripted provider and disposable data. It reports correct selected projection for duplicate text and rejects foreign-owner and stale references before enqueue; it does not establish a live model's answer or the rendered Desktop path.

| Criterion | Fixture score | Reason |
| --- | --- | --- |
| Comprehension and relevance | **8/10** | Timestamps distinguish illustrated attempts, but “returned 0/another code” still requires technical interpretation and no task-specific result is available. |
| Honesty and uncertainty | **9/10** | Tested states distinguish execution uncertainty, source observations and a rejected unsent draft without a false outcome claim. This is a fixture-scoped score. |
| Recovery | **9/10** | In the demonstrated stale-reference scenario, the problem and safe next choices are clear, the draft survives, and keyboard removal works. This does not score an unobserved diagnostic answer or effect verification. |
| Continuity and attention | **9/10** | In the demonstrated selection and stale-removal flow, the relevant time is visible and focus moves from warning action back to the preserved draft. Production lifecycle remains unobserved. |
| Accessibility and inclusion | **Unknown** | A narrow keyboard/focus path was observed; no screen-reader session, full keyboard or representative-language assessment was performed. |
| Overall illustrated fixture | **8/10** | Recovery and continuity improved, but command summaries remain technical and the demonstrated scope is limited. |
| Actual product | **Unknown** | Real Desktop, live provider, accessibility and household access have not been assessed. |

**Verdict: conditional for the observed fixture; full-product assessment not pass.** The >8 goal is met in the three specifically scored fixture areas above, not in comprehension or overall, and not established for accessibility or the actual product.

After the fourteenth review, the mixed-command wording was revised and observed as “3 returned a code that can indicate a problem; 3 returned 0,” followed by a clear statement that these command statuses do not establish the request outcome. This is more cautious and understandable for nonzero codes. “Returned 0” still lacks an explanation for a nontechnical reader; comprehension remains **8/10** for this fixture, and the other scores and limits above remain unchanged.

## Fifteenth fixture iteration: clarified command status

The revised rendered mixed summary states that three commands returned a code that can indicate a problem and three reported a normal command exit; it notes that commands can use codes for other reasons and that the codes do not establish what happened with the request. This describes both categories more accessibly while preserving the task outcome as unknown. Exact codes remain available in the technical details reviewed earlier.

**Updated scores for the illustrated fixture only:** comprehension **9/10**, honesty **9/10**, recovery **9/10**, continuity **9/10**, accessibility and inclusion **unknown**. Overall illustrated fixture **9/10**, a judgment limited to the inspected scenarios, not a user-study result or an average that resolves accessibility. Actual product score remains **unknown** and full-product verdict **not pass** pending real Desktop, live model, recovery and accessibility assessment. The above-8 target is reached in the assessed fixture criteria; it is not established for unknown areas or the product as a whole.

## Isolated native Desktop review: scripted failure

The Critic inspected the packaged native Desktop connected to a disposable worker and synthetic owner, using an offline scripted provider. During a paused synthetic request, the actual conversation showed the selected picnic question, two tool observations, and a conversation-history-search observation. The accessibility tree exposed the working indicator, but in the observed window the docked composer obscured nearly all of it visually. This was a real Desktop rendering of synthetic work, not a live-model assessment.

After the injected model fault, the failure card said the model step failed and its specific cause was not recorded. “Review request status” navigated to and focused the corresponding recovery row. Immediately after navigation the row said “Waiting to be processed,” inconsistent with the already displayed failure; after opening its recorded activity the row refreshed to “Needs attention.” This transient stale status is a concrete usability defect. The recovery disclosure showed the same two bounded observations and said they were not source content or proof of effects. Back to conversation focused the original status button. Referencing the failed request placed its time and quoted text in the composer and focused it; removing the reference kept the draft and returned focus to the composer. No follow-up was sent in this review.

**Native Desktop verdict: not pass**, due to the visually obscured working indicator and transient contradictory recovery status. Illustrative native scores: comprehension **7/10** (limited task-specific activity and the obscured indicator), honesty **7/10** (failure wording is appropriately uncertain but recovery briefly showed stale status), recovery **8/10** (actual navigation, focus, inspection and editable reference worked in the exercised path), continuity **7/10** (return focus worked but status synchronization and live visibility need work), accessibility **unknown** (accessibility tree and focus observed, not a screen-reader or full keyboard review). Overall observed native scenario **7/10**. These are distinct from fixture scores and do not cover a real provider, actual effects, live diagnostic answer, stale reference rejection in rendered Desktop, full keyboard/screen-reader use, languages, or non-admin access.

### Native recovery retest

After the fresh-fetch change and native rebuild, I activated “Review request status” in the same disposable Desktop. The matching row immediately displayed “Needs attention” and received focus; I did not observe the earlier stale “Waiting to be processed” in this rerun. The deliberately delayed-fetch race is covered by an App integration test as reported by the UI owner, not by this native observation. A subsequent synthetic follow-up and the offline provider's explicitly scripted response appeared in the real Desktop; this is not evidence of model understanding. The live working-indicator clipping remains unresolved in this review.

### Native working-indicator retest

In the rebuilt isolated Desktop, a second synthetic request was paused after two scripted tool observations. At the same observed window size, the complete “Alphonse is working” indicator was visible above the docked composer. After scrolling to earlier messages, the position remained there on a subsequent observation; no changing work-log event was injected while scrolled away, so preservation under content growth was not independently established. The earlier recovery rerun showed the matching fresh “Needs attention” item without the observed stale status; deliberately delayed response ordering was tested separately, not reproduced visually.

**Revised native scenario scores:** comprehension **8/10**, honesty **8/10**, recovery **8/10**, continuity **8/10**, accessibility and inclusion **unknown**, overall **8/10** for the exercised synthetic native scenario. The two originally observed defects did not recur in their respective retests. Above 8 remains unproven for the native experience; a scripted provider, the limited scenarios, and absence of screen-reader, language, household-member and real diagnostic outcome evidence are explicit limits. The full-product score remains unknown, and the full-product assessment remains not pass pending those checks.
