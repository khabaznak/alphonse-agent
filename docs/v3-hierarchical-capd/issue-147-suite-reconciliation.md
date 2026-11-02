# Full-suite reconciliation — issue #147

Date: November 2, 2026. All work used an isolated worktree; the original
checkout and its unrelated changes were preserved. No real account access or
provider writes were used. Providers and tools are synthetic; filesystem fixtures
are disposable.

## Baseline and environment

Clean current main: `c8b0231a` (merge of PR #146). Interpreter:
`/Users/alex/Code Projects/Alphonse/.venv/bin/python`, Python 3.12.11,
macOS 26.6.2 arm64, pytest 8.3.4, setuptools 82.0.1, pip 26.1.2.
Command: `python -m pytest -q` using that interpreter from the clean worktree.
Result: **37 failed, 1302 passed, 37 skipped in 34.08s**.

All 37 node IDs below exactly match the original inventory. None was already
fixed on current main. The issue records 37/1273/37 on
`03e3d4c12ef760ee08f1197516f373e2e469873c` and 37/1302/37 on
`287c090f07249675279d6be25ae32aecc87bedfa`; those historical totals are supplied
evidence, not new executions in this task. Current main reproduces the latter
totals and failing node set.

## Intended contracts

- **Jev:** [PDCA migration](https://github.com/khabaznak/alphonse-agent/commit/4c0ed096610a61c307b114e14081cb369218fc4c),
  [Check](../../alphonse/agent_v2/core/intelligence/pdca/nodes/check_node.py),
  [Act](../../alphonse/agent_v2/core/intelligence/pdca/nodes/act_node.py) and
  [acceptance planning](../../alphonse/agent_v2/core/intelligence/acceptance_planning.py).
  Plan authors criteria; Jev reviews bounded evidence and recommends routes.
  Missing decisions fail closed; response preparation and valid evidence remain
  deterministic requirements. Historical optional-provider fallback wording in
  Stage 6 predates this migration and does not justify restoring LLM review.
- **Conversation/project:** [conversation migration](https://github.com/khabaznak/alphonse-agent/commit/22c4ea2ad4df5b14f3065aaf8a9520d54a5f33bf),
  [context migration](https://github.com/khabaznak/alphonse-agent/commit/fbf6dfff),
  [progressive context contract](03-progressive-tool-revealing.md),
  [ProjectStore](../../alphonse/agent_v2/core/projects.py) and
  [inbound routing](../../alphonse/agent_v2/services/project_sessions.py).
  Conversations need no home project; project scoping is explicit at ingestion.
  Config JSON supplies optional context; legacy Markdown is retained read-only.
  Shared visibility requires explicit membership. Inaccessible private projects
  return not-found rather than leaking ownership metadata.
- **Skills:** [standard skills](../standard-skills-contract.md) and
  [progressive reveal](03-progressive-tool-revealing.md). Skill scripts and retired
  artifacts never become independent tool-curation candidates.
- **Decision inputs:** [System One](../../alphonse/agent_v2/system_one.py) uses
  Score for message intake, then finalized curated context for admission.
  [Context assembly](../../alphonse/agent_v2/core/intelligence/v3/context.py)
  must preserve useful bounded durable memory for both decision systems.
- **IPC:** [client](../../alphonse/agent_v2/ipc.py),
  [progress projection](../../alphonse/agent_v2/core/intelligence/v3/projections.py) and
  [installed distribution test](../../tests/test_skill_distribution.py).
  System One validation allows 105 seconds; public progress uses display names,
  attempt numbers and action fields while redacting credentials and private reasoning.

## Named inventory and resolutions

Classifications below describe the reproduced failure, including secondary
regressions exposed after replacing obsolete fixtures. No environment prerequisite
caused an inventoried failure. No blanket skips, xfails or deleted coverage were used.

1. `tests/test_agent_v2_acceptance_contract.py::test_interactive_completion_requires_prepared_response`

   **Fixture/test isolation defect.** Act lacked a synthetic Jev provider. Supply an explicit complete recommendation and retain the response-before-completion guard assertions (Jev contract).

2. `tests/test_agent_v2_do_node.py::test_silent_successful_bash_requires_model_authored_response_before_completion`

   **Fixture/test isolation defect.** Act lacked a synthetic Jev provider. Supply an explicit complete recommendation and retain the response-before-completion guard assertions (Jev contract).

3. `tests/test_agent_v2_ipc.py::test_system_one_settings_request_uses_validation_timeout`

   **Obsolete expectation.** Expected retired 35-second System One timeout; assert the current 105-second validation budget, leaving the model-provider timeout at 35 seconds (IPC contract).

4. `tests/test_agent_v2_ipc.py::test_v3_task_progress_projects_hierarchical_actions_without_private_reasoning`

   **Obsolete expectation.** Expected raw tool IDs or an older progress shape; assert display names, attempts and action fields while retaining exact sanitized arguments/results and secret exclusions (IPC contract).

5. `tests/test_agent_v2_ipc.py::test_desktop_task_progress_a2ui_renders_v3_capd_work_log`

   **Obsolete expectation.** Expected raw tool IDs or an older progress shape; assert display names, attempts and action fields while retaining exact sanitized arguments/results and secret exclusions (IPC contract).

6. `tests/test_agent_v2_ipc.py::test_desktop_project_ipc_uses_the_daemon_owned_store`

   **Obsolete expectation.** Used the retired Markdown write endpoint; exercise daemon-owned config write/read persistence instead (conversation/project contract).

7. `tests/test_agent_v2_memory_sessions.py::test_router_stamps_selected_session_before_queueing_and_supports_commands`

   **Obsolete expectation.** Implicit channel selection no longer attaches a project. Pass an explicit authorized project and retain session binding across TUI/Telegram and immutable queued-session assertions (conversation/project contract).

8. `tests/test_agent_v2_native_respond_tool.py::test_respond_descriptor_and_schema_are_visible_to_plan_prompt`

   **Obsolete expectation.** One wording assertion included the removed word “all”; retain native response descriptor/schema and Bash-operation prompt checks (IPC/prompt contract).

9. `tests/test_agent_v2_openai_codex_capd.py::test_act_node_uses_codex_provider_for_acceptance_criteria`

   **Obsolete expectation.** Act no longer authors criteria and Check no longer asks Codex to classify evidence. Exercise Codex criteria authoring in Plan, and Jev evidence review with an assertion that the Codex subprocess cannot run (Jev contract).

10. `tests/test_agent_v2_openai_codex_capd.py::test_check_node_uses_codex_provider_for_criteria_review`

   **Obsolete expectation.** Act no longer authors criteria and Check no longer asks Codex to classify evidence. Exercise Codex criteria authoring in Plan, and Jev evidence review with an assertion that the Codex subprocess cannot run (Jev contract).

11. `tests/test_agent_v2_pdca_graph.py::test_pdca_act_node_returns_task_state`

   **Fixture/test isolation defect.** Missing synthetic Act/Check decisions or a monkeypatch for a deleted LLM review helper. Supply opt-in decisions; retain execution counts, successful evidence, final response, terminal outcome and cycle-limit failure assertions (Jev contract).

12. `tests/test_agent_v2_pdca_graph.py::test_run_pdca_once_passes_context_tools_to_plan`

   **Fixture/test isolation defect.** Missing synthetic Act/Check decisions or a monkeypatch for a deleted LLM review helper. Supply opt-in decisions; retain execution counts, successful evidence, final response, terminal outcome and cycle-limit failure assertions (Jev contract).

13. `tests/test_agent_v2_pdca_graph.py::test_run_pdca_once_executes_planned_tool_and_returns_to_check`

   **Fixture/test isolation defect.** Missing synthetic Act/Check decisions or a monkeypatch for a deleted LLM review helper. Supply opt-in decisions; retain execution counts, successful evidence, final response, terminal outcome and cycle-limit failure assertions (Jev contract).

14. `tests/test_agent_v2_pdca_graph.py::test_run_pdca_once_marks_mission_success_after_check_completes_criteria`

   **Fixture/test isolation defect.** Missing synthetic Act/Check decisions or a monkeypatch for a deleted LLM review helper. Supply opt-in decisions; retain execution counts, successful evidence, final response, terminal outcome and cycle-limit failure assertions (Jev contract).

15. `tests/test_agent_v2_pdca_graph.py::test_act_route_after_wip_incomplete_criteria_can_continue_to_plan`

   **Fixture/test isolation defect.** Missing synthetic Act/Check decisions or a monkeypatch for a deleted LLM review helper. Supply opt-in decisions; retain execution counts, successful evidence, final response, terminal outcome and cycle-limit failure assertions (Jev contract).

16. `tests/test_agent_v2_pdca_processor.py::test_pdca_processor_runs_graph_and_returns_completed_result`

   **Obsolete expectation.** Unavailable acceptance planning must report failure, not successful completion; assert failure snapshots and cause. See the Jev contract.

17. `tests/test_agent_v2_pdca_processor.py::test_core_loop_can_use_pdca_processor_snapshot_metadata`

   **Obsolete expectation.** Unavailable acceptance planning must report failure, not successful completion; assert failure snapshots and cause. See the Jev contract.

18. `tests/test_agent_v2_pdca_processor.py::test_pdca_processor_snapshot_includes_planned_tool_call_when_present`

   **Fixture/test isolation defect.** Supply explicit synthetic decisions instead of a removed LLM helper; retain planned-call snapshots and verified terminal outcome assertions. See the Jev contract.

19. `tests/test_agent_v2_pdca_processor.py::test_pdca_processor_snapshot_includes_final_act_outcome`

   **Fixture/test isolation defect.** Supply explicit synthetic decisions instead of a removed LLM helper; retain planned-call snapshots and verified terminal outcome assertions. See the Jev contract.

20. `tests/test_agent_v2_projects.py::test_project_store_creates_directory_and_context_file`

   **Obsolete expectation.** Replace legacy Markdown writes with supported config writes and explicit context selection. Retain per-user/per-project sentinels and legacy write rejection. This exposes the renderer’s undefined administrator variable, fixed without granting additional access. See the conversation/project contract.

21. `tests/test_agent_v2_projects.py::test_project_store_filters_private_and_shared_projects`

   **Obsolete expectation.** Shared does not mean visible to every household member. Assert initial denial and explicitly grant membership before visibility checks. See the conversation/project contract.

22. `tests/test_agent_v2_projects.py::test_project_store_archives_restores_and_updates_metadata`

   **Obsolete expectation.** Inaccessible private-project edits return not-found; retain archive/restore and authorized metadata mutation checks. See the conversation/project contract.

23. `tests/test_agent_v2_projects.py::test_tui_project_selection_routes_future_messages`

   **Obsolete expectation.** Selected UI project does not implicitly scope ordinary conversations. Assert projectless queueing and retained selected-project config. See the conversation/project contract.

24. `tests/test_agent_v2_projects.py::test_prompt_render_paths_include_project_context`

   **Obsolete expectation.** Replace legacy Markdown writes with supported config writes and explicit context selection. Retain per-user/per-project sentinels and legacy write rejection. This exposes the renderer’s undefined administrator variable, fixed without granting additional access. See the conversation/project contract.

25. `tests/test_agent_v2_projects.py::test_capd_prompts_keep_user_and_project_contexts_isolated`

   **Obsolete expectation.** Replace legacy Markdown writes with supported config writes and explicit context selection. Retain per-user/per-project sentinels and legacy write rejection. This exposes the renderer’s undefined administrator variable, fixed without granting additional access. See the conversation/project contract.

26. `tests/test_agent_v2_projects.py::test_capd_prompt_uses_the_project_attached_to_each_new_task`

   **Obsolete expectation.** Replace legacy Markdown writes with supported config writes and explicit context selection. Retain per-user/per-project sentinels and legacy write rejection. This exposes the renderer’s undefined administrator variable, fixed without granting additional access. See the conversation/project contract.

27. `tests/test_agent_v2_system_one.py::test_jev_request_curation_receives_plan_instructions_and_bounded_context`

   **Obsolete expectation.** Assert retired artifact IDs are excluded while native tools and bounded request/project/memory context remain in the request. See skills and decision-input contracts.

28. `tests/test_agent_v2_system_one.py::test_jev_triages_each_plan_message_with_choice_and_relevance_fuse`

   **Obsolete expectation.** Replace Choice/Noul fixture responses with validated bounded Score responses and assert the current single question per message. See skills and decision-input contracts.

29. `tests/test_agent_v2_system_one.py::test_jev_admission_and_context_curation_share_one_evaluation`

   **Obsolete expectation.** Curation precedes admission. Assert two calls and that admission includes selected context while excluding rejected context. See skills and decision-input contracts.

30. `tests/test_agent_v2_users.py::test_non_admin_profile_projects_and_membership_are_isolated`

   **Obsolete expectation.** New projects create config JSON, not legacy context Markdown; private visibility still denies access after membership insertion until the owner explicitly shares the project. Retain unauthorized config-edit rejection and user profile/membership isolation assertions (conversation/project contract).

31. `tests/test_agent_v3_admission.py::test_direct_reply_receives_cross_project_history_global_and_selected_project_context`

   **Obsolete expectation.** Use real disposable config-backed project records; task-attached project context is still optional and explicitly selected by Jev. Retain cross-project authorized history, global invariants and rejected/legacy context exclusions (conversation/project contract).

32. `tests/test_agent_v3_admission.py::test_context_curation_separates_invariants_from_optional_candidates`

   **Obsolete expectation.** Use real disposable config-backed project records; task-attached project context is still optional and explicitly selected by Jev. Retain cross-project authorized history, global invariants and rejected/legacy context exclusions (conversation/project contract).

33. `tests/test_agent_v3_engine_rollout.py::test_v3_phase_planner_receives_project_context_and_durable_memory`

   **Obsolete expectation + product regression.** Replace legacy Markdown context writes and set explicit context selection. The supported fixture then reveals tail-only truncation dropping the authoritative journal path. Use the existing bounded head/tail renderer; preserve both the journal path and latest correction, truncation marker, read-only scope and native tool relevance (decision-input contract).

34. `tests/test_agent_v3_plan_intake.py::test_plan_intake_batches_eligible_messages_and_consumes_only_jev_selected_ids`

   **Obsolete expectation.** Jev judges all eligible authorized-user candidates, including independent-request candidates; assert both are presented but only the selected message is consumed (decision-input contract).

35. `tests/test_project_provenance_waiting.py::test_new_conversation_creates_protected_home_project`

   **Obsolete expectation.** Assert a normal conversation creates no project and queues projectless. See the conversation/project contract.

36. `tests/test_project_provenance_waiting.py::test_independent_message_receives_exact_turns_ahead_notice`

   **Obsolete expectation + product regression.** Remove implicit-home expectation. Preserve exact turns-ahead notice; fix the product gate that suppressed waiting notices whenever no project was attached. See the conversation/project contract.

37. `tests/test_skill_distribution.py::test_installed_distribution_includes_nested_skill_resources`

   **Fixture/test isolation defect.** Expected resources were read from the bytecode-polluted repository although the build source excluded bytecode. Derive expected contents from the filtered build input; filter copied fixture bytecode too; exercise clean and deliberately polluted disposable inputs and assert no packaged bytecode before imports. Preserve sdist→wheel→installed checks, nested binary/reference/script bytes, executable bit and installed-package provenance.

## Verification

Affected suites: `python -m pytest -q` with all 16 inventoried test modules plus
`tests/test_agent_v2_act_node.py` and `tests/test_agent_v2_check_node.py`:
**171 passed**. These include unavailable/ambiguous decisions, final-response
requirements, evidence review, project access and context isolation.

Tested commit: `6420ef473e47e4cf5c963968e852ce9dfbfa4f2b`.
Both checkouts were clean (`git status --porcelain` empty); final follow-up edits
only add this verification record.

- Fresh detached checkout `/private/tmp/alphonse-147-clean`:
  `python -m pytest -q -rs` → **1340 passed, 37 skipped in 34.21s**.
- Clean branch checkout `/private/tmp/alphonse-issue-147`, same commit:
  affected suites → **171 passed in 5.86s**, with this exact command:

```sh
python -m pytest -q \
  tests/test_agent_v2_acceptance_contract.py \
  tests/test_agent_v2_do_node.py \
  tests/test_agent_v2_ipc.py \
  tests/test_agent_v2_memory_sessions.py \
  tests/test_agent_v2_native_respond_tool.py \
  tests/test_agent_v2_openai_codex_capd.py \
  tests/test_agent_v2_pdca_graph.py \
  tests/test_agent_v2_pdca_processor.py \
  tests/test_agent_v2_projects.py \
  tests/test_agent_v2_system_one.py \
  tests/test_agent_v2_users.py \
  tests/test_agent_v3_admission.py \
  tests/test_agent_v3_engine_rollout.py \
  tests/test_agent_v3_plan_intake.py \
  tests/test_project_provenance_waiting.py \
  tests/test_skill_distribution.py \
  tests/test_agent_v2_act_node.py \
  tests/test_agent_v2_check_node.py
```

Use the interpreter path recorded above for both commands. Distribution tests
pass both first in isolation and after other suites, and include separate clean
and deliberately bytecode-polluted build-input cases. The extra parameterized
case explains the one-test increase beyond the 37 repaired tests.

The 37 skips are unchanged from baseline and explicitly optional:

- 34 parameter cases at `tests/test_agent_v2_skills.py:49` require the optional
  official `skills_ref.validator` conformance checkout.
- One case at `tests/test_agent_v2_skills.py:166` requires the same optional
  validator for the authoring template.
- Two cases at `tests/test_oauth_lifecycle.py:280` require optional
  `requests_oauthlib` for its real OAuth library contract. Synthetic adapter,
  grant and lifecycle tests still run without real accounts or provider writes.

No new skips/xfails were added. All 37 original failures are resolved through
supported-behavior replacement tests, explicit providers, or the three narrow
product fixes. No authorization, containment, cancellation, evidence or terminal
safeguard was relaxed. `git diff --check` also passes.
