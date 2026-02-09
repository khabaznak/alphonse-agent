# Ability Runtime Plan (JSON-Driven)

## Goal
Replace hardcoded flow logic with a JSON-based ability runtime. Abilities are data. The runtime handles parameter exploration, execution, and outputs uniformly. No parallel legacy path.

## Why change
- Current implementation hardcodes flows in `handle_incoming_message.py` and `cortex/graph.py`.
- Each new ability adds bespoke code and prompt keys.
- Slot filling is per-step and not contextual, causing repetitive prompts.
- The runtime should parse a full message and gather parameters before executing steps.

## Target Ability Spec (Draft)
```yaml
name: "Add User"
description: "Onboard a secondary user introduced by the admin"
intent_name: "core.onboarding.add_user"
input_parameters:
  - name: display_name
    type: person_name
    description: "Name of the new user"
    required: true
    order: 1
  - name: relationship
    type: relationship
    description: "Relationship of the new user to the admin"
    required: true
    order: 2
  - name: role
    type: role
    description: "Role in the household"
    required: false
    order: 3
  - name: channel
    type: channel
    description: "Preferred channel to link the user"
    required: false
    order: 4
step_sequence:
  - order: 1
    action: "ensure_user_record"
    input: [display_name, relationship, role]
  - order: 2
    action: "request_channel_link"
    input: [channel]
  - order: 3
    action: "ability.invoke"
    ability_intent: "onboarding.location.set_home"
    input: [address_text]
outputs:
  success:
    response_key: "core.onboarding.secondary.completed"
  need_more_info:
    response_key: "clarify.ability_parameters"
  failure:
    response_key: "error.execution_failed"
```

## Runtime Loop (Concept)
1. **Intent resolved → ability spec loaded**
2. **Parameter extraction**
   - Parse current message for any parameters in the spec.
   - Merge with previously collected state for the ability.
3. **Missing parameters**
   - If any required params missing: ask *contextual* question derived from spec + known values.
   - Store pending parameter requests in state.
4. **Execution mode**
   - If all required params collected: execute `step_sequence` in order.
   - Each step maps to a tool/action handler.
5. **Outputs**
   - Map result → output response keys.

## Data Model (Proposal)
Ability specs stored in nerve-db or JSON files (seed for core abilities).

**AbilitySpec fields**
- `intent_name`, `name`, `description`
- `input_parameters[]`
- `step_sequence[]`
- `outputs{success, need_more_info, failure}`

**Execution state**
- `ability_state`: { intent, params_collected, pending_param, last_prompt, status }

## Parameter Handling
- Parameters are typed. Runtime chooses parse/extract strategy.
- If a user provides multiple params in one message, all are captured in one pass.
- Clarify prompts should include known context values.

## Tool/Action Mapping
- `step_sequence.action` maps to tool handler functions.
- Tools are registered in a centralized registry.
- Steps can also invoke another ability via `action: ability.invoke`, enabling composition and reuse.

## Migration Plan (No parallel paths)
1. **Create ability runtime skeleton**
2. **Move one ability first**: `core.onboarding.add_user`
3. **Remove old hardcoded flow for that ability**
4. **Move remaining abilities incrementally**
5. **Delete unused slot-fill logic when coverage is complete**

## Immediate Next Steps
1. Define AbilitySpec schema in code
2. Implement ability runtime loop
3. Replace secondary onboarding flow with JSON ability
4. Update tests for new runtime
