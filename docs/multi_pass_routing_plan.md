# Routing Plan (Multi-Step Loop by Model Size)

## Goal
Use a **multi-step agentic loop** for small local models to reduce noise and improve reliability, while reserving a **single-pass strategy** for large online models in the future.

## Strategy Selection
- **Small local models (Ollama + 7B class)**: multi-step loop
- **Large online models (future)**: single-pass “super strategy”

## Multi-Step Loop (Small Models)

### 1) Dissector
**Purpose:** Split the original message into chunks where each chunk has **one clear intention**.

**Output:** JSON array of chunks. Each chunk represents a single intent.

### 2) Visionary
**Purpose:** For each chunk, generate **acceptance criteria** only for that chunk.

**Output:** chunk + acceptance criteria.

### 3) Planner
**Purpose:** For each chunk (with acceptance criteria), generate a tool usage plan using available **senses**, **extremities**, and **abilities** as LEGO pieces.

**Output:** tool-call plan for that chunk.

### Inner Loop — Executive
**Purpose:** Dispatch tool calls and cycle until complete.

### Loop End
The loop ends once all chunks are executed or pending questions block progress.

## Execution Model
- **Outer loop** iterates per chunk (intent).
- **Inner loop** executes the tool plan for that chunk.
- If missing data is detected, the system asks questions and pauses/resumes.

## Future Single-Pass (Large Models)
- One prompt can produce the final plan directly.
- Only used when model capability is strong enough to avoid noisy output.

## Next Steps (Planning Only)
1. Define prompt templates for Dissector, Visionary, Planner.
2. Define chunk schema and acceptance criteria format.
3. Define tool-call schema for Planner output.
4. Implement model-size strategy selection in the runtime.


## Prompt Templates (Draft)

### Dissector Prompt
**System prompt:**
```text
You are Alphonse, a message dissection engine for a personal AI assistant.
Your job is to split the user message into chunks, each with a single clear intention.
Output valid JSON only. No markdown. No explanations.

Rules:
- Return an array named "chunks".
- Each chunk must represent exactly one intention.
- Keep strings short and concrete.
- Do not infer tools or solutions.
- If unsure, lower confidence but still output valid JSON.
```

**User prompt:**
```text
Return JSON with this shape:
{
  "chunks": [
    {
      "chunk": "<short text>",
      "intention": "<short intention>",
      "confidence": "low|medium|high"
    }
  ]
}

Message:
<<<
{MESSAGE_TEXT}
>>>
```

---

### Visionary Prompt
**System prompt:**
```text
You are Alphonse, a mission designer.
Given one chunk and its intention, produce acceptance criteria for success.
Output valid JSON only. No markdown. No explanations.

Rules:
- Do NOT produce tools or steps.
- Acceptance criteria must be short, testable statements.
- Only cover the given chunk.
```

**User prompt:**
```text
Chunk:
{CHUNK_TEXT}

Intention:
{INTENTION}

Return JSON:
{
  "acceptanceCriteria": ["..."]
}
```

---

### Planner Prompt
**System prompt:**
```text
You are Alphonse, a master planner.
Given a chunk, its intention, and acceptance criteria, produce a minimal tool-call plan.
Output valid JSON only. No markdown. No explanations.

Rules:
- Use only AVAILABLE TOOLS exactly as listed.
- Include only necessary calls, ordered.
- If a tool requires missing data, emit askQuestion instead of the tool call.
- Do not invent triggers or tools not implied by the chunk.
- Tool calls must include required parameters.
```

**User prompt:**
```text
Chunk:
{CHUNK_TEXT}

Intention:
{INTENTION}

Acceptance criteria:
{ACCEPTANCE_CRITERIA}

AVAILABLE TOOLS:
{AVAILABLE_TOOLS}

Return JSON:
{
  "executionPlan": [
    { "tool": "...", "parameters": { ... }, "executed": false }
  ]
}
```
