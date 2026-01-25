# Atrium — Architectural Decisions

This document records deliberate architectural and philosophical decisions
that shape Atrium.

It exists to preserve intent and prevent unintentional drift.

---

## Decision 001 — Local-First Architecture

**Decision:** Atrium is designed to operate locally by default.

**Rationale:**
Local operation preserves privacy, resilience, and household sovereignty.
Cloud services may be integrated selectively, but never required.

**Alternatives Considered:**
- Cloud-first assistant architecture
- Fully managed third-party platforms

**Status:** Accepted

---

## Decision 002 — Explicit Constitution

**Decision:** Rex is governed by an explicit, written constitution.

**Rationale:**
Behavioral clarity and trust are foundational.
An explicit charter prevents accidental overreach as capabilities expand.

**Alternatives Considered:**
- Implicit behavior through prompts only
- Hardcoded constraints without documentation

**Status:** Accepted

---

## Decision 003 — Rex as Presence, Not Tool

**Decision:** Rex is treated as a resident presence, not as a utility function.

**Rationale:**
This framing encourages restraint, continuity, and respectful interaction.
It avoids command-and-control dynamics.

**Alternatives Considered:**
- Task-oriented assistant model
- Stateless request/response bot

**Status:** Accepted

---

## Decision 004 — Ollama for Local Inference

**Decision:** Use Ollama as the local LLM runtime for Rex reasoning.

**Rationale:**
Ollama better leverages Apple Silicon performance and provides a reliable
local chat interface.

**Alternatives Considered:**
- LocalAI runtime
- Cloud-hosted inference

**Status:** Accepted
