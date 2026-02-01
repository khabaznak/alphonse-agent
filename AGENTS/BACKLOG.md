# Backlog

## Intro

This backlog captures tasks to be accomplished as the project progresses.
Agents may look here to get a sense of what's next.
Each item should align with the PLAN.

## Tasks

1. [ ] Define initial event ingestion workflows for Alphonse.
2. [ ] Add monitoring of household sensors in a test mode.
3. [ ] Draft a permissions model for household members.
4. [ ] Add iOS push platform support alongside Android and web.
5. [ ] Plan A2A integration for Alphonse (LAN-only, respond-only).
6. [ ] Define Agent Card schema with identity fields (agent_id, name, version, owner, environment).
7. [ ] Define Agent Card handshake fields (a2a_endpoint, supported_protocols, auth_methods, public_key).
8. [ ] Define Agent Card capability fields (skills list, descriptions, IO summary, limits).
9. [ ] Define Agent Card operational context fields (availability, latency_profile, trust_level).
10. [ ] Add Settings-driven skills registry for A2A exposure.
11. [ ] Store A2A skills JSON in Settings (`a2a_skills`) and render in Agent Card.
12. [ ] Add A2A endpoints: `/a2a/agent-card` and `/a2a/tasks` on the existing FastAPI app.
13. [ ] Require `A2A_SHARED_SECRET` via `X-A2A-Secret` header for A2A requests.
14. [ ] Implement respond-only A2A task handling (no delegation).
15. [ ] Map A2A tasks to Signals → FSM → Actions → Intentions → (Narration) → Extremities.
16. [ ] Define task payload shape for skills (minimal, high-level input/output).
17. [ ] Decide defaults for agent_id, version, and owner fields.
18. [ ] Keep A2A discovery LAN-only and document exposure constraints.
19. [ ] Refactor to ensure a single nerve-db file exists in `alphonse/agent/nervous_system/db` (move from `alphonse/nervous_system/db`).
