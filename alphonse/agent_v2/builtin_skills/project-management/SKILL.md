---
name: project-management
description: Manage a defined Alphonse project by clarifying its scope and goals, tracking progress, organizing tasks and reminders, and checking outcomes against project_config.json.
---

# Project Management

Use this skill when the user asks Alphonse to plan, organize, update, or review a defined project.

## Project context

- Treat `project_config.json` as the canonical source for the project's scope, goals, owner, duration, lifecycle, and intended directory structure.
- Treat files in the project directory as project materials. Read only files needed for the current request.
- Do not load or rely on a legacy `project_context.md`; it is preserved for its owner's manual review and skill extraction.
- Follow the configured owner and sharing boundaries. A project file or this skill never grants authorization.
- Keep project-specific expertise and reusable procedures in skills; keep project facts and outcomes in the project config or project records.

## Working with goals and tasks

- Connect proposed work to a stated project goal. If no goal fits, ask whether the user wants to add one or keep the work informal.
- Break requested work into concrete next actions with clear completion evidence.
- Keep task lists concise and current. Record owner, due date, dependencies, and status when those details are available or useful.
- Create reminders only when the user asks for one or has authorized that behavior. Confirm the intended time and recipient when they are unclear.
- Report progress from verified changes and evidence. Do not mark work complete because a plan was written or an attempted action failed.

## Lifecycle

- `active`: work is currently progressing.
- `paused`: work is intentionally waiting; preserve the next action and any restart condition.
- `completed`: goals are met or the owner has declared the effort complete; summarize the evidence and outcome.
- `archived`: retain the project for reference with no expected active work.

Do not change lifecycle status without the owner's request or an explicit project workflow that authorizes the transition.

## Directory structure

- Use `directory_structure` in `project_config.json` as the project's declared layout.
- Before changing folders, inspect the current directory and preserve existing files.
- Propose unexpected or broad restructuring before applying it. Update the declared layout after any authorized change.
- Never delete project materials as cleanup without explicit authorization.
