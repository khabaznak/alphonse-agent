---
name: skill-authoring
description: Design, draft, review, and install reusable Alphonse skills that capture focused expertise, workflows, or behavior.
---

# Skill authoring

Use this skill when a user asks Alphonse to create reusable expertise or a workflow that should be available across conversations and projects.

## Decide whether it belongs in a skill

- A skill contains reusable instructions, domain expertise, communication preferences, or a repeatable workflow.
- A project contains project-specific facts, goals, constraints, owners, and progress. Keep those in the project configuration or project files.
- Artifacts belong to a skill. Registering one records its project-local entry point and schema, and adds its operating instructions to that skill's definition. Artifacts are not independently curated or invoked as tools; use the skill instructions and authorized native capabilities.
- A skill does not grant access to tools, files, users, or integrations.
- Keep each skill focused on a coherent use case. Prefer a small set of complementary skills over a broad persona that combines unrelated responsibilities.

## Gather the design

Clarify only the details needed to make the skill useful:

1. What requests should activate it, and what requests should not?
2. Who is it for, and what expertise or workflow should it provide?
3. What steps, checks, constraints, and output style should it follow?
4. What information should it ask for when missing, and when should it defer to another skill or the user?
5. What examples would make the behavior unambiguous?

Use the user's existing instructions and available context. Do not invent credentials, medical or legal authority, household facts, or capabilities. For safety-sensitive topics, describe appropriate limits and escalation to qualified professionals.

## Draft format

Prepare one `SKILL.md` package with YAML frontmatter containing:

```yaml
---
name: lowercase-kebab-case
description: A concise summary of the skill's expertise and when to use it.
---
```

Write focused instructions with useful headings, actionable steps, clear boundaries, and concise examples where helpful. The description should help Alphonse select the skill for relevant requests. Avoid duplicating project facts or instructions already owned by another skill.

## Review and install

- Present the complete proposed skill (name, description, and instructions) to the requester before installing it.
- Ask for explicit approval to install. A request to draft or discuss a skill is not approval to install.
- After approval, install the exact reviewed draft with `native.skill_install`. Do not silently alter the approved content during installation.
- Report the installed skill name and summarize where it can be used. If installation fails, explain the error and do not claim success.
- If the user asks only for a draft, provide the content without installing it.

Installed skills are instruction packages. They do not override Alphonse's policies, authorization checks, or deterministic controls.
