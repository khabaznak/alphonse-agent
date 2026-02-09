# Terminal Tool Plan

## Goal
Allow Alphonse to execute safe, user-approved shell commands within sandboxed directories.

## Requirements
- Auto-approve read-only commands.
- Explicit approval for write/dangerous commands.
- Sandbox directories stored in nerve-db (per user/workspace).
- Commands run synchronously for now.
- Full audit trail (command, output, exit code, approvals).

## Concepts

### Tool: `terminal`
Runs a single command in a sandbox root. The tool itself never selects a sandbox.

### Ability: `terminal.run_command`
Collects:
- `sandbox_id`
- `command`
- `cwd`

Steps:
1. Validate the sandbox exists and is active.
2. Parse command type (read-only vs write/dangerous).
3. If read-only, auto-approve and execute.
4. If write/dangerous, request approval.
5. Execute and log results.

## Policy (initial)

### Auto-approve
- `ls`, `pwd`, `rg`, `cat`, `head`, `tail`, `stat`

### Require approval
- `touch`, `mkdir`, `cp`, `mv`, `rm`

### Always reject
- `sudo`, `curl | sh`, package installs, `rm -rf /`

## Data Model (nerve-db)

### `terminal_sandboxes`
- `sandbox_id`
- `owner_principal_id`
- `label`
- `path`
- `is_active`
- `created_at`, `updated_at`

### `terminal_sessions`
- `session_id`
- `principal_id`
- `sandbox_id`
- `status` (`pending|approved|rejected|executed|failed`)
- `created_at`, `updated_at`

### `terminal_commands`
- `command_id`
- `session_id`
- `command`
- `cwd`
- `status`
- `stdout`
- `stderr`
- `exit_code`
- `requested_by`
- `approved_by`
- `created_at`, `updated_at`

## API Endpoints

### Sandboxes
- `GET /agent/terminal/sandboxes`
- `GET /agent/terminal/sandboxes/{sandbox_id}`
- `POST /agent/terminal/sandboxes`
- `PATCH /agent/terminal/sandboxes/{sandbox_id}`
- `DELETE /agent/terminal/sandboxes/{sandbox_id}`

### Commands + Approval
- `POST /agent/terminal/commands`
- `GET /agent/terminal/commands`
- `GET /agent/terminal/commands/{command_id}`
- `POST /agent/terminal/commands/{command_id}/approve`
- `POST /agent/terminal/commands/{command_id}/reject`
- `POST /agent/terminal/commands/{command_id}/finalize`

## UI Flow

1. User requests a file action.
2. Alphonse proposes command + sandbox.
3. UI shows approval prompt if needed.
4. Upon approval, run and show output.

## Notes
- Sandbox ownership should be per principal (admin or user).
- Multiple sandboxes are allowed per principal.
- Channel access should not override sandbox permissions.
