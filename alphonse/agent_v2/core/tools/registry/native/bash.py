"""v2-native Bash tool."""

from __future__ import annotations

import os
import json
import selectors
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

BASH_TOOL_ID = "native.bash"
BASH_TOOL_NAME = "bash"
MAX_OUTPUT_CHARS = 12000
TERMINATION_GRACE_SECONDS = 0.25

BASH_ARGUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "command": {
            "type": "string",
            "description": "Shell command to execute with bash.",
        },
        "skill_id": {"type": "string", "description": "Optional selected standard skill ID. Resolve cwd from its verified installed package, independent of project. Exposes ALPHONSE_SKILL_ROOT; no script registration needed. Does not grant effects or sandbox the shell."},
        "runtime": {"type": "string", "description": "Optional documented executable: PATH name, absolute host path, or package-relative path (e.g. .venv/bin/python). Checks availability and exposes ALPHONSE_SKILL_RUNTIME; quote it in command. Does not install dependencies or automatically invoke it."},
        "required_files": {"type": "array", "items": {"type": "string"}, "description": "Optional documented prerequisite files. Relative to resolved cwd; absolute paths denote explicit host configuration. Check existence/readability before dispatch. No files are created."},
        "binding_names": {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True,
                          "description": "Explicit logical binding roles configured for this actor. Trusted scripts read ALPHONSE_SKILL_BINDINGS JSON from their process environment; never put credential values in commands."},
        "verification": {
            "type": "object", "additionalProperties": False,
            "description": "Optional independent harness read-back of the authorized resource before and after this command; never relies on the command's own success claim.",
            "properties": {
                "source": {"type": "string", "enum": ["skill_data_json", "workspace_text", "public_json"],
                           "description": "Independent actor-authorized binding, scoped file, or nonce-aware public receipt."},
                "binding_name": {"type": "string", "description": "Selected data binding also supplied in binding_names."},
                "path": {"type": "string", "description": "Scoped workspace-relative text file path."},
                "url": {"type": "string", "description": "Public JSON read-back URL on an origin explicitly named in the request; no redirects."},
                "pointer": {"type": "string", "description": "JSON Pointer to the requested value in a bound JSON file or public receipt."},
                "nonce_pointer": {"type": "string", "description": "JSON Pointer to the per-action nonce returned by a nonce-v1 public receipt."},
                "event_pointer": {"type": "string", "description": "JSON Pointer to the new resource-specific event ID in a nonce-v1 public receipt."},
                "expected": {"type": ["string", "number", "boolean", "null"], "maxLength": 4096,
                             "description": "Exact value to verify against the independent source."},
                "require_change": {"type": "boolean", "description": "Require the source value or public event ID to change; required for workspace and public receipts."},
            }, "required": ["source", "expected"],
        },
        "cwd": {
            "type": "string",
            "description": "Optional working directory. With skill_id, omit for package root or supply a contained package-relative directory; absolute cwd must also be within that package. Otherwise defaults to the authorized project or the active owner’s ordinary workspace.",
        },
        "timeout_seconds": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": (
                "Optional total timeout in seconds. Omit it to let the command run until it exits; "
                "set it only when the command itself warrants a deadline."
            ),
        },
    },
    "required": ["command"],
}


def build_bash_tool_definition() -> ToolDefinition:
    """Build the native Bash tool definition."""
    descriptor = ToolDescriptor(
        tool_id=BASH_TOOL_ID,
        name=BASH_TOOL_NAME,
        kind=ToolKind.NATIVE,
        description=(
            "Execute one Bash command on the local host and return stdout/stderr. Use Bash for local file and project work: "
            "search filenames and file contents with find, rg, or grep; read files with cat or sed; create and edit "
            "files with shell redirection or standard command-line editors; and verify changes by reading them back "
            'or comparing diffs. For independently verifiable effects, verification can read an authorized skill data JSON field, '
            'a scoped workspace text file, or a public JSON endpoint on a user-provided origin before and after the command. Data bindings require owner-configured alphonse_verification_authority=resource_state. '
            'Public confirmation requires the service to return the per-action ALPHONSE_VERIFICATION_NONCE, a new event ID, and the matching value. '
            'A data binding read-back establishes only the bound data, not that a cache represents a physical device. '
            'For standard package scripts supply skill_id, follow the documented command and '
            'interpreter, and quote paths/ALPHONSE_SKILL_ROOT/ALPHONSE_SKILL_RUNTIME. Check dependencies with the '
            'documented runtime (for example imports) before mutation; missing prerequisites require actionable setup, '
            'never an automatic install or repeated failing route. A zero exit alone does not verify a write: read back '
            'effects, and on resume reconcile prior or uncertain writes before repeating them. Bash is trusted host '
            'execution, not a filesystem sandbox; cwd does not restrict host files, network or environment. Without skill_id, a '
            'project-associated task defaults to that project directory; a task without a project defaults to its active owner’s private ordinary workspace. Never ask for a project just to create ordinary notes. '
            "For an ordinary user file lookup, search likely user-accessible locations such as the user's home folder, "
            "Downloads, or a path the user supplied; report the scope actually searched and ask when the intended "
            "location is unclear. Bash replaces dedicated project search, project file read, and exact text edit tools. "
            "Use it for direct filesystem, process, build, test, diagnostic, and skill script creation or repair work. "
            "When a dedicated tool is absent, use bounded runtime/dependency checks and small temporary programs "
            "within authorized scope to assess feasible execution paths. Users need not name a library. "
            "Capability assessment alone does not authorize installs, external submissions or schedules. "

            "For configured accounts/data, supply skill_id and binding_names. The process receives ALPHONSE_SKILL_BINDINGS JSON; "
            "credentials stay out of command arguments and known literal credentials are redacted from output. "
            "Bash is trusted host execution, not a filesystem or credential sandbox. "
            "Do not use Bash for multi-tool orchestration or aggregation of tool results."
        ),
        argument_schema=dict(BASH_ARGUMENT_SCHEMA),
        capabilities=("shell", "local_execution"),
        tags=("native", "shell"),
    )
    return ToolDefinition(
        descriptor=descriptor,
        callable=execute_bash,
        argument_schema=dict(BASH_ARGUMENT_SCHEMA),
        enabled=True,
        accepts_context=True,
    )


def execute_bash(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    """Execute a bash command and return a JSON-safe result."""
    from alphonse.agent_v2.skill_bindings import ExecutionBindings

    prepared = ExecutionBindings({}, ())
    if "binding_names" in arguments:
        skill_id = arguments.get("skill_id")
        names = arguments.get("binding_names")
        if not isinstance(skill_id, str) or not isinstance(names, list) or not names:
            raise ValueError("skill_binding_arguments_required")
        if context is None or context.skill_bindings is None:
            raise ValueError("skill_binding_context_required")
        selection = context.task.metadata.get("v3_skill_selection") or {}
        if skill_id not in selection.get("selected_skill_ids", []):
            raise PermissionError("skill_binding_skill_not_selected")
        prepared = context.skill_bindings.prepare(actor_id=str(context.task.user or ""), skill_id=skill_id, names=names)
    try:
        return prepared.redact(_execute_bash(arguments, context=context, prepared=prepared))
    except Exception as exc:
        if prepared.values:
            # Keep binding failures static, but retain actionable preflight codes
            # without echoing configuration paths or credential-bearing errors.
            code = str(exc).split(":", 1)[0]
            hints = {
                "bash_runtime_unavailable": "configure the documented executable/environment before retrying",
                "bash_required_file_unavailable": "provide the documented configuration/resource before retrying",
                "bash_skill_cwd_outside_package": "choose a working directory within the selected package",
                "bash_required_file_outside_package": "keep relative prerequisite resources within the package",
                "bash_runtime_outside_package": "use a contained environment or explicit host executable",
            }
            if code in hints:
                raise RuntimeError(f"skill_binding_execution_failed: {code}; {hints[code]}") from None
            raise RuntimeError("skill_binding_execution_failed") from None
        raise


def _execute_bash(arguments: dict[str, Any], *, context: ToolExecutionContext | None, prepared: Any) -> dict[str, Any]:
    def output(value: Any) -> str:
        return _truncate_output(prepared.redact(_coerce_output(value)))

    command = str(arguments.get("command") or "").strip()
    if not command:
        raise ValueError("bash_command_required")

    bash_bin = shutil.which("bash") or "/bin/bash"
    if not Path(bash_bin).exists():
        raise RuntimeError("bash_executable_missing")

    if "artifact_id" in arguments:
        raise ValueError("legacy_artifact_execution_retired: migrate resource and replan using skill commands")
    skill_id = str(arguments.get("skill_id") or "").strip()
    skill_root = _skill_root(skill_id, context) if skill_id else None
    if skill_root:
        requested = Path(str(arguments.get("cwd") or "."))
        resolved = (skill_root / requested).resolve()
        if ".." in requested.parts or not resolved.is_relative_to(skill_root):
            raise ValueError("bash_skill_cwd_outside_package: use a directory within the selected package")
        cwd = _resolve_cwd(str(resolved))
    else:
        cwd = _resolve_cwd(arguments.get("cwd"), context=context)
    environment = dict(os.environ)
    environment.pop("ALPHONSE_SKILL_BINDINGS", None)
    environment.pop("ALPHONSE_VERIFICATION_NONCE", None)
    if skill_root:
        environment["ALPHONSE_SKILL_ROOT"] = str(skill_root)
    runtime = _resolve_runtime(arguments.get("runtime"), skill_root or Path(cwd))
    if runtime:
        environment["ALPHONSE_SKILL_RUNTIME"] = runtime
    for filename in arguments.get("required_files") or []:
        path = Path(str(filename)).expanduser()
        if not path.is_absolute():
            path = Path(cwd) / path
            if skill_root and (".." in Path(str(filename)).parts or not path.resolve().is_relative_to(skill_root)):
                raise ValueError("bash_required_file_outside_package: relative resources must remain in the package")
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ValueError(f"bash_required_file_unavailable: {path}; provide the documented configuration/resource before retrying")
    if prepared.values:
        environment["ALPHONSE_SKILL_BINDINGS"] = json.dumps(prepared.values, ensure_ascii=True)
    timeout_seconds = _coerce_timeout(arguments.get("timeout_seconds"))
    cancelled = context.cancellation_checker if context else None
    if cancelled and cancelled():
        return {"exit_code": -1, "stdout": "", "stderr": "Command cancelled before dispatch.",
                "cancelled": True, "timed_out": False, "cwd": cwd, "timeout_seconds": timeout_seconds, "duration_ms": 0}
    verification = before = None
    if "verification" in arguments:
        from .bash_verification import prepare_verification, observe
        verification = prepare_verification(arguments["verification"], arguments, context, prepared, cwd)
        before = observe(verification, before=True, context=context)
        if verification['nonce']:
            environment['ALPHONSE_VERIFICATION_NONCE'] = verification['nonce']
        if cancelled and cancelled():
            return {"exit_code": -1, "stdout": "", "stderr": "Command cancelled before dispatch.",
                    "cancelled": True, "timed_out": False, "cwd": cwd, "timeout_seconds": timeout_seconds, "duration_ms": 0}
    started_at = time.monotonic()

    try:
        process = subprocess.Popen(
            [bash_bin, "-lc", command],
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = _collect_output_until_exit(process, timeout_seconds, cancelled)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _stop_process_group(process, exc)
        return {
            "exit_code": -1,
            "stdout": output(stdout),
            "stderr": output(stderr or ("Command cancelled." if isinstance(exc, _CommandCancelled) else f"Command timed out after {timeout_seconds:g} seconds.")),
            "timed_out": not isinstance(exc, _CommandCancelled),
            "cancelled": isinstance(exc, _CommandCancelled),
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
            "duration_ms": _duration_ms(started_at),
        }

    result = {
        "exit_code": int(process.returncode),
        "stdout": output(stdout),
        "stderr": output(stderr),
        "cancelled": False,
        "timed_out": False,
        "cwd": cwd,
        "timeout_seconds": timeout_seconds,
        "duration_ms": _duration_ms(started_at),
    }
    if verification is not None and result["exit_code"] == 0:
        try:
            from .bash_verification import finish_verification, observe, output_reports_failure
            if output_reports_failure(result['stdout'], result['stderr']):
                raise ValueError('verification_action_reported_failure')
            if cancelled and cancelled():
                raise ValueError('verification_cancelled')
            after = observe(verification, before=False, context=context)
            if cancelled and cancelled():
                raise ValueError('verification_cancelled')
            result['verification'] = finish_verification(verification, before, after, context)
        except Exception:
            # The command may already have changed a resource. Preserve its
            # evidence, fail closed, and never automatically replay the action.
            result['verification_failed'] = True
            result['verification'] = {'status': 'unavailable', 'source': 'bash_independent_readback'}
    return result


class _CommandCancelled(subprocess.TimeoutExpired):
    """Internal signal carrying captured output through process-group cleanup."""


def _skill_root(skill_id: str, context: ToolExecutionContext | None) -> Path:
    store = context.skill_store if context else None
    if store is None and context is not None and context.skill_bindings is not None:
        store = context.skill_bindings.skills
    if context is None or store is None:
        raise ValueError("bash_skill_context_required")
    if not str(context.task.user or "").strip():
        raise PermissionError("bash_skill_authenticated_actor_required")
    selection = context.task.metadata.get("v3_skill_selection") or {}
    if skill_id not in selection.get("selected_skill_ids", []):
        raise PermissionError("bash_skill_not_selected")
    from alphonse.agent_v2.skill_resources import require_skill, pin_skill
    skill = require_skill(context.task, store, skill_id)
    pin_skill(context.task, skill, context.artifact_resources)
    root = Path(skill.directory).resolve(strict=True)
    if not root.is_dir() or not (root / "SKILL.md").is_file():
        raise ValueError("bash_skill_package_unavailable")
    return root


def _resolve_runtime(value: Any, root: Path) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    value = str(value).strip()
    path = Path(value).expanduser()
    if path.is_absolute():
        candidate = str(path)
    elif "/" in value:
        candidate_path = root / path
        # Permit interpreter links in contained environments, not symlinked parents.
        if ".." in path.parts or not candidate_path.parent.resolve().is_relative_to(root):
            raise ValueError("bash_runtime_outside_package")
        candidate = str(candidate_path)
    else:
        candidate = shutil.which(value) or ""
    if not candidate or not Path(candidate).is_file() or not os.access(candidate, os.X_OK):
        raise ValueError(f"bash_runtime_unavailable: {value}; configure the documented executable/environment before retrying")
    return str(Path(candidate).absolute())


def _stop_process_group(
    process: subprocess.Popen[bytes],
    timeout_error: subprocess.TimeoutExpired,
) -> tuple[str, str]:
    """Stop the shell and every child that inherited its captured streams."""
    _signal_process_group(process.pid, signal.SIGTERM)
    try:
        stdout, stderr = process.communicate(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_process_group(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    else:
        if _process_group_exists(process.pid):
            _signal_process_group(process.pid, signal.SIGKILL)
    return (
        _merge_output(timeout_error.stdout, stdout),
        _merge_output(timeout_error.stderr, stderr),
    )


def _collect_output_until_exit(
    process: subprocess.Popen[bytes],
    timeout_seconds: float | None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[str, str]:
    """Read available output until the shell exits, without awaiting orphaned pipe writers."""
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    streams = {
        process.stdout: stdout_chunks,
        process.stderr: stderr_chunks,
    }
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds

    with selectors.DefaultSelector() as selector:
        for stream in streams:
            if stream is None:
                continue
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)

        while True:
            if cancelled and cancelled():
                raise _CommandCancelled(process.args, timeout_seconds, output=b"".join(stdout_chunks), stderr=b"".join(stderr_chunks))
            exited = process.poll() is not None
            if exited:
                wait_seconds = 0
            elif deadline is None:
                wait_seconds = 0.02
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        process.args,
                        timeout_seconds,
                        output=b"".join(stdout_chunks),
                        stderr=b"".join(stderr_chunks),
                    )
                wait_seconds = min(remaining, 0.02)
            ready = selector.select(wait_seconds)
            for key, _ in ready:
                stream = key.fileobj
                chunks = streams[stream]
                while True:
                    try:
                        chunk = os.read(stream.fileno(), 65_536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        selector.unregister(stream)
                        break
                    chunks.append(chunk)

            if exited:
                # The shell's own writes are buffered by now. Do not wait for EOF
                # from a background descendant that inherited either pipe.
                for stream, chunks in streams.items():
                    if stream is None:
                        continue
                    while True:
                        try:
                            chunk = os.read(stream.fileno(), 65_536)
                        except BlockingIOError:
                            break
                        if not chunk:
                            break
                        chunks.append(chunk)
                    stream.close()
                return (
                    _coerce_output(b"".join(stdout_chunks)),
                    _coerce_output(b"".join(stderr_chunks)),
                )


def _signal_process_group(process_group_id: int, requested_signal: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, requested_signal)
    except ProcessLookupError:
        return


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False


def _duration_ms(started_at: float) -> int:
    return max(0, round((time.monotonic() - started_at) * 1000))


def _merge_output(initial: Any, final: Any) -> str:
    return _coerce_output(initial) + _coerce_output(final)


def _resolve_cwd(raw_cwd: Any, *, context: ToolExecutionContext | None = None) -> str:
    if raw_cwd is None or str(raw_cwd).strip() == "":
        project_root = _project_root_from_context(context)
        if project_root:
            return project_root
        if context is None:
            return str(Path.cwd())  # Standalone host API compatibility only.
        if str(context.task.project_id or "").strip():
            raise ValueError("bash_project_directory_unavailable")
        resolver = getattr(context.identity_resolver, "ordinary_workspace", None)
        if not callable(resolver):
            raise ValueError("ordinary_workspace_store_unavailable")
        return str(resolver(str(context.task.user or "")))
    path = Path(str(raw_cwd)).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"bash_cwd_not_found: {path}")
    if not path.is_dir():
        raise ValueError(f"bash_cwd_not_directory: {path}")
    return str(path)


def _project_root_from_context(context: ToolExecutionContext | None) -> str:
    if context is None or context.project_store is None:
        return ""
    task = context.task
    project_id = str(getattr(task, "project_id", "") or "").strip()
    if not project_id:
        return ""
    get_project = getattr(context.project_store, "get_project", None)
    if get_project is None:
        return ""
    project = get_project(project_id, requester_user_id=getattr(task, "user", None))
    if project is None:
        return ""
    root = Path(str(project.root_path)).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return ""
    return str(root)


def _coerce_timeout(raw_timeout: Any) -> float | None:
    if raw_timeout is None or raw_timeout == "":
        return None
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("bash_timeout_invalid") from exc
    if timeout <= 0:
        raise ValueError("bash_timeout_must_be_positive")
    return timeout


def _coerce_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _truncate_output(value: Any) -> str:
    rendered = _coerce_output(value)
    if len(rendered) <= MAX_OUTPUT_CHARS:
        return rendered
    return f"{rendered[:MAX_OUTPUT_CHARS]}\n... [truncated]"
