"""Project execution receipts for Act. Shell syntax and process status are observations."""
from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
from copy import deepcopy

_NON_EFFECT = {"native.respond": "prepared_response", "native.ask_question": "question"}
_NATIVE_READBACK = {
    "native.exact_text_edit": "file_readback",
    "native.skill_install": "skill_store_readback",
    "native.scheduled_task": "schedule_store_readback",
}
_OUTBOX_TOOLS = {"native.deliver_message", "native.send_attachment"}
_ARTIFACT_TOOLS = {"native.skill_artifact_register", "native.artifact_metadata_update"}
# A deliberately narrow shell grammar. Quoted literals may contain SQL or data,
# while expansions, chained commands, and shell operators remain opaque.
_WORD = r'''(?:'[^\r\n']*'|"[^\r\n"`$\\]*"|[a-zA-Z0-9_./:=+-])+'''
_LITERAL_COMMAND = re.compile(rf"[ \t]*{_WORD}(?:[ \t]+{_WORD})*[ \t]*")
_REDIRECT = re.compile(rf"[ \t]*(?P<args>printf(?:[ \t]+{_WORD})+)[ \t]+(?P<operator>>>?)[ \t]*(?P<path>{_WORD})[ \t]*")
_PATH = re.compile(r"[\w./ -]{1,512}", re.ASCII)
_SQL = re.compile(
    r"(SELECT\s+(?P<columns>[a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)\s+FROM"
    r"|INSERT\s+INTO|UPDATE|DELETE\s+FROM)"
    r"\s+(?P<relation>[a-zA-Z_]\w*)\b",
    re.IGNORECASE | re.DOTALL,
)
_HTTP_FAILURE = re.compile(
    r'\bHTTP(?:\s*Error|/\d+(?:\.\d+)?)?\s*[: ]\s*([45]\d\d)\b'
    r'|["\']?(?:http_status|status_code)["\']?\s*[:=]\s*["\']?([45]\d\d)\b'
    r'|^\s*([45]\d\d)\s+(?:Forbidden|Unauthorized|Not Found|Bad Request|Internal Server Error)\b',
    re.IGNORECASE | re.MULTILINE,
)
_DENIED = re.compile(r"\b(?:permission denied|access denied|forbidden|unauthorized)\b", re.IGNORECASE)
_RESULT_PRIVATE_KEYS = {"authorization", "api_key", "access_token", "refresh_token", "password",
                        "secret", "cookie", "headers", "delivery_channels"}


def _shell_operation(arguments, result):
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return {"kind": "unknown", "argument_coverage": "unavailable"}
    operation = {
        "kind": "unknown", "argument_coverage": "command_withheld",
        "command_digest": hashlib.sha256(command.encode()).hexdigest(),
    }
    artifact_id = arguments.get("artifact_id")
    if isinstance(artifact_id, str) and artifact_id.startswith("artifact.") and _PATH.fullmatch(artifact_id):
        operation["artifact_id"] = artifact_id
    # Recognize only single invocations with literal paths. Shell expansions,
    # scripts, pipelines and control flow remain opaque.
    if len(command) > 4096:
        return {**operation, "argument_coverage": "command_exceeds_projection_limit"}
    redirect = _REDIRECT.fullmatch(command)
    if not redirect and not _LITERAL_COMMAND.fullmatch(command):
        return operation
    argv = shlex.split(command) if not redirect else []
    verb = argv[0] if argv else ""
    path, relation, columns = "", None, None
    if redirect:
        path = shlex.split(redirect["path"])[0]
        kind = "mutation"
        verb = "append_file" if redirect["operator"] == ">>" else "write_file"
    elif verb in {"cat", "ls", "stat", "touch", "mkdir", "rm", "rmdir"} and len(argv) == 2:
        path = argv[1]
        kind = "inspection" if verb in {"cat", "ls", "stat"} else "mutation"
    elif verb == "sqlite3":
        read_only = len(argv) == 4 and argv[1] == "-readonly"
        if not read_only and len(argv) != 3:
            return operation
        path, query = argv[-2:]
        query = query.strip().removesuffix(";")
        match = _SQL.match(query) if ";" not in query else None
        if not match:
            return operation
        action = match.group(1)
        relation, columns = match.group("relation", "columns")
        is_select = action.upper().startswith("SELECT")
        if is_select and not re.fullmatch(
            r"""(?:\s+WHERE\s+[a-zA-Z_]\w*\s*=\s*(?:-?\d+|'(?:[^']|'')*'))?\s*""",
            query[match.end():], flags=re.IGNORECASE,
        ):
            return operation
        if not is_select and read_only:
            return operation
        kind = "inspection" if is_select and read_only else "mutation" if not is_select else "unknown"
        verb = "sqlite_select" if is_select else "sqlite_" + action.split()[0].lower()
    else:
        return operation
    if not _PATH.fullmatch(path) or path.startswith("-"):
        return operation
    cwd = result.get("cwd")
    # Older receipts sometimes retained only the explicit working directory.
    cwd = cwd if isinstance(cwd, str) else arguments.get("cwd")
    if not posixpath.isabs(path) and (not isinstance(cwd, str) or not posixpath.isabs(cwd)):
        return {**operation, "kind": kind, "name": verb, "target_status": "working_directory_unavailable"}
    resource = {"kind": "sqlite_relation" if relation else "path",
                "path": posixpath.normpath(posixpath.join(cwd or "", path))}
    if relation:
        resource["relation"] = relation.casefold()
    if columns:
        operation["selected_columns"] = [column.strip() for column in columns.split(",")]
    return {**operation, "kind": kind, "name": verb, "target": resource,
            "argument_coverage": "literal_operation_only"}


def _native_target(tool_id, arguments, result):
    if tool_id in {"native.scheduled_task", "native.scheduled_task_delivery"}:
        kind, identity = "schedule", result.get("scheduled_task_id") or arguments.get("scheduled_task_id")
    elif tool_id in _OUTBOX_TOOLS:
        kind, identity = "recipient", result.get("recipient_user_id") or arguments.get("recipient")
    elif tool_id in _ARTIFACT_TOOLS:
        kind = "artifact"
        record = result.get("artifact")
        identity = record.get("artifact_id") if isinstance(record, dict) else arguments.get("artifact_id")
    elif tool_id == "native.skill_install":
        kind = "skill"
        skill = result.get("skill")
        identity = skill.get("id") if isinstance(skill, dict) else None
        identity = identity or arguments.get("name")
    elif tool_id in {"native.read_project_file", "native.exact_text_edit"}:
        kind, identity = "project_path", result.get("path") or arguments.get("path")
        paths = result.get("affected_paths")
        if tool_id == "native.exact_text_edit" and isinstance(paths, list) and len(paths) == 1:
            identity = paths[0]
    else:
        return None
    if isinstance(identity, str) and identity:
        return {"kind": kind, "id": identity}


def _native_verified(tool_id, result, target):
    verification = result.get("verification")
    if not target or not isinstance(verification, dict) or verification.get("status") != "verified":
        return False
    if tool_id == "native.exact_text_edit":
        digest = result.get("after_sha256")
        return (isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest) is not None
                and digest == verification.get("post_write_sha256"))
    return (tool_id in _NATIVE_READBACK and verification.get("source") == _NATIVE_READBACK[tool_id]
            and (tool_id != "native.skill_install" or isinstance(result.get("skill"), dict)))


def _native_commit(tool_id, result, operation):
    if operation["kind"] != "mutation" or not operation.get("target"):
        return None
    if tool_id in _OUTBOX_TOOLS and result.get("status") == "queued" and result.get("outbox_message_id") and result.get("recipient_user_id"):
        return "outbox_enqueue"
    if tool_id in _ARTIFACT_TOOLS and isinstance(result.get("artifact"), dict) and result["artifact"].get("artifact_id"):
        return "artifact_catalog_commit"
    if tool_id == "native.scheduled_task_delivery" and result.get("status") == "updated" and result.get("scheduled_task_id"):
        return "schedule_store_commit"
    return None


def _output_status(value, *, inspect_stdout=False):
    """Identify explicit transport failure facts without interpreting arbitrary prose."""
    streams = ("stdout", "stderr") if inspect_stdout else ("stderr",)
    for stream in streams:
        text = value.get(stream)
        if not isinstance(text, str):
            continue
        match = _HTTP_FAILURE.search(text)
        if match:
            status = int(next(group for group in match.groups() if group))
            return ("denied" if status in {401, 403} else "failed", status)
    status = value.get("status_code")
    if type(status) is int and 400 <= status < 600:
        return ("denied" if status in {401, 403} else "failed", status)
    return None, None


def _result_view(value):
    """Retain whole values and record producer clipping or credential field omissions."""
    if isinstance(value, dict):
        rendered, coverage = {}, set()
        for key, item in value.items():
            if str(key).lower().replace("-", "_") in _RESULT_PRIVATE_KEYS:
                coverage.add("private_fields_omitted")
                continue
            rendered[key], child_coverage = _result_view(item)
            coverage.update(child_coverage)
        if value.get("truncated") is True:
            coverage.add("source_truncated")
        return rendered, coverage
    if isinstance(value, list):
        children = [_result_view(item) for item in value]
        return [item for item, _ in children], set().union(*(flags for _, flags in children))
    clipped = isinstance(value, str) and value.endswith("... [truncated]")
    return value, {"source_truncated"} if clipped else set()


def observation(entry, criterion_ids, *, include_result=True):
    ref = entry["evidence_ref"]
    tool_id = entry.get("tool_id")
    status = entry.get("status")
    arguments = entry["arguments"] if isinstance(entry.get("arguments"), dict) else {}
    result = entry.get("result")
    fields = result if isinstance(result, dict) else {}
    row = {key: deepcopy(entry[key]) for key in ("evidence_ref", "phase_id", "subgoal_id", "tool_id", "status") if key in entry}
    row["provenance"] = {
        "source": ("tool_invocation" if entry.get("invocation") else
                   "checkpoint_action" if status in {"planned", "running"} else "checkpoint_evidence"),
        **deepcopy(entry.get("invocation") or {}),
    }
    row["criterion_ids"] = criterion_ids
    operation = {"kind": _NON_EFFECT.get(tool_id, "unknown")}
    if tool_id == "native.bash":
        operation = _shell_operation(arguments, fields)
    elif tool_id in _NATIVE_READBACK or tool_id in _OUTBOX_TOOLS or tool_id in _ARTIFACT_TOOLS or tool_id in {
        "native.read_project_file", "native.scheduled_task_delivery",
    }:
        inspection = tool_id == "native.read_project_file" or arguments.get("operation") == "list"
        operation = {"kind": "inspection" if inspection else "mutation"}
        if target := _native_target(tool_id, arguments, fields):
            operation["target"] = target
    elif (entry.get("invocation") or {}).get("read_only") is True and tool_id not in _NON_EFFECT:
        operation = {"kind": "inspection"}
    row["operation"] = operation

    error = str(entry.get("error") or "")
    error_code = entry.get("error_code")
    denied = error_code == "PermissionError" or error.startswith("PermissionError:") or _DENIED.search(str(fields.get("stderr") or ""))
    failure, http_status = _output_status(
        fields, inspect_stdout=tool_id == "native.bash" and operation["kind"] == "unknown",
    )
    native_delivery_failed = tool_id in _OUTBOX_TOOLS and fields.get("status") in {
        "recipient_not_found", "recipient_ambiguous", "recipient_unreachable", "delivery_failed",
    }
    if denied or failure == "denied":
        outcome = "denied"
    elif status == "rejected":
        outcome = "rejected"
    elif failure or native_delivery_failed or status == "failed" or fields.get("timed_out") is True or (
        tool_id == "native.bash" and type(fields.get("exit_code")) is int and fields["exit_code"] != 0
    ):
        outcome = "failed"
    elif status in {"planned", "running"}:
        outcome = "invocation_outcome_unavailable"
    elif status == "waiting":
        outcome = "waiting"
    elif status == "success":
        outcome = "process_exited_zero" if tool_id == "native.bash" and fields.get("exit_code") == 0 else "returned"
    else:
        outcome = "unavailable"
    row["operational_outcome"] = outcome
    if http_status is not None:
        row["http_status"] = http_status
    if error_code and isinstance(error_code, str) and re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]{0,63}", error_code):
        row["error_code"] = error_code
    elif error:
        row["error_detail_withheld"] = True

    rendered, coverage = _result_view(result)
    if result is None:
        availability = "unavailable"
    elif tool_id == "native.bash":
        availability = "available" if any(str(fields.get(key) or "").strip() for key in ("stdout", "stderr")) else "empty"
    else:
        if isinstance(result, dict):
            metadata = {"status", "success", "ok", "duration_ms", "command", "arguments"}
            substantive = any(key not in metadata and item not in (None, "", [], {}) for key, item in rendered.items())
        else:
            substantive = bool(result) or type(result) in {bool, int, float}
        availability = "available" if substantive else "empty"
    row["result_coverage"] = {"status": availability if include_result else "omitted",
                              "limitations": sorted(coverage)}
    if include_result and result is not None:
        row["result"] = rendered
        if tool_id in {"native.scheduled_task", "native.scheduled_task_delivery"}:
            channels = fields.get("delivery_channels")
            if isinstance(channels, list):
                row["delivery_provider_keys"] = sorted({
                    channel["provider_key"] for channel in channels if isinstance(channel, dict)
                    and isinstance(channel.get("provider_key"), str)
                })

    observed = include_result and availability == "available" and "source_truncated" not in coverage
    row["observation_kind"] = (
        "conversation" if tool_id in _NON_EFFECT else
        "operational_result" if observed and outcome in {"returned", "process_exited_zero"} else "attempt"
    )
    if operation["kind"] in _NON_EFFECT.values() or outcome == "rejected":
        effect_status = "none"
    elif outcome == "invocation_outcome_unavailable":
        effect_status = "unknown"
    elif outcome in {"failed", "denied"}:
        effect_status = "failed_attempt_possible_partial_effect" if operation["kind"] == "mutation" else "unverified"
    elif include_result and outcome == "returned" and _native_verified(tool_id, fields, operation.get("target")):
        effect_status = "verified_existing_state" if fields.get("created") is False else "verified_persisted_effect"
        row["verification"] = {"source": _NATIVE_READBACK[tool_id], "readback_ref": ref,
                               "target": deepcopy(operation["target"])}
    elif observed and outcome == "returned" and (commit := _native_commit(tool_id, fields, operation)):
        effect_status = "committed_receipt"
        row["effect_receipt"] = {"source": commit, "target": deepcopy(operation["target"])}
        if commit == "outbox_enqueue":
            row["submitted_content"] = {
                key: arguments[key] for key in ("message", "asset_id") if isinstance(arguments.get(key), str)
            }
    elif operation["kind"] == "inspection":
        effect_status = "none"
        if (observed and operation.get("target") and
                ((tool_id == "native.bash" and operation.get("name") in {"cat", "stat", "sqlite_select"}
                  and outcome == "process_exited_zero" and str(fields.get("stdout") or "").strip() and not fields.get("stderr"))
                 or (tool_id == "native.read_project_file" and outcome == "returned"))):
            row["verification"] = {"source": "direct_read", "readback_ref": ref,
                                   "target": deepcopy(operation["target"])}
    else:
        effect_status = "attempted_unverified" if operation["kind"] == "mutation" else "unknown"
    row["effect_status"] = effect_status
    if operation["kind"] == "mutation" and outcome != "rejected":
        row["effect_ref"] = ref + ":effect"
    return row


def link_readbacks(rows):
    """Tie a fresh read to the most recent matching write, if one was recorded."""
    mutations = {}
    for row in rows:
        operation = row["operation"]
        target = operation.get("target")
        if not target:
            continue
        identity = json.dumps(target, sort_keys=True)
        if operation["kind"] == "mutation" and row.get("effect_ref"):
            mutations[identity] = row
        if row.get("verification", {}).get("source") != "direct_read":
            continue
        if write := mutations.get(identity):
            row["verification"]["preceding_mutation_ref"] = write["evidence_ref"]
            row["verification"]["preceding_mutation_outcome"] = write["operational_outcome"]
            row["verification"]["effect_ref"] = write["effect_ref"]


def requested_effect_evidence_refs(review_state):
    """Require a resource-specific receipt cited by Check. Act judges exact fulfillment."""
    refs = set()
    for row in review_state["execution_evidence"]:
        verification = row.get("verification")
        if not row["criterion_ids"]:
            continue
        if row.get("effect_receipt") or (verification and verification["source"] != "direct_read"):
            refs.add(row["evidence_ref"])
            continue
        if not verification or verification.get("preceding_mutation_omitted"):
            continue
        preceding = verification.get("preceding_mutation_outcome")
        if preceding == "process_exited_zero" or (preceding is None and row.get("target_in_planned_scope")):
            refs.add(row["evidence_ref"])
    return refs


def has_unusable_criterion_support(review_state):
    """Reject criteria supported solely by failed or empty execution receipts."""
    supporting = {}
    for row in review_state["execution_evidence"]:
        if row["operation"]["kind"] in _NON_EFFECT.values():
            continue
        usable = (row["operational_outcome"] in {"returned", "process_exited_zero"}
                  and row["result_coverage"]["status"] == "available"
                  and ("source_truncated" not in row["result_coverage"]["limitations"]
                       or row["effect_status"] == "verified_persisted_effect"))
        for criterion_id in row["criterion_ids"]:
            supporting[criterion_id] = supporting.get(criterion_id, False) or usable
    return any(not usable for usable in supporting.values())


def has_requested_effect_evidence(review_state):
    return bool(requested_effect_evidence_refs(review_state))
