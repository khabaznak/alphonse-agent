"""Deterministic A2UI v0.9.1 projection for trusted Alphonse UI facts.

The first catalog intentionally projects only persisted question interrupts. It
does not accept model-authored component trees or permit arbitrary actions.
"""

from __future__ import annotations

from typing import Any

from alphonse.agent_v2.core.questions import QuestionInterrupt


A2UI_VERSION = "v0.9.1"
ALPHONSE_DESKTOP_CATALOG_ID = "alphonse.desktop.catalog.v1"
SUPPORTED_COMPONENTS = frozenset({"Card", "Container", "Text", "Button", "ChoiceList", "TextInput", "Status"})


class A2UiAdapter:
    """Build a small, validated A2UI stream from known question records."""

    catalog_id = ALPHONSE_DESKTOP_CATALOG_ID

    def server_capabilities(self) -> dict[str, Any]:
        return {"supportedCatalogIds": [self.catalog_id], "inlineCatalogs": False}

    def question_opened(self, question: QuestionInterrupt) -> list[dict[str, Any]]:
        surface_id = surface_id_for_question(question.question_id)
        components = _question_components(question)
        data_model = {
            "question": {
                "id": question.question_id,
                "kind": question.kind,
                "message": question.message,
                "choices": [choice.to_dict() for choice in question.choices],
            },
            "answer": {"text": ""},
        }
        messages = [
            {
                "version": A2UI_VERSION,
                "createSurface": {
                    "surfaceId": surface_id,
                    "catalogId": self.catalog_id,
                    "sendDataModel": False,
                },
            },
            {"version": A2UI_VERSION, "updateComponents": {"surfaceId": surface_id, "components": components}},
            {"version": A2UI_VERSION, "updateDataModel": {"surfaceId": surface_id, "path": "/", "value": data_model}},
        ]
        for message in messages:
            validate_envelope(message)
        return messages

    def question_closed(self, question_id: str) -> dict[str, Any]:
        message = {"version": A2UI_VERSION, "deleteSurface": {"surfaceId": surface_id_for_question(question_id)}}
        validate_envelope(message)
        return message

    def scheduled_task_created(self, task: dict[str, Any], *, project_name: str = "") -> list[dict[str, Any]]:
        """Project a persisted scheduled-task record into a trusted confirmation card."""
        task_id = str(task.get("scheduled_task_id") or "").strip()
        if not task_id:
            raise ValueError("scheduled_task_id_required")
        surface_id = f"scheduled-task:{task_id}"
        name = str(task.get("name") or "Scheduled task").strip()
        description = str(task.get("description") or "").strip()
        schedule = str(task.get("schedule_summary") or "Scheduled").strip()
        next_run = str(task.get("next_run_at") or "").strip()
        timezone = str(task.get("timezone") or "UTC").strip() or "UTC"
        details = [schedule]
        if next_run:
            details.append(f"Next: {next_run}")
        details.append(timezone)
        if project_name:
            details.append(f"Project: {project_name}")
        components: list[dict[str, Any]] = [
            {"id": "root", "component": "Card", "children": ["title", "name", "details", "view"]},
            {"id": "title", "component": "Status", "text": "Scheduled"},
            {"id": "name", "component": "Text", "text": name},
            {"id": "details", "component": "Text", "text": " · ".join(details)},
            {"id": "view", "component": "Button", "label": "View task", "action": {"name": "view_scheduled_task", "context": {"scheduled_task_id": task_id}}},
        ]
        if description:
            components[0]["children"].insert(2, "description")
            components.append({"id": "description", "component": "Text", "text": description})
        messages = [
            {"version": A2UI_VERSION, "createSurface": {"surfaceId": surface_id, "catalogId": self.catalog_id, "sendDataModel": False}},
            {"version": A2UI_VERSION, "updateComponents": {"surfaceId": surface_id, "components": components}},
            {"version": A2UI_VERSION, "updateDataModel": {"surfaceId": surface_id, "path": "/", "value": {"scheduled_task_id": task_id}}},
        ]
        for message in messages:
            validate_envelope(message)
        return messages


def surface_id_for_question(question_id: str) -> str:
    return f"question:{str(question_id or '').strip()}"


def question_id_from_surface(surface_id: str) -> str:
    value = str(surface_id or "").strip()
    return value.removeprefix("question:") if value.startswith("question:") else ""


def validate_envelope(message: dict[str, Any]) -> None:
    """Reject unsafe or malformed messages before a renderer can see them."""
    if not isinstance(message, dict) or message.get("version") != A2UI_VERSION:
        raise ValueError("a2ui_version_invalid")
    kinds = [key for key in ("createSurface", "updateComponents", "updateDataModel", "deleteSurface") if key in message]
    if len(kinds) != 1:
        raise ValueError("a2ui_envelope_kind_required")
    payload = message[kinds[0]]
    if not isinstance(payload, dict) or not str(payload.get("surfaceId") or "").strip():
        raise ValueError("a2ui_surface_id_required")
    if kinds[0] == "createSurface" and payload.get("catalogId") != ALPHONSE_DESKTOP_CATALOG_ID:
        raise ValueError("a2ui_catalog_not_supported")
    if kinds[0] == "updateComponents":
        components = payload.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError("a2ui_components_required")
        identifiers: set[str] = set()
        for component in components:
            if not isinstance(component, dict):
                raise ValueError("a2ui_component_invalid")
            component_id = str(component.get("id") or "").strip()
            kind = str(component.get("component") or "").strip()
            if not component_id or component_id in identifiers or kind not in SUPPORTED_COMPONENTS:
                raise ValueError("a2ui_component_not_allowed")
            identifiers.add(component_id)


def _question_components(question: QuestionInterrupt) -> list[dict[str, Any]]:
    question_id = question.question_id
    items: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "children": ["title", "message", "body", "cancel"]},
        {"id": "title", "component": "Status", "text": "Alphonse needs your input"},
        {"id": "message", "component": "Text", "text": question.message},
        {"id": "body", "component": "Container", "children": []},
        {
            "id": "cancel",
            "component": "Button",
            "label": "Cancel",
            "action": {"name": "cancel_question", "context": {"question_id": question_id}},
        },
    ]
    body = items[3]["children"]
    if question.kind == "yes_no":
        for answer, label in ((True, "Yes"), (False, "No")):
            component_id = "answer_yes" if answer else "answer_no"
            items.append(
                {
                    "id": component_id,
                    "component": "Button",
                    "label": label,
                    "action": {"name": "answer_question", "context": {"question_id": question_id, "answer": answer}},
                }
            )
            body.append(component_id)
    elif question.kind == "single_choice":
        choice_ids: list[str] = []
        for choice in question.choices:
            component_id = f"choice_{choice.id}"
            choice_ids.append(component_id)
            items.append(
                {
                    "id": component_id,
                    "component": "Button",
                    "label": choice.label,
                    "action": {"name": "answer_question", "context": {"question_id": question_id, "choice_id": choice.id}},
                }
            )
        items.append({"id": "choices", "component": "ChoiceList", "children": choice_ids})
        body.append("choices")
    else:
        items.extend(
            [
                {"id": "answer_text", "component": "TextInput", "label": "Your answer", "value": {"path": "/answer/text"}},
                {
                    "id": "submit",
                    "component": "Button",
                    "label": "Answer",
                    "action": {"name": "answer_question", "context": {"question_id": question_id}},
                },
            ]
        )
        body.extend(["answer_text", "submit"])
    return items
