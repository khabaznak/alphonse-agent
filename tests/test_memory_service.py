from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent.cognition.memory import MemoryService
from alphonse.agent.cognition.memory import TimeRange


def test_append_episode_writes_expected_markdown_shape(tmp_path: Path) -> None:
    service = MemoryService(root_dir=tmp_path / "memory")
    out = service.append_episode(
        user_id="alex",
        mission_id="leads_veloswim",
        event_type="web_research",
        payload={
            "intent": "Encontrar número de sucursales y ubicaciones",
            "action": 'web.search("VeloSwim sucursales")',
            "result": "4 sucursales (GDL) según fuente",
            "next": "buscar contacto (operaciones/gerencia)",
        },
    )
    path = Path(out["path"])
    text = path.read_text(encoding="utf-8")
    assert text.startswith("### ")
    assert "| mission: leads_veloswim | event: web_research" in text
    assert "- intent: Encontrar número de sucursales y ubicaciones" in text
    assert "- action: web.search(\"VeloSwim sucursales\")" in text
    assert "- result: 4 sucursales (GDL) según fuente" in text
    assert "- next: buscar contacto (operaciones/gerencia)" in text


def test_append_episode_splits_file_when_limits_exceeded(tmp_path: Path) -> None:
    service = MemoryService(root_dir=tmp_path / "memory", max_lines_per_file=1, max_bytes_per_file=1024)
    first = service.append_episode(
        user_id="alex",
        mission_id="m1",
        event_type="event",
        payload={"intent": "uno"},
    )
    second = service.append_episode(
        user_id="alex",
        mission_id="m1",
        event_type="event",
        payload={"intent": "dos"},
    )
    first_path = Path(first["path"])
    second_path = Path(second["path"])
    assert first_path.name.endswith(".md")
    assert second_path.name.endswith(".md")
    assert first_path != second_path
    assert "__part02" in second_path.name


def test_mission_workspace_and_artifacts_roundtrip(tmp_path: Path) -> None:
    service = MemoryService(root_dir=tmp_path / "memory")
    mission = service.mission_upsert("alex", "leads_veloswim", status="active", title="Leads")
    assert mission["mission_id"] == "leads_veloswim"
    updated = service.mission_step_update(
        "alex",
        "leads_veloswim",
        "step_1",
        "done",
        summary="Found website",
    )
    assert isinstance(updated.get("steps"), list)
    assert any(str(item.get("step_id") or "") == "step_1" for item in updated["steps"] if isinstance(item, dict))
    service.upsert_workspace_pointer("alex", "leads_db_path", "/tmp/leads.db")
    assert service.get_workspace_pointer("alex", "leads_db_path") == "/tmp/leads.db"

    ref = service.put_artifact(
        mission_id="leads_veloswim",
        content={"url": "https://example.com"},
        mime="application/json",
        name_hint="web_snapshot",
    )
    artifact_path = Path(ref["path"])
    assert artifact_path.exists()
    assert artifact_path.suffix == ".json"
    assert ref["mission_id"] == "leads_veloswim"
    assert ref["user_id"] == "alex"
    assert service.get_mission("alex", "leads_veloswim") is not None
    active = service.list_active_missions("alex")
    assert any(str(item.get("mission_id") or "") == "leads_veloswim" for item in active if isinstance(item, dict))


def test_search_episodes_with_query_mission_and_time_range(tmp_path: Path) -> None:
    service = MemoryService(root_dir=tmp_path / "memory")
    service.append_episode(
        user_id="alex",
        mission_id="m_search",
        event_type="web_research",
        payload={"result": "4 sucursales GDL"},
    )
    hits = service.search_episodes(user_id="alex", query="sucursales", mission_id="m_search")
    assert hits
    assert any("sucursales" in str(item.get("text") or "").lower() for item in hits)

    start = datetime.now(timezone.utc) - timedelta(days=1)
    end = datetime.now(timezone.utc) + timedelta(days=1)
    ranged_hits = service.search_episodes(
        user_id="alex",
        query="sucursales",
        mission_id="m_search",
        time_range=TimeRange(start=start, end=end),
    )
    assert ranged_hits


def test_retention_prunes_old_daily_and_weekly_files(tmp_path: Path) -> None:
    service = MemoryService(root_dir=tmp_path / "memory")
    user = "alex"
    root = tmp_path / "memory" / user
    old_day = (datetime.now().date() - timedelta(days=60)).strftime("%Y-%m-%d")
    old_year = old_day[:4]
    old_month = old_day[5:7]
    episodic_old = root / "episodic" / old_year / old_month / f"{old_day}.md"
    episodic_old.parent.mkdir(parents=True, exist_ok=True)
    episodic_old.write_text("### old\n", encoding="utf-8")

    weekly_old_date = (datetime.now().date() - timedelta(days=120)).strftime("%Y-%m-%d")
    weekly_old_year = weekly_old_date[:4]
    weekly_old = root / "summaries" / "weekly" / weekly_old_year / f"week_{weekly_old_date}_W01.md"
    weekly_old.parent.mkdir(parents=True, exist_ok=True)
    weekly_old.write_text("# old weekly\n", encoding="utf-8")

    report = service.apply_retention(user_id=user)
    assert report["deleted_daily"] >= 1
    assert report["deleted_weekly"] >= 1
    assert not episodic_old.exists()
    assert not weekly_old.exists()
