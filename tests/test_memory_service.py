from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent.cognition.memory import MemoryService
from alphonse.agent.cognition.memory import TimeRange
from alphonse.agent.cognition.memory import append_conversation_transcript


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


def test_search_episodes_mission_filter_is_entry_scoped(tmp_path: Path) -> None:
    service = MemoryService(root_dir=tmp_path / "memory")
    service.append_episode(
        user_id="alex",
        mission_id="m_a",
        event_type="web_research",
        payload={"result": "client-checkpoint"},
    )
    service.append_episode(
        user_id="alex",
        mission_id="m_b",
        event_type="web_research",
        payload={"result": "client-checkpoint"},
    )
    hits = service.search_episodes(user_id="alex", query="client-checkpoint", mission_id="m_a")
    assert hits
    for hit in hits:
        path = Path(str(hit.get("path") or ""))
        line_no = int(hit.get("line") or 0)
        lines = path.read_text(encoding="utf-8").splitlines()
        mission = ""
        for idx in range(max(0, line_no - 1), -1, -1):
            text = lines[idx]
            if text.startswith("### "):
                mission = text.split("| mission:", 1)[1].split("| event:", 1)[0].strip()
                break
        assert mission == "m_a"


def test_search_episodes_treats_query_as_literal_string(tmp_path: Path) -> None:
    service = MemoryService(root_dir=tmp_path / "memory")
    needle = "ACME (North)+?"
    service.append_episode(
        user_id="alex",
        mission_id="m_literal",
        event_type="web_research",
        payload={"result": needle},
    )
    hits = service.search_episodes(user_id="alex", query=needle, mission_id="m_literal")
    assert hits


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


def test_append_conversation_transcript_persists_full_text(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    long_text = ("contact: +52 (33) 1234-5678\n" * 200).strip()
    out = append_conversation_transcript(
        user_id="alex",
        session_id="alex|2026-03-30",
        role="user",
        text=long_text,
        channel="telegram",
        correlation_id="corr-full-text-1",
    )
    assert isinstance(out, dict)
    path = Path(str((out or {}).get("path") or ""))
    data = path.read_text(encoding="utf-8")
    assert long_text in data


def test_llm_period_summaries_daily_weekly_monthly(monkeypatch, tmp_path: Path) -> None:
    class _FakeLLM:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            assert "Never invent data" in system_prompt
            assert "Required sections" in user_prompt
            return "## Overview\n- ok\n\n## Key Facts\n- fact"

    monkeypatch.setattr("alphonse.agent.cognition.memory.service.build_llm_client", lambda: _FakeLLM())
    service = MemoryService(root_dir=tmp_path / "memory")
    service.append_episode(
        user_id="alex",
        mission_id="m1",
        event_type="event",
        payload={"result": "Acme contact +52 33 1111 1111"},
    )
    now = datetime.now().astimezone()
    daily = service.generate_daily_summary(user_id="alex", reference=now)
    weekly = service.generate_weekly_summary(user_id="alex", reference=now)
    monthly = service.generate_monthly_summary(user_id="alex", reference=now)
    assert daily is not None
    assert weekly is not None
    assert monthly is not None


def test_period_summary_skips_when_llm_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "alphonse.agent.cognition.memory.service.build_llm_client",
        lambda: (_ for _ in ()).throw(RuntimeError("llm_down")),
    )
    service = MemoryService(root_dir=tmp_path / "memory")
    service.append_episode(
        user_id="alex",
        mission_id="m1",
        event_type="event",
        payload={"result": "some content"},
    )
    result = service.generate_daily_summary(user_id="alex", reference=datetime.now().astimezone())
    assert result is None


def test_run_maintenance_writes_period_summaries_on_boundaries(monkeypatch, tmp_path: Path) -> None:
    class _FakeLLM:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return "## Overview\n- ok"

    monkeypatch.setattr("alphonse.agent.cognition.memory.service.build_llm_client", lambda: _FakeLLM())
    monkeypatch.setenv("ALPHONSE_MEMORY_DAILY_SUMMARY_ENABLED", "1")
    monkeypatch.setenv("ALPHONSE_MEMORY_WEEKLY_SUMMARY_ENABLED", "1")
    monkeypatch.setenv("ALPHONSE_MEMORY_MONTHLY_SUMMARY_ENABLED", "1")
    service = MemoryService(root_dir=tmp_path / "memory")
    day_file = tmp_path / "memory" / "alex" / "episodic" / "2026" / "05" / "2026-05-31.md"
    day_file.parent.mkdir(parents=True, exist_ok=True)
    day_file.write_text(
        "### 2026-05-31 10:00:00 +0000 | mission: m1 | event: event\n- result: boundary data\n\n",
        encoding="utf-8",
    )
    boundary = datetime.fromisoformat("2026-06-01T12:00:00+00:00")
    report = service.run_maintenance(user_id="alex", now=boundary)
    assert int(report.get("daily_summaries_written") or 0) >= 1
    assert int(report.get("weekly_summaries_written") or 0) >= 1
    assert int(report.get("monthly_summaries_written") or 0) >= 1
