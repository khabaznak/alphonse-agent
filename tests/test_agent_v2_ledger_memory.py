from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

from alphonse.agent_v2.core.memory import LedgerMemory
from alphonse.agent_v2.memory_settings import MemorySettings
from alphonse.agent_v2.memory_settings import SQLiteMemorySettingsStore


def _memory(tmp_path: Path, *, bytes_: int = 512000, words: int = 500) -> LedgerMemory:
    store = SQLiteMemorySettingsStore(tmp_path / "settings.sqlite3")
    store.save(MemorySettings(bytes_, words))
    return LedgerMemory(users_root=lambda: tmp_path / "users", settings_store=store)


def _task(user: str = "alex", project: str = "", task_id: str = "task-1"):
    return SimpleNamespace(user=user, project_id=project, task_id=task_id, message_id="", goal="Plan the party", outcome=None, status="running")


def test_project_ledgers_are_scoped_per_user(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    alex, gaby = _task("alex", "party"), _task("gaby", "party")
    memory.start_task(alex); memory.event(alex, "Conversation", "- Alphonse: I can help.")
    memory.start_task(gaby)
    assert "I can help" in memory.latest_content(user_id="alex", project_id="party")
    assert "I can help" not in memory.latest_content(user_id="gaby", project_id="party")
    assert (tmp_path / "users" / "alex" / "projects" / "party" / "memory" / "ledger-0001.md").exists()


def test_rollover_links_previous_ledger_and_bounds_summary_words(tmp_path: Path) -> None:
    memory = _memory(tmp_path, bytes_=1024, words=3)
    first = _task()
    memory.start_task(first)
    memory.event(first, "Conversation", "word1 word2 word3 word4 " + "x" * 1100)
    second = _task(task_id="task-2")
    rendered = memory.start_task(second)
    assert "[ledger-0001.md](ledger-0001.md)" in rendered
    summary = rendered.split("## Compaction Summary\n", 1)[1].split("\n\n## Memory", 1)[0]
    assert len(summary.split()) <= 3


def test_generic_and_project_memory_do_not_mix(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.start_task(_task(project=""))
    memory.start_task(_task(project="home"))
    assert memory.latest_content(user_id="alex", project_id="") != memory.latest_content(user_id="alex", project_id="home")


def test_concurrent_ledger_appends_do_not_interleave_or_drop_events(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    task = _task(project="shared")
    memory.start_task(task)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda index: memory.event(task, "Conversation", f"- User: event-{index:03d}"), range(100)))

    content = memory.latest_content(user_id="alex", project_id="shared")

    for index in range(100):
        assert content.count(f"event-{index:03d}") == 1
