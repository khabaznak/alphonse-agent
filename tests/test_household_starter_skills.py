"""Real bundled packages, project-free composition, and registry-free extension."""
from pathlib import Path
import re

import pytest

from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.core.core import CoreLoopContext, ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.skill_resources import read_resource


STARTER_NAMES = {
    'family-calendar', 'family-coordination', 'meals-groceries', 'home-care',
    'household-administration', 'project-management', 'skill-authoring',
}


def test_clean_store_validates_starters_and_linked_resources(tmp_path):
    store = SkillStore(tmp_path / 'empty household skills')
    records = store.list_skills()
    assert {record.name for record in records} == STARTER_NAMES
    assert not store.diagnostics
    assert not list(store.skills_dir.iterdir())  # Defaults need no household copy.
    for record in records:
        assert store.get(record.skill_id) == record
        assert record.instructions
        # Local Markdown resource links must be real, contained and readable.
        for relative in re.findall(r'\]\(([^)]+)\)', record.instructions):
            if '://' in relative or relative.startswith('#'):
                continue
            assert not Path(relative).is_absolute()
            resolved = (Path(record.directory) / relative).resolve(strict=True)
            assert resolved.is_relative_to(Path(record.directory).resolve())
            assert store.read_file(record.skill_id, relative)




def test_third_party_package_roundtrip_without_registry_changes(tmp_path):
    source = tmp_path / 'source' / 'synthetic-checklist'
    (source / 'references').mkdir(parents=True)
    (source / 'SKILL.md').write_text(
        '---\nname: synthetic-checklist\ndescription: Prepare a synthetic checklist.\n---\n'
        'Read [format](references/format.md) when preparing the checklist.\n'
    )
    (source / 'references/format.md').write_text('Use a task and an observable check.\n')
    store = SkillStore(tmp_path / 'installed')
    before = {record.skill_id for record in store.list_skills()}
    record = store.install_directory(source)
    assert {r.skill_id for r in store.list_skills()} == before | {record.skill_id}
    assert store.read_file(record.skill_id, 'references/format.md') == (source / 'references/format.md').read_text()
    destination = tmp_path / 'exports'
    destination.mkdir()
    exported = store.export_directory(record.skill_id, destination)
    for path in source.rglob('*'):
        if path.is_file():
            assert (exported / path.relative_to(source)).read_bytes() == path.read_bytes()
