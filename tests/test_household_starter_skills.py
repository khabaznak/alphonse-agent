"""Real bundled packages, project-free composition, and registry-free extension."""
from pathlib import Path
import re

import pytest

from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.context import (
    selected_skill_guidance, skill_candidates_for_curation,
)
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


STARTER_NAMES = {
    'family-coordination', 'meals-groceries', 'home-care',
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


@pytest.mark.parametrize('prompt,names', [
    ('Plan next week\'s school pickups.', ['family-coordination']),
    ('Three quick dinners for four using rice.', ['meals-groceries']),
    ('Prepare an AC visit and track its warranty.', ['home-care', 'household-administration']),
    ('Plan dinners around late practices.', ['meals-groceries', 'family-coordination']),
    ('Plan our kitchen renovation.', ['project-management', 'home-care']),
    ('Improve this upkeep skill with CSV import.', ['skill-authoring', 'home-care']),
    ('What is 2 + 2?', []),
])
def test_representative_selections_load_complete_composed_guidance(tmp_path, prompt, names):
    """Exercise activation with reviewed selection inputs, not a fake Jev classifier."""
    store = SkillStore(tmp_path / 'skills')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=store)
    candidates = skill_candidates_for_curation(context)
    assert {candidate['id'] for candidate in candidates} == {
        f'skill:{name}' for name in STARTER_NAMES
    }
    # Discovery provides concise activation descriptions, not full bodies/resources.
    assert all(c['description'] == store.get(c['id']).description for c in candidates)
    task = TaskState(user='synthetic-member', project_id=None, goal=prompt)
    ids = [f'skill:{name}' for name in names]
    task.metadata['v3_skill_selection'] = {'selected_skill_ids': ids, 'primary_skill_ids': ids[:1]}
    guidance, loaded = selected_skill_guidance(task, context)
    assert loaded == ids
    for skill_id in ids:
        record = store.get(skill_id)
        assert record.instructions in guidance  # No truncated completion constraints.
        assert record.directory in guidance
    if not ids:
        assert guidance == ''


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
