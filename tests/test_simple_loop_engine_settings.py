import sqlite3

import pytest

from alphonse.agent_v2.intelligence_engine_settings import (
    IntelligenceEngineSettings,
    SQLiteIntelligenceEngineSettingsStore,
)
from alphonse.agent_v2.core.intelligence.task_state import TaskState


def test_legacy_preference_does_not_select_old_loop_or_rewrite_saved_state(tmp_path):
    path = tmp_path / 'settings.sqlite'
    store = SQLiteIntelligenceEngineSettingsStore(path)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO v2_intelligence_engine_settings VALUES (1, 'hierarchical_v3', '[\"old-project\"]', 'old')")
    assert store.get().engine_for('old-project') == 'single_loop_experimental'
    assert store.get().to_dict() == {'default_engine': 'single_loop_experimental', 'updated_at': 'old'}
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT default_engine FROM v2_intelligence_engine_settings').fetchone() == ('hierarchical_v3',)
    with pytest.raises(ValueError, match='intelligence_engine_invalid'):
        IntelligenceEngineSettings(default_engine='hierarchical_v3')
    assert TaskState().intelligence_engine == 'unassigned'
    assert TaskState.from_dict({'intelligence_engine': 'hierarchical_v3', 'intelligence_schema_version': 3}).intelligence_engine == 'hierarchical_v3'
