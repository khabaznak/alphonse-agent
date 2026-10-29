"""Synthetic account/data fixtures; never use household packages or databases."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash, MAX_OUTPUT_CHARS
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.skill_bindings import SkillBindings
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def fixture(tmp_path):
    users = V2UserStore(tmp_path / 'users.sqlite')
    users.onboard(display_name='Owner', user_id='owner', users_root=tmp_path / 'profiles')
    users.create_user(display_name='Member', user_id='member')
    users.create_user(display_name='Other', user_id='other')
    skills = SkillStore(tmp_path / 'skills')
    skills.create_skill('synthetic', 'Synthetic binding fixture', 'Use the configured account and records roles.')
    integrations = SQLiteIntegrationStore(tmp_path / 'integrations.sqlite')
    secret = 'synthetic-credential-"line\n雪'
    integrations.upsert(integration_id='test-account', provider_key='synthetic', display_name='Test',
                        enabled=True, secrets={'token': secret}, config={'endpoint': 'https://invalid.example'})
    bindings = SkillBindings(integrations, users, skills)
    bindings.configure(actor_id='owner', owner_id='member', binding_id='test-binding', skill_id='skill:synthetic',
                       kind='account', target='test-account', required_secrets=('token',))
    bindings.select(actor_id='member', skill_id='skill:synthetic', name='account', binding_id='test-binding')
    return SimpleNamespace(bindings=bindings, users=users, skills=skills, integrations=integrations,
                           secret=secret, tmp=tmp_path)


def prepare(f, actor='member', names=None):
    return f.bindings.prepare(actor_id=actor, skill_id='skill:synthetic', names=names or ['account'])


def context(f, *, actor='member', selected=True, project=None):
    task = TaskState(goal='Synthetic test', user=actor, project_id=project,
                     metadata={'v3_skill_selection': {'selected_skill_ids': ['skill:synthetic'] if selected else []}})
    return ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), skill_bindings=f.bindings)


def test_explicit_actor_choice_and_sharing_rechecked_at_use(fixture):
    f = fixture
    with pytest.raises(ValueError, match='choice_missing'):
        prepare(f, 'owner')  # Even administrators must explicitly select an account.
    with pytest.raises(PermissionError, match='access_denied'):
        f.bindings.select(actor_id='other', skill_id='skill:synthetic', name='account', binding_id='test-binding')
    with pytest.raises(PermissionError):
        f.bindings.set_access('test-binding', actor_id='other', shared_with=('other',))
    f.bindings.set_access('test-binding', actor_id='member', shared_with=('other',))
    f.bindings.select(actor_id='other', skill_id='skill:synthetic', name='account', binding_id='test-binding')
    assert prepare(f, 'other').values['account']['credentials']['token'] == f.secret
    f.bindings.set_access('test-binding', actor_id='member')
    with pytest.raises(PermissionError, match='access_denied'):
        prepare(f, 'other')
    f.bindings.set_access('test-binding', actor_id='member', enabled=False)
    with pytest.raises(PermissionError, match='revoked'):
        prepare(f)


@pytest.mark.parametrize('actor', ['unknown', '', 'member'])
def test_inactive_or_unknown_actor_denied(fixture, actor):
    fixture.users.update_user('member', is_active=False)
    with pytest.raises(PermissionError, match='actor_denied'):
        prepare(fixture, actor)


def test_inactive_owner_blocks_shared_use(fixture):
    f = fixture
    f.bindings.set_access('test-binding', actor_id='member', shared_with=('other',))
    f.bindings.select(actor_id='other', skill_id='skill:synthetic', name='account', binding_id='test-binding')
    f.users.update_user('member', is_active=False)
    with pytest.raises(PermissionError):
        prepare(f, 'other')


def test_only_admin_can_attach_household_account(fixture):
    with pytest.raises(PermissionError, match='admin_required'):
        fixture.bindings.configure(actor_id='other', owner_id='other', binding_id='stolen',
                                   skill_id='skill:synthetic', kind='account', target='test-account')


@pytest.mark.parametrize('change', ['disable', 'delete', 'secret'])
def test_account_availability_rechecked_without_fallback(fixture, change):
    f = fixture
    f.integrations.upsert(integration_id='alternative', provider_key='synthetic', display_name='Other', enabled=True,
                          secrets={'token': 'never-fall-back'})
    if change == 'disable':
        f.integrations.set_enabled('test-account', False)
    elif change == 'secret':
        f.integrations.remove_secret('test-account', 'token')
    else:
        with f.integrations._connect() as conn:
            conn.execute('DELETE FROM v2_integrations WHERE integration_id = ?', ('test-account',))
    with pytest.raises(ValueError, match='account_unavailable|credential_missing'):
        prepare(f)


def data_binding(f):
    path = f.tmp / 'household' / 'records.sqlite'
    path.parent.mkdir()
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE records (value TEXT)')
        conn.execute("INSERT INTO records VALUES ('disposable')")
    f.bindings.configure(actor_id='owner', owner_id='member', binding_id='records-binding', skill_id='skill:synthetic',
                         kind='data', target=str(path))
    f.bindings.select(actor_id='member', skill_id='skill:synthetic', name='records', binding_id='records-binding')
    return path


def test_data_independent_of_project_and_persistent_across_lifecycle(fixture):
    f = fixture
    path = data_binding(f)
    reopened = SkillBindings(SQLiteIntegrationStore(f.integrations.db_path), f.users, f.skills)
    assert reopened.prepare(actor_id='member', skill_id='skill:synthetic', names=['records']).values['records']['path'] == str(path)
    command = "printf '%s' \"$ALPHONSE_SKILL_BINDINGS\""
    arguments = {'command': command, 'skill_id': 'skill:synthetic', 'binding_names': ['records']}
    for project in [None, 'unrelated-project']:
        result = execute_bash(arguments, context=context(f, project=project))
        assert json.loads(result['stdout'])['records']['path'] == str(path)
    export = f.tmp / 'export'
    export.mkdir()
    exported = f.skills.export_directory('skill:synthetic', export)
    assert list(exported.iterdir()) == [exported / 'SKILL.md']
    assert f.secret not in (exported / 'SKILL.md').read_text()
    f.skills.replace_directory('skill:synthetic', exported)
    assert prepare(f, names=['records']).values['records']['path'] == str(path)
    f.skills.uninstall('skill:synthetic')
    with pytest.raises(ValueError, match='skill_missing'):
        prepare(f)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT value FROM records').fetchone()[0] == 'disposable'
    assert f.integrations.get('test-account').secrets['token'] == f.secret
    f.skills.install_directory(exported)
    assert prepare(f).values['account']['credentials']['token'] == f.secret


def test_missing_and_retargeted_data_denied(fixture):
    f = fixture
    path = data_binding(f)
    path.unlink()
    with pytest.raises(ValueError, match='data_missing'):
        prepare(f, names=['records'])
    target = f.tmp / 'replacement'
    target.write_text('other')
    path.symlink_to(target)
    with pytest.raises(ValueError, match='data_missing_or_changed'):
        prepare(f, names=['records'])


@pytest.mark.parametrize('location', ['inside', 'ancestor', 'missing', 'relative'])
def test_reject_invalid_or_packaged_data(fixture, location):
    f = fixture
    path = {'inside': f.skills.skills_dir / 'synthetic' / 'SKILL.md', 'ancestor': f.tmp,
            'missing': f.tmp / 'absent', 'relative': 'relative.sqlite'}[location]
    with pytest.raises(ValueError, match='data_'):
        f.bindings.configure(actor_id='owner', owner_id='member', binding_id='bad-data', skill_id='skill:synthetic',
                             kind='data', target=str(path))


def test_configuration_and_dependencies_need_no_script_registration(fixture):
    f = fixture
    f.bindings.configure(actor_id='owner', owner_id='member', binding_id='settings', skill_id='skill:synthetic',
                         kind='configuration', configuration={'timezone': 'UTC'}, executables=('missing-synthetic-executable',))
    f.bindings.select(actor_id='member', skill_id='skill:synthetic', name='settings', binding_id='settings')
    with pytest.raises(ValueError, match='dependency_missing'):
        prepare(f, names=['settings'])


def test_native_adapter_redacts_nested_results_and_errors(fixture):
    f = fixture
    assert f.secret not in repr(prepare(f))
    assert f.secret not in repr(f.integrations.get('test-account'))
    result = f.bindings.use(actor_id='member', skill_id='skill:synthetic', names=['account'],
                            operation=lambda values: {'echo': [values['account']['credentials']['token']]})
    assert result == {'echo': ['[REDACTED]']}
    def fail(values):
        raise ValueError(values['account']['credentials']['token'])
    with pytest.raises(RuntimeError, match='operation_failed') as error:
        f.bindings.use(actor_id='member', skill_id='skill:synthetic', names=['account'], operation=fail)
    assert error.value.__suppress_context__
    assert f.secret not in str(error.value)


def test_bash_redacts_literal_json_and_timeout_output_before_truncation(fixture):
    f = fixture
    args = {'command': 'printf "%s" "$ALPHONSE_SKILL_BINDINGS"', 'skill_id': 'skill:synthetic', 'binding_names': ['account']}
    result = execute_bash(args, context=context(f))
    assert json.dumps(f.secret)[1:-1] not in result['stdout']
    assert '[REDACTED]' in result['stdout']
    # A literal token straddling the output limit must be removed before clipping.
    f.integrations.upsert(integration_id='test-account', provider_key='synthetic', display_name='Test', enabled=True,
                          secrets={'token': 'synthetic-secret-end'})
    script = "printf '%*s' " + str(MAX_OUTPUT_CHARS - 5) + " ''; printf synthetic-secret-end >&2; printf synthetic-secret-end; sleep 2"
    args.update(command=script, timeout_seconds=0.05)
    result = execute_bash(args, context=context(f))
    assert result['timed_out']
    assert 'synthetic-secret' not in result['stdout'] + result['stderr']
    assert result['stderr'] == '[REDACTED]'


def test_bash_denies_unselected_or_unauthorized_bindings(fixture):
    args = {'command': 'exit 0', 'skill_id': 'skill:synthetic', 'binding_names': ['account']}
    with pytest.raises(PermissionError, match='not_selected'):
        execute_bash(args, context=context(fixture, selected=False))
    with pytest.raises(ValueError, match='choice_missing'):
        execute_bash(args, context=context(fixture, actor='other'))
    fixture.bindings.set_access('test-binding', actor_id='member', enabled=False)
    with pytest.raises(PermissionError, match='revoked'):
        execute_bash(args, context=context(fixture))


def test_loop_context_propagates_binding_service(fixture):
    ctx = CoreLoopContext(messages=InMemoryMessageQueue(), skill_bindings=fixture.bindings)
    assert ctx.tool_execution_context(TaskState(goal='test')).skill_bindings is fixture.bindings


def test_account_delivers_only_explicit_credential_keys_and_rotation(fixture):
    f = fixture
    f.integrations.upsert(integration_id='test-account', provider_key='synthetic', display_name='Test', enabled=True,
                          secrets={'token': 'rotated-synthetic-token', 'refresh': 'unrequested-secret'})
    assert prepare(f).values['account']['credentials'] == {'token': 'rotated-synthetic-token'}
    with f.integrations._connect() as conn:
        saved = dict(conn.execute('SELECT * FROM v2_skill_bindings').fetchone())
    assert 'rotated-synthetic-token' not in json.dumps(saved)
    assert 'unrequested-secret' not in json.dumps(saved)


def test_config_ready_and_native_sqlite_access(fixture):
    f = fixture
    data_binding(f)
    f.bindings.configure(actor_id='owner', owner_id='member', binding_id='settings', skill_id='skill:synthetic',
                         kind='configuration', configuration={'timezone': 'UTC'}, executables=('sh',))
    f.bindings.select(actor_id='member', skill_id='skill:synthetic', name='settings', binding_id='settings')
    def query(values):
        with sqlite3.connect(values['records']['path']) as conn:
            return {'row': conn.execute('SELECT value FROM records').fetchone()[0],
                    'timezone': values['settings']['configuration']['timezone']}
    assert f.bindings.use(actor_id='member', skill_id='skill:synthetic', names=['records', 'settings'], operation=query) == {
        'row': 'disposable', 'timezone': 'UTC'}


def test_bash_delivery_does_not_change_parent_or_ordinary_commands(fixture, monkeypatch):
    import os
    monkeypatch.setenv('ALPHONSE_SKILL_BINDINGS', 'inherited-value')
    result = execute_bash({'command': 'printf "%s" "$ALPHONSE_SKILL_BINDINGS"'})
    assert result['stdout'] == ''
    args = {'command': 'printf "%s" "$ALPHONSE_SKILL_BINDINGS"', 'skill_id': 'skill:synthetic', 'binding_names': ['account']}
    assert '[REDACTED]' in execute_bash(args, context=context(fixture))['stdout']
    assert os.environ['ALPHONSE_SKILL_BINDINGS'] == 'inherited-value'


def test_choice_cannot_cross_skill_and_identity_cannot_be_retargeted(fixture):
    f = fixture
    with pytest.raises(ValueError, match='skill_mismatch'):
        f.bindings.select(actor_id='member', skill_id='skill:another', name='account', binding_id='test-binding')
    with pytest.raises(ValueError, match='already_exists'):
        f.bindings.configure(actor_id='owner', owner_id='other', binding_id='test-binding', skill_id='skill:synthetic',
                             kind='configuration')
    assert prepare(f).values['account']['credentials']['token'] == f.secret
