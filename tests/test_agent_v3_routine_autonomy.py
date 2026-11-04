"""Disposable storage and scripted System Two regression seams.

These exercise real file effects and prompt wiring, not live-model reliability.
"""
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.core.core import ToolExecutionContext, CoreLoopContext
from alphonse.agent_v2.core.io.identity import V2IdentityResolver
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.routine_autonomy import ordinary_workspace_context, ROUTINE_AUTONOMY_GUIDANCE
from alphonse.agent_v2.core.intelligence.v3.planner import _strategic_plan_instructions
from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash


@pytest.fixture
def owners(tmp_path):
    store = V2UserStore(tmp_path / 'users.sqlite3')
    store.set_users_root(tmp_path / 'users')
    store.create_user(display_name='Alice', user_id='alice')
    store.create_user(display_name='Bob', user_id='bob')
    return store, V2IdentityResolver(user_store=store)


def context(resolver, user='alice', project_id='', project_store=None):
    return ToolExecutionContext(
        task=TaskState(user=user, project_id=project_id),
        messages=InMemoryMessageQueue(), identity_resolver=resolver, project_store=project_store,
    )


def test_notes_created_read_back_and_reported_without_project(owners):
    store, resolver = owners
    ctx = context(resolver)
    result = execute_bash({'command': "set -C; printf 'Household notes\\n' > household-notes.md && cat household-notes.md && pwd"}, context=ctx)
    path = store.ordinary_workspace('alice') / 'household-notes.md'
    assert result['exit_code'] == 0
    assert result['cwd'] == str(path.parent)
    assert result['stdout'] == f'Household notes\n{path.parent}\n'
    assert path.read_text() == 'Household notes\n'
    assert not (store.ordinary_workspace('bob') / path.name).exists()
    assert execute_bash({'command': 'test ! -e household-notes.md'}, context=context(resolver, 'bob'))['exit_code'] == 0
    assert path.parent.stat().st_mode & 0o077 == 0


def test_new_file_collision_preserves_existing_notes(owners):
    store, resolver = owners
    ctx = context(resolver)
    path = store.ordinary_workspace('alice') / 'notes.md'
    path.write_text('Existing notes')
    result = execute_bash({'command': "set -C; printf replacement > notes.md"}, context=ctx)
    assert result['exit_code'] != 0
    assert path.read_text() == 'Existing notes'
    result = execute_bash({'command': "set -C; printf new > notes-2.md && cat notes-2.md"}, context=ctx)
    assert result['exit_code'] == 0
    assert result['stdout'] == 'new'
    assert path.read_text() == 'Existing notes'


def test_explicit_destination_wins_over_project_and_default(owners, tmp_path):
    store, resolver = owners
    explicit = tmp_path / 'explicit'
    project = tmp_path / 'project'
    explicit.mkdir()
    project.mkdir()
    projects = SimpleNamespace(get_project=lambda _, requester_user_id: SimpleNamespace(root_path=project))
    ctx = context(resolver, project_id='project', project_store=projects)
    result = execute_bash({'command': 'pwd', 'cwd': str(explicit)}, context=ctx)
    assert result['stdout'].strip() == str(explicit)
    assert execute_bash({'command': 'pwd'}, context=ctx)['cwd'] == str(project)
    assert not (store.users_root() / 'alice' / 'workspace').exists()


def test_unavailable_explicit_or_project_destination_never_falls_back(owners, tmp_path):
    _, resolver = owners
    with pytest.raises(ValueError, match='bash_cwd_not_found'):
        execute_bash({'command': 'touch unintended', 'cwd': str(tmp_path / 'missing')}, context=context(resolver))
    with pytest.raises(ValueError, match='bash_project_directory_unavailable'):
        execute_bash({'command': 'touch unintended'}, context=context(resolver, project_id='inaccessible'))
    denied = SimpleNamespace(get_project=lambda *a, **kw: None)
    with pytest.raises(ValueError, match='bash_project_directory_unavailable'):
        execute_bash({'command': 'touch unintended'}, context=context(resolver, project_id='other-owner', project_store=denied))


def test_missing_inactive_owner_and_unavailable_storage_fail_closed(owners):
    store, resolver = owners
    for owner in ('', 'unknown'):
        with pytest.raises(ValueError, match='ordinary_workspace_owner_required'):
            execute_bash({'command': 'touch unintended'}, context=context(resolver, owner))
    store.update_user('bob', is_active=False)
    with pytest.raises(ValueError, match='ordinary_workspace_owner_required'):
        execute_bash({'command': 'touch unintended'}, context=context(resolver, 'bob'))
    workspace = store.users_root() / 'alice' / 'workspace'
    workspace.write_text('blocked storage')
    with pytest.raises((ValueError, OSError)):
        execute_bash({'command': 'touch unintended'}, context=context(resolver))
    assert workspace.read_text() == 'blocked storage'


@pytest.mark.parametrize('linked_component', ['owner', 'workspace'])
def test_linked_owner_storage_cannot_alias_another_owner(owners, linked_component):
    store, resolver = owners
    bob = store.ordinary_workspace('bob')
    alice = store.users_root() / 'alice'
    if linked_component == 'owner':
        (alice / 'user_context.md').unlink()
        alice.rmdir()
        alice.symlink_to(bob.parent, target_is_directory=True)
    else:
        (alice / 'workspace').symlink_to(bob, target_is_directory=True)
    with pytest.raises(ValueError, match='ordinary_workspace_symlink'):
        execute_bash({'command': 'touch unintended'}, context=context(resolver))
    assert not (bob / 'unintended').exists()


def test_workspace_migration_preserves_notes_and_reuses_existing_files(owners, tmp_path):
    store, _ = owners
    old = store.ordinary_workspace('alice')
    (old / 'notes.md').write_text('old notes')
    destination = tmp_path / 'new-workspace'
    target = destination / 'users' / 'alice' / 'workspace'
    target.mkdir(parents=True)
    (target / 'notes.md').write_text('newer notes')
    (old / 'keep.md').write_text('keep')
    store.set_workspace_root(destination)
    assert store.ordinary_workspace('alice') == target
    assert (target / 'notes.md').read_text() == 'newer notes'
    assert (target / 'keep.md').read_text() == 'keep'
    assert (old / 'notes.md').read_text() == 'old notes'


def test_owner_default_is_visible_before_creation_and_delegation_guidance_is_shared(owners):
    store, resolver = owners
    ctx = CoreLoopContext(messages=InMemoryMessageQueue(), identity_resolver=resolver)
    task = TaskState(user='alice', goal='Create a new notes file')
    default = store.ordinary_workspace('alice', create=False)
    assert str(default) in ordinary_workspace_context(task, ctx)
    assert not default.exists()
    assert ROUTINE_AUTONOMY_GUIDANCE in _strategic_plan_instructions()
    assert 'do not ask approval of that same choice again' in ROUTINE_AUTONOMY_GUIDANCE
    assert 'Conflicting explicit destinations require a narrow clarification' in ROUTINE_AUTONOMY_GUIDANCE


def test_delegated_filename_resumes_through_planner_and_executor_without_another_question(owners, tmp_path):
    # System Two's selection is scripted; real question persistence, plan validation,
    # shell execution and result evidence must carry the delegation through.
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
    from alphonse.agent_v2.core.intelligence.v3.contracts import (
        PhasePlan, PhaseSubgoal, SideEffectClass, MutationScope, PhaseStatus,
    )
    from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor
    from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
    from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection, SystemOneTacticalReview

    store, resolver = owners
    questions = SQLiteQuestionStore(tmp_path / 'questions.sqlite3')
    task = TaskState(task_id='notes', user='alice', intelligence_engine='hierarchical_v3', goal='Create household notes with the text: Shopping on Friday')
    task.set_acceptance_contract_from_markdown('- [ ] Household notes created and read back')
    task.hierarchical_state = {'phase': {'phase_id': 'ask-filename'}, 'active_subgoal_id': 'question'}
    question = questions.create_question(task=task, question='What filename should I use?')
    answered = questions.route_answer(respondent_user_id='alice', text='You choose the filename and default directory.', question_id=question.question_id)
    task = answered.resumed_task
    assert task is not None
    phase = PhasePlan('notes', 'Create and verify household notes', (PhaseSubgoal(
        'write', 'Create and read back notes', 'verified_notes', allowed_capabilities=('local_shell',),
        allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,),
    ),), authorized_capabilities=('local_shell',), mutation_scope=MutationScope(('household-notes.md',)))
    action = {'tool_id': 'native.bash', 'arguments': {
        'command': "set -C; printf 'Shopping on Friday\\n' > household-notes.md && cat household-notes.md && pwd",
    }, 'acceptance_questions': [{
        'question_id': 'notes-verified', 'type': 'noul', 'instructions': 'Are the created notes read back?',
        'criteria': {'true': 'Contents and destination verified.', 'false': 'Missing or failed evidence.'},
    }]}

    class Inference:
        outputs = iter([phase.to_dict(), action])
        requests = []

        def generate_json(self, request):
            self.requests.append(request)
            return SimpleNamespace(json_value=next(self.outputs))

    class Judgments:
        def curate_request_tools(self, *, tools, **_):
            return SystemOneToolRegistrySelection(selected_tool_ids=tuple(t.tool_id for t in tools))

        select_plan_tools = curate_request_tools

        def evaluate_tactical_progress(self, **_):
            return SystemOneTacticalReview(True, 0.99, True)

    tools = InMemoryToolRegistry()
    tools.register(build_bash_tool_definition())
    inference = Inference()
    ctx = CoreLoopContext(messages=InMemoryMessageQueue(), identity_resolver=resolver,
                          tools=tools, question_store=questions, inference=inference, system_one=Judgments())
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    state = HierarchicalCAPDProcessor._new_state(task, ctx)
    outcome = PhaseExecutor(reveal_policy=ToolRevealPolicy()).run(task, state, ctx)
    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert [a.tool_id for a in state.actions] == ['native.bash']
    assert questions.get_question(question.question_id).status == 'answered'
    assert (store.ordinary_workspace('alice') / 'household-notes.md').read_text() == 'Shopping on Friday\n'
    assert len(inference.requests) == 2
    for request in inference.requests:
        assert ROUTINE_AUTONOMY_GUIDANCE in request.prompt
        assert str(store.ordinary_workspace('alice')) in request.prompt
        assert 'You choose the filename' in request.prompt


def test_conflicting_explicit_destinations_wait_for_owner_without_touching_either(owners, tmp_path):
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    store, resolver = owners
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    (first / 'notes.md').write_text('keep first')
    (second / 'notes.md').write_text('keep second')
    questions = SQLiteQuestionStore(tmp_path / 'questions.sqlite3')
    task = TaskState(task_id='conflict', user='alice', goal=f'Create notes in {first}; save them in {second}')
    question = questions.create_question(task=task, question=f'Which destination should I use: {first} or {second}?')
    assert task.status == 'waiting_user'
    # Independent authorized inspection can proceed while the destination is unresolved.
    observed = execute_bash({'command': 'ls', 'cwd': str(first)}, context=context(resolver))
    assert observed['exit_code'] == 0
    assert questions.get_question(question.question_id).status == 'pending'
    assert (first / 'notes.md').read_text() == 'keep first'
    assert (second / 'notes.md').read_text() == 'keep second'
    assert not (store.users_root() / 'alice' / 'workspace').exists()
