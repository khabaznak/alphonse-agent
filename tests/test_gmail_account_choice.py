"""Synthetic account-choice receipts: no network, account, or provider access."""
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.integrations.gmail_choice import issue_choice, resolve_choice


def fixture():
    store = SQLiteQuestionStore()
    task = TaskState(task_id='task-1', user='owner', status='running', goal='Count unread')
    task.metadata['channel'] = {'integration_id': 'desktop', 'provider_key': 'tui',
        'channel_target': 'owner', 'alphonse_user_id': 'owner'}
    accounts = [{'connection_id': 'one', 'display_name': 'Personal'},
                {'connection_id': 'two', 'display_name': 'Other'}]
    return store, task, accounts


def resolve(store, task, accounts, question_id, *, label_id='INBOX', unit='messages'):
    return resolve_choice(store, task=task, question_id=question_id, accounts=accounts,
                          label_id=label_id, unit=unit)


def test_issued_question_requires_actual_authenticated_answer():
    store, task, accounts = fixture()
    question = issue_choice(store, task=task, accounts=accounts, label_id='INBOX', unit='messages')
    assert resolve(store, task, accounts, question.question_id) is None
    assert not store.route_answer(respondent_user_id='other', question_id=question.question_id,
                                  payload={'choice_id': 'two'}).handled
    answer = store.route_answer(respondent_user_id='owner', question_id=question.question_id,
                                payload={'choice_id': 'two'})
    assert answer.handled and not answer.invalid
    assert resolve(store, task, accounts, question.question_id) == 'two'
    assert resolve(store, task, accounts, question.question_id, label_id='Label_5') is None
    assert resolve(store, task, accounts, question.question_id, unit='threads') is None
    assert resolve(store, task, accounts[:1], question.question_id) is None
    assert resolve(store, task, accounts + [{'connection_id': 'three', 'display_name': 'New'}], question.question_id) is None
    task.metadata['channel']['channel_target'] = 'different'
    assert resolve(store, task, accounts, question.question_id) is None


def test_generic_model_authored_question_cannot_issue_challenge():
    store, task, accounts = fixture()
    choices = [{'id': item['connection_id'], 'label': f"{item['display_name']} ({item['connection_id']})"} for item in accounts]
    question = store.create_question(task=task, question='Which Gmail account should I use to count unread messages in INBOX?',
                                     kind='single_choice', choices=choices, respondent_user_id='owner',
                                     delivery_metadata={'origin_channel': task.metadata['channel']})
    assert store.route_answer(respondent_user_id='owner', question_id=question.question_id,
                              payload={'choice_id': 'two'}).handled
    assert resolve(store, task, accounts, question.question_id) is None


def test_other_task_or_actor_cannot_reuse_choice():
    store, task, accounts = fixture()
    question = issue_choice(store, task=task, accounts=accounts, label_id='INBOX', unit='messages')
    store.route_answer(respondent_user_id='owner', question_id=question.question_id, payload={'choice_id': 'one'})
    task.task_id = 'different'
    assert resolve(store, task, accounts, question.question_id) is None
    task.task_id = 'task-1'
    task.user = 'other'
    task.metadata['channel']['alphonse_user_id'] = 'other'
    assert resolve(store, task, accounts, question.question_id) is None


def test_newer_server_choice_supersedes_old_answer():
    store, task, accounts = fixture()
    first = issue_choice(store, task=task, accounts=accounts, label_id='INBOX', unit='messages')
    store.route_answer(respondent_user_id='owner', question_id=first.question_id, payload={'choice_id': 'one'})
    second = issue_choice(store, task=task, accounts=accounts, label_id='INBOX', unit='messages')
    assert resolve(store, task, accounts, first.question_id) is None
    store.route_answer(respondent_user_id='owner', question_id=second.question_id, payload={'choice_id': 'two'})
    assert resolve(store, task, accounts, second.question_id) == 'two'
    assert resolve(store, task, accounts, first.question_id) is None
