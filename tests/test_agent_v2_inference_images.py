from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.assets import AttachmentDescriptor, SQLiteAssetStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import (
    InferencePurpose, InferenceRequest, InferenceRouter, ModelProfile,
    OpenAICodexProvider, StubInferenceProvider,
)
from alphonse.agent_v2.core.inference.images import task_image_inputs
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.media_tools_settings import SQLiteMediaToolsSettingsStore

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


def _fixture(tmp_path):
    store = SQLiteAssetStore(':memory:', tmp_path / 'assets')
    image = store.register_bytes(owner_user_id='alex', descriptor=AttachmentDescriptor('receipt.png', 'image/png', len(PNG)), content=PNG, source='desktop')
    task = TaskState(user='alex', goal='Read the image', metadata={
        'asset_ids': [image.asset_id],
        'attachments': [{'asset_id': image.asset_id, 'mime_type': 'image/png', 'path': '/untrusted/path'}],
    })
    return store, image, task


def test_resolves_catalog_bytes_deduplicates_and_does_not_trust_manifest_path(tmp_path):
    store, image, task = _fixture(tmp_path)
    inputs = task_image_inputs(task, store)
    assert len(inputs) == 1
    assert inputs[0].asset_id == image.asset_id
    assert inputs[0].content == PNG
    assert 'content=' not in repr(inputs[0])


def test_cannot_attach_another_users_asset(tmp_path):
    store, _, task = _fixture(tmp_path)
    task.user = 'someone-else'
    with pytest.raises(ValueError, match='not_found_or_forbidden'):
        task_image_inputs(task, store)


def test_missing_corrupt_or_unregistered_images_fail_explicitly(tmp_path):
    store, image, task = _fixture(tmp_path)
    with pytest.raises(ValueError, match='store_unavailable'):
        task_image_inputs(task, None)
    Path(image.path).write_bytes(b'changed')
    with pytest.raises(ValueError, match='integrity_failed'):
        task_image_inputs(task, store)
    Path(image.path).unlink()
    with pytest.raises(ValueError, match='image_unavailable'):
        task_image_inputs(task, store)
    task.metadata = {'attachments': [{'mime_type': 'image/png'}]}
    with pytest.raises(ValueError, match='unregistered'):
        task_image_inputs(task, store)


@pytest.mark.parametrize('method,output', [('generate_markdown', 'visible text'), ('generate_json', '{"text":"visible text"}'), ('plan_tool_call', '{"tool_id":"native.respond"}')])
@pytest.mark.parametrize('interruptible', [False, True])
def test_selected_model_receives_image_bytes_and_cli_files_are_cleaned(tmp_path, monkeypatch, method, output, interruptible):
    store, image, task = _fixture(tmp_path)
    images = task_image_inputs(task, store)
    paths = []

    def run(command, **kwargs):
        assert command[command.index('--model') + 1] == 'selected-model'
        assert command[-1] == '-'
        paths.extend(Path(command[index + 1]) for index, arg in enumerate(command) if arg == '--image')
        assert len(paths) == 1
        assert paths[0].read_bytes() == PNG
        envelope = json.loads(kwargs.get('input') or kwargs['prompt'])
        assert envelope['image_inputs'] == [{'image_number': 1, 'asset_id': image.asset_id}]
        assert 'vision capabilities' in envelope['image_input_instructions']
        return SimpleNamespace(returncode=0, stdout=output, stderr='')

    monkeypatch.setattr('alphonse.agent_v2.core.inference.openai_codex.shutil.which', lambda _: '/bin/codex')
    monkeypatch.setattr('alphonse.agent_v2.core.inference.openai_codex.subprocess.run', run)
    monkeypatch.setattr('alphonse.agent_v2.core.inference.openai_codex._run_interruptible', run)
    router = InferenceRouter(provider=OpenAICodexProvider(), default_profile=ModelProfile(provider='openai_codex', model='selected-model', profile_id='selected'))
    getattr(router, method)(InferenceRequest(prompt='Read according to skill instructions', purpose=InferencePurpose.TACTICAL_ACTION, images=images, cancel_checker=(lambda: False) if interruptible else None))
    assert all(not path.exists() for path in paths)


def test_image_inputs_are_not_silently_dropped_by_unsupported_provider(tmp_path):
    store, _, task = _fixture(tmp_path)
    provider = StubInferenceProvider()
    provider.supports_image_input = False
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('other', 'text-only', 'test'))
    with pytest.raises(ValueError, match='image_input_unsupported'):
        router.generate_markdown(InferenceRequest(prompt='Read', purpose=InferencePurpose.FINAL_RESPONSE, images=task_image_inputs(task, store)))
    assert provider.requests == []


def test_legacy_ocr_settings_are_ignored_without_losing_speech_settings():
    store = SQLiteMediaToolsSettingsStore(':memory:')
    with store._connect() as connection:
        connection.execute('INSERT INTO v2_media_tools_settings(settings_id,payload,updated_at) VALUES (1,?,?)', (json.dumps({'ocr': {'enabled': True, 'model_id': 'old'}, 'stt': {'model': 'small'}}), 'now'))
    assert store.get().stt.model == 'small'
    assert 'ocr' not in store.get().to_dict()
    assert 'ocr' not in store.save(store.get()).to_dict()
    for operation in (lambda: store.update('ocr', {}), lambda: store.mark_verification('ocr', ready=True)):
        with pytest.raises(ValueError, match='media_tools_kind_invalid'):
            operation()


def test_direct_response_sees_images_and_selected_skill_instructions(tmp_path):
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    store, image, task = _fixture(tmp_path)
    task.metadata['v3_skill_selection'] = {'selected_skill_ids': ['skill:reading']}
    skill = SimpleNamespace(skill_id='skill:reading', name='Reading', description='Read documents', instructions='Extract only printed dates.', directory=str(tmp_path))
    provider = StubInferenceProvider(markdown_by_purpose={InferencePurpose.FINAL_RESPONSE: 'Visible date'})
    context = CoreLoopContext(messages=InMemoryMessageQueue(), asset_store=store, inference=InferenceRouter(provider=provider, default_profile=ModelProfile('stub', 'selected', 'selected')), skill_store=SimpleNamespace(get=lambda _: skill))
    HierarchicalCAPDProcessor._complete_direct_response(task, context)
    assert provider.requests[0].images[0].asset_id == image.asset_id
    assert 'Extract only printed dates.' in provider.requests[0].prompt
    assert task.status == 'completed'


def test_multiple_image_order_and_cleanup_on_model_rejection(tmp_path, monkeypatch):
    from dataclasses import replace
    store, _, task = _fixture(tmp_path)
    first = task_image_inputs(task, store)[0]
    paths = []

    def run(command, **kwargs):
        paths.extend(Path(command[i + 1]) for i, arg in enumerate(command) if arg == '--image')
        assert len(paths) == 2
        assert [row['asset_id'] for row in json.loads(kwargs['input'])['image_inputs']] == [first.asset_id, 'second']
        assert all(path.is_file() for path in paths)
        return SimpleNamespace(returncode=1, stdout='', stderr='This model does not support image inputs')

    monkeypatch.setattr('alphonse.agent_v2.core.inference.openai_codex.shutil.which', lambda _: '/bin/codex')
    monkeypatch.setattr('alphonse.agent_v2.core.inference.openai_codex.subprocess.run', run)
    with pytest.raises(ValueError, match='model_image_input_unsupported'):
        OpenAICodexProvider().generate_markdown(InferenceRequest(prompt='Compare', purpose=InferencePurpose.FINAL_RESPONSE, images=(first, replace(first, asset_id='second'))))
    assert all(not path.exists() for path in paths)


def test_runtime_wires_the_asset_catalog_into_core_inference(tmp_path):
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore
    users = V2UserStore(':memory:')
    users.onboard(display_name='Admin', users_root=tmp_path / 'users')
    assets = SQLiteAssetStore(':memory:', tmp_path / 'assets')
    runtime = build_runtime_host(user_store=users, asset_store=assets)
    assert runtime.core.asset_store is assets
    assert runtime.core.tools.get('native.analyze_image') is None
