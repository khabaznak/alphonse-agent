"""Opt-in configuration for the container-backed coding-skill service."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

from alphonse.agent_v2.coding_jobs import CodingJobStore
from alphonse.agent_v2.coding_skill_service import CodingSkillService
from alphonse.agent_v2.docker_coding_agents import DockerCodexSkillProvider, DockerCodexVerifier
from alphonse.agent_v2.docker_coding_sandbox import DockerCodingSandbox


def build_configured_coding_skill_service(*, users, skills):
    """Require an explicitly selected immutable image and a persistent household."""
    configured = getattr(users, 'coding_sandbox_image_id', lambda: '')
    image_id = (os.environ.get('ALPHONSE_CODEX_SANDBOX_IMAGE_ID', '') or configured()).strip()
    if not image_id:
        return None
    if sys.platform not in {'darwin', 'linux'}:
        raise ValueError('coding_container_unsupported_platform')
    if users.is_ephemeral or users.workspace_root() is None:
        raise ValueError('coding_skill_persistent_workspace_required')
    docker = shutil.which(os.environ.get('ALPHONSE_CODEX_DOCKER_BIN', 'docker'))
    if not docker:
        raise ValueError('coding_container_docker_unavailable')
    auth = Path(os.environ.get('CODEX_HOME') or (Path.home() / '.codex')) / 'auth.json'
    if auth.is_symlink() or not auth.is_file():
        raise ValueError('coding_agent_auth_unavailable')
    root = Path(users.workspace_root()) / '.coding-jobs'
    if root.is_symlink():
        raise ValueError('coding_skill_workspace_root_invalid')
    root.mkdir(mode=0o700, exist_ok=True)
    runner = DockerCodingSandbox(str(Path(docker).resolve()), image_id)
    return CodingSkillService(
        jobs=CodingJobStore(users.db_path), skills=skills, workspace_root=root,
        provider=DockerCodexSkillProvider(runner, auth, model=os.environ.get('ALPHONSE_CODEX_CODING_MODEL', '')),
        verifier=DockerCodexVerifier(runner), is_admin=users.is_admin,
    )
