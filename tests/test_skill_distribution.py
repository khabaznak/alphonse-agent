"""Build sdist -> wheel -> installed resource checks without touching source packages."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def run(code, cwd, **kwargs):
    return subprocess.run([sys.executable, '-c', code], cwd=cwd, check=True, capture_output=True, text=True, **kwargs)


def test_installed_distribution_includes_nested_skill_resources(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / 'source'
    source.mkdir()
    shutil.copy(repository / 'pyproject.toml', source)
    shutil.copytree(repository / 'alphonse', source / 'alphonse', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    fixture = repository / 'tests/fixtures/skills/sample'
    shutil.copytree(fixture, source / 'alphonse/agent_v2/builtin_skills/sample')
    dist = tmp_path / 'dist'
    dist.mkdir()
    run(f'from setuptools.build_meta import build_sdist; build_sdist({str(dist)!r})', source)
    with tarfile.open(next(dist.glob('*.tar.gz'))) as archive:
        archive.extractall(tmp_path / 'unpacked', filter='data')
    unpacked = next((tmp_path / 'unpacked').iterdir())
    run(f'from setuptools.build_meta import build_wheel; build_wheel({str(dist)!r})', unpacked)
    installed = tmp_path / 'installed'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-compile', '--target', str(installed), str(next(dist.glob('*.whl')))], check=True, capture_output=True, text=True)
    expected = {p.relative_to(fixture).as_posix(): p.read_bytes() for p in fixture.rglob('*') if p.is_file()}
    bundled = repository / 'alphonse/agent_v2/builtin_skills'
    starter_files = {p.relative_to(bundled).as_posix(): p.read_bytes() for p in bundled.rglob('*') if p.is_file()}
    starter_names = {'family-coordination', 'meals-groceries', 'home-care',
                     'household-administration', 'project-management', 'skill-authoring'}
    code = f'''
from importlib.resources import files
from alphonse.agent_v2.skills import SkillStore
from pathlib import Path
root = files('alphonse.agent_v2').joinpath('builtin_skills/sample')
for relative, content in {expected!r}.items():
    assert root.joinpath(relative).read_bytes() == content, relative
store = SkillStore({str(tmp_path / 'skills')!r})
assert root.joinpath('scripts/nested/read.sh').stat().st_mode & 0o111
assert store.get('sample').metadata['version'] == '1.0'
assert Path(store.get('sample').directory).is_relative_to({str(installed)!r})
assert 'sample' in {{record.name for record in store.list_skills()}}
assert {{record.name for record in store.list_skills()}} == {starter_names!r} | {{'sample'}}
bundled = files('alphonse.agent_v2').joinpath('builtin_skills')
for relative, content in {starter_files!r}.items():
    assert bundled.joinpath(relative).read_bytes() == content, relative
for name in {starter_names!r}:
    assert Path(store.get(name).directory).is_relative_to({str(installed)!r})
assert not list(store.skills_dir.iterdir())
assert not store.diagnostics
'''
    run(code, tmp_path, env={**os.environ, 'PYTHONPATH': str(installed)})
