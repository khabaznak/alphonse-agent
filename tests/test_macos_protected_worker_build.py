"""Offline checks of the candidate build's fail-closed preflight."""
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'desktop/scripts/build_protected_worker.py'
spec = importlib.util.spec_from_file_location('protected_worker_build', SCRIPT)
worker_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker_build)
IDENTITY = 'A' * 40
TEAM = 'AB12CD34EF'


def test_explicit_identity_must_be_available_and_output_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_build.sys, 'platform', 'darwin')
    monkeypatch.setattr(worker_build, '_run', lambda *a, **k: '     0 valid identities found\n')
    output = tmp_path / 'candidate'
    with pytest.raises(worker_build.BuildError, match='signing_identity_unavailable'):
        worker_build.prepare(IDENTITY, TEAM, output)
    assert not output.exists()
    with pytest.raises(worker_build.BuildError, match='signing_identity_or_team_invalid'):
        worker_build.prepare('-', TEAM, output)


def test_existing_output_is_never_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_build.sys, 'platform', 'darwin')
    output = tmp_path / 'candidate'
    output.mkdir()
    (output / 'existing').write_text('preserve')
    with pytest.raises(worker_build.BuildError, match='output_must_not_exist'):
        worker_build.prepare(IDENTITY, TEAM, output)
    assert (output / 'existing').read_text() == 'preserve'


def test_signing_identity_parser_rejects_substring_and_unselected_keys():
    listing = f'  1) {IDENTITY} "Developer ID Application: Example"\n'
    assert worker_build._identity_available(IDENTITY, listing)
    assert not worker_build._identity_available('B' * 40, listing)
    assert not worker_build._identity_available('-', listing)
    assert not worker_build._identity_available(IDENTITY, listing.rstrip() + ' (CSSMERR_TP_CERT_REVOKED)')


def test_bundle_rejects_escaping_symlink(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'escape').symlink_to(tmp_path)
    with pytest.raises(worker_build.BuildError, match='bundle_symlink_outside'):
        list(worker_build._macho_files(bundle))


def test_candidate_build_is_onedir_and_only_published_after_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_build.sys, 'platform', 'darwin')
    observed = []

    def command(args, **kwargs):
        observed.append(args)
        if args[0] == '/usr/bin/security':
            return f'  1) {IDENTITY} "Developer ID Application: Example"\n'
        output = Path(args[args.index('--distpath') + 1]) / 'alphonse-protected-worker'
        output.mkdir(parents=True)
        (output / 'candidate').write_text('fixture')
        return ''

    monkeypatch.setattr(worker_build, '_run', command)
    monkeypatch.setattr(worker_build, '_verify_bundle', lambda bundle, team: None)
    target = tmp_path / 'candidate'
    worker_build.prepare(IDENTITY, TEAM, target)
    assert (target / 'candidate').read_text() == 'fixture'
    assert '--onedir' in observed[1] and '--onefile' not in observed[1]
    assert observed[1][observed[1].index('--codesign-identity') + 1] == IDENTITY
    assert sorted(path.name for path in tmp_path.iterdir()) == ['candidate']

    def reject(*args):
        raise worker_build.BuildError('signature_invalid')

    monkeypatch.setattr(worker_build, '_verify_bundle', reject)
    rejected = tmp_path / 'rejected'
    with pytest.raises(worker_build.BuildError, match='signature_invalid'):
        worker_build.prepare(IDENTITY, TEAM, rejected)
    assert not rejected.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ['candidate']


def test_candidate_checks_identity_hardening_and_forbidden_entitlements(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    executable = bundle / 'alphonse-protected-worker'
    dependency = bundle / 'python.dylib'
    executable.write_bytes(bytes.fromhex('cffaedfe'))
    dependency.write_bytes(bytes.fromhex('cffaedfe'))
    observed = []

    def command(args):
        observed.append(args)
        return f'flags=0x10000(runtime)\nTeamIdentifier={TEAM}\n'

    monkeypatch.setattr(worker_build, '_run', command)
    monkeypatch.setattr(worker_build, '_entitlements', lambda _: {})
    worker_build._verify_bundle(bundle, TEAM)
    assert sum('--verify' in command for command in observed) == 2
    with pytest.raises(worker_build.BuildError, match='bundle_team_identity_mismatch'):
        worker_build._verify_bundle(bundle, 'XXXXXXXXXX')
    monkeypatch.setattr(worker_build, '_entitlements',
                        lambda _: {'com.apple.security.cs.disable-library-validation': True})
    with pytest.raises(worker_build.BuildError, match='worker_entitlement_prohibited'):
        worker_build._verify_bundle(bundle, TEAM)
    monkeypatch.setattr(worker_build, '_run', lambda args: f'flags=0x0()\nTeamIdentifier={TEAM}\n')
    with pytest.raises(worker_build.BuildError, match='worker_hardened_runtime_missing'):
        worker_build._verify_bundle(bundle, TEAM)
