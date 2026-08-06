from pathlib import Path

import pytest

from alphonse.agent_v2.assets import AttachmentDescriptor, MAX_ATTACHMENT_BYTES, SQLiteAssetStore


def test_asset_is_durable_private_and_deletable(tmp_path: Path) -> None:
    store = SQLiteAssetStore(tmp_path / "assets.sqlite", tmp_path / "files")
    asset = store.register_bytes(owner_user_id="gaby", descriptor=AttachmentDescriptor("recipe.pdf", "application/pdf", 3), content=b"pdf", source="telegram")
    assert Path(asset.path).read_bytes() == b"pdf"
    assert store.get(asset.asset_id, requester_user_id="alex") is None
    assert store.get(asset.asset_id, requester_user_id="alex", is_admin=lambda user: user == "alex") is not None
    assert store.delete(asset.asset_id, requester_user_id="gaby") is True
    assert not Path(asset.path).exists()


def test_asset_rejects_unsupported_and_oversize(tmp_path: Path) -> None:
    store = SQLiteAssetStore(tmp_path / "assets.sqlite", tmp_path / "files")
    with pytest.raises(ValueError, match="attachment_type_unsupported"):
        store.register_bytes(owner_user_id="gaby", descriptor=AttachmentDescriptor("x.exe", "application/octet-stream", 1), content=b"x", source="desktop")
    with pytest.raises(ValueError, match="attachment_too_large"):
        store.register_bytes(owner_user_id="gaby", descriptor=AttachmentDescriptor("x.pdf", "application/pdf", MAX_ATTACHMENT_BYTES + 1), content=b"x" * (MAX_ATTACHMENT_BYTES + 1), source="desktop")


def test_assets_are_written_to_the_owner_profile_and_legacy_assets_migrate(tmp_path: Path) -> None:
    users_root = tmp_path / "users"
    legacy_root = tmp_path / "legacy-assets"
    store = SQLiteAssetStore(tmp_path / "assets.sqlite", legacy_root, users_root=lambda: users_root)

    asset = store.register_bytes(owner_user_id="gaby", descriptor=AttachmentDescriptor("note.ogg", "audio/ogg", 3), content=b"ogg", source="telegram")

    assert Path(asset.path) == users_root / "gaby" / "assets" / asset.asset_id / "original.ogg"
    assert Path(asset.path).read_bytes() == b"ogg"

    legacy_store = SQLiteAssetStore(tmp_path / "legacy.sqlite", legacy_root)
    legacy = legacy_store.register_bytes(owner_user_id="alex", descriptor=AttachmentDescriptor("photo.jpg", "image/jpeg", 3), content=b"jpg", source="telegram")
    migrator = SQLiteAssetStore(tmp_path / "legacy.sqlite", legacy_root, users_root=lambda: users_root)

    assert migrator.migrate_to_user_directories() == {"migrated": 1, "missing": 0}
    moved = migrator.system_get(legacy.asset_id)
    assert moved is not None
    assert Path(moved.path) == users_root / "alex" / "assets" / legacy.asset_id / "original.jpg"
    assert Path(moved.path).read_bytes() == b"jpg"
    assert not Path(legacy.path).exists()
