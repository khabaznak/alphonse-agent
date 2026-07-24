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
