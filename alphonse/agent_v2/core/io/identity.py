"""Deterministic v2 identity and delivery resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent_v2.core.io.channels import ChannelAddress


@dataclass(frozen=True)
class IntegrationIdentity:
    """A configured integration instance."""

    integration_id: str
    provider_key: str


@dataclass(frozen=True)
class IdentityResolution:
    """Resolution result for user/address lookups."""

    resolved: bool
    alphonse_user_id: str = ""
    address: ChannelAddress | None = None
    reason: str = ""


class V2IdentityResolver:
    """Resolve provider identities and preferred outbound addresses."""

    def __init__(self, integrations: tuple[IntegrationIdentity, ...] | None = None, *, user_store: Any | None = None) -> None:
        configured = integrations or (IntegrationIdentity("tui", "tui"),)
        self._by_id = {
            str(item.integration_id or "").strip(): item
            for item in configured
            if str(item.integration_id or "").strip()
        }
        self._by_provider: dict[str, IntegrationIdentity] = {}
        for item in configured:
            provider = str(item.provider_key or "").strip().lower()
            if provider and provider not in self._by_provider:
                self._by_provider[provider] = item
        self._user_store = user_store

    def resolve_inbound_user(
        self,
        *,
        integration_id: str,
        provider_user_id: str,
        provider_key: str = "",
    ) -> IdentityResolution:
        """Resolve an inbound provider user to a canonical Alphonse user id."""
        integration = self.integration_for(integration_id=integration_id, provider_key=provider_key)
        provider_user = str(provider_user_id or "").strip()
        if not provider_user:
            return IdentityResolution(resolved=False, reason="provider_user_id_required")
        if integration.provider_key == "tui":
            return IdentityResolution(resolved=True, alphonse_user_id=provider_user)

        if self._user_store is not None:
            address = self._user_store.address_for_inbound(
                integration_id=integration.integration_id, provider_user_id=provider_user
            )
            if address is None:
                return IdentityResolution(resolved=False, reason="user_mapping_not_found")
            return IdentityResolution(resolved=True, alphonse_user_id=address.user_id)

        return IdentityResolution(resolved=False, reason="user_mapping_not_found")

    def resolve_outbound_address(
        self,
        *,
        alphonse_user_id: str,
        preferred_integration_id: str = "",
        fallback_address: ChannelAddress | None = None,
    ) -> IdentityResolution:
        """Resolve a canonical user id to a preferred concrete integration address."""
        user_id = str(alphonse_user_id or "").strip()
        if not user_id:
            return IdentityResolution(resolved=False, reason="alphonse_user_id_required")

        if fallback_address is not None and fallback_address.alphonse_user_id == user_id:
            return IdentityResolution(resolved=True, alphonse_user_id=user_id, address=fallback_address)

        if self._user_store is not None:
            address_record = self._user_store.address_for_outbound(user_id, integration_id=preferred_integration_id)
            if address_record is None:
                return IdentityResolution(resolved=False, alphonse_user_id=user_id, reason="preferred_delivery_not_found")
            return IdentityResolution(
                resolved=True,
                alphonse_user_id=user_id,
                address=ChannelAddress(
                    integration_id=address_record.integration_id,
                    provider_key=address_record.provider_key,
                    channel_target=address_record.channel_target,
                    alphonse_user_id=user_id,
                    provider_user_id=address_record.provider_user_id,
                ),
            )

        return IdentityResolution(resolved=False, alphonse_user_id=user_id, reason="preferred_delivery_not_found")

    def integration_for(self, *, integration_id: str = "", provider_key: str = "") -> IntegrationIdentity:
        """Return a configured integration, defaulting safely for simple providers."""
        integration_key = str(integration_id or "").strip()
        if integration_key and integration_key in self._by_id:
            return self._by_id[integration_key]
        provider = str(provider_key or "").strip().lower()
        if provider and provider in self._by_provider:
            return self._by_provider[provider]
        if integration_key:
            return IntegrationIdentity(integration_id=integration_key, provider_key=provider or integration_key.lower())
        if provider:
            return IntegrationIdentity(integration_id=provider, provider_key=provider)
        return IntegrationIdentity(integration_id="tui", provider_key="tui")

def resolve_provider_user_mapping(*, user_store: Any, alphonse_user_id: str, provider_key: str) -> str | None:
    """Return the provider user id mapped to a canonical Alphonse user."""
    user_id = str(alphonse_user_id or "").strip()
    if not user_id:
        return None
    address = user_store.address_for_outbound(user_id, integration_id=str(provider_key or "").strip())
    return address.provider_user_id if address is not None else None


def upsert_provider_user_mapping(
    *,
    user_store: Any,
    alphonse_user_id: str,
    provider_key: str,
    provider_user_id: str,
    display_name: str = "",
    is_active: bool = True,
) -> str:
    """Bind a provider user id to a canonical Alphonse user in the v2 store."""
    user_id = str(alphonse_user_id or "").strip()
    provider_user = str(provider_user_id or "").strip()
    if not user_id:
        raise ValueError("alphonse_user_id_required")
    if not provider_user:
        raise ValueError("provider_user_id_required")
    provider = str(provider_key or "").strip().lower()
    if not provider:
        raise ValueError("provider_key_required")
    if user_store.get_user(user_id) is None:
        user_store.create_user(display_name=str(display_name or user_id).strip() or user_id, user_id=user_id, is_active=is_active)
    address = user_store.bind_address(
        user_id=user_id,
        integration_id=provider,
        provider_key=provider,
        provider_user_id=provider_user,
        is_active=is_active,
    )
    return address.address_id
