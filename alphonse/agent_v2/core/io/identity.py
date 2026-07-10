"""Deterministic v2 identity and delivery resolution.

The first implementation wraps the existing v1 identity tables while exposing a
small v2 boundary. CAPD receives canonical Alphonse user ids; provider ids stay
inside integrations and this resolver.
"""

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

    def __init__(self, integrations: tuple[IntegrationIdentity, ...] | None = None) -> None:
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

        service_id = _resolve_service_id(integration.provider_key)
        if service_id is None:
            return IdentityResolution(resolved=False, reason="provider_not_registered")
        user_id = _identity().resolve_user_id(service_id=service_id, service_user_id=provider_user)
        if not user_id:
            return IdentityResolution(resolved=False, reason="user_mapping_not_found")
        return IdentityResolution(resolved=True, alphonse_user_id=user_id)

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

        if preferred_integration_id:
            integration = self.integration_for(integration_id=preferred_integration_id)
            address = self._address_for_integration(user_id=user_id, integration=integration)
            if address is not None:
                return IdentityResolution(resolved=True, alphonse_user_id=user_id, address=address)
            return IdentityResolution(resolved=False, alphonse_user_id=user_id, reason="delivery_mapping_not_found")

        if fallback_address is not None and fallback_address.alphonse_user_id == user_id:
            return IdentityResolution(resolved=True, alphonse_user_id=user_id, address=fallback_address)

        preferred_service_id = _identity().get_preferred_service_id(user_id)
        if preferred_service_id is not None:
            provider_key = _identity().resolve_service_key(preferred_service_id)
            integration = self.integration_for(provider_key=provider_key or "")
            address = self._address_for_integration(user_id=user_id, integration=integration, service_id=preferred_service_id)
            if address is not None:
                return IdentityResolution(resolved=True, alphonse_user_id=user_id, address=address)

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

    def _address_for_integration(
        self,
        *,
        user_id: str,
        integration: IntegrationIdentity,
        service_id: int | None = None,
    ) -> ChannelAddress | None:
        provider = str(integration.provider_key or "").strip().lower()
        if provider == "tui":
            return ChannelAddress(
                integration_id=str(integration.integration_id or "tui"),
                provider_key="tui",
                channel_target=user_id,
                alphonse_user_id=user_id,
                provider_user_id=user_id,
            )
        resolved_service_id = service_id if service_id is not None else _resolve_service_id(provider)
        if resolved_service_id is None:
            return None
        target = _identity().resolve_delivery_target(user_id=user_id, service_id=resolved_service_id)
        if not target:
            return None
        return ChannelAddress(
            integration_id=str(integration.integration_id or provider),
            provider_key=provider,
            channel_target=target,
            alphonse_user_id=user_id,
            provider_user_id=target,
        )


def resolve_provider_user_mapping(*, alphonse_user_id: str, provider_key: str) -> str | None:
    """Return the provider user id mapped to a canonical Alphonse user."""
    user_id = str(alphonse_user_id or "").strip()
    if not user_id:
        return None
    service_id = _resolve_service_id(provider_key)
    if service_id is None:
        return None
    return _identity().resolve_service_user_id(user_id=user_id, service_id=service_id)


def upsert_provider_user_mapping(
    *,
    alphonse_user_id: str,
    provider_key: str,
    provider_user_id: str,
    display_name: str = "",
    is_active: bool = True,
) -> str:
    """Bind a provider user id to a canonical Alphonse user via v1 identity tables."""
    user_id = str(alphonse_user_id or "").strip()
    provider_user = str(provider_user_id or "").strip()
    if not user_id:
        raise ValueError("alphonse_user_id_required")
    if not provider_user:
        raise ValueError("provider_user_id_required")
    service_id = _resolve_service_id(provider_key)
    if service_id is None:
        raise ValueError("provider_not_registered")

    existing_user_id = _identity().resolve_user_id(service_id=service_id, service_user_id=provider_user)
    if existing_user_id and str(existing_user_id) != user_id:
        raise ValueError("provider_user_already_mapped")

    if _identity().get_user(user_id) is None:
        _identity().upsert_user(
            {
                "user_id": user_id,
                "display_name": str(display_name or user_id).strip() or user_id,
                "is_active": True,
            }
        )
    return _identity().upsert_service_user_id(
        user_id=user_id,
        service_id=service_id,
        service_user_id=provider_user,
        is_active=is_active,
    )


def _resolve_service_id(provider_key: str) -> int | None:
    provider = str(provider_key or "").strip().lower()
    if not provider:
        return None
    if provider == "tui":
        return None
    return _identity().resolve_service_id(provider)


def _identity() -> Any:
    from alphonse.agent import identity

    return identity
