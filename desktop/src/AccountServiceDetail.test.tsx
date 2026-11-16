import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { AccountServiceDetail, type AccountServiceDetailProps } from "./AccountServiceDetail";
import type { Catalog, Connection } from "./integrationManagement";

const account = (overrides: Partial<Connection> = {}): Connection => ({
  connection_id: "one", provider: "google", display_name: "Gmail", revision: 1,
  status: "connected", health_code: "ok", capabilities: ["mail_read"], ...overrides,
});
const catalog: Catalog = { actor_user_id: "owner", owner_gmail_setup_only: true,
  providers: [{ provider: "google", capabilities: ["mail_read"] }], connections: [],
  grants: [], members: [], sources: [], skills: [], bindings: [], projects: [] };
const props = (value: Partial<AccountServiceDetailProps> = {}): AccountServiceDetailProps => ({
  catalog, provider: "google", capability: "mail_read", busy: false,
  onRefresh: vi.fn(), onConnect: vi.fn(), onReconnect: vi.fn(), onGrant: vi.fn(),
  onPause: vi.fn(), onDisconnect: vi.fn(), onRetryDeletion: vi.fn(), ...value,
});

describe("account service detail", () => {
  it("keeps same-named records separate and labels their distinct states", () => {
    const value = { ...catalog, connections: [account({ access_token_status: "refresh_required" }),
      account({ connection_id: "two", status: "reauth_required", capabilities: [] })] };
    const html = renderToStaticMarkup(<AccountServiceDetail {...props({ catalog: value })} />);
    expect(html).toContain("Gmail · account 1");
    expect(html).toContain("Gmail · account 2");
    expect(html).toContain("Reconnect required");
    expect(html).toContain("Finish connecting");
    expect(html).not.toContain("Capabilities for next consent");
    expect(html).toContain("<details class=\"integration-connect-form\"><summary>Add another account</summary>");
  });
  it("disables reconnect for an owner account with additional service permissions", () => {
    const value = { ...catalog, connections: [account({ status: "reauth_required", capabilities: ["mail_read", "calendar_read"] })] };
    const html = renderToStaticMarkup(<AccountServiceDetail {...props({ catalog: value })} />);
    expect(html).toContain("preserve its existing permissions");
    expect(html).toMatch(/<button type="button" disabled="">Finish connecting/);
  });
  it("uses Outlook-specific wording and reports missing provider configuration", () => {
    const html = renderToStaticMarkup(<AccountServiceDetail {...props({ provider: "microsoft" })} />);
    expect(html).toContain("Outlook / Hotmail");
    expect(html).toContain("Provider setup is required");
    expect(html).toContain("Continue to Microsoft");
    expect(html).not.toContain("Continue to Google");
  });
  it("explains when a connected Gmail account still needs an INBOX grant", () => {
    const html = renderToStaticMarkup(<AccountServiceDetail {...props({ catalog: { ...catalog, connections: [account()] } })} />);
    expect(html).toContain("Grant INBOX access before asking for the unread count");
    expect(html).toContain("Manage access");
  });
  it("does not label unavailable credentials or another service as connected", () => {
    const accounts = [account({ credential_health: 'credential_backend_unavailable', access_token_status: 'unavailable' }), account({ connection_id: 'two', capabilities: ['calendar_read'] })];
    const html = renderToStaticMarkup(<AccountServiceDetail {...props({ catalog: { ...catalog, connections: accounts } })} />);
    expect(html).toContain('Credentials unavailable');
    expect(html).toContain('Service not enabled');
    expect(html).not.toContain('Manage access');
  });
  it("does not claim a generic Outlook token needs reconnect merely because refresh is required", () => {
    const microsoft = account({ provider: 'microsoft', access_token_status: 'refresh_required' });
    const html = renderToStaticMarkup(<AccountServiceDetail {...props({ provider: 'microsoft', catalog: { ...catalog, owner_gmail_setup_only: false, connections: [microsoft] } })} />);
    expect(html).toContain('Connected');
    expect(html).not.toContain('Reconnect required');
  });
  it("offers visible removal and retains disconnected records for cleanup", () => {
    const accounts = [account({ status: 'reauth_required', capabilities: [] }), account({ connection_id: 'two', status: 'disconnected', credential_deletion_pending: true })];
    const html = renderToStaticMarkup(<AccountServiceDetail {...props({ catalog: { ...catalog, connections: accounts } })} />);
    expect(html).toContain('Remove account…');
    expect(html).toContain('Removed accounts (1) · cleanup needs attention');
    expect(html).toContain('Retry credential deletion');
  });
});
