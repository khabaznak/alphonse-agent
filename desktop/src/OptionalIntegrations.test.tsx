// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OptionalAccounts } from "./OptionalIntegrations";
import type { Catalog, Grant } from "./integrationManagement";

const invoke = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/core", () => ({ invoke }));
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const inbox: Grant = {
  grant_id: "inbox", connection_id: "gmail", grantor_user_id: "owner", beneficiary_user_id: "owner", project_id: null,
  operations: ["mail_content_read"], resources: ["INBOX"], recipient_user_ids: ["owner"],
  expires_at: null, revision: 1, revoked_at: "",
};
const catalog: Catalog = {
  actor_user_id: "owner", owner_gmail_setup_only: true, owner_gmail_content_setup: true, owner_gmail_label_setup: true,
  connections: [{ connection_id: "gmail", provider: "google", display_name: "Personal mail", revision: 4,
    status: "connected", health_code: "healthy", capabilities: ["mail_read"], access_token_status: "refresh_required" }],
  providers: [{ provider: "google", capabilities: ["mail_read"] }], grants: [inbox],
  members: [{ user_id: "owner", display_name: "Owner" }], sources: [], skills: [], bindings: [], projects: [],
};

async function mountAccounts() {
  const element = document.createElement("div");
  document.body.append(element);
  const root = createRoot(element);
  await act(async () => root.render(<OptionalAccounts identity="owner" />));
  const button = (name: string) => {
    const result = [...element.querySelectorAll("button")].find(node => (node.getAttribute("aria-label") || node.textContent) === name);
    if (!result) throw new Error(`Could not find button ${name}`);
    return result;
  };
  const click = (name: string) => act(async () => button(name).click());
  const select = (name: string) => {
    const label = [...element.querySelectorAll("label")].find(node => node.textContent === name);
    return element.querySelector<HTMLSelectElement>(`select[id="${label?.htmlFor}"]`)!;
  };
  const choose = (name: string, value: string) => act(async () => {
    const picker = select(name);
    picker.value = value;
    picker.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const close = async () => { await act(async () => root.unmount()); element.remove(); };
  return { element, button, click, select, choose, close };
}

describe("Google account permissions in Desktop settings", () => {
  afterEach(() => invoke.mockReset());

  it("loads nested labels on request, saves a standing grant, and revises or revokes that label only", async () => {
    let grants = [inbox];
    const listLabels = vi.fn((_values: unknown) => ({ labels: [{ id: "INBOX", name: "INBOX" }, { id: "Label_5", name: "School/Notices" }] }));
    const saveGrant = vi.fn((values: Grant) => {
      grants = [...grants.filter(g => g.grant_id !== values.grant_id), { ...values, grantor_user_id: "owner", revoked_at: "", revision: 2 }];
    });
    const revokeGrant = vi.fn(({ grant_id }: { grant_id: string }) => {
      grants = grants.map(g => g.grant_id === grant_id ? { ...g, revoked_at: "now" } : g);
    });
    invoke.mockImplementation(async (_command, { params }) => {
      switch (params.action) {
        case "catalog": return { ...catalog, grants };
        case "list_gmail_labels": return listLabels(params.values);
        case "save_grant": return saveGrant(params.values);
        case "revoke_grant": return revokeGrant(params.values);
      }
    });
    const ui = await mountAccounts();
    try {
      await ui.click("Manage Gmail");
      expect(listLabels).not.toHaveBeenCalled();
      expect(ui.button("Manage Inbox content access").disabled).toBe(false);
      await ui.click("Add label access");
      expect(listLabels).toHaveBeenCalledWith({ connection_id: "gmail", expected_revision: 4 });
      expect(ui.element.textContent).toContain("matching excerpts go to your configured AI model");
      expect([...ui.select("Gmail label").options].map(option => option.textContent))
        .toEqual(["Choose a label", "School/Notices"]);
      expect(ui.select("Access duration").value).toBe("standing");
      expect(ui.button("Authorize label access").disabled).toBe(true);
      await ui.choose("Gmail label", "Label_5");
      await ui.click("Authorize label access");
      expect(saveGrant).toHaveBeenCalledWith(expect.objectContaining({ connection_id: "gmail",
        operations: ["mail_content_read"], resources: ["Label_5"], gmail_label_name: "School/Notices",
        recipient_user_ids: ["owner"], expires_at: null }));
      expect(ui.element.textContent).toContain("School/Notices · Until revoked · Granted");
      await ui.click("← Gmail settings");
      await ui.click("Manage access to School/Notices");
      expect(ui.select("Gmail label").disabled).toBe(true);
      expect(ui.select("Access duration").value).toBe("standing");
      await ui.click("Cancel");
      await ui.click("Revoke access to School/Notices");
      expect(revokeGrant).toHaveBeenCalledWith({ grant_id: grants[1].grant_id, expected_revision: 2 });
      expect(grants[0]).toEqual(inbox);
    } finally {
      await ui.close();
    }
  });

  it("leaves unread-count access time-limited and directs expired Google consent back to Gmail settings", async () => {
    invoke.mockImplementation(async (_command, { params }) => {
      if (params.action === "catalog") return catalog;
      if (params.action === "list_gmail_labels") throw "OAuthError: oauth_reauth_required";
    });
    const ui = await mountAccounts();
    try {
      await ui.click("Manage Gmail");
      await ui.click("Manage unread-count access");
      expect(ui.element.querySelector("input[type=datetime-local]")).not.toBeNull();
      expect(ui.element.textContent).not.toContain("Choose an end date");
      await ui.click("← Gmail settings");
      await ui.click("Add label access");
      expect(ui.element.textContent).toContain("Return to Gmail settings and reconnect the account");
      expect(ui.element.querySelector("select")).toBeNull();
    } finally {
      await ui.close();
    }
  });

  it("keeps keyboard focus in Calendar editing settings when entering the exact grant form", async () => {
    const resource = "fixture@group.calendar.google.com";
    const calendarCatalog = {
      ...catalog, owner_google_setup_only: true,
      connections: [{ ...catalog.connections[0], connection_id: "calendar-edit", capabilities: ["calendar_write"] }],
      providers: [{ provider: "google", capabilities: ["calendar_write"] }],
      grants: [{ ...inbox, connection_id: "calendar-edit", operations: ["calendar_write"], resources: [resource],
        expires_at: new Date(Date.now() + 3600_000).toISOString() }],
    };
    invoke.mockImplementation(async (_command, { params }) => {
      if (params.action === "catalog") return calendarCatalog;
    });
    const ui = await mountAccounts();
    try {
      await ui.click("Manage Google Calendar editing");
      expect(document.activeElement?.textContent).toBe("Google Calendar editing");
      await ui.click("Manage access");
      expect(document.activeElement?.textContent).toBe("Allow approved edits to one Google calendar");
      const input = ui.element.querySelector<HTMLInputElement>("#owner-calendar-id");
      expect(input?.readOnly).toBe(true);
      expect(input?.value).toBe(resource);
      expect(input?.getAttribute("aria-describedby")).toBe("owner-calendar-id-help");
    } finally {
      await ui.close();
    }
  });
});
