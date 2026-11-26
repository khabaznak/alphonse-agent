// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OptionalAccounts } from "./OptionalIntegrations";
import { OUTLOOK_INBOX, OUTLOOK_INBOX_UNREAD, type Catalog, type Connection, type Grant } from "./integrationManagement";

const invoke = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/core", () => ({ invoke }));
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const catalog: Catalog = {
  actor_user_id: "owner", owner_gmail_setup_only: true, owner_google_setup_only: true, owner_outlook_inbox_setup: true,
  providers: [{ provider: "microsoft", capabilities: [OUTLOOK_INBOX_UNREAD] }],
  members: [{ user_id: "owner", display_name: "Owner" }], connections: [], grants: [],
  sources: [], skills: [], projects: [], bindings: [],
};
const account: Connection = {
  connection_id: "microsoft", provider: "microsoft", display_name: "Family email", revision: 4,
  status: "connected", health_code: "healthy", capabilities: [OUTLOOK_INBOX_UNREAD], access_token_status: "current",
};
const activeGrant: Grant = {
  connection_id: account.connection_id, grant_id: "inbox", grantor_user_id: "owner", beneficiary_user_id: "owner",
  operations: [OUTLOOK_INBOX_UNREAD], resources: [OUTLOOK_INBOX], recipient_user_ids: ["owner"],
  project_id: null, expires_at: "2099-01-01T10:00:00Z", revision: 2, revoked_at: "",
};

async function mountAccounts(options: { outlookRequested?: boolean; onConversation?: () => void } = {}) {
  const host = document.body.appendChild(document.createElement("div"));
  const root = createRoot(host);
  await act(async () => root.render(<OptionalAccounts identity="owner" {...options} />));
  const content = () => host.textContent || "";
  const button = (name: string) => {
    const target = [...host.querySelectorAll("button")]
      .find(node => (node.getAttribute("aria-label") || node.textContent) === name);
    if (!target) throw new Error(`Could not find ${name} in ${content()}`);
    return target;
  };
  const click = async (name: string) => act(async () => button(name).click());
  const enter = async (labelText: string, value: string) => act(async () => {
    const label = Array.from(host.querySelectorAll("label")).find(item => item.textContent === labelText)!;
    const input = document.getElementById(label.htmlFor) as HTMLInputElement;
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const close = async () => { await act(async () => root.unmount()); host.remove(); };
  return { host, content, button, click, enter, close };
}

describe("owner Outlook / Hotmail Inbox settings", () => {
  afterEach(() => { invoke.mockReset(); vi.useRealTimers(); });

  it("follows consent automatically and grants and revokes only the named private Inbox count", async () => {
    vi.useFakeTimers();
    let connections: Connection[] = [];
    let grants: Grant[] = [];
    let consentComplete = false;
    const saveGrant = vi.fn((values: Grant) => { grants = [{ ...values, grantor_user_id: "owner", revision: 1, revoked_at: "" }]; });
    const revokeGrant = vi.fn((_values: { grant_id: string; expected_revision: number }) => {
      grants = grants.map(saved => ({ ...saved, revoked_at: "now" }));
    });
    const onConversation = vi.fn();
    invoke.mockImplementation(async (command, { params }) => {
      if (command === "open_account_consent") return undefined;
      switch (params.action) {
        case "catalog": return { ...catalog, connections, grants };
        case "begin": return { authorization_url: "https://login.microsoftonline.com/mock", attempt_id: "consent-1" };
        case "oauth_attempt_status":
          if (consentComplete) connections = [account];
          return { status: consentComplete ? "connected" : "pending" };
        case "save_grant": return saveGrant(params.values);
        case "revoke_grant": return revokeGrant(params.values);
      }
    });
    const ui = await mountAccounts({ onConversation });
    try {
      await ui.click("Connect Outlook / Hotmail");
      expect(document.activeElement?.textContent).toBe("Outlook / Hotmail");
      await ui.enter("Account name in Alphonse", "Family email");
      await ui.click("Continue to Microsoft");
      expect(invoke).toHaveBeenCalledWith("daemon_request", expect.objectContaining({ params: {
        action: "begin", values: expect.objectContaining({
          provider: "microsoft", capabilities: [OUTLOOK_INBOX_UNREAD], display_name: "Family email", expected_revision: 0,
        }),
      }}));
      expect(invoke).toHaveBeenCalledWith("open_account_consent", { url: "https://login.microsoftonline.com/mock" });
      expect(ui.content()).toContain("Finish Microsoft sign-in in your browser");
      expect(ui.button("Continue to Microsoft").disabled).toBe(true);
      expect(document.activeElement).toBe(ui.button("Stop sign-in"));
      consentComplete = true;
      await act(async () => vi.advanceTimersByTimeAsync(1000));
      expect(ui.content()).toContain("Microsoft account connected.");
      expect(ui.content()).toContain("Inbox access needed");
      expect(saveGrant).not.toHaveBeenCalled();
      expect(ui.content()).not.toContain("Refresh account status");

      await ui.click("Grant Inbox access for Family email");
      expect(document.activeElement?.textContent).toBe("Inbox access for Family email");
      expect(ui.host.querySelector('input[type="text"]')).toBeNull();
      expect(ui.content()).not.toContain(OUTLOOK_INBOX);
      expect(ui.button("Confirm Inbox access").disabled).toBe(false);
      await ui.enter("Access expires (local time)", "2000-01-01T10:00");
      expect(ui.button("Confirm Inbox access").disabled).toBe(true);
      expect(ui.content()).toContain("Choose a date and time in the future.");
      await ui.enter("Access expires (local time)", "2099-01-01T10:00");
      await ui.click("Confirm Inbox access");
      expect(saveGrant).toHaveBeenCalledWith(expect.objectContaining({
        connection_id: account.connection_id, expected_connection_revision: account.revision,
        beneficiary_user_id: "owner", operations: [OUTLOOK_INBOX_UNREAD], resources: [OUTLOOK_INBOX],
        recipient_user_ids: ["owner"], project_id: null, expires_at: expect.any(String),
      }));
      expect(ui.content()).toContain("Inbox access granted");
      await ui.click("Open Conversation");
      expect(onConversation).toHaveBeenCalledOnce();

      await ui.click("Manage Inbox access for Family email");
      await ui.click("Revoke Inbox access");
      expect(revokeGrant).toHaveBeenCalledWith({ grant_id: grants[0].grant_id, expected_revision: 1 });
      expect(ui.content()).toContain("Inbox access revoked.");
      expect(ui.content()).not.toContain("Open Conversation");
    } finally {
      await ui.close();
    }
  });

  it("stops a pending attempt with the daemon and discards late status responses", async () => {
    let resolveRead: (result: unknown) => void = () => {};
    const status = new Promise(resolve => { resolveRead = resolve; });
    invoke.mockImplementation(async (command, { params }) => {
      if (command === "open_account_consent") return undefined;
      if (params.action === "catalog") return catalog;
      if (params.action === "begin") return { authorization_url: "https://login.microsoftonline.com/mock", attempt_id: "consent-2" };
      if (params.action === "oauth_attempt_status") return status;
      if (params.action === "cancel_oauth_attempt") return { status: "cancelled" };
    });
    const ui = await mountAccounts();
    try {
      await ui.click("Connect Outlook / Hotmail");
      await ui.enter("Account name in Alphonse", "Family email");
      await ui.click("Continue to Microsoft");
      await ui.click("Stop sign-in");
      expect(invoke).toHaveBeenCalledWith("daemon_request", expect.objectContaining({ params: {
        action: "cancel_oauth_attempt", values: { provider: "microsoft", attempt_id: "consent-2" },
      }}));
      await act(async () => resolveRead({ status: "connected" }));
      expect(ui.content()).toContain("Microsoft sign-in was cancelled");
      expect(ui.content()).not.toContain("Microsoft account connected.");
      expect(ui.button("Continue to Microsoft").disabled).toBe(false);
    } finally {
      await ui.close();
    }
  });

  it("opens account recovery directly and refreshes only with a live exact grant", async () => {
    const refresh = vi.fn();
    const connection = { ...account, access_token_status: "refresh_required" };
    invoke.mockImplementation(async (_command, { params }) => {
      if (params.action === "catalog") return { ...catalog, connections: [connection], grants: [activeGrant] };
      if (params.action === "refresh_outlook_inbox_access") {
        refresh(params.values);
        connection.access_token_status = "current";
      }
    });
    const ui = await mountAccounts({ outlookRequested: true });
    try {
      expect(ui.content()).toContain("Microsoft access expired");
      await ui.click("Refresh access for Family email");
      expect(refresh).toHaveBeenCalledWith({ connection_id: "microsoft", expected_revision: 4, grant_id: "inbox", grant_revision: 2 });
      expect(ui.content()).toContain("Inbox access granted");
    } finally {
      await ui.close();
    }
  });

  it("distinguishes account labels and directs a wrong-account reconnect without showing provider text", async () => {
    const wrongAccount = { ...account, connection_id: "other", access_token_status: "refresh_required" };
    invoke.mockImplementation(async (command, { params }) => {
      if (command === "open_account_consent") return undefined;
      if (params.action === "catalog") return { ...catalog, connections: [account, wrongAccount] };
      if (params.action === "begin") return { authorization_url: "https://login.microsoftonline.com/mock", attempt_id: "consent-3" };
      if (params.action === "oauth_attempt_status") return { status: "failed", reason: "account_mismatch", details: "PRIVATE_PROVIDER_TEXT" };
    });
    const ui = await mountAccounts();
    try {
      await ui.click("Manage Outlook / Hotmail");
      expect(ui.content()).toContain("Family email · account 1");
      await ui.click("Reconnect Family email · account 2");
      expect(ui.content()).toContain("Microsoft returned a different account.");
      expect(ui.content()).not.toContain("PRIVATE_PROVIDER_TEXT");
      expect(ui.button("Reconnect Family email · account 2").disabled).toBe(false);
    } finally {
      await ui.close();
    }
  });
});
