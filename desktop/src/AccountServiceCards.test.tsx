import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { AccountServiceCards } from "./OptionalIntegrations";
import type { Catalog } from "./integrationManagement";

const catalog: Catalog = { actor_user_id: "owner", connections: [{ connection_id: "mail", provider: "google", display_name: "Home", revision: 1, status: "connected", health_code: "ok", capabilities: ["mail_read"] }], grants: [], members: [], providers: [{ provider: "google", capabilities: ["mail_read", "calendar_read"] }], sources: [], skills: [], bindings: [], projects: [] };

describe("email and calendar integrations", () => {
  it("shows Google and Microsoft services with their actual availability", () => {
    const html = renderToStaticMarkup(<AccountServiceCards catalog={catalog} onChoose={() => {}} />);
    expect(html).toContain("Gmail");
    expect(html).toContain("Google Calendar");
    expect(html).toContain("Outlook / Hotmail");
    expect(html).toContain("Outlook Calendar");
    expect(html).toContain("Not yet available");
    expect(html).toContain("1 account connected");
    expect(html).toContain("Available to connect");
    expect(html).toContain("Provider setup required");
  });
  it("does not claim accounts are available when management fails", () => {
    const html = renderToStaticMarkup(<AccountServiceCards catalog={null} unavailable onChoose={() => {}} />);
    expect(html).toContain("Account management unavailable");
    expect(html).not.toContain("Available to connect");
  });
});
