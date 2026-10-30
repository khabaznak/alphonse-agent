import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { currentHousehold, HouseholdWorkspace, type HouseholdOverview } from "./HouseholdWorkspace";

const snapshot = { member: { user_id: "member", display_name: "Member" } } as HouseholdOverview;
describe("Household audience boundary", () => {
  it("discards snapshots from another member or a previous disclosure choice", () => {
    expect(currentHousehold(snapshot, "member", false, "member:false")).toBe(snapshot);
    expect(currentHousehold(snapshot, "admin", false, "member:false")).toBeNull();
    expect(currentHousehold(snapshot, "member", true, "member:false")).toBeNull();
    expect(currentHousehold(null, "member", false, "member:false")).toBeNull();
  });
  it("has a shared default, clear member context and admin-only member management", () => {
    const props = { user: "member", active: true, connected: true, isAdmin: false, onProject: () => {}, onAgenda: () => {}, onConversation: () => {}, onQuestions: () => {}, onMembers: () => {} };
    const html = renderToStaticMarkup(<HouseholdWorkspace {...props} />);
    expect(html).toContain('aria-pressed="true">Shared');
    expect(html).toContain("Current member:");
    expect(html).not.toContain("Manage members");
    expect(renderToStaticMarkup(<HouseholdWorkspace {...props} isAdmin />)).toContain("Manage members");
    expect(renderToStaticMarkup(<HouseholdWorkspace {...props} connected={false} />)).toContain("unavailable while disconnected");
  });
});
