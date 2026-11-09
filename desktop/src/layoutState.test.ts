import { describe, expect, it } from "vitest";
import { avatarState, avatarStateLabel, capdActivityLabel, canOpenSettingsSection, isCurrentDesktopAdmin, projectKey } from "./layoutState";

describe("Desktop layout state helpers", () => {
  it("recognizes the active administrator returned by the daemon without relying on a profile role label", () => {
    expect(isCurrentDesktopAdmin({ onboarded: true, user: { is_active: true } })).toBe(true);
    expect(isCurrentDesktopAdmin({ onboarded: true, user: { is_active: false } })).toBe(false);
    expect(isCurrentDesktopAdmin({ onboarded: false, user: null })).toBe(false);
  });
  it("maps CAPD activity into compact node labels", () => {
    expect(capdActivityLabel({ phase: "check", label: "" })).toBe("checking");
    expect(capdActivityLabel({ phase: "decide", label: "" })).toBe("planning");
    expect(capdActivityLabel({ phase: "tool", label: "" })).toBe("doing");
    expect(capdActivityLabel({ phase: "act", label: "" })).toBe("acting");
  });

  it("resolves avatar states with connection and error precedence", () => {
    expect(avatarState(false, "", "planning")).toBe("disconnected");
    expect(avatarState(true, "Request failed", "doing")).toBe("error");
    expect(avatarState(true, "", "planning")).toBe("planning");
    expect(avatarState(true, "", "doing")).toBe("doing");
    expect(avatarState(true, "", "checking")).toBe("checking");
    expect(avatarState(true, "", "acting")).toBe("acting");
    expect(avatarState(true, "", "unknown")).toBe("idle");
    expect(avatarStateLabel("checking")).toBe("Checking");
  });

  it("keeps optional account settings available to a household member", () => {
    expect(canOpenSettingsSection("integrations", false)).toBe(true);
    expect(canOpenSettingsSection("skills", false)).toBe(false);
    expect(canOpenSettingsSection("skills", true)).toBe(true);
  });

  it("uses a stable home bucket for blank projects", () => {
    expect(projectKey("")).toBe("__home__");
    expect(projectKey("project-a")).toBe("project-a");
  });
});
