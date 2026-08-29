import { describe, expect, it } from "vitest";
import { avatarState, avatarStateLabel, capdActivityLabel, projectKey } from "./layoutState";

describe("Desktop layout state helpers", () => {
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

  it("uses a stable home bucket for blank projects", () => {
    expect(projectKey("")).toBe("__home__");
    expect(projectKey("project-a")).toBe("project-a");
  });
});
