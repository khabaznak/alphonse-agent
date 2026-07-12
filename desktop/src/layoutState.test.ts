import { describe, expect, it } from "vitest";
import { agentStateLabel, capdActivityLabel, projectKey } from "./layoutState";

describe("Desktop layout state helpers", () => {
  it("maps CAPD activity into compact node labels", () => {
    expect(capdActivityLabel({ phase: "check", label: "" })).toBe("checking");
    expect(capdActivityLabel({ phase: "decide", label: "" })).toBe("planning");
    expect(capdActivityLabel({ phase: "tool", label: "" })).toBe("doing");
    expect(capdActivityLabel({ phase: "act", label: "" })).toBe("acting");
  });

  it("maps connection and work into top-bar state labels", () => {
    expect(agentStateLabel(false, false, 0)).toBe("Disconnected");
    expect(agentStateLabel(true, true, 0)).toBe("Error");
    expect(agentStateLabel(true, false, 1)).toBe("Working");
    expect(agentStateLabel(true, false, 0)).toBe("Idle");
  });

  it("uses a stable home bucket for blank projects", () => {
    expect(projectKey("")).toBe("__home__");
    expect(projectKey("project-a")).toBe("project-a");
  });
});
