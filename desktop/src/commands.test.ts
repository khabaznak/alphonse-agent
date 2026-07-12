import { describe, expect, it } from "vitest";
import { matchingCommands } from "./commands";

describe("Desktop slash commands", () => {
  it("includes the current TUI parity commands", () => {
    expect(matchingCommands("/agent")).toEqual(["/agent-config"]);
    expect(matchingCommands("/project")).toEqual(["/project", "/project-context"]);
  });
});
