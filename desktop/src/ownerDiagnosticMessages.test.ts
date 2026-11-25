import { describe, expect, it } from "vitest";
import { ownerFailureMessage } from "./ownerConversation";

describe("owner failure messages", () => {
  it("distinguishes time, model and reservation failures without exposing raw text", () => {
    expect(ownerFailureMessage("time_limit")).toContain("time limit");
    expect(ownerFailureMessage("model_timeout")).toContain("provider");
    expect(ownerFailureMessage("input_budget_limit")).toContain("reserved");
    expect(ownerFailureMessage("model_unavailable")).toContain("specific cause was not recorded");
    expect(ownerFailureMessage("model_timeout: private content")).not.toContain("private content");
  });
});
