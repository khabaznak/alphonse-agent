import { describe, expect, it } from "vitest";
import { applyA2uiEvent } from "./a2ui";

describe("A2UI surface state", () => {
  it("creates, updates, and deletes only catalog-approved surfaces", () => {
    const created = applyA2uiEvent({}, {
      type: "CUSTOM", name: "a2ui.envelope", value: {
        version: "v0.9.1", createSurface: { surfaceId: "question:q1", catalogId: "alphonse.desktop.catalog.v1" },
      },
    });
    const updated = applyA2uiEvent(created, {
      type: "CUSTOM", name: "a2ui.envelope", value: {
        version: "v0.9.1", updateComponents: { surfaceId: "question:q1", components: [{ id: "root", component: "Card" }] },
      },
    });
    expect(updated["question:q1"].components.root.component).toBe("Card");
    const deleted = applyA2uiEvent(updated, {
      type: "CUSTOM", name: "a2ui.envelope", value: { version: "v0.9.1", deleteSurface: { surfaceId: "question:q1" } },
    });
    expect(deleted).toEqual({});
  });
});
