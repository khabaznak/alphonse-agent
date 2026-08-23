import { describe, expect, it } from "vitest";
import { applyA2uiEvent, localDismissSurfaceId, readPointer, writePointer, type A2uiComponent, type A2uiSurface } from "./a2ui";

describe("A2UI surface state", () => {
  it("creates, updates, and deletes only catalog-approved surfaces", () => {
    const created = applyA2uiEvent({}, {
      type: "CUSTOM", name: "a2ui.envelope", value: {
        version: "v0.9.1", createSurface: { surfaceId: "question:q1", catalogId: "alphonse.desktop.catalog.v2" },
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

  it("accepts dismiss as a client-only action only for its scheduled-task surface", () => {
    const surface: A2uiSurface = { surfaceId: "scheduled-task:one", catalogId: "alphonse.desktop.catalog.v2", components: {}, dataModel: {} };
    const dismiss: A2uiComponent = { id: "dismiss", component: "Button", action: { name: "dismiss_surface", context: { surface_id: "scheduled-task:one" } } };

    expect(localDismissSurfaceId(surface, dismiss)).toBe("scheduled-task:one");
    expect(localDismissSurfaceId({ ...surface, surfaceId: "question:one" }, dismiss)).toBeNull();
    expect(localDismissSurfaceId(surface, { ...dismiss, action: { ...dismiss.action!, context: { surface_id: "scheduled-task:other" } } })).toBeNull();
  });

  it("supports typed v2 components and generic JSON Pointer bindings", () => {
    const model = writePointer({ answer: { values: ["one"] } }, "/answer/when", "2026-08-20T12:00");
    expect(readPointer(model, "/answer/when")).toBe("2026-08-20T12:00");
    const created = applyA2uiEvent({}, { type: "CUSTOM", name: "a2ui.envelope", value: { version: "v0.9.1", createSurface: { surfaceId: "v2", catalogId: "alphonse.desktop.catalog.v2" } } });
    const updated = applyA2uiEvent(created, { type: "CUSTOM", name: "a2ui.envelope", value: { version: "v0.9.1", updateComponents: { surfaceId: "v2", components: [{ id: "table", component: "Table", columns: [{ id: "name", label: "Name" }], rows: [{ cells: { name: "One" } }] }, { id: "bad-icon", component: "Icon", name: "untrusted" }] } } });
    expect(updated.v2.components.table.component).toBe("Table");
    expect(updated.v2.components["bad-icon"]).toBeUndefined();
  });
});
