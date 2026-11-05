import { describe, expect, it } from "vitest";
import { DESKTOP_CATALOG_ID, type A2uiSurface, type AgUiEvent } from "./a2ui";
import { projectTaskProgress } from "./taskProgress";

const old: A2uiSurface = {
  surfaceId: "task-progress:old", catalogId: DESKTOP_CATALOG_ID,
  components: { step_0: { id: "step_0", component: "Text", text: "Attempt 1 · failed" } }, dataModel: {},
};
const envelope = (value: object): { event: AgUiEvent } => ({
  event: { type: "CUSTOM", name: "a2ui.envelope", value: { version: "v0.9.1", ...value } },
});
const create = (id: string) => envelope({ createSurface: { surfaceId: `task-progress:${id}`, catalogId: DESKTOP_CATALOG_ID } });
const current = { [old.surfaceId]: old };

describe("task progress lifecycle", () => {
  it("preserves a closed failed attempt without requiring a final delivery", () => {
    const result = projectTaskProgress(current, {}, [envelope({ deleteSurface: { surfaceId: old.surfaceId } })], false);
    expect(result.surfaces).toEqual({});
    expect(result.archived.old).toEqual(old);
  });
  it("preserves independent task activity across polling and closes only its own task", () => {
    const result = projectTaskProgress(current, {}, [create("one"), create("two")], false);
    expect(result.archived).toEqual({});
    expect(Object.keys(result.surfaces)).toHaveLength(3);
    const next = projectTaskProgress(result.surfaces, result.archived, [envelope({ deleteSurface: { surfaceId: "task-progress:one" } })], false);
    expect(Object.keys(next.archived)).toEqual(["one"]);
    expect(Object.keys(next.surfaces)).toEqual(["task-progress:old", "task-progress:two"]);
  });
  it("retires stale progress when the daemon stops without a closing event", () => {
    const result = projectTaskProgress(current, {}, [], true);
    expect(result.archived.old.dataModel.status).toBe("interrupted");
    expect(projectTaskProgress(result.surfaces, result.archived, [], false).archived.old).toEqual(result.archived.old);
  });
  it("resumes archived tasks on fresh progress", () => {
    const result = projectTaskProgress({}, { old }, [create("old"), envelope({ updateDataModel: { surfaceId: old.surfaceId, path: "/", value: { status: "running" } } })], false);
    expect(result.archived.old).toBeUndefined();
    expect(result.surfaces[old.surfaceId]).toBeDefined();
  });
  it("keeps final details when creation, update, and closure share a poll", () => {
    const result = projectTaskProgress({}, {}, [create("old"), envelope({ updateComponents: {
      surfaceId: old.surfaceId, components: [old.components.step_0],
    } }), envelope({ deleteSurface: { surfaceId: old.surfaceId } })], false);
    expect(result.archived.old.components.step_0.text).toContain("failed");
  });
});
