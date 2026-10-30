import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { TaskProgressBubble } from "./TaskProgressBubble";
import { projectTaskProgress } from "./taskProgress";
import { buildConversationTimeline } from "./conversationTimeline";
import { DESKTOP_CATALOG_ID, type AgUiEvent } from "./a2ui";

const event = (value: object) => ({ event: { type: "CUSTOM", name: "a2ui.envelope", value: { version: "v0.9.1", ...value } } as AgUiEvent });
function update(taskId: string, status: string) {
  const surfaceId = `task-progress:${taskId}`;
  return [event({ createSurface: { surfaceId, catalogId: DESKTOP_CATALOG_ID } }), event({ updateComponents: { surfaceId, components: [
    { id: "summary", component: "Text", text: taskId },
    { id: "tool", component: "Text", text: "Tool: Device Status · failed" },
    { id: "step_0", component: "Text", text: "Attempt 1 · Device Status · failed\nOutput: Device unavailable" },
    { id: "task_context", component: "Text", text: "PRIVATE REASONING" },
  ] } }), event({ updateDataModel: { surfaceId, path: "/", value: {
    task_id: taskId, status, tool_id: "native.device_status", tool_name: "Device Status", tool_status: "failed", updated_at: "2026-10-29T18:00:00Z",
  } } })];
}

describe("poll projection through the production task renderer", () => {
  it.each([["admitted", "Request received"], ["queued", "Queued"], ["planned", "Ready to start"], ["blocked", "Blocked · action needed"], ["waiting_user", "Waiting for your answer"]])("presents %s without implying completion or active execution", (status, label) => {
    const result = projectTaskProgress({}, {}, update("request", status), false);
    const html = renderToStaticMarkup(<TaskProgressBubble surface={result.surfaces["task-progress:request"]} />);
    expect(html).toContain(label);
    expect(html).not.toContain("Completed");
    expect(html).not.toContain("task-progress-spinner");
    expect(html).toContain('data-task-id="request"');
  });
  it.each(["completed", "failed", "cancelled", "abandoned"])("collapses %s without a stale spinner or private checkpoint text", (status) => {
    const result = projectTaskProgress({}, {}, [...update("repair", status), event({ deleteSurface: { surfaceId: "task-progress:repair" } })], false);
    const html = renderToStaticMarkup(<TaskProgressBubble surface={result.archived.repair} archived />);
    expect(html).toContain('<details class="task-progress-log"><summary>Work log');
    expect(html).toContain("Device Status · failed");
    expect(html).toContain("Device unavailable");
    expect(html).toContain('<time dateTime="2026-10-29T18:00:00Z">');
    expect(html).not.toContain("task-progress-spinner");
    expect(html).not.toContain("PRIVATE REASONING");
    expect(result.archived.repair.dataModel.tool_id).toBe("native.device_status");
  });
  it("keeps both active tasks, preserves failed actions during retry, then retains the log beside its own request and reply", () => {
    let result = projectTaskProgress({}, {}, [...update("repair", "retrying"), ...update("receipt", "running")], false);
    expect(result.archived).toEqual({});
    expect(renderToStaticMarkup(<TaskProgressBubble surface={result.surfaces["task-progress:repair"]} />)).toContain("Waiting to retry");
    result = projectTaskProgress(result.surfaces, result.archived, [...update("repair", "failed"), event({ deleteSurface: { surfaceId: "task-progress:repair" } })], false);
    const items = buildConversationTimeline([{ id: "repair", role: "user", content: "Repair" }, { id: "reply", role: "assistant", content: "Could not repair" }], ["repair", "receipt"], {}, {}, Object.keys(result.archived));
    expect(items.map(i => i.key)).toEqual(["message:repair", "task:repair", "message:reply", "task:receipt"]);
    expect(renderToStaticMarkup(<TaskProgressBubble surface={result.surfaces["task-progress:receipt"]} />)).toContain("Alphonse is working");
    expect(renderToStaticMarkup(<TaskProgressBubble surface={result.archived.repair} archived />)).toContain("Work log · Failed");
  });
  it("shows disconnect as interrupted, never as completed, and resumes after a fresh poll", () => {
    const result = projectTaskProgress({}, {}, update("repair", "running"), false);
    const surface = result.surfaces["task-progress:repair"];
    const offline = renderToStaticMarkup(<TaskProgressBubble surface={surface} connected={false} />);
    expect(offline).toContain("Interrupted · outcome unconfirmed");
    expect(offline).not.toContain("task-progress-spinner");
    expect(renderToStaticMarkup(<TaskProgressBubble surface={surface} />)).toContain("Alphonse is working");
  });
  it("does not relabel an interruption as success or resurrect it on a create envelope alone", () => {
    let result = projectTaskProgress({}, {}, update("repair", "running"), true);
    expect(result.archived.repair.dataModel.status).toBe("interrupted");
    result = projectTaskProgress(result.surfaces, result.archived, [update("repair", "running")[0]], false);
    expect(result.archived.repair.dataModel.status).toBe("interrupted");
    result = projectTaskProgress(result.surfaces, result.archived, update("repair", "running"), false);
    expect(result.archived.repair).toBeUndefined();
  });
});
