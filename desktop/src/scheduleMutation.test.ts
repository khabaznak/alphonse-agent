import { describe, expect, it } from "vitest";
import { confirmedScheduleMutation } from "./scheduleMutation";

describe("schedule mutation confirmation", () => {
  it("accepts cancellation only for the selected schedule with cancelled status", () => {
    const cancelled = { scheduled_task_id: "task-1", status: "cancelled" };
    expect(confirmedScheduleMutation("cancel_schedule", "task-1", { task: cancelled })).toEqual(cancelled);
    expect(() => confirmedScheduleMutation("cancel_schedule", "task-1", { task: { ...cancelled, status: "active" } })).toThrow("not confirmed");
    expect(() => confirmedScheduleMutation("cancel_schedule", "task-1", { task: { ...cancelled, scheduled_task_id: "task-2" } })).toThrow("not confirmed");
    expect(() => confirmedScheduleMutation("cancel_schedule", "task-1", {})).toThrow("not confirmed");
  });
  it("requires explicit confirmation of deletion and checks pause and resume status", () => {
    expect(confirmedScheduleMutation("delete_scheduled_task", "task-1", { deleted: true })).toBeNull();
    expect(() => confirmedScheduleMutation("delete_scheduled_task", "task-1", { deleted: false })).toThrow("not confirmed");
    expect(() => confirmedScheduleMutation("pause_schedule", "task-1", { task: { scheduled_task_id: "task-1", status: "active" } })).toThrow("not confirmed");
    expect(confirmedScheduleMutation("resume_schedule", "task-1", { task: { scheduled_task_id: "task-1", status: "active" } })?.status).toBe("active");
  });
});
