/** Reject a schedule mutation response that does not confirm the intended record and state. */
export function confirmedScheduleMutation<T extends { scheduled_task_id: string; status: string }>(
  method: string, scheduledTaskId: string, result: { task?: T; deleted?: boolean },
): T | null {
  if (method === "delete_scheduled_task") {
    if (result.deleted !== true) throw new Error("Schedule deletion was not confirmed. Refresh before retrying.");
    return null;
  }
  const expected = ({ pause_schedule: "paused", resume_schedule: "active", cancel_schedule: "cancelled" } as Record<string, string>)[method];
  if (!result.task || result.task.scheduled_task_id !== scheduledTaskId || (expected && result.task.status !== expected)) {
    throw new Error("The schedule change was not confirmed. Refresh before retrying.");
  }
  return result.task;
}
