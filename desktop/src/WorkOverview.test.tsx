import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { daemonRequest } from "./api";
import { loadProjectTodos, projectProgress, ProjectTodoRow, todoMatches, type ProjectTodo } from "./WorkOverview";

vi.mock("./api", () => ({ daemonRequest: vi.fn() }));

const pending: ProjectTodo = { todo_id: "1", title: "Book train tickets", status: "pending" };
const completed: ProjectTodo = { todo_id: "2", title: "Choose destination", status: "completed" };
const cancelled: ProjectTodo = { todo_id: "3", title: "Cancelled item", status: "cancelled" };

describe("project to-dos in the work overview", () => {
  it("loads every page for the selected actor, owner and project", async () => {
    vi.mocked(daemonRequest).mockReset().mockResolvedValueOnce({ todos: [pending], todos_more: true }).mockResolvedValueOnce({ todos: [completed], todos_more: false });
    expect(await loadProjectTodos("alex", "owner", "trip")).toEqual([pending, completed]);
    expect(daemonRequest).toHaveBeenNthCalledWith(1, "project_records", { actor_user_id: "alex", owner_user_id: "owner", project_id: "trip", todos_offset: 0, limit: 100 });
    expect(daemonRequest).toHaveBeenNthCalledWith(2, "project_records", { actor_user_id: "alex", owner_user_id: "owner", project_id: "trip", todos_offset: 1, limit: 100 });
  });
  it("includes pending and completed to-dos in progress while excluding cancelled ones", () => {
    expect(projectProgress([pending, completed, cancelled], 3, 1)).toEqual({ total: 5, completed: 2 });
    expect(todoMatches(pending, "active")).toBe(true);
    expect(todoMatches(completed, "active")).toBe(false);
    expect(todoMatches(completed, "history")).toBe(true);
    expect(todoMatches(cancelled, "history")).toBe(true);
    expect(todoMatches(pending, "attention")).toBe(false);
  });
  it("renders a to-do with its status", () => {
    const html = renderToStaticMarkup(<ProjectTodoRow todo={pending} onOpen={() => {}} />);
    expect(html).toContain("Book train tickets");
    expect(html).toContain("To-do · Pending");
  });
});
