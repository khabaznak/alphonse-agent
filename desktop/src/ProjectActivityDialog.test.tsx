import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ProjectActivityDialog, ProjectScheduleList, ProjectTodoList } from "./ProjectActivityDialog";
import type { Project } from "./types";

const project: Project = { project_id: "project-1", name: "Weekend", description: "Plan the trip", root_path: "/tmp/weekend", visibility: "private", owner_user_id: "alex", status: "active", archived_at: null, created_at: "2026-01-01", updated_at: "2026-01-01" };
const render = (value = project) => renderToStaticMarkup(<ProjectActivityDialog project={value} user="alex" ownerId="alex" ownerName="Alex" onClose={() => {}} onSettings={() => {}} onDelegate={() => {}} onAgenda={() => {}} onUpdated={() => {}} onDeleted={() => {}} />);

describe("project activity", () => {
  it("exposes delegation and project management in the project dialog", () => {
    const html = render();
    expect(html).toContain("Weekend");
    expect(html).toContain("Scheduled work");
    expect(html).not.toContain("Tasks &amp; to-dos");
    expect(html).not.toContain("Related conversation");
    expect(html).toContain('class="project-activity-body"');
    expect(html).toContain("Delegate work");
    expect(html).toContain("Edit details");
    expect(html).toContain("Delete project");
  });
  it("does not offer delegation or editing for archived projects", () => {
    const html = render({ ...project, status: "archived" });
    expect(html).toContain('<button disabled="">Delegate work</button>');
    expect(html).toContain('<button disabled="">Edit details</button>');
  });
});


describe("project work lists", () => {
  it("distinguishes recurring and one-time work using the stored schedule kind", () => {
    const html = renderToStaticMarkup(<ProjectScheduleList items={[{ scheduled_task_id: "a", name: "Weekly check", status: "active", schedule_kind: "rrule", next_run_at: null }, { scheduled_task_id: "b", name: "Reminder", status: "paused", schedule_kind: "once", next_run_at: null }]} onAgenda={() => {}} />);
    expect(html).toContain("Weekly check");
    expect(html).toContain("Recurring");
    expect(html).toContain("One-time");
    expect(html).toContain("Paused");
  });
});


describe("project to-dos", () => {
  it("renders explicit to-dos and disables editing for another owner", () => {
    const items = [{ todo_id: "one", title: "Pack bags", status: "pending" as const }, { todo_id: "two", title: "Book train", status: "completed" as const }];
    const html = renderToStaticMarkup(<ProjectTodoList items={items} editable={false} busy={false} onChange={() => {}} />);
    expect(html).toContain("Pack bags");
    expect(html).toContain("Book train");
    expect(html.match(/disabled=""/g)).toHaveLength(2);
    expect(html).toContain('checked=""');
  });
});
