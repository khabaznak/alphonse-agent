import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ConversationMessage } from "./ConversationMessage";
import type { ChatMessage } from "./types";

const render = (message: ChatMessage, renderMarkdown = true) => renderToStaticMarkup(<ConversationMessage message={message} timezone="America/Mexico_City" projectName="Trip" renderMarkdown={renderMarkdown} />);

describe("conversation content", () => {
  it("shows a verified cancellation and running-work uncertainty even if prose omits it", () => {
    const html = render({ id: "a", role: "assistant", content: "Listo.", committed_effect: {
      kind: "scheduled_task_cancelled", schedule_id: "fixture", name: "Pantry",
      execution_may_have_started: true,
    } });
    expect(html).toContain("Listo.");
    expect(html).toContain("Scheduled task cancelled: Pantry.");
    expect(html).toContain("The task may already have started.");
  });
  it("keeps user input literal while rendering assistant GFM and accessible media", () => {
    const content = "## Plan\n\n- [x] Booked\n\n| Item | Cost |\n| --- | --- |\n| Rail | 20 |\n\n![Receipt](/receipt.png)\n\n[Itinerary](https://example.com/trip.pdf)";
    const assistant = render({ id: "a", role: "assistant", content });
    expect(assistant).toContain('aria-label="Alphonse"');
    expect(assistant).toContain('<h2>Plan</h2>');
    expect(assistant).toContain('<table>');
    expect(assistant).toContain('alt="Receipt"');
    expect(assistant).toContain('href="https://example.com/trip.pdf"');
    expect(assistant).toContain('class="message-resource"');
    expect(assistant).toContain('type="checkbox"');
    const user = render({ id: "u", role: "user", content });
    expect(user).toContain('aria-label="You"');
    expect(user).toContain("## Plan");
    expect(user).not.toContain('<h2>');
    expect(user).not.toContain('<img');
  });
  it("preserves source, project and exact timestamp metadata without inventing content", () => {
    const html = render({ id: "a", role: "assistant", content: "Could not save the file.", source: "telegram", project_id: "trip", created_at: "2026-10-29T18:30:00Z" });
    expect(html).toContain("Telegram");
    expect(html).toContain("Project · Trip");
    expect(html).toContain('dateTime="2026-10-29T18:30:00Z"');
    expect(html).toContain("Could not save the file.");
    expect(html).not.toContain("Booked");
  });
  it("retains plain-text diagnostics and does not execute embedded HTML or unsafe links", () => {
    expect(render({ id: "a", role: "assistant", content: "**plain**" }, false)).not.toContain('<strong>');
    const html = render({ id: "a", role: "assistant", content: '<script>alert(1)</script>\n\n[bad](javascript:alert)' });
    expect(html).not.toContain('<script>');
    expect(html).not.toContain('href="javascript:');
    expect(render({ id: "s", role: "system", content: "Connection lost" })).toContain('aria-label="System notice"');
  });
  it("renders private mail summaries without creating links or remote image requests", () => {
    const content = "![probe](https://example.invalid/image.png)\nhttps://example.invalid/link";
    const html = render({ id: "mail", role: "assistant", content, private_reply_plain_text: true });
    expect(html).toContain(content);
    expect(html).not.toContain("href=");
    expect(html).not.toContain("<img");
  });
  it("renders the private result and a collapsed plain-text source disclosure", () => {
    const message: ChatMessage = { id: "mail", role: "assistant", content: "fallback", private_reply_owner: "owner",
      private_reply_surface: { surfaceId: "private-gmail:mail", catalogId: "alphonse.desktop.catalog.v2", dataModel: {}, components: {
        root: { id: "root", component: "Card", children: ["sources"] },
        sources: { id: "sources", component: "Disclosure", label: "Reviewed sources", children: ["subject"] },
        subject: { id: "subject", component: "Text", text: "Packing [updated] <latest>" },
      } } };
    const html = render(message);
    expect(html).toContain("<summary>Reviewed sources</summary>");
    expect(html).not.toContain('<details open');
    expect(html).toContain("Packing [updated] &lt;latest&gt;");
    expect(html).not.toContain("fallback");
    expect(render({ ...message, private_reply_owner: undefined })).not.toContain("Packing");
  });
});
