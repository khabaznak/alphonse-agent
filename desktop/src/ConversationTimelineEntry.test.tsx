import { expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ConversationTimelineEntry } from "./ConversationTimelineEntry";
import { readConversationFailures, saveConversationFailures } from "./conversationFailures";
import { mergeFreshConversationHistory } from "./conversationHistory";
import type { ChatMessage } from "./types";

it("restores a dismissed failure only beneath its original request after reloading history", () => {
  const storageValues = new Map<string, string>();
  const storage = { getItem: (key: string) => storageValues.get(key) ?? null, setItem: (key: string, value: string) => storageValues.set(key, value) } as unknown as Storage;
  const failedId = "a".repeat(32);
  const laterId = "b".repeat(32);
  saveConversationFailures(storage, "owner", [{ messageId: failedId, failureCode: "source_unavailable", dismissed: true }]);
  const reloaded: ChatMessage[] = mergeFreshConversationHistory([
    { id: laterId, role: "user", content: "Later request", created_at: "2026-11-25T10:01:00Z" },
    { id: failedId, role: "user", content: "Earlier request", created_at: "2026-11-25T10:00:00Z" },
  ], []);
  const render = (user: string) => renderToStaticMarkup(<>{reloaded.map((message) => <ConversationTimelineEntry key={message.id} message={message} questions={[]} user={user} timezone="UTC" onDone={async () => {}} projectName="" memoize={false} renderMarkdown={false} failures={readConversationFailures(storage, user)} onFailureDismissedChange={() => {}} />)}</>);
  const html = render("owner");
  expect(html.indexOf("Earlier request")).toBeLessThan(html.indexOf("Request failed · View details"));
  expect(html.indexOf("Request failed · View details")).toBeLessThan(html.indexOf("Later request"));
  expect(html.match(/Request failed · View details/g)).toHaveLength(1);
  expect(render("other")).not.toContain("Request failed · View details");
});

it("offers a recovery route only when one was supplied for this request", () => {
  const message: ChatMessage = { id: "a".repeat(32), role: "user", content: "Request" };
  const common = { message, questions: [], user: "owner", timezone: "UTC", onDone: async () => {},
    projectName: "", memoize: false, renderMarkdown: false,
    failures: [{ messageId: message.id, failureCode: "time_limit", dismissed: false }],
    onFailureDismissedChange: () => {} };
  expect(renderToStaticMarkup(<ConversationTimelineEntry {...common} />)).not.toContain("Review request status");
  expect(renderToStaticMarkup(<ConversationTimelineEntry {...common} onReviewRecovery={() => {}} />)).toContain("Review request status");
});
