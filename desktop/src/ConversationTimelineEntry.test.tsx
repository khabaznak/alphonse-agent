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

it("only offers Outlook settings from a live recovery result for its owner", () => {
  const message: ChatMessage = { id: "owner:hot", role: "assistant", content: "Refresh Microsoft access.",
    private_reply_owner: "owner", private_outlook_inbox: { status: "refresh_required" } };
  const common = { message, questions: [], user: "owner", timezone: "UTC", onDone: async () => {},
    projectName: "", memoize: false, renderMarkdown: false, failures: [], onFailureDismissedChange: () => {},
    onOpenOutlookSettings: () => {} };
  expect(renderToStaticMarkup(<ConversationTimelineEntry {...common} />)).toContain("Open Outlook / Hotmail settings");
  expect(renderToStaticMarkup(<ConversationTimelineEntry {...common} user="other" />)).not.toContain("Open Outlook / Hotmail settings");
  expect(renderToStaticMarkup(<ConversationTimelineEntry {...common} message={{ ...message, private_outlook_inbox: {
    status: "complete", result: { count: 0, observed_at: "2026-11-26T17:00:00Z", timezone: "UTC" },
  } }} />))
    .not.toContain("Open Outlook / Hotmail settings");
});

it("keeps contextual recovery owner-scoped, and drops unsupported actions in model prose", () => {
  const id = "c".repeat(32);
  const message: ChatMessage = {
    id, role: "user", source: "desktop", content: "Fixture request",
    failure_recovery: {
      reply: "**Je ne peux pas confirmer le résultat.** [Consultez le journal](https://example.com)",
      language: "fr-CA", activity: { label: "<b>Voir l’activité</b>", detail: "Vérification enregistrée." },
    },
  };
  const common = { message, questions: [], user: "owner", timezone: "UTC", onDone: async () => {},
    projectName: "", memoize: false, renderMarkdown: true,
    failures: [{ messageId: id, failureCode: "source_unavailable", dismissed: false }], onFailureDismissedChange: () => {} };
  const ownerHtml = renderToStaticMarkup(<ConversationTimelineEntry {...common} showFailureRecovery />);
  expect(ownerHtml).toContain('lang="fr-CA"');
  expect(ownerHtml).toContain("&lt;b&gt;Voir l’activité&lt;/b&gt;");
  expect(ownerHtml).toContain("Consultez le journal");
  expect(ownerHtml).not.toContain("href=");
  expect(renderToStaticMarkup(<ConversationTimelineEntry {...common} />)).not.toContain("Je ne peux pas");
  const malformed = { ...message, failure_recovery: { reply: "", language: "fr-CA" } };
  expect(renderToStaticMarkup(<ConversationTimelineEntry {...common} message={malformed} showFailureRecovery />)).toContain("Alphonse could not complete this request");
});

it("places the failure before its closed work log, with an honest unverified-answer message", () => {
  const message: ChatMessage = { id: "a".repeat(32), role: "user", content: "What should we do next?" };
  const html = renderToStaticMarkup(<ConversationTimelineEntry message={message} questions={[]} user="owner" timezone="UTC" onDone={async () => {}} projectName="" memoize={false} renderMarkdown={false} failures={[{messageId: message.id, failureCode: "answer_unverified", dismissed: false}]} onFailureDismissedChange={() => {}} workLogRequest={async () => ({}) as never} />);
  expect(html).toContain("The specific reason was not recorded.");
  expect(html).not.toContain("could not verify enough source information");
  expect(html.indexOf("Alphonse could not complete this request")).toBeLessThan(html.indexOf('class="conversation-work-log"'));
  expect(html).toMatch(/<details id="conversation-work-log-[^"]+" class="conversation-work-log">/);
});

it("keeps the work log available beside a report from the final reserved call", () => {
  const message: ChatMessage = {
    id: "d".repeat(32), role: "user", source: "desktop", content: "Fixture request",
    failure_code: "action_limit", failure_recovery: { reply: "I found the catalog entry.", language: "en" },
  };
  const html = renderToStaticMarkup(<ConversationTimelineEntry message={message} questions={[]} user="owner"
    timezone="UTC" onDone={async () => {}} projectName="" memoize={false} renderMarkdown
    failures={[]} onFailureDismissedChange={() => {}} showFailureRecovery
    workLogRequest={async () => { throw new Error("static render"); }} />);
  expect(html).toContain("I found the catalog entry.");
  expect(html).toContain("Work log");
});
