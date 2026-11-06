import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ConversationTimelineMessage } from "./ConversationTimelineMessage";
import { mergeFreshConversationHistory } from "./conversationHistory";
import type { ChatMessage, Question } from "./types";

describe("pending questions in the conversation", () => {
  it("renders answer controls once at the original position across live and history merge", () => {
    const question: Question = { question_id: "q", task_id: "t", project_id: "", message: "Which one?", kind: "yes_no", choices: [], created_at: "2026-11-06T01:47:00Z" };
    const q: ChatMessage = { id: "question:q", role: "assistant", content: question.message, created_at: question.created_at, sequence: 4 };
    const answer: ChatMessage = { id: "later", role: "user", content: "Later turn", created_at: "2026-11-06T01:51:00Z", sequence: 5 };
    const ordered = mergeFreshConversationHistory([answer, q], [q]);
    const html = renderToStaticMarkup(<>{ordered.map((message) => <ConversationTimelineMessage key={message.id} message={message} questions={[question]} user="owner" timezone="UTC" onDone={async () => {}} projectName="" memoize={false} renderMarkdown />)}</>);
    expect(html.match(/Which one\?/g)).toHaveLength(1);
    expect(html.indexOf("Which one?")).toBeLessThan(html.indexOf("Later turn"));
    expect(html).toContain(">Yes</button>");
    expect(html).toContain('dateTime="2026-11-06T01:47:00Z"');
  });
});
