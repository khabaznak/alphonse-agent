import { ConversationMessage, MemoizedConversationMessage } from "./ConversationMessage";
import { QuestionCard } from "./QuestionCard";
import type { ChatMessage, Question } from "./types";

export function ConversationTimelineMessage({ message, questions, user, timezone, onDone, projectName, memoize, renderMarkdown }: {
  message: ChatMessage; questions: Question[]; user: string; timezone: string; onDone: () => Promise<void>;
  projectName: string; memoize: boolean; renderMarkdown: boolean;
}) {
  const question = questions.find((candidate) => message.id === `question:${candidate.question_id}`);
  if (question) return <QuestionCard question={question} user={user} timezone={timezone} onDone={onDone} />;
  const Bubble = memoize ? MemoizedConversationMessage : ConversationMessage;
  return <Bubble message={message} timezone={timezone} projectName={projectName} renderMarkdown={renderMarkdown} />;
}
