import { ConversationFailure } from "./ConversationFailure";
import { ConversationTimelineMessage } from "./ConversationTimelineMessage";
import type { ConversationFailureRecord } from "./conversationFailures";
import { ownerFailureMessage } from "./ownerConversation";
import type { ChatMessage, Question } from "./types";

/** Keep a locally observed failure directly after the matching conversation message. */
export function ConversationTimelineEntry({ message, questions, user, timezone, onDone, projectName, memoize, renderMarkdown, failures, onFailureDismissedChange }: {
  message: ChatMessage; questions: Question[]; user: string; timezone: string; onDone: () => Promise<void>;
  projectName: string; memoize: boolean; renderMarkdown: boolean; failures: ConversationFailureRecord[];
  onFailureDismissedChange: (messageId: string, dismissed: boolean) => void;
}) {
  const failure = failures.find((entry) => entry.messageId === message.id);
  return <>
    <ConversationTimelineMessage message={message} questions={questions} user={user} timezone={timezone} onDone={onDone} projectName={projectName} memoize={memoize} renderMarkdown={renderMarkdown} />
    {failure && <ConversationFailure message={ownerFailureMessage(failure.failureCode)} dismissed={failure.dismissed} onDismissedChange={(dismissed) => onFailureDismissedChange(failure.messageId, dismissed)} />}
  </>;
}
