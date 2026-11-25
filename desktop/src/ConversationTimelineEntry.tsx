import { ConversationFailure } from "./ConversationFailure";
import { ConversationWorkLog, focusRequestWorkLog } from "./ConversationWorkLog";
import { ConversationTimelineMessage } from "./ConversationTimelineMessage";
import type { ConversationFailureRecord } from "./conversationFailures";
import { ownerFailureMessage, type OwnerRequest } from "./ownerConversation";
import type { ChatMessage, Question } from "./types";

/** Keep a locally observed failure directly after the matching conversation message. */
export function ConversationTimelineEntry({ message, questions, user, timezone, onDone, projectName, memoize, renderMarkdown, failures, onFailureDismissedChange, workLogRequest, active, onReviewRecovery, onOpenModelSettings, onDraftQuestion, onActiveLogUpdate }: {
  message: ChatMessage; questions: Question[]; user: string; timezone: string; onDone: () => Promise<void>;
  projectName: string; memoize: boolean; renderMarkdown: boolean; failures: ConversationFailureRecord[];
  onFailureDismissedChange: (messageId: string, dismissed: boolean) => void;
  workLogRequest?: OwnerRequest; active?: boolean; onReviewRecovery?: (messageId: string) => void; onOpenModelSettings?: () => void; onDraftQuestion?: (messageId: string, content: string, createdAt?: string) => void; onActiveLogUpdate?: () => void;
}) {
  const failure = failures.find((entry) => entry.messageId === message.id);
  return <>
    <ConversationTimelineMessage message={message} questions={questions} user={user} timezone={timezone} onDone={onDone} projectName={projectName} memoize={memoize} renderMarkdown={renderMarkdown} />
    {workLogRequest && message.role === "user" && (!message.source || message.source === "desktop") && /^[0-9a-f]{32}$/.test(message.id) && <ConversationWorkLog request={workLogRequest} messageId={message.id} active={Boolean(active)} onActiveUpdate={onActiveLogUpdate} disclosureId={`conversation-work-log-${message.id}`} />}
    {failure && <ConversationFailure recoveryTriggerId={`owner-recovery-trigger-${message.id}`} message={ownerFailureMessage(failure.failureCode)} dismissed={failure.dismissed} onDismissedChange={(dismissed) => onFailureDismissedChange(failure.messageId, dismissed)} onReviewRecovery={onReviewRecovery ? () => onReviewRecovery(failure.messageId) : undefined} onViewWorkLog={workLogRequest && message.role === "user" && (!message.source || message.source === "desktop") && /^[0-9a-f]{32}$/.test(message.id) ? () => focusRequestWorkLog(message.id) : undefined} onDraftQuestion={onDraftQuestion && message.role === "user" ? () => onDraftQuestion(message.id, message.content, message.created_at) : undefined} hasWorkLog={Boolean(workLogRequest && message.role === "user" && (!message.source || message.source === "desktop") && /^[0-9a-f]{32}$/.test(message.id))} onOpenModelSettings={onOpenModelSettings && ["model_not_configured", "model_cli_missing", "model_auth_unverified", "model_auth_required", "model_cli_upgrade_required"].includes(failure.failureCode) ? onOpenModelSettings : undefined} />}
  </>;
}
