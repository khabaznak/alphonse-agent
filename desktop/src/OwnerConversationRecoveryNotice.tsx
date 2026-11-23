import type { OwnerConversation } from "./ownerConversation";

type Props = {
  state: ReturnType<OwnerConversation["blockingState"]>;
  retrying: boolean;
  onRetry: () => void;
};

export function OwnerConversationRecoveryNotice({ state, retrying, onRetry }: Props) {
  if (state === "acknowledgement") return <p className="composer-hint" role="status">
    The answer is shown. Saving it to conversation history is still unconfirmed. <button type="button" disabled={retrying} onClick={onRetry}>Retry saving answer</button>
  </p>;
  if (state === "uncertain") return <p className="composer-hint" role="status">The previous message or answer has an uncertain result. Check earlier work and recovery before sending another message.</p>;
  return null;
}
