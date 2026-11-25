import { useEffect, useState } from "react";
import type { OwnerRequest } from "./ownerConversation";
import { conversationSendLabel, type ConversationSendMode } from "./conversationSendMode";

type Suggestion = { task: string; prompt: string; user: string; mode: ConversationSendMode };

/** Advice applies only to the exact task and draft examined; it never sends by itself. */
export function ConversationSuggestion({ request, task, prompt, user, defaultMode, onSend }: {
  request: OwnerRequest; task?: string; prompt: string; user: string; defaultMode: ConversationSendMode;
  onSend: (mode: ConversationSendMode, task: string) => void;
}) {
  const [suggestion, setSuggestion] = useState<Suggestion | null>(null);
  useEffect(() => {
    let current = true;
    setSuggestion(null);
    if (!task || !user || !prompt.trim() || prompt.trimStart().startsWith("/")) return;
    const timer = window.setTimeout(() => {
      void request<{ suggestion: unknown }>("experimental_owner_message_suggestion", { active_message_id: task, prompt })
        .then((result) => {
          if (current && (result?.suggestion === "queue" || result?.suggestion === "steer")) {
            setSuggestion({ task, prompt, user, mode: result.suggestion });
          }
        }).catch(() => { if (current) setSuggestion(null); });
    }, 1000);
    return () => { current = false; window.clearTimeout(timer); };
  }, [request, task, prompt, user]);
  if (!suggestion || suggestion.task !== task || suggestion.prompt !== prompt || suggestion.user !== user || suggestion.mode === defaultMode) return null;
  const update = suggestion.mode === "steer";
  return <aside className="conversation-suggestion" aria-live="polite" aria-label="Alphonse suggestion">
    <span className="conversation-suggestion-speaker">Alphonse suggests</span>
    <strong>{update ? "This looks like an update to the current task." : "This looks like a separate follow-up."}</strong>
    <p>{update ? "Send it now for Alphonse to consider while working." : "Send it after the current task has a confirmed answer."}</p>
    <button type="button" onClick={() => onSend(suggestion.mode, suggestion.task)}>{conversationSendLabel(suggestion.mode)}</button>
  </aside>;
}
