import type { ReactNode, RefObject } from "react";
import { alternateSendMode, conversationSendLabel, primarySendMode, shouldSendOnKey, type ConversationSendMode } from "./conversationSendMode";
import { ArrowUp, Plus } from "lucide-react";

type Props = {
  prompt: string; task?: string; mode: ConversationSendMode; enterToSend: boolean; reviewDraft: boolean;
  textareaRef: RefObject<HTMLTextAreaElement | null>; commandMenuRef: RefObject<HTMLDetailsElement | null>;
  suggestions: readonly string[]; commands: readonly string[]; context: ReactNode; suggestion: ReactNode;
  onChange: (value: string) => void; onSend: (mode: ConversationSendMode | "normal", task?: string) => void;
  onReview: () => void;
};
export function ConversationComposer(props: Props) {
  const { prompt, task, mode, enterToSend, reviewDraft, textareaRef, commandMenuRef, suggestions, commands, context, suggestion, onChange, onSend, onReview } = props;
  const send = () => { if (!reviewDraft) onSend(primarySendMode(task, mode), task); };
  return <>
    {reviewDraft && <div className="conversation-draft-review" role="status"><strong>The active task changed while you were writing.</strong> Your draft is still here. Review it and the current send action before continuing. <button type="button" onClick={onReview}>I’ve reviewed my message</button></div>}
    {suggestion}
    <form className="composer" onSubmit={event => { event.preventDefault(); send(); }} onKeyDown={event => { if (event.key === "Escape" && commandMenuRef.current?.open) { event.preventDefault(); event.stopPropagation(); commandMenuRef.current.open = false; textareaRef.current?.focus(); } }}>
      {suggestions.length > 0 && <div className="suggestions">{suggestions.map(item => <button type="button" key={item} onClick={() => onChange(item)}>{item}</button>)}</div>}
      <textarea ref={textareaRef} aria-label="Message Alphonse" value={prompt} onKeyDown={event => { if (!shouldSendOnKey(event.key, event.shiftKey, enterToSend)) return; event.preventDefault(); send(); }} onChange={event => onChange(event.target.value)} placeholder="Ask Alphonse, or hand something over…" rows={1} />
      <div className="composer-bottom"><details ref={commandMenuRef} className="composer-command-menu"><summary aria-label="Conversation commands" title="Conversation commands"><Plus aria-hidden="true" /></summary><div><p>{prompt.trim() ? "Keep or send your draft before choosing a command." : "Choose a command to prepare in the composer."}</p>{commands.map(command => <button key={command} type="button" disabled={Boolean(prompt.trim())} onClick={() => { onChange(command); if (commandMenuRef.current) commandMenuRef.current.open = false; textareaRef.current?.focus(); }}>{command}</button>)}</div></details><span className="composer-context">{context}</span>{task ? <button className="send conversation-send-active" type="submit" disabled={reviewDraft}>{conversationSendLabel(mode)}</button> : <button className="send" type="submit" disabled={reviewDraft} aria-label="Send" title="Send message"><ArrowUp aria-hidden="true" /></button>}</div>
    </form>
    {task && <button type="button" className="conversation-send-alternative" disabled={reviewDraft} onClick={() => onSend(alternateSendMode(mode), task)}>{conversationSendLabel(alternateSendMode(mode))} →</button>}
  </>;
}
