import { memo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import { FileText } from "lucide-react";
import remarkGfm from "remark-gfm";
import { PrivateProviderResult } from "./a2ui";
import { formatMessageTime } from "./messageTime";
import type { ChatMessage, OutlookInboxResult } from "./types";

// Stable plugin identity and memoization matter for long histories in WebKit.
const REMARK_PLUGINS = [remarkGfm];
// File links remain ordinary links with the same Markdown URL safety policy.
// Do not infer execution capabilities or artifact permissions from a URL.
const MARKDOWN_COMPONENTS: Components = {
  a: ({ node: _node, href, children, ...props }) => {
    const file = /\.(?:pdf|docx?|xlsx?|csv|tsv|pptx?|zip|txt|md|json|html?)(?:[?#]|$)/i.test(href || "");
    return <a {...props} href={href} className={file ? "message-resource" : undefined}>{file && <FileText aria-hidden="true" size={16} />}{children}</a>;
  },
};
function sourceLabel(source: string): string {
  return source.replace(/[-_]/g, " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function OutlookInboxCount({ result }: { result: OutlookInboxResult }) {
  const count = new Intl.NumberFormat(result.locale).format(result.count);
  const checkedAt = new Intl.DateTimeFormat(result.locale, {
    timeZone: result.timezone, year: "numeric", month: "short", day: "numeric",
    hour: "numeric", minute: "2-digit", timeZoneName: "short",
  }).format(new Date(result.observed_at));
  return <section className="outlook-inbox-result" aria-label="Outlook / Hotmail Inbox result">
    <p className="outlook-inbox-source">Outlook / Hotmail · Inbox</p>
    <dl><dt>Unread messages</dt><dd><bdi lang={result.locale}>{count}</bdi></dd></dl>
    <p>Checked <time dateTime={result.observed_at} lang={result.locale} dir="auto">{checkedAt}</time></p>
  </section>;
}

export function ConversationMessage({ message, timezone, projectName = "", renderMarkdown = true }: { message: ChatMessage; timezone: string; projectName?: string; renderMarkdown?: boolean }) {
  const timestamp = message.created_at ? formatMessageTime(message.created_at, timezone) : null;
  return <article className={`message ${message.role} conversation-message`} data-conversation-message-id={message.id} aria-label={message.role === "assistant" ? "Alphonse" : message.role === "user" ? "You" : "System notice"}>
    {message.source && !["desktop", "ledger"].includes(message.source) && <small className="message-source" title={`Sent from ${message.source}`}>↗ {sourceLabel(message.source)}</small>}
    {message.project_id && <small className="message-project-association" title="Task project context">Project · {projectName || message.project_id}</small>}
    {message.role === "assistant" && message.private_reply_owner && message.private_outlook_inbox?.status === "complete"
      ? <OutlookInboxCount result={message.private_outlook_inbox.result} />
      : message.role === "assistant" && message.private_reply_owner && message.private_reply_surface
        ? <PrivateProviderResult key={message.private_reply_surface.surfaceId} surface={message.private_reply_surface} />
      : message.role === "assistant" && renderMarkdown && !message.private_reply_plain_text
        ? <div className="message-markdown"><ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={MARKDOWN_COMPONENTS}>{message.content}</ReactMarkdown></div>
        : message.content}
    {message.committed_effect?.kind === "scheduled_task_cancelled" && <aside aria-label="Verified cancellation">
      Scheduled task cancelled: {message.committed_effect.name}.
      {message.committed_effect.execution_may_have_started && " The task may already have started."}
    </aside>}
    {timestamp && <time className="message-timestamp" dateTime={message.created_at} title={timestamp.tooltip}>{timestamp.visible}</time>}
  </article>;
}

export const MemoizedConversationMessage = memo(ConversationMessage);

