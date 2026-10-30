import { memo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import { FileText } from "lucide-react";
import remarkGfm from "remark-gfm";
import { formatMessageTime } from "./messageTime";
import type { ChatMessage } from "./types";

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

export function ConversationMessage({ message, timezone, projectName = "", renderMarkdown = true }: { message: ChatMessage; timezone: string; projectName?: string; renderMarkdown?: boolean }) {
  const timestamp = message.created_at ? formatMessageTime(message.created_at, timezone) : null;
  return <article className={`message ${message.role} conversation-message`} aria-label={message.role === "assistant" ? "Alphonse" : message.role === "user" ? "You" : "System notice"}>
    {message.source && !["desktop", "ledger"].includes(message.source) && <small className="message-source" title={`Sent from ${message.source}`}>↗ {sourceLabel(message.source)}</small>}
    {message.project_id && <small className="message-project-association" title="Task project context">Project · {projectName || message.project_id}</small>}
    {message.role === "assistant" && renderMarkdown ? <div className="message-markdown"><ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={MARKDOWN_COMPONENTS}>{message.content}</ReactMarkdown></div> : message.content}
    {timestamp && <time className="message-timestamp" dateTime={message.created_at} title={timestamp.tooltip}>{timestamp.visible}</time>}
  </article>;
}

export const MemoizedConversationMessage = memo(ConversationMessage);

