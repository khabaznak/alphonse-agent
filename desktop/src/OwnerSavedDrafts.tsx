export type OwnerSavedDraft = {id: string; active_message_id: string; prompt: string; resolution: "unresolved" | "not_submitted" | "submitted"};
export function OwnerSavedDrafts({ items, prompt, onCheck, onRestore, onDelete }: {
  items: OwnerSavedDraft[]; prompt: string; onCheck: (item: OwnerSavedDraft) => void;
  onRestore: (item: OwnerSavedDraft) => void; onDelete: (item: OwnerSavedDraft) => void;
}) {
  if (!items.length) return null;
  return <section className="composer-hint" aria-label="Saved message drafts"><h3>Saved message drafts</h3>{items.map(draft =>
    <p key={draft.id}>{draft.prompt} · {draft.resolution === "unresolved" ? "Submission needs checking" : draft.resolution === "submitted" ? "Submitted" : "Not submitted"} {draft.resolution === "unresolved" && <button type="button" onClick={() => onCheck(draft)}>Check submission</button>} {draft.resolution === "not_submitted" && <button type="button" disabled={Boolean(prompt.trim())} onClick={() => onRestore(draft)}>Restore draft</button>} {draft.resolution !== "unresolved" && <button type="button" onClick={() => onDelete(draft)}>Delete saved draft</button>}</p>
  )}</section>;
}
