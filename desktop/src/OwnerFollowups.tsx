export type OwnerFollowup = { id: string; prompt: string; status: "queued" | "released" | "completed" | "needs_attention"; can_remove: boolean };

export function OwnerFollowups({ items, onRemove }: { items: OwnerFollowup[]; onRemove: (id: string) => void }) {
  if (!items.length) return null;
  const labels = { queued: "Queued", released: "Processing", completed: "Completed", needs_attention: "Needs attention" };
  return <section className="composer-hint" aria-label="Follow-ups"><h3>Follow-ups</h3>
    {items.map(item => <p key={item.id}>{item.prompt} · {labels[item.status]} {item.can_remove && <button type="button" onClick={() => onRemove(item.id)}>Remove</button>}</p>)}
    <p>Follow-ups wait for a confirmed previous answer. If work fails or delivery is uncertain, review it before continuing.</p>
  </section>;
}
