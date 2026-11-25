export type OwnerFollowup = { id: string; prompt: string; status: "queued" | "released" | "completed" | "needs_attention"; can_remove: boolean };

export function OwnerFollowups({ items, onRemove }: { items: OwnerFollowup[]; onRemove: (id: string) => void }) {
  if (!items.length) return null;
  const labels = { queued: "Waiting", released: "Processing", completed: "Completed", needs_attention: "Needs attention" };
  return <section className="composer-hint" aria-label="Messages to send later"><h3>Messages to send later</h3>
    {items.map(item => <p key={item.id}>{item.prompt} · {labels[item.status]} {item.can_remove && <button type="button" onClick={() => onRemove(item.id)}>Remove</button>}</p>)}
    <p>Messages wait until the previous answer is confirmed. If work fails or delivery is uncertain, review it before continuing.</p>
  </section>;
}
