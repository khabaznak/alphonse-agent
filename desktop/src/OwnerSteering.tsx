export type OwnerSteer = {id: string; prompt: string; status: string; can_remove: boolean; remove_blocked_reason?: string | null};
export function OwnerSteering({items, onRemove}: {items: OwnerSteer[]; onRemove: (id: string) => void}) {
  if (!items.length) return null;
  return <section className="composer-hint" aria-label="Updates to current task"><h3>Updates to current task</h3>{items.map(item =>
    <p key={item.id}>{item.prompt} · {item.status === "accepted" ? "Received · waiting to be considered" : item.status === "included_in_planning" ? "Included in planning · outcome not confirmed" : "Needs attention"} {item.can_remove ? <button type="button" onClick={() => onRemove(item.id)}>Remove update</button> : item.remove_blocked_reason === "dependent_followups" ? "Review dependent follow-ups before removing." : "Remove after the request is resolved."}</p>
  )}<p>Removing an update can remove context needed by later follow-ups or confirmations; review those first.</p></section>;
}
