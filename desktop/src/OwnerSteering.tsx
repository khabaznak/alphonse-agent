export type OwnerSteer = {id: string; prompt: string; status: string; can_remove: boolean; remove_blocked_reason?: string | null};
export function OwnerSteering({items, onRemove}: {items: OwnerSteer[]; onRemove: (id: string) => void}) {
  if (!items.length) return null;
  return <section className="composer-hint" aria-label="Steering messages"><h3>Steering messages</h3>{items.map(item =>
    <p key={item.id}>{item.prompt} · {item.status === "accepted" ? "Accepted" : item.status === "included_in_planning" ? "Included in planning" : "Needs attention"} {item.can_remove ? <button type="button" onClick={() => onRemove(item.id)}>Clear steering</button> : item.remove_blocked_reason === "dependent_followups" ? "Review dependent follow-ups before clearing." : "Clear after the request is resolved."}</p>
  )}<p>Clearing steering can remove context needed by later follow-ups or confirmations; review those first. Effect-specific confirmations after a steered question may need attention.</p></section>;
}
