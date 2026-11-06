import { useState } from "react";
import { daemonRequest } from "./api";

type Manifest = { decision_id: string; sources?: Array<{ source_id: string }>; projections: Array<{ records?: Array<{ source_id: string }> }> };
export function DecisionContext({ summary, manifests, actorUserId, taskId }: {
  summary: string; manifests: Manifest[]; actorUserId: string; taskId: string;
}) {
  const [expanded, setExpanded] = useState("");
  const expand = async (decisionId: string, sourceId: string) => {
    try {
      const result = await daemonRequest<{ status: string; source?: { items?: Array<{ content: string }> } }>("decision_context", {
        actor_user_id: actorUserId, task_id: taskId, decision_id: decisionId, source_id: sourceId,
      });
      setExpanded(result.status === "found" ? result.source?.items?.map((item) => item.content).join("\n\n") || "Source unavailable." : "Source unavailable or no longer authorized.");
    } catch { setExpanded("Source unavailable or no longer authorized."); }
  };
  const sources = manifests.flatMap((manifest) => [...new Set([...(manifest.sources || []), ...manifest.projections.flatMap((projection) => projection.records || [])].map((record) => record.source_id))]
    .filter((id) => /^(conversation:|memory:|task-evidence:|legacy-ledger:)/.test(id))
    .map((sourceId) => ({ decisionId: manifest.decision_id, sourceId })));
  return <details className="task-progress-context"><summary>Decision context</summary>
    <pre className="task-progress-detail">{summary || "Legacy work log · no call-time decision manifest was captured."}</pre>
    {sources.length > 0 && <><p>Expand a currently authorized source. This reads its current contents; captured decision text is not retained.</p>
      {sources.map(({ decisionId, sourceId }) => <button key={`${decisionId}:${sourceId}`} onClick={() => void expand(decisionId, sourceId)}>Read {sourceId}</button>)}
    </>}
    {expanded && <pre className="task-progress-detail">{expanded}</pre>}
  </details>;
}
