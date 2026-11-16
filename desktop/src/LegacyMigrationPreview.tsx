import { useEffect, useRef, useState } from "react";
import { daemonRequest } from "./api";

type LegacyResource = Record<string, unknown>;
type PreviewRequest = (method: string, params: Record<string, unknown>) => Promise<unknown>;

/** Inventory only: no migration, resource execution, or file movement. */
export async function loadLegacyMigrationPreview(user: string, request: PreviewRequest = daemonRequest): Promise<LegacyResource[]> {
  if (!user.trim()) throw new Error("A current user is required to review legacy resources.");
  const result = await request("legacy_skill_migration", {
    actor_user_id: user,
    operation: "preview",
  });
  if (!result || typeof result !== "object" || !("resources" in result) || !Array.isArray(result.resources)
      || result.resources.some((item) => !item || typeof item !== "object" || Array.isArray(item))) {
    throw new Error("The legacy resource report is unavailable.");
  }
  return result.resources as LegacyResource[];
}

function label(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

export function LegacyMigrationReport({ resources }: { resources: LegacyResource[] }) {
  if (!resources.length) return <p>No legacy resources need review.</p>;
  return <ul>{resources.map((resource, index) => {
    const source = resource.source && typeof resource.source === "object" && !Array.isArray(resource.source)
      ? resource.source as Record<string, unknown> : {};
    const id = label(resource.artifact_id, `Record ${index + 1}`);
    return <li key={`${id}-${index}`}>
      <strong>{label(source.name, id)}</strong>{" — "}
      <span>{label(resource.migration_state, "unknown state")}</span>{"; "}
      <span>{label(resource.reconciliation, "review required")}</span>
      <details>
        <summary>Review resource details</summary>
        <pre>{JSON.stringify(resource, null, 2)}</pre>
      </details>
    </li>;
  })}</ul>;
}

export function LegacyMigrationPreview({ user, request = daemonRequest }: { user: string; request?: PreviewRequest }) {
  const [result, setResult] = useState<{ user: string; resources: LegacyResource[] } | null>(null);
  const [error, setError] = useState<{ user: string; text: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const sequence = useRef(0);
  useEffect(() => {
    setResult(null);
    setError(null);
    setLoading(false);
    return () => { sequence.current += 1; };
  }, [user]);

  const load = async () => {
    const current = ++sequence.current;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const resources = await loadLegacyMigrationPreview(user, request);
      if (current === sequence.current) setResult({ user, resources });
    } catch (cause) {
      if (current === sequence.current) setError({ user, text: cause instanceof Error ? cause.message : "The report could not be loaded." });
    } finally {
      if (current === sequence.current) setLoading(false);
    }
  };

  return <section aria-label="Legacy resource migration review">
    <h3>Legacy resource review</h3>
    <p>Review retained resource locations and migration status. This report does not move files or change skills.</p>
    <button type="button" className="secondary" onClick={() => void load()} disabled={loading}>
      {loading ? "Loading report…" : "Review legacy resources"}
    </button>
    {error?.user === user && <p role="alert">{error.text}</p>}
    {result?.user === user && <LegacyMigrationReport resources={result.resources} />}
  </section>;
}
