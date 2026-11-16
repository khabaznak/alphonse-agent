import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LegacyMigrationPreview, LegacyMigrationReport, loadLegacyMigrationPreview } from "./LegacyMigrationPreview";

describe("legacy migration preview", () => {
  it("requests only the actor-scoped preview", async () => {
    const request = vi.fn(async () => ({ resources: [{ artifact_id: "artifact.fixture" }] }));
    await expect(loadLegacyMigrationPreview("owner", request)).resolves.toEqual([{ artifact_id: "artifact.fixture" }]);
    expect(request).toHaveBeenCalledExactlyOnceWith("legacy_skill_migration", { actor_user_id: "owner", operation: "preview" });
  });

  it("rejects malformed reports and a missing actor", async () => {
    const request = vi.fn(async () => ({ resources: [null] }));
    await expect(loadLegacyMigrationPreview("owner", request)).rejects.toThrow("report is unavailable");
    await expect(loadLegacyMigrationPreview("", request)).rejects.toThrow("current user is required");
    expect(request).toHaveBeenCalledOnce();
  });

  it("provides a read-only control and an explicit empty state", () => {
    const html = renderToStaticMarkup(<LegacyMigrationPreview user="owner" />);
    expect(html).toContain("Review legacy resources");
    expect(html).toContain("does not move files or change skills");
    expect(html).not.toContain("Apply");
    expect(renderToStaticMarkup(<LegacyMigrationReport resources={[]} />)).toContain("No legacy resources need review");
  });

  it("makes original report metadata reviewable without interpreting it as HTML", () => {
    const html = renderToStaticMarkup(<LegacyMigrationReport resources={[{
      artifact_id: "artifact.fixture", source: { name: "<script>unsafe</script>", owner_user_id: "owner" },
      reconciliation: "ambiguous", migration_state: "pending", location_candidates: [{ storage_kind: "project" }],
    }]} />);
    expect(html).toContain("ambiguous");
    expect(html).toContain("owner_user_id");
    expect(html).toContain("location_candidates");
    expect(html).not.toContain("<script>unsafe</script>");
  });
});
