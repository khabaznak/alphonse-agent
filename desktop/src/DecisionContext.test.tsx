import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DecisionContext } from "./DecisionContext";

describe("decision context work log", () => {
  it("labels older records without inventing a captured context", () => {
    const html = renderToStaticMarkup(<DecisionContext summary="" manifests={[]} actorUserId="alex" taskId="old" />);
    expect(html).toContain("Decision context");
    expect(html).toContain("Legacy work log");
    expect(html).toContain("no call-time decision manifest");
  });
  it("renders the server's captured inventory and authorized expansion controls", () => {
    const html = renderToStaticMarkup(<DecisionContext summary="planning · decision:captured · request:original · 340 tokens" manifests={[
      { decision_id: "decision:captured", projections: [{ records: [{ source_id: "conversation:prior" }, { source_id: "policy:hard" }] }] },
    ]} actorUserId="alex" taskId="task" />);
    expect(html).toContain("decision:captured");
    expect(html).toContain("request:original");
    expect(html).toContain("340 tokens");
    expect(html).toContain("Read conversation:prior");
    expect(html).not.toContain("Read policy:hard");
    expect(html).toContain("current contents");
    expect(html).toContain("captured decision text is not retained");
  });
});
