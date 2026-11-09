import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SkillOverviewCards } from "./App";

describe("skill overview", () => {
  it("renders the base assistant and every installed skill as a visible card", () => {
    const html = renderToStaticMarkup(<SkillOverviewCards user="fixture" onSelect={() => {}} skills={[
      { id: "skill:first", title: "First skill", description: "First description" },
      { id: "skill:second", title: "Second skill", description: "Second description" },
    ]} />);
    expect(html).toContain("Everyday assistant");
    expect(html).toContain("First skill");
    expect(html).toContain("Second skill");
    expect(html).not.toContain("<details");
    expect(html.match(/Manage skill/g)).toHaveLength(2);
  });
});
