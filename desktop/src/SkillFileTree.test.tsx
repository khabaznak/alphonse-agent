import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { buildSkillFileTree, SkillFileTree } from "./SkillFileTree";

describe("skill file directory view", () => {
  const files = [
    { path: "SKILL.md", size_bytes: 10, text_editable: true },
    { path: "scripts/nested/run.py", size_bytes: 12, text_editable: true },
    { path: "references/diagram.png", size_bytes: 32, text_editable: false },
  ];
  it("builds sorted nested directories, including empty folders", () => {
    const tree = buildSkillFileTree(files, ["empty", "scripts/nested"]);
    expect(tree.map(node => node.name)).toEqual(["empty", "references", "scripts", "SKILL.md"]);
    expect(tree[2].children[0].children[0].path).toBe("scripts/nested/run.py");
    expect(tree[0].children).toEqual([]);
  });
  it("renders folders as expandable controls and the selected file", () => {
    const html = renderToStaticMarkup(<SkillFileTree files={files} directories={["empty"]} selectedPath="SKILL.md" onSelect={() => {}} />);
    expect(html).toContain('aria-label="Skill package files"');
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain('aria-current="true"');
    expect(html).toContain("empty");
  });
});
