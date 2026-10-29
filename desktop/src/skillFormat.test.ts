import { describe, expect, it } from "vitest";
import { limitSkillText, skillTextLength, SKILL_DESCRIPTION_MAX_LENGTH } from "./skillFormat";

describe("Agent Skills description limits", () => {
  it.each([513, 1024])("accepts %i ASCII or supplementary Unicode characters", (length) => {
    for (const symbol of ["a", "😀"]) {
      const text = symbol.repeat(length);
      expect(limitSkillText(text, SKILL_DESCRIPTION_MAX_LENGTH)).toBe(text);
      expect(skillTextLength(text)).toBe(length);
    }
  });
  it("limits pasted text without splitting a supplementary character", () => {
    const text = "x".repeat(1023) + "😀extra";
    expect(limitSkillText(text, SKILL_DESCRIPTION_MAX_LENGTH)).toBe("x".repeat(1023) + "😀");
  });
});
