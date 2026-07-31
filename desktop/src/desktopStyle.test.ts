import { describe, expect, it } from "vitest";
import { parseDesktopStyle } from "./desktopStyle";

describe("parseDesktopStyle", () => {
  it("accepts the supported modern style", () => {
    expect(parseDesktopStyle("modern")).toBe("modern");
  });

  it("falls back to classic for missing or unsupported values", () => {
    expect(parseDesktopStyle(null)).toBe("classic");
    expect(parseDesktopStyle("classic")).toBe("classic");
    expect(parseDesktopStyle("midnight")).toBe("classic");
  });
});
