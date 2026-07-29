import { describe, expect, it } from "vitest";
import { formatMessageTime } from "./messageTime";

describe("formatMessageTime", () => {
  it("uses the configured timezone and exposes the full instant", () => {
    const result = formatMessageTime(
      "2026-07-29T18:30:00+00:00",
      "America/Mexico_City",
      new Date("2026-07-29T18:00:00+00:00"),
    );

    expect(result).not.toBeNull();
    expect(result?.visible).not.toMatch(/2026/);
    expect(result?.tooltip).toContain("America/Mexico_City");
    expect(result?.tooltip).toContain("2026-07-29T18:30:00.000Z");
  });

  it("adds a date for older messages and ignores invalid timestamps", () => {
    const result = formatMessageTime(
      "2026-07-20T15:00:00+00:00",
      "UTC",
      new Date("2026-07-29T18:00:00+00:00"),
    );

    expect(result?.visible).toMatch(/2026/);
    expect(formatMessageTime("not-a-date", "UTC")).toBeNull();
    expect(formatMessageTime("2026-07-20T15:00:00+00:00", "Not/A_Zone")?.tooltip).toContain("UTC");
  });
});
