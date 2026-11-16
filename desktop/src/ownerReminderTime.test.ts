import { afterEach, describe, expect, it } from "vitest";
import { resolveLocalReminderTime } from "./ownerReminderTime";

declare const process: { env: Record<string, string | undefined> };
const originalTimezone = process.env.TZ;
afterEach(() => { if (originalTimezone === undefined) delete process.env.TZ; else process.env.TZ = originalTimezone; });

describe("local reminder time", () => {
  it("rejects the spring-forward gap and the repeated fall-back hour", () => {
    process.env.TZ = "America/New_York";
    expect(() => resolveLocalReminderTime("2027-03-14T02:30")).toThrow("nonexistent_local_time");
    expect(() => resolveLocalReminderTime("2027-11-07T01:30")).toThrow("ambiguous_local_time");
    expect(resolveLocalReminderTime("2027-03-14T03:30")).toBe("2027-03-14T07:30:00.000Z");
    expect(resolveLocalReminderTime("2027-11-07T02:30")).toBe("2027-11-07T07:30:00.000Z");
  });

  it("accepts ordinary times and rejects malformed calendar dates", () => {
    process.env.TZ = "America/Mexico_City";
    expect(resolveLocalReminderTime("2027-05-10T09:30")).toBe("2027-05-10T15:30:00.000Z");
    expect(() => resolveLocalReminderTime("2027-02-30T09:30")).toThrow();
    expect(() => resolveLocalReminderTime("not a time")).toThrow("invalid_local_time");
  });
});
