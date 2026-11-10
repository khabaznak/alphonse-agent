import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { WidgetCatalogDialog } from "./App";
import { defaultDrawer } from "./atHandState";
import { readDesktopNotificationPreferences } from "./desktopNotifications";

describe("settings mockup controls", () => {
  it("lists real drawer widgets and identifies widgets already added", () => {
    const html = renderToStaticMarkup(<WidgetCatalogDialog state={defaultDrawer()} act={() => {}} member="alex" onClose={() => {}} />);
    expect(html).toContain("A little help, at hand.");
    expect(html).toContain("To-dos");
    expect(html).toContain("Added");
    expect(html).toContain("Add widget");
  });
  it("preserves existing notification settings while enabling new categories by default", () => {
    const storage = { getItem: () => JSON.stringify({ enabled: true, sound: false }) } as unknown as Storage;
    const prefs = readDesktopNotificationPreferences(storage);
    expect(prefs.questions).toBe(true);
    expect(prefs.workProblems).toBe(true);
    expect(prefs.sound).toBe(false);
  });
});
