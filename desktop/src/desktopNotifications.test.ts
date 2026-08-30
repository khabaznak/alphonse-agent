import { describe, expect, it } from "vitest";
import { createDesktopNotifier, DEFAULT_DESKTOP_NOTIFICATION_PREFERENCES, notificationForCompletion, notificationForQuestion, readDesktopNotificationPreferences } from "./desktopNotifications";

describe("Desktop notifications", () => {
  it("uses enabled, sound, and background-only preferences by default", () => {
    const storage = { getItem: () => null } as unknown as Storage;
    expect(readDesktopNotificationPreferences(storage)).toEqual(DEFAULT_DESKTOP_NOTIFICATION_PREFERENCES);
  });

  it("includes the event and project in alert text", () => {
    expect(notificationForQuestion({ question_id: "q1", project_id: "home", message: "Which room should I use?" }, "Home")).toEqual({
      id: "question:q1", title: "Alphonse needs your input", body: "Home: Which room should I use?",
    });
    expect(notificationForCompletion({ outbox_message_id: "d1", project_id: "home", message: "The task is complete." }, "Home")).toEqual({
      id: "completion:d1", title: "Task completed", body: "Home: The task is complete.",
    });
  });

  it("sends each background alert once and requests permission when needed", async () => {
    const sent: Array<{ id: string; sound: boolean }> = [];
    const played: string[] = [];
    const notifier = createDesktopNotifier({
      isFocused: () => false,
      isNativeRuntime: () => true,
      isPermissionGranted: async () => false,
      requestPermission: async () => "granted" as const,
      send: (alert, sound) => sent.push({ id: alert.id, sound }),
      playSound: async (path) => { played.push(path); },
    });
    const alert = notificationForCompletion({ outbox_message_id: "d1", project_id: "", message: "Done" });

    expect(await notifier.notify(alert, DEFAULT_DESKTOP_NOTIFICATION_PREFERENCES)).toBe(true);
    expect(await notifier.notify(alert, DEFAULT_DESKTOP_NOTIFICATION_PREFERENCES)).toBe(false);
    expect(sent).toEqual([{ id: "completion:d1", sound: true }]);
    expect(played).toEqual([""]);
  });

  it("still plays the configured sound when notification permission is denied", async () => {
    const played: string[] = [];
    const notifier = createDesktopNotifier({
      isFocused: () => false,
      isNativeRuntime: () => true,
      isPermissionGranted: async () => false,
      requestPermission: async () => "denied" as const,
      send: () => { throw new Error("must not send"); },
      playSound: async (path) => { played.push(path); },
    });

    expect(await notifier.notify(notificationForQuestion({ question_id: "q1", project_id: "", message: "Continue?" }), { ...DEFAULT_DESKTOP_NOTIFICATION_PREFERENCES, soundFile: "/tmp/alert.wav" })).toBe(false);
    expect(played).toEqual(["/tmp/alert.wav"]);
  });

  it("does not notify when focused, disabled, or outside Tauri", async () => {
    const sent: string[] = [];
    const alert = notificationForQuestion({ question_id: "q1", project_id: "", message: "Continue?" });
    const options = {
      isPermissionGranted: async () => true,
      requestPermission: async () => "granted" as const,
      send: (item: { id: string }) => sent.push(item.id),
    };

    expect(await createDesktopNotifier({ ...options, isFocused: () => true, isNativeRuntime: () => true }).notify(alert, DEFAULT_DESKTOP_NOTIFICATION_PREFERENCES)).toBe(false);
    expect(await createDesktopNotifier({ ...options, isFocused: () => false, isNativeRuntime: () => true }).notify(alert, { ...DEFAULT_DESKTOP_NOTIFICATION_PREFERENCES, enabled: false })).toBe(false);
    expect(await createDesktopNotifier({ ...options, isFocused: () => false, isNativeRuntime: () => false }).notify(alert, DEFAULT_DESKTOP_NOTIFICATION_PREFERENCES)).toBe(false);
    expect(sent).toEqual([]);
  });
});
