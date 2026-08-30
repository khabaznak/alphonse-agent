import { isPermissionGranted, requestPermission, sendNotification } from "@tauri-apps/plugin-notification";

export const DESKTOP_NOTIFICATION_PREFERENCES_KEY = "alphonse.desktop.notifications";
export const ALPHONSE_NOTIFICATION_ICON = "/alphonse-mascot.png";

export type DesktopNotificationPreferences = {
  enabled: boolean;
  sound: boolean;
  soundFile: string;
  onlyWhenUnfocused: boolean;
};

export type DesktopAlert = {
  id: string;
  title: string;
  body: string;
};

export const DEFAULT_DESKTOP_NOTIFICATION_PREFERENCES: DesktopNotificationPreferences = {
  enabled: true,
  sound: true,
  soundFile: "",
  onlyWhenUnfocused: true,
};

export function readDesktopNotificationPreferences(storage: Storage): DesktopNotificationPreferences {
  try {
    const value = JSON.parse(storage.getItem(DESKTOP_NOTIFICATION_PREFERENCES_KEY) || "{}");
    return {
      enabled: value.enabled !== false,
      sound: value.sound !== false,
      soundFile: typeof value.soundFile === "string" ? value.soundFile : "",
      onlyWhenUnfocused: value.onlyWhenUnfocused !== false,
    };
  } catch {
    return { ...DEFAULT_DESKTOP_NOTIFICATION_PREFERENCES };
  }
}

export function notificationForQuestion(question: { question_id: string; message: string; project_id: string }, projectName = ""): DesktopAlert {
  return {
    id: `question:${question.question_id}`,
    title: "Alphonse needs your input",
    body: withProject(question.message, projectName),
  };
}

export function notificationForCompletion(delivery: { outbox_message_id: string; message: string; project_id: string }, projectName = ""): DesktopAlert {
  return {
    id: `completion:${delivery.outbox_message_id}`,
    title: "Task completed",
    body: withProject(delivery.message, projectName),
  };
}

export function createDesktopNotifier(options: {
  isFocused: () => boolean;
  isNativeRuntime: () => boolean;
  isPermissionGranted?: () => Promise<boolean>;
  requestPermission?: () => Promise<"granted" | "denied" | "default">;
  send?: (alert: DesktopAlert, sound: boolean) => void;
  playSound?: (path: string) => Promise<void>;
}) {
  const notified = new Set<string>();
  const granted = options.isPermissionGranted || isPermissionGranted;
  const request = options.requestPermission || requestPermission;
  const deliver = options.send || ((alert: DesktopAlert) => sendNotification({
    title: alert.title,
    body: alert.body,
    icon: ALPHONSE_NOTIFICATION_ICON,
    autoCancel: true,
  }));

  return {
    async notify(alert: DesktopAlert, preferences: DesktopNotificationPreferences): Promise<boolean> {
      if (notified.has(alert.id) || !preferences.enabled || !options.isNativeRuntime()) return false;
      if (preferences.onlyWhenUnfocused && options.isFocused()) return false;
      if (preferences.sound) void options.playSound?.(preferences.soundFile).catch(() => undefined);
      let allowed = await granted();
      if (!allowed) allowed = await request() === "granted";
      notified.add(alert.id);
      if (!allowed) return false;
      try {
        deliver(alert, preferences.sound);
      } catch {
        return false;
      }
      return true;
    },
  };
}

function withProject(message: string, projectName: string): string {
  const preview = String(message || "").replace(/\s+/g, " ").trim().slice(0, 180);
  return projectName ? `${projectName}: ${preview}` : preview;
}
