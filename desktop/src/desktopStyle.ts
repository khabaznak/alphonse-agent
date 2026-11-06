export type DesktopStyle = "system" | "light" | "dark";
export type DesktopSkin = Exclude<DesktopStyle, "system">;

// Keep the former Classic/Modern key intact; neither value selects a color skin.
export const DESKTOP_STYLE_STORAGE_KEY = "alphonse.desktop.appearance";

export function parseDesktopStyle(value: string | null): DesktopStyle {
  return value === "light" || value === "dark" ? value : "system";
}

export function desktopStyleStorageKey(userId: string): string | null {
  return userId ? `${DESKTOP_STYLE_STORAGE_KEY}.${encodeURIComponent(userId)}` : null;
}

export function readDesktopStyle(storage: Pick<Storage, "getItem">, userId: string): DesktopStyle {
  const key = desktopStyleStorageKey(userId);
  try {
    return key ? parseDesktopStyle(storage.getItem(key)) : "system";
  } catch {
    return "system";
  }
}

export function saveDesktopStyle(storage: Pick<Storage, "setItem">, userId: string, style: DesktopStyle): void {
  const key = desktopStyleStorageKey(userId);
  try {
    if (key) storage.setItem(key, style);
  } catch {
    // Appearance still works for this session when local storage is unavailable.
  }
}

export function resolveDesktopSkin(style: DesktopStyle, systemDark: boolean): DesktopSkin {
  return style === "system" ? (systemDark ? "dark" : "light") : style;
}

/** Apply immediately and follow live system changes. Caller owns listener cleanup. */
export function followDesktopStyle(
  root: Pick<HTMLElement, "dataset">,
  media: Pick<MediaQueryList, "matches" | "addEventListener" | "removeEventListener">,
  style: DesktopStyle,
): () => void {
  const apply = () => { root.dataset.alphonseTheme = resolveDesktopSkin(style, media.matches); };
  apply();
  media.addEventListener("change", apply);
  return () => media.removeEventListener("change", apply);
}
