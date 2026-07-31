export type DesktopStyle = "classic" | "modern";

export const DESKTOP_STYLE_STORAGE_KEY = "alphonse.desktop.style";

export function parseDesktopStyle(value: string | null): DesktopStyle {
  return value === "modern" ? "modern" : "classic";
}
