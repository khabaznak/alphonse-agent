import { describe, expect, it, vi } from "vitest";
import { desktopStyleStorageKey, followDesktopStyle, parseDesktopStyle, readDesktopStyle, resolveDesktopSkin, saveDesktopStyle } from "./desktopStyle";

function memoryStorage() {
  const values = new Map<string, string>();
  return { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value); } };
}

describe("desktop appearance", () => {
  it("defaults missing, unsupported and former style values to the system skin", () => {
    for (const value of [null, "system", "classic", "modern", "midnight"]) expect(parseDesktopStyle(value)).toBe("system");
    expect(parseDesktopStyle("light")).toBe("light");
    expect(parseDesktopStyle("dark")).toBe("dark");
  });
  it("keeps preferences separate per household member and survives a new read", () => {
    const storage = memoryStorage();
    saveDesktopStyle(storage, "alex", "dark");
    saveDesktopStyle(storage, "family", "light");
    expect(readDesktopStyle(storage, "alex")).toBe("dark");
    expect(readDesktopStyle(storage, "family")).toBe("light");
    expect(readDesktopStyle(storage, "new-member")).toBe("system");
    saveDesktopStyle(storage, "alex", "system");
    expect(readDesktopStyle(storage, "alex")).toBe("system");
    expect(readDesktopStyle(storage, "family")).toBe("light");
    expect(desktopStyleStorageKey("a/b")).not.toBe(desktopStyleStorageKey("a%2Fb"));
  });
  it("does not save an anonymous preference or overwrite the previous style setting", () => {
    const storage = memoryStorage();
    storage.setItem("alphonse.desktop.style", "modern");
    saveDesktopStyle(storage, "", "dark");
    expect(desktopStyleStorageKey("")).toBeNull();
    expect(readDesktopStyle(storage, "")).toBe("system");
    expect(readDesktopStyle(storage, "alex")).toBe("system");
    expect(storage.getItem("alphonse.desktop.style")).toBe("modern");
  });
  it("remains usable when storage is unavailable", () => {
    const storage = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } };
    expect(readDesktopStyle(storage, "alex")).toBe("system");
    expect(() => saveDesktopStyle(storage, "alex", "dark")).not.toThrow();
  });
  it("resolves system changes while honoring explicit overrides", () => {
    expect(resolveDesktopSkin("system", false)).toBe("light");
    expect(resolveDesktopSkin("system", true)).toBe("dark");
    expect(resolveDesktopSkin("light", true)).toBe("light");
    expect(resolveDesktopSkin("dark", false)).toBe("dark");
  });
  it("updates a mounted skin live and removes the media listener on cleanup", () => {
    const root = { dataset: {} as DOMStringMap };
    const media = { matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() };
    const cleanup = followDesktopStyle(root, media, "system");
    expect(root.dataset.alphonseTheme).toBe("light");
    const update = media.addEventListener.mock.calls[0][1] as () => void;
    media.matches = true;
    update();
    expect(root.dataset.alphonseTheme).toBe("dark");
    cleanup();
    expect(media.removeEventListener).toHaveBeenCalledWith("change", update);
    const stop = followDesktopStyle(root, media, "light");
    (media.addEventListener.mock.calls[1][1] as () => void)();
    expect(root.dataset.alphonseTheme).toBe("light");
    stop();
  });
});
