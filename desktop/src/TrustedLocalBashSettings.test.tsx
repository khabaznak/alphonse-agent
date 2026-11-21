import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { readTrustedLocalBash, setTrustedLocalBash, restartTrustedLocalDaemon, TrustedLocalBashView } from "./TrustedLocalBashSettings";

describe("trusted local Bash settings", () => {
  it("reads and updates only the requested preference", async () => {
    const request = vi.fn(async (method: string) => method.startsWith("get_")
      ? { enabled: false, active: false, restart_required: false }
      : { enabled: true, active: false, restart_required: true });
    expect(await readTrustedLocalBash(request)).toEqual({ enabled: false, active: false, restart_required: false });
    expect(await setTrustedLocalBash(request, true)).toEqual({ enabled: true, active: false, restart_required: true });
    expect(request.mock.calls).toEqual([
      ["get_trusted_local_bash_settings", {}],
      ["set_trusted_local_bash_settings", { enabled: true }],
    ]);
  });

  it("rejects malformed daemon replies rather than displaying an incorrect setting", async () => {
    const request = vi.fn(async () => ({ enabled: true, active: "yes", restart_required: false }));
    await expect(readTrustedLocalBash(request)).rejects.toThrow("could not be checked");
    await expect(setTrustedLocalBash(request, false)).rejects.toThrow("could not be verified");
  });

  it("distinguishes saved from running state and explains host access", () => {
    const html = renderToStaticMarkup(<TrustedLocalBashView state={{ enabled: false, active: true, restart_required: true }}
      busy={false} notice="" onChange={() => {}} onRefresh={() => {}} onRestart={() => {}} />);
    expect(html).toContain("your files and network");
    expect(html).toContain("does not enforce a protected household authorization boundary");
    expect(html).toContain("Saved: Off. Running daemon: On.");
    expect(html).toContain("Restart the Alphonse daemon to apply this change.");
    expect(html).toContain("Restart Alphonse daemon");
    expect(html).toContain("may interrupt active work");
    expect(html).toContain("remains available until the daemon restarts");
  });

  it("shows unknown runtime state without claiming the new setting is active", () => {
    const html = renderToStaticMarkup(<TrustedLocalBashView state={{ enabled: true, active: null, restart_required: null }}
      busy={false} notice="" onChange={() => {}} onRefresh={() => {}} onRestart={() => {}} />);
    expect(html).toContain("Running daemon: Unknown");
    expect(html).toContain("running daemon state could not be verified");
  });

  it("reconciles a lost stop reply and verifies the new daemon before reporting status", async () => {
    let queries = 0;
    const request = vi.fn(async (method: string) => {
      if (method === "get_trusted_local_bash_settings") return { enabled: false, active: false, restart_required: false };
      queries++;
      if (queries === 2) throw new Error("disconnected");
      return { daemon_id: queries === 1 ? "old" : "new" };
    });
    const stop = vi.fn(async () => { throw new Error("reply lost"); });
    const ensure = vi.fn(async () => {});
    const result = await restartTrustedLocalDaemon(request, stop, ensure, async () => {});
    expect(result.active).toBe(false);
    expect(stop).toHaveBeenCalledOnce();
    expect(ensure).toHaveBeenCalledOnce();
  });

  it("does not repeat a stop or claim success when the old daemon stays running", async () => {
    const request = vi.fn(async () => ({ daemon_id: "old" }));
    const stop = vi.fn(async () => {});
    const ensure = vi.fn(async () => {});
    await expect(restartTrustedLocalDaemon(request, stop, ensure, async () => {})).rejects.toThrow("could not be confirmed");
    expect(stop).toHaveBeenCalledOnce();
    expect(ensure).not.toHaveBeenCalled();
  });

  it("handles a daemon that restarted before the next poll without starting another", async () => {
    let calls = 0;
    const request = vi.fn(async (method: string) => method === "runtime_identity"
      ? { daemon_id: calls++ === 0 ? "old" : "new" }
      : { enabled: true, active: true, restart_required: false });
    const stop = vi.fn(async () => {});
    const ensure = vi.fn(async () => {});
    expect((await restartTrustedLocalDaemon(request, stop, ensure, async () => {})).active).toBe(true);
    expect(stop).toHaveBeenCalledOnce();
    expect(ensure).not.toHaveBeenCalled();
  });
});
