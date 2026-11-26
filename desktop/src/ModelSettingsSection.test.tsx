// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { ModelSettingsSection } from "./ModelSettingsSection";

const invoke = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/core", () => ({ invoke }));
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

it("rechecks the Codex executable without saving, then saves Terra after a successful retry", async () => {
  const luna = "gpt-6-luna";
  const terra = "gpt-6.1-terra";
  const originalVersion = "codex-cli 0.156.1";
  const updatedVersion = "codex-cli 0.171.0-alpha.13.2";
  const datedSettings = { provider_key: "openai_codex", model_id: luna, cli_version: originalVersion,
    validated_at: "2026-11-26T16:00:00Z", validation_error: "" };
  let cliVersion = originalVersion;
  const catalog = () => ({
    provider_key: "openai_codex", display_name: "OpenAI Codex",
    cli_path: "/opt/homebrew/bin/codex", cli_version: cliVersion,
    catalog_cli_version: "0.171.0", catalog_cli_matches_runtime: cliVersion === updatedVersion,
    models: [
      { model_id: luna, display_name: "GPT-6-Luna" },
      ...(cliVersion === originalVersion ? [{ model_id: terra, display_name: "GPT-6.1 Terra" }] : []),
    ],
  });

  invoke.mockImplementation(async (_command, { method }) => {
    if (method === "inference_providers") return { providers: [catalog()] };
    if (method === "inference_settings") return { settings: datedSettings };
    if (method === "inference_models") return catalog();
    if (method === "set_inference_settings") {
      if (cliVersion === originalVersion) throw `ValueError: openai_codex_model_access_rejected: ${terra}`;
      return { settings: { ...datedSettings, model_id: terra, cli_version: updatedVersion } };
    }
    throw Error(`Unexpected daemon method ${method}`);
  });

  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  const button = (name: string) => {
    const result = [...container.querySelectorAll("button")].find(node => node.textContent?.trim() === name);
    if (!result) throw Error(`Missing button ${name}`);
    return result;
  };
  const methods = (method: string) => invoke.mock.calls.filter(([, request]) => request.method === method);
  const click = (name: string) => act(async () => button(name).click());

  try {
    await act(async () => root.render(<ModelSettingsSection />));
    const select = container.querySelector<HTMLSelectElement>("select")!;
    expect(select.value).toBe(luna);
    await act(async () => {
      select.value = terra;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await click("Validate & save");

    expect(container.querySelector('[role="alert"]')?.textContent).toContain("GPT-6.1 Terra");
    expect(container.textContent).not.toContain("ValueError:");
    expect(container.textContent).toContain("GPT-6-Luna");
    expect(select.value).toBe(terra);
    const recovery = [...container.querySelectorAll("summary")].find(node => node.textContent?.trim() === "Update Codex CLI");
    expect(recovery).toBeDefined();
    await act(async () => recovery?.click());
    await click("Refresh CLI status");
    expect(methods("inference_models")).toHaveLength(1);
    expect(container.textContent).toContain("0.156.1");

    cliVersion = updatedVersion;
    await click("Refresh CLI status");
    expect(methods("inference_models")).toHaveLength(2);
    expect(methods("set_inference_settings")).toHaveLength(1);
    expect(container.textContent).toContain("0.171.0");
    expect(select.value).toBe(terra);

    await click("Retry validation");
    expect(methods("set_inference_settings")).toHaveLength(2);
    expect(methods("set_inference_settings")[1][1].params).toEqual({ provider_key: "openai_codex", model_id: terra });
    expect(container.textContent).toContain("Model validated and saved");
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(button("Validate & save").disabled).toBe(true);
    expect(container.querySelector("details > summary")?.textContent).toContain("Update Codex CLI");
    expect(container.querySelector<HTMLDetailsElement>("details")?.open).toBe(false);
  } finally {
    await act(async () => root.unmount());
    container.remove();
    invoke.mockReset();
  }
});
