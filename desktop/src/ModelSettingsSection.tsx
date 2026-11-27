import { Brain, CircleAlert, CircleCheck, LoaderCircle } from "lucide-react";
import { type FormEvent, useEffect, useId, useRef, useState } from "react";
import { daemonRequest } from "./api";
import type { InferenceSettings } from "./types";
import "./modelSettings.css";

type Provider = {
  provider_key: string;
  display_name: string;
  models: Array<{ model_id: string; display_name: string }>;
  catalog_cli_version?: string;
  cli_version?: string;
  cli_path?: string;
};

type ValidationFailure = {
  code: string;
  diagnostic: string;
  provider: Provider;
  cliRecovery: boolean;
};

const CODEX_PROVIDER = "openai_codex";

function cliVersion(value?: string) {
  return value?.replace(/^codex(?:-cli)?\s+/i, "").trim() || "";
}

function catalogUsesNewerCli(provider: Provider) {
  const catalog = cliVersion(provider.catalog_cli_version).match(/^\d+\.\d+\.\d+/)?.[0];
  const runtime = cliVersion(provider.cli_version).match(/^\d+\.\d+\.\d+/)?.[0];
  return Boolean(catalog && runtime && catalog.localeCompare(runtime, "en", { numeric: true }) > 0);
}

function validationFailure(cause: unknown, provider: Provider): ValidationFailure {
  const message = cause instanceof Error ? cause.message : String(cause);
  const code = /^(?:[A-Za-z]+Error:\s*)?((?:openai_codex|inference)_[a-z_]+)\b/.exec(message)?.[1] || "";
  const exitCode = code === "openai_codex_exec_failed" ? /\bexit_code=-?\d+\b/.exec(message)?.[0] : undefined;
  const possibleVersionFailure = ["openai_codex_exec_failed", "openai_codex_model_access_rejected", "openai_codex_model_unavailable"].includes(code);
  return {
    code,
    diagnostic: exitCode ? `${code}: ${exitCode}` : code,
    provider,
    cliRecovery: provider.provider_key === CODEX_PROVIDER && (
      code === "openai_codex_cli_upgrade_required" || code === "openai_codex_cli_missing" ||
      (possibleVersionFailure && catalogUsesNewerCli(provider))
    ),
  };
}

function validationMessage(failure: ValidationFailure, provider: Provider) {
  if (failure.cliRecovery && provider.cli_version &&
      (provider.cli_version !== failure.provider.cli_version || provider.cli_path !== failure.provider.cli_path)) {
    return "The previous validation failed. Retry validation to check access with the current CLI.";
  }
  switch (failure.code) {
    case "openai_codex_cli_missing":
      return "Alphonse couldn’t find the Codex CLI. Install Codex, refresh its status, and retry validation.";
    case "openai_codex_auth_required":
    case "openai_codex_subscription_auth_unverified":
      return "Sign in to the Codex CLI with your ChatGPT account, then retry validation.";
    case "openai_codex_timeout":
      return "The access check timed out. Retry validation.";
    case "inference_settings_busy":
      return "Another model validation is in progress. Try again when it finishes.";
    case "inference_save_unverified":
      return "Alphonse couldn’t confirm that the model was saved. Retry validation.";
    default:
      if (failure.cliRecovery) return "The Codex CLI used by Alphonse returned an error. Update the CLI, then retry validation.";
      if (["openai_codex_model_access_rejected", "openai_codex_model_unavailable"].includes(failure.code)) {
        return "Codex couldn’t verify access to this model. Retry validation or choose another model.";
      }
      return "Alphonse couldn’t confirm access to this model. Retry validation.";
  }
}

export function ModelSettingsSection() {
  const [settings, setSettings] = useState<InferenceSettings | null>(null);
  const [provider, setProvider] = useState<Provider | null>(null);
  const [model, setModel] = useState("");
  const [knownModels, setKnownModels] = useState(new Map<string, string>());
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [loadFailed, setLoadFailed] = useState(false);
  const [operation, setOperation] = useState<"validate" | "refresh" | null>(null);
  const [failure, setFailure] = useState<ValidationFailure | null>(null);
  const [saved, setSaved] = useState(false);
  const [refreshStatus, setRefreshStatus] = useState("");
  const selectRef = useRef<HTMLSelectElement>(null);
  const id = useId();

  useEffect(() => {
    let active = true;
    void Promise.all([
      daemonRequest<{ providers: Provider[] }>("inference_providers"),
      daemonRequest<{ settings: InferenceSettings }>("inference_settings"),
    ]).then(([catalog, current]) => {
      if (!active) return;
      const selectedProvider = catalog.providers.find(item => item.provider_key === current.settings.provider_key) || catalog.providers[0];
      if (!selectedProvider) { setLoadFailed(true); return; }
      setProvider(selectedProvider);
      setSettings(current.settings);
      setKnownModels(new Map(selectedProvider.models.map(item => [item.model_id, item.display_name])));
      setModel(selectedProvider.provider_key === current.settings.provider_key ? current.settings.model_id : "");
    }).catch(() => {
      if (active) setLoadFailed(true);
    });
    return () => { active = false; };
  }, [loadAttempt]);

  if (!settings || !provider) {
    return <section className="model-settings" aria-label="Agent model settings">
      <div className="card model-settings-card">
        {loadFailed ? <>
          <div className="model-settings-feedback model-settings-error" role="alert">
            <CircleAlert aria-hidden="true" />
            <div><strong>Model settings couldn’t load</strong><p>Try loading them again.</p></div>
          </div>
          <div className="model-settings-actions model-settings-load-actions">
            <button type="button" className="primary" onClick={() => { setLoadFailed(false); setLoadAttempt(value => value + 1); }}>Try again</button>
          </div>
        </> : <div className="model-settings-feedback model-settings-loading" role="status">
          <LoaderCircle aria-hidden="true" /><strong>Loading model settings…</strong>
        </div>}
      </div>
    </section>;
  }

  const selected = provider.models.find(item => item.model_id === model);
  const selectedName = knownModels.get(model) || model;
  const currentName = knownModels.get(settings.model_id) || settings.model_id;
  const selectedIsSaved = settings.provider_key === provider.provider_key && settings.model_id === model;
  const hasDraft = Boolean(model) && !selectedIsSaved;
  const wantsValidation = hasDraft || (selectedIsSaved && !settings.validated_at);
  const canValidate = knownModels.has(model) && wantsValidation && !operation;
  const isValidating = operation === "validate";
  const version = cliVersion(provider.cli_version);
  const showCliRecovery = provider.provider_key === CODEX_PROVIDER && !isValidating;

  function changeModel(next: string) {
    setModel(next);
    setFailure(null);
    setSaved(false);
    setRefreshStatus("");
  }

  async function saveModel(event: FormEvent, submittedProvider: Provider) {
    event.preventDefault();
    if (!canValidate) return;
    setOperation("validate");
    setFailure(null);
    setSaved(false);
    setRefreshStatus("");
    try {
      const result = await daemonRequest<{ settings: InferenceSettings }>("set_inference_settings", {
        provider_key: submittedProvider.provider_key,
        model_id: model,
      });
      if (result.settings.provider_key !== submittedProvider.provider_key || result.settings.model_id !== model || !result.settings.validated_at) {
        throw new Error("inference_save_unverified");
      }
      setSettings(result.settings);
      setSaved(true);
    } catch (cause) {
      setFailure(validationFailure(cause, submittedProvider));
    } finally {
      setOperation(null);
    }
  }

  async function refreshCli(currentProvider: Provider) {
    if (operation) return;
    setOperation("refresh");
    setRefreshStatus("");
    try {
      const result = await daemonRequest<Provider>("inference_models", { provider_key: currentProvider.provider_key });
      setProvider(result);
      setKnownModels(previous => new Map([...previous, ...result.models.map(item => [item.model_id, item.display_name] as const)]));
      const checkedVersion = cliVersion(result.cli_version);
      const sameInstallation = checkedVersion === version && result.cli_path === currentProvider.cli_path;
      const cliStatus = !checkedVersion ? "Alphonse couldn’t read the version of its Codex CLI."
        : sameInstallation ? `Alphonse still sees Codex CLI ${checkedVersion} at this path.`
          : `Alphonse found Codex CLI ${checkedVersion}.`;
      const selectedStillListed = result.models.some(item => item.model_id === model);
      const catalogStatus = !result.models.length ? "The Codex model catalog is unavailable."
        : hasDraft && !selectedStillListed ? "The selected model is missing from the refreshed catalog." : "";
      const nextStep = checkedVersion && !sameInstallation && knownModels.has(model) && wantsValidation
        ? `Retry validation to check access to ${selectedName}.` : "";
      setRefreshStatus([cliStatus, catalogStatus, nextStep].filter(Boolean).join(" "));
    } catch {
      setRefreshStatus("Alphonse couldn’t refresh the Codex CLI status. Try again.");
    } finally {
      setOperation(null);
    }
  }

  return <section className="model-settings" aria-label="Agent model settings">
    <section className="card model-settings-card model-settings-current" aria-labelledby={`${id}-current`}>
      <div className="model-settings-current-model">
        <span className="model-settings-icon"><Brain aria-hidden="true" /></span>
        <div>
          <h2 id={`${id}-current`}>Current model</h2>
          <p>{currentName || "No model saved"}</p>
          {currentName && <small>{settings.provider_key === provider.provider_key ? provider.display_name : settings.provider_key}</small>}
        </div>
      </div>
      {currentName && (settings.validated_at ? <span className="model-settings-in-use">In use</span> : <small>Validation required</small>)}
    </section>

    <form className="card model-settings-card model-settings-change" aria-labelledby={`${id}-change`} onSubmit={event => void saveModel(event, provider)}>
      <header>
        <h2 id={`${id}-change`}>Change model</h2>
        <p className="model-settings-intro" id={`${id}-intro`}>Alphonse checks access with a live request before saving your selection.</p>
      </header>

      <div className="model-settings-fields">
        <div className="model-settings-field">
          <span className="model-settings-field-label">Provider</span>
          <span>{provider.display_name}</span>
        </div>
        <div className="model-settings-field">
          <div>
            <label className="model-settings-field-label" htmlFor={`${id}-model`}>Model</label>
            <small id={`${id}-model-help`}>{provider.models.length ? "Options come from the Codex catalog." : "The Codex model catalog is unavailable."}</small>
          </div>
          <select ref={selectRef} id={`${id}-model`} value={model} aria-describedby={`${id}-model-help`} disabled={Boolean(operation) || !provider.models.length} onChange={event => changeModel(event.target.value)}>
            {!model && <option value="" disabled>Choose a model</option>}
            {model && !selected && <option value={model} disabled>{selectedName} (unavailable in catalog)</option>}
            {provider.models.map(item => <option key={item.model_id} value={item.model_id}>{item.display_name}</option>)}
          </select>
        </div>
      </div>

      {(failure || showCliRecovery || isValidating || saved) && <div className="model-settings-result">
        {failure && <div className="model-settings-feedback model-settings-error" role="alert">
          <CircleAlert aria-hidden="true" />
          <div><strong>Couldn’t validate {selectedName}</strong><p>{validationMessage(failure, provider)}</p></div>
        </div>}
        {saved && <div className="model-settings-feedback model-settings-success" role="status">
          <CircleCheck aria-hidden="true" />
          <div><strong>Model validated and saved</strong><p>New tasks use {currentName}.</p></div>
        </div>}
        {showCliRecovery && <details className="model-settings-details">
          <summary>Update Codex CLI</summary>
          <div className="model-settings-cli-panel">
            {version ? <p>Alphonse uses <code>{provider.cli_path || "codex"}</code> (<code>{version}</code>).</p>
              : <p>Alphonse couldn’t read the CLI version at <code>{provider.cli_path || "codex"}</code>.</p>}
            <p>{version ? "Update that installation the same way you installed it, then refresh its status here."
              : "Install or repair the Codex CLI at this path, then refresh its status here."}</p>
            <div className="model-settings-actions">
              <button type="button" className="secondary" disabled={Boolean(operation)} onClick={() => void refreshCli(provider)}>{operation === "refresh" ? "Refreshing CLI status…" : "Refresh CLI status"}</button>
            </div>
            <p className="model-settings-cli-status" role="status">{refreshStatus}</p>
          </div>
        </details>}
        {failure && <details className="model-settings-details">
          <summary>Technical details</summary>
          <dl className="model-settings-diagnostics">
            <div><dt>Selected model</dt><dd><code>{model}</code></dd></div>
            {failure.provider.catalog_cli_version && <div><dt>Catalog version</dt><dd><code>{failure.provider.catalog_cli_version}</code></dd></div>}
            <div><dt>CLI at last failure</dt><dd><code>{cliVersion(failure.provider.cli_version) || "Unknown version"} · {failure.provider.cli_path || "codex"}</code></dd></div>
            {failure.diagnostic && <div><dt>CLI result</dt><dd><code>{failure.diagnostic}</code></dd></div>}
          </dl>
        </details>}
        {isValidating && <div className="model-settings-feedback model-settings-loading" role="status">
          <LoaderCircle aria-hidden="true" /><strong>Checking access with {provider.display_name}…</strong>
        </div>}
      </div>}

      <div className="model-settings-actions">
        <button type="submit" className="primary" aria-describedby={`${id}-intro`} disabled={!canValidate}>{isValidating ? "Validating…" : failure ? "Retry validation" : "Validate & save"}</button>
        {hasDraft && <button type="button" className="secondary" disabled={Boolean(operation)} onClick={() => { changeModel(settings.model_id); selectRef.current?.focus(); }}>Discard change</button>}
      </div>
    </form>
  </section>;
}
