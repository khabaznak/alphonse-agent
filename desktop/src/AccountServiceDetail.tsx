import { useState } from "react";
import type { Catalog, Connection } from "./integrationManagement";

export type AccountServiceDetailProps = {
  catalog: Catalog;
  provider: string;
  capability: string;
  busy: boolean;
  onRefresh: () => void;
  onConnect: (label: string, capabilities: string[]) => void;
  onReconnect: (connection: Connection, capabilities: string[]) => void;
  onGrant: (connection: Connection) => void;
  onPause: (connection: Connection) => void;
  onDisconnect: (connection: Connection) => void;
  onRetryDeletion: (connection: Connection) => void;
};

function accountStatus(account: Connection, ownerGmail: boolean, capability: string): string {
  if (account.credential_deletion_pending || account.revocation_pending) return "Removal needs attention";
  if (account.status === "connected" && !account.capabilities.includes(capability)) return "Service not enabled";
  if (account.status === "connected" && (account.credential_health === "absent" || account.credential_health === "credential_backend_unavailable" || account.access_token_status === "unavailable")) return "Credentials unavailable";
  if (ownerGmail && account.status === "connected" && account.access_token_status === "refresh_required") return "Reconnect required";
  if (account.status === "connected") return "Connected";
  if (account.status === "reauth_required") return "Finish connecting";
  if (account.status === "paused") return "Paused";
  if (account.status === "disconnected") return "Disconnected";
  return "Account needs attention";
}

/** An account is a stored connection, not a deduplicated provider identity. */
export function AccountServiceDetail(props: AccountServiceDetailProps) {
  const { catalog, provider, capability, busy } = props;
  const [label, setLabel] = useState("");
  const [newScopes, setNewScopes] = useState<string[]>([capability]);
  const [scopeSelections, setScopeSelections] = useState<Record<string, string[]>>({});
  const records = catalog.connections.filter(account => account.provider === provider);
  const availableScopes = catalog.providers.find(item => item.provider === provider)?.capabilities || [];
  const configured = catalog.providers.some(item => item.provider === provider && item.capabilities.includes(capability));
  const service = provider === "google" ? (capability === "calendar_read" ? "Google Calendar" : "Gmail") : "Outlook / Hotmail";
  const canAdd = configured && !busy;

  const visible = records.filter(account => account.status !== "disconnected");
  const removed = records.filter(account => account.status === "disconnected");
  const accountRow = (account: Connection) => {
        const index = records.indexOf(account);
        const repeated = records.filter(item => item.display_name === account.display_name).length > 1;
        const name = account.display_name || "Account";
        const title = repeated ? `${name} · account ${index + 1}` : name;
        const status = accountStatus(account, catalog.owner_gmail_setup_only === true && provider === 'google' && capability === 'mail_read', capability);
        const inboxGrant = catalog.grants.some(grant => grant.connection_id === account.connection_id
          && grant.beneficiary_user_id === catalog.actor_user_id && !grant.revoked_at
          && grant.operations.includes('mail_read') && grant.resources.includes('INBOX')
          && grant.recipient_user_ids.includes(catalog.actor_user_id) && !grant.project_id
          && Date.parse(grant.expires_at) > Date.now());
        const combined = catalog.owner_gmail_setup_only && account.capabilities.some(value => value !== "mail_read");
        const desired = scopeSelections[account.connection_id] || (account.capabilities.length ? account.capabilities : [capability]);
        const supported = desired.length > 0 && desired.every(value => availableScopes.includes(value));
        const reconnectable = configured && supported && !combined && !account.credential_deletion_pending;
        const reconnect = status === "Reconnect required" || status === "Finish connecting" || status === "Paused" || status === "Disconnected";
        return <section className="mockup-settings-row" key={account.connection_id} aria-label={title}>
          <div><h3>{title}</h3><p>{status}</p>
            {status === "Reconnect required" && <p>Consent is needed again before this account can be used.</p>}
            {status === "Credentials unavailable" && <p>Credential storage could not provide access. Check account and credential storage before retrying.</p>}
            {status === "Service not enabled" && <p>This saved connection does not include {service} access.</p>}
            {catalog.owner_gmail_setup_only && status === "Connected" && !inboxGrant && <p>Grant INBOX access before asking for the unread count.</p>}
            {combined && <p>This connection includes other services; reconnecting it here is unavailable to preserve its existing permissions.</p>}
            {account.credential_deletion_pending && <p>Local credential deletion is pending.</p>}
            {account.revocation_pending && <p>Provider revocation is pending; local access is stopped.</p>}
          </div>
          <div>
            {reconnect && <button type="button" disabled={busy || !reconnectable} onClick={() => props.onReconnect(account, desired)}>{status === "Finish connecting" ? "Finish connecting" : "Reconnect"}</button>}
            {status === "Connected" && <button type="button" disabled={busy} onClick={() => props.onGrant(account)}>Manage access</button>}
            <details><summary>More options</summary><div className="dialog-actions">
              {!catalog.owner_gmail_setup_only && availableScopes.length > 1 && <fieldset><legend>Permissions for reconnection</legend>{availableScopes.map(scope => <label key={scope}><input type="checkbox" checked={desired.includes(scope)} onChange={event => setScopeSelections(current => ({ ...current, [account.connection_id]: event.target.checked ? [...desired, scope] : desired.filter(value => value !== scope) }))} /> {scope === 'mail_read' ? 'Read mail' : scope === 'calendar_read' ? 'Read calendar events' : scope}</label>)}</fieldset>}
              {account.status === "connected" && status !== "Reconnect required" && <button type="button" disabled={busy || !reconnectable || account.credential_health === 'credential_backend_unavailable'} onClick={() => props.onReconnect(account, desired)}>Reconnect</button>}
              {account.status === "connected" && <button type="button" disabled={busy} onClick={() => props.onPause(account)}>Pause</button>}
              {account.credential_deletion_pending && <button type="button" disabled={busy} onClick={() => props.onRetryDeletion(account)}>Retry credential deletion</button>}
            </div></details>
            {account.status !== "disconnected" && <button type="button" className="integration-remove-account" disabled={busy} onClick={() => props.onDisconnect(account)}>Remove account…</button>}
          </div>
        </section>;
  };
  return <div className="account-service-detail">
    <p>{provider === "google" && capability === "mail_read"
      ? "Connect Gmail, then grant access to your INBOX to ask for an unread count."
      : `Connect an account to use ${service}.`}</p>
    <button type="button" className="integration-link" disabled={busy} onClick={props.onRefresh}>Refresh account status</button>
    {visible.length > 0 && <div className="mockup-settings-rows" aria-label={`${service} accounts`}>{visible.map(accountRow)}</div>}
    {removed.length > 0 && <details className="integration-removed-accounts"><summary>Removed accounts ({removed.length}){removed.some(account => account.credential_deletion_pending || account.revocation_pending) ? ' · cleanup needs attention' : ''}</summary><p>These recovery records remain available for pending cleanup or reconnection.</p><div className="mockup-settings-rows">{removed.map(accountRow)}</div></details>}
    {!configured && <p role="status">Provider setup is required before an account can be connected.</p>}
    <details className="integration-connect-form"><summary>{visible.length ? "Add another account" : "Connect an account"}</summary>
      <form onSubmit={event => { event.preventDefault(); if (canAdd && label.trim() && newScopes.length) props.onConnect(label.trim(), newScopes); }}>
        <div className="form-field"><label htmlFor="integration-account-label">Account label</label>
          <input id="integration-account-label" required value={label} onChange={event => setLabel(event.target.value)} disabled={!canAdd} />
        </div>{!catalog.owner_gmail_setup_only && availableScopes.length > 1 ? <fieldset><legend>Permissions</legend>{availableScopes.map(scope => <label key={scope}><input type="checkbox" checked={newScopes.includes(scope)} onChange={event => setNewScopes(event.target.checked ? [...newScopes, scope] : newScopes.filter(value => value !== scope))} /> {scope === 'mail_read' ? 'Read mail' : scope === 'calendar_read' ? 'Read calendar events' : scope}</label>)}</fieldset> : <p>{capability === "mail_read" ? "You will review mail access with the provider." : "You will review calendar access with the provider."}</p>}
        <button type="submit" disabled={!canAdd || !label.trim() || !newScopes.length}>Continue to {provider === "google" ? "Google" : "Microsoft"}</button>
      </form>
    </details>
  </div>;
}
