import { useState } from "react";
import { isGrantCurrent, isOwnerReadGrant, type Catalog, type Connection, type Grant } from "./integrationManagement";

export type AccountServiceDetailProps = {
  catalog: Catalog;
  provider: string;
  capability: string;
  busy: boolean;
  onRefresh: () => void;
  onConnect: (label: string, capabilities: string[]) => void;
  onReconnect: (connection: Connection, capabilities: string[]) => void;
  onRefreshAccess?: (connection: Connection, grant: Grant) => void;
  onGrant: (connection: Connection) => void;
  onGrantContent?: (connection: Connection) => void;
  onAddGmailLabel?: (connection: Connection) => void;
  onEditGmailLabel?: (connection: Connection, grant: Grant) => void;
  onPause: (connection: Connection) => void;
  onDisconnect: (connection: Connection) => void;
  onRetryDeletion: (connection: Connection) => void;
};

function accountStatus(account: Connection, needsCurrentToken: boolean, capability: string): string {
  if (account.credential_deletion_pending || account.revocation_pending) return "Removal needs attention";
  if (account.status === "connected" && !account.capabilities.includes(capability)) return "Service not enabled";
  if (account.status === "connected" && (account.credential_health === "absent" || account.credential_health === "credential_backend_unavailable" || account.access_token_status === "unavailable")) return "Credentials unavailable";
  if (needsCurrentToken && account.status === "connected" && account.access_token_status === "refresh_required") return "Reconnect required";
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
  const ownerSetup = catalog.owner_gmail_setup_only || catalog.owner_google_setup_only;
  const ownerCalendar = catalog.owner_google_setup_only === true && provider === 'google' && capability === 'calendar_read';
  const records = catalog.connections.filter(account => account.provider === provider
    && (!catalog.owner_google_setup_only || account.capabilities.length === 0 || account.capabilities.includes(capability)));
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
        const ownerGmail = catalog.owner_gmail_setup_only === true && provider === 'google' && capability === 'mail_read';
        const status = accountStatus(account, ownerGmail || ownerCalendar, capability);
        const readGrant = catalog.grants.find(grant => grant.connection_id === account.connection_id
          && isOwnerReadGrant(catalog, grant, capability)
          && isGrantCurrent(grant));
        const contentGrants = ownerGmail ? catalog.grants.filter(grant => grant.connection_id === account.connection_id
          && isOwnerReadGrant(catalog, grant, 'mail_content_read')) : [];
        const inboxGrant = contentGrants.find(grant => grant.resources[0] === 'INBOX' && isGrantCurrent(grant));
        const labelGrants = contentGrants.filter(grant => grant.resources[0] !== 'INBOX');
        const combined = ownerSetup && account.capabilities.some(value => value !== capability);
        const desired = scopeSelections[account.connection_id] || (account.capabilities.length ? account.capabilities : [capability]);
        const supported = desired.length > 0 && desired.every(value => availableScopes.includes(value));
        const reconnectable = configured && supported && !combined && !account.credential_deletion_pending;
        const reconnect = status === "Reconnect required" || status === "Finish connecting" || status === "Paused" || status === "Disconnected";
        const refreshable = ownerGmail && reconnectable && status === "Reconnect required" && Boolean(props.onRefreshAccess);
        const refreshGrant = readGrant || inboxGrant || labelGrants.find(isGrantCurrent);
        const manageable = status === "Connected" || refreshable;
        return <section className="mockup-settings-row" key={account.connection_id} aria-label={title}>
          <div><h3>{title}</h3><p>{refreshable ? "Access expired" : status}</p>
            {status === "Reconnect required" && !refreshable && <p>Consent is needed again before this account can be used.</p>}
            {status === "Credentials unavailable" && <p>Credential storage could not provide access. Check account and credential storage before retrying.</p>}
            {status === "Service not enabled" && <p>This saved connection does not include {service} access.</p>}
            {ownerGmail && manageable && !readGrant && !inboxGrant && <p>Grant INBOX access before asking for the unread count.</p>}
            {ownerCalendar && status === "Connected" && !readGrant && <p>Grant access to one calendar before asking about its events.</p>}
            {combined && <p>This connection includes other services; reconnecting it here is unavailable to preserve its existing permissions.</p>}
            {account.credential_deletion_pending && <p>Local credential deletion is pending.</p>}
            {account.revocation_pending && <p>Provider revocation is pending; local access is stopped.</p>}
          </div>
          <div>
            {reconnect && <button type="button" disabled={busy || !reconnectable} onClick={() => props.onReconnect(account, desired)}>{status === "Finish connecting" ? "Finish connecting" : "Reconnect"}</button>}
            {refreshable && refreshGrant && <button type="button" disabled={busy} onClick={() => props.onRefreshAccess?.(account, refreshGrant)}>Refresh access</button>}
            {manageable && ownerGmail && catalog.owner_gmail_content_setup && props.onGrantContent &&
              <button type="button" disabled={busy} onClick={() => props.onGrantContent?.(account)}>
                {inboxGrant ? 'Manage Inbox content access' : 'Grant Inbox content access'}
              </button>}
            {manageable && ownerGmail && catalog.owner_gmail_label_setup && props.onAddGmailLabel &&
              <button type="button" disabled={busy} onClick={() => props.onAddGmailLabel?.(account)}>Add label access</button>}
            {manageable && catalog.owner_gmail_label_setup && props.onEditGmailLabel && labelGrants.map(grant =>
              <button type="button" key={grant.grant_id} disabled={busy} onClick={() => props.onEditGmailLabel?.(account, grant)}>
                Manage access to {grant.gmail_label_name}
              </button>)}
            {(manageable || ownerCalendar && status === "Reconnect required") &&
              <button type="button" disabled={busy} onClick={() => props.onGrant(account)}>
                {ownerGmail && catalog.owner_gmail_content_setup ? 'Manage unread-count access' : 'Manage access'}
              </button>}
            <details><summary>More options</summary><div className="dialog-actions">
              {!ownerSetup && availableScopes.length > 1 && <fieldset><legend>Permissions for reconnection</legend>{availableScopes.map(scope => <label key={scope}><input type="checkbox" checked={desired.includes(scope)} onChange={event => setScopeSelections(current => ({ ...current, [account.connection_id]: event.target.checked ? [...desired, scope] : desired.filter(value => value !== scope) }))} /> {scope === 'mail_read' ? 'Read mail' : scope === 'calendar_read' ? 'Read calendar events' : scope}</label>)}</fieldset>}
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
      ? (catalog.owner_gmail_label_setup
        ? "Choose which Gmail labels Alphonse can search and summarize in Desktop. You can grant Inbox unread counts on their own."
        : catalog.owner_gmail_content_setup
        ? "Inbox content access enables search, summaries, and unread counts in Desktop. You can also grant access to unread counts alone."
        : "Connect Gmail, then grant access to your INBOX to ask for an unread count.")
      : ownerCalendar ? 'Connect Google Calendar, then choose one calendar for agenda questions in Desktop.' : `Connect an account to use ${service}.`}</p>
    <button type="button" className="integration-link" disabled={busy} onClick={props.onRefresh}>Refresh account status</button>
    {visible.length > 0 && <div className="mockup-settings-rows" aria-label={`${service} accounts`}>{visible.map(accountRow)}</div>}
    {removed.length > 0 && <details className="integration-removed-accounts"><summary>Removed accounts ({removed.length}){removed.some(account => account.credential_deletion_pending || account.revocation_pending) ? ' · cleanup needs attention' : ''}</summary><p>These recovery records remain available for pending cleanup or reconnection.</p><div className="mockup-settings-rows">{removed.map(accountRow)}</div></details>}
    {!configured && <p role="status">Provider setup is required before an account can be connected.</p>}
    <details className="integration-connect-form"><summary>{visible.length ? "Add another account" : "Connect an account"}</summary>
      <form onSubmit={event => { event.preventDefault(); if (canAdd && label.trim() && newScopes.length) props.onConnect(label.trim(), newScopes); }}>
        <div className="form-field"><label htmlFor="integration-account-label">Account label</label>
          <input id="integration-account-label" required value={label} onChange={event => setLabel(event.target.value)} disabled={!canAdd} />
        </div>{!ownerSetup && availableScopes.length > 1 ? <fieldset><legend>Permissions</legend>{availableScopes.map(scope => <label key={scope}><input type="checkbox" checked={newScopes.includes(scope)} onChange={event => setNewScopes(event.target.checked ? [...newScopes, scope] : newScopes.filter(value => value !== scope))} /> {scope === 'mail_read' ? 'Read mail' : scope === 'calendar_read' ? 'Read calendar events' : scope}</label>)}</fieldset> : <p>{capability === "mail_read" ? "You will review mail access with the provider." : "You will review calendar access with the provider."}</p>}
        <button type="submit" disabled={!canAdd || !label.trim() || !newScopes.length}>Continue to {provider === "google" ? "Google" : "Microsoft"}</button>
      </form>
    </details>
  </div>;
}
