import { useCallback, useEffect, useId, useRef, useState } from "react";
import { openAccountConsent, optionalManagement } from "./api";
import { IntegrationIcon } from "./IntegrationIcon";
import { AccountServiceDetail } from "./AccountServiceDetail";
import { OutlookInboxSettings } from "./OutlookInboxSettings";
import { Catalog, Connection, Definition, GmailLabel, Grant, Member, OUTLOOK_INBOX_UNREAD, Policy, ProjectChecks, ResponseFence, days, defaultDefinition, editableDefinition, friendly, isGrantCurrent, isOwnerCalendarEditGrant, isOwnerReadGrant, labels, sourceDefinition, validateDefinition } from "./integrationManagement";

function useManaged<T>(action: string, values: Record<string, unknown>, identity: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const fence = useRef(new ResponseFence());
  const serialized = JSON.stringify(values);
  const refresh = useCallback(async () => {
    const ticket = fence.current.start();
    try {
      const result = await optionalManagement<T>(action, JSON.parse(serialized));
      if (fence.current.current(ticket)) { setData(result); setError(""); }
    } catch (cause) { if (fence.current.current(ticket)) { setData(null); setError(String(cause instanceof Error ? cause.message : cause)); } }
  }, [action, serialized, identity]);
  useEffect(() => { setData(null); void refresh(); const timer = window.setInterval(() => void refresh(), 15000); return () => { window.clearInterval(timer); fence.current.invalidate(); }; }, [refresh]);
  return { data, error, refresh };
}
function RecipientPicker({ members, selected, onChange, allowed }: { members: Member[]; selected: string[]; onChange: (v: string[]) => void; allowed?: string[] }) {
  return <fieldset className="form-field"><legend>Recipients</legend>{members.filter(m => !allowed || allowed.includes(m.user_id)).map(m => <label key={m.user_id}><input type="checkbox" checked={selected.includes(m.user_id)} onChange={e => onChange(e.target.checked ? [...selected, m.user_id] : selected.filter(id => id !== m.user_id))} /> {m.display_name}</label>)}{allowed?.length === 0 && <p>No recipients are authorized for these sources. Ask the account owner to revise the grant.</p>}</fieldset>;
}
function Coverage({ value }: { value: Record<string, unknown> }) {
  const sources = Object.values((value.sources || {}) as Record<string, { resources?: string[]; coverage_complete?: boolean; observed_at?: string }>);
  const gaps = Array.isArray(value.gaps) ? value.gaps.filter((g): g is string => typeof g === 'string') : [];
  return <div><p>{value.acquisition_complete === true ? 'Acquisition finished for this run.' : 'Acquisition coverage is not yet complete.'}</p>{sources.map((s, n) => <p key={n}>{s.resources?.join(', ') || `Source ${n + 1}`}: {s.coverage_complete === true ? 'Selected source covered' : 'Partial source coverage'}{s.observed_at && ` · observed ${s.observed_at}`}</p>)}{gaps.map(g => <p key={g}>Coverage gap: {friendly(g)}</p>)}</div>;
}
function StatusText({ children }: { children: React.ReactNode }) { return <p role="status" aria-live="polite">{children}</p>; }

type AccountService = { name: string; provider: string; capability: string; description: string; icon: "gmail" | "calendar" | "outlook" };
const ACCOUNT_SERVICES: AccountService[] = [
  { name: "Gmail", icon: "gmail", provider: "google", capability: "mail_read", description: "Read mail from an authorized Google account." },
  { name: "Google Calendar", icon: "calendar", provider: "google", capability: "calendar_read", description: "Read events from an authorized Google calendar." },
  { name: "Google Calendar editing", icon: "calendar", provider: "google", capability: "calendar_write", description: "Review and apply changes to events in one authorized calendar." },
  { name: "Outlook / Hotmail", icon: "outlook", provider: "microsoft", capability: "mail_read", description: "Read mail from an authorized Microsoft account." },
];

export function AccountServiceCards({ catalog, unavailable = false, onChoose }: { catalog: Catalog | null; unavailable?: boolean; onChoose: (provider: string, capability: string) => void }) {
  return <div className="integration-service-grid">{ACCOUNT_SERVICES.filter(service => service.capability !== "calendar_write"
    || catalog?.owner_google_setup_only && catalog.providers.some(item => item.provider === "google" && item.capabilities.includes("calendar_write"))).map(base => {
    const outlookInbox = catalog?.owner_outlook_inbox_setup && base.provider === "microsoft";
    const service = outlookInbox ? { ...base, capability: OUTLOOK_INBOX_UNREAD, description: "Check the unread count in your Outlook or Hotmail Inbox." } : base;
    const provider = catalog?.providers.find(item => item.provider === service.provider);
    const configured = Boolean(provider?.capabilities.includes(service.capability));
    const ownerSetup = catalog?.owner_gmail_setup_only || catalog?.owner_google_setup_only;
    const needsCurrentToken = ownerSetup && service.capability !== "calendar_write";
    const unsupported = Boolean(ownerSetup && !outlookInbox && service.name !== "Gmail"
      && !(catalog?.owner_google_setup_only && service.icon === "calendar"));
    const connections = (catalog?.connections || []).filter(item => item.provider === service.provider
      && (item.capabilities.includes(service.capability) || outlookInbox && !item.capabilities.length));
    const usable = connections.filter(item => item.status === "connected"
      && item.credential_health !== "absent" && item.credential_health !== "credential_backend_unavailable"
      && item.access_token_status !== "unavailable"
      && !(needsCurrentToken && item.access_token_status === "refresh_required")
      && (!outlookInbox || (item.capabilities.includes(service.capability) && !item.credential_deletion_pending && !item.revocation_pending)));
    const connected = usable.length;
    const existing = connections.some(account => !outlookInbox || account.status !== "disconnected"
      || account.credential_deletion_pending || account.revocation_pending);
    const inboxGrants = outlookInbox ? catalog.grants.filter(grant => isOwnerReadGrant(catalog, grant, OUTLOOK_INBOX_UNREAD)
      && isGrantCurrent(grant)) : [];
    const hasInboxAccess = inboxGrants.length === 1 && usable.some(account => account.connection_id === inboxGrants[0].connection_id);
    const status = !catalog ? (unavailable ? "Account management unavailable" : "Checking availability…")
      : unsupported ? "Not yet available in this version"
      : outlookInbox && !configured ? "Provider setup required"
      : outlookInbox && inboxGrants.length > 1 ? "Inbox grants need attention"
      : outlookInbox && connected ? (hasInboxAccess ? "Inbox access granted" : "Connected · Inbox access needed")
      : connected ? `${connected} ${connected === 1 ? "account" : "accounts"} connected`
      : existing ? "Account needs attention" : configured ? "Available to connect" : "Provider setup required";
    const action = unsupported ? "Unavailable" : existing ? "Manage" : configured ? "Connect" : "View setup";
    return <article className="mockup-settings-card integration-service-card" key={service.name}>
      <div className="integration-card-heading"><IntegrationIcon service={service.icon} /><div><h3>{service.name}</h3><p>{service.description}</p></div></div>
      <small className="integration-card-status">{status}</small>
      <div className="integration-card-footer">
        <span>{service.icon === "calendar" ? "Calendar" : "Email"}</span>
        <button type="button" data-outlook-inbox-card={outlookInbox || undefined} data-calendar-edit-card={service.capability === "calendar_write" || undefined} disabled={!catalog || unsupported}
          aria-label={`${action}${action === "View setup" ? " for" : ""} ${service.name}`}
          onClick={() => onChoose(service.provider, service.capability)}>{action}{!unsupported && " →"}</button>
      </div>
    </article>;
  })}<article className="mockup-settings-card integration-service-card"><div className="integration-card-heading"><IntegrationIcon service="outlook" /><div><h3>Outlook Calendar</h3><p>Access events from a Microsoft calendar.</p></div></div><small className="integration-card-status">Not yet available</small><div className="integration-card-footer"><span>Calendar</span></div></article></div>;
}

export function OptionalAccounts({ identity, messaging, onConversation, outlookRequested, onOutlookOpened }: {
  identity: string; messaging?: React.ReactNode; onConversation?: () => void; outlookRequested?: boolean; onOutlookOpened?: () => void;
}) {
  const managed = useManaged<Catalog>('catalog', {}, identity);
  const mismatch = Boolean(managed.data && managed.data.actor_user_id !== identity);
  const data = mismatch ? null : managed.data; const error = mismatch ? 'The active account session does not match the selected household member. Reconnect the correct session to manage accounts.' : managed.error; const refresh = managed.refresh;
  const [view, setView] = useState<'overview' | 'service' | 'access'>('overview');
  const [notice, setNotice] = useState(""); const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState('google'); const [serviceCapability, setServiceCapability] = useState("mail_read");
  const [grantOperation, setGrantOperation] = useState("mail_read");
  const [grantFor, setGrantFor] = useState<Connection | null>(null); const [editingGrant, setEditingGrant] = useState<Grant | undefined>();
  const [editingGmailLabel, setEditingGmailLabel] = useState(false);
  const [accessFromService, setAccessFromService] = useState(false);
  const [disconnectFor, setDisconnectFor] = useState<Connection | null>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const chooseService = (nextProvider: string, capability: string) => {
    if (capability === OUTLOOK_INBOX_UNREAD) setNotice("");
    setProvider(nextProvider); setServiceCapability(capability); setGrantFor(null); setView("service");
  };
  useEffect(() => {
    if (!outlookRequested || !data?.owner_outlook_inbox_setup) return;
    chooseService("microsoft", OUTLOOK_INBOX_UNREAD);
    onOutlookOpened?.();
  }, [outlookRequested, data?.owner_outlook_inbox_setup, onOutlookOpened]);
  const outlookService = serviceCapability === OUTLOOK_INBOX_UNREAD;
  const calendarEditService = serviceCapability === "calendar_write";
  useEffect(() => {
    if (view !== "service" || !(outlookService || calendarEditService)) return;
    heading.current?.focus();
  }, [outlookService, calendarEditService, view]);
  const returnToOverview = () => {
    setView("overview"); setGrantFor(null); setDisconnectFor(null);
    if (outlookService || calendarEditService) window.requestAnimationFrame(() => {
      const selector = outlookService ? "[data-outlook-inbox-card]" : "[data-calendar-edit-card]";
      const control = heading.current?.parentElement?.querySelector<HTMLButtonElement>(`button${selector}:not(:disabled)`);
      (control || heading.current)?.focus();
    });
  };
  const run = async (action: string, values: Record<string, unknown>) => {
    setBusy(true); setNotice("");
    try { await optionalManagement(action, values); setNotice(action === "refresh_gmail_access" ? "Gmail access refreshed." : "Saved. Dependent checks may require review."); await refresh(); }
    catch (cause) { setNotice(cause instanceof Error ? cause.message : String(cause)); await refresh(); }
    finally { setBusy(false); }
  };
  const begin = async (label: string, capabilities: string[], connection?: Connection) => {
    setBusy(true); setNotice("");
    try {
      const result = await optionalManagement<{ authorization_url: string }>('begin', { connection_id: connection?.connection_id || `account_${crypto.randomUUID()}`, provider: connection?.provider || provider, expected_revision: connection?.revision || 0, display_name: connection?.display_name || label, capabilities });
      await openAccountConsent(result.authorization_url); setNotice("Complete consent in your browser, then refresh account health."); await refresh();
    } catch (cause) { setNotice(cause instanceof Error ? cause.message : String(cause)); await refresh(); }
    finally { setBusy(false); }
  };
  const service = ACCOUNT_SERVICES.find(item => item.provider === provider && item.capability === serviceCapability) || ACCOUNT_SERVICES.find(item => item.provider === provider);
  const selectGrant = (account: Connection, operation: string, grant?: Grant, gmailLabel = false) => {
    setEditingGmailLabel(gmailLabel);
    if (view === "service") setAccessFromService(true);
    setNotice("");
    setGrantFor(account); setGrantOperation(operation); setEditingGrant(grant); setView("access");
  };
  const manageGrant = (account: Connection, operation: string) => {
    if (!data) return;
    const existing = data.grants.find(grant => grant.connection_id === account.connection_id
      && (operation === 'calendar_write' ? isOwnerCalendarEditGrant(data, grant) : isOwnerReadGrant(data, grant, operation))
      && (operation !== 'mail_content_read' || grant.resources[0] === 'INBOX'));
    selectGrant(account, operation, existing);
  };
  const currentGrantAccount = data?.connections.find(account => account.connection_id === grantFor?.connection_id);
  const currentGmailLabelGrant = data?.grants.find(grant => grant.grant_id === editingGrant?.grant_id && !grant.revoked_at);
  const finishGrant = () => {
    setGrantFor(null);
    heading.current?.focus();
    void refresh();
  };
  return <section className="optional-management"><h3 tabIndex={-1} ref={heading} className="integration-context-heading"
    data-outlook-settings-heading={view === "service" && outlookService || undefined}>{view === "overview" ? "Personal accounts" : view === "access" ? (data?.owner_gmail_setup_only ? "Account access" : "Account access and skills") : service?.name}</h3><StatusText>{error || notice}</StatusText>
    {view === "overview" ? <><div className="integration-owner-note"><strong>{data ? `Managing accounts as ${data.members.find(m => m.user_id === data.actor_user_id)?.display_name || data.actor_user_id}` : "Personal accounts"}</strong><p>Each household member connects their own accounts and chooses what Alphonse can access.</p></div><AccountServiceCards catalog={data} unavailable={Boolean(error)} onChoose={chooseService} />{data && <button className="integration-link" onClick={() => { setAccessFromService(false); setView("access"); }}>{data.owner_gmail_setup_only ? "Review access grants →" : "Review access grants and skill accounts →"}</button>}{messaging}<p className="integration-privacy"><strong>Personal by default.</strong> Access to an account requires explicit permission. Connecting an account does not start monitoring.</p></> : <><button className="integration-back" onClick={returnToOverview}>← All integrations</button>{data && <>
      {view === "access" && accessFromService && <button className="integration-back" onClick={() => { setView("service"); finishGrant(); }}>← {service?.name} settings</button>}
      <p>Managing accounts as {data.members.find(m => m.user_id === data.actor_user_id)?.display_name || data.actor_user_id}.</p>
      {view === "service" && outlookService && data.owner_outlook_inbox_setup && <OutlookInboxSettings
        key={data.actor_user_id} catalog={data} onRefresh={refresh} onConversation={onConversation} />}
      {view === "service" && !outlookService && <><AccountServiceDetail key={`${provider}:${serviceCapability}`} catalog={data} provider={provider} capability={serviceCapability} busy={busy}
        onRefresh={() => void refresh()} onConnect={(label, scopes) => void begin(label, scopes)}
        onReconnect={(connection, scopes) => void begin(connection.display_name, scopes, connection)}
        onRefreshAccess={(connection, grant) => void run('refresh_gmail_access', { connection_id: connection.connection_id,
          expected_revision: connection.revision, grant_id: grant.grant_id, grant_revision: grant.revision })}
        onGrant={connection => manageGrant(connection, serviceCapability)}
        onGrantContent={connection => manageGrant(connection, 'mail_content_read')}
        onAddGmailLabel={connection => selectGrant(connection, 'mail_content_read', undefined, true)}
        onEditGmailLabel={(connection, grant) => selectGrant(connection, 'mail_content_read', grant, true)}
        onPause={connection => void run('pause', { connection_id: connection.connection_id, expected_revision: connection.revision })}
        onDisconnect={setDisconnectFor}
        onRetryDeletion={connection => void run('retry_deletion', { connection_id: connection.connection_id, expected_revision: connection.revision })} />
        {disconnectFor && <div className="project-detail" role="group" aria-label="Confirm account removal"><p>Remove {disconnectFor.display_name}? Alphonse will stop local access and revoke its grants. Provider revocation or credential cleanup may remain pending. A recovery record is retained.</p><button disabled={busy} onClick={() => { void run('disconnect', { connection_id: disconnectFor.connection_id, expected_revision: disconnectFor.revision }); setDisconnectFor(null); heading.current?.focus(); }}>Remove account</button><button disabled={busy} onClick={() => { setDisconnectFor(null); heading.current?.focus(); }}>Keep account</button></div>}
      </>}{view === "access" && <>
      {editingGmailLabel && currentGrantAccount && data.owner_gmail_label_setup && (!editingGrant || currentGmailLabelGrant)
        ? <OwnerGmailLabelGrant key={`${data.actor_user_id}:${currentGrantAccount.connection_id}:${currentGrantAccount.revision}:${currentGmailLabelGrant?.grant_id || "new"}:${currentGmailLabelGrant?.revision || 0}`}
            account={currentGrantAccount} grant={currentGmailLabelGrant} catalog={data} onCancel={finishGrant}
            onSave={() => { setNotice("Gmail label access saved."); finishGrant(); }} />
        : grantFor && !editingGmailLabel && <GrantEditor key={`${grantFor.connection_id}:${grantOperation}:${editingGrant?.grant_id || "new"}`} account={grantFor} grant={editingGrant} operation={grantOperation} catalog={data} onDone={finishGrant} />}
      <h4>Explicit grants</h4><p>Grants you gave or received appear here.</p>{data.grants.map(g => <GrantCard key={g.grant_id} grant={g} catalog={data} busy={busy} run={run} onEdit={() => {
        if (g.operations.length === 1 && g.operations[0] === OUTLOOK_INBOX_UNREAD) {
          setNotice(""); chooseService("microsoft", OUTLOOK_INBOX_UNREAD); return;
        }
        const account = data.connections.find(c => c.connection_id === g.connection_id);
        if (account) selectGrant(account, g.operations[0], g,
          g.operations[0] === 'mail_content_read' && g.resources[0] !== 'INBOX' && Boolean(g.gmail_label_name));
      }} />)}
      {!data.owner_gmail_setup_only && <><h4>Choose a skill account role</h4><p>A grant permits access; selecting a role chooses which permitted account a skill uses.</p><BindingPicker catalog={data} run={run} busy={busy} /></>}
    </>}</>}</>}</section>;
}
function BindingPicker({ catalog, run, busy }: { catalog: Catalog; run: (a: string, v: Record<string, unknown>) => Promise<void>; busy: boolean }) {
  const [binding, setBinding] = useState(""); const [role, setRole] = useState("mail");
  return <form onSubmit={e => { e.preventDefault(); const b = catalog.bindings.find(b => b.binding_id === binding); if (b) void run('select_binding', { skill_id: b.skill_id, binding_id: binding, name: role }); }}><div className="form-field"><label htmlFor="optional-binding">Granted skill account</label><select id="optional-binding" required value={binding} onChange={e => setBinding(e.target.value)}><option value="">Choose account binding</option>{catalog.bindings.map(b => <option key={b.binding_id} value={b.binding_id}>{catalog.skills.find(s => s.skill_id === b.skill_id)?.name} · {b.grant_id}</option>)}</select><label htmlFor="optional-role">Skill role (from its instructions)</label><input id="optional-role" required pattern={"[a-z][a-z0-9_\\-]{0,63}"} value={role} onChange={e => setRole(e.target.value)} /></div><button disabled={busy || !binding}>Use account for this role</button></form>;
}
function GrantCard({ grant: g, catalog, busy, run, onEdit }: { grant: Grant; catalog: Catalog; busy: boolean; onEdit: () => void; run: (a: string, v: Record<string, unknown>) => Promise<void> }) {
  const [skill, setSkill] = useState("");
  const isGmailContent = g.operations.length === 1 && g.operations[0] === 'mail_content_read';
  const isOutlookInbox = g.operations.length === 1 && g.operations[0] === OUTLOOK_INBOX_UNREAD;
  const isCalendarEdit = isOwnerCalendarEditGrant(catalog, g);
  const resource = isOutlookInbox ? "Outlook / Hotmail Inbox" : isGmailContent && g.gmail_label_name ? g.gmail_label_name : g.resources.join(', ');
  const accountName = catalog.connections.find(account => account.connection_id === g.connection_id)?.display_name || g.connection_id;
  const status = g.revoked_at ? 'Revoked' : isGrantCurrent(g) ? 'Granted' : 'Expired';
  return <article className="project-detail">
    <h5>{catalog.members.find(m => m.user_id === g.beneficiary_user_id)?.display_name} · {accountName}</h5>
    <p>{g.operations.map(o => labels[o] || friendly(o)).join(', ')} in {resource} · {g.expires_at
      ? `Expires ${new Date(g.expires_at).toLocaleString()}`
      : isGmailContent ? 'Until revoked' : 'No expiry available'} · {status}</p>
    {isCalendarEdit || (isGmailContent || isOutlookInbox) && isOwnerReadGrant(catalog, g) ? <p>Private Desktop conversation</p>
      : <p>Recipients: {g.recipient_user_ids.map(id => catalog.members.find(m => m.user_id === id)?.display_name || id).join(', ')} · Project: {catalog.projects.find(p => p.project_id === g.project_id)?.name || (g.project_id ? 'Restricted project' : 'Any permitted project')}</p>}
    {g.grantor_user_id === catalog.actor_user_id && !g.revoked_at && <>
      {(!(catalog.owner_gmail_setup_only || catalog.owner_google_setup_only) || isOwnerReadGrant(catalog, g) || isCalendarEdit) &&
        <button disabled={busy} onClick={onEdit} aria-label={isGmailContent ? `Revise access to ${resource}` : undefined}>Revise access grant</button>}
      <button disabled={busy} aria-label={isGmailContent ? `Revoke access to ${resource}` : undefined}
        onClick={() => void run('revoke_grant', { grant_id: g.grant_id, expected_revision: g.revision })}>Revoke grant</button>
      {!catalog.owner_gmail_setup_only && !isCalendarEdit && <><label>Associate with skill<select value={skill} onChange={e => setSkill(e.target.value)}><option value="">Choose skill</option>{catalog.skills.map(s => <option key={s.skill_id} value={s.skill_id}>{s.name}</option>)}</select></label><button disabled={busy || !skill} onClick={() => void run('bind_grant', { grant_id: g.grant_id, skill_id: skill })}>Create skill binding</button></>}
    </>}
  </article>;
}

function grantErrorCode(cause: unknown): string {
  const message = cause instanceof Error ? cause.message : String(cause);
  return message.replace(/^\w+: /, "");
}

function gmailLabelError(cause: unknown, fallback: string): string {
  switch (grantErrorCode(cause)) {
    case 'oauth_refresh_required':
      return 'Gmail access expired. Return to Gmail settings to refresh access or reconnect, then try again.';
    case 'owner_only_account_scope_unavailable':
    case 'oauth_connection_unavailable':
    case 'oauth_reauth_required':
      return 'Gmail read access is unavailable. Return to Gmail settings and reconnect the account.';
    case 'oauth_provider_unconfigured':
      return 'Gmail account setup is unavailable. Check the provider configuration before trying again.';
    case 'oauth_provider_unavailable':
      return 'Google could not refresh Gmail access. Try again.';
    case 'oauth_revision_conflict':
      return 'The Gmail connection changed. Return to Gmail settings, refresh account status, and try again.';
    case 'gmail_label_selection_stale':
      return 'The selected Gmail label changed. Load Gmail labels again and review the selection.';
    default:
      return fallback;
  }
}

function useOwnerGrantExpiry(grant: Grant | undefined, allowStanding: boolean) {
  const id = useId();
  const [untilRevoked, setUntilRevoked] = useState(allowStanding && (!grant || grant.expires_at === null));
  const [date, setDate] = useState(localInput(grant?.expires_at || null));
  const expiry = new Date(date);
  const valid = untilRevoked || expiry.getTime() > Date.now();
  return { id, allowStanding, untilRevoked, setUntilRevoked, date, setDate, expiry, valid };
}

function OwnerGrantExpiry({ expiry }: { expiry: ReturnType<typeof useOwnerGrantExpiry> }) {
  const { id, allowStanding, untilRevoked, setUntilRevoked, date, setDate } = expiry;
  return <>
    {allowStanding && <><label htmlFor={`${id}-mode`}>Access duration</label>
      <select id={`${id}-mode`} value={untilRevoked ? "standing" : "date"} onChange={e => setUntilRevoked(e.target.value === "standing")}>
        <option value="standing">Until revoked</option><option value="date">Choose an end date</option>
      </select></>}
    {!untilRevoked && <><label htmlFor={`${id}-date`}>Access expires (local time)</label>
      <input id={`${id}-date`} type="datetime-local" required value={date} onChange={e => setDate(e.target.value)} /></>}
  </>;
}

function OwnerGmailLabelGrant({ account, catalog, grant, onCancel, onSave }: { account: Connection; catalog: Catalog; grant?: Grant; onCancel: () => void; onSave: () => void }) {
  const id = useId();
  const heading = useRef<HTMLHeadingElement>(null);
  const fence = useRef(new ResponseFence());
  const [options, setOptions] = useState<GmailLabel[] | null>(null);
  const [labelId, setLabelId] = useState(grant?.resources[0] || "");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState("");
  const expiry = useOwnerGrantExpiry(grant, true);
  const loadLabels = useCallback(async () => {
    const ticket = fence.current.start();
    setOptions(null);
    setNotice("");
    setLoading(true);
    try {
      const result = await optionalManagement<{ labels: GmailLabel[] }>('list_gmail_labels', {
        connection_id: account.connection_id, expected_revision: account.revision,
      });
      if (fence.current.current(ticket)) setOptions(result.labels);
    } catch (cause) {
      if (fence.current.current(ticket)) setNotice(gmailLabelError(cause, 'Gmail labels could not be loaded. Try again.'));
    } finally {
      if (fence.current.current(ticket)) setLoading(false);
    }
  }, [account.connection_id, account.revision]);
  useEffect(() => {
    heading.current?.focus();
    void loadLabels();
    return () => fence.current.invalidate();
  }, [loadLabels]);
  const grantedIds = new Set(catalog.grants.filter(g => g.connection_id === account.connection_id
    && isOwnerReadGrant(catalog, g, 'mail_content_read')).map(g => g.resources[0]));
  const choices = options?.filter(label => grant
    ? label.id === grant.resources[0]
    : label.id !== 'INBOX' && !grantedIds.has(label.id));
  const selected = choices?.find(label => label.id === labelId);
  const save = async () => {
    if (!selected || !expiry.valid || saving || loading) return;
    const ticket = fence.current.start();
    setSaving(true);
    setNotice("");
    try {
      await optionalManagement('save_grant', {
        grant_id: grant?.grant_id || `grant_${crypto.randomUUID()}`,
        connection_id: account.connection_id,
        expected_revision: grant?.revision || 0,
        beneficiary_user_id: catalog.actor_user_id,
        operations: ['mail_content_read'],
        resources: [selected.id],
        gmail_label_name: selected.name,
        recipient_user_ids: [catalog.actor_user_id],
        expires_at: expiry.untilRevoked ? null : expiry.expiry.toISOString(),
        project_id: null,
      });
      if (fence.current.current(ticket)) onSave();
    } catch (cause) {
      if (fence.current.current(ticket)) {
        if (grantErrorCode(cause) === 'gmail_label_selection_stale') {
          setOptions(null);
          setLabelId(grant?.resources[0] || "");
        }
        setNotice(gmailLabelError(cause, 'Gmail label access could not be saved. Try again.'));
      }
    } finally {
      if (fence.current.current(ticket)) setSaving(false);
    }
  };
  return <section className="project-detail">
    <h4 ref={heading} tabIndex={-1}>{grant ? 'Manage Gmail label access' : 'Add Gmail label access'}</h4>
    <p>Alphonse can find and summarize messages under one label for your private Desktop conversation. For summaries, matching excerpts go to your configured AI model.</p>
    <StatusText>{loading ? 'Loading Gmail labels…' : notice}</StatusText>
    {!loading && !options && <button type="button" disabled={saving} onClick={() => void loadLabels()}>Load Gmail labels again</button>}
    {choices?.length === 0 && <p>{grant ? 'Gmail no longer lists this label on the account.' : 'All available labels already have access or use the Inbox permission.'}</p>}
    {choices && choices.length > 0 && <form onSubmit={e => { e.preventDefault(); void save(); }}>
      <fieldset disabled={saving}>
        <label htmlFor={`${id}-label`}>Gmail label</label>
        <select id={`${id}-label`} required value={labelId} disabled={Boolean(grant)} onChange={e => setLabelId(e.target.value)}>
          <option value="">Choose a label</option>
          {choices.map(label => <option key={label.id} value={label.id}>{label.name}</option>)}
        </select>
        {grant && selected && selected.name !== grant.gmail_label_name && <p>Gmail now calls this label {selected.name}. Review the name before saving.</p>}
        <OwnerGrantExpiry expiry={expiry} />
        <button disabled={!selected || !expiry.valid}>{grant ? 'Save label access' : 'Authorize label access'}</button>
      </fieldset>
    </form>}
    <button type="button" disabled={saving} onClick={onCancel}>Cancel</button>
  </section>;
}

function GrantEditor({ account, catalog, grant, operation, onDone }: { account: Connection; catalog: Catalog; grant?: Grant; operation: string; onDone: () => void }) {
  if (catalog.owner_gmail_setup_only || catalog.owner_google_setup_only) return <OwnerReadGrant account={account} catalog={catalog} grant={grant} operation={operation} onDone={onDone} />;
  return <GeneralGrantEditor account={account} catalog={catalog} grant={grant} onDone={onDone} />;
}
function OwnerReadGrant({ account, catalog, grant, operation, onDone }: { account: Connection; catalog: Catalog; grant?: Grant; operation: string; onDone: () => void }) {
  const calendarEdit = operation === 'calendar_write';
  const calendar = operation === 'calendar_read' || calendarEdit;
  const gmailContent = operation === 'mail_content_read';
  const [resource, setResource] = useState(calendar ? grant?.resources[0] || '' : 'INBOX');
  const expiry = useOwnerGrantExpiry(grant, gmailContent && catalog.owner_gmail_label_setup === true);
  const [busy, setBusy] = useState(false); const [notice, setNotice] = useState("");
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => { if (calendarEdit) heading.current?.focus(); }, [calendarEdit]);
  const save = async () => { setBusy(true); setNotice(""); try {
    await optionalManagement('save_grant', { grant_id: grant?.grant_id || `grant_${crypto.randomUUID()}`, connection_id: account.connection_id, expected_revision: grant?.revision || 0, beneficiary_user_id: catalog.actor_user_id, operations: [operation], resources: [resource.trim()], recipient_user_ids: [catalog.actor_user_id], expires_at: expiry.untilRevoked ? null : expiry.expiry.toISOString(), project_id: null });
    onDone();
  } catch (cause) { setNotice(cause instanceof Error ? cause.message : String(cause)); setBusy(false); } };
  return <section className="project-detail">
    <h4 ref={heading} tabIndex={calendarEdit ? -1 : undefined}>{calendarEdit ? 'Allow approved edits to one Google calendar' : calendar ? 'Allow access to one Google calendar' : gmailContent ? 'Allow access to Gmail Inbox content' : 'Allow access to your Gmail INBOX'}</h4>
    {gmailContent ? <p>This lets Alphonse count unread Inbox mail and find matching messages in your private Desktop conversation. For summaries, matching excerpts go to your configured AI model.</p>
      : <p>{calendarEdit ? 'Each event edit requires your approval in the private Desktop conversation.' : calendar ? 'This lets you ask for event titles and times in your private Desktop conversation until the grant expires.' : 'This lets you ask for the unread INBOX count until the grant expires.'}</p>}
    <StatusText>{notice}</StatusText>
    <form className={calendarEdit ? "calendar-write-grant" : undefined} onSubmit={e => { e.preventDefault(); if (expiry.valid && resource.trim() && !busy) void save(); }}>
      {calendar && <><label htmlFor="owner-calendar-id">Exact calendar ID</label><input id="owner-calendar-id" aria-describedby="owner-calendar-id-help" type="text" required readOnly={calendarEdit && Boolean(grant)} maxLength={512} value={resource} onChange={e => setResource(e.target.value)} />
        <small id="owner-calendar-id-help">Copy the calendar ID from Google Calendar’s settings for that calendar, under Integrate calendar.</small></>}
      <OwnerGrantExpiry expiry={expiry} />
      <button disabled={busy || !expiry.valid || !resource.trim()}>{calendarEdit ? 'Authorize calendar edits' : calendar ? 'Authorize calendar access' : gmailContent ? 'Authorize Inbox content access' : 'Authorize INBOX access'}</button>
    </form><button disabled={busy} onClick={onDone}>Cancel</button>
  </section>;
}
function GeneralGrantEditor({ account, catalog, grant, onDone }: { account: Connection; catalog: Catalog; grant?: Grant; onDone: () => void }) {
  const id = useId(); const [beneficiary, setBeneficiary] = useState(grant?.beneficiary_user_id || ""); const [operations, setOperations] = useState<string[]>(grant?.operations || []); const [resources, setResources] = useState(grant?.resources.join("\n") || ""); const [recipients, setRecipients] = useState<string[]>(grant?.recipient_user_ids || []); const [expires, setExpires] = useState(localInput(grant?.expires_at || null)); const [project, setProject] = useState(grant?.project_id || ""); const [review, setReview] = useState(false); const [notice, setNotice] = useState(""); const [busy, setBusy] = useState(false); const focus = useRef<HTMLHeadingElement>(null);
  useEffect(() => { focus.current?.focus(); }, [review]);
  const save = async () => { setBusy(true); try { await optionalManagement('save_grant', { grant_id: grant?.grant_id || `grant_${crypto.randomUUID()}`, connection_id: account.connection_id, expected_revision: grant?.revision || 0, beneficiary_user_id: beneficiary, operations, resources: resources.split('\n').map(s => s.trim()).filter(Boolean), recipient_user_ids: recipients, expires_at: new Date(expires).toISOString(), project_id: project || null }); onDone(); } catch (cause) { setNotice(String(cause)); setBusy(false); } };
  return <section className="project-detail"><h4 tabIndex={-1} ref={focus}>Grant access to {account.display_name}</h4><StatusText>{notice}</StatusText>{review ? <><p>Allow {catalog.members.find(m => m.user_id === beneficiary)?.display_name} to {operations.map(o => labels[o] || friendly(o)).join(', ')} in {resources.split('\n').join(', ')} until {expires} (local time), for {catalog.projects.find(p => p.project_id === project)?.name || (project ? `restricted project ${project}` : 'any permitted project')}.</p><p>Findings may go only to {recipients.map(id => catalog.members.find(m => m.user_id === id)?.display_name).join(', ')}. This grant starts no monitoring.</p><button disabled={busy} onClick={() => void save()}>Authorize this grant</button><button disabled={busy} onClick={() => setReview(false)}>Revise grant</button></> : <form onSubmit={e => { e.preventDefault(); setReview(true); }}><fieldset disabled={busy}><legend>Access and recipients</legend><label htmlFor={`${id}-beneficiary`}>Who may use this account?</label><select id={`${id}-beneficiary`} disabled={!!grant} required value={beneficiary} onChange={e => setBeneficiary(e.target.value)}><option value="">Choose member</option>{catalog.members.map(m => <option key={m.user_id} value={m.user_id}>{m.display_name}</option>)}</select><fieldset><legend>Allowed capabilities</legend>{account.capabilities.map(o => <label key={o}><input type="checkbox" checked={operations.includes(o)} onChange={e => setOperations(e.target.checked ? [...operations, o] : operations.filter(k => k !== o))} /> {labels[o] || friendly(o)}</label>)}</fieldset><div className="form-field"><label htmlFor={`${id}-resources`}>Exact permitted source IDs (one per line)</label><textarea id={`${id}-resources`} required value={resources} onChange={e => setResources(e.target.value)} placeholder="For example INBOX for Gmail, or an exact Outlook folder ID" /><small>Use exact IDs for mail labels/folders or other sources. Wildcards are not permitted.</small><label htmlFor={`${id}-project`}>Project restriction</label><select id={`${id}-project`} disabled={!!grant} value={project} onChange={e => setProject(e.target.value)}><option value="">Any permitted project</option>{catalog.projects.map(p => <option key={p.project_id} value={p.project_id}>{p.name}</option>)}</select><label htmlFor={`${id}-expires`}>Grant expires (local time)</label><input id={`${id}-expires`} type="datetime-local" required value={expires} onChange={e => setExpires(e.target.value)} /></div><RecipientPicker members={catalog.members} selected={recipients} onChange={setRecipients} /><button disabled={!operations.length || !recipients.length}>Review grant</button></fieldset></form>}<button disabled={busy} onClick={onDone}>Cancel grant</button></section>;
}

export function ProjectCheckPanel({ projectId, identity, readOnly = false }: { projectId: string; identity: string; readOnly?: boolean }) {
  const catalog = useManaged<Catalog>('catalog', {}, identity);
  const project = useManaged<ProjectChecks>('project', { project_id: projectId }, `${identity}:${projectId}`);
  const [editing, setEditing] = useState<Policy | 'new' | null>(null); const [notice, setNotice] = useState(""); const [busy, setBusy] = useState(false); const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => { setEditing(null); setNotice(""); }, [identity, projectId]);
  const lifecycle = async (p: Policy, status: string) => { setBusy(true); try { await optionalManagement('set_status', { policy_id: p.policy_id, expected_revision: p.revision, status }); setNotice(`Check ${status}.`); } catch (cause) { setNotice(String(cause)); } finally { await project.refresh(); setBusy(false); } };
  return <section className="optional-management"><h3 tabIndex={-1} ref={heading}>Project checks</h3><p>Checks run only inside their authorized retrieval window. Findings and delivery receipts remain separate from acquisition success.</p><StatusText>{project.error || catalog.error || notice}</StatusText><button onClick={() => void project.refresh()}>Refresh checks</button>
    {project.data?.checks.map(({ policy: p, runs, next_run_at }) => <article key={p.policy_id} className="project-detail"><h4>{p.query.text || 'Project check'}</h4><p>Status: {friendly(p.status)} · Next run: {next_run_at ? new Date(next_run_at).toLocaleString(undefined, { timeZone: p.timezone }) : 'Not scheduled'} · {p.timezone}</p><p>{p.trigger.weekdays.map(d => days[d]).join(', ')} at {p.trigger.local_time}; retrieval {p.retrieval_window.start}–{p.retrieval_window.end}; lookback {p.initial_lookback_days} days.</p><p>Last run: {runs[0] ? `${runs[0].window_start} · ${friendly(runs[0].outcome || runs[0].state)}` : 'Never run'}</p>
      {!readOnly && <div className="dialog-actions"><button disabled={busy} onClick={() => setEditing(p)}>Revise or authorize</button><button disabled={busy || p.status !== 'active'} onClick={() => void lifecycle(p, 'paused')}>Pause check</button><button disabled={busy || p.status !== 'paused'} onClick={() => void lifecycle(p, 'active')}>Resume authorized scope</button><button disabled={busy || p.status === 'ended'} onClick={() => void lifecycle(p, 'ended')}>End check</button></div>}
      <details><summary>Run health and coverage ({runs.length})</summary>{runs.length === 0 && <p>No run receipts yet. An offline or missed window cannot count as a successful check.</p>}{runs.map(r => <div key={r.check_run_id}><p>{r.window_start}–{r.window_end}: {friendly(r.outcome || r.state)} {r.error_code && `· ${friendly(r.error_code)}`}{r.stale && ' · stale window'}{r.offline && ' · worker offline'}{r.pending_analysis && ' · pending analysis'}</p><Coverage value={r.coverage} /></div>)}</details>
    </article>)}
    <h4>Findings and delivery outcomes</h4>{project.data?.findings.map(f => <article className="project-detail" key={f.finding_id}><h5>{f.summary}</h5><p>{friendly(f.state)} · Due: {f.due_at || 'Unknown'} · Timezone: {f.timezone || 'Unknown'}</p>{f.uncertainty.map(u => <p key={u}>{u}</p>)}<p>Delivery: {Object.entries(f.deliveries).map(([id, d]) => `${catalog.data?.members.find(m => m.user_id === id)?.display_name || id}: ${friendly(d.state)}`).join('; ') || 'No delivery receipt'}</p><details><summary>Finding coverage</summary><Coverage value={f.coverage} /></details></article>)}{project.data?.findings.length === 0 && <p>No findings are currently available to you.</p>}
    {!readOnly && !editing && <button disabled={!catalog.data || !project.data} onClick={() => setEditing('new')}>Add project check</button>}
    {editing && catalog.data && project.data && <CheckEditor key={editing === 'new' ? 'new' : `${editing.policy_id}:${editing.revision}`} projectId={projectId} policy={editing === 'new' ? undefined : editing} catalog={catalog.data} sources={project.data.sources} onDone={() => { setEditing(null); heading.current?.focus(); void project.refresh(); }} />}
  </section>;
}
function localInput(iso: string | null): string {
  if (!iso) return '';
  const date = new Date(iso); return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}
function CheckEditor({ projectId, policy, catalog, sources, onDone }: { projectId: string; policy?: Policy; catalog: Catalog; sources: ProjectChecks['sources']; onDone: () => void }) {
  const [definition, setDefinition] = useState<Definition>(() => policy ? editableDefinition(policy) : defaultDefinition(projectId));
  const [review, setReview] = useState<{ review_id: string; summary: Definition & { goal: { goals: string[]; scope: string } } } | null>(null);
  const [notice, setNotice] = useState(""); const [busy, setBusy] = useState(false); const focus = useRef<HTMLHeadingElement>(null); const alive = useRef(true); const id = useId();
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useEffect(() => { focus.current?.focus(); }, [review]);
  const update = (patch: Partial<Definition>) => { setReview(null); setDefinition(d => ({ ...d, ...patch })); };
  const allowed = catalog.members.map(m => m.user_id).filter(id => definition.source_bindings.every(s => sources.find(c => c.binding_id === s.binding_id && c.operation === s.operation)?.recipient_user_ids.includes(id)));
  const submit = async () => {
    const invalid = validateDefinition(definition); if (invalid) { setNotice(invalid); return; }
    setBusy(true); setNotice("");
    try { const result = await optionalManagement<typeof review>('review', { definition, policy_id: policy?.policy_id || null, expected_revision: policy?.revision || 0 }); if (alive.current) setReview(result); }
    catch (cause) { if (alive.current) setNotice(String(cause)); } finally { if (alive.current) setBusy(false); }
  };
  const apply = async (activate: boolean) => {
    if (!review) return; setBusy(true);
    try { await optionalManagement('apply_review', { review_id: review.review_id, activate }); if (alive.current) onDone(); }
    catch (cause) { if (alive.current) { setNotice(String(cause)); setReview(null); setBusy(false); } }
  };
  if (review) return <section className="project-detail"><h4 tabIndex={-1} ref={focus}>Authorize this concrete check</h4><p>Project goals: {review.summary.goal.goals.join('; ') || review.summary.goal.scope || 'No additional project goals'}</p><p>Check goal: {definition.query.text}</p><ul>{definition.source_bindings.map(s => <li key={`${s.binding_id}:${s.operation}`}>{sources.find(c => c.binding_id === s.binding_id)?.display_name || s.connection_id}: {labels[s.operation] || friendly(s.operation)} in {s.resources.join(', ')}</li>)}</ul><p>{definition.trigger.weekdays.map(d => days[d]).join(', ')} at {definition.trigger.local_time} ({definition.timezone}), retrieval only {definition.retrieval_window.start}–{definition.retrieval_window.end}. Initial lookback: {definition.initial_lookback_days} days.</p><p>From {definition.duration.starts_at} until {definition.duration.ongoing ? 'explicitly ongoing' : definition.duration.ends_at}. Missed windows: {definition.missed_run?.kind === 'bounded_late' ? `allow at most ${definition.missed_run.max_lateness_seconds} seconds late, still inside the window` : 'wait for the next allowed window'}.</p><p>Recipients: {definition.recipient_user_ids.map(id => catalog.members.find(m => m.user_id === id)?.display_name || id).join(', ')}. Notifications: {friendly(definition.notification_mode)}.</p><p>Per run: {Object.entries(definition.budgets).map(([k, v]) => `${friendly(k)}: ${v}`).join('; ')}.</p><p>Activation authorizes these recurring checks without further approval inside this scope. This review expires in five minutes; changed project or grant scope requires a fresh review.</p><button disabled={busy} onClick={() => void apply(true)}>Authorize and activate check</button><button disabled={busy} onClick={() => void apply(false)}>Save disabled draft</button><button disabled={busy} onClick={() => setReview(null)}>Revise check</button><button disabled={busy} onClick={onDone}>Cancel</button></section>;
  return <section className="project-detail"><h4 tabIndex={-1} ref={focus}>{policy ? 'Revise check' : 'New project check'}</h4><StatusText>{notice}</StatusText><form onSubmit={e => { e.preventDefault(); void submit(); }}><fieldset disabled={busy}><legend>Check scope</legend>
    <div className="form-field"><label htmlFor={`${id}-goal`}>What should Alphonse check?</label><textarea id={`${id}-goal`} required value={definition.query.text} onChange={e => update({ query: { text: e.target.value } })} /></div>
    <fieldset><legend>Authorized accounts and sources</legend>{sources.map(s => <label key={`${s.binding_id}:${s.operation}`}><input type="checkbox" checked={definition.source_bindings.some(v => v.binding_id === s.binding_id && v.operation === s.operation)} onChange={e => { const selected = e.target.checked ? [...definition.source_bindings, sourceDefinition(s)] : definition.source_bindings.filter(v => !(v.binding_id === s.binding_id && v.operation === s.operation)); update({ source_bindings: selected, skill_hints: [...new Set(selected.map(s => s.skill_id))], recipient_user_ids: [] }); }} /> {s.display_name} · {labels[s.operation] || friendly(s.operation)} · {s.resources.join(', ')} · {s.role}</label>)}{sources.length === 0 && <p>No usable skill sources. Have the owner grant access, associate it with a skill, and explicitly select the skill role in Integrations.</p>}</fieldset>
    <RecipientPicker members={catalog.members} selected={definition.recipient_user_ids} allowed={allowed} onChange={v => update({ recipient_user_ids: v })} />
    <fieldset><legend>Check days</legend>{days.map((day, n) => <label key={day}><input type="checkbox" checked={definition.trigger.weekdays.includes(n)} onChange={e => { const selected = e.target.checked ? [...definition.trigger.weekdays, n].sort() : definition.trigger.weekdays.filter(v => v !== n); update({ trigger: { ...definition.trigger, weekdays: selected }, retrieval_window: { ...definition.retrieval_window, weekdays: selected } }); }} /> {day}</label>)}</fieldset>
    <div className="form-field"><label htmlFor={`${id}-time`}>Check time</label><input id={`${id}-time`} required type="time" value={definition.trigger.local_time} onChange={e => update({ trigger: { ...definition.trigger, local_time: e.target.value } })} /><label htmlFor={`${id}-timezone`}>Schedule timezone</label><input id={`${id}-timezone`} required value={definition.timezone} onChange={e => update({ timezone: e.target.value })} /><label htmlFor={`${id}-window-start`}>Retrieval window starts</label><input id={`${id}-window-start`} type="time" required value={definition.retrieval_window.start} onChange={e => update({ retrieval_window: { ...definition.retrieval_window, start: e.target.value } })} /><label htmlFor={`${id}-window-end`}>Retrieval window ends (exclusive)</label><input id={`${id}-window-end`} type="time" required value={definition.retrieval_window.end} onChange={e => update({ retrieval_window: { ...definition.retrieval_window, end: e.target.value } })} /><small>Applies on the selected days, in the schedule timezone. Acquisition and analysis stop at the end, even in a one-minute window.</small></div>
    <div className="form-field"><label htmlFor={`${id}-lookback`}>Initial lookback (days)</label><input id={`${id}-lookback`} type="number" min={1} max={365} required value={definition.initial_lookback_days} onChange={e => update({ initial_lookback_days: Number(e.target.value) })} /><label htmlFor={`${id}-starts`}>Starts (your local time)</label><input id={`${id}-starts`} type="datetime-local" required value={localInput(definition.duration.starts_at)} onChange={e => e.target.value && update({ duration: { ...definition.duration, starts_at: new Date(e.target.value).toISOString() } })} /><label><input type="checkbox" checked={definition.duration.ongoing} onChange={e => update({ duration: { ...definition.duration, ongoing: e.target.checked, ends_at: null } })} /> Explicitly ongoing</label>{!definition.duration.ongoing && <><label htmlFor={`${id}-ends`}>Ends (your local time)</label><input id={`${id}-ends`} type="datetime-local" required value={localInput(definition.duration.ends_at)} onChange={e => update({ duration: { ...definition.duration, ends_at: e.target.value ? new Date(e.target.value).toISOString() : null } })} /></>}</div>
    <fieldset><legend>Per-run budgets</legend>{Object.entries(definition.budgets).map(([key, value]) => <div key={key} className="form-field"><label htmlFor={`${id}-${key}`}>{friendly(key)}</label><input id={`${id}-${key}`} type="number" required min={key === 'max_model_calls' ? 0 : 1} max={key === 'max_model_calls' ? 10000 : 1000000000} value={value} onChange={e => update({ budgets: { ...definition.budgets, [key]: Number(e.target.value) } })} /></div>)}</fieldset>
    <div className="form-field"><label htmlFor={`${id}-missed`}>Missed check</label><select id={`${id}-missed`} value={definition.missed_run?.kind || 'next_allowed_window'} onChange={e => update({ missed_run: e.target.value === 'bounded_late' ? { kind: 'bounded_late', max_lateness_seconds: 60 } : { kind: 'next_allowed_window' } })}><option value="next_allowed_window">Wait for next allowed window</option><option value="bounded_late">Allow a bounded late start inside the window</option></select>{definition.missed_run?.kind === 'bounded_late' && <><label htmlFor={`${id}-late`}>Maximum lateness (seconds)</label><input id={`${id}-late`} type="number" min={1} max={86400} value={definition.missed_run.max_lateness_seconds} onChange={e => update({ missed_run: { kind: 'bounded_late', max_lateness_seconds: Number(e.target.value) } })} /></>}<label htmlFor={`${id}-notification`}>Notifications</label><select id={`${id}-notification`} value={definition.notification_mode} onChange={e => update({ notification_mode: e.target.value })}><option value="material_changes">Material changes only</option><option value="requested_digest">Requested digest</option></select></div>
    <button>Review authorization</button><button type="button" onClick={onDone}>Cancel</button></fieldset></form></section>;
}
export function CheckAgenda({ identity }: { identity: string }) {
  const catalog = useManaged<Catalog>('catalog', {}, identity); const [projectId, setProjectId] = useState("");
  return <section className="optional-management"><FindingInbox identity={identity} /><h3>Project check agenda</h3><StatusText>{catalog.error}</StatusText><label htmlFor="check-agenda-project">Project</label><select id="check-agenda-project" value={projectId} onChange={e => setProjectId(e.target.value)}><option value="">Choose project</option>{catalog.data?.projects.map(p => <option key={p.project_id} value={p.project_id}>{p.name}</option>)}</select>{projectId && <ProjectCheckPanel key={`${identity}:${projectId}`} projectId={projectId} identity={identity} />}</section>;
}

export function FindingInbox({ identity }: { identity: string }) {
  const { data, error } = useManaged<{ findings: ProjectChecks['findings'] }>('findings', {}, identity);
  return <section className="optional-management"><h3>Findings for you</h3><StatusText>{error}</StatusText>{data?.findings.map(f => <article key={f.finding_id}><h4>{f.summary}</h4><p>{friendly(f.state)} · Due: {f.due_at || 'Unknown'} · {f.timezone || 'Timezone unknown'}</p>{f.uncertainty.map(u => <p key={u}>{u}</p>)}<p>Delivery: {Object.values(f.deliveries).map(d => friendly(d.state)).join(', ') || 'No delivery receipt'}</p></article>)}{data?.findings.length === 0 && <p>No findings are currently available to you.</p>}</section>;
}
