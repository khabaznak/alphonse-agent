import { useCallback, useEffect, useId, useRef, useState } from "react";
import type { RefObject } from "react";
import { openAccountConsent, optionalManagement } from "./api";
import {
  OUTLOOK_INBOX, OUTLOOK_INBOX_UNREAD, ResponseFence, isGrantCurrent, isOwnerReadGrant,
  type Catalog, type Connection, type Grant,
} from "./integrationManagement";

const MICROSOFT = "microsoft";
const STATUS_INTERVAL_MS = 1000;

type ConsentStatus = {
  status: "pending" | "connected" | "cancelled" | "expired" | "failed";
  reason?: string;
};
type ConsentResponse = { authorization_url: string; attempt_id: string };
type AccountState = "connected" | "refresh" | "reconnect" | "unfinished" | "paused" | "credentials" | "removed" | "cleanup";
type AccountEdit = { kind: "access" | "remove"; connectionId: string };

function accountState(account: Connection): AccountState {
  if (account.credential_deletion_pending) return "cleanup";
  if (account.status === "disconnected") return "removed";
  if (account.status === "paused") return "paused";
  if (account.status === "reauth_required") return account.capabilities.length ? "reconnect" : "unfinished";
  if (account.status !== "connected" || account.credential_health === "absent"
      || account.credential_health === "credential_backend_unavailable" || account.access_token_status === "unavailable") {
    return "credentials";
  }
  return account.access_token_status === "refresh_required" ? "refresh" : "connected";
}

const ACCOUNT_STATUS: Record<AccountState, string> = {
  connected: "Connected",
  refresh: "Microsoft access expired",
  reconnect: "Reconnect required",
  unfinished: "Connection incomplete",
  paused: "Paused",
  credentials: "Credentials unavailable",
  removed: "Removed in Alphonse",
  cleanup: "Removal needs attention",
};

function consentMessage(result: ConsentStatus): string {
  switch (result.status) {
    case "connected": return "Microsoft account connected.";
    case "cancelled": return "Microsoft sign-in was cancelled. Start again when you’re ready.";
    case "expired": return "Microsoft sign-in expired. Start a new sign-in.";
    case "failed":
      switch (result.reason) {
        case "account_mismatch": return "Microsoft returned a different account. Reconnect and choose the original account in the browser.";
        case "consent_denied": return "Microsoft denied access. Check the account’s consent settings and try again.";
        case "consent_incomplete": return "Microsoft did not grant Inbox access. Try again and review the requested permission.";
        case "connection_changed": return "The saved account changed during sign-in. Review its current status before trying again.";
        default: return "Microsoft sign-in could not be completed. Try again.";
      }
    default: return "Finish Microsoft sign-in in your browser, then return here.";
  }
}

function accountError(cause: unknown, fallback: string): string {
  const code = cause instanceof Error ? cause.message.split(": ").at(-1) : null;
  switch (code) {
    case "outlook_inbox_grant_conflict":
    case "outlook_inbox_grant_ambiguous":
    case "outlook_inbox_grant_exists":
      return "Another account already has Inbox access. Revoke that access before granting it here.";
    case "oauth_revision_conflict":
    case "account_revision_conflict":
    case "grant_revision_conflict":
    case "outlook_inbox_connection_changed":
      return "The saved account or permission changed. Review its current status and try again.";
    case "account_connection_unavailable":
      return "The Microsoft account is unavailable. Review its current status and try again.";
    case "oauth_credential_cleanup_required":
      return "Finish removing the old credentials before connecting this account again.";
    case "oauth_reauth_required":
      return "Reconnect the Microsoft account, then try again.";
    default:
      return fallback;
  }
}

function consentRequest(action: "oauth_attempt_status" | "cancel_oauth_attempt", attemptId: string) {
  return optionalManagement<ConsentStatus>(action, { provider: MICROSOFT, attempt_id: attemptId });
}

function inboxAccountTitle(records: Connection[], account: Connection): string {
  const label = account.display_name || "Microsoft account";
  const sameName = records.filter(record => record.display_name === account.display_name);
  return sameName.length > 1
    ? `${label} · account ${sameName.findIndex(record => record.connection_id === account.connection_id) + 1}`
    : label;
}

export function OutlookInboxSettings({ catalog, onRefresh, onConversation }: {
  catalog: Catalog;
  onRefresh: () => Promise<void>;
  onConversation?: () => void;
}) {
  const id = useId();
  const panelTitle = useRef<HTMLHeadingElement>(null);
  const editTitle = useRef<HTMLHeadingElement>(null);
  const stopSignIn = useRef<HTMLButtonElement>(null);
  const actionTrigger = useRef<HTMLElement | null>(null);
  const consentFence = useRef(new ResponseFence());
  const pendingAttempt = useRef<string | null>(null);
  const submitting = useRef(false);
  const alive = useRef(false);
  const [label, setLabel] = useState("");
  const [attempt, setAttempt] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [pollNotice, setPollNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [edit, setEdit] = useState<AccountEdit | null>(null);

  const configured = catalog.providers.some(item => item.provider === MICROSOFT && item.capabilities.includes(OUTLOOK_INBOX_UNREAD));
  const records = catalog.connections.filter(account => account.provider === MICROSOFT
    && (!account.capabilities.length || account.capabilities.includes(OUTLOOK_INBOX_UNREAD)));
  const visible = records.filter(account => account.status !== "disconnected");
  const removed = records.filter(account => account.status === "disconnected");
  const grants = catalog.grants.filter(grant => isOwnerReadGrant(catalog, grant, OUTLOOK_INBOX_UNREAD));
  const liveGrants = grants.filter(isGrantCurrent);
  const awaitingConsent = attempt !== null;
  const locked = busy || awaitingConsent;
  const chosenAccount = records.find(account => account.connection_id === edit?.connectionId);
  const chosenGrant = liveGrants.find(grant => grant.connection_id === chosenAccount?.connection_id)
    || grants.find(grant => grant.connection_id === chosenAccount?.connection_id);
  const ready = configured && liveGrants.length === 1 && visible.some(account =>
    account.connection_id === liveGrants[0].connection_id && accountState(account) === "connected");

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      consentFence.current.invalidate();
      const attemptId = pendingAttempt.current;
      pendingAttempt.current = null;
      if (attemptId) void consentRequest("cancel_oauth_attempt", attemptId).catch(() => {});
    };
  }, []);

  const finishConsent = useCallback(async (attemptId: string, result: ConsentStatus) => {
    if (result.status === "pending" || pendingAttempt.current !== attemptId) return;
    pendingAttempt.current = null;
    setAttempt(null);
    setNotice(consentMessage(result));
    setPollNotice("");
    setStopping(false);
    if (result.status === "connected") setLabel("");
    await onRefresh();
    if (alive.current && !pendingAttempt.current && !submitting.current) panelTitle.current?.focus();
  }, [onRefresh]);

  useEffect(() => {
    if (!attempt) return;
    stopSignIn.current?.focus({ preventScroll: true });
    let timer: number | undefined;
    let stopped = false;
    const poll = async () => {
      try {
        const result = await consentRequest("oauth_attempt_status", attempt);
        if (!stopped && pendingAttempt.current === attempt) {
          setPollNotice("");
          await finishConsent(attempt, result);
        }
      } catch {
        if (!stopped && pendingAttempt.current === attempt) {
          setPollNotice("Couldn’t check sign-in status. Alphonse will try again.");
        }
      } finally {
        if (!stopped && pendingAttempt.current === attempt) timer = window.setTimeout(() => void poll(), STATUS_INTERVAL_MS);
      }
    };
    void poll();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [attempt, finishConsent]);

  const begin = async (account?: Connection) => {
    if (!configured || submitting.current || pendingAttempt.current) return;
    const accountLabel = account?.display_name || label.trim();
    if (!accountLabel) return;
    const ticket = consentFence.current.start();
    submitting.current = true;
    setBusy(true);
    setNotice("");
    try {
      const result = await optionalManagement<ConsentResponse>("begin", {
        provider: MICROSOFT,
        connection_id: account?.connection_id || `account_${crypto.randomUUID()}`,
        expected_revision: account?.revision || 0,
        display_name: accountLabel,
        capabilities: [OUTLOOK_INBOX_UNREAD],
      });
      if (!consentFence.current.current(ticket)) {
        if (result.attempt_id) void consentRequest("cancel_oauth_attempt", result.attempt_id).catch(() => {});
        return;
      }
      if (!result.attempt_id) {
        setNotice("This version could not track Microsoft sign-in. Update Alphonse and try again.");
        return;
      }
      pendingAttempt.current = result.attempt_id;
      setAttempt(result.attempt_id);
      try {
        await openAccountConsent(result.authorization_url);
      } catch {
        try {
          const stopped = await consentRequest("cancel_oauth_attempt", result.attempt_id);
          if (consentFence.current.current(ticket) && pendingAttempt.current === result.attempt_id) {
            await finishConsent(result.attempt_id, stopped);
            setNotice("Alphonse couldn’t open Microsoft sign-in. Try again.");
          }
        } catch {
          if (consentFence.current.current(ticket)) setPollNotice("Alphonse couldn’t open Microsoft sign-in. Stop this attempt before trying again.");
        }
      }
    } catch (cause) {
      if (consentFence.current.current(ticket)) {
        setNotice(accountError(cause, "Microsoft sign-in could not be started. Try again."));
        await onRefresh();
      }
    } finally {
      if (consentFence.current.current(ticket)) {
        submitting.current = false;
        setBusy(false);
      }
    }
  };

  const cancel = async () => {
    if (!attempt || stopping) return;
    setStopping(true);
    try {
      const result = await consentRequest("cancel_oauth_attempt", attempt);
      await finishConsent(attempt, result);
    } catch {
      if (alive.current && pendingAttempt.current === attempt) setPollNotice("Alphonse couldn’t stop sign-in yet. Try again.");
    } finally {
      if (alive.current) setStopping(false);
    }
  };

  const openEditor = (kind: AccountEdit["kind"], account: Connection, trigger: HTMLElement) => {
    actionTrigger.current = trigger;
    setNotice("");
    setEdit({ kind, connectionId: account.connection_id });
  };
  useEffect(() => {
    if (edit) editTitle.current?.focus();
  }, [edit]);

  const closeEditor = () => {
    setEdit(null);
    window.requestAnimationFrame(() => {
      if (!alive.current) return;
      if (actionTrigger.current?.isConnected && !actionTrigger.current.matches(":disabled")) actionTrigger.current.focus();
      else panelTitle.current?.focus();
    });
  };

  const mutate = async (action: string, values: Record<string, unknown>, success: string, fallback: string, close = false) => {
    if (submitting.current || pendingAttempt.current) return;
    const trigger = document.activeElement;
    submitting.current = true;
    setBusy(true);
    setNotice("");
    let saved = false;
    try {
      await optionalManagement(action, values);
      if (!alive.current) return;
      saved = true;
      setNotice(success);
    } catch (cause) {
      if (alive.current) setNotice(accountError(cause, fallback));
    } finally {
      if (alive.current) {
        await onRefresh();
        submitting.current = false;
        setBusy(false);
        if (saved && close) closeEditor();
        else window.requestAnimationFrame(() => {
          if (!alive.current || document.activeElement !== document.body) return;
          if (trigger instanceof HTMLElement && trigger.isConnected && !trigger.matches(":disabled")) trigger.focus();
          else panelTitle.current?.focus();
        });
      }
    }
  };

  const grantAccess = (account: Connection, expiresAt: string) => {
    void mutate("save_grant", {
      grant_id: chosenGrant?.grant_id || `grant_${crypto.randomUUID()}`,
      connection_id: account.connection_id,
      expected_connection_revision: account.revision,
      expected_revision: chosenGrant?.revision || 0,
      beneficiary_user_id: catalog.actor_user_id,
      operations: [OUTLOOK_INBOX_UNREAD],
      resources: [OUTLOOK_INBOX],
      recipient_user_ids: [catalog.actor_user_id],
      expires_at: expiresAt,
      project_id: null,
    }, "Inbox access saved. You can ask for the unread count in Conversation.", "Inbox access could not be saved. Try again.", true);
  };

  const renderAccount = (account: Connection) => {
    const state = accountState(account);
    const title = inboxAccountTitle(records, account);
    const grant = liveGrants.find(item => item.connection_id === account.connection_id)
      || grants.find(item => item.connection_id === account.connection_id);
    const grantCurrent = Boolean(grant && isGrantCurrent(grant));
    const allowedForAnotherAccount = liveGrants.some(item => item.connection_id !== account.connection_id);
    const canReconnect = configured && !account.credential_deletion_pending && !account.revocation_pending
      && account.credential_health !== "credential_backend_unavailable";
    const connectAction = state === "unfinished" ? "Finish connecting" : state === "paused" ? "Resume access" : "Reconnect";
    const refresh = state === "refresh" && grant && grantCurrent;
    const reconnect = state === "reconnect" || state === "paused" || state === "unfinished" || state === "credentials"
      || (state === "refresh" && !refresh);
    return <section className="mockup-settings-row outlook-account-row" key={account.connection_id} aria-label={title}>
      <div>
        <h5>{title}</h5>
        <p>{ACCOUNT_STATUS[state]}</p>
        {state === "connected" && <p>{grantCurrent ? "Inbox access granted" : grant ? "Inbox access expired" : "Inbox access needed"}</p>}
        {grantCurrent && grant?.expires_at && <p>Access ends {new Date(grant.expires_at).toLocaleString()}</p>}
        {state === "paused" && <p>Resume the connection to check unread messages.</p>}
        {state === "credentials" && <p>{canReconnect ? "Reconnect to restore credentials." : "Credential storage is unavailable. Try again when it’s available."}</p>}
        {account.credential_deletion_pending && <p>Local credential deletion is pending.</p>}
        {account.revocation_pending && <p>To withdraw Microsoft consent, remove Alphonse from the apps connected to your Microsoft account.</p>}
      </div>
      <div className="outlook-account-actions">
        {reconnect && <button type="button" disabled={locked || !canReconnect} onClick={() => void begin(account)}
          aria-label={`${connectAction} ${title}`}>{connectAction}</button>}
        {refresh && <button type="button" disabled={locked || !canReconnect} aria-label={`Refresh access for ${title}`}
          onClick={() => void mutate("refresh_outlook_inbox_access", {
            connection_id: account.connection_id, expected_revision: account.revision,
            grant_id: grant.grant_id, grant_revision: grant.revision,
          }, "Microsoft access refreshed.", "Microsoft access could not be refreshed. Try again.")}>Refresh access</button>}
        {configured && state === "connected" && !allowedForAnotherAccount && !grantCurrent && <button type="button" disabled={locked}
          onClick={event => openEditor("access", account, event.currentTarget)}
          aria-label={`${grant ? "Renew" : "Grant"} Inbox access for ${title}`}>{grant ? "Renew Inbox access" : "Grant Inbox access"}</button>}
        {grant && (grantCurrent || allowedForAnotherAccount || state !== "connected") && <button type="button" disabled={locked}
          onClick={event => openEditor("access", account, event.currentTarget)}
          aria-label={`Manage Inbox access for ${title}`}>Manage Inbox access</button>}
        {state === "connected" && allowedForAnotherAccount && !grant && <p>Inbox access is already granted to another account.</p>}
        {state !== "removed" && (state !== "cleanup" || account.status !== "disconnected" || account.credential_deletion_pending)
          && <details><summary aria-label={`More options for ${title}`}>More options</summary>
          <div className="dialog-actions">
            {state === "connected" && canReconnect && <button type="button" disabled={locked}
              aria-label={`Reconnect ${title}`} onClick={() => void begin(account)}>Reconnect</button>}
            {state === "connected" && <button type="button" disabled={locked} onClick={() => void mutate("pause", {
              connection_id: account.connection_id, expected_revision: account.revision,
            }, "Microsoft account paused.", "Microsoft account could not be paused. Try again.")}>Pause</button>}
            {account.credential_deletion_pending && <button type="button" disabled={locked} onClick={() => void mutate("retry_deletion", {
              connection_id: account.connection_id, expected_revision: account.revision,
            }, "Credential cleanup checked.", "Credential cleanup could not be completed. Try again.")}>Retry credential deletion</button>}
            {account.status !== "disconnected" && <button type="button" disabled={locked}
              onClick={event => openEditor("remove", account, event.currentTarget)}>Remove account…</button>}
          </div>
        </details>}
      </div>
    </section>;
  };

  return <div className="account-service-detail outlook-inbox-settings">
    <h4 ref={panelTitle} tabIndex={-1}>Inbox unread count</h4>
    <p>Connect a personal Microsoft account, then grant access to its Inbox unread count. Alphonse checks only the count when you ask in your private Desktop conversation.</p>
    <p className="outlook-status" role="status">{edit?.kind === "access" ? "" : notice || (awaitingConsent && consentMessage({ status: "pending" }))}</p>
    {pollNotice && <p className="outlook-status" role="status">{pollNotice}</p>}
    {awaitingConsent && <button type="button" ref={stopSignIn} aria-disabled={stopping} onClick={() => void cancel()}>{stopping ? "Stopping sign-in…" : "Stop sign-in"}</button>}
    {!configured && <div className="project-detail">
      <h5>Microsoft setup required</h5>
      <p>Ask your Alphonse administrator to configure the Microsoft connection before signing in.</p>
      <div className="dialog-actions"><button type="button" onClick={() => void onRefresh()}>Check setup again</button></div>
    </div>}
    {liveGrants.length > 1 && <p role="alert">More than one Inbox grant is active. Keep only the grant for the Inbox you want to check.</p>}
    {visible.length > 0 && <div className="mockup-settings-rows" aria-label="Microsoft accounts">{visible.map(renderAccount)}</div>}
    {ready && !edit && onConversation && <div className="project-detail">
      <h5>Ask in Conversation</h5>
      <p>Try “How many unread messages are in my Hotmail Inbox?”</p>
      <div className="dialog-actions"><button type="button" onClick={onConversation}>Open Conversation</button></div>
    </div>}
    {chosenAccount && edit?.kind === "access" && <InboxAccessEditor key={chosenAccount.connection_id}
      title={inboxAccountTitle(records, chosenAccount)} grant={chosenGrant} heading={editTitle} notice={notice}
      editable={configured && accountState(chosenAccount) === "connected"
        && !liveGrants.some(grant => grant.connection_id !== chosenAccount.connection_id)}
      busy={locked} onSave={expiresAt => grantAccess(chosenAccount, expiresAt)}
      onRevoke={grant => void mutate("revoke_grant", { grant_id: grant.grant_id, expected_revision: grant.revision },
        "Inbox access revoked.", "Inbox access could not be revoked. Try again.", true)}
      onClose={closeEditor} />}
    {chosenAccount && edit?.kind === "remove" && <section className="project-detail">
      <h5 ref={editTitle} tabIndex={-1}>Remove {inboxAccountTitle(records, chosenAccount)}?</h5>
      <p>Removal stops Inbox access in Alphonse. Manage Microsoft consent separately in your Microsoft account.</p>
      <div className="dialog-actions"><button type="button" disabled={locked} onClick={() => void mutate("disconnect", {
        connection_id: chosenAccount.connection_id, expected_revision: chosenAccount.revision,
      }, "Microsoft account removed from Alphonse.", "Microsoft account could not be removed. Try again.", true)}>Remove account</button>
      <button type="button" disabled={locked} onClick={closeEditor}>Keep account</button></div>
    </section>}
    {configured && !edit && <form className="integration-connect-form outlook-connect-form" onSubmit={event => { event.preventDefault(); void begin(); }}>
      <h5>{visible.length ? "Connect another Microsoft account" : "Connect a Microsoft account"}</h5>
      <div className="form-field"><label htmlFor={`${id}-label`}>Account name in Alphonse</label>
        <input id={`${id}-label`} value={label} required maxLength={120} disabled={locked} onChange={event => setLabel(event.target.value)} />
      </div>
      <p>In the browser, choose the Microsoft account whose Inbox you want to check.</p>
      <button type="submit" disabled={locked || !label.trim()}>Continue to Microsoft</button>
    </form>}
    {removed.some(account => account.credential_deletion_pending || account.revocation_pending) &&
      <details className="integration-removed-accounts"><summary>Removed accounts
        {removed.some(account => account.credential_deletion_pending) && " · credential cleanup needed"}</summary>
        <div className="mockup-settings-rows">{removed.filter(account => account.credential_deletion_pending || account.revocation_pending).map(renderAccount)}</div>
      </details>}
  </div>;
}

function grantDate(grant?: Grant): string {
  const currentExpiry = grant && isGrantCurrent(grant) ? grant.expires_at : null;
  const date = currentExpiry ? new Date(currentExpiry) : new Date();
  if (!currentExpiry) date.setDate(date.getDate() + 30);
  date.setMinutes(date.getMinutes() - date.getTimezoneOffset());
  return date.toISOString().slice(0, 16);
}

function InboxAccessEditor({ title, grant, heading, notice, editable, busy, onSave, onRevoke, onClose }: {
  title: string;
  grant?: Grant;
  heading: RefObject<HTMLHeadingElement | null>;
  notice: string;
  editable: boolean;
  busy: boolean;
  onSave: (expiresAt: string) => void;
  onRevoke: (grant: Grant) => void;
  onClose: () => void;
}) {
  const id = useId();
  const [expiresAt, setExpiresAt] = useState(grantDate(grant));
  const expiry = new Date(expiresAt);
  const valid = expiry.getTime() > Date.now();
  return <section className="project-detail">
    <h5 ref={heading} tabIndex={-1}>Inbox access for {title}</h5>
    <p>Allow Alphonse to check the unread message count in this Inbox when you ask in your private Desktop conversation.</p>
    <p role="status" className="outlook-status">{notice}</p>
    <form className="outlook-grant-form" onSubmit={event => {
      event.preventDefault();
      if (!busy && editable && valid) onSave(expiry.toISOString());
    }}>
      {editable && <div className="form-field"><label htmlFor={`${id}-expiry`}>Access expires (local time)</label>
        <input id={`${id}-expiry`} type="datetime-local" required value={expiresAt} disabled={busy}
          aria-describedby={`${id}-expiry-help`} aria-invalid={!valid}
          onChange={event => setExpiresAt(event.target.value)} />
        <p id={`${id}-expiry-help`} aria-live="polite">{valid
          ? "You can change the end date or revoke access earlier."
          : "Choose a date and time in the future."}</p>
      </div>}
      <div className="dialog-actions">
        {editable && <button type="submit" disabled={busy || !valid}>{grant ? "Save Inbox access" : "Confirm Inbox access"}</button>}
        {grant && <button type="button" disabled={busy} onClick={() => onRevoke(grant)}>Revoke Inbox access</button>}
        <button type="button" disabled={busy} onClick={onClose}>Close</button>
      </div>
    </form>
  </section>;
}
