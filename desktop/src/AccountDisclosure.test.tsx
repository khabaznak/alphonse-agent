import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { AccountConsentGuidance, AccountDisclosure, accountActionError } from './OptionalIntegrations';
import type { Connection } from './integrationManagement';

describe('account processing disclosure', () => {
  it('explains external processing and local retention with an unchecked acknowledgement', () => {
    const html = renderToStaticMarkup(<AccountDisclosure accepted={false} onChange={() => {}} />);
    expect(html).toContain('email or calendar information');
    expect(html).toContain('TypeSafe and OpenAI');
    expect(html).toContain('local storage or backups');
    expect(html).toContain('I have reviewed how account information may be processed');
    expect(html).not.toContain('checked=""');
  });
});

describe('account setup errors', () => {
  it('offers recovery for account mismatch, cleanup, and stale state', () => {
    expect(accountActionError(new Error('OAuthError: oauth_account_mismatch'))).toContain('original account');
    expect(accountActionError('oauth_credential_cleanup_required')).toContain('Retry deletion');
    expect(accountActionError('oauth_revision_conflict')).toContain('Refresh the account list');
  });
  it('does not display unrecognized errors or private provider details', () => {
    const message = accountActionError('unexpected synthetic private detail');
    expect(message).toContain('could not be completed');
    expect(message).not.toContain('synthetic private detail');
  });
});

describe('account consent guidance', () => {
  const connection: Connection = { connection_id: 'synthetic', provider: 'google', display_name: 'Test', revision: 1, status: 'reauth_required', health_code: 'consent_required', capabilities: [] };
  it('explains how to continue an existing consent record without guessing account identity', () => {
    const html = renderToStaticMarkup(<AccountConsentGuidance connection={connection} />);
    expect(html).toContain('unfinished connection');
    expect(html).toContain('Reconnect with consent');
    expect(html).toContain('do not establish that two records are the same account');
  });
  it('does not show consent guidance for a connected account or another recovery state', () => {
    expect(renderToStaticMarkup(<AccountConsentGuidance connection={{ ...connection, status: 'connected', health_code: 'healthy' }} />)).toBe('');
    expect(renderToStaticMarkup(<AccountConsentGuidance connection={{ ...connection, health_code: 'restore_reconciliation_required' }} />)).toBe('');
  });
});
