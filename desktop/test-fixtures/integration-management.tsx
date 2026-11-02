// Browser-only synthetic transport; no daemon, accounts or real provider calls.
import React from 'react';
import { createRoot } from 'react-dom/client';
import { OptionalAccounts, ProjectCheckPanel } from '../src/OptionalIntegrations';
import '../src/styles.css';
const source = { skill_id: 'skill:school', role: 'mail', binding_id: 'school-mail', connection_id: 'parent-mail', grant_id: 'school-grant', operation: 'mail_read', resources: ['INBOX'], recipient_user_ids: ['parent'], project_id: null, display_name: 'School mail' };
const members = [{ user_id: 'parent', display_name: 'Parent' }, { user_id: 'child', display_name: 'Child' }, { user_id: 'other', display_name: 'Other member' }];
const connection = { connection_id: 'parent-mail', provider: 'google', display_name: 'School mail', status: 'connected', health_code: 'healthy', capabilities: ['mail_read'], credential_health: 'available', revision: 1 };
const state = { checks: [] as unknown[], findings: [], sources: [source] };
let review: unknown;
Object.assign(window, { __fixtureCalls: [], __TAURI_INTERNALS__: { invoke: async (command: string, args: { method: string; params: { action: string; values: Record<string, unknown> } }) => {
  if (command === 'open_account_consent') return;
  if (command !== 'daemon_request' || args.method !== 'optional_management') throw new Error('Fixture rejects unrelated calls');
  const { action, values } = args.params;
  (window as unknown as { __fixtureCalls: unknown[] }).__fixtureCalls.push({ action, values });
  if (action === 'catalog') return { actor_user_id: 'parent', members, connections: [connection], grants: [], providers: [{ provider: 'google', capabilities: ['mail_read', 'calendar_read'] }], sources: [source], skills: [{ skill_id: 'skill:school', name: 'School' }], bindings: [], projects: [{ project_id: 'school', name: 'School term' }] };
  if (action === 'project') return state;
  if (action === 'review') { review = { review_id: 'synthetic-review', summary: { ...(values.definition as object), goal: { goals: ['Track school commitments'], scope: 'School term' } } }; return review; }
  if (action === 'apply_review') { state.checks = [{ policy: { ...(review as { summary: object }).summary, policy_id: 'school-check', revision: 1, status: values.activate ? 'active' : 'disabled' }, runs: [], next_run_at: '2026-11-03T14:00:00Z' }]; return state.checks[0]; }
  if (action === 'pause' || action === 'disconnect') { connection.status = action === 'pause' ? 'paused' : 'disconnected'; connection.revision++; return connection; }
  throw new Error('Unsupported fixture action');
} } });
createRoot(document.getElementById('root')!).render(<main style={{ maxWidth: 860, margin: 'auto', padding: 24 }}><h1>Integration management test fixture</h1><OptionalAccounts identity="parent" /><ProjectCheckPanel projectId="school" identity="parent" /></main>);
