import { describe, it, expect } from 'vitest';
import { defaultDefinition, editableDefinition, groupAccountRecords, ResponseFence, sourceDefinition, validateDefinition } from './integrationManagement';
import type { Connection } from './integrationManagement';

describe('account record grouping', () => {
  it('keeps same-label records separate across connected and recovery states', () => {
    const record = (connection_id: string, status: string): Connection => ({ connection_id, status, provider: 'google', display_name: 'Mail', revision: 1, health_code: '', capabilities: [] });
    const records = [record('connected-one', 'connected'), record('unfinished', 'reauth_required'), record('connected-two', 'connected'), record('disconnected', 'disconnected')];
    const groups = groupAccountRecords(records);
    expect(groups.connected.map(c => c.connection_id)).toEqual(['connected-one', 'connected-two']);
    expect(groups.other.map(c => c.connection_id)).toEqual(['unfinished', 'disconnected']);
    expect([...groups.connected, ...groups.other]).toHaveLength(records.length);
  });
});

describe('bounded check editor', () => {
  const valid = () => {
    const d = defaultDefinition('project-one'); d.query.text = 'School deadlines';
    d.source_bindings = [{ skill_id: 'skill:school', role: 'mail', binding_id: 'b', connection_id: 'c', grant_id: 'g', operation: 'mail_read', resources: ['INBOX'] }];
    d.recipient_user_ids = ['parent']; d.duration.ongoing = true; return d;
  };
  it('requires explicit sources, recipients, goal and finite duration or ongoing choice', () => {
    const d = defaultDefinition('project-one'); expect(validateDefinition(d)).toMatch(/goal/);
    d.query.text = 'Deadlines'; expect(validateDefinition(d)).toMatch(/source/);
    d.source_bindings = valid().source_bindings; expect(validateDefinition(d)).toMatch(/receive/);
    d.recipient_user_ids = ['parent']; expect(validateDefinition(d)).toMatch(/end/);
    d.duration.ongoing = true; expect(validateDefinition(d)).toBeNull();
  });
  it('accepts a one-minute window and treats its end as exclusive', () => {
    const d = valid(); d.retrieval_window.end = '08:01'; expect(validateDefinition(d)).toBeNull();
    d.trigger.local_time = '08:01'; expect(validateDefinition(d)).toMatch(/window/);
  });
  it('rejects invalid timezone, empty days, fractional budgets and excessive lookback', () => {
    const d = valid(); d.timezone = 'somewhere'; expect(validateDefinition(d)).toMatch(/timezone/);
    d.timezone = 'UTC'; d.trigger.weekdays = []; expect(validateDefinition(d)).toMatch(/day/);
    d.trigger.weekdays = [0]; d.budgets.max_pages = 1.1; expect(validateDefinition(d)).toMatch(/budget/);
    d.budgets.max_pages = 1; d.initial_lookback_days = 366; expect(validateDefinition(d)).toMatch(/Lookback/);
  });
  it('revision editing drops server authority and pins before reauthorization', () => {
    const p = { ...valid(), policy_id: 'check-one', revision: 9, status: 'active', authorized_by: 'other' };
    Object.assign(p.source_bindings[0], { grant_revision: 10, connection_revision: 2 });
    const d = editableDefinition(p); expect(d).not.toHaveProperty('authorized_by'); expect(d).not.toHaveProperty('revision');
    expect(d.source_bindings[0]).not.toHaveProperty('grant_revision');
    expect(d.source_bindings).toEqual([sourceDefinition(p.source_bindings[0] as never)]);
  });
});

describe('response fencing', () => {
  it('rejects an older refresh, identity response or unmounted component result', () => {
    const fence = new ResponseFence(); const old = fence.start(); const current = fence.start();
    expect(fence.current(old)).toBe(false); expect(fence.current(current)).toBe(true);
    fence.invalidate(); expect(fence.current(current)).toBe(false);
  });
});
