-- Relay transport tables (Supabase Postgres)
create extension if not exists pgcrypto;
create table if not exists relay_channels (
  id uuid primary key,
  owner_user_id uuid,
  alphonse_id text not null,
  device_id text not null,
  created_at timestamptz default now(),
  expires_at timestamptz,
  last_seen_alphonse timestamptz,
  last_seen_device timestamptz
);

create table if not exists relay_messages (
  id uuid primary key,
  channel_id uuid references relay_channels(id) on delete cascade,
  sender text not null,
  type text not null,
  ts timestamptz default now(),
  correlation_id uuid,
  device_id text not null,
  alphonse_id text not null,
  payload jsonb,
  schema_version int default 1,
  delivered_to_alphonse boolean default false,
  delivered_to_device boolean default false
);

create index if not exists relay_messages_alphonse_idx
  on relay_messages (alphonse_id, delivered_to_alphonse, sender, type, ts);

create index if not exists relay_messages_channel_idx
  on relay_messages (channel_id, ts);
