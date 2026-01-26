create table if not exists public.push_devices (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id),
  token text not null,
  platform text not null default 'android',
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists push_devices_token_key
  on public.push_devices (token);
