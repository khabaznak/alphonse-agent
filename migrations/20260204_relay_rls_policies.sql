-- Suggested RLS policies for relay tables (Supabase).
-- Apply separately; adjust to your auth model.

-- Enable RLS
alter table relay_channels enable row level security;
alter table relay_messages enable row level security;

-- Example: allow service role full access (service key bypasses RLS).
-- For authenticated users, scope by owner_user_id or device_id as needed.

-- relay_channels: owner can select/update their rows
create policy "relay_channels_owner_select"
on relay_channels for select
using (auth.uid() = owner_user_id);

create policy "relay_channels_owner_update"
on relay_channels for update
using (auth.uid() = owner_user_id);

-- relay_messages: owner can read messages for their channels
create policy "relay_messages_owner_select"
on relay_messages for select
using (
  exists (
    select 1
    from relay_channels rc
    where rc.id = relay_messages.channel_id
      and rc.owner_user_id = auth.uid()
  )
);

-- relay_messages: allow insert from authenticated users (mobile) into their channel
create policy "relay_messages_owner_insert"
on relay_messages for insert
with check (
  exists (
    select 1
    from relay_channels rc
    where rc.id = relay_messages.channel_id
      and rc.owner_user_id = auth.uid()
  )
);
