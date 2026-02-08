alter table public.push_devices
  drop constraint if exists push_devices_owner_id_fkey;

alter table public.push_devices
  add constraint push_devices_owner_id_fkey
  foreign key (owner_id) references public.users (id);
