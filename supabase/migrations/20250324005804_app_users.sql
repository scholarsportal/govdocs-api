create table app_users (
    id uuid primary key references auth.users(id) on delete cascade,
    first_name text,
    last_name text,
    position_title text,
    avatar_id uuid references storage.objects(id),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- create a storage bucket for avatars
insert into storage.buckets (id, name, public)
values ('avatars', 'avatars', true);


create policy "Users can upload their avatar"
on storage.objects for insert to authenticated with check (
  bucket_id = 'avatars' and
    owner = (select auth.uid()) and
    path_tokens[1]::uuid = (select auth.uid()) -- Ensure the file name starts with the user's id
);


create policy "Anyone can view user's avatar"
on storage.objects for select to authenticated using (
  bucket_id = 'avatars'
);

create policy "Users can remove their avatar"
on storage.objects for delete to authenticated using (
  bucket_id = 'avatars' and
    owner = (select auth.uid())
);

create policy "Users can replace their avatar"
on storage.objects for update to authenticated using (
  bucket_id = 'avatars' and
    owner = (select auth.uid())
);


-- add function to automatically create a user profile
create or replace function public.handle_new_user()
returns trigger as $$
begin
    insert into public.app_users (id, first_name, last_name, position_title)
    values (
        new.id,
        coalesce(new.raw_user_meta_data->>'first_name', ''),
        coalesce(new.raw_user_meta_data->>'last_name', ''),
        coalesce(new.raw_user_meta_data->>'position_title', '')
    );

    return new;
end;
$$ language plpgsql security definer;

-- trigger to call the function when a new user is created using the supabase auth service
create trigger on_auth_user_created
after insert on auth.users
for each row
execute function public.handle_new_user();

-- create a view for easier access to user data with avatar urls
create or replace view user_profiles as
select 
    au.id,
    au.first_name,
    au.last_name,
    au.position_title,
    au.created_at,
    au.updated_at,
    so.name as avatar_name,
    case 
        when so.id is not null then 
            concat('storage/v1/object/public/', so.bucket_id, '/', so.name)
        else null
    end as avatar_url
from public.app_users au
left join storage.objects so on so.id = au.avatar_id
where so.bucket_id = 'avatars' ;


-- trigger function to update app_user.avatar_id when a new avatar is uploaded
create or replace function update_user_avatar()
returns trigger as $$
begin
    update public.app_users
    set avatar_id = new.id
    where id = (new.path_tokens[1]::uuid);
    return new;
end;
$$ language plpgsql security definer;

-- trigger to call the function when a new avatar is uploaded
create constraint trigger on_avatar_uploaded
after insert on storage.objects
deferrable initially deferred
for each row
when (new.bucket_id = 'avatars')
execute function update_user_avatar();