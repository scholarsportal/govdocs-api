create table if not exists govdoc_files (
  id uuid primary key default gen_random_uuid(),
  file_name text not null,
  file_url text not null,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);